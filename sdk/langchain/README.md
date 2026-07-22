# ContextStream — LangChain Integration

Two integration patterns. Use whichever fits your setup.

---

## Pattern 1: `VMMMemory` — drop-in for `ConversationBufferMemory`

Replaces LangChain's `ConversationBufferMemory`. Instead of accumulating raw
tool outputs in the conversation buffer, it pages each output to cold storage
and injects the compressed ledger as the `{history}` variable.

**Zero changes to your prompt or chain.**

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic

from core.engine import ContextStreamEngine
from sdk.langchain.memory import VMMMemory

# 1. Create engine
engine = ContextStreamEngine(model="claude-haiku-4-5-20251001")
engine.init("Diagnose production OOM incident")

# 2. Swap in VMMMemory — same interface as ConversationBufferMemory
memory = VMMMemory(engine=engine, task_description="Diagnose production OOM")

# 3. Wire to chain — prompt uses {history} exactly as before
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="{history}\nHuman: {input}\nAssistant:"
)
chain = LLMChain(llm=ChatAnthropic(model="claude-haiku-4-5-20251001"),
                 prompt=prompt, memory=memory)

# 4. Run — tool outputs are automatically paged, history stays compressed
chain.run("kubectl describe the OOMKilled pod")
chain.run("show me the deployment manifest")

# Each {history} injection is the compressed ledger — not raw outputs
```

### save_context() — manually page a tool result

```python
memory.save_context(
    inputs={"tool": "kubectl", "input": "describe pod web-backend"},
    outputs={"output": raw_kubectl_output},   # raw — gets paged immediately
)

# load_memory_variables returns compressed lessons, not raw output
history = memory.load_memory_variables({})["history"]
```

### Tool name detection

`VMMMemory` auto-detects the tool name from the inputs dict:

| Input key | Example |
|---|---|
| `inputs["tool"]` | `{"tool": "kubectl", ...}` |
| `inputs["action"].tool` | ReAct `AgentAction` object |
| `outputs["tool_name"]` | some chain styles |
| fallback | `"unknown"` → format auto-detection routes schema |

---

## Pattern 2: `ContextStreamCallbackHandler` — zero-change agent integration

Attaches to any LangChain agent or tool via the callbacks interface.
Intercepts every `on_tool_end` automatically — no changes to agent logic.

```python
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain_anthropic import ChatAnthropic

from core.engine import ContextStreamEngine
from sdk.langchain.callback import ContextStreamCallbackHandler

# 1. Create handler
engine = ContextStreamEngine(model="claude-haiku-4-5-20251001")
handler = ContextStreamCallbackHandler(engine=engine)

# 2. Attach to agent — that's it
llm   = ChatAnthropic(model="claude-haiku-4-5-20251001")
tools = load_tools(["llm-math", "serpapi"])
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# Every tool call is intercepted — raw output paged, lesson extracted
agent.run(
    "What is the memory usage on node-3 and is it correlated with OOMKills?",
    callbacks=[handler],
)

# Inspect compressed context at any point
print(handler.get_context())
print(handler.status())
```

### What fires automatically

| LangChain callback | ContextStream action |
|---|---|
| `on_tool_start(tool_name, input)` | `engine.before_tool_call()` — prepares ForkContext |
| `on_tool_end(output)` | `engine.after_tool_call()` — pages output, extracts lesson |
| `on_tool_error(error)` | pages error as low-confidence lesson (not silently dropped) |
| `on_agent_action(action)` | extracts task description from ReAct thought |
| `on_agent_finish(finish)` | logs final answer (dry-run mode only) |

### Parallel tool calls

The handler uses `run_id` to track concurrent tool calls — each fork context
is keyed by `run_id` so parallel tools don't collide:

```python
# LangChain passes run_id automatically — no action needed
# Handler maintains: {run_id: ForkContext} dict
```

### Inject context into the next LLM call

```python
# Get compressed ledger string for manual injection
compressed = handler.get_context()

messages = [
    SystemMessage(content=f"[Investigation context]\n{compressed}"),
    HumanMessage(content="What is the root cause?"),
]
response = llm.invoke(messages)
```

---

## Dry-run mode

Test your integration without making real LLM extraction calls:

```python
engine = ContextStreamEngine(dry_run=True)
memory  = VMMMemory(engine=engine)
handler = ContextStreamCallbackHandler(engine=engine)

# All decisions logged, no LLM calls made
memory.save_context(inputs={"tool": "sql"}, outputs={"output": raw_sql})

for event in engine.dry_run_log:
    print(f"[{event.event}] {event.detail}")
```

---

## Choosing between the two patterns

| | `VMMMemory` | `ContextStreamCallbackHandler` |
|---|---|---|
| Best for | Chains with explicit memory | Agents with automatic tool calls |
| Integration | `memory=VMMMemory(engine)` | `callbacks=[handler]` |
| Tool name source | `inputs["tool"]` or `action.tool` | `serialized["name"]` from LangChain |
| Works with LCEL | Yes (via `RunnableWithMessageHistory`) | Yes (attach to Runnable) |
| Parallel tools | No (sequential memory) | Yes (run_id keyed) |

Use both together for maximum coverage:

```python
memory  = VMMMemory(engine=engine)
handler = ContextStreamCallbackHandler(engine=engine)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
chain.run("investigate the incident", callbacks=[handler])
```
