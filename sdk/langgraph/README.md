# ContextStream — LangGraph Integration

Intercepts tool calls in a LangGraph `StateGraph` agent transparently.
Raw output never enters main context — only compressed lessons (~40 tokens each).

## Setup

```python
from core.engine import ContextStreamEngine
from sdk.langgraph.plugin import ContextStreamPlugin

engine = ContextStreamEngine(model="claude-haiku-4-5-20251001")
engine.init("Your agent task")
cs = ContextStreamPlugin(engine)
```

---

## Pattern 1: `wrap_tool_node` — wrap an existing node (one line)

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class AgentState(TypedDict):
    task: str
    messages: list
    tool_input: str
    tool_output: str   # raw output lands here briefly, evicted after compression

def call_kubectl(state):
    return {"tool_output": run_kubectl(state["tool_input"])}

graph = StateGraph(AgentState)

# Before: graph.add_node("call_tool", call_kubectl)
# After:
graph.add_node("call_tool", cs.wrap_tool_node(
    call_kubectl,
    tool_name="kubectl",       # routes to correct extraction schema
    output_key="tool_output",  # where raw output lives in state
    task_key="task",           # where task description lives
))
```

After the node runs:
- `state["tool_output"]` → `None` (evicted)
- `state["messages"]` → appended with compressed lesson SystemMessage
- `state["_cs_lesson_refs"]` → list of `LessonReference` objects

---

## Pattern 2: `before_node` / `after_node` — manual wiring

More control. Attach hooks to separate graph nodes.

```python
graph.add_node("before_kubectl", cs.before_node("kubectl", task_key="task"))
graph.add_node("call_kubectl",   call_kubectl_fn)
graph.add_node("after_kubectl",  cs.after_node("kubectl", output_key="tool_output"))

graph.add_edge("before_kubectl", "call_kubectl")
graph.add_edge("call_kubectl",   "after_kubectl")
```

`before_node` attaches `ForkContext` to `state["_cs_fork_ctx"]`.
`after_node` reads it, pages output, clears it.

---

## Pattern 3: `inject_context` — prepend ledger before LLM node

Call this before the node that invokes the LLM so the agent sees all compressed lessons:

```python
def llm_node(state):
    # Inject compressed ledger as SystemMessage[0]
    patch = cs.inject_context(state)
    state = {**state, **patch}

    # state["messages"][0] now contains the full lesson ledger
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}
```

---

## Dry-run mode

```python
engine = ContextStreamEngine(dry_run=True)
cs = ContextStreamPlugin(engine)

# No LLM extraction calls — all decisions logged
state = {"task": "test", "messages": [], "tool_output": "kubectl output here"}
wrapped = cs.wrap_tool_node(lambda s: {"tool_output": "raw data"}, tool_name="kubectl")
wrapped(state)

for event in engine.dry_run_log:
    print(f"[{event.event}] {event.detail}")
```

---

## State key reference

| Key | Set by | Cleared by | Notes |
|---|---|---|---|
| `_cs_fork_ctx` | `before_node` | `after_node` | Transient — do not read directly |
| `_cs_lesson_refs` | `after_node` | never | Accumulated `LessonReference` list |
| `_cs_raw_output` | internal | `after_node` | Not used in current impl |
| `tool_output` | your tool fn | `after_node` | Set to `None` after compression |
| `messages` | `after_node` | never | Compressed lesson appended as `SystemMessage` |

---

## Full example

See [`examples/sre_agent/`](../../examples/sre_agent/) for a complete 4-tool agent
(kubectl → SQL → bash → file) with dry-run smoke test at
[`sdk/langgraph/examples/dry_run_agent.py`](examples/dry_run_agent.py).

```bash
python sdk/langgraph/examples/dry_run_agent.py
```
