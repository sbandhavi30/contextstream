# Quickstart

Get ContextStream running in 5 minutes.

## Prerequisites

- Python 3.10+
- An Anthropic API key ([get one here](https://console.anthropic.com))

## Install

```bash
git clone https://github.com/sbandhavi30/contextstream.git
cd contextstream

# Install dependencies
pip install anthropic pydantic pyyaml

# Set your API key
cp .env.example .env
# Edit .env — replace the placeholder with your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

## Run the benchmarks

```bash
# SRE incident demo — OOMKilled pod (kubectl → SQL → bash → file)
python examples/sre_agent/compare.py

# ETL pipeline demo — query bloat (SQL EXPLAIN → table stats → REST → bash)
python examples/etl_agent/compare.py
```

Both print a side-by-side table: baseline (raw context stuffing) vs ContextStream (compressed lessons). Typical output: 55–72% token reduction, same or better diagnosis quality.

---

## Use in your own agent — 3 patterns

### Pattern A: Direct API (any framework)

```python
from core.engine import ContextStreamEngine

engine = ContextStreamEngine(model="claude-haiku-4-5-20251001")
engine.init("Your agent task description")

# Before tool executes
fork_ctx = engine.before_tool_call("kubectl", task_description="check OOMKilled pods")

# Run your tool, get raw output
raw_output = run_kubectl("describe pod web-backend")

# After tool — pages output, extracts lesson, appends to ledger
ref = engine.after_tool_call(fork_ctx, iter([raw_output]))
print(f"Lesson extracted: conf={ref.confidence:.2f}")

# Inject compressed context into your LLM call
context = engine.render_context()   # ~40 tokens per tool call, not 8000
```

### Pattern B: LangChain

```python
from core.engine import ContextStreamEngine
from sdk.langchain.memory import VMMMemory
from sdk.langchain.callback import ContextStreamCallbackHandler

engine = ContextStreamEngine(model="claude-haiku-4-5-20251001")

# Drop-in memory replacement
memory = VMMMemory(engine=engine)
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# OR: zero-change callback
agent.run("diagnose the incident", callbacks=[ContextStreamCallbackHandler(engine=engine)])
```

→ Full guide: [`sdk/langchain/README.md`](sdk/langchain/README.md)

### Pattern C: LangGraph

```python
from core.engine import ContextStreamEngine
from sdk.langgraph.plugin import ContextStreamPlugin

engine = ContextStreamEngine(model="claude-haiku-4-5-20251001")
cs = ContextStreamPlugin(engine)

# Wrap any tool node — one line
graph.add_node("call_tool", cs.wrap_tool_node(call_tool_fn, tool_name="kubectl"))
```

→ Full guide: [`sdk/langgraph/README.md`](sdk/langgraph/README.md)

---

## Dry-run mode (test without API calls)

```python
engine = ContextStreamEngine(dry_run=True)
engine.init("Test task")

fork_ctx = engine.before_tool_call("sql")
engine.after_tool_call(fork_ctx, iter(["SELECT * FROM orders -- 1842 rows"]))

# See every decision logged
for event in engine.dry_run_log:
    print(f"[{event.event}] {event.detail}")
```

No API calls made. Raw output paged to in-memory store. All extraction decisions logged.

---

## Add a custom tool schema

If your agent uses a tool not in the built-in 5 (kubectl, sql, rest_api, bash, file):

```bash
python scripts/new_schema.py
# Walks you through 4 questions → writes YAML to configs/tool_schemas/
```

Or manually create `configs/tool_schemas/my_tool.yaml`:

```yaml
description: "Extracts lessons from my_tool output"
base: bash          # inherit from nearest built-in schema
model: claude-haiku-4-5-20251001
fields:
  my_field: "string — description of what to extract"
  confidence: "float 0.0-1.0"
```

→ Full guide: [`CONTRIBUTING_SCHEMAS.md`](CONTRIBUTING_SCHEMAS.md)

---

## Check engine status

```python
print(engine.status())
# {
#   "session_id": "abc123",
#   "tokens_used": 1842,
#   "tokens_limit": 200000,
#   "pressure": "normal",
#   "active_lessons": 4,
#   "eviction_bytes": 14821
# }
```
