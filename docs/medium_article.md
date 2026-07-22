# Your LLM Agent Is Drowning in Its Own Tool Outputs

## I built an OS-style Virtual Memory Manager for LLM context windows — here's why it matters

---

Every agent framework has the same silent killer.

You build an agent. It works great for 5–10 tool calls. Then somewhere around tool call 15, something changes. The model starts missing obvious connections. It repeats work it already did. Sometimes it just stops reasoning well altogether.

You've hit context rot.

---

## The Problem Nobody Talks About

Here's what your agent's context window actually looks like after a real investigation:

```
System prompt:              ~800 tokens
kubectl describe pod:     8,432 tokens
SQL query result:         6,100 tokens
REST API response:        4,200 tokens
bash df output:           1,840 tokens
deployment manifest:      2,100 tokens
... 12 more tool calls
─────────────────────────────────────
Total:               ~95,000 tokens
```

Every LLM call sends that entire window. The model tries to attend to all of it. Research on "lost-in-the-middle" attention shows that tokens buried deep in a large context receive dramatically less attention than tokens near the beginning or end.

Worse: most of those 95,000 tokens are raw tool outputs the model already used. The kubectl log that told you the pod was OOMKilled? Still in the context. The SQL result that showed session bloat? Still there. Page after page of operational noise the model has to wade through to find the three facts it actually needs.

Three failure modes compound this:

| Failure | Mechanism |
|---|---|
| Lost-in-the-middle | Attention degrades for tokens buried in large context |
| Context rot | Accumulated raw output obscures signal with noise |
| Agent death | Context limit hit mid-task, agent cannot continue |

This is an infrastructure-layer problem being patched — poorly — at the application layer. Every team building agents is writing their own ad-hoc summarization logic, their own context trimming heuristics, their own "compress this before it overflows" patches.

There's a better abstraction.

---

## The OS Analogy

Operating systems solved an identical problem decades ago.

Your laptop has 16GB of RAM, but can run programs that collectively need 200GB. The OS manages this through virtual memory: pages data in and out of physical RAM based on what the process actually needs right now, keeps a page table as a semantic index, and makes the whole thing transparent to the running process.

LLM agents have the same structure:

- **Physical RAM** = the context window (limited, expensive)
- **Virtual address space** = the full history of tool outputs (unbounded)
- **Page table** = index of what was learned, where raw data lives, how confident we are
- **Working set** = the compressed lessons the model needs for the current reasoning step

ContextStream applies this analogy directly. It's an LLM Virtual Memory Manager (VMM).

---

## How It Works

Four mechanics, each solving a specific failure mode:

### 1. Page OUT immediately

The moment a tool returns output, it streams to cold storage — never materializes in the main context. The framework gets back a pointer (a `raw_ref`), not the data.

```python
fork_ctx = engine.before_tool_call("kubectl")
ref = engine.after_tool_call(fork_ctx, iter([raw_kubectl_output]))
# raw_kubectl_output: 8,432 tokens → eviction store
# ref: lightweight LessonReference pointer → main context
```

### 2. Extract a typed structured lesson

A cheap LLM (Claude Haiku) compresses the raw output into a typed JSON schema — not freeform summarization, but structured fields with confidence scoring:

```
[LESSON] [kubectl] [conf=0.92]
Pod web-backend-6f9d7c4-xkv2p in namespace production OOMKilled —
memory limit 512Mi breached after 7 restarts.
| memory=512Mi, restarts=7
```

That's ~40 tokens. Down from 8,432.

The schema forces specificity. No vague "memory issue detected." The root cause must name the exact resource, exact metric with units, and specific condition. If the extractor can't fill a field with confidence, the confidence score drops — and the caller decides whether to re-page the raw data.

Built-in schemas for 5 tool types: kubectl, SQL, REST API, bash, file. Custom tools get a YAML registry with base schema inheritance.

### 3. Append-only ledger

Lessons append to a ledger. The ledger never mutates.

This isn't just clean design — it's a hard requirement for KV cache efficiency. vLLM and other inference engines cache KV state by prefix hash. Any mutation to the context prefix = full KV recompute. By keeping the ledger append-only, ContextStream guarantees the KV cache hit rate never regresses as the agent runs longer.

```
=== CONTEXT LEDGER ===
[SYSTEM_INSTRUCTION] Investigate OOM on web-backend pod
[LESSON] [kubectl] [conf=0.92] Pod web-backend OOMKilled, memory=512Mi, restarts=7
[LESSON] [sql]     [conf=0.93] 847 active sessions, top-5 consume 191MB total
[LESSON] [bash]    [conf=0.92] Node disk 93% full, memory 94% utilized
[LESSON] [file]    [conf=0.92] NODE_OPTIONS=2048MB heap vs 512Mi limit — guarantees OOMKill
======================
```

The main agent sees this — ~200 tokens — instead of 22,000+ tokens of raw output.

### 4. Tombstone deduplication

State changes in real systems. A pod that was OOMKilled gets restarted. A query that was failing after a fix now succeeds.

When new evidence supersedes old, ContextStream doesn't mutate the old lesson. It appends a tombstone:

```
[TOMBSTONE — supersedes les_362f4e4a]
(newer observation on same entity)
Table 'orders': 18.3M dead tuples (27.12% bloat) — autovacuum untuned
```

The model's natural recency bias does the rest. It sees the tombstone marker near the top of recent context and weights the newer information higher. No special handling required.

In the ETL benchmark, two SQL tools ran on the same `orders` table. The deduplicator automatically tombstoned the EXPLAIN lesson when the table stats lesson arrived — richer signal, same entity. The ledger stayed clean without any manual logic.

---

## Benchmarks

Two end-to-end scenarios, measured on Claude Haiku.

### Scenario 1: SRE OOM Incident

Tools: kubectl describe → SQL session query → bash resource check → deployment manifest

**The incident:** web-backend pods OOMKilled repeatedly. Find root cause.

| Metric | Baseline | ContextStream |
|---|---|---|
| Prompt tokens | 1,430 | **646** |
| Token reduction | — | **55%** |
| Raw data in context | yes | **never** |

Both agents identified the root cause correctly. ContextStream was more specific: it caught the exact 4:1 heap/container ratio (`NODE_OPTIONS=--max-old-space-size=2048` against a 512Mi pod limit) and explicitly labeled it "guarantees OOMKill." The baseline diagnosis was correct but vaguer.

### Scenario 2: ETL Pipeline Failure

Tools: SQL EXPLAIN ANALYZE → pg_stat_user_tables → REST pipeline run history → bash pg_stat_activity

**The incident:** `etl_orders_daily` failing 4 consecutive days with QueryTimeout. Runtime jumped from 36s to 900s.

| Metric | Baseline | ContextStream |
|---|---|---|
| Prompt tokens | 2,520 | **692** |
| Token reduction | — | **72.5%** |
| Tombstone fired | n/a | **yes** |

The SQL-heavy scenario compresses harder — EXPLAIN output is dense and repetitive. ContextStream's diagnosis added `autovacuum_vacuum_cost_limit` tuning and a bloat verification query that the baseline missed.

---

## vLLM Integration

ContextStream is designed to complement, not compete with, vLLM's native agentic cache APIs.

The upcoming [RFC #37168](https://github.com/vllm-project/vllm/issues/37168) introduces `POST /release_kv_cache` and `cache_salt` session isolation. RFC #48168 adds Agent Session/Correlation-ID hints for prefix cache management.

Division of responsibility:
- **vLLM owns**: KV block lifecycle, cache invalidation mechanics, memory tiering
- **ContextStream owns**: semantic eviction policy, tool-aware lesson extraction, conflict resolution

`budget.py` will call `POST /release_kv_cache` on tombstone events — explicit KV block invalidation when a lesson is superseded. `fork_manager.py` passes `cache_salt` per sub-agent fork — each tool extraction gets isolated KV scope.

---

## Framework Integration

### LangChain — drop-in memory replacement

```python
from sdk.langchain.memory import VMMMemory

# Before:
# chain = LLMChain(llm=llm, prompt=prompt, memory=ConversationBufferMemory())

# After (one line change — prompt {history} variable unchanged):
chain = LLMChain(llm=llm, prompt=prompt, memory=VMMMemory(engine=engine))
```

Or attach as a callback to any existing agent — zero changes to agent logic:

```python
from sdk.langchain.callback import ContextStreamCallbackHandler

agent.run("diagnose the incident",
          callbacks=[ContextStreamCallbackHandler(engine=engine)])
```

Every `on_tool_end` is intercepted automatically.

### LangGraph — wrap any tool node

```python
from sdk.langgraph.plugin import ContextStreamPlugin

cs = ContextStreamPlugin(engine)
graph.add_node("call_tool", cs.wrap_tool_node(call_tool_fn, tool_name="kubectl"))
```

### Direct API — any framework

```python
engine = ContextStreamEngine(model="claude-haiku-4-5-20251001")
engine.init("Diagnose cluster OOM")

fork_ctx = engine.before_tool_call("kubectl")
ref = engine.after_tool_call(fork_ctx, iter([raw_output]))

context = engine.render_context()  # inject into your LLM call
```

---

## Dry-Run Mode

Before committing to live extraction, test every decision:

```python
engine = ContextStreamEngine(dry_run=True)
engine.init("Test task")

fork_ctx = engine.before_tool_call("sql")
engine.after_tool_call(fork_ctx, iter(["SELECT ... 1842 rows -- 3042ms"]))

for event in engine.dry_run_log:
    print(f"[{event.event}] {event.detail}")

# [FORK_PREPARE][sql] fork_id=fork_a997f364 deps=0
# [STREAM_ABSORBED][sql] raw_size=183 chars → ref=sql_34602e75
# [LESSON][sql] conf=0.50 reason=no conflict text_tokens~37
```

No API calls. All raw output goes to in-memory store. Every paging, extraction, and deduplication decision is logged.

---

## What's Next

**v0.1 is complete.** Core engine (7 components), LangGraph + LangChain SDKs, extractor eval (86% pass rate on Haiku), two benchmark scenarios.

**v0.2** — vLLM native integration: `POST /release_kv_cache` on tombstone, `cache_salt` fork isolation, DRAM eviction store. Pending RFC #37168 merge (Q3 2026).

**v1.0** — OpenAI-compatible HTTP proxy (zero code change), CrewAI/LlamaIndex integrations, Redis/S3 eviction backends, full benchmarking harness.

---

## Try It

```bash
git clone https://github.com/sbandhavi30/contextstream.git
cd contextstream
pip install anthropic pydantic pyyaml
cp .env.example .env  # add ANTHROPIC_API_KEY

# Run both benchmarks
python examples/sre_agent/compare.py
python examples/etl_agent/compare.py
```

The benchmarks run against live Anthropic API. You'll see the full before/after comparison: raw token counts, compressed lesson output, final diagnosis from both agents.

---

I'd love feedback from anyone building long-horizon agents, working on inference optimization, or thinking about context management at scale. The vLLM KV cache coordination angle in particular feels underexplored — if you're on that team or adjacent to it, I'd like to talk.

**GitHub:** [github.com/sbandhavi30/contextstream](https://github.com/sbandhavi30/contextstream)

---

*Tags: LLM, AI Agents, Machine Learning, Open Source, Inference Optimization, LangChain, LangGraph, vLLM*
