# ContextStream

**An intelligent context operating system for LLM agents.**

ContextStream sits between your agent framework and the LLM inference engine, managing the context window like an OS manages RAM — paging out raw tool outputs, distilling structured lessons, and keeping the active context lean and causally coherent across unlimited tool calls.

---

## The Problem

Every agent framework forces developers to hand-write summarization logic to prevent context overflow. After 15–20 tool calls, the context window fills with raw tool outputs — kubectl logs, SQL results, API responses — and model performance degrades badly.

Three failure modes:

| Failure | Mechanism |
|---|---|
| Lost-in-the-middle | Attention degrades for tokens buried in large context |
| Context rot | Accumulated raw output obscures signal with noise |
| Agent death | Context limit hit mid-task, agent cannot continue |

This is an infrastructure-layer problem being patched at the application layer.

---


## The Approach

ContextStream applies the OS virtual memory analogy directly:

- **Page OUT** — raw tool output streams to cold storage, never enters main context
- **Extract** — a cheap LLM (Haiku / GPT-4o-mini) compresses it into a typed structured lesson
- **Append** — only the lesson (~40 tokens) appends to the main context (append-only, never mutates)
- **Fork** — tool calls execute in isolated sub-agent processes with clean, minimal context

```
Without ContextStream:                With ContextStream:
┌─────────────────────────┐           ┌─────────────────────────┐
│ System prompt           │           │ System prompt   [cached]│
│ kubectl output: 8,432t  │           │ Lesson 1: OOM@490Mi 40t │
│ SQL result: 6,100t      │           │ Lesson 2: etl-nightly 38t│
│ API response: 4,200t    │           │ Lesson 3: headroom 2Gi 35t│
│ ...12 more tool outputs │           │ Current task            │
│ Total: ~95,000 tokens   │           │ Total: ~650 tokens      │
└─────────────────────────┘           └─────────────────────────┘
         Lost-in-middle                    Full causal chain preserved
```

---

## Benchmarks

Two end-to-end demos measured on `claude-haiku-4-5-20251001`:

### SRE Incident — OOMKilled pod (kubectl → SQL → bash → file)

| Metric | Baseline | ContextStream |
|---|---|---|
| Prompt tokens | 1,430 | **646** |
| Token reduction | — | **55%** |
| Raw data in main context | yes | **never** |

### ETL Pipeline Failure — query bloat (SQL EXPLAIN → table stats → REST → bash)

| Metric | Baseline | ContextStream |
|---|---|---|
| Prompt tokens | 2,520 | **692** |
| Token reduction | — | **72.5%** |
| Tombstone fired | n/a | **yes** — 2 SQL lessons on same table auto-deduplicated |

In both cases diagnosis quality is equal or better — the compressed lesson retains the operative signal.

```bash
cp .env.example .env  # add ANTHROPIC_API_KEY
python examples/sre_agent/compare.py    # OOM incident
python examples/etl_agent/compare.py   # ETL bloat incident
```

---

## Key Design Decisions

**Append-only ledger** — the main context prefix never changes. vLLM and other inference engines cache KV state by prefix hash. Any mutation = full recompute. ContextStream's append-only invariant means KV cache hit rate never regresses.

**Sub-agent forking** — raw tool output goes into an isolated sub-agent's context, gets compressed to a lesson, sub-agent dies. Main agent only sees lessons. This is how the OS analogy actually holds — processes don't share address space.

**Structured extraction** — lessons are typed JSON schemas per tool type, not freeform summaries. `{condition, metric, cause, entity_targets, confidence}`. If a field can't be filled, confidence drops — not silently dropped.

**Tombstone deduplication** — when a lesson is superseded (state changed), a tombstone token appends. No mutation. Model's recency bias handles the state correction naturally.

---

## Architecture

```
contextstream/
├── core/
│   ├── ledger.py          # append-only context ledger (KV-cache safe)
│   ├── budget.py          # token budget tracker + pre-flight overflow check
│   ├── page_table.py      # index: active lessons, confidence, raw data pointers
│   ├── extractor.py       # mini-LLM engine → typed JSON lesson
│   ├── deduplicator.py    # conflict resolver: tombstone/override, never mutates
│   ├── fork_manager.py    # sub-agent lifecycle + dynamic dependency injection
│   └── eviction_store.py  # cold storage: local disk / Redis / S3
├── sdk/
│   ├── langgraph/         # before_node / after_node hook integration
│   ├── langchain/         # VMMMemory drop-in + callback handler
│   └── raw/               # VMMOpenAI, VMMAnthropic wrappers (v2)
├── proxy/                 # OpenAI-compatible HTTP proxy (v2)
├── bench/                 # benchmarking harness (v2)
└── configs/               # tool schema registry (kubectl, SQL, REST, bash, file)
```

**Plugin hook interface** — four methods, attaches to any framework:

```python
class VMMPlugin:
    def before_tool_call(self, tool_name: str, tool_input: dict) -> None
    def after_tool_call(self, tool_name: str, raw_output_stream: Iterator[str]) -> LessonReference
    def on_token_pressure(self, level: float, ledger: Ledger) -> None
    def on_agent_uncertainty(self, query: str) -> list[LessonReference]
```

**LangGraph** — `before_node` / `after_node` / `wrap_tool_node` hooks:

```python
from sdk.langgraph.plugin import ContextStreamPlugin

cs = ContextStreamPlugin(engine)
graph.add_node("call_tool", cs.wrap_tool_node(call_tool_fn, tool_name="kubectl"))
```

**LangChain** — two patterns:

```python
# Pattern 1: drop-in memory replacement
from sdk.langchain.memory import VMMMemory
chain = LLMChain(llm=llm, prompt=prompt, memory=VMMMemory(engine=engine))

# Pattern 2: zero-change callback handler
from sdk.langchain.callback import ContextStreamCallbackHandler
agent.run("diagnose OOM", callbacks=[ContextStreamCallbackHandler(engine=engine)])
```

See [`sdk/langchain/README.md`](sdk/langchain/README.md) for full usage guide.

---

## vLLM Integration

ContextStream is designed to complement vLLM's native agentic cache APIs:

| vLLM RFC | ContextStream integration |
|---|---|
| [#37168](https://github.com/vllm-project/vllm/issues/37168) `POST /release_kv_cache` | `budget.py` calls on tombstone — explicit KV block invalidation |
| [#37168](https://github.com/vllm-project/vllm/issues/37168) `cache_salt` session ID | `fork_manager.py` — sub-agent forks get isolated KV scope |
| [#48168](https://github.com/vllm-project/vllm/issues/48168) Agent Session/Correlation-ID | Fork parent/child relationship signaling |
| [#7697](https://github.com/vllm-project/vllm/issues/7697) DRAM/disk KV tiering | `eviction_store.py` DRAM backend |
| [#5557](https://github.com/vllm-project/vllm/issues/5557) Disaggregated prefilling | Sub-agent forks as dedicated prefill workers |

**Division of responsibility:**
- vLLM owns: KV block lifecycle, cache invalidation mechanics, memory tiering
- ContextStream owns: semantic eviction policy, tool-aware lesson extraction, conflict resolution

---

## Roadmap

**v0.1 — Core Engine (complete)**
- [x] `core/` engine: ledger, budget, page_table, extractor, deduplicator, fork_manager, eviction_store
- [x] LangGraph SDK integration (`sdk/langgraph/`)
- [x] Dry-run mode (show decisions without modifying context)
- [x] 5 built-in tool schemas: kubectl, SQL, REST API, bash, file
- [x] Format auto-detection (regex, <1ms, no LLM)
- [x] YAML schema registry for custom tool schemas
- [x] Schema contribution tooling (`scripts/new_schema.py`, `scripts/validate_schemas.py`)
- [x] Extractor eval harness with 14 ground-truth cases (86% pass rate @ Haiku, 79% @ Sonnet)
- [x] SRE agent benchmark: 55% token reduction, diagnosis quality preserved

**v0.2 — vLLM Native Integration (Q3 2026, pending RFC #37168 merge)**
- [ ] `POST /release_kv_cache` on tombstone events
- [ ] `cache_salt` session isolation per fork
- [ ] DRAM eviction store backend

**v1.0 — Full Platform**
- [ ] OpenAI-compatible HTTP proxy (zero code change deployment)
- [x] LangChain integration (`VMMMemory` + `ContextStreamCallbackHandler`)
- [ ] CrewAI / LlamaIndex integrations
- [ ] Redis / S3 eviction backends
- [ ] Benchmarking harness (VMM vs RAG vs full-context)
- [ ] OpenTelemetry tracing + Prometheus metrics

---

## Status

v0.1 complete. Core engine (7 components), LangGraph SDK, extractor eval (86% Haiku / 79% Sonnet), SRE agent benchmark (55% token reduction). Schema registry with YAML contribution system live.

Contributions, feedback, and collaboration welcome — especially from teams working on long-horizon agent workloads or vLLM agentic cache management.

---

## Research Context

- [Is Progressive Disclosure All You Need for Long-Context Agents?](https://arxiv.org/abs/2504.01954)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [vLLM RFC #37168: Active Coordination and Two-Zone KV Cache Scheduling](https://github.com/vllm-project/vllm/issues/37168)
- [vLLM Q3 2026 Roadmap](https://github.com/vllm-project/vllm/issues/48168)
