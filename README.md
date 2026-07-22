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
│   ├── langchain/         # VMMMemory drop-in (v2)
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

**v0.1 — Core Engine (in progress)**
- [ ] `core/` engine: ledger, budget, page_table, extractor, deduplicator, fork_manager, eviction_store
- [ ] LangGraph SDK integration
- [ ] Dry-run mode (show decisions without modifying context)
- [ ] 5 built-in tool schemas: kubectl, SQL, REST API, bash, file

**v0.2 — vLLM Native Integration (Q3 2026, pending RFC #37168 merge)**
- [ ] `POST /release_kv_cache` on tombstone events
- [ ] `cache_salt` session isolation per fork
- [ ] DRAM eviction store backend

**v1.0 — Full Platform**
- [ ] OpenAI-compatible HTTP proxy (zero code change deployment)
- [ ] LangChain / CrewAI / LlamaIndex integrations
- [ ] Redis / S3 eviction backends
- [ ] Benchmarking harness (VMM vs RAG vs full-context)
- [ ] OpenTelemetry tracing + Prometheus metrics

---

## Status

Early-stage research and design. Architecture validated against vLLM RFC feedback. Core engine implementation in progress.

Contributions, feedback, and collaboration welcome — especially from teams working on long-horizon agent workloads or vLLM agentic cache management.

---

## Research Context

- [Is Progressive Disclosure All You Need for Long-Context Agents?](https://arxiv.org/abs/2504.01954)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [vLLM RFC #37168: Active Coordination and Two-Zone KV Cache Scheduling](https://github.com/vllm-project/vllm/issues/37168)
- [vLLM Q3 2026 Roadmap](https://github.com/vllm-project/vllm/issues/48168)
