# SRE Agent Benchmark

Benchmarks ContextStream vs naive context-stuffing on a synthetic production incident.

## Scenario

A `web-backend` pod is OOMKilled repeatedly in the `production` namespace.
The agent runs 4 tools to investigate:

| Tool | What it returns |
|---|---|
| `kubectl describe pod` | OOMKilled, 7 restarts, memory limit 512Mi |
| `SQL query` | 847 active sessions, top session 48 MB, top-5 = 191 MB total |
| `bash df/free/top` | Node disk 93% full, system memory 94% utilized |
| `file read` | Deployment manifest: `NODE_OPTIONS=--max-old-space-size=2048`, `SESSION_CACHE_MAX_MB=unlimited` |

## Results (claude-haiku-4-5-20251001)

| Metric | Baseline | ContextStream |
|---|---|---|
| Context chars | 3,804 | 1,554 |
| Diagnosis prompt tokens | 1,430 | 646 |
| Token reduction | — | **55%** |
| Raw data in main context | yes | never |

Both agents identify the correct root cause. ContextStream diagnosis is more specific
(identifies the 4:1 heap/container ratio as a guarantee of failure).

## Run

```bash
# From repo root
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# Run comparison (quiet mode — summary only)
python examples/sre_agent/compare.py --quiet

# Run comparison with per-step detail
python examples/sre_agent/compare.py

# Run ContextStream agent only
python examples/sre_agent/contextstream_agent.py

# Run baseline only
python examples/sre_agent/baseline_agent.py

# Test with Sonnet
python examples/sre_agent/compare.py --model claude-sonnet-4-6
```

## Files

```
examples/sre_agent/
├── scenario.py              # synthetic tool outputs + task definition
├── baseline_agent.py        # naive: all raw outputs concatenated into prompt
├── contextstream_agent.py   # contextstream: paged outputs, compressed lessons
└── compare.py               # runs both, prints side-by-side table
```

## What the compressed ledger looks like

After 4 tool calls, ContextStream's main context contains:

```
=== CONTEXT LEDGER ===
[SYSTEM_INSTRUCTION] Agent initialized. Task: Production incident...
[LESSON] [kubectl] [conf=0.92] Pod web-backend OOMKilled — memory limit 512Mi breached after 7 restarts...
[LESSON] [sql]     [conf=0.92] top session 48.4MB, top-5 = 191MB, 847 active sessions...
[LESSON] [bash]    [conf=0.92] disk 93% full, memory 94% utilized, 184Mi free...
[LESSON] [file]    [conf=0.92] NODE_OPTIONS=2048MB heap vs 512Mi limit — guarantees OOMKill...
======================
```

646 tokens vs 1,430 for the baseline. The signal that matters — the 4:1 heap/container mismatch —
survives compression and is explicitly flagged in the file lesson.
