# ETL Pipeline Agent Benchmark

Benchmarks ContextStream vs naive context-stuffing on a synthetic ETL pipeline failure.

## Scenario

`etl_orders_daily` has failed 4 consecutive days with QueryTimeout.
Runtime jumped from ~36s to 900s. The agent runs 4 tools to investigate:

| Tool | What it returns |
|---|---|
| `EXPLAIN ANALYZE` | Parallel Seq Scan on orders, 45s execution, 16.3M filtered rows, disk sort |
| `pg_stat_user_tables` | 27.12% dead tuple bloat on orders (18.3M dead), idx_scan=0 for 8 days |
| `REST /pipeline/runs` | 4 consecutive failures at 900s timeout, last success 2026-07-16 at 36s |
| `bash pg_stat_activity` | 38 connections blocked on DataFileRead, longest query 38m14s |

## Results (claude-haiku-4-5-20251001)

| Metric | Baseline | ContextStream |
|---|---|---|
| Context chars | 6,587 | 1,825 |
| Prompt tokens | 2,520 | **692** |
| Token reduction | — | **72.5%** |
| Raw data in main context | yes | never |

Both agents correctly diagnose: vacuum bloat + missing index.
ContextStream adds `autovacuum_vacuum_cost_limit` tuning and a bloat verification query.

## Notable: Tombstone deduplication in action

Two SQL tools ran sequentially on the same `orders` table.
ContextStream automatically tombstoned the EXPLAIN lesson when the table stats lesson
arrived — higher information density, same entity:

```
[LESSON]    [sql] Table 'orders' seq scan filtered 16.3M rows, exec=45112ms
[TOMBSTONE] [sql] supersedes above — orders 27.12% dead tuple bloat, idx_scan=0 for 8 days
```

Main context stays clean. No redundant information.

## Compressed ledger (692 tokens vs 2,520 baseline)

```
[SYSTEM_INSTRUCTION] Data pipeline incident: etl_orders_daily failed 4 consecutive days...
[LESSON]    [sql]      Seq scan filtered 16.3M rows, no index on (status, created_at), exec=45112ms
[TOMBSTONE] [sql]      orders: 18.3M dead tuples (27.12%), idx_scan=0, autovacuum untuned
[LESSON]    [rest_api] 4 consecutive failures at 900s timeout, avg success was 36.6s
[LESSON]    [bash]     38 connections blocked on DataFileRead, longest query 38m14s
```

## Run

```bash
# From repo root
cp .env.example .env  # add ANTHROPIC_API_KEY

python examples/etl_agent/compare.py           # Haiku (default)
python examples/etl_agent/compare.py --quiet   # summary only
python examples/etl_agent/compare.py --model claude-sonnet-4-6
```
