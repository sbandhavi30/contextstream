"""
scenario.py — Synthetic ETL pipeline degradation incident.

etl_orders_daily has been failing for 3 days.
Query latency spiked from ~400ms to 45s. Pipeline jobs timing out.

Tool 1: sql_explain      — EXPLAIN ANALYZE on the slow ETL query
Tool 2: sql_table_stats  — pg_stat_user_tables bloat + seq scan counts
Tool 3: rest_pipeline    — pipeline job run history (REST API)
Tool 4: bash_pg_activity — pg_stat_activity + connection counts

Agent task: identify root cause, recommend minimum fix.
"""

SQL_EXPLAIN_OUTPUT = """\
Query: SELECT o.id, o.customer_id, o.total_amount, o.status,
              c.email, c.tier, c.region
       FROM orders o
       JOIN customers c ON o.customer_id = c.id
       WHERE o.created_at >= NOW() - INTERVAL '7 days'
         AND o.status IN ('pending', 'processing')
       ORDER BY o.created_at DESC;

EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT):

Gather Merge  (cost=847291.42..1092847.13 rows=48271 width=124)
              (actual time=44821.341..44997.118 rows=51847 loops=1)
  Workers Planned: 2
  Workers Launched: 2
  ->  Sort  (cost=846291.40..846412.02 rows=48276 width=124)
            (actual time=44803.211..44831.442 rows=17282 loops=3)
        Sort Key: o.created_at DESC
        Sort Method: external merge  Disk: 14872kB
        ->  Parallel Hash Join  (cost=234812.00..842331.12 rows=48276 width=124)
                                (actual time=18234.112..44712.341 rows=17282 loops=3)
              Hash Cond: (o.customer_id = c.id)
              ->  Parallel Seq Scan on orders  (cost=0.00..591234.12 rows=48276 width=72)
                                               (actual time=0.041..39841.223 rows=17282 loops=3)
                    Filter: ((status = ANY ('{pending,processing}'::text[]))
                             AND (created_at >= (now() - '7 days'::interval)))
                    Rows Removed by Filter: 16284381
              ->  Hash  (cost=112341.00..112341.00 rows=984127 width=52)
                        (actual time=2341.112..2341.112 rows=984127 loops=1)
                    Buckets: 1048576  Batches: 2  Memory Usage: 32768kB
                    ->  Seq Scan on customers  (cost=0.00..112341.00 rows=984127 width=52)
                                               (actual time=0.022..1841.334 rows=984127 loops=1)

Planning Time: 42.831 ms
Execution Time: 45,112.447 ms
"""

SQL_TABLE_STATS_OUTPUT = """\
Query: SELECT relname, n_live_tup, n_dead_tup,
              ROUND(n_dead_tup::numeric / NULLIF(n_live_tup + n_dead_tup, 0) * 100, 2) AS dead_pct,
              last_vacuum, last_autovacuum, last_analyze, last_autoanalyze,
              seq_scan, seq_tup_read, idx_scan, idx_tup_fetch
       FROM pg_stat_user_tables
       WHERE relname IN ('orders', 'customers')
       ORDER BY n_dead_tup DESC;

Results:
relname   | n_live_tup | n_dead_tup | dead_pct | last_autovacuum          | last_autoanalyze         | seq_scan | idx_scan
----------|------------|------------|----------|--------------------------|--------------------------|----------|---------
orders    | 49,284,127 | 18,341,029 | 27.12    | 2026-07-14 03:22:11 UTC  | 2026-07-14 03:22:11 UTC  | 8,847    | 0
customers | 984,127    |    284,031 | 22.41    | 2026-07-19 02:11:44 UTC  | 2026-07-19 02:11:44 UTC  | 51,823   | 12,841

Notes:
- orders.idx_scan = 0 for the past 8 days (index not used)
- orders table last grew by 4.2M rows on 2026-07-14 (migration job)
- autovacuum_vacuum_scale_factor for orders = 0.2 (default, not tuned for large table)
- No manual VACUUM or ANALYZE run since 2026-07-14 migration

Execution time: 312ms
"""

REST_PIPELINE_OUTPUT = """\
GET /api/v2/pipelines/etl_orders_daily/runs?limit=10

HTTP/1.1 200 OK
Content-Type: application/json

{
  "pipeline_id": "etl_orders_daily",
  "runs": [
    {"run_id": "run_20260721_0300", "status": "failed",   "started": "2026-07-21T03:00:01Z", "finished": "2026-07-21T03:15:01Z", "duration_s": 900,  "error": "QueryTimeout: execution exceeded 900s limit"},
    {"run_id": "run_20260720_0300", "status": "failed",   "started": "2026-07-20T03:00:02Z", "finished": "2026-07-20T03:15:02Z", "duration_s": 900,  "error": "QueryTimeout: execution exceeded 900s limit"},
    {"run_id": "run_20260719_0300", "status": "failed",   "started": "2026-07-19T03:00:01Z", "finished": "2026-07-19T03:15:01Z", "duration_s": 900,  "error": "QueryTimeout: execution exceeded 900s limit"},
    {"run_id": "run_20260718_0300", "status": "failed",   "started": "2026-07-18T03:00:00Z", "finished": "2026-07-18T03:15:00Z", "duration_s": 900,  "error": "QueryTimeout: execution exceeded 900s limit"},
    {"run_id": "run_20260717_0300", "status": "warning",  "started": "2026-07-17T03:00:01Z", "finished": "2026-07-17T03:12:44Z", "duration_s": 764,  "error": null},
    {"run_id": "run_20260716_0300", "status": "success",  "started": "2026-07-16T03:00:00Z", "finished": "2026-07-16T03:00:38Z", "duration_s": 38,   "error": null},
    {"run_id": "run_20260715_0300", "status": "success",  "started": "2026-07-15T03:00:01Z", "finished": "2026-07-15T03:00:41Z", "duration_s": 41,   "error": null},
    {"run_id": "run_20260714_0300", "status": "success",  "started": "2026-07-14T03:00:02Z", "finished": "2026-07-14T03:00:36Z", "duration_s": 36,   "error": null},
    {"run_id": "run_20260713_0300", "status": "success",  "started": "2026-07-13T03:00:00Z", "finished": "2026-07-13T03:00:33Z", "duration_s": 33,   "error": null},
    {"run_id": "run_20260712_0300", "status": "success",  "started": "2026-07-12T03:00:01Z", "finished": "2026-07-12T03:00:35Z", "duration_s": 35,   "error": null}
  ],
  "summary": {
    "consecutive_failures": 4,
    "last_success": "2026-07-16T03:00:38Z",
    "avg_duration_success": 36.6,
    "avg_duration_failure": 900.0
  }
}

Latency: 84ms
"""

BASH_PG_OUTPUT = """\
$ psql -c "SELECT state, wait_event_type, wait_event, COUNT(*) FROM pg_stat_activity WHERE datname='orders_db' GROUP BY 1,2,3 ORDER BY 4 DESC;" && psql -c "SELECT count(*) as total_connections, max_conn FROM pg_stat_activity, (SELECT setting::int as max_conn FROM pg_settings WHERE name='max_connections') s GROUP BY max_conn;"

 state  | wait_event_type |    wait_event    | count
--------|-----------------|------------------|-------
 active | Lock            | relation         |    14
 active | IO              | DataFileRead     |    38
 idle   |                 |                  |    91
 active | Client          | ClientRead       |     6
(4 rows)

 total_connections | max_conn
-------------------|----------
               149 | 200
(1 row)

$ psql -c "SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE state = 'active' AND wait_event_type = 'IO' ORDER BY duration DESC LIMIT 3;"

  pid  |    duration     | query
-------|-----------------|-------------------------------------------------------
 28441 | 00:38:14.112341 | SELECT o.id, o.customer_id ... FROM orders o JOIN ...
 28887 | 00:31:07.884210 | SELECT o.id, o.customer_id ... FROM orders o JOIN ...
 28103 | 00:22:41.334891 | SELECT o.id, o.customer_id ... FROM orders o JOIN ...

Exit code: 0
"""

TOOL_OUTPUTS = {
    "sql_explain_query":    SQL_EXPLAIN_OUTPUT,
    "sql_table_stats":      SQL_TABLE_STATS_OUTPUT,
    "rest_pipeline_status": REST_PIPELINE_OUTPUT,
    "bash_pg_activity":     BASH_PG_OUTPUT,
}

TOOL_CS_NAME = {
    "sql_explain_query":    "sql",
    "sql_table_stats":      "sql",
    "rest_pipeline_status": "rest_api",
    "bash_pg_activity":     "bash",
}

AGENT_TASK = (
    "Data pipeline incident: etl_orders_daily has failed 4 consecutive days with QueryTimeout. "
    "Last success was 2026-07-16. Runtime jumped from ~36s to 900s (timeout). "
    "Identify root cause and recommend the minimum fix to restore the pipeline."
)

DIAGNOSIS_PROMPT_TEMPLATE = """\
You are a database reliability engineer diagnosing a pipeline failure.

{context}

Based on the evidence above, provide:
1. Root cause (one sentence — name exact table, metric, value, and what changed when)
2. Contributing factors (bullet list, max 3)
3. Recommended fix (exact SQL or config change to run — be specific)
"""
