"""
format_detector.py — Detects output format from raw text, no LLM call.

Used as fallback when tool_name has no registered schema.
Runs on first 400 chars only — fast, deterministic.

Returns a schema key from TOOL_SCHEMAS or a detected format string
that schema_registry.py maps to the nearest built-in schema.
"""

from __future__ import annotations

import json
import re


# ---------------------------------------------------------------------------
# Format signatures — ordered by specificity, first match wins
# ---------------------------------------------------------------------------

def detect_format(raw_output: str) -> str:
    """
    Returns one of:
      'kubectl'  — k8s CLI output
      'sql'      — SQL result or error
      'rest_api' — HTTP response
      'file'     — config / YAML / key-value file content
      'bash'     — generic CLI / shell output (fallback)
    """
    sample = raw_output[:400].strip()
    if not sample:
        return "bash"

    # k8s: kubectl describe / logs / top / get output
    if _is_kubectl(sample):
        return "kubectl"

    # HTTP response: method + path, or status line, or JSON with status_code
    if _is_rest(sample):
        return "rest_api"

    # SQL: query keywords or error codes
    if _is_sql(sample):
        return "sql"

    # Structured file: YAML / TOML / dotenv / JSON config (not an HTTP response)
    if _is_file(sample):
        return "file"

    # Default
    return "bash"


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def _is_kubectl(s: str) -> bool:
    # High-confidence markers — only appear in kubectl output, not YAML files
    strong_markers = [
        r"^Name:\s+\S",
        r"^Namespace:\s+\S",
        r"^Status:\s+(Running|Failed|Pending|Terminating|CrashLoopBackOff|OOMKilled|Evicted)",
        r"Restart Count:",
        r"OOMKill",
        r"CrashLoopBackOff",
        r"kubectl\s+\w+",
        r"^KIND:\s+[A-Z]",          # kubectl table header — all-caps value (not YAML 'kind: Deployment')
        r"Controlled By:",
        r"^NAMESPACE\s+NAME\s+",    # kubectl get table header
        r"^Events:",
        r"^IP:\s+\d+\.\d+",
        r"Container ID:\s+containerd",
    ]
    # Exclude pure YAML files — they have apiVersion but none of the strong markers
    if re.search(r"^apiVersion:\s+\S", s, re.MULTILINE) and not re.search(
        r"(Restart Count:|Controlled By:|^Events:|OOMKill|CrashLoop)", s, re.MULTILINE
    ):
        return False
    return any(re.search(p, s, re.MULTILINE | re.IGNORECASE) for p in strong_markers)


def _is_rest(s: str) -> bool:
    rest_markers = [
        r"^(GET|POST|PUT|PATCH|DELETE)\s+/",
        r"^HTTP/[12]",
        r"^HTTP/[12]\.[01]\s+[1-5]\d{2}",
        r'"status_code"\s*:\s*[1-5]\d{2}',
        r"^[1-5]\d{2}\s+(OK|Created|Not Found|Internal Server Error|Unauthorized|Forbidden|Bad Request)",
        r"Content-Type:\s+application/json",
        r"X-Request-Id:",
        r"X-Response-Time:",
    ]
    return any(re.search(p, s, re.MULTILINE | re.IGNORECASE) for p in rest_markers)


def _is_sql(s: str) -> bool:
    sql_markers = [
        r"^(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|EXPLAIN)\s+",
        r"^ERROR\s+\d+",               # MySQL/MariaDB error
        r"^(ORA|PG|SQLITE)-\d+",       # Oracle/Postgres/SQLite error codes
        r"rows?\s+affected",
        r"Query OK",
        r"^\s*\|.+\|.+\|",            # table output with pipes
        r"rows in set",
        r"^psql:",
        r"SQLSTATE",
    ]
    return any(re.search(p, s, re.MULTILINE | re.IGNORECASE) for p in sql_markers)


def _is_file(s: str) -> bool:
    file_markers = [
        r"^apiVersion:",               # Kubernetes YAML
        r"^kind:\s+\w+",
        r"^---\s*$",                   # YAML document separator
        r"^\[[\w\s]+\]$",             # TOML section header
        r"^[A-Z_]{2,}=\S",            # dotenv KEY=VALUE
        r"^\s{2,}\w[\w_-]+:\s+\S",    # indented YAML key: value
        r"^<\?xml",                    # XML
        r"^\{[\s\S]{0,50}\"[\w]+\":", # JSON object
    ]
    # Exclude if already matched as REST (JSON HTTP response)
    if _is_rest(s):
        return False
    return any(re.search(p, s, re.MULTILINE) for p in file_markers)


# ---------------------------------------------------------------------------
# Format → model routing hint (complements TOOL_MODEL_MAP)
# ---------------------------------------------------------------------------

FORMAT_MODEL_HINTS: dict[str, str] = {
    "kubectl":  "claude-haiku-4-5-20251001",
    "bash":     "claude-haiku-4-5-20251001",
    "sql":      "claude-sonnet-4-6",
    "rest_api": "claude-sonnet-4-6",
    "file":     "claude-sonnet-4-6",
}
