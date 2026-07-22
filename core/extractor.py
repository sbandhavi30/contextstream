"""
extractor.py — Structured lesson extraction from raw tool output.

Uses a cheap LLM (Haiku / GPT-4o-mini) with tool-specific schemas.
Never freeform summarization — typed fields only.
If a field cannot be filled with confidence, it stays None.
Confidence drops with ambiguity; caller decides whether to re-page.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Iterator, Optional

from pydantic import BaseModel, Field

# Lazy import to avoid circular deps — registry imports from extractor
def _get_registry():
    from core.schema_registry import get_registry
    return get_registry()


# ---------------------------------------------------------------------------
# Base lesson model
# ---------------------------------------------------------------------------

class Lesson(BaseModel):
    lesson_id: str
    tool_source: str
    root_cause: str
    entity_targets: list[str] = Field(default_factory=list)
    metric_impact: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    raw_ref: str
    fields: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool-specific extraction schemas
# ---------------------------------------------------------------------------

KUBECTL_SCHEMA = {
    "resource_type": "string — 'pod' | 'deployment' | 'node' | 'service' | 'other'",
    "resource_name": "string — exact k8s resource name",
    "namespace": "string | null — k8s namespace if present",
    "condition": "string — 'OOMKilled' | 'CrashLoopBackOff' | 'Pending' | 'Running' | 'Error' | 'Evicted' | other observed condition",
    "metric": "string | null — exact observed metric with unit e.g. 'memory=490Mi' 'cpu=2.3' 'restarts=7'",
    "cause": "string | null — root cause if determinable from output",
    "event_timestamp": "string | null — ISO timestamp of the event if present in output",
    "confidence": "float 0.0-1.0 — lower if output is ambiguous or truncated",
}

SQL_SCHEMA = {
    "query_type": "string — 'SELECT' | 'INSERT' | 'UPDATE' | 'DELETE' | 'DDL' | 'EXPLAIN'",
    "tables_affected": "list[string] — table names present in query or result",
    "rows_affected": "integer | null — row count if present",
    "result_summary": "string — what the query returned or did, one sentence",
    "error_code": "string | null — SQL error code if present e.g. 'ORA-00942' 'ERROR 1064'",
    "error_message": "string | null — error message verbatim if present",
    "execution_ms": "integer | null — execution time in ms if present",
    "confidence": "float 0.0-1.0",
}

REST_SCHEMA = {
    "method": "string — 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'",
    "endpoint": "string — URL path only, no host, no query params",
    "status_code": "integer — HTTP status code",
    "entity": "string | null — primary resource being operated on e.g. 'user' 'order' 'payment'",
    "entity_id": "string | null — ID of the entity if present",
    "state_change": "string | null — what changed e.g. 'user.status: active→suspended'",
    "error_message": "string | null — error body if status >= 400",
    "latency_ms": "integer | null — response time if present",
    "confidence": "float 0.0-1.0 — MUST be < 0.75 if error_message is generic e.g. 'unexpected error' 'internal error' 'something went wrong'. MUST be < 0.65 if status=5xx AND error_message gives no specific cause.",
}

BASH_SCHEMA = {
    "command": "string — the command that was executed (truncate at 120 chars)",
    "exit_code": "integer | null — exit code if present",
    "outcome": "string — 'success' | 'failure' | 'partial' | 'unknown'",
    "key_output": "string — the single most diagnostic line or value from stdout/stderr",
    "error_type": "string | null — class of error if failed e.g. 'permission denied' 'not found' 'timeout'",
    "side_effects": "list[string] — files created/deleted/modified, processes started/killed, if determinable",
    "confidence": "float 0.0-1.0",
}

FILE_SCHEMA = {
    "file_path": "string — path of the file read",
    "file_type": "string — 'config' | 'log' | 'code' | 'data' | 'manifest' | 'other'",
    "key_values": "dict — COPY the key name EXACTLY as it appears in the file. Examples: file has 'DB_URL=...' → key is 'DB_URL' not 'database_url'. File has 'memory: 512Mi' → key is 'memory' not 'memory_limit'. Up to 5 most operationally relevant entries.",
    "anomalies": "list[string] — values that are missing, commented-out, misconfigured, or unexpected. Format: '<exact_key>: <observed value or absence> — <why anomalous>'",
    "references": "list[string] — hostnames, file paths, service names referenced in the file",
    "confidence": "float 0.0-1.0",
}

TOOL_SCHEMAS: dict[str, dict] = {
    "kubectl": KUBECTL_SCHEMA,
    "sql": SQL_SCHEMA,
    "rest_api": REST_SCHEMA,
    "bash": BASH_SCHEMA,
    "file": FILE_SCHEMA,
}

# Haiku wins structured operational outputs (kubectl, bash).
# Sonnet wins schema/API outputs (sql, rest_api, file).
# Override per-tool; fallback to constructor default.
TOOL_MODEL_MAP: dict[str, str] = {
    "kubectl":  "claude-haiku-4-5-20251001",
    "bash":     "claude-haiku-4-5-20251001",
    "sql":      "claude-sonnet-4-6",
    "rest_api": "claude-sonnet-4-6",
    "file":     "claude-sonnet-4-6",
}


# ---------------------------------------------------------------------------
# Extraction prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a structured data extractor for an LLM agent memory system.
Your only job is to extract a JSON object from raw tool output.
Rules:
- Output valid JSON only. No prose, no markdown, no explanation.
- Fill every field if evidence exists in the raw output.
- Set a field to null if evidence is absent or ambiguous.
- Set confidence < 0.7 if output is truncated, ambiguous, or contradictory.
- Never infer or hallucinate values not present in the raw output.
- If the raw output is empty or unparseable, return {"confidence": 0.1, "root_cause": "unparseable output"} plus nulls.
- root_cause MUST include: the exact resource/entity name, exact metric values with units, and the specific condition. Bad: "memory issue detected". Good: "Pod web-backend OOMKilled — memory limit 512Mi breached after 7 restarts".
- confidence calibration: complete unambiguous output with all key fields present → confidence ≥ 0.85. Partial or ambiguous output → confidence 0.55–0.75. Generic error with no specific cause → confidence ≤ 0.65. Empty/unparseable → confidence ≤ 0.20."""

def build_extraction_prompt(tool_name: str, raw_output: str, schema: dict) -> str:
    schema_str = json.dumps(schema, indent=2)
    # Hard cap raw output at 6000 chars — extractor model has small context
    truncated = raw_output[:6000]
    truncation_note = "\n[OUTPUT TRUNCATED AT 6000 CHARS]" if len(raw_output) > 6000 else ""
    return f"""Tool: {tool_name}

Extract a JSON object with exactly these fields:
{schema_str}

Also include these fields in every response:
- "root_cause": "string — ONE sentence. Must name the exact resource, exact metric with units, and specific condition. No vague prose."
- "entity_targets": ["exact resource names, IDs, table names, file paths affected — no generic words like 'database' or 'server'"]
- "metric_impact": "string | null — primary quantitative value with unit e.g. 'memory=490Mi' '1842 rows' 'latency=3042ms'"

Raw tool output:
---
{truncated}{truncation_note}
---

JSON:"""


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

class Extractor:
    """
    Calls a cheap LLM to extract a structured lesson from raw tool output.
    Model is configurable — default Haiku for cost, Sonnet for accuracy.
    """

    def __init__(
        self,
        eviction_store,
        model: str = "claude-haiku-4-5-20251001",
        fallback_model: str = "claude-sonnet-4-6",
        confidence_threshold: float = 0.6,
    ):
        self.store = eviction_store
        self.model = model
        self.fallback_model = fallback_model
        self.confidence_threshold = confidence_threshold

    def extract(self, tool_name: str, raw_ref: str, lesson_id: str) -> Lesson:
        raw_output = self.store.fetch(raw_ref)

        # Schema + model resolved via registry (user-defined → builtin → auto-detect)
        registry = _get_registry()
        schema = registry.resolve_schema(tool_name, raw_output)
        primary_model = registry.resolve_model(tool_name, raw_output, default=self.model)

        prompt = build_extraction_prompt(tool_name, raw_output, schema)
        result = self._call_llm(prompt, model=primary_model)

        # Retry with fallback model if confidence too low
        if result.get("confidence", 0) < self.confidence_threshold:
            result = self._call_llm(prompt, model=self.fallback_model)

        return Lesson(
            lesson_id=lesson_id,
            tool_source=tool_name,
            root_cause=result.pop("root_cause", "extraction failed"),
            entity_targets=result.pop("entity_targets", []),
            metric_impact=result.pop("metric_impact", None),
            confidence=result.pop("confidence", 0.5),
            raw_ref=raw_ref,
            fields=result,  # remaining tool-specific fields
        )

    def _call_llm(self, prompt: str, model: str) -> dict:
        """Call Anthropic API and parse JSON response."""
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("pip install anthropic") from e

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = message.content[0].text.strip()

        # Strip markdown fences if model wraps in ```json ... ```
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # Last-resort: extract first {...} block
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"confidence": 0.1, "root_cause": f"unparseable response: {raw_text[:120]}"}
