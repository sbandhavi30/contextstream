"""
extractor.py — Structured lesson extraction from raw tool output.

Uses a cheap LLM (Haiku / GPT-4o-mini) with tool-specific schemas.
Never freeform summarization — typed fields only.
If a field cannot be filled with confidence, it stays None.
Confidence drops with ambiguity; caller decides whether to re-page.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterator, Optional

from pydantic import BaseModel, Field


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
    "confidence": "float 0.0-1.0",
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
    "key_values": "dict[string, string] — up to 5 most relevant key-value pairs or settings extracted",
    "anomalies": "list[string] — any values that look misconfigured, missing, or unexpected",
    "references": "list[string] — other files or services this file references",
    "confidence": "float 0.0-1.0",
}

TOOL_SCHEMAS: dict[str, dict] = {
    "kubectl": KUBECTL_SCHEMA,
    "sql": SQL_SCHEMA,
    "rest_api": REST_SCHEMA,
    "bash": BASH_SCHEMA,
    "file": FILE_SCHEMA,
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
- If the raw output is empty or unparseable, return {"confidence": 0.1, "root_cause": "unparseable output"} plus nulls."""

def build_extraction_prompt(tool_name: str, raw_output: str, schema: dict) -> str:
    schema_str = json.dumps(schema, indent=2)
    # Hard cap raw output at 6000 chars — extractor model has small context
    truncated = raw_output[:6000]
    truncation_note = "\n[OUTPUT TRUNCATED AT 6000 CHARS]" if len(raw_output) > 6000 else ""
    return f"""Tool: {tool_name}

Extract a JSON object with exactly these fields:
{schema_str}

Also include these fields in every response:
- "root_cause": "string — one sentence synthesis of what happened or was found"
- "entity_targets": ["list of affected entity names"]
- "metric_impact": "string | null — primary quantitative impact if any"

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
        schema = TOOL_SCHEMAS.get(tool_name, BASH_SCHEMA)  # bash as default
        prompt = build_extraction_prompt(tool_name, raw_output, schema)

        result = self._call_llm(prompt, model=self.model)

        # Retry with stronger model if confidence too low
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
        """Stub — replace with actual Anthropic/OpenAI SDK call."""
        raise NotImplementedError(
            "Wire up to Anthropic or OpenAI SDK. "
            "See configs/tool_schemas/ for expected output examples."
        )
