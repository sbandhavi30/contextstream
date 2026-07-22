"""
schema_registry.py — Loads and resolves extraction schemas for any tool.

Resolution order (first match wins):
  1. Exact tool name in user-defined registry (configs/tool_schemas/*.yaml)
  2. Exact tool name in built-in TOOL_SCHEMAS
  3. Auto-detected format from raw output (format_detector.py)
  4. Fallback: bash schema

User-defined schemas live in configs/tool_schemas/<tool_name>.yaml.
Each YAML file defines:
  - fields: dict of field_name → description (same format as built-in schemas)
  - base: optional — inherit from a built-in schema ('bash'|'sql'|'rest_api'|'kubectl'|'file')
  - model: optional — preferred model override for this tool
  - description: optional — human-readable description

Example: configs/tool_schemas/jira.yaml
  description: "Jira API tool output"
  base: rest_api
  model: claude-sonnet-4-6
  fields:
    issue_key: "string — Jira issue key e.g. 'PROJ-123'"
    summary: "string — issue title"
    status: "string — 'To Do' | 'In Progress' | 'Done' | 'Blocked'"
    assignee: "string | null — assignee display name"
    priority: "string | null — 'Highest'|'High'|'Medium'|'Low'|'Lowest'"
    error: "string | null — API error if present"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from core.extractor import TOOL_SCHEMAS, BASH_SCHEMA, TOOL_MODEL_MAP
from core.format_detector import detect_format, FORMAT_MODEL_HINTS


# Default location — can be overridden via CONTEXTSTREAM_SCHEMA_DIR env var
_DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "configs" / "tool_schemas"


class SchemaRegistry:
    """
    Resolves the best extraction schema + model for any tool name.
    Loads user-defined schemas from YAML on first use (lazy, cached).
    """

    def __init__(self, schema_dir: Path | str | None = None):
        self._dir = Path(schema_dir or os.environ.get(
            "CONTEXTSTREAM_SCHEMA_DIR", _DEFAULT_SCHEMA_DIR
        ))
        self._user_schemas: dict[str, dict] = {}
        self._user_models: dict[str, str] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_schema(self, tool_name: str, raw_output: str = "") -> dict[str, Any]:
        """Return the best extraction schema for this tool + output."""
        self._ensure_loaded()

        # 1. User-defined exact match
        if tool_name in self._user_schemas:
            return self._user_schemas[tool_name]

        # 2. Built-in exact match
        if tool_name in TOOL_SCHEMAS:
            return TOOL_SCHEMAS[tool_name]

        # 3. Auto-detect from output shape
        if raw_output:
            detected = detect_format(raw_output)
            return TOOL_SCHEMAS.get(detected, BASH_SCHEMA)

        # 4. Fallback
        return BASH_SCHEMA

    def resolve_model(self, tool_name: str, raw_output: str = "",
                      default: str = "claude-haiku-4-5-20251001") -> str:
        """Return the best extraction model for this tool."""
        self._ensure_loaded()

        # 1. User-defined model override
        if tool_name in self._user_models:
            return self._user_models[tool_name]

        # 2. Built-in tool model map
        if tool_name in TOOL_MODEL_MAP:
            return TOOL_MODEL_MAP[tool_name]

        # 3. Auto-detect format → model hint
        if raw_output:
            detected = detect_format(raw_output)
            return FORMAT_MODEL_HINTS.get(detected, default)

        return default

    def list_registered(self) -> dict[str, str]:
        """Return {tool_name: source} for all registered schemas."""
        self._ensure_loaded()
        result = {k: "builtin" for k in TOOL_SCHEMAS}
        result.update({k: "user-defined" for k in self._user_schemas})
        return result

    # ------------------------------------------------------------------
    # YAML loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._dir.exists():
            return
        if not _YAML_AVAILABLE:
            return

        for yaml_path in sorted(self._dir.glob("*.yaml")):
            tool_name = yaml_path.stem
            try:
                raw = yaml.safe_load(yaml_path.read_text())
                if not isinstance(raw, dict):
                    continue

                fields = raw.get("fields", {})
                base = raw.get("base")

                # Merge with base schema if specified
                if base and base in TOOL_SCHEMAS:
                    merged = dict(TOOL_SCHEMAS[base])
                    merged.update(fields)
                    schema = merged
                else:
                    schema = fields

                if schema:
                    self._user_schemas[tool_name] = schema

                if "model" in raw:
                    self._user_models[tool_name] = raw["model"]

            except Exception:
                # Bad YAML — skip silently, don't crash the engine
                continue

    def reload(self) -> None:
        """Force re-read of all YAML files (useful in long-running processes)."""
        self._loaded = False
        self._user_schemas.clear()
        self._user_models.clear()
        self._ensure_loaded()


# Module-level singleton — shared across all Extractor instances
_registry: SchemaRegistry | None = None

def get_registry(schema_dir: Path | str | None = None) -> SchemaRegistry:
    global _registry
    if _registry is None:
        _registry = SchemaRegistry(schema_dir)
    return _registry
