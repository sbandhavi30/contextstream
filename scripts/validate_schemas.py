#!/usr/bin/env python3
"""
validate_schemas.py — Validates all schemas in configs/tool_schemas/.

Checks:
  - YAML parses without error
  - Required keys present (fields, confidence field)
  - Base schema exists if specified
  - No field name collisions that would shadow base fields silently
  - model value is a known model ID

Usage:
  python scripts/validate_schemas.py
  python scripts/validate_schemas.py --fix   # auto-add missing confidence field
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run: pip install pyyaml")
    sys.exit(1)

from core.extractor import TOOL_SCHEMAS

SCHEMA_DIR = ROOT / "configs" / "tool_schemas"
KNOWN_MODELS = {
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "gpt-4o-mini",
    "gpt-4o",
}


def validate_file(path: Path, fix: bool = False) -> list[str]:
    """Returns list of error strings. Empty = valid."""
    errors: list[str] = []
    tool_name = path.stem

    # Parse
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"]

    if not isinstance(raw, dict):
        return ["Root must be a YAML mapping"]

    # Required: description
    if not raw.get("description"):
        errors.append("Missing 'description' key")

    # Base schema check
    base = raw.get("base")
    if base and base not in TOOL_SCHEMAS:
        errors.append(f"Unknown base schema '{base}' — must be one of {list(TOOL_SCHEMAS.keys())}")

    # Fields
    fields = raw.get("fields", {})
    if not isinstance(fields, dict):
        errors.append("'fields' must be a mapping")
    elif not fields:
        errors.append("'fields' is empty — add at least one tool-specific field")

    # Confidence field
    if fields and "confidence" not in fields:
        if fix:
            fields["confidence"] = "float 0.0-1.0"
            raw["fields"] = fields
            path.write_text(yaml.dump(raw, default_flow_style=False, allow_unicode=True))
            print(f"  FIXED: added 'confidence' field to {path.name}")
        else:
            errors.append("Missing 'confidence' field in fields — needed for calibration")

    # Model check
    model = raw.get("model")
    if model and model not in KNOWN_MODELS:
        errors.append(f"Unknown model '{model}' — known: {sorted(KNOWN_MODELS)}")

    # Shadow check — user field names that override base fields (warn, not error)
    if base and base in TOOL_SCHEMAS:
        base_keys = set(TOOL_SCHEMAS[base].keys())
        user_keys = set(fields.keys()) if isinstance(fields, dict) else set()
        shadows = base_keys & user_keys - {"confidence"}
        if shadows:
            errors.append(f"WARN: fields shadow base '{base}': {sorted(shadows)} — intentional override or typo?")

    # Field description quality check
    if isinstance(fields, dict):
        for fname, fdesc in fields.items():
            if not isinstance(fdesc, str) or len(fdesc) < 10:
                errors.append(f"Field '{fname}': description too short or not a string")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Auto-fix minor issues")
    args = parser.parse_args()

    if not SCHEMA_DIR.exists():
        print(f"Schema dir not found: {SCHEMA_DIR}")
        sys.exit(1)

    yamls = sorted(SCHEMA_DIR.glob("*.yaml"))
    if not yamls:
        print("No schemas found.")
        sys.exit(0)

    total = 0
    failed = 0
    warned = 0

    for path in yamls:
        errors = validate_file(path, fix=args.fix)
        hard_errors = [e for e in errors if not e.startswith("WARN")]
        warnings    = [e for e in errors if e.startswith("WARN")]

        if hard_errors:
            print(f"FAIL  {path.name}")
            for e in hard_errors: print(f"      {e}")
            failed += 1
        elif warnings:
            print(f"WARN  {path.name}")
            for w in warnings: print(f"      {w}")
            warned += 1
        else:
            print(f"OK    {path.name}")
        total += 1

    print(f"\n{total} schemas — {total-failed-warned} OK, {warned} warnings, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
