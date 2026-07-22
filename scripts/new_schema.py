#!/usr/bin/env python3
"""
new_schema.py — Interactive scaffold for adding a new tool schema.

Usage:
  python scripts/new_schema.py
  python scripts/new_schema.py --tool jira --base rest_api --model sonnet

Guides the developer through:
  1. Tool name + description
  2. Base schema selection
  3. Field definition (interactive loop)
  4. Model selection
  5. Writes YAML + validation + test stub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCHEMA_DIR = ROOT / "configs" / "tool_schemas"
TEST_DIR   = ROOT / "bench" / "eval_datasets"

BUILTIN_BASES = ["bash", "sql", "rest_api", "kubectl", "file"]
MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
}
MODEL_GUIDANCE = {
    "bash":     "haiku",
    "kubectl":  "haiku",
    "sql":      "sonnet",
    "rest_api": "sonnet",
    "file":     "sonnet",
}


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or default


def ask_choice(prompt: str, choices: list[str], default: str = "") -> str:
    print(f"{prompt}")
    for i, c in enumerate(choices, 1):
        marker = " (recommended)" if c == default else ""
        print(f"  {i}. {c}{marker}")
    while True:
        raw = input(f"Choice [1-{len(choices)}]: ").strip()
        if not raw and default:
            return default
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            pass
        print("  Invalid — enter a number.")


def collect_fields(base: str) -> dict[str, str]:
    print("\nDefine fields for this tool (beyond what the base schema provides).")
    print("Format: field_name → description string")
    print("Examples:")
    print("  issue_key      → string — Jira issue key e.g. 'PROJ-123'")
    print("  rows_affected  → integer | null — row count if present")
    print("  error          → string | null — error message if present")
    print("Type 'done' when finished.\n")

    from core.extractor import TOOL_SCHEMAS
    base_keys = set(TOOL_SCHEMAS.get(base, {}).keys())
    if base_keys:
        print(f"Base '{base}' already provides: {', '.join(sorted(base_keys))}")
        print("Only add fields NOT in the base.\n")

    fields: dict[str, str] = {}
    while True:
        name = input("  field name (or 'done'): ").strip()
        if name.lower() in ("done", ""):
            break
        if name in base_keys:
            print(f"  '{name}' already in base schema — skip or use different name.")
            continue
        desc = input(f"  description for '{name}': ").strip()
        if desc:
            fields[name] = desc
        print()
    return fields


def write_yaml(tool_name: str, description: str, base: str,
               model_key: str, fields: dict[str, str]) -> Path:
    lines = [
        f'description: "{description}"',
        f"base: {base}",
        f"model: {MODELS[model_key]}",
        "fields:",
    ]
    if fields:
        for k, v in fields.items():
            lines.append(f'  {k}: "{v}"')
    else:
        lines.append("  # Add your fields here")

    path = SCHEMA_DIR / f"{tool_name}.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_test_stub(tool_name: str) -> Path:
    stub_path = TEST_DIR / f"{tool_name}_eval.json"
    if stub_path.exists():
        return stub_path

    content = f"""[
  {{
    "id": "{tool_name}_001",
    "tool": "{tool_name}",
    "description": "TODO: describe what this raw output represents",
    "raw_output": "TODO: paste real tool output here",
    "expected": {{
      "root_cause": "TODO: expected one-sentence synthesis",
      "entity_targets": ["TODO: affected entity names"],
      "confidence_min": 0.80
    }}
  }}
]
"""
    stub_path.write_text(content)
    return stub_path


def validate_schema(tool_name: str) -> bool:
    """Run the schema registry loader and check the schema resolves cleanly."""
    from core.schema_registry import SchemaRegistry
    reg = SchemaRegistry()
    reg.reload()
    schema = reg.resolve_schema(tool_name)
    if not schema:
        print(f"  ERROR: schema for '{tool_name}' resolved empty.")
        return False
    if "confidence" not in schema:
        print(f"  WARNING: schema has no 'confidence' field — add it for calibration.")
    print(f"  OK: {len(schema)} fields resolved ({', '.join(list(schema.keys())[:4])}...)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a new ContextStream tool schema")
    parser.add_argument("--tool",  default=None, help="Tool name e.g. 'jira'")
    parser.add_argument("--base",  default=None, choices=BUILTIN_BASES)
    parser.add_argument("--model", default=None, choices=list(MODELS.keys()))
    args = parser.parse_args()

    print("\n=== ContextStream: New Tool Schema ===\n")

    # Tool name
    tool_name = args.tool or ask("Tool name (snake_case, matches how agent calls the tool)")
    if not tool_name:
        print("Tool name required."); sys.exit(1)

    existing = SCHEMA_DIR / f"{tool_name}.yaml"
    if existing.exists():
        print(f"\nSchema already exists at {existing}")
        overwrite = ask("Overwrite? (y/n)", "n")
        if overwrite.lower() != "y":
            print("Aborted."); sys.exit(0)

    # Description
    description = ask("One-line description e.g. 'Jira REST API — issue reads and transitions'")

    # Base schema
    base = args.base or ask_choice(
        "\nBase schema to inherit from:",
        BUILTIN_BASES,
        default="bash"
    )

    # Model
    recommended_model = MODEL_GUIDANCE.get(base, "sonnet")
    model_key = args.model or ask_choice(
        "\nExtraction model:",
        list(MODELS.keys()),
        default=recommended_model
    )

    # Fields
    print()
    fields = collect_fields(base)

    # Write files
    yaml_path = write_yaml(tool_name, description, base, model_key, fields)
    test_path = write_test_stub(tool_name)

    print(f"\n{'='*40}")
    print(f"Schema written:    {yaml_path.relative_to(ROOT)}")
    print(f"Test stub written: {test_path.relative_to(ROOT)}")
    print(f"\nValidating...")
    ok = validate_schema(tool_name)

    if ok:
        print(f"\nNext steps:")
        print(f"  1. Add real raw_output examples to {test_path.relative_to(ROOT)}")
        print(f"  2. Run: python bench/eval_extractor.py --tool {tool_name}")
        print(f"  3. Iterate schema until pass rate >= 80%")
        print(f"  4. Submit PR: git add {yaml_path.relative_to(ROOT)} {test_path.relative_to(ROOT)}")
    else:
        print(f"\nSchema validation failed — check {yaml_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
