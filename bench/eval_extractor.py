"""
eval_extractor.py — Evaluates extractor.py output quality against ground truth.

Scoring:
  - field_coverage:    fraction of non-null expected fields that are non-null in output
  - field_accuracy:    fraction of filled fields that match expected value (fuzzy string match)
  - confidence_bound:  pass if output confidence is within expected min/max bounds
  - root_cause_score:  semantic similarity of root_cause to expected (keyword overlap)

Usage:
  python bench/eval_extractor.py --model claude-haiku-4-5-20251001
  python bench/eval_extractor.py --model claude-sonnet-4-6 --tool kubectl
"""

from __future__ import annotations

import argparse
import json
import statistics
import uuid
from pathlib import Path
from typing import Any


EVAL_PATH = Path(__file__).parent / "eval_datasets" / "extractor_eval.json"


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_field_coverage(output: dict, expected: dict) -> float:
    """Fraction of expected non-null fields that are non-null in output."""
    expected_fields = {k: v for k, v in expected.items()
                       if not k.startswith("confidence") and v is not None
                       and k not in ("root_cause", "entity_targets", "metric_impact")}
    if not expected_fields:
        return 1.0
    filled = sum(1 for k in expected_fields if output.get(k) is not None)
    return filled / len(expected_fields)


def score_field_accuracy(output: dict, expected: dict) -> float:
    """Fraction of expected non-null fields where output matches expected."""
    expected_fields = {k: v for k, v in expected.items()
                       if not k.startswith("confidence") and v is not None
                       and k not in ("root_cause", "entity_targets", "metric_impact")}
    if not expected_fields:
        return 1.0
    correct = 0
    for k, exp_val in expected_fields.items():
        out_val = output.get(k)
        if out_val is None:
            continue
        if isinstance(exp_val, int):
            correct += 1 if int(out_val) == exp_val else 0
        elif isinstance(exp_val, list):
            exp_set = {str(x).lower() for x in exp_val}
            out_set = {str(x).lower() for x in (out_val if isinstance(out_val, list) else [out_val])}
            correct += 1 if exp_set == out_set else (0.5 if exp_set & out_set else 0)
        else:
            correct += 1 if str(out_val).lower() == str(exp_val).lower() else 0
    return correct / len(expected_fields)


def score_confidence_bound(output_conf: float, expected: dict) -> bool:
    """Pass if output confidence is within expected min/max bounds."""
    if "confidence_min" in expected and output_conf < expected["confidence_min"]:
        return False
    if "confidence_max" in expected and output_conf > expected["confidence_max"]:
        return False
    return True


def score_root_cause(output_rc: str, expected_rc: str) -> float:
    """Keyword overlap between output and expected root_cause."""
    if not output_rc or not expected_rc:
        return 0.0
    exp_words = set(expected_rc.lower().split()) - {"a", "an", "the", "is", "in", "on", "at", "of", "to"}
    out_words = set(output_rc.lower().split())
    overlap = exp_words & out_words
    return len(overlap) / len(exp_words) if exp_words else 0.0


def score_entity_targets(output_entities: list, expected_entities: list) -> float:
    """Overlap between expected and output entity_targets."""
    if not expected_entities:
        return 1.0
    exp = {str(e).lower() for e in expected_entities}
    out = {str(e).lower() for e in (output_entities or [])}
    return len(exp & out) / len(exp)


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval(model: str, tool_filter: str | None = None) -> None:
    from core.extractor import Extractor, TOOL_SCHEMAS, build_extraction_prompt

    cases = json.loads(EVAL_PATH.read_text())
    if tool_filter:
        cases = [c for c in cases if c["tool"] == tool_filter]

    results = []
    for case in cases:
        tool = case["tool"]
        raw = case["raw_output"]
        expected = case["expected"]

        # Build prompt and call extractor directly (bypasses eviction store for eval)
        schema = TOOL_SCHEMAS.get(tool, {})
        prompt = build_extraction_prompt(tool, raw, schema)

        # Wire to actual model call here
        output = _call_model(model, prompt)

        scores = {
            "id": case["id"],
            "tool": tool,
            "description": case["description"],
            "field_coverage": score_field_coverage(output, expected),
            "field_accuracy": score_field_accuracy(output, expected),
            "confidence_bound_pass": score_confidence_bound(output.get("confidence", 0), expected),
            "root_cause_score": score_root_cause(output.get("root_cause", ""), expected.get("root_cause", "")),
            "entity_target_score": score_entity_targets(output.get("entity_targets", []), expected.get("entity_targets", [])),
            "output_confidence": output.get("confidence"),
            "output_root_cause": output.get("root_cause"),
        }
        scores["composite"] = statistics.mean([
            scores["field_coverage"],
            scores["field_accuracy"],
            1.0 if scores["confidence_bound_pass"] else 0.0,
            scores["root_cause_score"],
            scores["entity_target_score"],
        ])
        results.append(scores)
        _print_case(scores)

    _print_summary(results, model)


def _call_model(model: str, prompt: str) -> dict:
    """Replace with real SDK call. Returns parsed JSON dict."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system="You are a structured data extractor. Output valid JSON only.",
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    # Strip markdown code fences if model wraps output
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _print_case(s: dict) -> None:
    status = "PASS" if s["composite"] >= 0.75 else "WARN" if s["composite"] >= 0.55 else "FAIL"
    print(f"[{status}] {s['id']} ({s['tool']}) — composite={s['composite']:.2f} "
          f"cov={s['field_coverage']:.2f} acc={s['field_accuracy']:.2f} "
          f"rc={s['root_cause_score']:.2f} conf_bound={'OK' if s['confidence_bound_pass'] else 'FAIL'}")


def _print_summary(results: list, model: str) -> None:
    composites = [r["composite"] for r in results]
    by_tool: dict[str, list] = {}
    for r in results:
        by_tool.setdefault(r["tool"], []).append(r["composite"])

    print(f"\n{'='*60}")
    print(f"Model: {model}  |  Cases: {len(results)}")
    print(f"Overall composite: {statistics.mean(composites):.3f}  "
          f"(min={min(composites):.2f} max={max(composites):.2f})")
    print("\nBy tool:")
    for tool, scores in sorted(by_tool.items()):
        print(f"  {tool:12s}  mean={statistics.mean(scores):.3f}  n={len(scores)}")
    pass_rate = sum(1 for r in results if r["composite"] >= 0.75) / len(results)
    print(f"\nPass rate (composite >= 0.75): {pass_rate:.0%}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--tool", default=None, help="Filter to single tool type")
    args = parser.parse_args()
    run_eval(args.model, args.tool)
