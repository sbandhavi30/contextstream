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
import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Allow running from any directory: bench/eval_extractor.py or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env if present (ANTHROPIC_API_KEY etc.)
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

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


def _score_value(out_val, exp_val) -> float:
    """Score a single field value. Returns 0.0–1.0."""
    if out_val is None:
        return 0.0
    if isinstance(exp_val, int):
        try:
            return 1.0 if int(out_val) == exp_val else 0.0
        except (ValueError, TypeError):
            return 0.0
    if isinstance(exp_val, list):
        exp_set = {str(x).lower() for x in exp_val}
        out_list = out_val if isinstance(out_val, list) else [out_val]
        out_set = {str(x).lower() for x in out_list}
        if exp_set == out_set:
            return 1.0
        overlap = exp_set & out_set
        return len(overlap) / len(exp_set) if exp_set else 0.0
    if isinstance(exp_val, dict):
        # Dict fields (e.g. key_values): score key overlap + value match
        if not isinstance(out_val, dict):
            return 0.0
        if not exp_val:
            return 1.0
        key_scores = []
        for ek, ev in exp_val.items():
            # Accept case-insensitive key match
            ov = out_val.get(ek) or out_val.get(ek.lower()) or out_val.get(ek.upper())
            if ov is None:
                key_scores.append(0.0)
            else:
                # Value: exact or token-F1
                key_scores.append(_str_similarity(str(ov), str(ev)))
        return sum(key_scores) / len(key_scores)
    # String: exact match = 1.0, else token-F1 for fuzzy credit
    exp_str = str(exp_val).lower()
    out_str = str(out_val).lower()
    if exp_str == out_str:
        return 1.0
    return _str_similarity(out_str, exp_str)


def _str_similarity(a: str, b: str) -> float:
    """Token-level F1 between two strings. Gives credit for partial matches."""
    import re
    stopwords = {"a", "an", "the", "is", "in", "on", "at", "of", "to", "and", "or"}
    def tok(s): return [t for t in re.split(r"\W+", s.lower()) if t and t not in stopwords]
    at, bt = tok(a), tok(b)
    if not at or not bt:
        return 0.0
    from collections import Counter
    ac, bc = Counter(at), Counter(bt)
    common = sum(min(ac[t], bc[t]) for t in ac)
    if not common:
        return 0.0
    p = common / len(at)
    r = common / len(bt)
    return 2 * p * r / (p + r)


def score_field_accuracy(output: dict, expected: dict) -> float:
    """Per-field accuracy with type-aware scoring (int, list, dict, fuzzy string)."""
    expected_fields = {k: v for k, v in expected.items()
                       if not k.startswith("confidence") and v is not None
                       and k not in ("root_cause", "entity_targets", "metric_impact")}
    if not expected_fields:
        return 1.0
    scores = [_score_value(output.get(k), v) for k, v in expected_fields.items()]
    return sum(scores) / len(scores)


def score_confidence_bound(output_conf: float, expected: dict) -> bool:
    """Pass if output confidence is within expected min/max bounds."""
    if "confidence_min" in expected and output_conf < expected["confidence_min"]:
        return False
    if "confidence_max" in expected and output_conf > expected["confidence_max"]:
        return False
    return True


_STOPWORDS = {"a", "an", "the", "is", "in", "on", "at", "of", "to", "was", "has",
              "are", "be", "for", "with", "that", "this", "it", "and", "or", "not"}

def _tokens(text: str) -> list[str]:
    import re
    return [t for t in re.split(r"\W+", text.lower()) if t and t not in _STOPWORDS]

def score_root_cause(output_rc: str, expected_rc: str) -> float:
    """Token-level F1 (SQuAD-style) — handles valid paraphrases, not just exact keyword match."""
    if not output_rc or not expected_rc:
        return 0.0
    exp = _tokens(expected_rc)
    out = _tokens(output_rc)
    if not exp or not out:
        return 0.0
    exp_counts: dict[str, int] = {}
    for t in exp:
        exp_counts[t] = exp_counts.get(t, 0) + 1
    out_counts: dict[str, int] = {}
    for t in out:
        out_counts[t] = out_counts.get(t, 0) + 1
    common = sum(min(exp_counts.get(t, 0), out_counts.get(t, 0)) for t in out_counts)
    if common == 0:
        return 0.0
    precision = common / len(out)
    recall = common / len(exp)
    return 2 * precision * recall / (precision + recall)


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
