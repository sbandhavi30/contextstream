"""
compare.py — Run baseline vs ContextStream side-by-side and print comparison.

Usage:
    python examples/sre_agent/compare.py
    python examples/sre_agent/compare.py --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from examples.sre_agent.baseline_agent import run_baseline
from examples.sre_agent.contextstream_agent import run_contextstream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--quiet", action="store_true", help="suppress per-step output")
    args = parser.parse_args()

    verbose = not args.quiet

    print("\n" + "█" * 60)
    print("  SRE AGENT BENCHMARK: Baseline vs ContextStream")
    print("█" * 60 + "\n")

    baseline = run_baseline(model=args.model, verbose=verbose)
    print()
    cs       = run_contextstream(model=args.model, verbose=verbose)

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    token_reduction = baseline["diagnosis_prompt_tokens"] - cs["diagnosis_prompt_tokens"]
    reduction_pct   = token_reduction / baseline["diagnosis_prompt_tokens"] * 100
    char_reduction  = baseline["raw_context_chars"] - cs["compressed_context_chars"]
    char_pct        = char_reduction / baseline["raw_context_chars"] * 100

    print("\n" + "=" * 60)
    print("  COMPARISON RESULTS")
    print("=" * 60)
    print(f"  Model: {args.model}\n")

    print(f"  {'Metric':<35} {'Baseline':>12} {'ContextStream':>14}")
    print(f"  {'-'*35} {'-'*12} {'-'*14}")
    print(f"  {'Context chars':<35} {baseline['raw_context_chars']:>12,} {cs['compressed_context_chars']:>14,}")
    print(f"  {'Diagnosis prompt tokens':<35} {baseline['diagnosis_prompt_tokens']:>12,} {cs['diagnosis_prompt_tokens']:>14,}")
    print(f"  {'Response tokens':<35} {baseline['response_tokens']:>12,} {cs['response_tokens']:>14,}")
    print(f"  {'LLM response time (s)':<35} {baseline['elapsed_s']:>12.2f} {cs['elapsed_s']:>14.2f}")
    print()
    print(f"  Token reduction (diagnosis call): {token_reduction:,} tokens  ({reduction_pct:.1f}% fewer)")
    print(f"  Context size reduction:           {char_reduction:,} chars   ({char_pct:.1f}% smaller)")
    print(f"  Raw data in eviction store:       {cs['eviction_bytes']:,} bytes  (never touched main context)")
    print()
    print(f"  Active lessons in page table:     {cs['active_lessons']}")

    print("\n" + "=" * 60)
    print("  BASELINE DIAGNOSIS")
    print("=" * 60)
    print(baseline["diagnosis"])

    print("\n" + "=" * 60)
    print("  CONTEXTSTREAM DIAGNOSIS")
    print("=" * 60)
    print(cs["diagnosis"])
    print()


if __name__ == "__main__":
    main()
