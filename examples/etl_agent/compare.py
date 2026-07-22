"""
compare.py — Baseline vs ContextStream for ETL pipeline incident.

Usage:
    python examples/etl_agent/compare.py
    python examples/etl_agent/compare.py --model claude-sonnet-4-6
    python examples/etl_agent/compare.py --quiet
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import anthropic

from core.engine import ContextStreamEngine
from examples.etl_agent.scenario import (
    AGENT_TASK, DIAGNOSIS_PROMPT_TEMPLATE, TOOL_CS_NAME, TOOL_OUTPUTS
)


def count_tokens(client: anthropic.Anthropic, text: str, model: str) -> int:
    return client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    ).input_tokens


def run_baseline(client: anthropic.Anthropic, model: str, verbose: bool) -> dict:
    if verbose:
        print("=" * 60)
        print("BASELINE — Naive context stuffing")
        print("=" * 60)

    parts = []
    for name, output in TOOL_OUTPUTS.items():
        if verbose:
            print(f"  [TOOL] {name} → {len(output):,} chars")
        parts.append(f"=== {name} ===\n{output}")

    raw_context = "\n\n".join(parts)
    prompt = DIAGNOSIS_PROMPT_TEMPLATE.format(
        context=f"Tool outputs:\n\n{raw_context}"
    )
    tokens = count_tokens(client, prompt, model)

    if verbose:
        print(f"\n  Prompt tokens: {tokens:,}")

    t0 = time.time()
    resp = client.messages.create(
        model=model, max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - t0
    diagnosis = resp.content[0].text

    if verbose:
        print(f"  Response time: {elapsed:.2f}s\n")
        print("--- DIAGNOSIS ---")
        print(diagnosis)
        print("-" * 40)

    return {
        "mode": "baseline",
        "raw_chars": len(raw_context),
        "prompt_tokens": tokens,
        "response_tokens": resp.usage.output_tokens,
        "elapsed_s": round(elapsed, 2),
        "diagnosis": diagnosis,
    }


def run_contextstream(client: anthropic.Anthropic, model: str, verbose: bool) -> dict:
    if verbose:
        print("=" * 60)
        print("CONTEXTSTREAM — Compressed lesson context")
        print("=" * 60)

    engine = ContextStreamEngine(model=model, session_id="etl_demo", dry_run=False)
    engine.init(AGENT_TASK)

    for name, raw_output in TOOL_OUTPUTS.items():
        cs_name = TOOL_CS_NAME[name]
        fork_ctx = engine.before_tool_call(cs_name, task_description=AGENT_TASK)
        ref = engine.after_tool_call(fork_ctx, iter([raw_output]))
        if verbose:
            print(f"  [TOOL] {name} → {len(raw_output):,} chars → lesson conf={ref.confidence:.2f}")

    compressed = engine.render_context()
    if verbose:
        print(f"\n  Compressed ledger:\n{compressed}")

    prompt = DIAGNOSIS_PROMPT_TEMPLATE.format(
        context=f"Compressed investigation ledger:\n\n{compressed}"
    )
    tokens = count_tokens(client, prompt, model)

    if verbose:
        print(f"\n  Prompt tokens: {tokens:,}")

    t0 = time.time()
    resp = client.messages.create(
        model=model, max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - t0
    diagnosis = resp.content[0].text

    if verbose:
        print(f"  Response time: {elapsed:.2f}s\n")
        print("--- DIAGNOSIS ---")
        print(diagnosis)
        print("-" * 40)

    status = engine.status()
    return {
        "mode": "contextstream",
        "compressed_chars": len(compressed),
        "prompt_tokens": tokens,
        "response_tokens": resp.usage.output_tokens,
        "elapsed_s": round(elapsed, 2),
        "eviction_bytes": status["eviction_bytes"],
        "active_lessons": status["active_lessons"],
        "diagnosis": diagnosis,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    verbose = not args.quiet

    print("\n" + "█" * 60)
    print("  ETL PIPELINE AGENT BENCHMARK: Baseline vs ContextStream")
    print("█" * 60 + "\n")

    b = run_baseline(client, args.model, verbose)
    print()
    cs = run_contextstream(client, args.model, verbose)

    token_saved = b["prompt_tokens"] - cs["prompt_tokens"]
    pct_saved   = token_saved / b["prompt_tokens"] * 100
    char_saved  = b["raw_chars"] - cs["compressed_chars"]
    char_pct    = char_saved / b["raw_chars"] * 100

    print("\n" + "=" * 60)
    print("  COMPARISON RESULTS")
    print("=" * 60)
    print(f"  Model: {args.model}\n")
    print(f"  {'Metric':<35} {'Baseline':>12} {'ContextStream':>14}")
    print(f"  {'-'*35} {'-'*12} {'-'*14}")
    print(f"  {'Context chars':<35} {b['raw_chars']:>12,} {cs['compressed_chars']:>14,}")
    print(f"  {'Prompt tokens':<35} {b['prompt_tokens']:>12,} {cs['prompt_tokens']:>14,}")
    print(f"  {'Response tokens':<35} {b['response_tokens']:>12,} {cs['response_tokens']:>14,}")
    print(f"  {'LLM response time (s)':<35} {b['elapsed_s']:>12.2f} {cs['elapsed_s']:>14.2f}")
    print()
    print(f"  Token reduction:  {token_saved:,} tokens  ({pct_saved:.1f}% fewer)")
    print(f"  Context reduction: {char_saved:,} chars   ({char_pct:.1f}% smaller)")
    print(f"  Eviction store:   {cs['eviction_bytes']:,} bytes  (never in main context)")
    print(f"  Active lessons:   {cs['active_lessons']}")

    print("\n" + "=" * 60)
    print("  BASELINE DIAGNOSIS")
    print("=" * 60)
    print(b["diagnosis"])

    print("\n" + "=" * 60)
    print("  CONTEXTSTREAM DIAGNOSIS")
    print("=" * 60)
    print(cs["diagnosis"])
    print()


if __name__ == "__main__":
    main()
