"""
baseline_agent.py — Naive context-stuffing agent (no ContextStream).

All 4 raw tool outputs concatenated directly into the LLM prompt.
Represents the status quo in most agent frameworks today.

Measures:
  - Total tokens sent to LLM for final diagnosis call
  - Final diagnosis quality (same prompt, just more tokens)
"""

from __future__ import annotations

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

from examples.sre_agent.scenario import (
    AGENT_TASK, DIAGNOSIS_PROMPT_TEMPLATE, TOOL_OUTPUTS
)


def count_tokens(client: anthropic.Anthropic, text: str, model: str) -> int:
    """Use Anthropic token counting API."""
    response = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return response.input_tokens


def run_baseline(model: str = "claude-haiku-4-5-20251001", verbose: bool = True) -> dict:
    client = anthropic.Anthropic()

    if verbose:
        print("=" * 60)
        print("BASELINE AGENT — Naive context stuffing")
        print("=" * 60)

    # Step 1: "Execute" all tools — concatenate raw outputs into context
    raw_context_parts = []
    for tool_name, output in TOOL_OUTPUTS.items():
        if verbose:
            print(f"  [TOOL] {tool_name} → {len(output)} chars raw output added to context")
        raw_context_parts.append(f"=== {tool_name} output ===\n{output}")

    raw_context = "\n\n".join(raw_context_parts)

    # Step 2: Build diagnosis prompt with full raw context
    context_block = f"Tool outputs from investigation:\n\n{raw_context}"
    diagnosis_prompt = DIAGNOSIS_PROMPT_TEMPLATE.format(context=context_block)

    # Step 3: Count tokens BEFORE calling LLM
    token_count = count_tokens(client, diagnosis_prompt, model)

    if verbose:
        print(f"\n  Raw context size: {len(raw_context):,} chars")
        print(f"  Diagnosis prompt tokens: {token_count:,}")
        print(f"\n  Calling {model} for diagnosis...")

    t0 = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": diagnosis_prompt}],
    )
    elapsed = time.time() - t0
    diagnosis = response.content[0].text

    if verbose:
        print(f"  LLM response time: {elapsed:.2f}s")
        print(f"\n--- DIAGNOSIS ---")
        print(diagnosis)
        print("-" * 40)

    return {
        "mode": "baseline",
        "model": model,
        "raw_context_chars": len(raw_context),
        "diagnosis_prompt_tokens": token_count,
        "response_tokens": response.usage.output_tokens,
        "total_tokens": token_count + response.usage.output_tokens,
        "elapsed_s": round(elapsed, 2),
        "diagnosis": diagnosis,
    }


if __name__ == "__main__":
    result = run_baseline(verbose=True)
    print(f"\nTokens used for diagnosis call: {result['diagnosis_prompt_tokens']:,}")
