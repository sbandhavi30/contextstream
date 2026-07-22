"""
contextstream_agent.py — SRE agent with ContextStream.

Each tool output is:
  1. Paged to eviction store (never enters main context)
  2. Compressed to a typed lesson by a cheap LLM (Haiku)
  3. Only the lesson (~40 tokens) appended to the context ledger

Final diagnosis call receives the compressed ledger, not raw outputs.

Measures:
  - Total tokens sent to LLM for final diagnosis call
  - Compression ratio vs baseline
  - Lesson quality (do lessons contain the signal needed for correct diagnosis?)
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

from core.engine import ContextStreamEngine
from examples.sre_agent.scenario import (
    AGENT_TASK, DIAGNOSIS_PROMPT_TEMPLATE, TOOL_CS_NAME, TOOL_OUTPUTS
)


def count_tokens(client: anthropic.Anthropic, text: str, model: str) -> int:
    response = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return response.input_tokens


def run_contextstream(
    model: str = "claude-haiku-4-5-20251001",
    extractor_model: str = "claude-haiku-4-5-20251001",
    verbose: bool = True,
) -> dict:
    client = anthropic.Anthropic()

    if verbose:
        print("=" * 60)
        print("CONTEXTSTREAM AGENT — Compressed lesson context")
        print("=" * 60)

    engine = ContextStreamEngine(
        model=model,
        session_id="sre_demo",
        dry_run=False,
    )
    engine.init(AGENT_TASK)

    lessons_text: list[str] = []
    extraction_tokens_total = 0

    # Step 1: Execute each tool — page output, extract lesson
    for tool_name, raw_output in TOOL_OUTPUTS.items():
        cs_name = TOOL_CS_NAME[tool_name]

        if verbose:
            print(f"\n  [TOOL] {tool_name}")
            print(f"    raw output: {len(raw_output):,} chars → eviction store")

        # before_tool_call — prepares fork context
        fork_ctx = engine.before_tool_call(cs_name, task_description=AGENT_TASK)

        # after_tool_call — pages output, extracts lesson, appends to ledger
        ref = engine.after_tool_call(fork_ctx, iter([raw_output]))

        if verbose:
            print(f"    lesson: conf={ref.confidence:.2f}  tombstone={ref.is_tombstone}")

        # Track extraction cost (counted against extractor, not main agent)
        lesson_line = f"[{cs_name.upper()}] conf={ref.confidence:.2f} id={ref.lesson_id}"
        lessons_text.append(lesson_line)

    # Step 2: Render compressed ledger
    compressed_context = engine.render_context()

    if verbose:
        print(f"\n  Compressed ledger:\n{compressed_context}")

    # Step 3: Build diagnosis prompt with compressed context only
    context_block = f"Compressed investigation ledger (lessons extracted from tool outputs):\n\n{compressed_context}"
    diagnosis_prompt = DIAGNOSIS_PROMPT_TEMPLATE.format(context=context_block)

    # Step 4: Count tokens
    token_count = count_tokens(client, diagnosis_prompt, model)

    if verbose:
        print(f"\n  Compressed context size: {len(compressed_context):,} chars")
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

    status = engine.status()

    return {
        "mode": "contextstream",
        "model": model,
        "extractor_model": extractor_model,
        "compressed_context_chars": len(compressed_context),
        "diagnosis_prompt_tokens": token_count,
        "response_tokens": response.usage.output_tokens,
        "total_tokens": token_count + response.usage.output_tokens,
        "elapsed_s": round(elapsed, 2),
        "diagnosis": diagnosis,
        "eviction_bytes": status["eviction_bytes"],
        "active_lessons": status["active_lessons"],
    }


if __name__ == "__main__":
    result = run_contextstream(verbose=True)
    print(f"\nTokens used for diagnosis call: {result['diagnosis_prompt_tokens']:,}")
