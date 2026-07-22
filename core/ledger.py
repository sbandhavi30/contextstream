"""
ledger.py — Append-only context ledger.

Every lesson, tombstone, and system instruction appended as a new block.
Nothing is ever modified or deleted. Prefix never changes = KV cache coherence.
render_prompt() produces the exact string injected into the LLM context.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Iterator

from pydantic import BaseModel, Field


class PayloadType(str, Enum):
    LESSON              = "LESSON"
    TOMBSTONE           = "TOMBSTONE"
    SYSTEM_INSTRUCTION  = "SYSTEM_INSTRUCTION"
    AGENT_OBSERVATION   = "AGENT_OBSERVATION"


class LedgerEntry(BaseModel):
    sequence_id:  int
    payload_type: PayloadType
    text_content: str
    timestamp:    datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    lesson_id:    str | None = None     # back-reference for tombstone resolution
    tool_source:  str | None = None
    confidence:   float | None = None


class Ledger:
    """
    Append-only sequence of LedgerEntry blocks.
    render_prompt() is the only method that touches the LLM context.
    """

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self._entries: list[LedgerEntry] = []

    # ------------------------------------------------------------------
    # Write path — append only
    # ------------------------------------------------------------------

    def append_lesson(
        self,
        text: str,
        lesson_id: str,
        tool_source: str = "",
        confidence: float | None = None,
    ) -> int:
        return self._append(PayloadType.LESSON, text, lesson_id, tool_source, confidence)

    def append_tombstone(
        self,
        text: str,
        superseded_lesson_id: str,
        tool_source: str = "",
    ) -> int:
        return self._append(PayloadType.TOMBSTONE, text, superseded_lesson_id, tool_source)

    def append_system(self, text: str) -> int:
        return self._append(PayloadType.SYSTEM_INSTRUCTION, text)

    def append_observation(self, text: str) -> int:
        return self._append(PayloadType.AGENT_OBSERVATION, text)

    def _append(
        self,
        payload_type: PayloadType,
        text: str,
        lesson_id: str | None = None,
        tool_source: str | None = None,
        confidence: float | None = None,
    ) -> int:
        seq = len(self._entries)
        entry = LedgerEntry(
            sequence_id=seq,
            payload_type=payload_type,
            text_content=text,
            lesson_id=lesson_id,
            tool_source=tool_source or None,
            confidence=confidence,
        )
        self._entries.append(entry)
        return seq

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def render_prompt(self, max_entries: int | None = None) -> str:
        """
        Render active context for injection into LLM prompt.
        Append-only: same entries always produce same prefix string.
        max_entries: if set, render only last N entries (for tight budgets).
        """
        entries = self._entries[-max_entries:] if max_entries else self._entries
        lines = ["=== CONTEXT LEDGER ==="]
        for e in entries:
            ts = e.timestamp.strftime("%H:%M:%SZ")
            tag = e.payload_type.value
            conf = f" [conf={e.confidence:.2f}]" if e.confidence is not None else ""
            src  = f" [{e.tool_source}]" if e.tool_source else ""
            lines.append(f"[{ts}][{tag}]{src}{conf} {e.text_content}")
        lines.append("======================")
        return "\n".join(lines)

    def token_estimate(self, chars_per_token: float = 4.0) -> int:
        """Fast token estimate: total chars / chars_per_token."""
        total = sum(len(e.text_content) for e in self._entries)
        return int(total / chars_per_token)

    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def tail(self, n: int = 5) -> list[LedgerEntry]:
        return self._entries[-n:]

    def __len__(self) -> int:
        return len(self._entries)

    def iter_by_type(self, payload_type: PayloadType) -> Iterator[LedgerEntry]:
        return (e for e in self._entries if e.payload_type == payload_type)
