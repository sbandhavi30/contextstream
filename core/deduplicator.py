"""
deduplicator.py — Conflict resolver for incoming lessons.

Before a lesson hits the ledger, checks if it supersedes an existing
active lesson. If conflict detected, generates a tombstone text instead
of a plain lesson text. Never mutates past entries — append-only invariant
is preserved throughout.

Conflict heuristics (in order of confidence):
  1. Same entity targets + same tool + state changed (e.g. 404→200)
  2. Same entity targets + same tool + higher confidence than existing
  3. Same entity targets + same tool + newer timestamp supersedes old
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.extractor import Lesson
from core.page_table import PageTable, PageTableEntry


# ---------------------------------------------------------------------------
# State-change vocabulary for heuristic #1
# ---------------------------------------------------------------------------

# Pairs: (old_signal, new_signal) — if old entry contains left AND
# incoming lesson contains right → definite supersede
STATE_TRANSITIONS = [
    (r"\b404\b",          r"\b200\b"),
    (r"\b500\b",          r"\b200\b"),
    (r"\bfailed\b",       r"\bsuccess(ful)?\b"),
    (r"\berror\b",        r"\bsuccess(ful)?\b"),
    (r"\bdown\b",         r"\bup\b"),
    (r"\bcrashloop\b",    r"\brunning\b"),
    (r"\boomkilled\b",    r"\brunning\b"),
    (r"\bpending\b",      r"\brunning\b"),
    (r"\bsuspended\b",    r"\bactive\b"),
    (r"\bcancelled\b",    r"\bactive\b"),
    (r"\binactive\b",     r"\bactive\b"),
    (r"\blocked\b",       r"\bopen\b"),
]


@dataclass
class DeduplicationResult:
    text:           str         # text to append to ledger
    is_tombstone:   bool        # True = conflict resolved, False = plain lesson
    conflict_id:    str | None  # lesson_id being superseded, if any
    reason:         str         # human-readable reason for the decision


class Deduplicator:
    """
    Evaluates incoming Lesson against PageTable.
    Returns DeduplicationResult — caller appends result.text to Ledger.
    """

    def __init__(self, page_table: PageTable, confidence_gap: float = 0.15):
        self.page_table = page_table
        # Minimum confidence gap to trigger supersede on confidence-only basis
        self.confidence_gap = confidence_gap

    def process(self, lesson: Lesson) -> DeduplicationResult:
        candidates = self.page_table.find_conflicts(
            entity_targets=lesson.entity_targets,
            tool_source=lesson.tool_source,
        )

        if not candidates:
            return self._plain(lesson)

        # Check each candidate for conflict type
        for candidate in sorted(candidates, key=lambda e: e.confidence, reverse=True):
            result = self._check_conflict(lesson, candidate)
            if result:
                return result

        # No conflict resolved — plain append
        return self._plain(lesson)

    # ------------------------------------------------------------------
    # Conflict checks
    # ------------------------------------------------------------------

    def _check_conflict(
        self, lesson: Lesson, existing: PageTableEntry
    ) -> DeduplicationResult | None:

        # 1. Explicit state transition (highest confidence conflict)
        for old_pat, new_pat in STATE_TRANSITIONS:
            old_match = re.search(old_pat, existing.root_cause_summary, re.IGNORECASE)
            new_match = re.search(new_pat, lesson.root_cause, re.IGNORECASE)
            if old_match and new_match:
                return self._tombstone(
                    lesson, existing,
                    f"state transition: {old_match.group()} → {new_match.group()}"
                )

        # 2. Significantly higher confidence on same entity+tool
        if lesson.confidence - existing.confidence >= self.confidence_gap:
            return self._tombstone(
                lesson, existing,
                f"higher confidence: {lesson.confidence:.2f} > {existing.confidence:.2f}"
            )

        # 3. Newer timestamp on same entity+tool (recency supersedes)
        if (lesson.timestamp and existing.timestamp
                and lesson.timestamp > existing.timestamp
                and lesson.confidence >= existing.confidence):
            return self._tombstone(
                lesson, existing,
                "newer observation on same entity"
            )

        return None

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _tombstone(
        self, lesson: Lesson, existing: PageTableEntry, reason: str
    ) -> DeduplicationResult:
        text = (
            f"[TOMBSTONE — supersedes {existing.lesson_id}] "
            f"({reason}) "
            f"{lesson.root_cause}"
        )
        if lesson.metric_impact:
            text += f" | {lesson.metric_impact}"
        return DeduplicationResult(
            text=text,
            is_tombstone=True,
            conflict_id=existing.lesson_id,
            reason=reason,
        )

    @staticmethod
    def _plain(lesson: Lesson) -> DeduplicationResult:
        text = lesson.root_cause
        if lesson.metric_impact:
            text += f" | {lesson.metric_impact}"
        return DeduplicationResult(
            text=text,
            is_tombstone=False,
            conflict_id=None,
            reason="no conflict",
        )
