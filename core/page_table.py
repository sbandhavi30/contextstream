"""
page_table.py — Semantic index of all lessons the agent knows.

Tracks: active/inactive state, confidence, entity targets, tool source,
raw_ref pointer, tombstone relationships.

Used by:
  - fork_manager: inject relevant prior context into sub-agents
  - deduplicator: detect conflicts before appending to ledger
  - budget:       identify low-confidence lessons for eviction candidates
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from pydantic import BaseModel, Field


class PageTableEntry(BaseModel):
    lesson_id:      str
    is_active:      bool = True
    confidence:     float
    raw_ref:        str
    tool_source:    str
    entity_targets: list[str]   = Field(default_factory=list)
    dependency_tags: list[str]  = Field(default_factory=list)
    tombstone_of:   str | None  = None
    sequence_id:    int         = 0
    timestamp:      datetime    = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Denormalized root_cause for deduplication without cold-store fetch
    root_cause_summary: str     = ""


class PageTable:
    """
    In-memory index. Fast lookups by tool, entity, confidence.
    Never stores raw data — only metadata + pointers.
    """

    def __init__(self):
        self._table: dict[str, PageTableEntry] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def register(self, entry: PageTableEntry) -> None:
        """Add new entry. If it tombstones an old one, mark old as inactive."""
        if entry.tombstone_of and entry.tombstone_of in self._table:
            self._table[entry.tombstone_of].is_active = False
        self._table[entry.lesson_id] = entry

    # ------------------------------------------------------------------
    # Query — used by fork_manager for dependency injection
    # ------------------------------------------------------------------

    def get_dependencies_for_tool(
        self,
        tool_name: str,
        max_results: int = 5,
    ) -> list[PageTableEntry]:
        """
        Return active entries most relevant to this tool call.
        Priority: same tool source first, then by confidence descending.
        """
        active = [e for e in self._table.values() if e.is_active]
        same_tool = [e for e in active if e.tool_source == tool_name]
        other     = [e for e in active if e.tool_source != tool_name]
        ranked = (
            sorted(same_tool, key=lambda e: e.confidence, reverse=True) +
            sorted(other,     key=lambda e: e.confidence, reverse=True)
        )
        return ranked[:max_results]

    def get_by_entity(self, entity: str) -> list[PageTableEntry]:
        """Return active entries that mention a specific entity."""
        entity_lower = entity.lower()
        return [
            e for e in self._table.values()
            if e.is_active and any(entity_lower in t.lower() for t in e.entity_targets)
        ]

    def get_active(self) -> list[PageTableEntry]:
        return [e for e in self._table.values() if e.is_active]

    def get_low_confidence(self, threshold: float = 0.6) -> list[PageTableEntry]:
        """Candidates for re-paging — agent may need to revisit raw source."""
        return [e for e in self._table.values() if e.is_active and e.confidence < threshold]

    def get(self, lesson_id: str) -> PageTableEntry | None:
        return self._table.get(lesson_id)

    # ------------------------------------------------------------------
    # Conflict detection — used by deduplicator
    # ------------------------------------------------------------------

    def find_conflicts(
        self,
        entity_targets: list[str],
        tool_source: str,
    ) -> list[PageTableEntry]:
        """
        Find active entries that share entity targets and tool source.
        These are candidates for tombstoning.
        """
        incoming_entities = {e.lower() for e in entity_targets}
        return [
            e for e in self._table.values()
            if e.is_active
            and e.tool_source == tool_source
            and bool(incoming_entities & {t.lower() for t in e.entity_targets})
        ]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def active_count(self) -> int:
        return sum(1 for e in self._table.values() if e.is_active)

    def total_count(self) -> int:
        return len(self._table)

    def summary(self) -> str:
        active = self.active_count()
        total  = self.total_count()
        tombstoned = sum(1 for e in self._table.values() if e.tombstone_of)
        return f"PageTable: {active} active / {total} total / {tombstoned} tombstoned"

    def __len__(self) -> int:
        return len(self._table)

    def __iter__(self) -> Iterator[PageTableEntry]:
        return iter(self._table.values())
