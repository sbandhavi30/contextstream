"""
engine.py — ContextStreamEngine: coordinates all core components.

This is the single entry point for frameworks and SDKs.
Implements the VMMPlugin hook interface:
  - before_tool_call()
  - after_tool_call()
  - on_token_pressure()
  - on_agent_uncertainty()

Dry-run mode: logs all decisions without modifying context.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable, Iterator

from core.budget import BudgetTracker, PressureLevel
from core.deduplicator import Deduplicator
from core.eviction_store import EvictionStore, MemoryBackend
from core.extractor import Extractor, Lesson
from core.fork_manager import ForkContext, ForkManager, ForkResult
from core.ledger import Ledger, PayloadType
from core.page_table import PageTable, PageTableEntry


@dataclass
class LessonReference:
    """Lightweight pointer returned to framework after tool call."""
    lesson_id:   str
    confidence:  float
    tool_source: str
    is_tombstone: bool
    tombstone_of: str | None = None


@dataclass
class DryRunEvent:
    event:       str
    tool_name:   str
    detail:      str
    token_delta: int = 0


class ContextStreamEngine:
    """
    The VMM for LLM agents.

    Usage (direct):
        engine = ContextStreamEngine(model="claude-sonnet-4-6")
        engine.init("Diagnose cluster OOM issue")

        fork_ctx = engine.before_tool_call("kubectl", "describe OOMKilled pods")
        raw_stream = run_kubectl(...)
        ref = engine.after_tool_call(fork_ctx, iter([raw_stream]))

        prompt = engine.render_context()  # inject into LLM call

    Usage (dry-run):
        engine = ContextStreamEngine(dry_run=True)
        # All decisions logged, context never modified
        for event in engine.dry_run_log:
            print(event)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        session_id: str = "",
        dry_run: bool = False,
        eviction_backend=None,
        max_injected_lessons: int = 5,
        confidence_threshold: float = 0.6,
    ):
        self.session_id  = session_id or uuid.uuid4().hex[:8]
        self.model       = model
        self.dry_run     = dry_run

        # Core components
        backend = eviction_backend or (MemoryBackend() if dry_run else None)
        self.store        = EvictionStore(backend)
        self.ledger       = Ledger(session_id=self.session_id)
        self.page_table   = PageTable()
        self.budget       = BudgetTracker(model=model)
        self.extractor    = Extractor(
            eviction_store=self.store,
            confidence_threshold=confidence_threshold,
        )
        self.fork_manager = ForkManager(
            eviction_store=self.store,
            page_table=self.page_table,
            max_injected_lessons=max_injected_lessons,
        )
        self.deduplicator = Deduplicator(page_table=self.page_table)

        self.dry_run_log: list[DryRunEvent] = []
        self._pressure_callbacks: list[Callable] = []

        # Wire budget pressure callbacks
        self.budget.on_pressure(PressureLevel.CRITICAL, self._on_critical_pressure)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(self, task_description: str) -> None:
        """Set the initial task context. Call once before agent loop starts."""
        text = f"Agent initialized. Task: {task_description}"
        if not self.dry_run:
            seq = self.ledger.append_system(text)
            self.budget.update(text)
        else:
            self._log("INIT", "system", text)

    # ------------------------------------------------------------------
    # VMMPlugin hook interface
    # ------------------------------------------------------------------

    def before_tool_call(
        self,
        tool_name: str,
        task_description: str = "",
    ) -> ForkContext:
        """
        Called before tool executes.
        Prepares sub-agent fork with dependency-injected context.
        Returns ForkContext — pass to after_tool_call().
        """
        status = self.budget.preflight()
        if status.pressure == PressureLevel.CRITICAL:
            self._log("PREFLIGHT_WARN", tool_name,
                      f"Critical budget: {status.used_tokens}/{status.limit_tokens} tokens")

        ctx = self.fork_manager.prepare_fork(
            tool_name=tool_name,
            task_description=task_description,
            session_id=self.session_id,
        )

        if self.dry_run:
            self._log("FORK_PREPARE", tool_name,
                      f"fork_id={ctx.fork_id} deps={len(ctx.injected_lessons)}")

        return ctx

    def after_tool_call(
        self,
        fork_ctx: ForkContext,
        raw_output_stream: Iterator[str],
    ) -> LessonReference:
        """
        Called after tool returns output stream.
        Full cycle: absorb → extract → deduplicate → append to ledger.
        Returns lightweight LessonReference — framework never holds raw data.
        """
        # 1. Absorb raw stream into cold storage
        fork_result = self.fork_manager.absorb_output(fork_ctx, raw_output_stream)

        if self.dry_run:
            self._log("STREAM_ABSORBED", fork_ctx.tool_name,
                      f"raw_size={fork_result.raw_size} chars → ref={fork_result.raw_ref}",
                      token_delta=-fork_result.raw_size // 4)

        # 2. Extract structured lesson
        lesson_id = f"les_{uuid.uuid4().hex[:8]}"
        lesson    = self._extract(fork_ctx.tool_name, fork_result.raw_ref, lesson_id)

        # 3. Deduplicate / conflict resolve
        dedup_result = self.deduplicator.process(lesson)

        # 4. Append to ledger (or dry-run log)
        lesson_tokens = len(dedup_result.text) // 4
        if not self.dry_run:
            if dedup_result.is_tombstone:
                self.ledger.append_tombstone(
                    text=dedup_result.text,
                    superseded_lesson_id=dedup_result.conflict_id or "",
                    tool_source=fork_ctx.tool_name,
                )
            else:
                self.ledger.append_lesson(
                    text=dedup_result.text,
                    lesson_id=lesson_id,
                    tool_source=fork_ctx.tool_name,
                    confidence=lesson.confidence,
                )
            self.budget.update(dedup_result.text)
        else:
            payload = "TOMBSTONE" if dedup_result.is_tombstone else "LESSON"
            self._log(payload, fork_ctx.tool_name,
                      f"conf={lesson.confidence:.2f} reason={dedup_result.reason} "
                      f"text_tokens~{lesson_tokens}",
                      token_delta=lesson_tokens)

        # 5. Update PageTable
        entry = PageTableEntry(
            lesson_id=lesson_id,
            confidence=lesson.confidence,
            raw_ref=fork_result.raw_ref,
            tool_source=fork_ctx.tool_name,
            entity_targets=lesson.entity_targets,
            dependency_tags=[fork_ctx.tool_name],
            tombstone_of=dedup_result.conflict_id,
            sequence_id=len(self.ledger),
            root_cause_summary=lesson.root_cause,
            timestamp=lesson.timestamp,
        )
        if not self.dry_run:
            self.page_table.register(entry)

        return LessonReference(
            lesson_id=lesson_id,
            confidence=lesson.confidence,
            tool_source=fork_ctx.tool_name,
            is_tombstone=dedup_result.is_tombstone,
            tombstone_of=dedup_result.conflict_id,
        )

    def on_token_pressure(self, level: PressureLevel, ledger: Ledger) -> None:
        """Called by budget tracker at pressure thresholds. Hook for SDK integrations."""
        if self.dry_run:
            self._log("PRESSURE", "budget",
                      f"level={level.value} used={self.budget._used}")

    def on_agent_uncertainty(self, query: str) -> list[LessonReference]:
        """
        Called when agent signals low confidence on a topic.
        Returns lessons that may need re-paging from cold storage.
        """
        low_conf = self.page_table.get_low_confidence(threshold=0.65)
        refs = [
            LessonReference(
                lesson_id=e.lesson_id,
                confidence=e.confidence,
                tool_source=e.tool_source,
                is_tombstone=False,
            )
            for e in low_conf
        ]
        if self.dry_run:
            self._log("UNCERTAINTY_RECALL", "agent",
                      f"query='{query}' candidates={len(refs)}")
        return refs

    # ------------------------------------------------------------------
    # Context rendering
    # ------------------------------------------------------------------

    def render_context(self) -> str:
        """Render current ledger for injection into LLM prompt."""
        return self.ledger.render_prompt()

    def status(self) -> dict:
        budget = self.budget._status()
        return {
            "session_id":    self.session_id,
            "model":         self.model,
            "dry_run":       self.dry_run,
            "ledger_entries": len(self.ledger),
            "active_lessons": self.page_table.active_count(),
            "tokens_used":   budget.used_tokens,
            "tokens_limit":  budget.limit_tokens,
            "pressure":      budget.pressure.value,
            "eviction_bytes": self.store.total_size_bytes(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract(self, tool_name: str, raw_ref: str, lesson_id: str) -> Lesson:
        try:
            return self.extractor.extract(tool_name, raw_ref, lesson_id)
        except NotImplementedError:
            # Extractor not wired to LLM yet — return stub lesson for testing
            raw = self.store.fetch(raw_ref)
            return Lesson(
                lesson_id=lesson_id,
                tool_source=tool_name,
                root_cause=f"[stub] raw output from {tool_name}: {raw[:120]}",
                entity_targets=[],
                confidence=0.5,
                raw_ref=raw_ref,
            )

    def _on_critical_pressure(self, status) -> None:
        if self.dry_run:
            self._log("CRITICAL_PRESSURE", "budget",
                      f"used={status.used_tokens} limit={status.limit_tokens}")

    def _log(self, event: str, tool_name: str, detail: str, token_delta: int = 0) -> None:
        self.dry_run_log.append(DryRunEvent(
            event=event, tool_name=tool_name,
            detail=detail, token_delta=token_delta,
        ))
        print(f"[DRY-RUN][{event}][{tool_name}] {detail}")
