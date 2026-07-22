"""
fork_manager.py — Sub-agent lifecycle and dynamic dependency injection.

When a tool executes:
  1. Queries PageTable for relevant prior context (dependency injection)
  2. Builds isolated sub-agent context (tool schema + task + injected deps)
  3. Streams raw output to EvictionStore — never touches main context
  4. Returns raw_ref + injected context for Extractor

Sub-agents are logical, not OS processes — they represent an isolated
LLM call with a clean minimal context window.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

from core.eviction_store import EvictionStore
from core.page_table import PageTable, PageTableEntry


@dataclass
class ForkContext:
    """Context injected into a sub-agent for tool execution."""
    fork_id:          str
    tool_name:        str
    task_description: str
    injected_lessons: list[str]       # rendered lesson summaries from PageTable
    session_id:       str = ""
    created_at:       datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def render(self) -> str:
        """Minimal context string for the sub-agent."""
        lines = [
            f"Tool: {self.tool_name}",
            f"Task: {self.task_description}",
        ]
        if self.injected_lessons:
            lines.append("Prior context:")
            for lesson in self.injected_lessons:
                lines.append(f"  - {lesson}")
        return "\n".join(lines)


@dataclass
class ForkResult:
    fork_id:    str
    tool_name:  str
    raw_ref:    str             # pointer to raw output in EvictionStore
    fork_ctx:   ForkContext     # context the sub-agent saw
    raw_size:   int             # chars in raw output


class ForkManager:
    """
    Manages sub-agent fork lifecycle.
    Coordinates EvictionStore (raw storage) and PageTable (dependency lookup).
    """

    def __init__(
        self,
        eviction_store: EvictionStore,
        page_table: PageTable,
        max_injected_lessons: int = 5,
    ):
        self.store                 = eviction_store
        self.page_table            = page_table
        self.max_injected_lessons  = max_injected_lessons
        self._active_forks: dict[str, ForkContext] = {}

    def prepare_fork(
        self,
        tool_name: str,
        task_description: str,
        session_id: str = "",
    ) -> ForkContext:
        """
        Build sub-agent context before tool executes.
        Called in before_tool_call hook.
        """
        fork_id = f"fork_{uuid.uuid4().hex[:8]}"

        # Dynamic dependency injection — query PageTable for relevant lessons
        deps = self.page_table.get_dependencies_for_tool(
            tool_name, max_results=self.max_injected_lessons
        )
        injected = [
            f"[{e.tool_source}] {e.root_cause_summary} (conf={e.confidence:.2f})"
            for e in deps
        ]

        ctx = ForkContext(
            fork_id=fork_id,
            tool_name=tool_name,
            task_description=task_description,
            injected_lessons=injected,
            session_id=session_id,
        )
        self._active_forks[fork_id] = ctx
        return ctx

    def absorb_output(
        self,
        fork_ctx: ForkContext,
        raw_stream: Iterator[str],
    ) -> ForkResult:
        """
        Swallow raw tool output stream into EvictionStore.
        Called in after_tool_call hook — raw data never enters main context.
        """
        raw_ref = self.store.save_stream(
            raw_stream,
            tool_name=fork_ctx.tool_name,
            session_id=fork_ctx.session_id,
        )
        raw_size = len(self.store.fetch(raw_ref))
        self._active_forks.pop(fork_ctx.fork_id, None)

        return ForkResult(
            fork_id=fork_ctx.fork_id,
            tool_name=fork_ctx.tool_name,
            raw_ref=raw_ref,
            fork_ctx=fork_ctx,
            raw_size=raw_size,
        )

    def active_fork_count(self) -> int:
        return len(self._active_forks)

    def get_fork(self, fork_id: str) -> ForkContext | None:
        return self._active_forks.get(fork_id)
