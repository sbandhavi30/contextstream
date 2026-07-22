"""
callback.py — ContextStreamCallbackHandler: intercepts tool outputs automatically.

Usage:
    from sdk.langchain.callback import ContextStreamCallbackHandler
    from core.engine import ContextStreamEngine

    engine = ContextStreamEngine(model="claude-haiku-4-5-20251001")
    handler = ContextStreamCallbackHandler(engine=engine)

    # Attach to any agent or chain
    agent.run("diagnose cluster OOM", callbacks=[handler])

    # Or set globally on the LLM
    llm = ChatAnthropic(callbacks=[handler])

How it works:
    on_tool_start(tool_name, input_str) — calls engine.before_tool_call()
    on_tool_end(output)                 — calls engine.after_tool_call(), pages output
    on_tool_error(error)                — pages error output as low-confidence lesson
    on_agent_action(action)             — extracts task description from agent thoughts
    on_agent_finish(finish)             — logs final answer token count

The handler is stateless between tool calls except for the active ForkContext
(stored per run_id to support parallel tool calls).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from core.engine import ContextStreamEngine


class ContextStreamCallbackHandler:
    """
    LangChain BaseCallbackHandler-compatible implementation.

    Implements the subset of callback methods needed for tool interception.
    Does NOT require langchain to be installed in core — duck-typed interface.
    """

    def __init__(
        self,
        engine: ContextStreamEngine,
        task_description: str = "",
    ):
        self.engine           = engine
        self.task_description = task_description
        # run_id → ForkContext — supports concurrent tool calls
        self._active_forks: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Fires before tool executes. Prepares ForkContext."""
        tool_name = serialized.get("name", "unknown")
        task = self.task_description or input_str[:200]

        fork_ctx = self.engine.before_tool_call(
            tool_name=tool_name,
            task_description=task,
        )
        key = str(run_id) if run_id else tool_name
        self._active_forks[key] = fork_ctx

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Fires after tool returns. Pages output, extracts lesson."""
        key = str(run_id) if run_id else self._last_fork_key()
        fork_ctx = self._active_forks.pop(key, None)

        if fork_ctx is None:
            # on_tool_start wasn't called — create minimal fork context
            fork_ctx = self.engine.before_tool_call(
                tool_name="unknown",
                task_description=self.task_description,
            )

        self.engine.after_tool_call(fork_ctx, iter([str(output)]))

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Pages tool errors as low-confidence lessons (don't silently drop)."""
        key = str(run_id) if run_id else self._last_fork_key()
        fork_ctx = self._active_forks.pop(key, None)

        if fork_ctx is None:
            fork_ctx = self.engine.before_tool_call(
                tool_name="unknown",
                task_description=self.task_description,
            )

        error_output = f"Tool error: {type(error).__name__}: {error}"
        self.engine.after_tool_call(fork_ctx, iter([error_output]))

    # ------------------------------------------------------------------
    # Agent callbacks — extract task description from agent thoughts
    # ------------------------------------------------------------------

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Extract task context from ReAct agent's thought."""
        if hasattr(action, "log") and action.log and not self.task_description:
            # First line of ReAct thought = high-level task
            first_line = action.log.strip().splitlines()[0]
            self.task_description = first_line[:200]

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Log final answer — useful for dry-run mode."""
        if self.engine.dry_run:
            output = getattr(finish, "return_values", {}).get("output", "")
            self.engine._log("AGENT_FINISH", "agent", f"output_len={len(output)} chars")

    # ------------------------------------------------------------------
    # Context access
    # ------------------------------------------------------------------

    def get_context(self) -> str:
        """Return compressed ledger for injection into next LLM call."""
        return self.engine.render_context()

    def status(self) -> dict:
        return self.engine.status()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _last_fork_key(self) -> str:
        """Fallback when run_id is None — returns most recently added key."""
        if self._active_forks:
            return next(reversed(self._active_forks))
        return ""
