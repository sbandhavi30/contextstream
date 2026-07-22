"""
memory.py — VMMMemory: drop-in replacement for ConversationBufferMemory.

Usage:
    from sdk.langchain.memory import VMMMemory
    from core.engine import ContextStreamEngine

    engine = ContextStreamEngine(model="claude-haiku-4-5-20251001")
    memory = VMMMemory(engine=engine)

    # Drop into any LangChain chain exactly like ConversationBufferMemory
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

How it works:
    save_context(inputs, outputs) — when a tool returns output, page it to
        eviction store and extract a lesson. Only the lesson enters memory.
    load_memory_variables()       — returns compressed ledger as the
        "history" key, never raw tool outputs.

The memory_key defaults to "history" to match ConversationBufferMemory's
interface — existing prompts that reference {history} work unchanged.

Tool detection heuristic:
    If inputs contains a "tool" key, use it as tool_name.
    If inputs contains "action" (ReAct style), extract tool from there.
    Otherwise falls back to "unknown" and format_detector auto-routes.
"""

from __future__ import annotations

from typing import Any

from core.engine import ContextStreamEngine


class VMMMemory:
    """
    Drop-in for ConversationBufferMemory.

    Implements the LangChain BaseMemory interface:
        memory_variables  — list of keys this memory provides
        load_memory_variables(inputs) -> dict
        save_context(inputs, outputs) -> None
        clear() -> None
    """

    memory_key: str = "history"
    input_key:  str = "input"
    output_key: str = "output"

    def __init__(
        self,
        engine: ContextStreamEngine,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        task_description: str = "",
    ):
        self.engine           = engine
        self.memory_key       = memory_key
        self.input_key        = input_key
        self.output_key       = output_key
        self.task_description = task_description
        self._call_count      = 0

    # ------------------------------------------------------------------
    # BaseMemory interface
    # ------------------------------------------------------------------

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return compressed ledger as {history: <ledger text>}."""
        return {self.memory_key: self.engine.render_context()}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """
        Called after each chain step. Pages tool output to eviction store,
        extracts lesson, appends to ledger.

        inputs  — chain inputs for this step (may contain 'tool', 'action')
        outputs — chain outputs (tool result or LLM response)
        """
        tool_name = self._detect_tool(inputs, outputs)
        raw_output = self._extract_raw(outputs)

        if not raw_output.strip():
            return

        task = self.task_description or inputs.get(self.input_key, "")
        fork_ctx = self.engine.before_tool_call(tool_name, task_description=task)
        self.engine.after_tool_call(fork_ctx, iter([raw_output]))
        self._call_count += 1

    def clear(self) -> None:
        """Reset — reinitialise engine state."""
        self._call_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_tool(self, inputs: dict, outputs: dict) -> str:
        """
        Best-effort tool name extraction from LangChain step dicts.
        ReAct agents set inputs["action"] = AgentAction(tool=...).
        Tool nodes set inputs["tool"] = "kubectl".
        Falls back to "unknown" — format_detector handles routing.
        """
        # Direct key
        if "tool" in inputs:
            return str(inputs["tool"])

        # ReAct AgentAction object
        action = inputs.get("action")
        if action is not None and hasattr(action, "tool"):
            return str(action.tool)

        # Output key hint (some chains set "tool_name" in output)
        if "tool_name" in outputs:
            return str(outputs["tool_name"])

        return "unknown"

    def _extract_raw(self, outputs: dict) -> str:
        """Pull raw string from outputs dict."""
        # Standard LangChain output keys in priority order
        for key in ("output", "result", "text", "tool_output", "observation"):
            val = outputs.get(key)
            if val is not None:
                return str(val)
        # Fallback: join all string values
        parts = [str(v) for v in outputs.values() if isinstance(v, str)]
        return "\n".join(parts)
