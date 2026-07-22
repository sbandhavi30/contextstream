"""
plugin.py — ContextStream LangGraph integration.

Drop-in middleware for LangGraph StateGraph agents.

Usage:
    from sdk.langgraph.plugin import ContextStreamPlugin
    from langgraph.graph import StateGraph

    engine = ContextStreamEngine(model="claude-sonnet-4-6")
    cs = ContextStreamPlugin(engine)

    graph = StateGraph(AgentState)
    graph.add_node("call_tool", cs.wrap_tool_node(call_tool_fn))
    # or wire hooks manually:
    graph.add_node("call_tool", call_tool_fn)
    graph.add_node("call_tool", cs.before_node("call_tool"))
    graph.add_node("call_tool", cs.after_node("call_tool"))

The plugin intercepts tool calls transparently.
Main agent context only ever sees compressed lessons (~40 tokens),
never raw tool output.

LangGraph state key convention:
    state["messages"]        — standard message list (appended to by plugin)
    state["_cs_fork_ctx"]    — transient: ForkContext between before/after
    state["_cs_lesson_refs"] — accumulated LessonReferences this session
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterator

# LangGraph types — imported lazily so the core engine doesn't require langgraph installed
def _langgraph_message():
    from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
    return AIMessage, SystemMessage, ToolMessage


class ContextStreamPlugin:
    """
    Wraps a ContextStreamEngine with LangGraph-compatible hooks.

    Hooks:
        before_node(tool_name) → LangGraph node function
        after_node(tool_name)  → LangGraph node function
        wrap_tool_node(fn)     → wraps existing node function with both hooks

    All hooks read/write standard LangGraph state dict.
    No modifications to existing agent logic required.
    """

    # State keys used internally — prefixed to avoid collision
    _FORK_CTX_KEY    = "_cs_fork_ctx"
    _LESSON_REFS_KEY = "_cs_lesson_refs"
    _RAW_OUTPUT_KEY  = "_cs_raw_output"

    def __init__(self, engine):
        """
        Args:
            engine: ContextStreamEngine instance (from core.engine)
        """
        self.engine = engine

    # ------------------------------------------------------------------
    # Public hook builders
    # ------------------------------------------------------------------

    def before_node(self, tool_name: str, task_key: str = "task") -> Callable:
        """
        Build a LangGraph node function to call before tool execution.

        Args:
            tool_name: name of the tool (e.g. 'kubectl', 'sql', 'bash')
            task_key:  state key holding current task description (default: 'task')

        Returns:
            Node function: state -> state_patch
        """
        def _before(state: dict) -> dict:
            task_desc = state.get(task_key, "")
            fork_ctx = self.engine.before_tool_call(
                tool_name=tool_name,
                task_description=task_desc,
            )
            return {self._FORK_CTX_KEY: fork_ctx}

        _before.__name__ = f"cs_before_{tool_name}"
        return _before

    def after_node(self, tool_name: str, output_key: str = "tool_output") -> Callable:
        """
        Build a LangGraph node function to call after tool execution.

        Reads raw tool output from state[output_key] (str or list[str]).
        Writes compressed lesson back to state["messages"] as a SystemMessage.

        Args:
            tool_name:  name of the tool
            output_key: state key holding raw tool output string/list

        Returns:
            Node function: state -> state_patch
        """
        def _after(state: dict) -> dict:
            fork_ctx = state.get(self._FORK_CTX_KEY)
            if fork_ctx is None:
                # No fork context — before_node wasn't called; skip gracefully
                return {}

            raw = state.get(output_key, "")
            stream: Iterator[str] = _to_stream(raw)

            ref = self.engine.after_tool_call(fork_ctx, stream)

            # Append compressed lesson to message list
            AIMessage, SystemMessage, ToolMessage = _langgraph_message()
            lesson_text = _lesson_ref_to_text(ref)
            lesson_msg = SystemMessage(content=lesson_text)

            existing_msgs = state.get("messages", [])
            existing_refs = state.get(self._LESSON_REFS_KEY, [])

            return {
                "messages": existing_msgs + [lesson_msg],
                self._LESSON_REFS_KEY: existing_refs + [ref],
                self._FORK_CTX_KEY: None,   # clear transient state
                output_key: None,           # evict raw output from state
            }

        _after.__name__ = f"cs_after_{tool_name}"
        return _after

    def wrap_tool_node(
        self,
        tool_fn: Callable,
        tool_name: str,
        input_key: str = "tool_input",
        output_key: str = "tool_output",
        task_key: str = "task",
    ) -> Callable:
        """
        Wrap an existing tool node function with before/after hooks.

        The wrapped function:
            1. Calls before_tool_call
            2. Calls tool_fn(state) to get raw output
            3. Stores raw output in state[output_key]
            4. Calls after_tool_call
            5. Returns state patch with lesson (raw output evicted)

        Args:
            tool_fn:    existing LangGraph node function
            tool_name:  tool name for schema routing
            input_key:  state key with tool input (for task_description)
            output_key: state key to write raw output into temporarily
            task_key:   state key with task description

        Returns:
            Wrapped node function
        """
        before = self.before_node(tool_name, task_key=task_key)
        after  = self.after_node(tool_name, output_key=output_key)

        def _wrapped(state: dict) -> dict:
            # 1. Before hook
            patch = before(state)
            state = {**state, **patch}

            # 2. Execute tool
            raw_result = tool_fn(state)

            # Normalize: tool_fn may return state patch or raw string
            if isinstance(raw_result, dict):
                raw_output = raw_result.get(output_key, "")
            else:
                raw_output = str(raw_result) if raw_result is not None else ""

            state = {**state, output_key: raw_output}

            # 3. After hook — compresses output, appends lesson
            after_patch = after(state)
            return after_patch

        _wrapped.__name__ = f"cs_wrapped_{tool_name}"
        return _wrapped

    # ------------------------------------------------------------------
    # Context injection
    # ------------------------------------------------------------------

    def inject_context(self, state: dict, messages_key: str = "messages") -> dict:
        """
        Prepend current ContextStream ledger as a SystemMessage.

        Call this before invoking the LLM node to ensure the agent
        sees all compressed lessons (not raw outputs).

        Args:
            state:        current LangGraph state
            messages_key: state key holding message list

        Returns:
            State patch with updated messages list
        """
        AIMessage, SystemMessage, ToolMessage = _langgraph_message()
        context_text = self.engine.render_context()
        if not context_text.strip():
            return {}

        context_msg = SystemMessage(content=f"[ContextStream]\n{context_text}")
        existing = state.get(messages_key, [])

        # Replace existing ContextStream system message if present (keep list clean)
        filtered = [m for m in existing if not (
            hasattr(m, "content") and
            isinstance(m.content, str) and
            m.content.startswith("[ContextStream]")
        )]

        return {messages_key: [context_msg] + filtered}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Delegate to engine status — for logging/debugging."""
        return self.engine.status()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_stream(raw: Any) -> Iterator[str]:
    """Normalize various raw output types to Iterator[str]."""
    if isinstance(raw, str):
        yield raw
    elif isinstance(raw, (list, tuple)):
        for chunk in raw:
            yield str(chunk)
    elif raw is None:
        yield ""
    else:
        yield str(raw)


def _lesson_ref_to_text(ref) -> str:
    """Render a LessonReference as a terse context string (~40 tokens)."""
    if ref.is_tombstone:
        tag = f"[TOMBSTONE supersedes {ref.tombstone_of}]"
    else:
        tag = f"[LESSON {ref.lesson_id}]"

    conf_str = f"conf={ref.confidence:.2f}"
    return f"{tag} [{ref.tool_source}] {conf_str}"
