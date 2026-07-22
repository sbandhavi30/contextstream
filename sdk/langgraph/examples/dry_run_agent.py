"""
dry_run_agent.py — Smoke test for ContextStream LangGraph integration.

Simulates a 3-tool LangGraph agent (kubectl → sql → bash) without
LangGraph installed. Uses plugin hooks directly to verify:
  - before_node fires and attaches ForkContext
  - after_node compresses output to lesson, evicts raw from state
  - messages list gets lesson SystemMessages (not raw output)
  - engine status reflects correct token/lesson counts

Run:
    cd contextstream
    python sdk/langgraph/examples/dry_run_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core.engine import ContextStreamEngine
from sdk.langgraph.plugin import ContextStreamPlugin, _lesson_ref_to_text


# ---------------------------------------------------------------------------
# Fake LangGraph message types (no langgraph required for smoke test)
# ---------------------------------------------------------------------------

class SystemMessage:
    def __init__(self, content: str):
        self.content = content
    def __repr__(self):
        return f"SystemMessage({self.content[:80]!r})"


# Monkeypatch lazy import in plugin so it uses our fakes
import sdk.langgraph.plugin as _plugin_mod
_plugin_mod._langgraph_message = lambda: (None, SystemMessage, None)


# ---------------------------------------------------------------------------
# Simulated tool functions
# ---------------------------------------------------------------------------

KUBECTL_OUTPUT = """\
Name:               web-backend-6f9d7c4-xkv2p
Namespace:          production
Status:             OOMKilled
Restart Count:      7
Limits:
  memory:           512Mi
  cpu:              500m
Last State: Terminated
  Reason:           OOMKilled
  Exit Code:        137
"""

SQL_OUTPUT = """\
SELECT * FROM orders WHERE status='pending' AND created_at < NOW() - INTERVAL 7 DAY;
-- 1842 rows
-- Execution time: 3042ms
-- Rows: id, customer_id, total_amount, status, created_at
"""

BASH_OUTPUT = """\
$ df -h /var/lib/kubelet
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1        50G   47G  3.0G  94% /var/lib/kubelet
Exit code: 0
"""

def fake_kubectl(state: dict) -> str:
    return KUBECTL_OUTPUT

def fake_sql(state: dict) -> str:
    return SQL_OUTPUT

def fake_bash(state: dict) -> str:
    return BASH_OUTPUT


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ContextStream LangGraph Plugin — Dry Run")
    print("=" * 60)

    engine = ContextStreamEngine(
        model="claude-sonnet-4-6",
        dry_run=True,
        session_id="langgraph_smoke_test",
    )
    cs = ContextStreamPlugin(engine)
    engine.init("Diagnose OOM and disk pressure on production cluster")

    # Simulate LangGraph state
    state: dict = {
        "task": "Investigate web-backend OOMKilled pod and disk usage",
        "messages": [],
        "_cs_lesson_refs": [],
    }

    # -----------------------------------------------------------------------
    # Tool 1: kubectl
    # -----------------------------------------------------------------------
    print("\n--- Tool 1: kubectl ---")
    before_fn = cs.before_node("kubectl", task_key="task")
    after_fn   = cs.after_node("kubectl", output_key="tool_output")

    state.update(before_fn(state))
    assert "_cs_fork_ctx" in state and state["_cs_fork_ctx"] is not None
    print(f"  fork_ctx attached: {state['_cs_fork_ctx'].fork_id}")

    raw = fake_kubectl(state)
    state["tool_output"] = raw
    state.update(after_fn(state))

    assert state.get("tool_output") is None, "Raw output should be evicted from state"
    assert state.get("_cs_fork_ctx") is None, "fork_ctx should be cleared"
    assert len(state["messages"]) == 1
    assert state["messages"][0].content.startswith("[LESSON")
    print(f"  lesson appended: {state['messages'][0].content}")
    print(f"  lesson_refs: {len(state['_cs_lesson_refs'])}")

    # -----------------------------------------------------------------------
    # Tool 2: sql — use wrap_tool_node
    # -----------------------------------------------------------------------
    print("\n--- Tool 2: sql (wrap_tool_node) ---")
    wrapped_sql = cs.wrap_tool_node(
        fake_sql,
        tool_name="sql",
        output_key="tool_output",
        task_key="task",
    )
    patch = wrapped_sql(state)
    state.update(patch)

    assert len(state["messages"]) == 2
    print(f"  lesson appended: {state['messages'][-1].content}")
    print(f"  lesson_refs: {len(state['_cs_lesson_refs'])}")

    # -----------------------------------------------------------------------
    # Tool 3: bash
    # -----------------------------------------------------------------------
    print("\n--- Tool 3: bash (wrap_tool_node) ---")
    wrapped_bash = cs.wrap_tool_node(
        fake_bash,
        tool_name="bash",
        output_key="tool_output",
        task_key="task",
    )
    patch = wrapped_bash(state)
    state.update(patch)

    assert len(state["messages"]) == 3
    print(f"  lesson appended: {state['messages'][-1].content}")
    print(f"  lesson_refs: {len(state['_cs_lesson_refs'])}")

    # -----------------------------------------------------------------------
    # Context injection check
    # -----------------------------------------------------------------------
    print("\n--- inject_context ---")
    patch = cs.inject_context(state, messages_key="messages")
    state.update(patch)
    # inject_context prepends ContextStream ledger as SystemMessage[0]
    first_msg = state["messages"][0]
    assert first_msg.content.startswith("[ContextStream]"), (
        f"Expected [ContextStream] prefix, got: {first_msg.content[:60]}"
    )
    print(f"  context injected: {first_msg.content[:120]!r}")
    print(f"  total messages: {len(state['messages'])}")

    # -----------------------------------------------------------------------
    # Engine status
    # -----------------------------------------------------------------------
    print("\n--- Engine Status ---")
    s = cs.status()
    for k, v in s.items():
        print(f"  {k}: {v}")

    assert s["dry_run"] is True
    assert s["ledger_entries"] == 0  # dry_run skips ledger writes
    assert s["active_lessons"] == 0  # dry_run skips page_table writes

    # -----------------------------------------------------------------------
    # Dry run log
    # -----------------------------------------------------------------------
    print("\n--- Dry Run Log ---")
    for ev in engine.dry_run_log:
        print(f"  [{ev.event}][{ev.tool_name}] {ev.detail}")

    print("\n" + "=" * 60)
    print("PASS — all assertions passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
