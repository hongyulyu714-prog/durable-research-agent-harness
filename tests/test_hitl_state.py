import asyncio
import json

import pytest

from research_agent.models import RunState, RunStatus, ToolCall
from research_agent.registry import ToolRegistry
from research_agent.state_store import ApprovalService, CheckpointError, StateStore
from research_agent.trace import TraceRecorder


def paused_state() -> RunState:
    state = RunState.new("write a short report", run_id="run_hitl")
    state.pending_approval = ToolCall(
        "write-1", "write_report", {"path": "brief.md", "content": "draft"}
    )
    state.transition_to(RunStatus.WAITING_APPROVAL)
    return state


def write_registry(calls: list[dict[str, str]]) -> ToolRegistry:
    registry = ToolRegistry()

    async def write_report(**arguments: str) -> dict[str, str]:
        calls.append(arguments)
        return {"written": arguments["path"]}

    from research_agent.models import ToolSpec

    registry.register(ToolSpec("write_report", risk="write"), write_report)
    return registry


def test_waiting_approval_is_persisted_without_executing_write(tmp_path):
    store = StateStore(tmp_path / "runs")
    state = RunState.new("write a short report", run_id="run_hitl")
    call = ToolCall("write-1", "write_report", {"path": "brief.md", "content": "draft"})
    ApprovalService(store, write_registry([])).request_approval(state, call)

    restored = store.load("run_hitl")
    assert restored.status is RunStatus.WAITING_APPROVAL
    assert restored.pending_approval is not None
    assert not (tmp_path / "brief.md").exists()
    assert TraceRecorder(store.root, "run_hitl").events()[-1]["type"] == "approval_requested"


def test_new_service_approves_exactly_one_persisted_call(tmp_path):
    store = StateStore(tmp_path / "runs")
    store.save(paused_state())
    writes: list[dict[str, str]] = []

    resumed = asyncio.run(
        ApprovalService(store, write_registry(writes)).resume("run_hitl", approve=True)
    )

    assert writes == [{"path": "brief.md", "content": "draft"}]
    assert resumed.status is RunStatus.RUNNING
    assert resumed.pending_approval is None
    assert resumed.completed_call_ids == {"write-1"}
    assert json.loads(resumed.messages[-1]["content"])["ok"] is True


def test_rejection_emits_model_visible_result_and_does_not_execute(tmp_path):
    store = StateStore(tmp_path / "runs")
    store.save(paused_state())
    writes: list[dict[str, str]] = []

    resumed = asyncio.run(
        ApprovalService(store, write_registry(writes)).resume(
            "run_hitl", reject_reason="please shorten the report"
        )
    )

    payload = json.loads(resumed.messages[-1]["content"])
    assert writes == []
    assert payload["error"]["kind"] == "approval_rejected"
    assert payload["error"]["message"] == "please shorten the report"


def test_corrupt_checkpoint_and_illegal_transition_are_rejected(tmp_path):
    store = StateStore(tmp_path / "runs")
    path = store.checkpoint_path("run_broken")
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(CheckpointError, match="corrupt"):
        store.load("run_broken")

    state = RunState.new("done")
    state.transition_to(RunStatus.COMPLETED)
    with pytest.raises(ValueError, match="illegal"):
        state.transition_to(RunStatus.RUNNING)


def test_trace_redacts_secret_and_content(tmp_path):
    recorder = TraceRecorder(tmp_path, "run_trace")
    recorder.record("tool_started", api_key="secret-key", content="full report body")
    event = recorder.events()[0]

    assert event["api_key"] == "[REDACTED]"
    assert event["content"]["char_count"] == len("full report body")
