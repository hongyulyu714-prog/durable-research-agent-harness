from __future__ import annotations

import asyncio
import json

import pytest

from research_agent.llm import LLMResponse, MockLLM
from research_agent.loop import AgentLoop
from research_agent.models import RunState, RunStatus, ToolCall, ToolResult
from research_agent.registry import ToolRegistry, ToolSpec
from research_agent.state_store import (
    ApprovalRecoveryRequired,
    ApprovalService,
    CheckpointError,
    StateStore,
)
from research_agent.trace import TraceRecorder


def write_registry(calls: list[dict[str, str]]) -> ToolRegistry:
    registry = ToolRegistry()

    async def write_report(**arguments: str) -> dict[str, str]:
        calls.append(arguments)
        return {"written": arguments["path"]}

    registry.register(
        ToolSpec(
            name="write_report",
            description="Write a report after approval.",
            parameters_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            risk="write",
            parallel_safe=False,
            timeout_seconds=1,
            max_retries=0,
        ),
        write_report,
    )
    return registry


def read_write_registry(reads: list[dict[str, str]], writes: list[dict[str, str]]) -> ToolRegistry:
    registry = write_registry(writes)

    async def read_file(**arguments: str) -> dict[str, str]:
        reads.append(arguments)
        return {"path": arguments["path"], "content": "source material"}

    registry.register(
        ToolSpec(
            name="read_file",
            description="Read source material.",
            parameters_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            risk="read",
            parallel_safe=True,
            timeout_seconds=1,
            max_retries=0,
        ),
        read_file,
    )
    return registry


async def no_execute(_: list[ToolCall]) -> list[object]:
    raise AssertionError("a pending write must not reach the ordinary executor")


def test_loop_pauses_then_new_service_approves_once_and_continues(tmp_path):
    store = StateStore(tmp_path / "runs")
    writes: list[dict[str, str]] = []
    registry = write_registry(writes)
    call = ToolCall(
        id="write-1", name="write_report", arguments={"path": "brief.md", "content": "draft"}
    )
    state = RunState.new("write a short report", run_id="run_hitl")
    first_loop = AgentLoop(
        MockLLM([LLMResponse.with_tool_calls([call])]),
        registry,
        no_execute,
        approval_gate=ApprovalService(store, registry),
    )

    paused = asyncio.run(first_loop.run(state))

    assert paused.status is RunStatus.WAITING_APPROVAL
    assert store.load("run_hitl").pending_approval == call
    assert writes == []

    second_loop = AgentLoop(
        MockLLM([LLMResponse.final("The report has been saved.")]), registry, no_execute
    )
    resumed = asyncio.run(
        ApprovalService(store, registry, continuation=second_loop.run).resume(
            "run_hitl", approve=True
        )
    )

    assert writes == [{"path": "brief.md", "content": "draft"}]
    assert resumed.status is RunStatus.COMPLETED
    assert resumed.pending_approval is None
    assert resumed.completed_call_ids == {"write-1"}
    assert json.loads(second_loop.llm.requests[0][0][-1]["content"])["ok"] is True


def test_same_turn_read_and_write_settle_every_tool_call_before_next_model_turn(tmp_path):
    store = StateStore(tmp_path / "runs")
    reads: list[dict[str, str]] = []
    writes: list[dict[str, str]] = []
    registry = read_write_registry(reads, writes)
    read_call = ToolCall(id="read-1", name="read_file", arguments={"path": "notes.md"})
    write_call = ToolCall(
        id="write-1", name="write_report", arguments={"path": "brief.md", "content": "draft"}
    )

    async def execute_reads(calls: list[ToolCall]) -> list[ToolResult]:
        assert [call.id for call in calls] == ["read-1"]
        return [ToolResult.success(call, await registry.invoke(call)) for call in calls]

    first_loop = AgentLoop(
        MockLLM([LLMResponse.with_tool_calls([read_call, write_call])]),
        registry,
        execute_reads,
        approval_gate=ApprovalService(store, registry),
    )
    paused = asyncio.run(first_loop.run(RunState.new("research", run_id="run_mixed")))

    assert paused.status is RunStatus.WAITING_APPROVAL
    assert reads == [{"path": "notes.md"}]
    assert writes == []
    assert [
        message["tool_call_id"] for message in paused.messages if message["role"] == "tool"
    ] == ["read-1"]

    second_loop = AgentLoop(
        MockLLM([LLMResponse.final("Saved after reading the source.")]), registry, execute_reads
    )
    resumed = asyncio.run(
        ApprovalService(store, registry, continuation=second_loop.run).resume(
            "run_mixed", approve=True
        )
    )

    assert writes == [{"path": "brief.md", "content": "draft"}]
    assert resumed.status is RunStatus.COMPLETED
    tool_ids = [
        message["tool_call_id"]
        for message in second_loop.llm.requests[0][0]
        if message["role"] == "tool"
    ]
    assert tool_ids == ["read-1", "write-1"]


def test_multiple_write_calls_are_approved_and_settled_in_order(tmp_path):
    store = StateStore(tmp_path / "runs")
    writes: list[dict[str, str]] = []
    registry = write_registry(writes)
    calls = [
        ToolCall(id="write-1", name="write_report", arguments={"path": "one.md", "content": "one"}),
        ToolCall(id="write-2", name="write_report", arguments={"path": "two.md", "content": "two"}),
    ]
    loop = AgentLoop(
        MockLLM([LLMResponse.with_tool_calls(calls)]),
        registry,
        no_execute,
        approval_gate=ApprovalService(store, registry),
    )
    asyncio.run(loop.run(RunState.new("write twice", run_id="run_many_writes")))

    first = asyncio.run(ApprovalService(store, registry).resume("run_many_writes", approve=True))
    assert first.status is RunStatus.WAITING_APPROVAL
    assert first.pending_approval is not None and first.pending_approval.id == "write-2"
    assert [message["tool_call_id"] for message in first.messages if message["role"] == "tool"] == [
        "write-1"
    ]

    finish_loop = AgentLoop(MockLLM([LLMResponse.final("Both saved.")]), registry, no_execute)
    last = asyncio.run(
        ApprovalService(store, registry, continuation=finish_loop.run).resume(
            "run_many_writes", approve=True
        )
    )
    assert last.status is RunStatus.COMPLETED
    assert [item["path"] for item in writes] == ["one.md", "two.md"]


def test_rejection_emits_model_visible_result_and_does_not_execute(tmp_path):
    store = StateStore(tmp_path / "runs")
    writes: list[dict[str, str]] = []
    service = ApprovalService(store, write_registry(writes))
    state = RunState.new("write", run_id="run_reject")
    service.request_approval(
        state, ToolCall("write-1", "write_report", {"path": "brief.md", "content": "draft"})
    )

    resumed = asyncio.run(service.resume("run_reject", reject_reason="please shorten the report"))

    payload = json.loads(resumed.messages[-1]["content"])
    assert writes == []
    assert payload["error"]["kind"] == "approval_rejected"
    assert payload["error"]["message"] == "please shorten the report"


def test_started_execution_marker_blocks_blind_repeat_after_save_failure(tmp_path, monkeypatch):
    store = StateStore(tmp_path / "runs")
    writes: list[dict[str, str]] = []
    service = ApprovalService(store, write_registry(writes))
    state = RunState.new("write", run_id="run_retry_fence")
    service.request_approval(
        state, ToolCall("write-1", "write_report", {"path": "brief.md", "content": "draft"})
    )
    original_save = store.save
    call_count = 0

    def fail_after_effect(candidate: RunState):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise CheckpointError("simulated post-write checkpoint failure")
        return original_save(candidate)

    monkeypatch.setattr(store, "save", fail_after_effect)
    with pytest.raises(CheckpointError):
        asyncio.run(service.resume("run_retry_fence", approve=True))

    assert writes == [{"path": "brief.md", "content": "draft"}]
    # A fresh process sees the durable fence and refuses an unsafe retry.
    fresh_service = ApprovalService(StateStore(store.root), write_registry(writes))
    with pytest.raises(ApprovalRecoveryRequired):
        asyncio.run(fresh_service.resume("run_retry_fence", approve=True))
    assert writes == [{"path": "brief.md", "content": "draft"}]


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
