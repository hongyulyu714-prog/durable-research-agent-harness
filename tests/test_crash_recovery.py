"""Offline fault-injection proofs for durable approval recovery."""

from __future__ import annotations

import asyncio

import pytest

from research_agent.cli import ApplicationService
from research_agent.llm import LLMResponse, MockLLM
from research_agent.loop import AgentLoop
from research_agent.models import RunState, RunStatus, ToolCall
from research_agent.registry import ToolRegistry
from research_agent.scheduler import AsyncToolScheduler
from research_agent.state_store import (
    ApprovalRecoveryRequired,
    ApprovalService,
    CheckpointError,
    StateStore,
)
from research_agent.tools import build_local_registry
from research_agent.trace import TraceRecorder


def _file_registry(workspace, writes: list[dict[str, str]]) -> ToolRegistry:
    """Use real workspace tools while counting only side-effecting invocations."""

    base = build_local_registry(workspace=workspace)
    read = base.get("read_file")
    write = base.get("write_report")
    assert read is not None and write is not None

    registry = ToolRegistry()
    registry.register(read.spec, read.handler)

    async def counted_write(path: str, content: str):
        writes.append({"path": path, "content": content})
        return await write.handler(path=path, content=content)

    registry.register(write.spec, counted_write)
    return registry


def _paused_run(store: StateStore, workspace, writes: list[dict[str, str]]) -> RunState:
    registry = _file_registry(workspace, writes)
    scheduler = AsyncToolScheduler(registry, retry_base_seconds=0)
    read = ToolCall("read-source", "read_file", {"path": "source.txt"})
    write = ToolCall(
        "write-brief",
        "write_report",
        {"path": "brief.md", "content": "# Durable brief\n\nCheckpoint before approval."},
    )
    loop = AgentLoop(
        MockLLM([LLMResponse.with_tool_calls([read, write])]),
        registry,
        scheduler,
        approval_gate=ApprovalService(store, registry),
    )
    return asyncio.run(loop.run(RunState.new("summarize the local source", run_id="fault-proof")))


def test_fresh_service_recovers_checkpoint_and_writes_real_report_once(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "data").mkdir(parents=True)
    (workspace / "reports").mkdir()
    (workspace / "data" / "source.txt").write_text("durable source\n", encoding="utf-8")
    store = StateStore(tmp_path / "runs")
    writes: list[dict[str, str]] = []

    paused = _paused_run(store, workspace, writes)

    assert paused.status is RunStatus.WAITING_APPROVAL
    assert writes == []
    checkpoint = store.load("fault-proof")
    assert checkpoint.pending_approval is not None
    assert checkpoint.pending_approval.id == "write-brief"
    assert not (workspace / "reports" / "brief.md").exists()

    # New registry, scheduler, loop and store model a new process that knows
    # only the saved checkpoint, not the original in-memory AgentLoop.
    fresh_registry = _file_registry(workspace, writes)
    fresh_loop = AgentLoop(
        MockLLM([LLMResponse.final("The approved report was written.")]),
        fresh_registry,
        AsyncToolScheduler(fresh_registry, retry_base_seconds=0),
    )
    resumed = asyncio.run(
        ApplicationService(
            StateStore(store.root), fresh_registry, continuation=fresh_loop.run
        ).resume("fault-proof", approve=True, reject_reason=None)
    )

    expected_content = "# Durable brief\n\nCheckpoint before approval."
    assert resumed.status is RunStatus.COMPLETED
    assert writes == [{"path": "brief.md", "content": expected_content}]
    assert (workspace / "reports" / "brief.md").read_text(encoding="utf-8") == expected_content
    assert {"approval_granted", "tool_finished", "run_completed"}.issubset(
        {event["type"] for event in TraceRecorder(store.root, "fault-proof").events()}
    )


def test_crash_after_real_write_requires_manual_recovery_instead_of_retry(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "data").mkdir(parents=True)
    (workspace / "reports").mkdir()
    (workspace / "data" / "source.txt").write_text("durable source\n", encoding="utf-8")
    store = StateStore(tmp_path / "runs")
    writes: list[dict[str, str]] = []

    paused = _paused_run(store, workspace, writes)
    assert paused.status is RunStatus.WAITING_APPROVAL

    first_registry = _file_registry(workspace, writes)
    original_save = store.save
    save_calls = 0

    def fail_result_checkpoint(state: RunState):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise CheckpointError("simulated crash before the result checkpoint")
        return original_save(state)

    monkeypatch.setattr(store, "save", fail_result_checkpoint)
    with pytest.raises(CheckpointError, match="simulated crash"):
        asyncio.run(ApprovalService(store, first_registry).resume("fault-proof", approve=True))

    report = workspace / "reports" / "brief.md"
    assert report.read_text(encoding="utf-8") == "# Durable brief\n\nCheckpoint before approval."
    assert len(writes) == 1

    # The retry fence was atomically persisted *before* the file write.  A new
    # process must stop for manual inspection rather than duplicate the effect.
    fresh_registry = _file_registry(workspace, writes)
    with pytest.raises(ApprovalRecoveryRequired, match="may already have run"):
        asyncio.run(
            ApprovalService(StateStore(store.root), fresh_registry).resume(
                "fault-proof", approve=True
            )
        )

    assert len(writes) == 1
    assert report.read_text(encoding="utf-8") == "# Durable brief\n\nCheckpoint before approval."
    assert "approval_recovery_required" in {
        event["type"] for event in TraceRecorder(store.root, "fault-proof").events()
    }
