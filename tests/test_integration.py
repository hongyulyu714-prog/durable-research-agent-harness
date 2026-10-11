"""Offline end-to-end proof of the durable research workflow.

This deliberately uses a scripted model and controlled async tools so the
acceptance path is deterministic and never needs an API key or network access.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from research_agent.cli import ApplicationService
from research_agent.llm import LLMResponse, MockLLM
from research_agent.loop import AgentLoop
from research_agent.models import RunState, RunStatus, ToolCall
from research_agent.registry import ToolRegistry, ToolSpec
from research_agent.scheduler import AsyncToolScheduler
from research_agent.state_store import ApprovalService, StateStore
from research_agent.trace import TraceRecorder


def _spec(name: str, *, risk: str = "read", parallel_safe: bool = True) -> ToolSpec:
    properties = {"path": {"type": "string"}}
    if name == "fetch_url":
        properties = {"url": {"type": "string"}}
    if name == "write_report":
        properties = {"path": {"type": "string"}, "content": {"type": "string"}}
    return ToolSpec(
        name=name,
        description=f"integration fixture for {name}",
        parameters_schema={
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        risk=risk,  # type: ignore[arg-type]
        parallel_safe=parallel_safe,
        timeout_seconds=0.5,
        max_retries=0,
    )


@pytest.mark.asyncio
async def test_durable_research_flow_records_concurrency_failure_and_cross_process_resume(tmp_path):
    """Read and web work overlap; a web failure never blocks approved output."""

    registry = ToolRegistry()
    started: set[str] = set()
    both_started = asyncio.Event()
    writes: list[dict[str, str]] = []

    async def wait_for_peer(name: str) -> None:
        started.add(name)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()

    async def read_file(path: str) -> dict[str, str]:
        await wait_for_peer("read_file")
        await asyncio.sleep(0.04)
        return {"path": path, "content": "local source"}

    async def fetch_url(url: str) -> dict[str, str]:
        await wait_for_peer("fetch_url")
        await asyncio.sleep(0.02)
        raise httpx.ConnectError(f"simulated unreachable source: {url}")

    async def write_report(path: str, content: str) -> dict[str, str]:
        writes.append({"path": path, "content": content})
        return {"path": path, "status": "written"}

    registry.register(_spec("read_file"), read_file)
    registry.register(_spec("fetch_url"), fetch_url)
    registry.register(_spec("write_report", risk="write", parallel_safe=False), write_report)

    store = StateStore(tmp_path / "runs")
    scheduler = AsyncToolScheduler(registry, max_concurrency=2, retry_base_seconds=0)
    read_call = ToolCall("read-1", "read_file", {"path": "source.md"})
    web_call = ToolCall("web-1", "fetch_url", {"url": "https://example.test/source"})
    write_call = ToolCall(
        "write-1",
        "write_report",
        {"path": "brief.md", "content": "# Research brief\n\nBased on available sources."},
    )
    first_loop = AgentLoop(
        MockLLM(
            [
                LLMResponse.with_tool_calls([read_call, web_call]),
                LLMResponse.with_tool_calls([write_call]),
            ]
        ),
        registry,
        scheduler,
        approval_gate=ApprovalService(store, registry),
    )

    async def start(task: str) -> RunState:
        return await first_loop.run(RunState.new(task, run_id="offline_integration"))

    paused = await ApplicationService(store, registry, start=start).run("Prepare a research brief.")

    assert paused.status is RunStatus.WAITING_APPROVAL
    assert started == {"read_file", "fetch_url"}
    assert writes == []
    assert [result["tool_call_id"] for result in _tool_payloads(paused)] == ["read-1", "web-1"]
    assert _tool_payloads(paused)[0]["ok"] is True
    assert _tool_payloads(paused)[1]["error"]["kind"] == "network_error"
    batch = next(event for event in paused.events if event.kind == "tool_batch_completed")
    assert batch.data["failed_call_count"] == 1
    assert batch.data["duration_ms"] < batch.data["aggregate_tool_duration_ms"]

    # A fresh application service mirrors a new CLI process: it only sees the
    # persisted checkpoint and must execute the saved write exactly once.
    final_loop = AgentLoop(
        MockLLM([LLMResponse.final("The approved report was written.")]), registry, scheduler
    )
    resumed = await ApplicationService(
        StateStore(store.root), registry, continuation=final_loop.run
    ).resume("offline_integration", approve=True, reject_reason=None)

    assert resumed.status is RunStatus.COMPLETED
    assert writes == [
        {"path": "brief.md", "content": "# Research brief\n\nBased on available sources."}
    ]
    assert [payload["tool_call_id"] for payload in _tool_payloads(resumed)] == [
        "read-1",
        "web-1",
        "write-1",
    ]
    final_turn = final_loop.llm.requests[0][0]
    assert [message["tool_call_id"] for message in final_turn if message["role"] == "tool"] == [
        "read-1",
        "web-1",
        "write-1",
    ]

    recorder = TraceRecorder(store.root, "offline_integration")
    trace = recorder.events()
    event_types = [event["type"] for event in trace]
    assert {
        "llm_response",
        "tool_result",
        "tool_batch_completed",
        "approval_requested",
        "approval_granted",
        "tool_started",
        "tool_finished",
        "run_completed",
    }.issubset(event_types)
    state_events = [event for event in trace if event.get("source") == "run_state"]
    assert len({event["state_event_id"] for event in state_events}) == len(state_events)
    summary = recorder.summary(resumed)
    assert summary["status"] == "completed"
    assert summary["tool_failure_count"] == 1
    assert summary["approval_decisions"] == ["approval_granted"]


def _tool_payloads(state: RunState) -> list[dict[str, object]]:
    return [
        json.loads(message["content"])
        for message in state.messages
        if message["role"] == "tool"
    ]
