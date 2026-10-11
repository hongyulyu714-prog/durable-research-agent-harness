from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from research_agent.llm import LLMResponse, MockLLM
from research_agent.loop import AgentLoop
from research_agent.models import RunState, RunStatus, ToolCall, ToolResult


class FakeRegistry:
    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": "lookup"}}]

    def validate_calls(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        outcomes = []
        for call in calls:
            if call.name != "lookup":
                outcomes.append(
                    ToolResult.failure(
                        call, kind="unknown_tool", message=f"unknown tool: {call.name}"
                    )
                )
            elif not isinstance(call.arguments, dict) or "query" not in call.arguments:
                outcomes.append(
                    ToolResult.failure(
                        call, kind="invalid_arguments", message="lookup requires a query"
                    )
                )
        return outcomes


def test_loop_returns_tool_result_to_second_model_turn() -> None:
    call = ToolCall(id="call_lookup", name="lookup", arguments={"query": "Ada Lovelace"})
    llm = MockLLM(
        [
            LLMResponse.with_tool_calls([call]),
            LLMResponse.final("Ada Lovelace was a pioneer."),
        ]
    )

    async def execute(calls: Sequence[ToolCall]) -> list[ToolResult]:
        return [ToolResult.success(calls[0], {"fact": "first programmer"})]

    state = asyncio.run(
        AgentLoop(llm, FakeRegistry(), execute).run(RunState.new("Who was Ada Lovelace?"))
    )

    assert state.status is RunStatus.COMPLETED
    assert state.final_answer == "Ada Lovelace was a pioneer."
    assert len(llm.requests) == 2
    second_turn = llm.requests[1][0]
    tool_message = next(message for message in second_turn if message["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_lookup"
    assert json.loads(tool_message["content"])["data"] == {"fact": "first programmer"}


def test_invalid_calls_return_structured_results_and_loop_continues() -> None:
    bad_call = ToolCall(id="call_bad", name="not_registered", arguments={})
    llm = MockLLM(
        [
            LLMResponse.with_tool_calls([bad_call]),
            LLMResponse.final("I cannot use that tool."),
        ]
    )

    async def execute(_: Sequence[ToolCall]) -> list[ToolResult]:
        raise AssertionError("invalid calls must not reach the executor")

    state = asyncio.run(
        AgentLoop(llm, FakeRegistry(), execute).run(RunState.new("Try an unknown tool"))
    )

    assert state.status is RunStatus.COMPLETED
    error = json.loads(next(m for m in llm.requests[1][0] if m["role"] == "tool")["content"])
    assert error["error"]["kind"] == "unknown_tool"


def test_max_steps_stops_an_infinite_tool_loop() -> None:
    call = ToolCall(id="call_1", name="lookup", arguments={"query": "again"})
    llm = MockLLM([LLMResponse.with_tool_calls([call]), LLMResponse.with_tool_calls([call])])

    async def execute(calls: Sequence[ToolCall]) -> list[ToolResult]:
        return [ToolResult.success(calls[0], "done")]

    state = asyncio.run(
        AgentLoop(llm, FakeRegistry(), execute, max_steps=2).run(RunState.new("Loop"))
    )

    assert state.status is RunStatus.FAILED
    assert state.failure_reason == "max_steps_exceeded"
    assert state.step == 2
    assert len(llm.requests) == 2


def test_non_json_tool_data_becomes_a_structured_execution_failure() -> None:
    call = ToolCall(id="call_path", name="lookup", arguments={"query": "path"})
    llm = MockLLM([LLMResponse.with_tool_calls([call]), LLMResponse.final("Recovered")])

    async def execute(calls: Sequence[ToolCall]) -> list[ToolResult]:
        return [ToolResult.success(calls[0], Path("not-json-serializable"))]

    state = asyncio.run(AgentLoop(llm, FakeRegistry(), execute).run(RunState.new("Return a path")))

    assert state.status is RunStatus.COMPLETED
    result = json.loads(next(m for m in llm.requests[1][0] if m["role"] == "tool")["content"])
    assert result["error"]["kind"] == "execution_error"
    assert result["tool_name"] == "lookup"


def test_executor_result_with_mismatched_tool_name_becomes_execution_failure() -> None:
    call = ToolCall(id="call_lookup", name="lookup", arguments={"query": "name"})
    llm = MockLLM([LLMResponse.with_tool_calls([call]), LLMResponse.final("Recovered")])

    async def execute(_: Sequence[ToolCall]) -> list[ToolResult]:
        return [ToolResult("call_lookup", "different_tool", True, {"unexpected": True})]

    state = asyncio.run(AgentLoop(llm, FakeRegistry(), execute).run(RunState.new("Check name")))

    assert state.status is RunStatus.COMPLETED
    result = json.loads(next(m for m in llm.requests[1][0] if m["role"] == "tool")["content"])
    assert result["error"]["kind"] == "execution_error"
    assert result["tool_name"] == "lookup"
