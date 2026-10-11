"""The project-owned multi-turn Agent control loop.

No Agent framework owns this loop: the harness asks the model for a proposal,
validates it, executes approved calls, returns results to the model, and decides
when the run terminates.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

from research_agent.llm import LLMAdapter
from research_agent.models import RunState, RunStatus, ToolCall, ToolResult


class ToolRegistry(Protocol):
    """The narrow registry boundary needed by the loop.

    ``validate_calls`` returns a ``ToolResult`` for each rejected call (such as
    an unknown tool or invalid parameters), and no result for valid calls.  The
    scheduler receives only the remaining valid calls.
    """

    def schemas(self) -> Sequence[Mapping[str, Any]]: ...

    def validate_calls(self, calls: Sequence[ToolCall]) -> Sequence[ToolResult]: ...

    def get_spec(self, name: str) -> Any | None: ...


ToolExecutor = Callable[
    [Sequence[ToolCall]], Awaitable[Sequence[ToolResult]] | Sequence[ToolResult]
]


class ToolScheduler(Protocol):
    """The small part of a scheduler the loop needs after validation."""

    def execute(
        self, calls: Sequence[ToolCall]
    ) -> Awaitable[Sequence[ToolResult]] | Sequence[ToolResult]: ...


class ApprovalGate(Protocol):
    """Durably pauses a run before a write-capable tool can execute."""

    def request_approval(self, state: RunState, call: ToolCall) -> RunState: ...


def tool_result_message(result: ToolResult) -> dict[str, str]:
    """Encode an outcome in the standard ``role=tool`` follow-up message."""

    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "name": result.tool_name,
        "content": json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True),
    }


class AgentLoop:
    """Advance a :class:`RunState` until the model answers or the harness stops.

    ``executor`` is intentionally injected.  Issue #3 supplies the concurrent
    scheduler; keeping it at this boundary lets this loop be unit tested with
    simple offline fake tools.
    """

    def __init__(
        self,
        llm: LLMAdapter,
        registry: ToolRegistry,
        executor: ToolExecutor | ToolScheduler,
        *,
        max_steps: int = 8,
        approval_gate: ApprovalGate | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least one")
        self.llm = llm
        self.registry = registry
        self.executor: ToolExecutor = executor if callable(executor) else executor.execute
        self.max_steps = max_steps
        self.approval_gate = approval_gate

    async def run(self, state: RunState) -> RunState:
        """Run only while active; terminal and paused states are left unchanged.

        A max-step overrun becomes ``FAILED`` with ``failure_reason`` set to
        ``max_steps_exceeded``.  Invalid/unknown calls become normal tool
        messages so the next model turn can repair its request.
        """

        while state.status is RunStatus.RUNNING:
            if state.step >= self.max_steps:
                state.status = RunStatus.FAILED
                state.failure_reason = "max_steps_exceeded"
                state.record("run_failed", reason=state.failure_reason, max_steps=self.max_steps)
                break

            response = await self.llm.complete(state.messages, self.registry.schemas())
            state.record("llm_response", has_tool_calls=bool(response.tool_calls))

            if response.is_final:
                state.messages.append(response.assistant_message())
                state.final_answer = response.content
                state.status = RunStatus.COMPLETED
                state.record("run_completed")
                break

            if not response.tool_calls:
                state.status = RunStatus.FAILED
                state.failure_reason = "invalid_llm_response"
                state.record("run_failed", reason=state.failure_reason)
                break

            state.messages.append(response.assistant_message())
            approval_call = self._approval_call(response.tool_calls)
            if approval_call is not None:
                self.approval_gate.request_approval(state, approval_call)
                break
            results = await self._process_calls(response.tool_calls)
            for result in results:
                state.messages.append(tool_result_message(result))
                state.completed_call_ids.add(result.tool_call_id)
                state.record("tool_result", tool_call_id=result.tool_call_id, ok=result.ok)
            state.step += 1

        return state

    def _approval_call(self, calls: Sequence[ToolCall]) -> ToolCall | None:
        """Return the first validated write request when HITL is configured.

        This check belongs before the executor.  The scheduler may reject
        writes defensively, but it must never be the component that decides
        whether a human approval checkpoint should be created.
        """

        if self.approval_gate is None:
            return None
        for call in calls:
            spec = self.registry.get_spec(call.name)
            if (
                spec is not None
                and spec.risk == "write"
                and self.registry.validate_call(call) is None
            ):
                return call
        return None

    async def _process_calls(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        """Validate, execute valid calls, and restore original call ordering."""

        if len({call.id for call in calls}) != len(calls):
            return [
                ToolResult.failure(
                    call,
                    kind="duplicate_tool_call_id",
                    message="each tool call in a model response must have a unique id",
                )
                for call in calls
            ]

        validation_results = list(self.registry.validate_calls(calls))
        self._validate_result_ids(validation_results, calls, source="validator")
        validation_results = self._match_result_names(validation_results, calls, source="validator")
        rejected_ids = {result.tool_call_id for result in validation_results}
        valid_calls = [call for call in calls if call.id not in rejected_ids]

        execution_results: list[ToolResult] = []
        if valid_calls:
            try:
                output = self.executor(valid_calls)
                if inspect.isawaitable(output):
                    output = await output
                execution_results = list(output)
                self._validate_result_ids(execution_results, valid_calls, source="executor")
                execution_results = self._match_result_names(
                    execution_results, valid_calls, source="executor"
                )
            except Exception as exc:
                execution_results = [
                    ToolResult.failure(
                        call,
                        kind="execution_error",
                        message=f"tool execution failed: {exc}",
                    )
                    for call in valid_calls
                ]

        by_id = {result.tool_call_id: result for result in validation_results + execution_results}
        for call in valid_calls:
            by_id.setdefault(
                call.id,
                ToolResult.failure(
                    call,
                    kind="missing_tool_result",
                    message="the tool executor returned no result for this call",
                ),
            )
        return [by_id[call.id] for call in calls]

    @staticmethod
    def _validate_result_ids(
        results: Sequence[ToolResult], calls: Sequence[ToolCall], *, source: str
    ) -> None:
        expected = {call.id for call in calls}
        ids = [result.tool_call_id for result in results]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{source} returned duplicate tool_call_id values")
        unknown = set(ids) - expected
        if unknown:
            raise ValueError(f"{source} returned results for unknown call IDs: {sorted(unknown)}")

    @staticmethod
    def _match_result_names(
        results: Sequence[ToolResult], calls: Sequence[ToolCall], *, source: str
    ) -> list[ToolResult]:
        """Reject results whose ID was preserved but whose tool identity was not."""

        calls_by_id = {call.id: call for call in calls}
        mismatch_kind = "execution_error" if source == "executor" else "validation_error"
        matched: list[ToolResult] = []
        for result in results:
            call = calls_by_id[result.tool_call_id]
            if result.tool_name == call.name:
                matched.append(result)
                continue
            matched.append(
                ToolResult.failure(
                    call,
                    kind=mismatch_kind,
                    message=(
                        f"{source} returned tool_name {result.tool_name!r} for "
                        f"call {call.id!r}; expected {call.name!r}"
                    ),
                )
            )
        return matched
