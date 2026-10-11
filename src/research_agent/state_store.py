"""Atomic JSON checkpoint persistence and restore.

Checkpoints are intentionally small, local files.  A temporary file is fsynced
and then atomically replaced so a kill or power loss never leaves a half-JSON
checkpoint at the canonical path.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from research_agent.models import RunState, RunStatus, ToolCall, ToolResult
from research_agent.registry import ToolRegistry
from research_agent.trace import TraceRecorder


class CheckpointError(RuntimeError):
    """A checkpoint is absent, malformed, or not safe to address."""


class ApprovalRecoveryRequired(RuntimeError):
    """A write may have happened before its result checkpoint was committed."""


class StateStore:
    """Read and write ``runs/<run_id>/checkpoint.json`` atomically."""

    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def run_directory(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise CheckpointError("run_id must be a single non-empty path component")
        return self.root / run_id

    def checkpoint_path(self, run_id: str) -> Path:
        return self.run_directory(run_id) / "checkpoint.json"

    def save(self, state: RunState) -> Path:
        """Durably replace the checkpoint for *state* and return its path."""

        target = self.checkpoint_path(state.run_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".checkpoint-", suffix=".tmp", dir=target.parent, text=True
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            self._fsync_directory(target.parent)
        except OSError as exc:
            raise CheckpointError(f"could not save checkpoint for {state.run_id}: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def load(self, run_id: str) -> RunState:
        """Load a validated state without silently accepting a corrupt file."""

        path = self.checkpoint_path(run_id)
        try:
            with path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
        except FileNotFoundError as exc:
            raise CheckpointError(f"checkpoint not found for run {run_id}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"checkpoint for run {run_id} is corrupt: {exc}") from exc
        if not isinstance(raw, dict):
            raise CheckpointError(f"checkpoint for run {run_id} must be a JSON object")
        try:
            state = RunState.from_dict(raw)
        except ValueError as exc:
            raise CheckpointError(f"checkpoint for run {run_id} is invalid: {exc}") from exc
        if state.run_id != run_id:
            raise CheckpointError("checkpoint run_id does not match its directory")
        return state

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        """Persist the rename on POSIX; harmlessly skip platforms without it."""

        if os.name == "nt":
            return
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


Continuation = Callable[[RunState], Awaitable[RunState] | RunState]


class ApprovalService:
    """Apply a human decision without asking the model to propose a new write.

    The service intentionally executes a persisted ``pending_approval`` by its
    saved id, name, and arguments.  It never accepts a replacement call from
    the resuming process.  ``continuation`` is optional to keep this boundary
    independent from the AgentLoop implementation.
    """

    def __init__(
        self,
        store: StateStore,
        registry: ToolRegistry,
        *,
        continuation: Continuation | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.continuation = continuation

    def request_approval(self, state: RunState, call: ToolCall) -> RunState:
        """Persist a high-risk call before exposing its approval request.

        The loop calls this instead of directly mutating ``pending_approval``.
        Saving first means a new CLI process always sees the exact call that
        was proposed, even if the original process exits immediately after it
        reports that approval is needed.
        """

        if state.status is not RunStatus.RUNNING:
            raise ValueError("approval can only be requested from a running state")
        if call.id in state.completed_call_ids:
            raise ValueError("cannot request approval for an already completed call")
        registered = self.registry.get(call.name)
        if registered is None or registered.spec.risk != "write":
            raise ValueError("approval can only be requested for a registered write tool")
        if self.registry.validate_call(call) is not None:
            raise ValueError("approval can only be requested for a valid tool call")
        state.pending_approval = call
        state.approval_execution_started = False
        previous = state.status
        state.transition_to(RunStatus.WAITING_APPROVAL)
        self.store.save(state)
        tracer = TraceRecorder(self.store.root, state.run_id)
        tracer.record(
            "state_transition",
            state=state,
            from_status=previous.value,
            to_status=state.status.value,
        )
        tracer.record("approval_requested", state=state, call_id=call.id, tool=call.name)
        tracer.write_summary(state)
        return state

    async def resume(
        self,
        run_id: str,
        *,
        approve: bool = False,
        reject_reason: str | None = None,
    ) -> RunState:
        """Load a paused run, record one decision, then optionally continue it."""

        if approve == (reject_reason is not None):
            raise ValueError("choose exactly one of approve or reject_reason")
        state = self.store.load(run_id)
        if state.status is not RunStatus.WAITING_APPROVAL or state.pending_approval is None:
            raise ValueError("only a run waiting for approval can be resumed")

        tracer = TraceRecorder(self.store.root, run_id)
        call = state.pending_approval
        if approve:
            if state.approval_execution_started:
                tracer.record(
                    "approval_recovery_required",
                    state=state,
                    call_id=call.id,
                    tool=call.name,
                )
                raise ApprovalRecoveryRequired(
                    "the approved write may already have run; "
                    "inspect the target before resolving it"
                )
            tracer.record("approval_granted", state=state, call_id=call.id, tool=call.name)
            # This checkpoint is the retry fence.  If the process fails after
            # the side effect but before the result is saved, the next resume
            # sees this marker and refuses to execute the write again.
            state.approval_execution_started = True
            self.store.save(state)
            tracer.record("tool_execution_started", state=state, call_id=call.id, tool=call.name)
            result = await self._execute_persisted_call(state, call, tracer)
        else:
            reason = reject_reason or "approval rejected"
            tracer.record(
                "approval_rejected", state=state, call_id=call.id, tool=call.name, reason=reason
            )
            result = ToolResult.failure(call, kind="approval_rejected", message=reason)

        state.messages.append(_tool_result_message(result))
        state.completed_call_ids.add(call.id)
        state.pending_approval = None
        state.approval_execution_started = False
        previous = state.status
        state.transition_to(RunStatus.RUNNING)
        tracer.record(
            "state_transition",
            state=state,
            from_status=previous.value,
            to_status=state.status.value,
        )
        self.store.save(state)

        if self.continuation is not None:
            continued = self.continuation(state)
            state = await continued if inspect.isawaitable(continued) else continued
            self.store.save(state)
        tracer.write_summary(state)
        return state

    async def _execute_persisted_call(
        self, state: RunState, call: ToolCall, tracer: TraceRecorder
    ) -> ToolResult:
        if call.id in state.completed_call_ids:
            return ToolResult.failure(
                call,
                kind="already_completed",
                message="the persisted approval call was already completed; it was not run again",
            )
        registered = self.registry.get(call.name)
        if registered is None:
            return ToolResult.failure(
                call, kind="unknown_tool", message=f"tool is not registered: {call.name}"
            )
        if registered.spec.risk != "write":
            return ToolResult.failure(
                call,
                kind="approval_not_required",
                message=f"persisted call {call.name} is not a write tool",
            )
        started = time.perf_counter()
        tracer.record("tool_started", state=state, call_id=call.id, tool=call.name)
        try:
            data = await self.registry.invoke(call)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1_000)
            result = ToolResult.failure(
                call,
                kind="execution_error",
                message=f"{type(exc).__name__}: {exc}",
                attempts=1,
                duration_ms=duration_ms,
            )
            tracer.record(
                "tool_error",
                state=state,
                call_id=call.id,
                tool=call.name,
                error_kind=result.error.kind if result.error else "execution_error",
                duration_ms=duration_ms,
            )
            return result
        duration_ms = round((time.perf_counter() - started) * 1_000)
        result = ToolResult.success(call, data=data, attempts=1, duration_ms=duration_ms)
        tracer.record(
            "tool_finished",
            state=state,
            call_id=call.id,
            tool=call.name,
            ok=True,
            duration_ms=duration_ms,
        )
        return result


def _tool_result_message(result: ToolResult) -> dict[str, str]:
    """Render a stored result in the model's normal tool-message protocol."""

    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "name": result.tool_name,
        "content": json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True),
    }
