"""Serializable contracts shared by the harness modules.

The contracts deliberately use the standard library rather than provider SDK
objects.  A checkpoint or a recorded test trajectory can therefore be encoded
as JSON and restored without an OpenAI client being installed or configured.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


def _utc_now() -> str:
    """Return an RFC 3339 timestamp suitable for JSONL traces."""

    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, init=False)
class ToolCall:
    """A single tool call proposed by a model.

    ``id`` and ``name`` are accepted as aliases because that is the naming used
    by the OpenAI tool-call payload.  The longer names are kept in serialized
    state to make the identifier's purpose unambiguous.
    """

    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, Any] | str

    def __init__(
        self,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        arguments: Mapping[str, Any] | str | None = None,
        *,
        id: str | None = None,
        name: str | None = None,
    ) -> None:
        if tool_call_id is not None and id is not None and tool_call_id != id:
            raise ValueError("tool_call_id and id must match when both are supplied")
        if tool_name is not None and name is not None and tool_name != name:
            raise ValueError("tool_name and name must match when both are supplied")
        resolved_id = tool_call_id if tool_call_id is not None else id
        resolved_name = tool_name if tool_name is not None else name
        if not resolved_id:
            raise ValueError("a tool call requires a non-empty tool_call_id")
        if not resolved_name:
            raise ValueError("a tool call requires a non-empty tool_name")
        object.__setattr__(self, "tool_call_id", resolved_id)
        object.__setattr__(self, "tool_name", resolved_name)
        object.__setattr__(self, "arguments", {} if arguments is None else arguments)

    @property
    def id(self) -> str:
        return self.tool_call_id

    @property
    def name(self) -> str:
        return self.tool_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments)
            if isinstance(self.arguments, Mapping)
            else self.arguments,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolCall:
        return cls(
            tool_call_id=value.get("tool_call_id", value.get("id")),
            tool_name=value.get("tool_name", value.get("name")),
            arguments=value.get("arguments", {}),
        )


@dataclass(frozen=True)
class ToolError:
    """A model-visible, structured reason why a tool call did not succeed."""

    kind: str
    message: str
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolError:
        return cls(
            kind=str(value["kind"]),
            message=str(value["message"]),
            retryable=bool(value.get("retryable", False)),
        )


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one proposed call, always tied to its original ID."""

    tool_call_id: str
    tool_name: str
    ok: bool
    data: JSONValue | Any = None
    error: ToolError | None = None
    attempts: int = 0
    duration_ms: int = 0

    def __post_init__(self) -> None:
        if not self.tool_call_id:
            raise ValueError("a tool result requires tool_call_id")
        if not self.tool_name:
            raise ValueError("a tool result requires tool_name")
        if self.ok and self.error is not None:
            raise ValueError("a successful tool result cannot carry an error")
        if not self.ok and self.error is None:
            raise ValueError("an unsuccessful tool result requires an error")
        try:
            # Round-tripping makes later message/checkpoint serialization safe
            # even when a tool handed us a mutable mapping or a tuple.
            normalized_data = json.loads(json.dumps(self.data, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("tool result data must be JSON serializable") from exc
        object.__setattr__(self, "data", normalized_data)

    @property
    def id(self) -> str:
        return self.tool_call_id

    @property
    def name(self) -> str:
        return self.tool_name

    @classmethod
    def success(
        cls,
        call: ToolCall,
        data: JSONValue | Any = None,
        *,
        attempts: int = 1,
        duration_ms: int = 0,
    ) -> ToolResult:
        return cls(call.id, call.name, True, data, None, attempts, duration_ms)

    @classmethod
    def failure(
        cls,
        call: ToolCall,
        *,
        kind: str,
        message: str,
        retryable: bool = False,
        attempts: int = 0,
        duration_ms: int = 0,
    ) -> ToolResult:
        return cls(
            call.id,
            call.name,
            False,
            None,
            ToolError(kind=kind, message=message, retryable=retryable),
            attempts,
            duration_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "ok": self.ok,
            "data": self.data,
            "error": None if self.error is None else self.error.to_dict(),
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolResult:
        error = value.get("error")
        return cls(
            tool_call_id=str(value["tool_call_id"]),
            tool_name=str(value["tool_name"]),
            ok=bool(value["ok"]),
            data=value.get("data"),
            error=ToolError.from_dict(error) if isinstance(error, Mapping) else None,
            attempts=int(value.get("attempts", 0)),
            duration_ms=int(value.get("duration_ms", 0)),
        )


class RunStatus(StrEnum):
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Event:
    """A small, serializable state/trace event without secret-bearing payloads."""

    kind: str
    step: int
    data: dict[str, JSONValue | Any] = field(default_factory=dict)
    at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Event:
        return cls(
            kind=str(value["kind"]),
            step=int(value["step"]),
            data=dict(value.get("data", {})),
            at=str(value.get("at", _utc_now())),
        )


@dataclass
class RunState:
    """The complete, JSON-serializable state needed by the control loop."""

    run_id: str
    status: RunStatus
    task: str
    messages: list[dict[str, Any]]
    step: int = 0
    completed_call_ids: set[str] = field(default_factory=set)
    pending_calls: list[ToolCall] = field(default_factory=list)
    pending_approval: ToolCall | None = None
    final_answer: str | None = None
    failure_reason: str | None = None
    events: list[Event] = field(default_factory=list)

    @classmethod
    def new(cls, task: str, *, run_id: str | None = None) -> RunState:
        return cls(
            run_id=run_id or uuid4().hex,
            status=RunStatus.RUNNING,
            task=task,
            messages=[{"role": "user", "content": task}],
        )

    def record(self, kind: str, **data: JSONValue | Any) -> None:
        self.events.append(Event(kind=kind, step=self.step, data=dict(data)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "task": self.task,
            "messages": self.messages,
            "step": self.step,
            "completed_call_ids": sorted(self.completed_call_ids),
            "pending_calls": [call.to_dict() for call in self.pending_calls],
            "pending_approval": None
            if self.pending_approval is None
            else self.pending_approval.to_dict(),
            "final_answer": self.final_answer,
            "failure_reason": self.failure_reason,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunState:
        pending_approval = value.get("pending_approval")
        return cls(
            run_id=str(value["run_id"]),
            status=RunStatus(value["status"]),
            task=str(value["task"]),
            messages=[dict(message) for message in value.get("messages", [])],
            step=int(value.get("step", 0)),
            completed_call_ids=set(value.get("completed_call_ids", [])),
            pending_calls=[ToolCall.from_dict(call) for call in value.get("pending_calls", [])],
            pending_approval=ToolCall.from_dict(pending_approval)
            if isinstance(pending_approval, Mapping)
            else None,
            final_answer=value.get("final_answer"),
            failure_reason=value.get("failure_reason"),
            events=[Event.from_dict(event) for event in value.get("events", [])],
        )
