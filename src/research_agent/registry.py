"""The model-visible tool catalogue and its untrusted-input boundary.

The registry deliberately contains no scheduling or approval decisions.  It
describes a callable to a model, validates the model's proposed arguments, and
offers the corresponding local handler to the project-owned scheduler.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from research_agent.models import ToolCall, ToolResult

Risk = Literal["read", "write"]
ToolHandler = Callable[..., Any | Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    """Static execution metadata controlled by the harness, not the model."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    risk: Risk
    parallel_safe: bool
    timeout_seconds: float
    max_retries: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if not self.description:
            raise ValueError(f"tool {self.name!r} needs a description")
        if self.risk not in {"read", "write"}:
            raise ValueError("risk must be 'read' or 'write'")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.parameters_schema.get("type") != "object":
            raise ValueError("parameters_schema must be a JSON Schema object")

    def openai_schema(self) -> dict[str, Any]:
        """Render the native function-tool object accepted by compatible APIs."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


@dataclass(frozen=True)
class RegisteredTool:
    """A ``ToolSpec`` paired with a local implementation."""

    spec: ToolSpec
    handler: ToolHandler


class ToolRegistry:
    """Register tools once and validate every model-originated call.

    Validation failures are returned as :class:`ToolResult` values instead of
    exceptions.  This lets the loop put the failure back into the tool-message
    protocol and gives the model an opportunity to repair its request.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.name in self._tools:
            raise ValueError(f"a tool named {spec.name!r} is already registered")
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def schemas(self) -> list[dict[str, Any]]:
        """Return JSON-serializable OpenAI-compatible tool definitions."""

        return [tool.spec.openai_schema() for tool in self._tools.values()]

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def get_spec(self, name: str) -> ToolSpec | None:
        tool = self.get(name)
        return None if tool is None else tool.spec

    def get_handler(self, name: str) -> ToolHandler | None:
        tool = self.get(name)
        return None if tool is None else tool.handler

    def arguments_for(self, call: ToolCall) -> dict[str, Any]:
        """Decode a model call's JSON object arguments without executing it."""

        arguments = call.arguments
        if isinstance(arguments, Mapping):
            return dict(arguments)
        if not isinstance(arguments, str):
            raise ValueError("arguments must be a JSON object")
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"arguments are not valid JSON: {exc.msg}") from exc
        if not isinstance(decoded, dict):
            raise ValueError("arguments must decode to a JSON object")
        return decoded

    def validate_call(self, call: ToolCall) -> ToolResult | None:
        """Return one model-visible error, or ``None`` for an executable call."""

        registered = self.get(call.name)
        if registered is None:
            return ToolResult.failure(
                call,
                kind="unknown_tool",
                message=f"tool {call.name!r} is not registered",
            )
        try:
            arguments = self.arguments_for(call)
            _validate_schema(arguments, registered.spec.parameters_schema)
        except ValueError as exc:
            return ToolResult.failure(call, kind="invalid_arguments", message=str(exc))
        return None

    def validate_calls(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        return [result for call in calls if (result := self.validate_call(call)) is not None]

    async def invoke(self, call: ToolCall) -> Any:
        """Invoke one already-validated tool, awaiting either handler style.

        The scheduler owns timeouts, retries, policy checks, and conversion to
        ``ToolResult``.  Keeping this method thin avoids accidentally moving
        execution policy into the registry.
        """

        registered = self.get(call.name)
        if registered is None:
            raise LookupError(f"tool {call.name!r} is not registered")
        arguments = self.arguments_for(call)
        result = registered.handler(**arguments)
        if inspect.isawaitable(result):
            return await result
        return result


def _validate_schema(value: Any, schema: Mapping[str, Any], *, field: str = "arguments") -> None:
    """Validate the small, explicit JSON-Schema subset used by project tools.

    API-facing schemas remain ordinary JSON Schema.  Local validation supports
    the object/string/integer/number/boolean/array constraints used here and is
    deliberately strict about unexpected object keys.
    """

    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                raise ValueError(f"{field}.{name} is required")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise ValueError(f"{field} contains unsupported fields: {', '.join(unknown)}")
        for name, item in value.items():
            child_schema = properties.get(name)
            if child_schema is not None:
                _validate_schema(item, child_schema, field=f"{field}.{name}")
        return

    if expected_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string")
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{field} must not be empty")
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{field} must be an integer")
    elif expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field} must be a number")
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be a boolean")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{field} must be an array")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, field=f"{field}[{index}]")

    allowed = schema.get("enum")
    if allowed is not None and value not in allowed:
        raise ValueError(f"{field} must be one of {allowed}")
