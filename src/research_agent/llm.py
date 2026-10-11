"""Small model adapters used by the project-owned control loop."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from research_agent.models import ToolCall


@dataclass(frozen=True)
class LLMResponse:
    """A provider-neutral model response consumed by :class:`AgentLoop`."""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def is_final(self) -> bool:
        """A final answer has text and makes no further tool request."""

        return self.content is not None and not self.tool_calls

    @classmethod
    def final(cls, content: str) -> LLMResponse:
        return cls(content=content)

    @classmethod
    def with_tool_calls(
        cls, tool_calls: Sequence[ToolCall], content: str | None = None
    ) -> LLMResponse:
        return cls(content=content, tool_calls=tuple(tool_calls))

    def assistant_message(self) -> dict[str, Any]:
        """Render the response in the OpenAI-compatible messages protocol."""

        if not self.tool_calls:
            return {"role": "assistant", "content": self.content}
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False)
                        if isinstance(call.arguments, Mapping)
                        else call.arguments,
                    },
                }
                for call in self.tool_calls
            ],
        }


class LLMAdapter(Protocol):
    """The only model capability the control loop needs."""

    async def complete(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> LLMResponse: ...


# Kept as a readable synonym for callers that prefer the provider-neutral term.
ModelResponse = LLMResponse


def _coerce_response(response: LLMResponse | Mapping[str, Any]) -> LLMResponse:
    if isinstance(response, LLMResponse):
        return response
    calls = tuple(ToolCall.from_dict(call) for call in response.get("tool_calls", []))
    return LLMResponse(content=response.get("content"), tool_calls=calls)


@dataclass
class MockLLM:
    """A deterministic offline model adapter driven by a scripted response list."""

    responses: Sequence[LLMResponse | Mapping[str, Any]]
    requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = field(
        default_factory=list, init=False
    )
    _next_response: int = field(default=0, init=False)

    @property
    def calls(self) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
        """Alias retained for concise assertions in tests."""

        return self.requests

    async def complete(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> LLMResponse:
        self.requests.append((copy.deepcopy(list(messages)), copy.deepcopy(list(tools))))
        if self._next_response >= len(self.responses):
            raise RuntimeError("MockLLM script was exhausted before the run completed")
        response = _coerce_response(self.responses[self._next_response])
        self._next_response += 1
        return response
