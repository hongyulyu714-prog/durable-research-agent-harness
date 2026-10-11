"""Small model adapters used by the project-owned control loop."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import AsyncOpenAI

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


class OpenAICompatibleLLM:
    """Minimal adapter around the Chat Completions API.

    The SDK is used only to make one provider request.  Converting its reply
    into the project's serializable contracts happens here, while the agent
    loop and all tool-execution decisions remain project-owned.

    ``client`` is injectable so the composition root can be exercised without
    credentials or network access.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY must not be empty")
        if not model.strip():
            raise ValueError("OPENAI_MODEL must not be empty")
        self.model = model
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            tools=list(tools),
        )
        choices = _response_field(response, "choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise RuntimeError("OpenAI-compatible API returned no completion choices")
        message = _response_field(choices[0], "message")
        if message is None:
            raise RuntimeError("OpenAI-compatible API returned a choice without a message")

        content = _response_field(message, "content")
        if content is not None and not isinstance(content, str):
            raise RuntimeError("OpenAI-compatible API returned non-text assistant content")
        raw_calls = _response_field(message, "tool_calls") or ()
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
            raise RuntimeError("OpenAI-compatible API returned invalid tool_calls")

        calls: list[ToolCall] = []
        for raw_call in raw_calls:
            function = _response_field(raw_call, "function")
            call_id = _response_field(raw_call, "id")
            name = _response_field(function, "name") if function is not None else None
            arguments = _response_field(function, "arguments") if function is not None else None
            if not isinstance(call_id, str) or not call_id:
                raise RuntimeError("OpenAI-compatible API returned a tool call without an id")
            if not isinstance(name, str) or not name:
                raise RuntimeError(
                    "OpenAI-compatible API returned a tool call without a function name"
                )
            if not isinstance(arguments, (str, Mapping)):
                raise RuntimeError(
                    "OpenAI-compatible API returned a tool call without JSON function arguments"
                )
            calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
        return LLMResponse(content=content, tool_calls=tuple(calls))


def _response_field(value: Any, name: str) -> Any:
    """Read a field from either SDK objects or dict-shaped compatible clients."""

    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


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
