"""Offline proof that the scheduler provides real async isolation."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from research_agent.models import ToolCall
from research_agent.registry import ToolHandler, ToolRegistry, ToolSpec
from research_agent.scheduler import AsyncToolScheduler
from research_agent.tools.web import fetch_url


def spec(
    name: str,
    *,
    timeout_seconds: float = 1,
    max_retries: int = 0,
    risk: str = "read",
    parallel_safe: bool = True,
    parameters_schema: dict[str, object] | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Test tool {name}",
        parameters_schema=parameters_schema or {"type": "object"},
        risk=risk,  # type: ignore[arg-type]
        parallel_safe=parallel_safe,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def make_registry(*entries: tuple[ToolSpec, ToolHandler]) -> ToolRegistry:
    registry = ToolRegistry()
    for spec, handler in entries:
        registry.register(spec, handler)
    return registry


@pytest.mark.asyncio
async def test_parallel_calls_take_roughly_the_longest_single_call() -> None:
    async def delayed(value: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return value

    registry = make_registry((spec("delayed"), delayed))
    scheduler = AsyncToolScheduler(registry, max_concurrency=3)
    calls = [
        ToolCall(f"c{number}", "delayed", {"value": str(number), "delay": 0.10})
        for number in range(3)
    ]

    started = time.perf_counter()
    results = await scheduler.execute(calls)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.20, "three 100ms calls should overlap, not total 300ms"
    assert [result.data for result in results] == ["0", "1", "2"]
    assert all(result.ok for result in results)


@pytest.mark.asyncio
async def test_timeout_and_exception_do_not_cancel_healthy_siblings() -> None:
    async def tool(mode: str) -> str:
        if mode == "slow":
            await asyncio.sleep(0.10)
        if mode == "explode":
            raise RuntimeError("boom")
        return "healthy"

    registry = make_registry((spec("tool", timeout_seconds=0.03), tool))
    results = await AsyncToolScheduler(registry).execute(
        [
            ToolCall("timeout", "tool", {"mode": "slow"}),
            ToolCall("error", "tool", {"mode": "explode"}),
            ToolCall("healthy", "tool", {"mode": "ok"}),
        ]
    )

    assert [result.tool_call_id for result in results] == ["timeout", "error", "healthy"]
    assert results[0].error is not None and results[0].error.kind == "timeout"
    assert results[1].error is not None and results[1].error.kind == "exception"
    assert results[2].ok and results[2].data == "healthy"


@pytest.mark.asyncio
async def test_semaphore_caps_peak_concurrency_and_retry_is_bounded() -> None:
    running = 0
    peak = 0
    attempts = 0

    async def measured(call: int) -> int:
        nonlocal running, peak, attempts
        attempts += 1
        if call == 0 and attempts == 1:
            raise httpx.ConnectError("temporary")
        running += 1
        peak = max(peak, running)
        try:
            await asyncio.sleep(0.02)
        finally:
            running -= 1
        return call

    registry = make_registry(
        (spec("measured", max_retries=1), measured)
    )
    results = await AsyncToolScheduler(
        registry, max_concurrency=2, retry_base_seconds=0
    ).execute([ToolCall(str(index), "measured", {"call": index}) for index in range(5)])

    assert peak <= 2
    assert all(result.ok for result in results)
    assert results[0].attempts == 2


@pytest.mark.asyncio
async def test_write_tools_are_not_auto_executed() -> None:
    called = False

    async def write() -> None:
        nonlocal called
        called = True

    registry = make_registry(
        (spec("write", risk="write", parallel_safe=False), write)
    )
    result = (await AsyncToolScheduler(registry).execute([ToolCall("w1", "write", {})]))[0]

    assert not called
    assert result.error is not None and result.error.kind == "approval_required"
    assert result.attempts == 0


@pytest.mark.asyncio
async def test_scheduler_invokes_validated_json_arguments_through_registry() -> None:
    async def lookup(query: str) -> str:
        return f"found {query}"

    registry = make_registry(
        (
            spec(
                "lookup",
                parameters_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            lookup,
        )
    )
    call = ToolCall(id="json-call", name="lookup", arguments='{"query":"Ada"}')

    assert registry.validate_call(call) is None
    result = (await AsyncToolScheduler(registry).execute([call]))[0]

    assert result.ok
    assert result.data == "found Ada"


@pytest.mark.asyncio
async def test_fetch_url_uses_mock_transport_and_reports_truncation() -> None:
    async def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="abcdef")

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        result = await fetch_url("https://example.test/page", max_response_chars=3, client=client)

    assert result == {
        "url": "https://example.test/page",
        "status_code": 200,
        "text": "abc",
        "truncated": True,
        "original_char_count": 6,
    }
