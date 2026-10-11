"""Offline proof that the scheduler provides real async isolation."""

from __future__ import annotations

import asyncio
import ipaddress
import time

import httpx
import pytest

from research_agent.models import ToolCall
from research_agent.registry import ToolHandler, ToolRegistry, ToolSpec
from research_agent.scheduler import AsyncToolScheduler
from research_agent.tools.web import UnsafeUrlError, fetch_url


@pytest.fixture(autouse=True)
def offline_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep web-tool tests deterministic and independent of host DNS settings."""

    async def public_dns(*_: object) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        return {ipaddress.ip_address("8.8.8.8")}

    monkeypatch.setattr("research_agent.tools.web._resolved_addresses", public_dns)


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/report.txt",
        "http://localhost:8000/",
        "https://127.0.0.1/",
        "https://[::1]/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.9/",
        "https://metadata.google.internal/computeMetadata/v1/",
    ],
)
async def test_fetch_url_rejects_unsafe_targets_before_request(url: str) -> None:
    requested = False

    async def response(_: httpx.Request) -> httpx.Response:
        nonlocal requested
        requested = True
        return httpx.Response(200, text="not reached")

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(UnsafeUrlError):
            await fetch_url(url, client=client)

    assert not requested


@pytest.mark.asyncio
async def test_fetch_url_checks_each_redirect_target_before_request() -> None:
    requested_urls: list[str] = []

    async def response(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(UnsafeUrlError):
            await fetch_url("https://example.test/start", client=client)

    assert requested_urls == ["https://example.test/start"]


@pytest.mark.asyncio
async def test_fetch_url_rejects_hostnames_resolving_to_private_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def private_dns(*_: object) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        return {ipaddress.ip_address("192.168.1.10")}

    monkeypatch.setattr("research_agent.tools.web._resolved_addresses", private_dns)
    transport = httpx.MockTransport(lambda _: httpx.Response(200))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(UnsafeUrlError):
            await fetch_url("https://private-name.test/", client=client)


@pytest.mark.asyncio
async def test_fetch_url_stops_streaming_after_its_byte_limit() -> None:
    class CountingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.chunks_read = 0

        async def __aiter__(self):
            for _ in range(20):
                self.chunks_read += 1
                yield b"abcd"

        async def aclose(self) -> None:
            pass

    stream = CountingStream()

    async def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        result = await fetch_url(
            "https://example.test/large",
            max_response_chars=8,
            max_response_bytes=8,
            client=client,
        )

    assert result == {
        "url": "https://example.test/large",
        "status_code": 200,
        "text": "abcdabcd",
        "truncated": True,
        "original_char_count": 8,
    }
    assert stream.chunks_read <= 4
    assert stream.chunks_read < 20


@pytest.mark.asyncio
async def test_scheduler_returns_unsafe_url_as_structured_non_retryable_error() -> None:
    async def response(_: httpx.Request) -> httpx.Response:
        raise AssertionError("unsafe URL must not reach the transport")

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        async def web_tool(url: str) -> dict[str, object]:
            return await fetch_url(url, client=client)

        registry = make_registry((spec("fetch_url", max_retries=2), web_tool))
        result = (
            await AsyncToolScheduler(registry).execute(
                [ToolCall("unsafe", "fetch_url", {"url": "http://127.0.0.1/"})]
            )
        )[0]

    assert result.error is not None
    assert result.error.kind == "unsafe_url"
    assert not result.error.retryable
    assert result.attempts == 1
