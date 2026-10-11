"""Concurrent, timeout-bounded, failure-isolating tool execution."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import httpx

from research_agent.models import ToolCall, ToolError, ToolResult
from research_agent.registry import ToolRegistry, ToolSpec


class AsyncToolScheduler:
    """Run independent read-only tools concurrently without coupling failures.

    A call is only eligible for the concurrent batch if its registered policy
    says both ``risk == 'read'`` and ``parallel_safe``.  Writes are represented
    as a structured policy error: the approval layer, not the scheduler, must
    execute them later.  This makes an accidental scheduler call unable to
    bypass human approval.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_concurrency: int = 3,
        retry_base_seconds: float = 0.1,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if retry_base_seconds < 0:
            raise ValueError("retry_base_seconds must be non-negative")
        self._registry = registry
        self._max_concurrency = max_concurrency
        self._retry_base_seconds = retry_base_seconds

    async def execute(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        """Execute one model turn and return results in the original order."""

        semaphore = asyncio.Semaphore(self._max_concurrency)
        results: list[ToolResult | None] = [None] * len(calls)
        parallel: list[tuple[int, ToolCall, ToolSpec]] = []
        sequential: list[tuple[int, ToolCall, ToolSpec]] = []

        for index, call in enumerate(calls):
            registered = self._registry.get(call.name)
            if registered is None:
                results[index] = self._error_result(
                    call, "unknown_tool", f"tool is not registered: {call.tool_name}"
                )
                continue

            spec = registered.spec
            if spec.risk != "read":
                results[index] = self._error_result(
                    call,
                    "approval_required",
                    f"{call.tool_name} is a write tool and cannot be auto-executed",
                )
            elif spec.parallel_safe:
                parallel.append((index, call, spec))
            else:
                sequential.append((index, call, spec))

        # Each worker catches expected errors itself.  Consequently gather does
        # not cancel healthy siblings when one tool times out or raises.
        if parallel:
            completed = await asyncio.gather(
                *(
                    self._run_call(call, spec, semaphore)
                    for _, call, spec in parallel
                )
            )
            for (index, _, _), result in zip(parallel, completed, strict=True):
                results[index] = result

        # Non-parallel read operations retain their safety constraint while
        # still receiving the same timeout, retry, and failure-isolation rules.
        for index, call, spec in sequential:
            results[index] = await self._run_call(call, spec, semaphore)

        return [result for result in results if result is not None]

    async def _run_call(
        self,
        call: ToolCall,
        spec: ToolSpec,
        semaphore: asyncio.Semaphore,
    ) -> ToolResult:
        started = time.perf_counter()
        attempts = 0
        while True:
            attempts += 1
            try:
                # The permit covers actual external work, and is released
                # before retry backoff so sleeping calls do not block I/O.
                async with semaphore:
                    async with asyncio.timeout(spec.timeout_seconds):
                        data = await self._registry.invoke(call)
                return ToolResult.success(
                    call, data, attempts=attempts, duration_ms=self._elapsed_ms(started)
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # converted into the tool protocol below
                error = self._classify_error(exc, call.tool_name, spec.timeout_seconds)
                should_retry = (
                    spec.risk == "read"
                    and error.retryable
                    and attempts <= spec.max_retries
                )
                if not should_retry:
                    return ToolResult.failure(
                        call,
                        kind=error.kind,
                        message=error.message,
                        retryable=error.retryable,
                        attempts=attempts,
                        duration_ms=self._elapsed_ms(started),
                    )
                await asyncio.sleep(self._retry_base_seconds * (2 ** (attempts - 1)))

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return round((time.perf_counter() - started) * 1_000)

    @staticmethod
    def _error_result(call: ToolCall, kind: str, message: str) -> ToolResult:
        return ToolResult.failure(call, kind=kind, message=message, attempts=0)

    @staticmethod
    def _classify_error(exc: BaseException, tool_name: str, timeout: float) -> ToolError:
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
            return ToolError(
                kind="timeout",
                message=f"{tool_name} exceeded {timeout} seconds",
                retryable=True,
            )
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return ToolError(
                kind="http_error",
                message=f"{tool_name} received HTTP {status}",
                retryable=status == 408 or status == 429 or 500 <= status <= 599,
            )
        if isinstance(exc, httpx.RequestError):
            return ToolError(kind="network_error", message=str(exc), retryable=True)
        kind = getattr(exc, "kind", "exception")
        retryable = bool(getattr(exc, "retryable", False))
        return ToolError(
            kind=kind,
            message=f"{type(exc).__name__}: {exc}",
            retryable=retryable,
        )
