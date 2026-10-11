"""A real asynchronous, bounded web-fetching tool."""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_TIMEOUT_SECONDS = 8.0
DEFAULT_MAX_RESPONSE_CHARS = 20_000


async def fetch_url(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch a URL with ``httpx.AsyncClient`` and return bounded text.

    The caller gets an explicit ``truncated`` flag instead of silently losing
    data.  HTTP error responses are raised so the scheduler can decide whether
    a 5xx error is eligible for a retry; 4xx responses remain non-retryable.
    Passing a client is useful for a shared connection pool or ``MockTransport``
    in offline tests.
    """

    if max_response_chars < 0:
        raise ValueError("max_response_chars must be non-negative")

    owns_client = client is None
    active_client = client or httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(timeout_seconds),
    )
    try:
        response = await active_client.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        text = response.text
        truncated = len(text) > max_response_chars
        return {
            "url": str(response.url),
            "status_code": response.status_code,
            "text": text[:max_response_chars],
            "truncated": truncated,
            "original_char_count": len(text),
        }
    finally:
        if owns_client:
            await active_client.aclose()
