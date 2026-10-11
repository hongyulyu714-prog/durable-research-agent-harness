"""A real asynchronous web-fetching tool with network and resource bounds."""

from __future__ import annotations

import asyncio
import codecs
import ipaddress
import socket
from typing import Any
from urllib.parse import urlsplit

import httpx

DEFAULT_TIMEOUT_SECONDS = 8.0
DEFAULT_MAX_RESPONSE_CHARS = 20_000
DEFAULT_MAX_RESPONSE_BYTES = 1_000_000
MAX_REDIRECTS = 10

_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_METADATA_HOSTS = {
    "metadata",
    "metadata.google",
    "metadata.google.internal",
    "instance-data",
}


class UnsafeUrlError(ValueError):
    """A non-retryable error for URLs outside the web tool's network boundary."""

    kind = "unsafe_url"
    retryable = False


async def _validate_target(url: str) -> None:
    """Reject non-web, local, private, and known metadata-service targets.

    DNS results are checked as a defence against hostnames which resolve to a
    private address.  An unresolvable hostname is left to ``httpx`` so callers
    get its normal, retryable network failure instead of a policy error.
    """

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError("fetch_url only permits http and https URLs")
    if not parsed.hostname:
        raise UnsafeUrlError("fetch_url requires a hostname")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise UnsafeUrlError("fetch_url does not permit localhost targets")
    if hostname in _METADATA_HOSTS or hostname.endswith(".metadata.google.internal"):
        raise UnsafeUrlError("fetch_url does not permit metadata-service targets")

    try:
        addresses = {ipaddress.ip_address(hostname)}
    except ValueError:
        addresses = await _resolved_addresses(hostname, parsed.port)

    for address in addresses:
        if not address.is_global:
            raise UnsafeUrlError("fetch_url does not permit local or private network targets")


async def _resolved_addresses(
    hostname: str, port: int | None
) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Return DNS answers when available without making DNS failure a policy error."""

    try:
        records = await asyncio.get_running_loop().getaddrinfo(
            hostname,
            port or 443,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror:
        return set()
    return {ipaddress.ip_address(record[4][0]) for record in records}


def _decode_bounded_body(response: httpx.Response, body: bytes) -> str:
    """Decode the collected prefix without splitting a multibyte character."""

    encoding = response.encoding or "utf-8"
    decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
    return decoder.decode(body, final=True)


async def fetch_url(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch a safe URL with bounded streaming reads and return bounded text.

    The caller gets an explicit ``truncated`` flag instead of silently losing
    data.  HTTP error responses are raised so the scheduler can decide whether
    a 5xx error is eligible for a retry; 4xx responses remain non-retryable.
    Redirects are followed one at a time so every destination is checked before
    it is requested.  Passing a client is useful for a shared connection pool
    or ``MockTransport`` in offline tests.
    """

    if max_response_chars < 0:
        raise ValueError("max_response_chars must be non-negative")
    if max_response_bytes < 1:
        raise ValueError("max_response_bytes must be at least 1")

    owns_client = client is None
    active_client = client or httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(timeout_seconds),
    )
    try:
        current_url = url
        for redirect_count in range(MAX_REDIRECTS + 1):
            await _validate_target(current_url)
            request = active_client.build_request("GET", current_url, timeout=timeout_seconds)
            response = await active_client.send(request, stream=True, follow_redirects=False)

            if response.status_code not in _REDIRECT_STATUS_CODES:
                break
            try:
                location = response.headers.get("location")
                if location is None:
                    response.raise_for_status()
                    raise httpx.RemoteProtocolError("redirect response has no Location header")
                if redirect_count == MAX_REDIRECTS:
                    raise httpx.TooManyRedirects(
                        f"exceeded {MAX_REDIRECTS} redirects while requesting {url!r}"
                    )
                current_url = str(response.url.join(location))
            finally:
                await response.aclose()
        else:  # pragma: no cover - the explicit redirect bound exits above.
            raise AssertionError("redirect loop did not terminate")

        try:
            response.raise_for_status()
            body = bytearray()
            byte_limit_hit = False
            chunks = response.aiter_bytes(chunk_size=min(64 * 1024, max_response_bytes))
            async for chunk in chunks:
                remaining = max_response_bytes - len(body)
                if len(chunk) > remaining:
                    body.extend(chunk[:remaining])
                    byte_limit_hit = True
                    break
                body.extend(chunk)
                if len(body) == max_response_bytes:
                    # One more bounded chunk tells a full boundary from a
                    # truncated body without retaining it.
                    try:
                        byte_limit_hit = bool(await anext(chunks))
                    except StopAsyncIteration:
                        pass
                    break

            full_text = _decode_bounded_body(response, bytes(body))
            return {
                "url": str(response.url),
                "status_code": response.status_code,
                "text": full_text[:max_response_chars],
                "truncated": byte_limit_hit or len(full_text) > max_response_chars,
                "original_char_count": len(full_text),
            }
        finally:
            await response.aclose()
    finally:
        if owns_client:
            await active_client.aclose()
