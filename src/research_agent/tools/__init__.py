"""Built-in, narrow-scope tools exposed through :class:`ToolRegistry`."""

from __future__ import annotations

from pathlib import Path

from research_agent.registry import ToolRegistry, ToolSpec
from research_agent.tools.files import read_file, write_report
from research_agent.tools.web import fetch_url


def build_local_registry(*, workspace: Path | str | None = None) -> ToolRegistry:
    """Create the registry containing the two local filesystem tools.

    ``workspace`` is captured in the handlers rather than exposed to the model,
    so a tool call cannot redirect its own security boundary.
    """

    registry = ToolRegistry()

    async def read_file_in_workspace(path: str):
        return await read_file(path, workspace=workspace)

    async def write_report_in_workspace(path: str, content: str):
        return await write_report(path, content, workspace=workspace)

    async def fetch_public_url(url: str):
        return await fetch_url(url)

    registry.register(
        ToolSpec(
            name="fetch_url",
            description="Fetch bounded text from a public HTTP or HTTPS URL.",
            parameters_schema={
                "type": "object",
                "properties": {"url": {"type": "string", "minLength": 1}},
                "required": ["url"],
                "additionalProperties": False,
            },
            risk="read",
            parallel_safe=True,
            timeout_seconds=10.0,
            max_retries=2,
        ),
        fetch_public_url,
    )
    registry.register(
        ToolSpec(
            name="read_file",
            description="Read a UTF-8 text file relative to the read-only data directory.",
            parameters_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1}},
                "required": ["path"],
                "additionalProperties": False,
            },
            risk="read",
            parallel_safe=True,
            timeout_seconds=5.0,
            max_retries=0,
        ),
        read_file_in_workspace,
    )
    registry.register(
        ToolSpec(
            name="write_report",
            description="Write a UTF-8 report relative to the reports directory after approval.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            risk="write",
            parallel_safe=False,
            timeout_seconds=10.0,
            max_retries=0,
        ),
        write_report_in_workspace,
    )
    return registry


__all__ = ["build_local_registry", "fetch_url", "read_file", "write_report"]
