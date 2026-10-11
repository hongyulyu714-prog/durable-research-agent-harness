"""Narrow, workspace-scoped file tools.

These functions only perform I/O.  Whether ``write_report`` is allowed is
decided by :mod:`research_agent.policy` before the scheduler invokes it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from research_agent.policy import resolve_under


def _workspace_dir(workspace: Path | str | None, name: str) -> Path:
    base = Path.cwd() if workspace is None else Path(workspace)
    return (base / name).resolve()


async def read_file(path: str, *, workspace: Path | str | None = None) -> dict[str, str]:
    """Read one UTF-8 text file whose path is relative to ``data/``."""

    target = resolve_under(_workspace_dir(workspace, "data"), path)
    content = await asyncio.to_thread(target.read_text, encoding="utf-8")
    return {"path": path, "content": content}


async def write_report(
    path: str, content: str, *, workspace: Path | str | None = None
) -> dict[str, str | int]:
    """Write one UTF-8 report whose path is relative to ``reports/``.

    Parent directories are deliberately not created: callers can only write
    into an already-provisioned report workspace, avoiding a second implicit
    filesystem capability.
    """

    target = resolve_under(_workspace_dir(workspace, "reports"), path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"report directory does not exist: {target.parent}")
    await asyncio.to_thread(target.write_text, content, encoding="utf-8")
    return {"path": path, "bytes_written": len(content.encode("utf-8"))}
