"""Narrow, workspace-scoped file tools.

These functions only perform I/O.  Whether ``write_report`` is allowed is
decided by :mod:`research_agent.policy` before the scheduler invokes it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from research_agent.policy import WorkspacePathError, resolve_under


def _workspace_dir(workspace: Path | str | None, name: str) -> Path:
    """Return one logical tool root after proving it stays in the workspace.

    The logical root must be checked *before* it becomes the trusted root for
    ``resolve_under``.  Otherwise a ``reports`` or ``data`` symlink to an
    external directory would make that external directory appear safe.
    """

    workspace_root = (Path.cwd() if workspace is None else Path(workspace)).resolve()
    logical_root = workspace_root / name
    try:
        logical_root.resolve().relative_to(workspace_root)
    except ValueError as exc:
        raise WorkspacePathError(
            f"the {name} directory must not resolve outside the workspace"
        ) from exc
    return logical_root


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
