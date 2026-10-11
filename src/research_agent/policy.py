"""Harness-owned risk decisions and secure workspace path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class _ToolMetadata(Protocol):
    name: str
    risk: str


class PolicyAction(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require-approval"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyDecision:
    """A decision made solely from trusted harness metadata."""

    action: PolicyAction
    reason: str

    @property
    def allowed(self) -> bool:
        return self.action is PolicyAction.ALLOW

    @property
    def requires_approval(self) -> bool:
        return self.action is PolicyAction.REQUIRE_APPROVAL

    @property
    def denied(self) -> bool:
        return self.action is PolicyAction.DENY


class PolicyEngine:
    """Classify tool calls before execution.

    A model's prose is intentionally not an input to this API.  Only the
    registry's trusted risk metadata determines whether a call can run.
    """

    def decide(self, tool: _ToolMetadata | None, *, approved: bool = False) -> PolicyDecision:
        if tool is None:
            return PolicyDecision(PolicyAction.DENY, "unknown tools are never executable")
        if tool.risk == "read":
            return PolicyDecision(PolicyAction.ALLOW, "read-only tool")
        if tool.risk == "write":
            if approved:
                return PolicyDecision(PolicyAction.ALLOW, "write explicitly approved by harness")
            return PolicyDecision(
                PolicyAction.REQUIRE_APPROVAL,
                "write operations require human approval",
            )
        return PolicyDecision(PolicyAction.DENY, f"unsupported risk class: {tool.risk!r}")

    classify = decide


class WorkspacePathError(ValueError):
    """A requested file path lies outside the tool's assigned directory."""


def resolve_under(root: Path, requested_path: str | Path) -> Path:
    """Safely resolve a *relative* path below ``root``.

    Rejecting absolute and ``..`` inputs explicitly makes the tool contract
    clear.  ``resolve`` followed by ``relative_to`` then catches a symlink in
    the allowed directory whose target escapes the workspace.
    """

    candidate = Path(requested_path)
    if candidate.is_absolute():
        raise WorkspacePathError("absolute paths are not allowed")
    if ".." in candidate.parts:
        raise WorkspacePathError("path traversal ('..') is not allowed")
    root_resolved = root.resolve()
    resolved = (root_resolved / candidate).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise WorkspacePathError("path escapes the allowed workspace directory") from exc
    return resolved
