from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_agent.models import ToolCall
from research_agent.policy import PolicyAction, PolicyEngine, WorkspacePathError
from research_agent.tools import build_local_registry
from research_agent.tools.files import read_file, write_report


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "data" / "source.txt").write_text("trusted local research", encoding="utf-8")
    return tmp_path


@pytest.mark.asyncio
async def test_local_tools_read_and_write_only_in_their_own_directories(workspace: Path) -> None:
    assert await read_file("source.txt", workspace=workspace) == {
        "path": "source.txt",
        "content": "trusted local research",
    }

    result = await write_report("brief.md", "# Brief", workspace=workspace)
    assert result["bytes_written"] == len(b"# Brief")
    assert (workspace / "reports" / "brief.md").read_text(encoding="utf-8") == "# Brief"

    with pytest.raises(FileNotFoundError):
        await read_file("source.txt", workspace=workspace / "reports")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["../outside.txt", "/tmp/outside.txt"])
async def test_file_tools_reject_traversal_and_absolute_paths(workspace: Path, path: str) -> None:
    with pytest.raises(WorkspacePathError):
        await read_file(path, workspace=workspace)
    with pytest.raises(WorkspacePathError):
        await write_report(path, "nope", workspace=workspace)


@pytest.mark.asyncio
async def test_file_tools_reject_symlink_escape(workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    (workspace / "data" / "escape.txt").symlink_to(outside)

    with pytest.raises(WorkspacePathError):
        await read_file("escape.txt", workspace=workspace)


def test_schemas_are_openai_compatible_and_validation_is_model_visible(workspace: Path) -> None:
    registry = build_local_registry(workspace=workspace)
    schema = {item["function"]["name"]: item for item in registry.schemas()}
    assert schema["read_file"]["type"] == "function"
    assert schema["read_file"]["function"]["parameters"]["type"] == "object"

    unknown = registry.validate_call(ToolCall(id="unknown", name="shell", arguments={}))
    assert unknown is not None
    assert unknown.error is not None and unknown.error.kind == "unknown_tool"

    bad = registry.validate_call(ToolCall(id="bad", name="read_file", arguments={"other": 1}))
    assert bad is not None
    assert bad.error is not None and bad.error.kind == "invalid_arguments"

    valid_json = ToolCall(id="ok", name="read_file", arguments=json.dumps({"path": "source.txt"}))
    assert registry.validate_call(valid_json) is None
    assert registry.arguments_for(valid_json) == {"path": "source.txt"}


def test_write_requires_approval_and_model_text_cannot_change_policy(workspace: Path) -> None:
    registry = build_local_registry(workspace=workspace)
    policy = PolicyEngine()
    assert policy.decide(registry.get_spec("read_file")).action is PolicyAction.ALLOW
    assert policy.decide(registry.get_spec("write_report")).action is PolicyAction.REQUIRE_APPROVAL
    approved_write = policy.decide(registry.get_spec("write_report"), approved=True)
    assert approved_write.action is PolicyAction.ALLOW
    assert policy.decide(None).action is PolicyAction.DENY
