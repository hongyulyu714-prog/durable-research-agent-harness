"""Offline coverage for the runnable CLI composition root."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent.llm import LLMResponse, MockLLM, OpenAICompatibleLLM
from research_agent.models import RunStatus, ToolCall
from research_agent.runtime import RuntimeConfigurationError, build_application_service


def make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "data").mkdir(parents=True)
    (workspace / "reports").mkdir()
    (workspace / "data" / "source.txt").write_text("research source", encoding="utf-8")
    return workspace


def test_openai_compatible_adapter_preserves_function_call_contract() -> None:
    received: dict[str, object] = {}

    class FakeCompletions:
        async def create(self, **kwargs: object) -> dict[str, object]:
            received.update(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": "I will inspect the source.",
                            "tool_calls": [
                                {
                                    "id": "call_read",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"source.txt"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    llm = OpenAICompatibleLLM(api_key="test-key", model="test-model", client=client)
    response = asyncio.run(
        llm.complete([{"role": "user", "content": "Read"}], [{"type": "function"}])
    )

    assert received["model"] == "test-model"
    assert response.content == "I will inspect the source."
    assert response.tool_calls == (
        ToolCall("call_read", "read_file", '{"path":"source.txt"}'),
    )


def test_runtime_runs_pauses_and_resumes_without_provider_credentials(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    call = ToolCall(
        "call_write",
        "write_report",
        {"path": "brief.md", "content": "# Durable brief"},
    )
    app = build_application_service(
        runs_dir=tmp_path / "runs",
        workspace=workspace,
        llm=MockLLM(
            [
                LLMResponse.with_tool_calls([call]),
                LLMResponse.final("The brief is ready."),
            ]
        ),
    )

    paused = asyncio.run(app.run("Create a brief."))
    assert paused.status is RunStatus.WAITING_APPROVAL
    assert app.store.load(paused.run_id).pending_approval == call
    assert not (workspace / "reports" / "brief.md").exists()

    completed = asyncio.run(app.resume(paused.run_id, approve=True, reject_reason=None))
    assert completed.status is RunStatus.COMPLETED
    assert completed.final_answer == "The brief is ready."
    assert (workspace / "reports" / "brief.md").read_text(encoding="utf-8") == "# Durable brief"
    assert app.store.load(completed.run_id).status is RunStatus.COMPLETED


def test_runtime_reports_actionable_missing_provider_configuration(tmp_path: Path) -> None:
    with pytest.raises(RuntimeConfigurationError, match="OPENAI_API_KEY, OPENAI_MODEL"):
        build_application_service(
            runs_dir=tmp_path / "runs",
            workspace=make_workspace(tmp_path),
            environment={},
        )


def test_cli_run_uses_the_runtime_composition_root(tmp_path: Path, monkeypatch, capsys) -> None:
    """The public command no longer falls back to an unconfigured service."""

    import research_agent.runtime as runtime
    from research_agent.cli import main

    workspace = make_workspace(tmp_path)
    app = build_application_service(
        runs_dir=tmp_path / "runs",
        workspace=workspace,
        llm=MockLLM([LLMResponse.final("Done.")]),
    )
    received: dict[str, object] = {}

    def fake_builder(**kwargs: object):
        received.update(kwargs)
        return app

    monkeypatch.setattr(runtime, "build_application_service", fake_builder)
    main(
        [
            "--runs-dir",
            str(tmp_path / "runs"),
            "run",
            "Complete the task.",
            "--workspace",
            str(workspace),
        ]
    )

    assert received["workspace"] == str(workspace)
    assert received["runs_dir"] == Path(tmp_path / "runs")
    assert capsys.readouterr().out.startswith("completed:")
