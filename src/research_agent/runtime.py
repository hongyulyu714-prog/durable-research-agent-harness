"""Composition root for a runnable, durable local research agent."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from research_agent.cli import ApplicationService
from research_agent.llm import LLMAdapter, OpenAICompatibleLLM
from research_agent.loop import AgentLoop
from research_agent.models import RunState
from research_agent.scheduler import AsyncToolScheduler
from research_agent.state_store import ApprovalService, StateStore
from research_agent.tools import build_local_registry

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class RuntimeConfigurationError(RuntimeError):
    """The command cannot start because required provider configuration is absent."""


def build_application_service(
    *,
    runs_dir: str | Path,
    workspace: str | Path,
    llm: LLMAdapter | None = None,
    client: Any | None = None,
    environment: Mapping[str, str] | None = None,
    max_steps: int = 8,
) -> ApplicationService:
    """Wire the project-owned runtime without relying on dependency injection.

    Tests may provide a scripted ``llm`` or an OpenAI-compatible fake client;
    ordinary CLI execution builds the adapter from environment variables.
    """

    workspace_root = Path(workspace).expanduser().resolve()
    if not workspace_root.is_dir():
        raise RuntimeConfigurationError(
            f"--workspace must name an existing directory: {workspace_root}"
        )

    active_llm = llm or build_openai_llm(client=client, environment=environment)
    store = StateStore(runs_dir)
    registry = build_local_registry(workspace=workspace_root)
    scheduler = AsyncToolScheduler(registry)
    approvals = ApprovalService(store, registry)
    loop = AgentLoop(active_llm, registry, scheduler, max_steps=max_steps, approval_gate=approvals)
    approvals.continuation = loop.run

    async def start(task: str) -> RunState:
        return await loop.run(RunState.new(task))

    return ApplicationService(
        store,
        registry,
        start=start,
        continuation=loop.run,
    )


def build_openai_llm(
    *,
    client: Any | None = None,
    environment: Mapping[str, str] | None = None,
) -> OpenAICompatibleLLM:
    """Read OpenAI-compatible provider settings and construct the LLM adapter."""

    env = os.environ if environment is None else environment
    api_key = env.get("OPENAI_API_KEY", "").strip()
    model = env.get("OPENAI_MODEL", "").strip()
    base_url = env.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).strip()
    settings = (
        ("OPENAI_API_KEY", api_key),
        ("OPENAI_MODEL", model),
        ("OPENAI_BASE_URL", base_url),
    )
    missing = [name for name, value in settings if not value]
    if missing:
        rendered = ", ".join(missing)
        raise RuntimeConfigurationError(
            f"missing provider configuration: {rendered}. "
            "Copy .env.example, set the variables, and export/source them before running."
        )
    return OpenAICompatibleLLM(
        api_key=api_key,
        model=model,
        base_url=base_url,
        client=client,
    )
