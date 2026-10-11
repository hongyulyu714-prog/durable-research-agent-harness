"""Command-line entrypoint.

The implementation will expose `run`, `resume`, and `inspect` without putting
workflow logic in the CLI layer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from research_agent import __version__
from research_agent.models import RunState
from research_agent.registry import ToolRegistry
from research_agent.state_store import ApprovalService, StateStore
from research_agent.tools import build_local_registry
from research_agent.trace import TraceRecorder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-agent")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--runs-dir", default="runs", help="directory containing run checkpoints")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="start a research task")
    run_parser.add_argument("task", help="natural-language task")
    run_parser.add_argument(
        "--workspace",
        required=True,
        help="directory that contains the data/ and reports/ tool boundaries",
    )

    resume_parser = subparsers.add_parser("resume", help="resume a paused run")
    resume_parser.add_argument("run_id")
    decision = resume_parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", metavar="REASON")
    resume_parser.add_argument(
        "--workspace",
        required=True,
        help="the same workspace directory used when the run was started",
    )

    inspect_parser = subparsers.add_parser("inspect", help="show a run summary")
    inspect_parser.add_argument("run_id")
    return parser


class ApplicationService:
    """Thin CLI-facing application service; orchestration remains outside argparse."""

    def __init__(
        self,
        store: StateStore,
        registry: ToolRegistry | None = None,
        *,
        start: Callable[[str], Awaitable[RunState] | RunState] | None = None,
        continuation: Callable[[RunState], Awaitable[RunState] | RunState] | None = None,
    ) -> None:
        self.store = store
        self.registry = registry or build_local_registry()
        self.start = start
        self.continuation = continuation

    async def run(self, task: str) -> RunState:
        if self.start is None:
            raise RuntimeError(
                "no run service is configured; inject an AgentLoop-backed start service"
            )
        state = self.start(task)
        state = await state if asyncio.iscoroutine(state) else state
        self.store.save(state)
        TraceRecorder(self.store.root, state.run_id).write_summary(state)
        return state

    async def resume(self, run_id: str, *, approve: bool, reject_reason: str | None) -> RunState:
        return await ApprovalService(
            self.store, self.registry, continuation=self.continuation
        ).resume(run_id, approve=approve, reject_reason=reject_reason)

    def inspect(self, run_id: str) -> dict[str, Any]:
        state = self.store.load(run_id)
        return TraceRecorder(self.store.root, run_id).summary(state)


def main(argv: list[str] | None = None, *, service: ApplicationService | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if service is not None:
            app = service
        elif args.command == "inspect":
            # Inspection is entirely local and must remain available even if
            # provider credentials have expired or are not configured.
            app = ApplicationService(StateStore(Path(args.runs_dir)))
        else:
            # Imported lazily to keep this CLI module usable as the lightweight
            # application-service boundary in offline unit tests.
            from research_agent.runtime import build_application_service

            app = build_application_service(
                runs_dir=Path(args.runs_dir), workspace=args.workspace
            )
        if args.command == "run":
            state = asyncio.run(app.run(args.task))
            print(f"{state.status.value}: {state.run_id}")
        elif args.command == "resume":
            state = asyncio.run(
                app.resume(args.run_id, approve=args.approve, reject_reason=args.reject)
            )
            print(f"{state.status.value}: {state.run_id}")
        else:
            print(
                json.dumps(app.inspect(args.run_id), ensure_ascii=False, indent=2, sort_keys=True)
            )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
