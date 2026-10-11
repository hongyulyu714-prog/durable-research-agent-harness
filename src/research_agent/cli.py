"""Command-line entrypoint.

The implementation will expose `run`, `resume`, and `inspect` without putting
workflow logic in the CLI layer.
"""

from __future__ import annotations

import argparse

from research_agent import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-agent")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="start a research task")
    run_parser.add_argument("task", help="natural-language task")

    resume_parser = subparsers.add_parser("resume", help="resume a paused run")
    resume_parser.add_argument("run_id")
    decision = resume_parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", metavar="REASON")

    inspect_parser = subparsers.add_parser("inspect", help="show a run summary")
    inspect_parser.add_argument("run_id")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    parser.error(f"'{args.command}' is scaffolded but not implemented yet")
