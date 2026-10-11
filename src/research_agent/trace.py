"""Append-only, redacted JSONL events and concise run summaries."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_agent.models import RunState

_SENSITIVE_NAME_PARTS = ("api_key", "apikey", "authorization", "password", "secret", "token")
_CONTENT_NAME_PARTS = ("content", "text", "body", "data", "arguments")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def redact(value: Any, *, key: str | None = None) -> Any:
    """Retain event shape while excluding keys and bulky untrusted content."""

    normalized = (key or "").lower()
    if any(part in normalized for part in _SENSITIVE_NAME_PARTS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(item_key): redact(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item, key=key) for item in value]
    if isinstance(value, str) and any(part in normalized for part in _CONTENT_NAME_PARTS):
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return {"sha256_16": digest, "char_count": len(value)}
    return value


class TraceRecorder:
    """Append redacted trace records at ``runs/<run_id>/events.jsonl``."""

    def __init__(self, root: str | Path, run_id: str) -> None:
        self.run_id = run_id
        self.run_directory = Path(root) / run_id
        self.path = self.run_directory / "events.jsonl"

    def record(
        self, event_type: str, *, state: RunState | None = None, **data: Any
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "ts": _timestamp(),
            "run_id": self.run_id,
            "type": event_type,
        }
        if state is not None:
            event.update({"step": state.step, "status": state.status.value})
        event.update(redact(data))
        self.run_directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"corrupt JSONL trace at line {line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"trace line {line_number} is not an object")
                records.append(record)
        return records

    def summary(self, state: RunState) -> dict[str, Any]:
        records = self.events()
        types = Counter(str(record.get("type", "unknown")) for record in records)
        failures = sum(
            1 for record in records if record.get("type") in {"tool_error", "tool_failed"}
        )
        decisions = [
            record["type"]
            for record in records
            if record.get("type") in {"approval_granted", "approval_rejected"}
        ]
        return {
            "run_id": state.run_id,
            "status": state.status.value,
            "step": state.step,
            "pending_approval": None
            if state.pending_approval is None
            else state.pending_approval.tool_name,
            "completed_call_count": len(state.completed_call_ids),
            "event_count": len(records),
            "event_types": dict(sorted(types.items())),
            "tool_failure_count": failures,
            "approval_decisions": decisions,
        }

    def write_summary(self, state: RunState) -> Path:
        target = self.run_directory / "summary.json"
        self.run_directory.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.summary(state), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target
