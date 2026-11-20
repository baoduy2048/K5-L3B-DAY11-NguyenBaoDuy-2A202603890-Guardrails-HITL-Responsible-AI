"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, tuple[float, int]] = {}
        self._request_ids: set[str] = set()

    def record_input(
        self,
        *,
        user_id: str,
        text: str,
        request_id: str | None = None,
    ) -> str:
        """Store an input and return its correlation ID."""
        key = request_id or f"{user_id}:{uuid4().hex}"
        if key in self._request_ids:
            raise ValueError(f"request_id has already been used: {key}")

        started_at = utc_now_iso()
        self.logs.append(
            {
                "request_id": key,
                "user_id": user_id,
                "input": text,
                "timestamp": started_at,
                "started_at": started_at,
            }
        )
        self._open[key] = (perf_counter(), len(self.logs) - 1)
        self._request_ids.add(key)
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Store the response, decision, and elapsed time for an input."""
        key = request_id
        if key is None:
            for candidate, (_, index) in reversed(tuple(self._open.items())):
                if self.logs[index]["user_id"] == user_id:
                    key = candidate
                    break

        pending = self._open.get(key) if key is not None else None
        if pending is not None:
            _, index = pending
            if self.logs[index]["user_id"] != user_id:
                raise ValueError("request_id does not belong to user_id")
            self._open.pop(key)

        completed_at = utc_now_iso()
        if pending is None:
            self.logs.append(
                {
                    "request_id": key,
                    "user_id": user_id,
                    "started_at": None,
                    "timestamp": completed_at,
                }
            )
            entry = self.logs[-1]
            latency_ms = None
        else:
            started_at, index = pending
            entry = self.logs[index]
            latency_ms = round((perf_counter() - started_at) * 1000, 3)

        entry.update(
            {
                "output": text,
                "blocked": blocked,
                "layer": layer,
                "latency_ms": latency_ms,
                "completed_at": completed_at,
            }
        )

    def export_json(self, filepath: str | None = None) -> str:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
