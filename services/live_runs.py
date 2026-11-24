from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any
import itertools


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat()


@dataclass
class LiveRunRecord:
    run_id: str
    ticket_id: int
    source: str
    status: str = "running"
    stage: str = "start"
    started_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    completed_at: datetime | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


class LiveRunsRegistry:
    def __init__(self, retention_seconds: int = 60):
        self.retention_seconds = retention_seconds
        self._lock = Lock()
        self._counter = itertools.count(1)
        self._runs: dict[str, LiveRunRecord] = {}

    def start_run(self, *, ticket_id: int, source: str) -> str:
        with self._lock:
            now = _utc_now()
            run_id = f"run-{next(self._counter)}"
            record = LiveRunRecord(run_id=run_id, ticket_id=ticket_id, source=source)
            self._runs[run_id] = record
            record.events.append(self._build_event(record, event_type="run_started", message="Run started"))
            return run_id

    def emit(self, run_id: str, *, stage: str, message: str, detail: str | None = None) -> None:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return
            record.stage = stage
            record.updated_at = _utc_now()
            record.events.append(self._build_event(record, event_type="step", message=message, detail=detail))

    def complete(self, run_id: str, *, message: str, result: dict[str, Any] | None = None) -> None:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return
            now = _utc_now()
            record.status = "completed"
            record.stage = "done"
            record.updated_at = now
            record.completed_at = now
            record.events.append(self._build_event(record, event_type="run_completed", message=message, result=result))

    def fail(self, run_id: str, *, message: str, error: str) -> None:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return
            now = _utc_now()
            record.status = "failed"
            record.stage = "failed"
            record.updated_at = now
            record.completed_at = now
            record.events.append(self._build_event(record, event_type="run_failed", message=message, detail=error))

    def list_active_runs(self) -> list[dict[str, Any]]:
        self.cleanup_expired()
        with self._lock:
            runs = [
                {
                    "run_id": record.run_id,
                    "ticket_id": record.ticket_id,
                    "source": record.source,
                    "stage": record.stage,
                    "status": record.status,
                    "started_at": _iso(record.started_at),
                    "updated_at": _iso(record.updated_at),
                }
                for record in self._runs.values()
                if record.status == "running"
            ]
        return sorted(runs, key=lambda item: item["started_at"], reverse=True)

    def get_events(self, run_id: str, *, after_index: int = -1) -> list[dict[str, Any]]:
        self.cleanup_expired()
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return []
            return [dict(event) for event in record.events[after_index + 1:]]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        self.cleanup_expired()
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return None
            return {
                "run_id": record.run_id,
                "ticket_id": record.ticket_id,
                "source": record.source,
                "status": record.status,
                "stage": record.stage,
                "started_at": _iso(record.started_at),
                "updated_at": _iso(record.updated_at),
                "completed_at": _iso(record.completed_at) if record.completed_at else None,
            }

    def cleanup_expired(self, *, now: float | None = None) -> None:
        current_time = datetime.fromtimestamp(now, timezone.utc) if now is not None else _utc_now()
        cutoff = current_time - timedelta(seconds=self.retention_seconds)
        with self._lock:
            expired = [
                run_id
                for run_id, record in self._runs.items()
                if record.completed_at is not None and record.completed_at < cutoff
            ]
            for run_id in expired:
                self._runs.pop(run_id, None)

    def _build_event(
        self,
        record: LiveRunRecord,
        *,
        event_type: str,
        message: str,
        detail: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timestamp = _utc_now()
        event = {
            "type": event_type,
            "run_id": record.run_id,
            "ticket_id": record.ticket_id,
            "source": record.source,
            "stage": record.stage,
            "message": message,
            "timestamp": _iso(timestamp),
            "timestamp_epoch": timestamp.timestamp(),
        }
        if detail:
            event["detail"] = detail
        if result is not None:
            event["result"] = result
        return event


class ActivityReporter:
    def __init__(self, registry: LiveRunsRegistry, run_id: str):
        self.registry = registry
        self.run_id = run_id

    def step(self, stage: str, message: str, detail: str | None = None) -> None:
        self.registry.emit(self.run_id, stage=stage, message=message, detail=detail)

    def complete(self, message: str, result: dict[str, Any] | None = None) -> None:
        self.registry.complete(self.run_id, message=message, result=result)

    def fail(self, message: str, error: str) -> None:
        self.registry.fail(self.run_id, message=message, error=error)


_registry: LiveRunsRegistry | None = None


def get_live_runs_registry() -> LiveRunsRegistry:
    global _registry
    if _registry is None:
        _registry = LiveRunsRegistry()
    return _registry
