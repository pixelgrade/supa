import asyncio
from datetime import datetime, timedelta, timezone

import pytest

import main

from services.live_runs import LiveRunsRegistry
from services.startup_recovery import (
    run_periodic_recovery,
    run_startup_conversation_recovery,
    run_startup_note_recovery,
)


@pytest.mark.asyncio
async def test_main_starts_one_retention_worker_and_shutdown_cancels_it(
    monkeypatch, tmp_path
):
    selected_db = str(tmp_path / "selected.db")
    retention_calls = []
    retention_task = asyncio.create_task(asyncio.sleep(3600))
    other_tasks = [asyncio.create_task(asyncio.sleep(3600)) for _ in range(3)]
    task_iterator = iter(other_tasks)

    monkeypatch.setattr(main, "get_supa_db_path", lambda: selected_db)
    monkeypatch.setattr(main, "recover_agent_operations_on_startup", lambda: None)
    monkeypatch.setattr(main, "schedule_startup_recovery", lambda: next(task_iterator))
    monkeypatch.setattr(main, "schedule_periodic_recovery", lambda: next(task_iterator))
    monkeypatch.setattr(
        main,
        "schedule_autosend_worker",
        lambda *, helpscout: next(task_iterator),
    )

    monkeypatch.setattr(main.extensions, "start_workers", lambda phase, attach: None)

    def fake_schedule_retention(db_path, **kwargs):
        retention_calls.append((db_path, kwargs))
        return retention_task

    monkeypatch.setattr(
        main,
        "schedule_event_retention_worker",
        fake_schedule_retention,
        raising=False,
    )

    await main.startup_recovery()

    assert retention_calls == [(selected_db, {})]
    assert "event_retention_task" in main.RECOVERY_TASK_NAMES
    assert main.app.state.event_retention_task is retention_task

    await main.shutdown_recovery_tasks()
    assert retention_task.cancelled()
    assert main.app.state.event_retention_task is None


def _conversation(conversation_id: int, note_created_at: datetime, source_type: str = "web") -> dict:
    note_timestamp = note_created_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "id": conversation_id,
        "subject": f"Ticket {conversation_id}",
        "primaryCustomer": {"id": 99, "first": "Reader", "last": "Oak", "email": "reader@woodland.example"},
        "tags": [],
        "_embedded": {
            "threads": [
                {
                    "type": "note",
                    "body": "<p>Please use the documented workaround.</p>",
                    "createdAt": note_timestamp,
                    "source": {"type": source_type},
                },
                {
                    "type": "customer",
                    "body": "<p>Need help</p>",
                    "createdBy": {"type": "customer"},
                    "source": {"type": "email"},
                },
            ]
        },
    }


def _new_conversation(
    conversation_id: int,
    customer_created_at: datetime,
    *,
    status: str = "active",
    source_type: str = "email",
) -> dict:
    customer_timestamp = customer_created_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "id": conversation_id,
        "subject": f"Ticket {conversation_id}",
        "status": status,
        "primaryCustomer": {"id": 99, "first": "Reader", "last": "Oak", "email": "reader@woodland.example"},
        "tags": [],
        "_embedded": {
            "threads": [
                {
                    "type": "customer",
                    "body": "<p>Need help</p>",
                    "createdAt": customer_timestamp,
                    "createdBy": {"type": "customer"},
                    "source": {"type": source_type},
                },
            ]
        },
    }


class FakeHelpScout:
    def __init__(self, conversations: list[dict]):
        self.conversations = conversations

    async def get_recent_conversations(self, *, limit: int = 100, statuses: tuple[str, ...] = ("active", "pending")):
        return self.conversations[:limit]


class FakeProcessor:
    def __init__(self, conversations: list[dict]):
        self.helpscout = FakeHelpScout(conversations)
        self.calls: list[tuple[int, str, bool]] = []

    async def process_ticket(self, payload, trigger="new_conversation", activity=None):
        self.calls.append((payload["id"], trigger, activity is not None))
        return {
            "ticket_id": payload["id"],
            "status": "processed",
            "classification": "support",
            "delivery_mode": "draft",
            "response_posted": True,
            "confidence_status": "manual_review",
        }


@pytest.mark.parametrize("recover,builder", [
    (run_startup_note_recovery, _conversation),
    (run_startup_conversation_recovery, _new_conversation),
])
async def test_recovery_leaves_ignored_conversations_for_manual_review(monkeypatch, recover, builder):
    now = datetime(2026, 9, 13, 14, 0, tzinfo=timezone.utc)
    ignored = builder(900, now - timedelta(minutes=2))
    ignored["tags"] = [{"tag": "supa-ignore"}]
    ordinary = builder(901, now - timedelta(minutes=2))
    processor = FakeProcessor([ignored, ordinary])
    monkeypatch.setattr("services.startup_recovery.get_ticket_log_by_helpscout_id", lambda *a, **k: None)
    recovered = await recover(processor=processor, registry=LiveRunsRegistry(), now=now)
    assert recovered == [901]
    assert [row[0] for row in processor.calls] == [901]


@pytest.mark.asyncio
async def test_startup_recovery_processes_recent_human_note(monkeypatch):
    now = datetime(2026, 4, 4, 5, 30, tzinfo=timezone.utc)
    processor = FakeProcessor([_conversation(401, now - timedelta(minutes=5))])
    registry = LiveRunsRegistry(retention_seconds=60)

    monkeypatch.setattr("services.startup_recovery.get_ticket_log_by_helpscout_id", lambda *args, **kwargs: None)

    recovered = await run_startup_note_recovery(
        processor=processor,
        registry=registry,
        now=now,
        window_minutes=15,
    )

    assert recovered == [401]
    assert processor.calls == [(401, "note_trigger", True)]

    run = registry.get_run("run-1")
    assert run is not None
    assert run["status"] == "completed"
    assert run["source"] == "startup_recovery"

    events = registry.get_events("run-1", after_index=-1)
    assert [event["type"] for event in events] == ["run_started", "step", "run_completed"]
    assert events[1]["stage"] == "recovery"


@pytest.mark.asyncio
async def test_startup_recovery_skips_old_api_and_already_processed_notes(monkeypatch):
    now = datetime(2026, 4, 4, 5, 30, tzinfo=timezone.utc)
    recoverable = _conversation(401, now - timedelta(minutes=4))
    older_than_window = _conversation(402, now - timedelta(minutes=16))
    api_note = _conversation(403, now - timedelta(minutes=3), source_type="api")
    already_processed = _conversation(404, now - timedelta(minutes=2))
    processor = FakeProcessor([recoverable, older_than_window, api_note, already_processed])
    registry = LiveRunsRegistry(retention_seconds=60)

    def fake_ticket_log(_db_path=None, helpscout_id=""):
        if helpscout_id == "404":
            return {
                "helpscout_id": helpscout_id,
                "created_at": "2026-04-04 05:29:30",
                "status": "processed",
                "trigger": "new_conversation",
            }
        return None

    monkeypatch.setattr("services.startup_recovery.get_ticket_log_by_helpscout_id", fake_ticket_log)

    recovered = await run_startup_note_recovery(
        processor=processor,
        registry=registry,
        now=now,
        window_minutes=15,
    )

    assert recovered == [401]
    assert processor.calls == [(401, "note_trigger", True)]


@pytest.mark.asyncio
async def test_startup_recovery_processes_recent_mobile_note(monkeypatch):
    now = datetime(2026, 4, 6, 6, 0, tzinfo=timezone.utc)
    processor = FakeProcessor([_conversation(405, now - timedelta(minutes=5), source_type="mobile")])
    registry = LiveRunsRegistry(retention_seconds=60)

    monkeypatch.setattr("services.startup_recovery.get_ticket_log_by_helpscout_id", lambda *args, **kwargs: None)

    recovered = await run_startup_note_recovery(
        processor=processor,
        registry=registry,
        now=now,
        window_minutes=15,
    )

    assert recovered == [405]
    assert processor.calls == [(405, "note_trigger", True)]


@pytest.mark.asyncio
async def test_startup_recovery_processes_recent_unlogged_new_conversation(monkeypatch):
    now = datetime(2026, 4, 5, 15, 0, tzinfo=timezone.utc)
    processor = FakeProcessor([_new_conversation(406, now - timedelta(minutes=5))])
    registry = LiveRunsRegistry(retention_seconds=60)

    monkeypatch.setattr("services.startup_recovery.get_ticket_log_by_helpscout_id", lambda *args, **kwargs: None)

    recovered = await run_startup_conversation_recovery(
        processor=processor,
        registry=registry,
        now=now,
        window_minutes=15,
    )

    assert recovered == [406]
    assert processor.calls == [(406, "new_conversation", True)]

    run = registry.get_run("run-1")
    assert run is not None
    assert run["status"] == "completed"
    assert run["source"] == "startup_recovery"


@pytest.mark.asyncio
async def test_startup_recovery_skips_logged_or_old_new_conversations(monkeypatch):
    now = datetime(2026, 4, 5, 15, 0, tzinfo=timezone.utc)
    recoverable = _new_conversation(406, now - timedelta(minutes=5))
    already_processed = _new_conversation(407, now - timedelta(minutes=4))
    older_than_window = _new_conversation(408, now - timedelta(minutes=16))
    api_sourced = _new_conversation(409, now - timedelta(minutes=3), source_type="api")
    pending = _new_conversation(410, now - timedelta(minutes=2), status="pending")
    processor = FakeProcessor([recoverable, already_processed, older_than_window, api_sourced, pending])
    registry = LiveRunsRegistry(retention_seconds=60)

    def fake_ticket_log(_db_path=None, helpscout_id=""):
        if helpscout_id == "407":
            return {
                "helpscout_id": helpscout_id,
                "created_at": "2026-04-05 14:58:30",
                "status": "processed",
                "trigger": "new_conversation",
            }
        return None

    monkeypatch.setattr("services.startup_recovery.get_ticket_log_by_helpscout_id", fake_ticket_log)

    recovered = await run_startup_conversation_recovery(
        processor=processor,
        registry=registry,
        now=now,
        window_minutes=15,
    )

    assert recovered == [406, 409, 410]
    assert processor.calls == [
        (406, "new_conversation", True),
        (409, "new_conversation", True),
        (410, "new_conversation", True),
    ]


@pytest.mark.asyncio
async def test_periodic_recovery_catches_conversation_that_appears_after_startup(monkeypatch):
    now = datetime(2026, 4, 5, 15, 0, tzinfo=timezone.utc)
    processor = FakeProcessor([])
    registry = LiveRunsRegistry(retention_seconds=60)
    sleep_calls = 0

    monkeypatch.setattr("services.startup_recovery.get_ticket_log_by_helpscout_id", lambda *args, **kwargs: None)

    async def fake_sleep(_seconds: int) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        processor.helpscout.conversations = [
            _new_conversation(411, now - timedelta(minutes=3), source_type="beacon-v2")
        ]

    await run_periodic_recovery(
        processor=processor,
        registry=registry,
        now=now,
        interval_seconds=60,
        iterations=2,
        sleep_func=fake_sleep,
    )

    assert sleep_calls == 1
    assert processor.calls == [(411, "new_conversation", True)]


@pytest.mark.asyncio
async def test_periodic_recovery_still_runs_conversation_pass_when_note_pass_times_out(monkeypatch):
    now = datetime(2026, 4, 5, 15, 0, tzinfo=timezone.utc)
    processor = FakeProcessor([])
    registry = LiveRunsRegistry(retention_seconds=60)
    conversation_calls: list[tuple[datetime | None, int]] = []

    async def fake_note_recovery(**_kwargs):
        raise TimeoutError("helpscout timeout")

    async def fake_conversation_recovery(**kwargs):
        conversation_calls.append((kwargs.get("now"), kwargs.get("window_minutes")))
        return [412]

    monkeypatch.setattr("services.startup_recovery.run_startup_note_recovery", fake_note_recovery)
    monkeypatch.setattr("services.startup_recovery.run_startup_conversation_recovery", fake_conversation_recovery)

    await run_periodic_recovery(
        processor=processor,
        registry=registry,
        now=now,
        conversation_window_minutes=45,
        iterations=1,
    )

    assert conversation_calls == [(now, 45)]
