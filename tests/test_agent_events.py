import asyncio
import json
import sqlite3
from base64 import urlsafe_b64decode
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest

from db.database import init_db
from services import agent_events as agent_events_module
from services.agent_events import (
    AgentEventAckError,
    AgentEventCursorExpiredError,
    AgentEventStartError,
    AgentEventStorageError,
    acknowledge_event,
    append_event,
    append_supa_event,
    lease_next_event,
    normalize_webhook_event,
    prune_events,
    reset_event_consumer,
    run_event_retention_worker,
    schedule_event_retention_worker,
)


UTC = timezone.utc
RECEIVED_AT = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def event_db(tmp_path):
    db_path = str(tmp_path / "agent-events.db")
    init_db(db_path)
    return db_path


def _webhook_payload(*, thread):
    return {
        "id": 602,
        "number": 1442,
        "subject": "Private customer question",
        "status": "active",
        "mailboxId": 22,
        "createdAt": "2026-09-24T10:00:00Z",
        "updatedAt": "2026-09-24T11:59:59Z",
        "tags": [{"tag": "support"}],
        "_embedded": {"threads": [thread]},
    }


class _SequenceTokenBytes:
    def __init__(self):
        self.calls = 0

    def __call__(self, size):
        assert size == 32
        self.calls += 1
        return bytes([self.calls]) * size


def _append_feed_events(event_db, count, *, start_at=RECEIVED_AT):
    return [
        append_supa_event(
            event_db,
            type="supa.run.completed",
            run_id=f"run-{index}",
            dedupe_key=f"run:{index}:completed",
            summary={"status": "succeeded"},
            clock=lambda index=index: start_at + timedelta(seconds=index),
        )
        for index in range(1, count + 1)
    ]


@pytest.mark.parametrize(
    (
        "raw_event_type",
        "thread_type",
        "actor_type",
        "expected_type",
    ),
    (
        (
            "convo.created",
            "customer",
            "customer",
            "helpscout.conversation.created",
        ),
        (
            "convo.customer.reply.created",
            "customer",
            "customer",
            "helpscout.customer.replied",
        ),
        (
            "convo.note.created",
            "note",
            "user",
            "helpscout.note.created",
        ),
        (
            "convo.agent.reply.created",
            "message",
            "user",
            "helpscout.agent.replied",
        ),
        (
            "convo.agent.reply.created",
            "message",
            "system_user",
            "helpscout.agent.replied",
        ),
    ),
)
def test_normalize_webhook_event_maps_names_ids_actors_and_timestamps(
    raw_event_type,
    thread_type,
    actor_type,
    expected_type,
):
    payload = _webhook_payload(
        thread={
            "id": 991,
            "type": thread_type,
            "state": "published",
            "body": "This body must not be journaled.",
            "createdAt": "2026-09-24T11:58:00-04:00",
            "createdBy": {"id": 71, "type": actor_type, "email": "private@example.com"},
        }
    )

    event = normalize_webhook_event(
        raw_event_type,
        payload,
        clock=lambda: RECEIVED_AT,
    )

    assert event.type == expected_type
    assert event.conversation_id == "602"
    assert event.thread_id == "991"
    assert event.run_id is None
    assert event.source == "helpscout"
    assert event.actor_type == actor_type
    assert event.occurred_at == datetime(2026, 9, 24, 15, 58, tzinfo=UTC)
    assert event.received_at == RECEIVED_AT
    assert event.dedupe_key.startswith("helpscout:")
    assert len(event.dedupe_key) == len("helpscout:") + 64


def test_normalize_webhook_event_uses_canonical_payload_hash_fallback(event_db):
    first_payload = {
        "id": 602,
        "status": "pending",
        "tags": [{"tag": "support"}, {"tag": "follow-up"}],
    }
    same_payload_different_key_order = {
        "tags": [{"tag": "support"}, {"tag": "follow-up"}],
        "status": "pending",
        "id": 602,
    }

    first = normalize_webhook_event(
        "convo.status",
        first_payload,
        clock=lambda: RECEIVED_AT,
    )
    retry = normalize_webhook_event(
        "convo.status",
        same_payload_different_key_order,
        clock=lambda: RECEIVED_AT + timedelta(minutes=5),
    )

    assert first.type == "helpscout.conversation.status.changed"
    assert first.thread_id is None
    assert first.dedupe_key == retry.dedupe_key
    assert first.occurred_at == RECEIVED_AT
    assert retry.occurred_at == RECEIVED_AT + timedelta(minutes=5)
    assert append_event(event_db, first) == append_event(event_db, retry)
    with sqlite3.connect(event_db) as db:
        assert db.execute("SELECT COUNT(*) FROM agent_events").fetchone()[0] == 1


def test_append_event_deduplicates_webhook_retries_without_replacing_original(event_db):
    payload = _webhook_payload(
        thread={
            "id": 991,
            "type": "customer",
            "state": "published",
            "createdAt": "2026-09-24T11:58:00Z",
            "createdBy": {"type": "customer"},
            "body": "Please help",
        }
    )
    original = normalize_webhook_event(
        "convo.customer.reply.created",
        payload,
        clock=lambda: RECEIVED_AT,
    )
    retry = normalize_webhook_event(
        "convo.customer.reply.created",
        payload,
        clock=lambda: RECEIVED_AT + timedelta(minutes=1),
    )

    first_row = append_event(event_db, original)
    retry_row = append_event(event_db, retry)

    assert first_row == retry_row
    assert first_row["event_id"] == 1
    assert first_row["received_at"] == "2026-09-24T12:00:00.000000Z"
    with sqlite3.connect(event_db) as db:
        assert db.execute("SELECT COUNT(*) FROM agent_events").fetchone()[0] == 1


def test_distinct_help_scout_threads_produce_distinct_events(event_db):
    first_thread = {
        "id": 991,
        "type": "customer",
        "state": "published",
        "createdAt": "2026-09-24T11:58:00Z",
        "createdBy": {"type": "customer"},
    }
    second_thread = {
        "id": 992,
        "type": "customer",
        "state": "published",
        "createdAt": "2026-09-24T12:01:00Z",
        "createdBy": {"type": "customer"},
    }
    first_payload = _webhook_payload(thread=first_thread)
    second_payload = _webhook_payload(thread=first_thread)
    second_payload["_embedded"]["threads"].append(second_thread)

    first = append_event(
        event_db,
        normalize_webhook_event(
            "convo.customer.reply.created",
            first_payload,
            clock=lambda: RECEIVED_AT,
        ),
    )
    second = append_event(
        event_db,
        normalize_webhook_event(
            "convo.customer.reply.created",
            second_payload,
            clock=lambda: RECEIVED_AT + timedelta(minutes=2),
        ),
    )

    assert first["event_id"] != second["event_id"]
    assert first["thread_id"] == "991"
    assert second["thread_id"] == "992"


def test_missing_thread_ids_use_payload_hash_to_avoid_timestamp_collisions(event_db):
    first_payload = _webhook_payload(
        thread={
            "type": "customer",
            "createdAt": "2026-09-24T11:58:00Z",
            "createdBy": {"type": "customer"},
            "body": "First legacy reply",
        }
    )
    second_payload = _webhook_payload(
        thread={
            "type": "customer",
            "createdAt": "2026-09-24T11:58:00Z",
            "createdBy": {"type": "customer"},
            "body": "Different legacy reply at the same timestamp",
        }
    )

    first = normalize_webhook_event(
        "convo.customer.reply.created",
        first_payload,
        clock=lambda: RECEIVED_AT,
    )
    second = normalize_webhook_event(
        "convo.customer.reply.created",
        second_payload,
        clock=lambda: RECEIVED_AT,
    )

    assert first.thread_id is None
    assert second.thread_id is None
    assert first.dedupe_key != second.dedupe_key
    assert append_event(event_db, first)["event_id"] != append_event(
        event_db, second
    )["event_id"]


def test_help_scout_summary_is_compact_allowlisted_and_contains_no_message_or_secret(
    event_db,
):
    raw_token = "A" * 43
    payload = _webhook_payload(
        thread={
            "id": 991,
            "type": "customer",
            "state": "published",
            "createdAt": "2026-09-24T11:58:00Z",
            "createdBy": {
                "type": "customer",
                "email": "private@example.com",
            },
            "body": f"Full customer body with Bearer {raw_token}",
            "_embedded": {
                "attachments": [
                    {"filename": f"private-{raw_token}.txt", "size": 123}
                ]
            },
        }
    )
    payload["subject"] = f"Secret sk-{raw_token}"
    payload["preview"] = "Full preview must not be stored"
    payload["primaryCustomer"] = {
        "email": "private@example.com",
        "first": "Private",
    }
    payload["authorization"] = f"Bearer {raw_token}"

    event = normalize_webhook_event(
        "convo.customer.reply.created",
        payload,
        clock=lambda: RECEIVED_AT,
    )
    encoded = json.dumps(event.summary, sort_keys=True)

    assert event.summary == {
        "attachment_count": 1,
        "conversation_number": "1442",
        "mailbox_id": "22",
        "status": "active",
        "tag_count": 1,
        "thread_state": "published",
        "thread_type": "customer",
    }
    assert raw_token not in encoded
    assert "Full customer body" not in encoded
    assert "Full preview" not in encoded
    assert "private@example.com" not in encoded
    assert "filename" not in encoded

    append_event(event_db, event)
    with sqlite3.connect(event_db) as db:
        stored = db.execute("SELECT * FROM agent_events").fetchone()
    stored_text = json.dumps(tuple(stored), default=str)
    assert raw_token not in stored_text
    assert "Full customer body" not in stored_text
    assert "Full preview" not in stored_text
    assert "private@example.com" not in stored_text


def test_append_supa_event_uses_namespace_dedupes_and_redacts_summary(event_db):
    raw_token = "B" * 43
    summary = {
        "operation_id": "op-123",
        "request_id": "req-123",
        "token_id": "token-123",
        "token_name": f"laptop Bearer {raw_token}",
        "operation_name": "draft.save",
        "status": "saved",
        "result": "sk-abcdefghijklmnop",
        "body": f"<p>Full body {raw_token}</p>",
        "authorization": f"Bearer {raw_token}",
        "nested": {"secret": raw_token},
        7: "non-string keys are not allowlisted",
    }

    first = append_supa_event(
        event_db,
        type="supa.draft.saved",
        conversation_id=602,
        thread_id=992,
        dedupe_key=f"operation:{raw_token}",
        summary=summary,
        clock=lambda: RECEIVED_AT,
    )
    retry = append_supa_event(
        event_db,
        type="supa.draft.saved",
        conversation_id=602,
        thread_id=992,
        dedupe_key=f"operation:{raw_token}",
        summary={"status": "should-not-replace-original"},
        clock=lambda: RECEIVED_AT + timedelta(minutes=1),
    )

    assert retry == first
    assert first["type"] == "supa.draft.saved"
    assert first["source"] == "supa"
    assert first["actor_type"] == "system"
    assert first["summary"] == {
        "operation_id": "op-123",
        "operation_name": "draft.save",
        "request_id": "req-123",
        "status": "saved",
        "token_id": "token-123",
        "token_name": "laptop Bearer [redacted]",
    }
    assert raw_token not in json.dumps(first, sort_keys=True)

    with sqlite3.connect(event_db) as db:
        row = db.execute(
            "SELECT dedupe_key, summary_json FROM agent_events"
        ).fetchone()
    assert raw_token not in row[0]
    assert raw_token not in row[1]


def test_append_supa_event_carries_run_id_without_full_result(event_db):
    event = append_supa_event(
        event_db,
        type="supa.run.completed",
        conversation_id=602,
        run_id="run-123",
        dedupe_key="run:run-123:completed",
        summary={
            "status": "succeeded",
            "result": "ok",
            "full_result": "large private task output",
        },
        clock=lambda: RECEIVED_AT,
    )

    assert event["run_id"] == "run-123"
    assert event["summary"] == {"result": "ok", "status": "succeeded"}
    assert "large private task output" not in json.dumps(event)


@pytest.mark.parametrize(
    ("column", "corrupt_value"),
    (
        ("event_type", "helpscout.customer.replied"),
        ("conversation_id", "Bearer stored-secret"),
        ("thread_id", "not a thread ID"),
        ("run_id", "not a run ID"),
        ("source", "untrusted-source"),
        ("actor_type", "untrusted-actor"),
        ("occurred_at", "not-a-timestamp"),
        ("received_at", "not-a-timestamp"),
        ("summary_json", '{"body":"stored secret"}'),
    ),
)
def test_append_replay_rejects_corrupt_stored_event_fields(
    event_db,
    column,
    corrupt_value,
):
    append_supa_event(
        event_db,
        type="supa.run.completed",
        conversation_id=602,
        thread_id=992,
        run_id="run-123",
        dedupe_key="run:run-123:completed",
        summary={"status": "succeeded"},
        clock=lambda: RECEIVED_AT,
    )
    with sqlite3.connect(event_db) as db:
        db.execute(f"UPDATE agent_events SET {column} = ?", (corrupt_value,))
        db.commit()

    with pytest.raises(AgentEventStorageError):
        append_supa_event(
            event_db,
            type="supa.run.completed",
            conversation_id=602,
            thread_id=992,
            run_id="run-123",
            dedupe_key="run:run-123:completed",
            summary={"status": "succeeded"},
            clock=lambda: RECEIVED_AT,
        )


def test_prune_events_removes_only_rows_older_than_thirty_day_cutoff(event_db):
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    cutoff = now - timedelta(days=30)
    event_times = (
        ("old", cutoff - timedelta(microseconds=1)),
        ("boundary", cutoff),
        ("new", cutoff + timedelta(microseconds=1)),
    )
    events = [
        append_supa_event(
            event_db,
            type="supa.run.completed",
            run_id=f"run-{label}",
            dedupe_key=f"run:{label}",
            summary={"status": "succeeded"},
            clock=lambda event_time=event_time: event_time,
        )
        for label, event_time in event_times
    ]
    with sqlite3.connect(event_db) as db:
        db.execute(
            """
            INSERT INTO agent_event_consumers (
                token_id, consumer_name, acknowledged_event_id, start_mode
            ) VALUES (?, ?, ?, ?)
            """,
            ("token-a", "laptop", events[0]["event_id"], "oldest"),
        )
        db.commit()

    deleted = prune_events(event_db, before=cutoff)

    assert deleted == 1
    with sqlite3.connect(event_db) as db:
        remaining = db.execute(
            "SELECT run_id FROM agent_events ORDER BY id"
        ).fetchall()
        consumer = db.execute(
            """
            SELECT acknowledged_event_id
            FROM agent_event_consumers
            WHERE token_id = 'token-a' AND consumer_name = 'laptop'
            """
        ).fetchone()
    assert remaining == [("run-boundary",), ("run-new",)]
    assert consumer == (events[0]["event_id"],)


def test_new_consumer_defaults_to_now_and_receives_only_future_events(event_db):
    existing = _append_feed_events(event_db, 2)
    token_bytes = _SequenceTokenBytes()

    assert (
        lease_next_event(
            event_db,
            token_id="token-a",
            consumer_name="laptop",
            clock=lambda: RECEIVED_AT + timedelta(minutes=1),
            token_bytes=token_bytes,
        )
        is None
    )
    future = append_supa_event(
        event_db,
        type="supa.run.completed",
        run_id="run-future",
        dedupe_key="run:future:completed",
        summary={"status": "succeeded"},
        clock=lambda: RECEIVED_AT + timedelta(minutes=2),
    )

    leased = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        start="oldest",
        clock=lambda: RECEIVED_AT + timedelta(minutes=3),
        token_bytes=token_bytes,
    )

    assert leased["event_id"] == future["event_id"]
    assert leased["event_id"] > existing[-1]["event_id"]
    assert token_bytes.calls == 1
    with sqlite3.connect(event_db) as db:
        row = db.execute(
            """
            SELECT acknowledged_event_id, start_mode
            FROM agent_event_consumers
            WHERE token_id = 'token-a' AND consumer_name = 'laptop'
            """
        ).fetchone()
    assert row == (existing[-1]["event_id"], "now")


def test_new_consumer_can_start_at_oldest_or_an_explicit_retained_event(event_db):
    events = _append_feed_events(event_db, 3)

    oldest = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="oldest-reader",
        start="oldest",
        clock=lambda: RECEIVED_AT + timedelta(minutes=1),
        token_bytes=_SequenceTokenBytes(),
    )
    explicit = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="explicit-reader",
        start=events[1]["event_id"],
        clock=lambda: RECEIVED_AT + timedelta(minutes=1),
        token_bytes=_SequenceTokenBytes(),
    )

    assert oldest["event_id"] == events[0]["event_id"]
    assert explicit["event_id"] == events[1]["event_id"]
    with sqlite3.connect(event_db) as db:
        modes = db.execute(
            """
            SELECT consumer_name, start_mode
            FROM agent_event_consumers
            ORDER BY consumer_name
            """
        ).fetchall()
    assert modes == [
        ("explicit-reader", "event_id"),
        ("oldest-reader", "oldest"),
    ]


def test_one_outstanding_lease_reuses_token_then_redelivers_with_fresh_token(event_db):
    events = _append_feed_events(event_db, 2)
    token_bytes = _SequenceTokenBytes()
    lease_time = RECEIVED_AT + timedelta(minutes=1)

    first = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        start="oldest",
        clock=lambda: lease_time,
        token_bytes=token_bytes,
    )
    before_expiry = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        clock=lambda: lease_time + timedelta(seconds=59),
        token_bytes=token_bytes,
    )
    after_expiry = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        clock=lambda: lease_time + timedelta(seconds=60),
        token_bytes=token_bytes,
    )

    assert before_expiry == first
    assert first["event_id"] == events[0]["event_id"]
    assert after_expiry["event_id"] == events[0]["event_id"]
    assert after_expiry["event_id"] != events[1]["event_id"]
    assert after_expiry["ack_token"] != first["ack_token"]
    assert token_bytes.calls == 2
    decoded = urlsafe_b64decode(first["ack_token"] + "=")
    assert len(decoded) == 32

    with sqlite3.connect(event_db) as db:
        row = db.execute(
            """
            SELECT lease_token_hash, last_ack_token_hash
            FROM agent_event_consumers
            WHERE token_id = 'token-a' AND consumer_name = 'laptop'
            """
        ).fetchone()
    assert len(row[0]) == 64
    assert row[0] not in {first["ack_token"], after_expiry["ack_token"]}
    assert row[1] is None

    with pytest.raises(AgentEventAckError):
        acknowledge_event(
            event_db,
            token_id="token-a",
            consumer_name="laptop",
            event_id=first["event_id"],
            ack_token=first["ack_token"],
            clock=lambda: lease_time + timedelta(seconds=61),
        )


def test_active_lease_recovers_after_process_restart_without_waiting(event_db):
    events = _append_feed_events(event_db, 2)
    token_bytes = _SequenceTokenBytes()
    lease_time = RECEIVED_AT + timedelta(minutes=1)
    first = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        start="oldest",
        clock=lambda: lease_time,
        token_bytes=token_bytes,
    )

    # Raw capabilities are intentionally process-local; reconstructing the
    # service leaves only the persisted one-way hash.
    with agent_events_module._ACK_TOKEN_CACHE_LOCK:
        agent_events_module._ACK_TOKEN_CACHE.clear()

    recovered = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        clock=lambda: lease_time + timedelta(seconds=1),
        token_bytes=token_bytes,
    )

    assert recovered["event_id"] == events[0]["event_id"]
    assert recovered["event_id"] != events[1]["event_id"]
    assert recovered["ack_token"] != first["ack_token"]
    assert token_bytes.calls == 2
    with pytest.raises(AgentEventAckError):
        acknowledge_event(
            event_db,
            token_id="token-a",
            consumer_name="laptop",
            event_id=first["event_id"],
            ack_token=first["ack_token"],
            clock=lambda: lease_time + timedelta(seconds=2),
        )

    assert acknowledge_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        event_id=recovered["event_id"],
        ack_token=recovered["ack_token"],
        clock=lambda: lease_time + timedelta(seconds=2),
    )["acknowledged"] is True


def test_concurrent_next_calls_share_one_outstanding_lease(event_db):
    event = _append_feed_events(event_db, 1)[0]
    lease_time = RECEIVED_AT + timedelta(minutes=1)
    token_bytes = _SequenceTokenBytes()
    ready = Barrier(2)

    def lease() -> dict:
        ready.wait()
        return lease_next_event(
            event_db,
            token_id="token-a",
            consumer_name="laptop",
            start="oldest",
            clock=lambda: lease_time,
            token_bytes=token_bytes,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        leases = list(executor.map(lambda _index: lease(), range(2)))

    assert leases[0] == leases[1]
    assert leases[0]["event_id"] == event["event_id"]
    assert token_bytes.calls == 1
    with sqlite3.connect(event_db) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM agent_event_consumers"
        ).fetchone()[0] == 1


def test_acknowledgement_enforces_ownership_and_is_idempotent(event_db):
    events = _append_feed_events(event_db, 2)
    now = RECEIVED_AT + timedelta(minutes=1)
    lease = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        start="oldest",
        clock=lambda: now,
        token_bytes=_SequenceTokenBytes(),
    )

    invalid_attempts = (
        {"token_id": "token-b", "consumer_name": "laptop"},
        {"token_id": "token-a", "consumer_name": "desktop"},
        {
            "token_id": "token-a",
            "consumer_name": "laptop",
            "event_id": events[1]["event_id"],
        },
        {
            "token_id": "token-a",
            "consumer_name": "laptop",
            "ack_token": "Z" * 43,
        },
    )
    for overrides in invalid_attempts:
        arguments = {
            "token_id": "token-a",
            "consumer_name": "laptop",
            "event_id": lease["event_id"],
            "ack_token": lease["ack_token"],
            **overrides,
        }
        with pytest.raises(AgentEventAckError):
            acknowledge_event(
                event_db,
                **arguments,
                clock=lambda: now + timedelta(seconds=1),
            )

    acknowledged = acknowledge_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        event_id=lease["event_id"],
        ack_token=lease["ack_token"],
        clock=lambda: now + timedelta(seconds=1),
    )
    replayed = acknowledge_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        event_id=lease["event_id"],
        ack_token=lease["ack_token"],
        clock=lambda: now + timedelta(days=1),
    )

    assert acknowledged == {
        "acknowledged": True,
        "event_id": lease["event_id"],
        "replayed": False,
    }
    assert replayed == {**acknowledged, "replayed": True}
    with sqlite3.connect(event_db) as db:
        row = db.execute(
            """
            SELECT acknowledged_event_id, leased_event_id, lease_expires_at,
                   lease_token_hash, last_ack_event_id, last_ack_token_hash
            FROM agent_event_consumers
            WHERE token_id = 'token-a' AND consumer_name = 'laptop'
            """
        ).fetchone()
    assert row[:4] == (lease["event_id"], None, None, None)
    assert row[4] == lease["event_id"]
    assert len(row[5]) == 64
    assert lease["ack_token"] not in row


def test_consumers_advance_sequentially_and_are_independent(event_db):
    events = _append_feed_events(event_db, 3)
    now = RECEIVED_AT + timedelta(minutes=1)
    first_a = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="shared-name",
        start="oldest",
        clock=lambda: now,
        token_bytes=_SequenceTokenBytes(),
    )
    first_b = lease_next_event(
        event_db,
        token_id="token-b",
        consumer_name="shared-name",
        start="oldest",
        clock=lambda: now,
        token_bytes=_SequenceTokenBytes(),
    )
    acknowledge_event(
        event_db,
        token_id="token-a",
        consumer_name="shared-name",
        event_id=first_a["event_id"],
        ack_token=first_a["ack_token"],
        clock=lambda: now + timedelta(seconds=1),
    )
    second_a = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="shared-name",
        clock=lambda: now + timedelta(seconds=2),
        token_bytes=_SequenceTokenBytes(),
    )
    still_first_b = lease_next_event(
        event_db,
        token_id="token-b",
        consumer_name="shared-name",
        clock=lambda: now + timedelta(seconds=2),
        token_bytes=_SequenceTokenBytes(),
    )

    assert first_a["event_id"] == first_b["event_id"] == events[0]["event_id"]
    assert second_a["event_id"] == events[1]["event_id"]
    assert still_first_b == first_b

    old_ack_retry = acknowledge_event(
        event_db,
        token_id="token-a",
        consumer_name="shared-name",
        event_id=first_a["event_id"],
        ack_token=first_a["ack_token"],
        clock=lambda: now + timedelta(seconds=3),
    )
    assert old_ack_retry["replayed"] is True
    assert (
        lease_next_event(
            event_db,
            token_id="token-a",
            consumer_name="shared-name",
            clock=lambda: now + timedelta(seconds=4),
            token_bytes=_SequenceTokenBytes(),
        )["event_id"]
        == events[1]["event_id"]
    )


def test_reset_clears_lease_moves_cursor_and_writes_safe_audit(event_db):
    events = _append_feed_events(event_db, 3)
    now = RECEIVED_AT + timedelta(minutes=1)
    lease = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        start="oldest",
        clock=lambda: now,
        token_bytes=_SequenceTokenBytes(),
    )

    reset = reset_event_consumer(
        event_db,
        token_id="token-a",
        token_name="Operator laptop",
        consumer_name="laptop",
        to="now",
        request_id="request-reset-1",
        clock=lambda: now + timedelta(seconds=1),
    )

    assert reset == {
        "acknowledged_event_id": events[-1]["event_id"],
        "consumer_name": "laptop",
        "start_mode": "now",
    }
    with pytest.raises(AgentEventAckError):
        acknowledge_event(
            event_db,
            token_id="token-a",
            consumer_name="laptop",
            event_id=lease["event_id"],
            ack_token=lease["ack_token"],
            clock=lambda: now + timedelta(seconds=2),
        )
    assert (
        lease_next_event(
            event_db,
            token_id="token-a",
            consumer_name="laptop",
            clock=lambda: now + timedelta(seconds=2),
            token_bytes=_SequenceTokenBytes(),
        )
        is None
    )

    with sqlite3.connect(event_db) as db:
        consumer = db.execute(
            """
            SELECT leased_event_id, lease_expires_at, lease_token_hash,
                   last_ack_event_id, last_ack_token_hash
            FROM agent_event_consumers
            WHERE token_id = 'token-a' AND consumer_name = 'laptop'
            """
        ).fetchone()
        audit = db.execute(
            """
            SELECT token_id, token_name, scope, operation_name, target_id,
                   request_id, request_hash, result, metadata_json
            FROM agent_audit
            WHERE operation_name = 'events.reset'
            """
        ).fetchone()
    assert consumer == (None, None, None, None, None)
    assert audit[:6] == (
        "token-a",
        "Operator laptop",
        "events",
        "events.reset",
        "laptop",
        "request-reset-1",
    )
    assert len(audit[6]) == 64
    assert audit[7] == "succeeded"
    assert json.loads(audit[8]) == {
        "consumer_name": "laptop",
        "from_acknowledged_event_id": events[0]["event_id"] - 1,
        "start_mode": "now",
        "to_acknowledged_event_id": events[-1]["event_id"],
    }
    assert lease["ack_token"] not in json.dumps(tuple(audit))


def test_reset_can_replay_from_oldest_or_an_explicit_event(event_db):
    events = _append_feed_events(event_db, 3)
    now = RECEIVED_AT + timedelta(minutes=1)
    assert (
        lease_next_event(
            event_db,
            token_id="token-a",
            consumer_name="laptop",
            clock=lambda: now,
            token_bytes=_SequenceTokenBytes(),
        )
        is None
    )

    reset_event_consumer(
        event_db,
        token_id="token-a",
        token_name="Laptop",
        consumer_name="laptop",
        to="oldest",
        request_id="request-reset-oldest",
        clock=lambda: now + timedelta(seconds=1),
    )
    oldest = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        clock=lambda: now + timedelta(seconds=2),
        token_bytes=_SequenceTokenBytes(),
    )
    assert oldest["event_id"] == events[0]["event_id"]

    reset_event_consumer(
        event_db,
        token_id="token-a",
        token_name="Laptop",
        consumer_name="laptop",
        to=events[2]["event_id"],
        request_id="request-reset-explicit",
        clock=lambda: now + timedelta(seconds=3),
    )
    explicit = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        clock=lambda: now + timedelta(seconds=4),
        token_bytes=_SequenceTokenBytes(),
    )
    assert explicit["event_id"] == events[2]["event_id"]


def test_reset_audit_failure_rolls_back_cursor_and_preserves_active_lease(event_db):
    event = _append_feed_events(event_db, 1)[0]
    now = RECEIVED_AT + timedelta(minutes=1)
    lease = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        start="oldest",
        clock=lambda: now,
        token_bytes=_SequenceTokenBytes(),
    )
    with sqlite3.connect(event_db) as db:
        db.execute(
            """
            CREATE TRIGGER reject_event_reset_audit
            BEFORE INSERT ON agent_audit
            WHEN NEW.operation_name = 'events.reset'
            BEGIN
                SELECT RAISE(ABORT, 'audit unavailable');
            END
            """
        )
        db.commit()

    with pytest.raises(AgentEventStorageError):
        reset_event_consumer(
            event_db,
            token_id="token-a",
            token_name="Laptop",
            consumer_name="laptop",
            to="now",
            request_id="request-reset-fails",
            clock=lambda: now + timedelta(seconds=1),
        )

    acknowledged = acknowledge_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        event_id=event["event_id"],
        ack_token=lease["ack_token"],
        clock=lambda: now + timedelta(seconds=2),
    )
    assert acknowledged["acknowledged"] is True
    with sqlite3.connect(event_db) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM agent_audit WHERE operation_name = 'events.reset'"
        ).fetchone()[0] == 0


def test_retention_marks_missed_cursor_expired_until_explicit_reset(event_db):
    now = RECEIVED_AT + timedelta(days=40)
    old_events = _append_feed_events(event_db, 2, start_at=RECEIVED_AT)
    retained = append_supa_event(
        event_db,
        type="supa.run.completed",
        run_id="run-retained",
        dedupe_key="run:retained:completed",
        summary={"status": "succeeded"},
        clock=lambda: now - timedelta(days=1),
    )
    lease = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        start="oldest",
        clock=lambda: now - timedelta(hours=1),
        token_bytes=_SequenceTokenBytes(),
    )
    assert lease["event_id"] == old_events[0]["event_id"]

    assert (
        prune_events(
            event_db,
            before=now - timedelta(days=30),
            clock=lambda: now,
        )
        == 2
    )

    with pytest.raises(AgentEventCursorExpiredError) as caught:
        lease_next_event(
            event_db,
            token_id="token-a",
            consumer_name="laptop",
            clock=lambda: now,
            token_bytes=_SequenceTokenBytes(),
        )
    assert caught.value.code == "cursor_expired"
    assert caught.value.details == {
        "current_head_event_id": retained["event_id"],
        "oldest_retained_event_id": retained["event_id"],
    }

    reset_event_consumer(
        event_db,
        token_id="token-a",
        token_name="Laptop",
        consumer_name="laptop",
        to="oldest",
        request_id="request-reset-expired",
        clock=lambda: now + timedelta(seconds=1),
    )
    replay = lease_next_event(
        event_db,
        token_id="token-a",
        consumer_name="laptop",
        clock=lambda: now + timedelta(seconds=2),
        token_bytes=_SequenceTokenBytes(),
    )
    assert replay["event_id"] == retained["event_id"]


def test_explicit_pruned_start_returns_cursor_expired_details_without_consumer(event_db):
    now = RECEIVED_AT + timedelta(days=40)
    old = _append_feed_events(event_db, 1, start_at=RECEIVED_AT)[0]
    retained = append_supa_event(
        event_db,
        type="supa.run.completed",
        run_id="run-retained",
        dedupe_key="run:retained:completed",
        summary={"status": "succeeded"},
        clock=lambda: now,
    )
    prune_events(
        event_db,
        before=now - timedelta(days=30),
        clock=lambda: now,
    )

    with pytest.raises(AgentEventCursorExpiredError) as caught:
        lease_next_event(
            event_db,
            token_id="token-a",
            consumer_name="laptop",
            start=old["event_id"],
            clock=lambda: now,
            token_bytes=_SequenceTokenBytes(),
        )
    assert caught.value.details == {
        "current_head_event_id": retained["event_id"],
        "oldest_retained_event_id": retained["event_id"],
    }
    with sqlite3.connect(event_db) as db:
        assert db.execute("SELECT COUNT(*) FROM agent_event_consumers").fetchone()[0] == 0


def test_revoked_token_replacement_is_an_independent_consumer(event_db):
    event = _append_feed_events(event_db, 1)[0]
    now = RECEIVED_AT + timedelta(minutes=1)
    token_bytes = _SequenceTokenBytes()
    old_lease = lease_next_event(
        event_db,
        token_id="revoked-token-id",
        consumer_name="laptop",
        start="oldest",
        clock=lambda: now,
        token_bytes=token_bytes,
    )
    replacement_lease = lease_next_event(
        event_db,
        token_id="replacement-token-id",
        consumer_name="laptop",
        start="oldest",
        clock=lambda: now,
        token_bytes=token_bytes,
    )

    assert old_lease["event_id"] == replacement_lease["event_id"] == event["event_id"]
    with pytest.raises(AgentEventAckError):
        acknowledge_event(
            event_db,
            token_id="replacement-token-id",
            consumer_name="laptop",
            event_id=old_lease["event_id"],
            ack_token=old_lease["ack_token"],
            clock=lambda: now + timedelta(seconds=1),
        )
    with sqlite3.connect(event_db) as db:
        assert db.execute("SELECT COUNT(*) FROM agent_event_consumers").fetchone()[0] == 2


def test_short_random_source_rolls_back_consumer_and_lease(event_db):
    _append_feed_events(event_db, 1)

    with pytest.raises(ValueError, match="at least 32 bytes"):
        lease_next_event(
            event_db,
            token_id="token-a",
            consumer_name="laptop",
            start="oldest",
            clock=lambda: RECEIVED_AT + timedelta(minutes=1),
            token_bytes=lambda _size: b"x" * 31,
        )

    with sqlite3.connect(event_db) as db:
        assert db.execute("SELECT COUNT(*) FROM agent_event_consumers").fetchone()[0] == 0


def test_future_explicit_start_is_rejected_without_creating_consumer(event_db):
    events = _append_feed_events(event_db, 1)

    with pytest.raises(AgentEventStartError):
        lease_next_event(
            event_db,
            token_id="token-a",
            consumer_name="laptop",
            start=events[0]["event_id"] + 100,
            clock=lambda: RECEIVED_AT + timedelta(minutes=1),
            token_bytes=_SequenceTokenBytes(),
        )

    with sqlite3.connect(event_db) as db:
        assert db.execute("SELECT COUNT(*) FROM agent_event_consumers").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_retention_worker_runs_immediately_every_six_hours_and_survives_errors(
    tmp_path, caplog
):
    now = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
    cutoffs = []
    sleeps = []

    def fake_prune(_db_path, *, before, clock):
        cutoffs.append(before)
        assert clock() == now
        if len(cutoffs) == 1:
            raise RuntimeError("synthetic private retention failure")
        return 3

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise asyncio.CancelledError

    with caplog.at_level("ERROR"):
        with pytest.raises(asyncio.CancelledError):
            await run_event_retention_worker(
                str(tmp_path / "events.db"),
                clock=lambda: now,
                sleep=fake_sleep,
                prune=fake_prune,
            )

    assert cutoffs == [now - timedelta(days=30)] * 2
    assert sleeps == [21_600, 21_600]
    assert "Event retention prune failed" in caplog.text
    assert "synthetic private retention failure" not in caplog.text


@pytest.mark.asyncio
async def test_scheduled_retention_prunes_old_events_and_keeps_consumer_state(event_db):
    now = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
    old = append_supa_event(
        event_db,
        type="supa.run.completed",
        run_id="run-retention-old",
        dedupe_key="run-retention-old:completed",
        summary={"status": "succeeded"},
        clock=lambda: now - timedelta(days=31),
    )
    retained = append_supa_event(
        event_db,
        type="supa.run.completed",
        run_id="run-retention-new",
        dedupe_key="run-retention-new:completed",
        summary={"status": "succeeded"},
        clock=lambda: now - timedelta(days=29),
    )
    lease_next_event(
        event_db,
        token_id="token-retention",
        consumer_name="retention-reader",
        start="oldest",
        clock=lambda: now - timedelta(days=31),
        token_bytes=lambda size: b"r" * size,
    )
    first_sleep = asyncio.Event()

    async def fake_sleep(seconds):
        assert seconds == 21_600
        first_sleep.set()
        await asyncio.Future()

    task = schedule_event_retention_worker(
        event_db,
        clock=lambda: now,
        sleep=fake_sleep,
    )
    await asyncio.wait_for(first_sleep.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with sqlite3.connect(event_db) as db:
        assert db.execute(
            "SELECT id FROM agent_events ORDER BY id"
        ).fetchall() == [(retained["event_id"],)]
        consumer = db.execute(
            """
            SELECT token_id, consumer_name, start_mode
            FROM agent_event_consumers
            """
        ).fetchone()
    assert old["event_id"] < retained["event_id"]
    assert consumer == ("token-retention", "retention-reader", "cursor_expired")
