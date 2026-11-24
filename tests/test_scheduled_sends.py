"""Scheduled agent replies (`supa draft schedule`): booking, send-time holds, restarts, auth, CLI.

The sender tests run the real ConversationGateway send path against the fake Help Scout
client of the write tests, so they exercise the same code `draft send` / `reply send` use.
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from api import agent as agent_api
from db.database import init_db
from main import app
from services import scheduled_sends
from services.agent_auth import AgentIdentity, write_token_records
from services.conversation_gateway import (
    ConversationGateway,
    ConversationGatewayError,
    draft_version,
)
from services.helpscout import MutationResult
from supa_cli.main import main as cli_main
from tests.test_agent_conversation_allowlist import (
    ALLOWED,
    CASE_HEADERS,
    CASE_TOKEN,
    FULL_TOKEN,
    OTHER,
    _assert_not_allowed,
    _conversation_result,
    _record,
)
from tests.test_agent_writes import FakeDraftClient, _conversation, _customer, _draft


CONVERSATION = 321
CUSTOMER_THREAD = 40
CUSTOMER_AT = datetime(2026, 9, 30, 11, 50, tzinfo=timezone.utc)
BOOKED_AT = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
SEND_IDENTITY = AgentIdentity(
    token_id="case-token-1",
    name="case-004",
    scopes=frozenset({"read", "draft", "send", "note"}),
    conversations=frozenset({CONVERSATION}),
)


def _at(value: datetime):
    return lambda: value


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _note(thread_id: int, created_at: str, *, source: str = "web", author: str = "user",
          body: str = "<p>Wait for my call before replying.</p>") -> dict:
    return {
        "id": thread_id,
        "type": "note",
        "state": "published",
        "body": body,
        "createdAt": created_at,
        "createdBy": {"type": author, "id": 9},
        "source": {"type": source},
    }


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "scheduled.db")
    init_db(path)
    return path


def _client(*, drafts: list[dict] | None = None) -> FakeDraftClient:
    return FakeDraftClient(
        _conversation(
            CONVERSATION,
            threads=[_customer(CUSTOMER_THREAD, "<p>Can you look at my header?</p>", _iso(CUSTOMER_AT))],
            drafts=drafts,
        )
    )


async def _book(client, db_path, *, key="sched-1", identity=SEND_IDENTITY, **kwargs) -> dict:
    kwargs.setdefault("body", "Hello Ada,\n\nThe header is fixed.")
    kwargs.setdefault("body_format", "text")
    expected_thread = kwargs.pop("expected_latest_thread_id", CUSTOMER_THREAD)
    return await scheduled_sends.schedule_send(
        gateway=ConversationGateway(client, db_path=db_path),
        db_path=db_path,
        identity=identity,
        conversation_id=CONVERSATION,
        expected_latest_thread_id=expected_thread,
        idempotency_key=key,
        clock=_at(BOOKED_AT),
        rng=random.Random(7),
        **kwargs,
    )


async def _process(client, db_path, row_id, *, now=None, identity=SEND_IDENTITY) -> str:
    return await scheduled_sends.process_row(
        row_id,
        gateway=ConversationGateway(client, db_path=db_path),
        db_path=db_path,
        clock=_at(now or BOOKED_AT + timedelta(hours=2)),
        identity_lookup=lambda _token_id: identity,
    )


def _sends(client) -> list[tuple]:
    return [call for call in client.calls if call[0] == "send"]


def _events(db_path) -> list[tuple[str, str]]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT event_type, summary_json FROM agent_events ORDER BY id").fetchall()
    finally:
        conn.close()
    return [(row[0], json.loads(row[1]).get("status")) for row in rows]


# --- send_at ---------------------------------------------------------------------------


def test_send_at_is_30_to_60_minutes_after_the_customer_message():
    now = CUSTOMER_AT + timedelta(minutes=5)
    rng = random.Random(1)
    picks = [
        scheduled_sends.compute_send_at(now=now, customer_message_at=CUSTOMER_AT, rng=rng)
        for _ in range(500)
    ]
    assert all(CUSTOMER_AT + timedelta(minutes=30) <= p <= CUSTOMER_AT + timedelta(minutes=60) for p in picks)
    # Random across the window, not a fixed delay.
    assert max(picks) - min(picks) > timedelta(minutes=25)


def test_custom_after_customer_window_is_respected():
    now = CUSTOMER_AT
    for seed in range(50):
        pick = scheduled_sends.compute_send_at(
            now=now, customer_message_at=CUSTOMER_AT, after_customer=(10, 12), rng=random.Random(seed)
        )
        assert CUSTOMER_AT + timedelta(minutes=10) <= pick <= CUSTOMER_AT + timedelta(minutes=12)


def test_past_window_sends_2_to_8_minutes_from_now():
    now = CUSTOMER_AT + timedelta(hours=3)
    rng = random.Random(2)
    picks = [
        scheduled_sends.compute_send_at(now=now, customer_message_at=CUSTOMER_AT, rng=rng)
        for _ in range(300)
    ]
    assert all(now + timedelta(minutes=2) <= p <= now + timedelta(minutes=8) for p in picks)


def test_send_at_is_never_earlier_than_one_minute_from_now():
    now = CUSTOMER_AT + timedelta(minutes=29, seconds=50)
    pick = scheduled_sends.compute_send_at(
        now=now, customer_message_at=CUSTOMER_AT, after_customer=(30, 30), rng=random.Random(3)
    )
    assert pick == now + timedelta(minutes=1)
    explicit = scheduled_sends.compute_send_at(
        now=now, customer_message_at=CUSTOMER_AT, send_at=now - timedelta(hours=1)
    )
    assert explicit == now + timedelta(minutes=1)


async def test_booking_stores_the_body_and_the_window_from_the_customer_message(db_path):
    client = _client()

    booked = await _book(client, db_path)

    assert booked["status"] == "queued"
    assert booked["source"] == "body"
    assert booked["customer_message_at"] == "2026-09-30T11:50:00Z"
    send_at = datetime.fromisoformat(booked["send_at"].replace("Z", "+00:00"))
    assert CUSTOMER_AT + timedelta(minutes=30) <= send_at <= CUSTOMER_AT + timedelta(minutes=60)
    assert "body" not in booked
    # Booking writes nothing to Help Scout: no draft, no placeholder to clean up later.
    assert [call[0] for call in client.calls] == ["get"]


async def test_booking_refuses_a_stale_latest_thread(db_path):
    client = _client()
    client.conversation["_embedded"]["threads"].append(
        _customer(41, "<p>One more thing</p>", "2026-09-30T11:58:00Z")
    )

    with pytest.raises(ConversationGatewayError) as raised:
        await _book(client, db_path)

    assert raised.value.code == "conversation_changed"
    assert raised.value.status == 409
    assert raised.value.retryable is False
    assert raised.value.details == {
        "thread_id": 41,
        "latest_actionable_thread_id": 41,
        "given_thread_id": 40,
    }
    assert str(raised.value) == "The latest actionable thread is now 41, not 40."
    assert scheduled_sends.latest_for_conversation(db_path, CONVERSATION) is None


@pytest.mark.parametrize("given_thread_id", [700, 701])
async def test_booking_refuses_a_draft_as_expected_thread(db_path, given_thread_id):
    drafts = [
        _draft(700, "<p>Older</p>", "2026-09-30T11:55:00Z"),
        _draft(701, "<p>Draft</p>", "2026-09-30T11:56:00Z"),
    ]
    client = _client(drafts=drafts)

    with pytest.raises(ConversationGatewayError) as raised:
        await _book(client, db_path, expected_latest_thread_id=given_thread_id)

    assert raised.value.code == "wrong_expected_thread"
    assert raised.value.status == 409
    assert raised.value.retryable is False
    assert raised.value.details == {
        "latest_actionable_thread_id": CUSTOMER_THREAD,
        "given_thread_id": given_thread_id,
        "draft_thread_id": given_thread_id,
    }
    assert "not the draft's thread" in str(raised.value)
    assert scheduled_sends.latest_for_conversation(db_path, CONVERSATION) is None
    assert [call[0] for call in client.calls] == ["get"]


async def test_rebooking_replaces_the_queued_row_and_same_key_replays(db_path):
    client = _client()
    first = await _book(client, db_path, key="sched-a")
    replay = await _book(client, db_path, key="sched-a")
    second = await _book(client, db_path, key="sched-b", body="Second version")

    assert replay["schedule_id"] == first["schedule_id"]
    assert second["schedule_id"] != first["schedule_id"]
    assert scheduled_sends.get_row(db_path, first["schedule_id"])["status"] == "cancelled"
    assert scheduled_sends.get_row(db_path, first["schedule_id"])["hold_reason"] == "replaced"
    with pytest.raises(ConversationGatewayError) as conflict:
        await _book(client, db_path, key="sched-a", body="Different body")
    assert conflict.value.code == "idempotency_conflict"


# --- sender ----------------------------------------------------------------------------


async def test_sender_sends_the_unchanged_reply_pending_through_the_reply_path(db_path):
    client = _client()
    booked = await _book(client, db_path)

    outcome = await _process(client, db_path, booked["schedule_id"])

    assert outcome == "sent"
    [send] = _sends(client)
    assert send[1] == CONVERSATION
    assert send[3] == "pending"
    assert "The header is fixed." in send[2]
    assert ("ensure_status", CONVERSATION, "pending") in client.calls
    row = scheduled_sends.get_row(db_path, booked["schedule_id"])
    assert row["status"] == "sent"
    assert row["sent_thread_id"] == 990
    assert row["completed_at"] is not None


async def test_sender_holds_when_the_customer_replied(db_path):
    client = _client()
    booked = await _book(client, db_path)
    client.conversation["_embedded"]["threads"].append(
        _customer(41, "<p>Also the footer</p>", "2026-09-30T12:20:00Z")
    )

    outcome = await _process(client, db_path, booked["schedule_id"])

    assert outcome == "held:customer_replied"
    assert _sends(client) == []
    row = scheduled_sends.get_row(db_path, booked["schedule_id"])
    assert (row["status"], row["hold_reason"]) == ("held", "customer_replied")
    assert ("supa.scheduled_send.held", "customer_replied") in _events(db_path)


async def test_sender_holds_when_the_saved_draft_changed(db_path):
    draft = _draft(50, "<p>Hello Ada, the header is fixed.</p>", "2026-09-30T11:55:00Z")
    client = _client(drafts=[draft])
    booked = await _book(client, db_path, body=None, body_format=None,
                         expected_version=draft_version(draft))
    assert booked["source"] == "draft"
    for thread in client.conversation["_embedded"]["threads"]:
        if thread["id"] == 50:
            thread["body"] = "<p>Support rewrote this.</p>"

    outcome = await _process(client, db_path, booked["schedule_id"])

    assert outcome == "held:draft_changed"
    assert _sends(client) == []


async def test_sender_sends_the_draft_snapshot_and_blanks_the_draft(db_path):
    draft = _draft(50, "<p>Hello Ada, the header is fixed.</p>", "2026-09-30T11:55:00Z")
    client = _client(drafts=[draft])
    booked = await _book(client, db_path, body=None, body_format=None,
                         expected_version=draft_version(draft))

    assert await _process(client, db_path, booked["schedule_id"]) == "sent"

    [send] = _sends(client)
    assert "the header is fixed" in send[2]
    # The draft thread is neutralized like every draft send (Help Scout cannot delete it).
    assert ("delete", CONVERSATION, 50) in client.calls


async def test_sender_holds_when_a_draft_appeared(db_path):
    client = _client()
    booked = await _book(client, db_path)
    client.conversation["_embedded"]["threads"].append(
        _draft(51, "<p>Support started writing</p>", "2026-09-30T12:10:00Z")
    )

    assert await _process(client, db_path, booked["schedule_id"]) == "held:draft_changed"
    assert _sends(client) == []


async def test_sender_holds_on_a_note_from_a_person_but_not_an_api_note(db_path):
    client = _client()
    booked = await _book(client, db_path)
    client.conversation["_embedded"]["threads"].append(
        _note(60, "2026-09-30T12:05:00Z", source="api", body="<p>Case agent note</p>")
    )
    client.conversation["_embedded"]["threads"].append(
        _note(61, "2026-09-30T11:30:00Z")  # before the booking
    )
    assert await _process(client, db_path, booked["schedule_id"]) == "sent"

    client2 = _client()
    booked2 = await _book(client2, db_path, key="sched-2")
    client2.conversation["_embedded"]["threads"].append(_note(62, "2026-09-30T12:05:00Z"))
    assert await _process(client2, db_path, booked2["schedule_id"]) == "held:note_added"
    assert _sends(client2) == []


async def test_sender_holds_when_someone_else_replied(db_path):
    client = _client()
    booked = await _book(client, db_path)
    reply = _draft(70, "<p>Support answered himself</p>", "2026-09-30T12:15:00Z")
    reply.update(state="published", draft=False)
    client.conversation["_embedded"]["threads"].append(reply)

    assert await _process(client, db_path, booked["schedule_id"]) == "held:agent_replied"
    assert _sends(client) == []


@pytest.mark.parametrize("status", ["closed", "spam"])
async def test_sender_holds_closed_or_spam(db_path, status):
    client = _client()
    booked = await _book(client, db_path)
    client.conversation["status"] = status

    assert await _process(client, db_path, booked["schedule_id"]) == f"held:conversation_{status}"
    assert _sends(client) == []


async def test_sender_rechecks_the_token_at_send_time(db_path):
    client = _client()
    revoked = await _book(client, db_path, key="sched-r")
    assert await _process(client, db_path, revoked["schedule_id"], identity=None) == "held:token_revoked"

    narrowed = await _book(client, db_path, key="sched-n")
    moved = AgentIdentity(token_id=SEND_IDENTITY.token_id, name="case-004",
                          scopes=SEND_IDENTITY.scopes, conversations=frozenset({555}))
    assert await _process(client, db_path, narrowed["schedule_id"], identity=moved) == (
        "held:conversation_not_allowed"
    )
    assert _sends(client) == []


async def test_upstream_error_retries_with_backoff_then_sends(db_path):
    client = _client()
    booked = await _book(client, db_path)
    client.send_result = MutationResult(False, 503, detail="upstream_server_error")
    client.apply_send = False
    first_try = BOOKED_AT + timedelta(hours=2)

    assert await _process(client, db_path, booked["schedule_id"], now=first_try) == (
        "retry:upstream_server_error"
    )
    row = scheduled_sends.get_row(db_path, booked["schedule_id"])
    assert row["status"] == "queued" and row["attempts"] == 1
    assert row["next_attempt_at"] == _iso(first_try + timedelta(seconds=60))
    assert scheduled_sends.due_row_ids(db_path, first_try + timedelta(seconds=30)) == []

    client.send_result = MutationResult(True, 201, resource_id=990)
    client.apply_send = True
    later = first_try + timedelta(seconds=61)
    assert scheduled_sends.due_row_ids(db_path, later) == [booked["schedule_id"]]
    assert await _process(client, db_path, booked["schedule_id"], now=later) == "sent"
    assert len(_sends(client)) == 2  # one rejected upstream, one delivered
    assert scheduled_sends.get_row(db_path, booked["schedule_id"])["attempts"] == 2


async def test_upstream_errors_fail_after_the_last_attempt(db_path):
    client = _client()
    booked = await _book(client, db_path)
    client.send_result = MutationResult(False, 503, detail="upstream_server_error")
    client.apply_send = False
    now = BOOKED_AT + timedelta(hours=2)
    outcomes = []
    for _ in range(scheduled_sends.MAX_ATTEMPTS):
        outcomes.append(await _process(client, db_path, booked["schedule_id"], now=now))
        now += timedelta(hours=1)

    assert outcomes[-1] == "failed:upstream_server_error"
    assert scheduled_sends.get_row(db_path, booked["schedule_id"])["status"] == "failed"
    assert ("supa.scheduled_send.failed", "upstream_server_error") in _events(db_path)


async def test_restart_after_a_recorded_send_never_sends_twice(db_path):
    client = _client()
    booked = await _book(client, db_path)
    assert await _process(client, db_path, booked["schedule_id"]) == "sent"
    # The daemon died after the ledger recorded the send but before the row was updated.
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE scheduled_sends SET status = 'queued', completed_at = NULL, sent_thread_id = NULL "
        "WHERE id = ?", (booked["schedule_id"],),
    )
    conn.commit()
    conn.close()

    restarted = await scheduled_sends.tick(
        gateway=ConversationGateway(client, db_path=db_path),
        db_path=db_path,
        clock=_at(BOOKED_AT + timedelta(hours=3)),
        identity_lookup=lambda _token_id: SEND_IDENTITY,
    )

    assert restarted == [(booked["schedule_id"], "sent")]
    assert len(_sends(client)) == 1
    assert scheduled_sends.get_row(db_path, booked["schedule_id"])["sent_thread_id"] == 990


async def test_restart_after_an_interrupted_send_fails_instead_of_resending(db_path):
    client = _client()
    booked = await _book(client, db_path)
    client.raise_on_send = asyncio.CancelledError()  # the daemon stops mid-request

    with pytest.raises(asyncio.CancelledError):
        await _process(client, db_path, booked["schedule_id"])
    client.raise_on_send = None

    # A new process releases the stale claim; the row is still queued with its ledger key.
    scheduled_sends.release_claims(db_path)
    outcomes = await scheduled_sends.tick(
        gateway=ConversationGateway(client, db_path=db_path),
        db_path=db_path,
        clock=_at(BOOKED_AT + timedelta(hours=2, minutes=1)),
        identity_lookup=lambda _token_id: SEND_IDENTITY,
    )

    assert outcomes == [(booked["schedule_id"], "failed:outcome_unknown")]
    assert len(_sends(client)) == 1


async def test_queued_row_overdue_after_downtime_sends_on_the_next_tick(db_path):
    client = _client()
    booked = await _book(client, db_path)
    # Nothing is due before send_at; days later (daemon was down) it goes on the first tick.
    assert await scheduled_sends.tick(
        gateway=ConversationGateway(client, db_path=db_path), db_path=db_path,
        clock=_at(BOOKED_AT), identity_lookup=lambda _t: SEND_IDENTITY,
    ) == []
    outcomes = await scheduled_sends.tick(
        gateway=ConversationGateway(client, db_path=db_path), db_path=db_path,
        clock=_at(BOOKED_AT + timedelta(days=2)), identity_lookup=lambda _t: SEND_IDENTITY,
    )
    assert outcomes == [(booked["schedule_id"], "sent")]


async def test_cancel_stops_a_queued_reply(db_path):
    client = _client()
    booked = await _book(client, db_path)

    cancelled = scheduled_sends.cancel_send(db_path=db_path, identity=SEND_IDENTITY,
                                            conversation_id=CONVERSATION)

    assert (cancelled["status"], cancelled["hold_reason"]) == ("cancelled", "cancelled_by_agent")
    assert await _process(client, db_path, booked["schedule_id"]) == "skipped"
    assert _sends(client) == []
    with pytest.raises(ConversationGatewayError) as missing:
        scheduled_sends.cancel_send(db_path=db_path, identity=SEND_IDENTITY,
                                    conversation_id=CONVERSATION)
    assert missing.value.code == "schedule_not_found"


# --- API and conversation allowlists ---------------------------------------------------


@pytest.fixture
def schedule_api(tmp_path, monkeypatch):
    registry_path = tmp_path / "private-agent-tokens.json"
    write_token_records(
        registry_path,
        [
            _record(CASE_TOKEN, "token-case-1", "case-004", conversations=[ALLOWED]),
            _record(FULL_TOKEN, "token-full-1", "full", scopes=("read",)),
        ],
    )
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(registry_path))
    db = str(tmp_path / "api.db")
    init_db(db)
    monkeypatch.setattr(agent_api, "get_supa_db_path", lambda: db)
    gateway = AsyncMock()
    dependency = agent_api.get_conversation_gateway
    app.dependency_overrides[dependency] = lambda: gateway
    try:
        yield TestClient(app), gateway, db
    finally:
        app.dependency_overrides.pop(dependency, None)


SCHEDULE_BODY = {
    "body": "Hello",
    "body_format": "text",
    "expected_latest_thread_id": 40,
    "idempotency_key": "k-schedule",
}


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/agent/v1/conversations/{id}/draft/schedule", SCHEDULE_BODY),
        ("get", "/agent/v1/conversations/{id}/draft/schedule", None),
        ("post", "/agent/v1/conversations/{id}/draft/schedule/cancel", {}),
    ],
)
def test_case_token_is_refused_other_conversations_before_any_work(schedule_api, method, path, payload):
    client, gateway, db = schedule_api
    kwargs = {"headers": CASE_HEADERS}
    if payload is not None:
        kwargs["json"] = payload

    response = getattr(client, method)(path.format(id=OTHER), **kwargs)

    _assert_not_allowed(response)
    assert gateway.mock_calls == []
    assert scheduled_sends.latest_for_conversation(db, OTHER) is None


def test_schedule_requires_the_send_scope(schedule_api):
    client, gateway, _db = schedule_api
    response = client.post(
        f"/agent/v1/conversations/{ALLOWED}/draft/schedule",
        headers={"Authorization": f"Bearer {FULL_TOKEN}"},
        json=SCHEDULE_BODY,
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "insufficient_scope"
    assert gateway.mock_calls == []


def test_merged_id_resolving_outside_the_list_is_refused(schedule_api):
    client, gateway, db = schedule_api
    gateway.get.return_value = _conversation_result(OTHER, ALLOWED)

    response = client.post(
        f"/agent/v1/conversations/{ALLOWED}/draft/schedule", headers=CASE_HEADERS, json=SCHEDULE_BODY
    )

    _assert_not_allowed(response)
    assert scheduled_sends.latest_for_conversation(db, OTHER) is None
    assert scheduled_sends.latest_for_conversation(db, ALLOWED) is None


def test_case_token_books_reads_and_cancels_its_own_conversation(schedule_api):
    client, gateway, _db = schedule_api
    conversation = _conversation_result(ALLOWED, ALLOWED)
    conversation["latest_actionable_thread_id"] = 40
    gateway.get.return_value = conversation

    booked = client.post(
        f"/agent/v1/conversations/{ALLOWED}/draft/schedule", headers=CASE_HEADERS,
        json={**SCHEDULE_BODY, "after_customer_min_minutes": 30, "after_customer_max_minutes": 60},
    )
    assert booked.status_code == 200, booked.text
    assert booked.json()["result"]["status"] == "queued"
    assert booked.json()["result"]["token_name"] == "case-004"

    shown = client.get(f"/agent/v1/conversations/{ALLOWED}/draft/schedule", headers=CASE_HEADERS)
    assert shown.json()["result"]["scheduled_send"]["schedule_id"] == booked.json()["result"]["schedule_id"]
    conversation_view = client.get(f"/agent/v1/conversations/{ALLOWED}", headers=CASE_HEADERS)
    assert conversation_view.json()["result"]["scheduled_send"]["status"] == "queued"

    cancelled = client.post(
        f"/agent/v1/conversations/{ALLOWED}/draft/schedule/cancel", headers=CASE_HEADERS, json={}
    )
    assert cancelled.json()["result"]["status"] == "cancelled"


def test_invalid_send_at_is_rejected(schedule_api):
    client, gateway, _db = schedule_api
    response = client.post(
        f"/agent/v1/conversations/{ALLOWED}/draft/schedule", headers=CASE_HEADERS,
        json={**SCHEDULE_BODY, "send_at": "tomorrow"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_send_at"


# --- CLI -------------------------------------------------------------------------------


def _cli(argv, handler, stdin_text=""):
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli_main(
        argv,
        transport=httpx.MockTransport(handler),
        environ={"SUPA_URL": "https://supa.example.test", "SUPA_TOKEN": "A" * 43},
        stdin=io.StringIO(stdin_text),
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def _envelope(result):
    return httpx.Response(200, json={"api_version": "v1", "request_id": "req_t", "result": result})


def test_cli_schedule_sends_body_window_and_key():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return _envelope({"schedule_id": 1, "status": "queued"})

    code, stdout, _ = _cli(
        ["draft", "schedule", "321", "--body-file", "-", "--body-format", "text",
         "--expected-thread", "40", "--after-customer", "30-60", "--idempotency-key", "sched-321-v1"],
        handler, stdin_text="Hello Ada",
    )

    assert code == 0
    assert seen == [("POST", "/agent/v1/conversations/321/draft/schedule", {
        "body": "Hello Ada",
        "body_format": "text",
        "expected_latest_thread_id": 40,
        "after_customer_min_minutes": 30,
        "after_customer_max_minutes": 60,
        "idempotency_key": "sched-321-v1",
    })]
    assert json.loads(stdout)["result"]["idempotency_key"] == "sched-321-v1"


def test_cli_schedule_saved_draft_at_an_explicit_time():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _envelope({"schedule_id": 1, "status": "queued"})

    code, _, _ = _cli(
        ["draft", "schedule", "321", "--expected-version", "a" * 64, "--expected-thread", "40",
         "--send-at", "2026-10-01T09:30:00Z", "--idempotency-key", "sched-2"],
        handler,
    )

    assert code == 0
    assert seen == [{
        "expected_latest_thread_id": 40,
        "expected_version": "a" * 64,
        "send_at": "2026-10-01T09:30:00Z",
        "idempotency_key": "sched-2",
    }]


@pytest.mark.parametrize(
    "argv",
    [
        ["draft", "schedule", "321", "--expected-thread", "40"],  # neither body nor version
        ["draft", "schedule", "321", "--body-file", "-", "--expected-thread", "40"],
        ["draft", "schedule", "321", "--expected-version", "a" * 64, "--expected-thread", "40",
         "--after-customer", "60-30"],
        ["draft", "schedule", "321", "--expected-version", "a" * 64, "--expected-thread", "40",
         "--send-at", "2026-10-01T09:30:00"],  # no timezone
        ["draft", "schedule", "321", "--expected-version", "a" * 64, "--expected-thread", "40",
         "--send-at", "2026-10-01T09:30:00Z", "--after-customer", "30-60"],
    ],
)
def test_cli_schedule_rejects_bad_usage_without_a_request(argv):
    def handler(request):  # pragma: no cover - must not be reached
        raise AssertionError("no request expected")

    code, _, stderr = _cli(argv, handler)

    assert code == 2
    assert json.loads(stderr)["error"]["code"] == "usage_error"


@pytest.mark.parametrize(
    ("command", "method", "path"),
    [
        ("schedule-get", "GET", "/agent/v1/conversations/321/draft/schedule"),
        ("schedule-cancel", "POST", "/agent/v1/conversations/321/draft/schedule/cancel"),
    ],
)
def test_cli_schedule_get_and_cancel_paths(command, method, path):
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return _envelope({"scheduled_send": None})

    code, _, _ = _cli(["draft", command, "321"], handler)

    assert code == 0
    assert seen == [(method, path)]


# --- daemon wiring -----------------------------------------------------------------------


def test_worker_is_started_with_the_daemon():
    import main

    assert "scheduled_send_task" in main.RECOVERY_TASK_NAMES


async def test_worker_loop_sends_due_rows_with_the_current_token(db_path, monkeypatch):
    client = _client()
    booked = await _book(client, db_path)
    looked_up = []
    monkeypatch.setattr(scheduled_sends, "current_identity",
                        lambda token_id: looked_up.append(token_id) or SEND_IDENTITY)
    monkeypatch.setattr(scheduled_sends, "_utc_now", lambda: BOOKED_AT + timedelta(hours=2))

    await scheduled_sends.run_worker_loop(
        db_path=db_path,
        gateway_factory=lambda: ConversationGateway(client, db_path=db_path),
        iterations=1,
    )

    assert looked_up == [SEND_IDENTITY.token_id]
    assert scheduled_sends.get_row(db_path, booked["schedule_id"])["status"] == "sent"
