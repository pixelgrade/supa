import asyncio
import copy
import json
import logging
import re
import tomllib
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from api import agent as agent_api
from api import dashboard as dashboard_api
from db.database import init_db
from main import app
from services.agent_auth import hash_token, write_token_records
from services.agent_auth import AgentIdentity
from services.agent_operations import (
    AgentOperationConflictError,
    AgentOperationDataError,
    AgentOperationStorageError,
    reserve_operation,
)
from services.agent_events import append_supa_event, prune_events
from services.conversation_locks import (
    ConversationBusyError,
    ConversationLockLostError,
    ConversationLockStorageError,
)
from services.conversation_gateway import AttachmentData, ConversationGatewayError
from services.helpscout import (
    HelpScoutAuthError,
    HelpScoutProtocolError,
    HelpScoutTransportError,
    HelpScoutUpstreamError,
)


API_VERSION = "v1"
READ_TOKEN = "synthetic-read-token"
READ_HEADERS = {"Authorization": f"Bearer {READ_TOKEN}"}
DRAFT_TOKEN = "synthetic-draft-token"
DRAFT_HEADERS = {"Authorization": f"Bearer {DRAFT_TOKEN}"}
SEND_TOKEN = "synthetic-send-token"
SEND_HEADERS = {"Authorization": f"Bearer {SEND_TOKEN}"}
LIMITED_TOKEN = "synthetic-limited-token"
LIMITED_HEADERS = {"Authorization": f"Bearer {LIMITED_TOKEN}"}
REQUEST_ID_PATTERN = re.compile(r"^req_[0-9a-f]{32}$")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_RESOURCE_ID = (1 << 63) - 1
AGENT_READ_PATHS = (
    "/agent/v1/health",
    "/agent/v1/desk",
    "/agent/v1/mailbox/insights",
    "/agent/v1/conversations/321/brief",
    "/agent/v1/conversations/321/precedents",
    "/agent/v1/capabilities",
    "/agent/v1/conversations",
    "/agent/v1/conversations/321",
    "/agent/v1/conversations/321/draft",
    "/agent/v1/conversations/321/attachments/701",
)
with (PROJECT_ROOT / "pyproject.toml").open("rb") as _project_file:
    PROJECT_VERSION = tomllib.load(_project_file)["project"]["version"]


def _token_record(
    token: str,
    *,
    token_id: str = "token-read-123",
    name: str = "review-desk",
    scopes: tuple[str, ...] = ("read",),
) -> dict[str, object]:
    return {
        "token_id": token_id,
        "name": name,
        "hash": hash_token(token),
        "scopes": list(scopes),
        "created_at": "2026-09-24T12:00:00+00:00",
    }


def test_task_api_scope_idempotency_and_owner_reads(agent_client, tmp_path, monkeypatch):
    client, _registry = agent_client
    db_path = str(tmp_path / "task-api.db")
    init_db(db_path)
    monkeypatch.setattr(agent_api, "get_supa_db_path", lambda: db_path)
    scheduled = []
    monkeypatch.setattr(agent_api, "schedule_agent_run", lambda run_id: scheduled.append(run_id))
    payload = {
        "task_name": "ticket.process",
        "arguments": {"conversation_id": 321},
        "idempotency_key": "process-321",
    }
    denied = client.post("/agent/v1/tasks", headers=READ_HEADERS, json=payload)
    assert denied.status_code == 403
    started = client.post("/agent/v1/tasks", headers=LIMITED_HEADERS, json=payload)
    assert started.status_code == 200
    run = started.json()["result"]
    assert run["status"] == "queued"
    assert run["delivery_ceiling"] == "draft"
    assert scheduled == [run["run_id"]]
    again = client.post("/agent/v1/tasks", headers=LIMITED_HEADERS, json=payload)
    assert again.json()["result"]["run_id"] == run["run_id"]
    assert scheduled == [run["run_id"], run["run_id"]]  # claim gate prevents duplicate execution
    assert client.get(f"/agent/v1/runs/{run['run_id']}", headers=LIMITED_HEADERS).status_code == 200
    assert client.get(f"/agent/v1/runs/{run['run_id']}", headers=READ_HEADERS).status_code == 403
    events = client.get(f"/agent/v1/runs/{run['run_id']}/events?after=0&wait=0", headers=LIMITED_HEADERS)
    assert events.status_code == 200
    assert events.json()["result"]["events"] == []
    allow_send = {**payload, "idempotency_key": "process-send-321", "arguments": {"conversation_id": 321, "allow_send": True}}
    assert client.post("/agent/v1/tasks", headers=LIMITED_HEADERS, json=allow_send).status_code == 403
    assert client.post("/agent/v1/tasks", headers=LIMITED_HEADERS, json={**payload, "task_name": "shell.exec"}).status_code == 400


@pytest.fixture
def agent_client(tmp_path, monkeypatch):
    registry_path = tmp_path / "private-agent-tokens.json"
    write_token_records(
        registry_path,
        [
            _token_record(READ_TOKEN, scopes=("send", "read", "events")),
            _token_record(
                "synthetic-no-read-token",
                token_id="token-no-read-456",
                name="publisher",
                scopes=("send",),
            ),
            _token_record(
                DRAFT_TOKEN,
                token_id="token-draft-789",
                name="draft-writer",
                scopes=("draft",),
            ),
            _token_record(
                SEND_TOKEN,
                token_id="token-send-901",
                name="sender",
                scopes=("send",),
            ),
            _token_record(
                LIMITED_TOKEN,
                token_id="token-limited-234",
                name="limited-agent",
                scopes=("read", "draft", "events", "task"),
            ),
        ],
    )
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(registry_path))
    monkeypatch.delenv("HELPSCOUT_WEBHOOK_SECRET", raising=False)
    return TestClient(app), registry_path


@pytest.fixture
def event_api_client(agent_client, tmp_path, monkeypatch):
    client, registry_path = agent_client
    db_path = str(tmp_path / "agent-api-events.db")
    init_db(db_path)
    monkeypatch.setenv("HELPSCOUT_WEBHOOK_SECRET", "synthetic-webhook-secret")
    monkeypatch.setattr(agent_api, "get_supa_db_path", lambda: db_path)
    return client, registry_path, db_path


@pytest.fixture
def gateway_client(agent_client):
    client, registry_path = agent_client
    gateway = AsyncMock()
    dependency = agent_api.get_conversation_gateway
    app.dependency_overrides[dependency] = lambda: gateway
    try:
        yield client, registry_path, gateway
    finally:
        app.dependency_overrides.pop(dependency, None)


def _normalized_conversation(*, draft: dict | None = None) -> dict:
    return {
        "id": 321,
        "requested_id": 123,
        "resolved_id": 321,
        "was_merged": True,
        "number": 9001,
        "subject": "Synthetic conversation",
        "status": "pending",
        "tags": ["field-journal", "follow-up"],
        "mailbox": {"id": 1, "name": "Support"},
        "assignee": {
            "id": 9,
            "type": "user",
            "name": "Desk Operator",
            "email": "operator@example.com",
        },
        "customer": {
            "id": 8,
            "type": "customer",
            "name": "Mica Fern",
            "email": "mica@example.com",
        },
        "timestamps": {
            "created_at": "2026-09-20T08:00:00Z",
            "updated_at": "2026-09-22T12:00:00Z",
            "user_updated_at": "2026-09-22T11:30:00Z",
            "closed_at": None,
            "customer_waiting_since": "2026-09-21T10:00:00Z",
        },
        "help_scout_url": "https://secure.helpscout.net/conversation/321",
        "waiting_on": "agent",
        "latest_actionable_thread_id": 40,
        "threads": [
            {
                "id": 40,
                "type": "customer",
                "state": "published",
                "draft": False,
                "is_internal_note": False,
                "superseded_by_reply": False,
                "author": {
                    "id": 8,
                    "type": "customer",
                    "role": "customer",
                    "name": "Mica Fern",
                    "email": "mica@example.com",
                },
                "source": {"type": "email", "via": "helpdesk"},
                "body_html": "<p>Please help.</p>",
                "body_text": "Please help.",
                "created_at": "2026-09-21T10:00:00Z",
                "updated_at": None,
                "opened_at": None,
                "attachments": [
                    {
                        "id": 701,
                        "filename": "screen.png",
                        "mime_type": "image/png",
                        "size": 2048,
                        "is_image": True,
                        "is_blank_transport_artifact": False,
                    }
                ],
            }
        ],
        "draft": draft,
        "version": "a" * 64,
    }


def _assert_error(
    response,
    *,
    status: int,
    code: str,
    message: str,
    retryable: bool = False,
) -> dict:
    assert response.status_code == status
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body == {
        "api_version": API_VERSION,
        "request_id": body["request_id"],
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "details": {},
        },
    }
    assert REQUEST_ID_PATTERN.fullmatch(body["request_id"])
    return body


@contextmanager
def _temporary_app_route(route):
    app.router.routes.append(route)
    try:
        yield
    finally:
        app.router.routes.remove(route)


@pytest.mark.parametrize("path", AGENT_READ_PATHS)
@pytest.mark.parametrize(
    "authorization",
    [None, "", "Basic abc", "Bearer", "Bearer ", "Bearer one two", "Bearer invalid"],
)
def test_agent_read_routes_return_stable_401_envelopes(
    agent_client, path, authorization
):
    client, _ = agent_client
    headers = {} if authorization is None else {"Authorization": authorization}

    response = client.get(path, headers=headers)

    _assert_error(
        response,
        status=401,
        code="authentication_failed",
        message="Authentication failed",
    )
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("path", AGENT_READ_PATHS)
def test_agent_read_routes_require_read_scope(agent_client, path):
    client, _ = agent_client

    response = client.get(
        path,
        headers={"Authorization": "Bearer synthetic-no-read-token"},
    )

    _assert_error(
        response,
        status=403,
        code="insufficient_scope",
        message="Insufficient scope",
    )
    assert "www-authenticate" not in response.headers


def test_health_returns_safe_versioned_success_with_server_generated_ids(agent_client):
    client, registry_path = agent_client

    first = client.get(
        "/agent/v1/health",
        headers={**READ_HEADERS, "X-Request-ID": "untrusted-correlation-value"},
    )
    second = client.get("/agent/v1/health", headers=READ_HEADERS)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["cache-control"] == "no-store"
    assert second.headers["cache-control"] == "no-store"
    first_body = first.json()
    second_body = second.json()
    assert first_body == {
        "api_version": API_VERSION,
        "request_id": first_body["request_id"],
        "result": {"status": "ok", "server_version": PROJECT_VERSION},
    }
    assert REQUEST_ID_PATTERN.fullmatch(first_body["request_id"])
    assert REQUEST_ID_PATTERN.fullmatch(second_body["request_id"])
    assert first_body["request_id"] != second_body["request_id"]
    assert "untrusted-correlation-value" not in first.text
    assert READ_TOKEN not in first.text
    assert str(registry_path) not in first.text


def test_capabilities_reports_identity_scopes_and_truthful_read_surface(agent_client):
    client, registry_path = agent_client

    response = client.get("/agent/v1/capabilities", headers=READ_HEADERS)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    from services.extensions import agent_commands, agent_tasks
    commands = body["result"]["commands"]
    commands["available"] = [key for key in commands["available"] if key not in {*agent_commands(), *agent_tasks()}]
    body["result"]["tasks"] = [task for task in body["result"]["tasks"] if task["name"] not in agent_tasks()]
    assert body == {
        "api_version": API_VERSION,
        "request_id": body["request_id"],
        "result": {
            "api_version": API_VERSION,
            "server_version": PROJECT_VERSION,
            "identity": {
                "name": "review-desk",
                "token_id": "token-read-123",
            },
            "scopes": ["events", "read", "send"],
            "conversations": None,
            "commands": {
                "available": [
                    "attachment.download",
                    "capabilities",
                    "conversation.brief",
                    "conversation.create",
                    "conversation.discard",
                    "conversation.get",
                    "conversation.list",
                    "conversation.precedents",
                    "conversation.search",
                    "conversation.state",
                    "desk",
                    "draft.get",
                    "draft.save",
                    "draft.schedule",
                    "draft.schedule.cancel",
                    "draft.schedule.get",
                    "draft.send",
                    "health",
                    "mailbox.insights",
                    "mailbox.operations",
                    "note.add",
                    "note.update",
                    "operation.get",
                    "reply.send",
                    "request.add",
                    "request.done",
                    "run.get",
                    "run.watch",
                    "ticket.evaluate",
                    "ticket.process",
                    "ticket.recover",
                ],
                "planned": [],
            },
            "tasks": [
                {"name": "ticket.recover", "required_scopes": ["draft", "read", "task"]},
                {"name": "ticket.evaluate", "required_scopes": ["read", "task"]},
                {"name": "ticket.process", "required_scopes": ["draft", "read", "task"]},
            ],
        },
    }
    assert REQUEST_ID_PATTERN.fullmatch(body["request_id"])
    assert READ_TOKEN not in response.text
    assert hash_token(READ_TOKEN) not in response.text
    assert str(registry_path) not in response.text
    assert "SUPA_AGENT_TOKENS_FILE" not in response.text


def test_capabilities_advertises_event_commands_only_with_signed_webhooks(
    agent_client, monkeypatch
):
    client, _ = agent_client

    disabled = client.get("/agent/v1/capabilities", headers=READ_HEADERS)
    monkeypatch.setenv("HELPSCOUT_WEBHOOK_SECRET", "synthetic-webhook-secret")
    enabled = client.get("/agent/v1/capabilities", headers=READ_HEADERS)

    disabled_commands = disabled.json()["result"]["commands"]["available"]
    enabled_commands = enabled.json()["result"]["commands"]["available"]
    assert not any(command.startswith("events.") for command in disabled_commands)
    assert enabled_commands == sorted(
        [*disabled_commands, "events.ack", "events.next", "events.reset"]
    )


def test_event_routes_require_events_scope_and_signed_webhook_configuration(
    agent_client, monkeypatch, tmp_path
):
    client, _ = agent_client
    db_path = str(tmp_path / "disabled-events.db")
    init_db(db_path)
    monkeypatch.setattr(agent_api, "get_supa_db_path", lambda: db_path)

    disabled = client.post(
        "/agent/v1/events/next",
        headers=READ_HEADERS,
        json={"consumer": "support-agent", "wait": 0},
    )
    _assert_error(
        disabled,
        status=503,
        code="event_feed_unavailable",
        message="Event feed is unavailable",
    )

    monkeypatch.setenv("HELPSCOUT_WEBHOOK_SECRET", "synthetic-webhook-secret")
    forbidden = client.post(
        "/agent/v1/events/next",
        headers=DRAFT_HEADERS,
        json={"consumer": "support-agent", "wait": 0},
    )
    _assert_error(
        forbidden,
        status=403,
        code="insufficient_scope",
        message="Insufficient scope",
    )


def test_event_next_is_immediate_and_consumer_is_namespaced_by_token(
    event_api_client, monkeypatch
):
    client, _, db_path = event_api_client
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(agent_api, "_event_clock", lambda: now, raising=False)
    event = append_supa_event(
        db_path,
        type="supa.run.completed",
        run_id="run-immediate",
        dedupe_key="run-immediate:completed",
        summary={"status": "succeeded"},
        clock=lambda: now,
    )

    first = client.post(
        "/agent/v1/events/next",
        headers=READ_HEADERS,
        json={"consumer": "shared-name", "start": "oldest", "wait": 0},
    )
    second = client.post(
        "/agent/v1/events/next",
        headers=LIMITED_HEADERS,
        json={"consumer": "shared-name", "start": "oldest", "wait": 0},
    )

    assert first.status_code == second.status_code == 200
    assert first.headers["cache-control"] == second.headers["cache-control"] == "no-store"
    first_event = first.json()["result"]["event"]
    second_event = second.json()["result"]["event"]
    assert first_event["event_id"] == second_event["event_id"] == event["event_id"]
    assert first_event["type"] == "supa.run.completed"
    assert first_event["summary"] == {"status": "succeeded"}
    assert first_event["lease_expires_at"] == "2026-09-25T12:01:00.000000Z"
    assert len(first_event["ack_token"]) == 43
    assert first_event["ack_token"] != second_event["ack_token"]
    assert READ_TOKEN not in first.text
    assert hash_token(READ_TOKEN) not in first.text


@pytest.mark.parametrize("wait", [-1, 56, True, 1.5, "1"])
def test_event_next_rejects_out_of_bounds_or_non_integer_wait(
    event_api_client, wait
):
    client, _, _ = event_api_client

    response = client.post(
        "/agent/v1/events/next",
        headers=READ_HEADERS,
        json={"consumer": "support-agent", "wait": wait},
    )

    _assert_error(
        response,
        status=422,
        code="invalid_request",
        message="Invalid request",
    )


def test_event_next_long_poll_rechecks_database_after_fake_notification_wait(
    event_api_client, monkeypatch
):
    client, _, db_path = event_api_client
    wall_now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    monotonic_now = 10.0
    waits: list[float] = []

    monkeypatch.setattr(agent_api, "_event_clock", lambda: wall_now, raising=False)
    monkeypatch.setattr(
        agent_api,
        "_event_poll_monotonic",
        lambda: monotonic_now,
        raising=False,
    )

    async def fake_wait(_generation, timeout):
        nonlocal monotonic_now
        waits.append(timeout)
        monotonic_now += timeout
        append_supa_event(
            db_path,
            type="supa.run.completed",
            run_id="run-during-wait",
            dedupe_key="run-during-wait:completed",
            summary={"status": "succeeded"},
            clock=lambda: wall_now,
        )

    monkeypatch.setattr(
        agent_api,
        "_wait_for_event_notification",
        fake_wait,
        raising=False,
    )

    response = client.post(
        "/agent/v1/events/next",
        headers=READ_HEADERS,
        json={"consumer": "polling-agent", "start": "now", "wait": 5},
    )

    assert response.status_code == 200
    assert response.json()["result"]["event"]["run_id"] == "run-during-wait"
    assert waits == [1.0]


def test_event_next_empty_timeout_uses_fake_clock_without_sleeping(
    event_api_client, monkeypatch
):
    client, _, _ = event_api_client
    monotonic_now = 40.0
    waits: list[float] = []

    monkeypatch.setattr(
        agent_api,
        "_event_poll_monotonic",
        lambda: monotonic_now,
        raising=False,
    )

    async def fake_wait(_generation, timeout):
        nonlocal monotonic_now
        waits.append(timeout)
        monotonic_now += timeout

    monkeypatch.setattr(
        agent_api,
        "_wait_for_event_notification",
        fake_wait,
        raising=False,
    )

    response = client.post(
        "/agent/v1/events/next",
        headers=READ_HEADERS,
        json={"consumer": "empty-agent", "start": "now", "wait": 3},
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["result"] == {"event": None}
    assert waits == [1.0, 1.0, 1.0]


def test_event_ack_is_idempotent_and_reset_is_explicit(event_api_client, monkeypatch):
    client, _, db_path = event_api_client
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(agent_api, "_event_clock", lambda: now, raising=False)
    event = append_supa_event(
        db_path,
        type="supa.run.completed",
        run_id="run-ack",
        dedupe_key="run-ack:completed",
        summary={"status": "succeeded"},
        clock=lambda: now,
    )
    lease = client.post(
        "/agent/v1/events/next",
        headers=READ_HEADERS,
        json={"consumer": "ack-agent", "start": "oldest", "wait": 0},
    ).json()["result"]["event"]

    payload = {"consumer": "ack-agent", "ack_token": lease["ack_token"]}
    first = client.post(
        f"/agent/v1/events/{event['event_id']}/ack",
        headers=READ_HEADERS,
        json=payload,
    )
    replay = client.post(
        f"/agent/v1/events/{event['event_id']}/ack",
        headers=READ_HEADERS,
        json=payload,
    )
    reset = client.post(
        "/agent/v1/events/reset",
        headers=READ_HEADERS,
        json={"consumer": "ack-agent", "to": "now"},
    )

    assert first.status_code == replay.status_code == reset.status_code == 200
    assert first.json()["result"] == {
        "acknowledged": True,
        "event_id": event["event_id"],
        "replayed": False,
    }
    assert replay.json()["result"] == {
        "acknowledged": True,
        "event_id": event["event_id"],
        "replayed": True,
    }
    assert reset.json()["result"] == {
        "acknowledged_event_id": event["event_id"],
        "consumer_name": "ack-agent",
        "start_mode": "now",
    }


def test_event_endpoints_return_typed_start_and_ack_errors(event_api_client):
    client, _, db_path = event_api_client
    event = append_supa_event(
        db_path,
        type="supa.run.completed",
        run_id="run-errors",
        dedupe_key="run-errors:completed",
        summary={"status": "succeeded"},
    )

    invalid_start = client.post(
        "/agent/v1/events/next",
        headers=READ_HEADERS,
        json={"consumer": "future-agent", "start": event["event_id"] + 10, "wait": 0},
    )
    invalid_ack = client.post(
        f"/agent/v1/events/{event['event_id']}/ack",
        headers=READ_HEADERS,
        json={"consumer": "future-agent", "ack_token": "a" * 43},
    )

    start_body = invalid_start.json()
    assert invalid_start.status_code == 400
    assert start_body["error"] == {
        "code": "event_start_invalid",
        "message": "Event start is invalid",
        "retryable": False,
        "details": {
            "event_id": event["event_id"] + 10,
            "current_head_event_id": event["event_id"],
        },
    }
    _assert_error(
        invalid_ack,
        status=409,
        code="event_ack_invalid",
        message="Event acknowledgement is invalid or expired",
    )


def test_event_cursor_expired_details_are_sanitized(event_api_client, monkeypatch):
    client, _, db_path = event_api_client
    old_time = datetime(2026, 8, 1, tzinfo=timezone.utc)
    current_time = datetime(2026, 9, 25, tzinfo=timezone.utc)
    monkeypatch.setattr(agent_api, "_event_clock", lambda: current_time, raising=False)
    old = append_supa_event(
        db_path,
        type="supa.run.completed",
        run_id="run-old",
        dedupe_key="run-old:completed",
        summary={"status": "succeeded"},
        clock=lambda: old_time,
    )
    leased = client.post(
        "/agent/v1/events/next",
        headers=READ_HEADERS,
        json={"consumer": "expired-agent", "start": "oldest", "wait": 0},
    )
    assert leased.status_code == 200
    retained = append_supa_event(
        db_path,
        type="supa.run.completed",
        run_id="run-retained",
        dedupe_key="run-retained:completed",
        summary={"status": "succeeded"},
        clock=lambda: current_time,
    )
    assert prune_events(
        db_path,
        before=old_time + timedelta(days=1),
        clock=lambda: current_time,
    ) == 1

    response = client.post(
        "/agent/v1/events/next",
        headers=READ_HEADERS,
        json={"consumer": "expired-agent", "wait": 0},
    )

    body = response.json()
    assert response.status_code == 409
    assert body["error"] == {
        "code": "cursor_expired",
        "message": "Event cursor expired",
        "retryable": False,
        "details": {
            "oldest_retained_event_id": retained["event_id"],
            "current_head_event_id": retained["event_id"],
        },
    }
    assert str(db_path) not in response.text
    assert old["run_id"] not in response.text


def test_revoked_event_token_cannot_resume_consumer(event_api_client):
    client, registry_path, _ = event_api_client
    initialized = client.post(
        "/agent/v1/events/next",
        headers=READ_HEADERS,
        json={"consumer": "revoked-agent", "start": "now", "wait": 0},
    )
    assert initialized.status_code == 200
    write_token_records(
        registry_path,
        [
            _token_record(
                LIMITED_TOKEN,
                token_id="token-limited-234",
                name="limited-agent",
                scopes=("read", "draft", "events", "task"),
            )
        ],
    )

    response = client.post(
        "/agent/v1/events/next",
        headers=READ_HEADERS,
        json={"consumer": "revoked-agent", "wait": 0},
    )

    _assert_error(
        response,
        status=401,
        code="authentication_failed",
        message="Authentication failed",
    )


def test_agent_router_does_not_change_dashboard_auth_response(agent_client, monkeypatch):
    client, _ = agent_client
    monkeypatch.setenv("ADMIN_API_KEY", "synthetic-admin-key")

    response = client.get("/dashboard/queue")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid admin key"}
    assert "cache-control" not in response.headers


def test_unexpected_agent_exception_uses_safe_error_envelope_and_logs_context(
    agent_client, monkeypatch, caplog
):
    client, _ = agent_client
    sensitive_exception = "synthetic internal detail that must not leak"

    def fail_success_response(*_args, **_kwargs):
        raise RuntimeError(sensitive_exception)

    monkeypatch.setattr(agent_api, "success_response", fail_success_response)
    safe_client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="api.agent"):
        response = safe_client.get("/agent/v1/health", headers=READ_HEADERS)

    body = _assert_error(
        response,
        status=500,
        code="internal_error",
        message="Internal server error",
        retryable=True,
    )
    assert sensitive_exception not in response.text
    assert READ_TOKEN not in response.text
    error_records = [
        record
        for record in caplog.records
        if record.name == "api.agent" and record.levelno >= logging.ERROR
    ]
    assert len(error_records) == 1
    [record] = error_records
    assert record.exc_info is not None
    message = record.getMessage()
    assert f"request_id={body['request_id']}" in message
    assert "method=GET" in message
    assert "path=/agent/v1/health" in message
    assert READ_TOKEN not in message


def test_unknown_agent_path_uses_not_found_envelope(agent_client):
    client, _ = agent_client

    response = client.get("/agent/v1/not-a-real-resource")

    _assert_error(
        response,
        status=404,
        code="not_found",
        message="Resource not found",
    )


def test_wrong_agent_method_uses_method_not_allowed_envelope(agent_client):
    client, _ = agent_client

    response = client.post("/agent/v1/health", headers=READ_HEADERS)

    _assert_error(
        response,
        status=405,
        code="method_not_allowed",
        message="Method not allowed",
    )
    assert response.headers["allow"] == "GET"


@pytest.mark.parametrize(
    "path",
    ["/agent/v1/health/", "/agent/v1/capabilities/"],
)
def test_agent_trailing_slash_redirect_is_rejected_with_not_found_envelope(
    agent_client, path
):
    client, _ = agent_client

    response = client.get(path, headers=READ_HEADERS, follow_redirects=False)

    _assert_error(
        response,
        status=404,
        code="not_found",
        message="Resource not found",
    )
    assert "location" not in response.headers


def test_agent_redirect_policy_does_not_change_non_agent_redirects(agent_client):
    client, _ = agent_client

    response = client.get("/health/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"].endswith("/health")
    assert "cache-control" not in response.headers
    assert response.content == b""


def test_agent_boundary_leaves_unrelated_404_and_500_responses_unchanged(
    agent_client, monkeypatch
):
    client, _ = agent_client

    not_found = client.get("/not-an-agent-route")

    assert not_found.status_code == 404
    assert not_found.json() == {"detail": "Not Found"}
    assert "cache-control" not in not_found.headers

    sensitive_exception = "unrelated dashboard exception"

    def fail_processor():
        raise RuntimeError(sensitive_exception)

    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.setattr(dashboard_api, "get_processor", fail_processor)
    safe_client = TestClient(app, raise_server_exceptions=False)

    server_error = safe_client.get("/dashboard/queue")

    assert server_error.status_code == 500
    assert server_error.text == "Internal Server Error"
    assert "cache-control" not in server_error.headers


def test_agent_stream_failure_discards_partial_bytes_and_returns_one_500_envelope(
    agent_client, monkeypatch
):
    client, _ = agent_client
    generated_ids: list[str] = []

    def fixed_request_id():
        request_id = "req_a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4"
        generated_ids.append(request_id)
        return request_id

    monkeypatch.setattr(agent_api, "_new_request_id", fixed_request_id)

    async def stream_then_fail(_request):
        async def body():
            yield b"partial-private-bytes"
            raise RuntimeError("private generator failure")

        return StreamingResponse(body(), media_type="application/octet-stream")

    route = Route(
        "/agent/v1/_test/stream-failure",
        stream_then_fail,
        methods=["GET"],
    )
    with _temporary_app_route(route):
        response = TestClient(app, raise_server_exceptions=False).get(route.path)

    body = _assert_error(
        response,
        status=500,
        code="internal_error",
        message="Internal server error",
        retryable=True,
    )
    assert response.content == response.text.encode()
    assert "partial-private-bytes" not in response.text
    assert "private generator failure" not in response.text
    assert body["request_id"] == generated_ids[0]
    assert generated_ids == [body["request_id"]]


@pytest.mark.parametrize(
    ("limit_name", "middleware_options"),
    [
        ("bytes", {"max_buffered_bytes": 128, "max_buffered_messages": 100}),
        ("messages", {"max_buffered_bytes": 4096, "max_buffered_messages": 4}),
    ],
)
def test_agent_boundary_rejects_oversized_responses_without_partial_output(
    limit_name, middleware_options, monkeypatch, caplog
):
    request_id = f"req_{'a' * 32 if limit_name == 'bytes' else 'b' * 32}"
    monkeypatch.setattr(agent_api, "_new_request_id", lambda: request_id)
    generator_closed = False

    async def oversized_stream(_request):
        async def body():
            nonlocal generator_closed
            try:
                if limit_name == "bytes":
                    yield b"private-response-byte" * 32
                else:
                    for _ in range(20):
                        yield b"x"
            finally:
                generator_closed = True

        return StreamingResponse(body(), media_type="application/octet-stream")

    local_app = FastAPI()
    local_app.router.routes.append(
        Route("/agent/v1/_test/oversized", oversized_stream, methods=["GET"])
    )
    bounded_app = agent_api.AgentAPIBoundaryMiddleware(
        local_app,
        **middleware_options,
    )

    with caplog.at_level(logging.ERROR, logger="api.agent"):
        response = TestClient(bounded_app, raise_server_exceptions=False).get(
            "/agent/v1/_test/oversized"
        )

    body = _assert_error(
        response,
        status=500,
        code="response_too_large",
        message="Agent API response exceeded its buffer limit",
        retryable=False,
    )
    assert body["request_id"] == request_id
    assert "private-response-byte" not in response.text
    assert generator_closed is True
    records = [record for record in caplog.records if record.name == "api.agent"]
    assert len(records) == 1
    assert f"request_id={request_id}" in records[0].getMessage()


def test_post_completion_exception_replays_completed_response_and_logs(
    agent_client, monkeypatch, caplog
):
    client, _ = agent_client
    request_id = "req_deadbeefdeadbeefdeadbeefdeadbeef"
    monkeypatch.setattr(agent_api, "_new_request_id", lambda: request_id)

    def fail_after_response():
        raise RuntimeError("post-completion private failure")

    async def completed_then_fail(_request):
        return Response(
            b"complete-response",
            status_code=202,
            media_type="text/plain",
            background=BackgroundTask(fail_after_response),
        )

    route = Route(
        "/agent/v1/_test/post-completion",
        completed_then_fail,
        methods=["GET"],
    )
    with _temporary_app_route(route), caplog.at_level(
        logging.ERROR, logger="api.agent"
    ):
        response = TestClient(app, raise_server_exceptions=False).get(route.path)

    assert response.status_code == 202
    assert response.content == b"complete-response"
    assert response.headers["cache-control"] == "no-store"
    records = [record for record in caplog.records if record.name == "api.agent"]
    assert len(records) == 1
    assert records[0].exc_info is not None
    message = records[0].getMessage()
    assert "post-completion" in message.lower()
    assert f"request_id={request_id}" in message


@pytest.mark.parametrize(
    ("status", "code", "message", "retryable"),
    [
        (400, "invalid_request", "Invalid request", False),
        (500, "internal_error", "Internal server error", True),
    ],
)
def test_raw_matched_error_response_is_replaced_with_canonical_envelope(
    agent_client, status, code, message, retryable
):
    client, _ = agent_client

    async def raw_error():
        return JSONResponse(
            {"raw": "private details must not be reflected"},
            status_code=status,
            headers={
                "WWW-Authenticate": "Bearer",
                "X-Unapproved-Upstream": "private-header",
            },
        )

    route = agent_api.AgentAPIRoute(
        path=f"/agent/v1/_test/raw-{status}",
        endpoint=raw_error,
        methods=["GET"],
    )
    with _temporary_app_route(route):
        response = client.get(route.path)

    _assert_error(
        response,
        status=status,
        code=code,
        message=message,
        retryable=retryable,
    )
    assert response.headers["www-authenticate"] == "Bearer"
    assert "x-unapproved-upstream" not in response.headers
    assert "private details" not in response.text


def test_agent_boundary_preserves_complete_multichunk_streams_and_head_semantics(
    agent_client,
):
    client, _ = agent_client

    async def stream_chunks(_request):
        async def body():
            yield b"first-"
            yield b"second"

        return StreamingResponse(body(), media_type="text/plain")

    route = Route("/agent/v1/_test/chunks", stream_chunks, methods=["GET"])
    with _temporary_app_route(route):
        response = client.get(route.path)
        head = client.head(route.path)

    assert response.status_code == 200
    assert response.content == b"first-second"
    assert response.headers["cache-control"] == "no-store"
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["cache-control"] == "no-store"


async def test_agent_boundary_does_not_swallow_cancellation():
    async def cancelled_app(_scope, _receive, _send):
        raise asyncio.CancelledError

    async def receive():
        return {"type": "http.disconnect"}

    async def send(_message):
        pytest.fail("a cancelled response must not emit output")

    middleware = agent_api.AgentAPIBoundaryMiddleware(cancelled_app)
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "https",
        "path": "/agent/v1/health",
        "raw_path": b"/agent/v1/health",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 443),
        "state": {},
    }

    with pytest.raises(asyncio.CancelledError):
        await middleware(scope, receive, send)


def test_handled_agent_errors_do_not_log_unexpected_exception(agent_client, caplog):
    client, _ = agent_client

    with caplog.at_level(logging.ERROR, logger="api.agent"):
        unauthorized = client.get("/agent/v1/health")
        missing = client.get("/agent/v1/missing")
        wrong_method = client.post("/agent/v1/health", headers=READ_HEADERS)

    assert unauthorized.status_code == 401
    assert missing.status_code == 404
    assert wrong_method.status_code == 405
    assert [record for record in caplog.records if record.name == "api.agent"] == []


def test_request_validation_error_uses_agent_error_envelope(agent_client):
    client, _ = agent_client

    async def validate_count(count: int):
        return {"count": count}

    route = agent_api.AgentAPIRoute(
        path="/agent/v1/_test/validation",
        endpoint=validate_count,
        methods=["GET"],
    )
    with _temporary_app_route(route):
        response = client.get(f"{route.path}?count=not-an-integer")

    _assert_error(
        response,
        status=422,
        code="invalid_request",
        message="Invalid request",
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, ("invalid_request", "Invalid request", False)),
        (409, ("conflict", "Request conflicts with current state", False)),
        (429, ("rate_limited", "Too many requests", True)),
        (500, ("internal_error", "Internal server error", True)),
        (502, ("upstream_error", "Upstream service failed", True)),
    ],
)
def test_http_error_contract_has_stable_representative_mappings(status, expected):
    assert agent_api._http_error_contract(status) == expected


def test_source_tree_server_version_matches_authoritative_project_metadata():
    assert agent_api.SERVER_VERSION == PROJECT_VERSION


def test_server_version_fallback_reads_project_metadata(tmp_path, monkeypatch):
    metadata_path = tmp_path / "pyproject.toml"
    metadata_path.write_text('[project]\nversion = "9.8.7"\n', encoding="utf-8")

    def package_not_installed(_name):
        raise agent_api.PackageNotFoundError("supa")

    monkeypatch.setattr(agent_api, "version", package_not_installed)
    monkeypatch.setattr(
        agent_api,
        "PROJECT_METADATA_PATH",
        metadata_path,
        raising=False,
    )

    assert agent_api._server_version() == "9.8.7"


def test_server_version_prefers_installed_metadata_and_fails_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_api, "version", lambda _name: "7.6.5")
    monkeypatch.setattr(
        agent_api,
        "PROJECT_METADATA_PATH",
        tmp_path / "missing.toml",
        raising=False,
    )
    assert agent_api._server_version() == "7.6.5"

    def package_not_installed(_name):
        raise agent_api.PackageNotFoundError("supa")

    monkeypatch.setattr(agent_api, "version", package_not_installed)
    assert agent_api._server_version() == "unknown"


def test_get_conversation_returns_normalized_merged_result(gateway_client):
    client, _, gateway = gateway_client
    conversation = _normalized_conversation()
    gateway.get.return_value = conversation

    response = client.get("/agent/v1/conversations/123", headers=READ_HEADERS)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body == {
        "api_version": API_VERSION,
        "request_id": body["request_id"],
        # The booked-reply summary (supa draft schedule); null without a booking.
        "result": {**conversation, "scheduled_send": None},
    }
    assert body["result"]["requested_id"] == 123
    assert body["result"]["resolved_id"] == 321
    assert body["result"]["was_merged"] is True
    gateway.get.assert_awaited_once_with(123)


def test_get_conversation_carries_each_threads_superseded_by_reply(gateway_client):
    client, _, gateway = gateway_client
    conversation = _normalized_conversation()
    conversation["threads"][0]["superseded_by_reply"] = True
    gateway.get.return_value = conversation

    response = client.get("/agent/v1/conversations/123", headers=READ_HEADERS)

    assert response.status_code == 200
    assert response.json()["result"]["threads"][0]["superseded_by_reply"] is True


def test_list_conversations_passes_bounded_filters_and_opaque_cursor(gateway_client):
    client, _, gateway = gateway_client
    conversation = _normalized_conversation()
    gateway.list.return_value = {
        "conversations": [conversation],
        "next_cursor": "eyJwYWdlIjoyLCJ2IjoxfQ",
    }

    response = client.get(
        "/agent/v1/conversations",
        params={
            "status": "pending",
            "tag": "field-journal",
            "query": "header spacing",
            "cursor": "opaque-input",
            "limit": 37,
        },
        headers=READ_HEADERS,
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["api_version"] == API_VERSION
    assert REQUEST_ID_PATTERN.fullmatch(body["request_id"])
    assert body["result"] == {
        "conversations": [conversation],
        "next_cursor": "eyJwYWdlIjoyLCJ2IjoxfQ",
    }
    gateway.list.assert_awaited_once_with(
        status="pending",
        tag="field-journal",
        query="header spacing",
        cursor="opaque-input",
        limit=37,
    )


@pytest.mark.parametrize("limit", [0, 101])
def test_list_conversations_rejects_out_of_bounds_limit_before_gateway(
    gateway_client, limit
):
    client, _, gateway = gateway_client

    response = client.get(
        "/agent/v1/conversations",
        params={"limit": limit},
        headers=READ_HEADERS,
    )

    _assert_error(
        response,
        status=422,
        code="invalid_request",
        message="Invalid request",
    )
    gateway.list.assert_not_awaited()


@pytest.mark.parametrize(
    "path",
    [
        "/agent/v1/conversations/0",
        "/agent/v1/conversations/-1",
        f"/agent/v1/conversations/{MAX_RESOURCE_ID + 1}",
        "/agent/v1/conversations/not-an-integer",
        "/agent/v1/conversations/321/attachments/0",
        f"/agent/v1/conversations/321/attachments/{MAX_RESOURCE_ID + 1}",
    ],
)
def test_conversation_routes_reject_invalid_resource_ids_without_redirect(
    gateway_client, path
):
    client, _, gateway = gateway_client

    response = client.get(path, headers=READ_HEADERS, follow_redirects=False)

    _assert_error(
        response,
        status=422,
        code="invalid_request",
        message="Invalid request",
    )
    assert "location" not in response.headers
    gateway.get.assert_not_awaited()
    gateway.attachment.assert_not_awaited()


@pytest.mark.parametrize(
    "draft",
    [
        None,
        {
            "thread_id": 60,
            "type": "message",
            "state": "draft",
            "body_html": "<p>Draft</p>",
            "body_text": "Draft",
            "created_at": "2026-09-22T12:00:00Z",
            "updated_at": None,
            "version": "b" * 64,
        },
    ],
)
def test_get_draft_returns_newest_live_draft_or_null(gateway_client, draft):
    client, _, gateway = gateway_client
    gateway.get.return_value = _normalized_conversation(draft=draft)

    response = client.get("/agent/v1/conversations/321/draft", headers=READ_HEADERS)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["result"] == {
        "draft": draft,
        "latest_actionable_thread_id": 40,
    }
    gateway.get.assert_awaited_once_with(321)


def test_put_draft_requires_draft_scope_and_forwards_strict_write(gateway_client):
    client, _, gateway = gateway_client
    gateway.save_draft.return_value = {
        "action": "created",
        "conversation_id": 321,
        "thread_id": 701,
        "body_sha256": "c" * 64,
        "version": "d" * 64,
    }
    payload = {
        "body": "Hello\n\nWorld",
        "body_format": "text",
        "expected_version": None,
        "idempotency_key": "draft-create-1",
    }

    denied = client.put(
        "/agent/v1/conversations/321/draft",
        headers=SEND_HEADERS,
        json=payload,
    )
    _assert_error(
        denied,
        status=403,
        code="insufficient_scope",
        message="Insufficient scope",
    )
    gateway.save_draft.assert_not_awaited()

    response = client.put(
        "/agent/v1/conversations/321/draft",
        headers=DRAFT_HEADERS,
        json=payload,
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["result"] == gateway.save_draft.return_value
    gateway.save_draft.assert_awaited_once_with(
        321,
        body="Hello\n\nWorld",
        body_format="text",
        expected_version=None,
        idempotency_key="draft-create-1",
        identity=AgentIdentity(
            token_id="token-draft-789",
            name="draft-writer",
            scopes=frozenset({"draft"}),
        ),
        request_id=body["request_id"],
    )


@pytest.mark.parametrize("path", ["draft/send", "replies"])
def test_send_routes_reject_tokens_without_send_scope(gateway_client, path):
    client, _, gateway = gateway_client
    payload = {
        "expected_latest_thread_id": 40,
        "idempotency_key": "send-denied-1",
    }
    if path == "draft/send":
        payload["expected_version"] = "b" * 64
    else:
        payload.update({"body": "Hello", "body_format": "text"})

    response = client.post(
        f"/agent/v1/conversations/321/{path}",
        headers=LIMITED_HEADERS,
        json=payload,
    )

    _assert_error(
        response,
        status=403,
        code="insufficient_scope",
        message="Insufficient scope",
    )
    gateway.send_draft.assert_not_awaited()
    gateway.send_reply.assert_not_awaited()


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        (
            "PUT",
            "/agent/v1/conversations/321/draft",
            {
                "body": "Hello",
                "body_format": "text",
                "expected_version": None,
                "idempotency_key": "unauthenticated-draft",
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/draft/send",
            {
                "expected_version": "a" * 64,
                "expected_latest_thread_id": 40,
                "idempotency_key": "unauthenticated-draft-send",
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/replies",
            {
                "body": "Hello",
                "body_format": "text",
                "expected_latest_thread_id": 40,
                "idempotency_key": "unauthenticated-reply",
            },
        ),
    ],
)
def test_write_routes_require_bearer_authentication(
    gateway_client, method, path, payload
):
    client, _, gateway = gateway_client

    response = client.request(method, path, json=payload)

    _assert_error(
        response,
        status=401,
        code="authentication_failed",
        message="Authentication failed",
    )
    assert response.headers["www-authenticate"] == "Bearer"
    gateway.save_draft.assert_not_awaited()
    gateway.send_draft.assert_not_awaited()
    gateway.send_reply.assert_not_awaited()


@pytest.mark.parametrize("conversation_id", ["0", "-1", str(MAX_RESOURCE_ID + 1), "true"])
def test_write_routes_reject_invalid_conversation_ids_before_gateway(
    gateway_client, conversation_id
):
    client, _, gateway = gateway_client

    response = client.post(
        f"/agent/v1/conversations/{conversation_id}/replies",
        headers=SEND_HEADERS,
        json={
            "body": "Hello",
            "body_format": "text",
            "expected_latest_thread_id": 40,
            "idempotency_key": "invalid-conversation-id",
        },
    )

    _assert_error(
        response,
        status=422,
        code="invalid_request",
        message="Invalid request",
    )
    gateway.send_reply.assert_not_awaited()


@pytest.mark.parametrize("method", ["draft", "reply"])
def test_send_routes_require_send_scope_and_forward_preconditions(
    gateway_client, method
):
    client, _, gateway = gateway_client
    result = {
        "action": "sent",
        "body_sha256": "e" * 64,
        "conversation_id": 321,
        "status": "sent",
        "thread_id": 990,
        "warnings": [],
    }
    if method == "draft":
        gateway.send_draft.return_value = result
        path = "/agent/v1/conversations/321/draft/send"
        payload = {
            "expected_version": "b" * 64,
            "expected_latest_thread_id": 40,
            "idempotency_key": "draft-send-1",
        }
    else:
        gateway.send_reply.return_value = result
        path = "/agent/v1/conversations/321/replies"
        payload = {
            "body": "Hello",
            "body_format": "text",
            "expected_latest_thread_id": 40,
            "idempotency_key": "reply-send-1",
        }

    response = client.post(path, headers=SEND_HEADERS, json=payload)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["result"] == result
    expected_identity = AgentIdentity(
        token_id="token-send-901",
        name="sender",
        scopes=frozenset({"send"}),
    )
    if method == "draft":
        gateway.send_draft.assert_awaited_once_with(
            321,
            expected_version="b" * 64,
            expected_latest_thread_id=40,
            idempotency_key="draft-send-1",
            identity=expected_identity,
            request_id=body["request_id"],
            cc=None,
        )
    else:
        gateway.send_reply.assert_awaited_once_with(
            321,
            body="Hello",
            body_format="text",
            expected_latest_thread_id=40,
            idempotency_key="reply-send-1",
            identity=expected_identity,
            request_id=body["request_id"],
            cc=None,
        )


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        (
            "PUT",
            "/agent/v1/conversations/321/draft",
            {
                "body": "Hello",
                "body_format": "text",
                "idempotency_key": "missing-expected-version",
            },
        ),
        (
            "PUT",
            "/agent/v1/conversations/321/draft",
            {
                "body": "Hello",
                "body_format": "markdown",
                "expected_version": None,
                "idempotency_key": "invalid-format",
            },
        ),
        (
            "PUT",
            "/agent/v1/conversations/321/draft",
            {
                "body": "Hello",
                "body_format": "text",
                "expected_version": "A" * 64,
                "idempotency_key": "uppercase-version",
            },
        ),
        (
            "PUT",
            "/agent/v1/conversations/321/draft",
            {
                "body": "Hello",
                "body_format": "text",
                "expected_version": None,
                "idempotency_key": "extra-field",
                "unexpected": True,
            },
        ),
        (
            "PUT",
            "/agent/v1/conversations/321/draft",
            {
                "body": ["not", "text"],
                "body_format": "text",
                "expected_version": None,
                "idempotency_key": "non-string-body",
            },
        ),
        (
            "PUT",
            "/agent/v1/conversations/321/draft",
            {
                "body": "Hello",
                "body_format": "text",
                "expected_version": 123,
                "idempotency_key": "non-string-version",
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/draft/send",
            {
                "expected_latest_thread_id": 40,
                "idempotency_key": "missing-draft-version",
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/draft/send",
            {
                "expected_version": None,
                "expected_latest_thread_id": 40,
                "idempotency_key": "null-draft-version",
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/replies",
            {
                "body": "Hello",
                "body_format": "text",
                "expected_latest_thread_id": True,
                "idempotency_key": "boolean-thread-id",
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/replies",
            {
                "body": "Hello",
                "body_format": "text",
                "expected_latest_thread_id": MAX_RESOURCE_ID + 1,
                "idempotency_key": "oversized-thread-id",
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/replies",
            {
                "body": "Hello",
                "body_format": "text",
                "expected_latest_thread_id": 40,
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/replies",
            {
                "body": "Hello",
                "body_format": "text",
                "expected_latest_thread_id": 40,
                "idempotency_key": "x" * 201,
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/replies",
            {
                "body": "Hello",
                "body_format": "text",
                "expected_latest_thread_id": 40,
                "idempotency_key": 123,
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/replies",
            {
                "body": "Hello",
                "body_format": "text",
                "expected_latest_thread_id": 40,
                "idempotency_key": "   ",
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/replies",
            {
                "body": "Hello",
                "body_format": "text",
                "expected_latest_thread_id": 40,
                "idempotency_key": "prefix_" + "A" * 43 + "_suffix",
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/replies",
            {
                "body": "",
                "body_format": "text",
                "expected_latest_thread_id": 40,
                "idempotency_key": "empty-body",
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/replies",
            {
                "body": "x" * (1024 * 1024 + 1),
                "body_format": "text",
                "expected_latest_thread_id": 40,
                "idempotency_key": "oversized-body",
            },
        ),
    ],
)
def test_write_request_models_reject_invalid_or_missing_fields_before_gateway(
    gateway_client, method, path, payload
):
    client, _, gateway = gateway_client
    headers = DRAFT_HEADERS if method == "PUT" else SEND_HEADERS

    response = client.request(method, path, headers=headers, json=payload)

    _assert_error(
        response,
        status=422,
        code="invalid_request",
        message="Invalid request",
    )
    gateway.save_draft.assert_not_awaited()
    gateway.send_draft.assert_not_awaited()
    gateway.send_reply.assert_not_awaited()


@pytest.mark.parametrize(
    ("failure", "status", "code", "message", "retryable"),
    [
        (
            ConversationGatewayError(
                "private draft conflict",
                code="draft_conflict",
                status=409,
                retryable=False,
            ),
            409,
            "draft_conflict",
            "The draft changed after it was read",
            False,
        ),
        (
            ConversationLockLostError("321"),
            409,
            "conversation_lock_lost",
            "The conversation lock was lost",
            False,
        ),
        (
            ConversationGatewayError(
                "private locked response",
                code="conversation_locked",
                status=412,
                retryable=False,
            ),
            412,
            "conversation_locked",
            "Conversation is locked upstream",
            False,
        ),
        (
            ConversationGatewayError(
                "private rate response",
                code="upstream_rate_limited",
                status=429,
                retryable=True,
            ),
            429,
            "upstream_rate_limited",
            "Too many requests",
            True,
        ),
        (
            ConversationGatewayError(
                "private upstream response",
                code="upstream_auth_error",
                status=502,
                retryable=False,
            ),
            502,
            "upstream_auth_error",
            "Upstream authentication failed",
            False,
        ),
        (
            ConversationGatewayError(
                "private upstream response",
                code="upstream_server_error",
                status=503,
                retryable=True,
            ),
            503,
            "upstream_server_error",
            "Upstream service failed",
            True,
        ),
        (
            ConversationBusyError("321"),
            503,
            "conversation_busy",
            "Conversation is busy",
            True,
        ),
        (
            ConversationGatewayError(
                "private timeout response",
                code="request_timeout",
                status=504,
                retryable=True,
            ),
            504,
            "request_timeout",
            "Upstream service timed out",
            True,
        ),
        (
            ConversationGatewayError(
                "private corrupt result",
                code="invalid_operation_result",
                status=500,
                retryable=False,
            ),
            500,
            "invalid_operation_result",
            "Stored operation result is invalid",
            False,
        ),
        (
            ConversationLockStorageError("321"),
            500,
            "conversation_lock_storage_error",
            "The conversation lock store is unavailable",
            False,
        ),
    ],
)
def test_typed_write_failures_preserve_safe_status_code_and_retryability(
    gateway_client, failure, status, code, message, retryable
):
    client, _, gateway = gateway_client
    gateway.send_reply.side_effect = failure

    response = client.post(
        "/agent/v1/conversations/321/replies",
        headers=SEND_HEADERS,
        json={
            "body": "private customer body",
            "body_format": "text",
            "expected_latest_thread_id": 40,
            "idempotency_key": "typed-error-key",
        },
    )

    _assert_error(
        response,
        status=status,
        code=code,
        message=message,
        retryable=retryable,
    )
    assert "private" not in response.text


def test_write_conflict_exposes_only_validated_current_version(gateway_client):
    client, _, gateway = gateway_client
    gateway.save_draft.side_effect = ConversationGatewayError(
        "private body and auth=secret",
        code="draft_conflict",
        status=409,
        details={
            "current_version": "a" * 64,
            "thread_id": 701,
            "payload": "private body",
            "api_key": "secret",
        },
    )

    response = client.put(
        "/agent/v1/conversations/321/draft",
        headers=DRAFT_HEADERS,
        json={
            "body": "private body",
            "body_format": "text",
            "expected_version": None,
            "idempotency_key": "conflict-details",
        },
    )

    assert response.status_code == 409
    assert response.json()["error"] == {
        "code": "draft_conflict",
        "message": "The draft changed after it was read",
        "retryable": False,
        "details": {"current_version": "a" * 64, "thread_id": 701},
    }
    assert "private" not in response.text
    assert "secret" not in response.text


def test_unexpected_write_failure_logs_no_exception_text_or_request_body(
    gateway_client, caplog
):
    client, _, gateway = gateway_client
    sensitive_body = "private-message-body-never-log"
    sensitive_error = "private-exception-text-never-log"
    gateway.send_reply.side_effect = RuntimeError(sensitive_error)

    with caplog.at_level(logging.ERROR, logger="api.agent"):
        response = client.post(
            "/agent/v1/conversations/321/replies",
            headers=SEND_HEADERS,
            json={
                "body": sensitive_body,
                "body_format": "text",
                "expected_latest_thread_id": 40,
                "idempotency_key": "unexpected-error",
            },
        )

    _assert_error(
        response,
        status=500,
        code="internal_error",
        message="Internal server error",
        retryable=True,
    )
    assert sensitive_body not in response.text
    assert sensitive_error not in response.text
    assert sensitive_body not in caplog.text
    assert sensitive_error not in caplog.text
    assert "exception_type=RuntimeError" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_write_response_validation_blocks_malformed_or_secret_results(
    gateway_client, caplog
):
    client, _, gateway = gateway_client
    sensitive = "private-warning-never-return-or-log"
    gateway.send_reply.return_value = {
        "action": "sent",
        "body_sha256": "a" * 64,
        "conversation_id": 321,
        "status": "sent_with_warning",
        "thread_id": 990,
        "warnings": [sensitive],
    }

    with caplog.at_level(logging.ERROR, logger="api.agent"):
        response = client.post(
            "/agent/v1/conversations/321/replies",
            headers=SEND_HEADERS,
            json={
                "body": "Hello",
                "body_format": "text",
                "expected_latest_thread_id": 40,
                "idempotency_key": "malformed-response",
            },
        )

    _assert_error(
        response,
        status=500,
        code="internal_error",
        message="Internal server error",
        retryable=True,
    )
    assert sensitive not in response.text
    assert sensitive not in caplog.text


def test_same_write_request_replay_returns_same_validated_result(gateway_client):
    client, _, gateway = gateway_client
    result = {
        "action": "sent",
        "body_sha256": "a" * 64,
        "conversation_id": 321,
        "status": "sent",
        "thread_id": 990,
        "warnings": [],
    }
    gateway.send_reply.return_value = result
    payload = {
        "body": "Hello",
        "body_format": "text",
        "expected_latest_thread_id": 40,
        "idempotency_key": "replayed-api-key",
    }

    first = client.post(
        "/agent/v1/conversations/321/replies",
        headers=SEND_HEADERS,
        json=payload,
    )
    second = client.post(
        "/agent/v1/conversations/321/replies",
        headers=SEND_HEADERS,
        json=payload,
    )

    assert first.status_code == second.status_code == 200
    assert first.json()["result"] == second.json()["result"] == result
    assert gateway.send_reply.await_count == 2


def test_same_key_different_request_conflict_is_canonical(gateway_client):
    client, _, gateway = gateway_client
    gateway.send_reply.side_effect = AgentOperationConflictError("same-key")

    response = client.post(
        "/agent/v1/conversations/321/replies",
        headers=SEND_HEADERS,
        json={
            "body": "Different body",
            "body_format": "text",
            "expected_latest_thread_id": 40,
            "idempotency_key": "same-key",
        },
    )

    _assert_error(
        response,
        status=409,
        code="idempotency_conflict",
        message="Idempotency key conflict",
    )


def test_operation_get_is_owner_only_and_returns_sanitized_public_record(
    agent_client, tmp_path, monkeypatch
):
    client, _ = agent_client
    db_path = tmp_path / "operations.db"
    init_db(db_path)
    monkeypatch.setattr(agent_api, "get_supa_db_path", lambda: str(db_path))
    owner = AgentIdentity(
        token_id="token-draft-789",
        name="draft-writer",
        scopes=frozenset({"draft"}),
    )
    reserve_operation(
        db_path=db_path,
        idempotency_key="owner-operation-key",
        identity=owner,
        scope="draft",
        operation_name="draft.save",
        target_id=321,
        request_id="req_owner_operation",
        request={"body": "Hello", "body_format": "text"},
        request_metadata={
            "body_sha256": "a" * 64,
            "expected_version": None,
        },
    )

    owned = client.get(
        "/agent/v1/operations/owner-operation-key",
        headers=DRAFT_HEADERS,
    )
    denied = client.get(
        "/agent/v1/operations/owner-operation-key",
        headers=READ_HEADERS,
    )

    assert owned.status_code == 200
    assert owned.headers["cache-control"] == "no-store"
    result = owned.json()["result"]
    assert result["idempotency_key"] == "owner-operation-key"
    assert result["operation_name"] == "draft.save"
    assert result["target_id"] == "321"
    assert result["state"] == "reserved"
    assert "token_id" not in result
    assert "token_name" not in result
    _assert_error(
        denied,
        status=404,
        code="operation_not_found",
        message="Operation not found",
    )


def test_operation_get_requires_authentication_without_touching_store(
    agent_client, monkeypatch
):
    client, _ = agent_client
    operation_get = Mock()
    monkeypatch.setattr(agent_api, "get_operation", operation_get)

    response = client.get("/agent/v1/operations/owner-operation-key")

    _assert_error(
        response,
        status=401,
        code="authentication_failed",
        message="Authentication failed",
    )
    operation_get.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "status", "code", "message", "retryable"),
    [
        (
            AgentOperationStorageError(),
            503,
            "operation_storage_error",
            "Operation store is unavailable",
            True,
        ),
        (
            AgentOperationDataError(),
            500,
            "operation_data_error",
            "Stored operation data is invalid",
            False,
        ),
    ],
)
def test_operation_get_maps_typed_storage_and_data_failures(
    agent_client,
    monkeypatch,
    failure,
    status,
    code,
    message,
    retryable,
):
    client, _ = agent_client
    monkeypatch.setattr(agent_api, "get_operation", Mock(side_effect=failure))

    response = client.get(
        "/agent/v1/operations/owner-operation-key",
        headers=DRAFT_HEADERS,
    )

    _assert_error(
        response,
        status=status,
        code=code,
        message=message,
        retryable=retryable,
    )


def test_operation_unknown_resolution_is_not_exposed_remotely(agent_client):
    client, _ = agent_client

    response = client.post(
        "/agent/v1/operations/unknown-operation-key",
        headers=SEND_HEADERS,
        json={"outcome": "succeeded", "reason": "remote resolution"},
    )

    _assert_error(
        response,
        status=405,
        code="method_not_allowed",
        message="Method not allowed",
    )


@pytest.mark.parametrize(
    ("method", "path", "headers", "payload"),
    [
        (
            "PUT",
            "/agent/v1/conversations/321/draft/",
            DRAFT_HEADERS,
            {
                "body": "Hello",
                "body_format": "text",
                "expected_version": None,
                "idempotency_key": "no-redirect-draft",
            },
        ),
        (
            "POST",
            "/agent/v1/conversations/321/replies/",
            SEND_HEADERS,
            {
                "body": "Hello",
                "body_format": "text",
                "expected_latest_thread_id": 40,
                "idempotency_key": "no-redirect-reply",
            },
        ),
    ],
)
def test_write_route_trailing_slashes_never_redirect(
    agent_client, method, path, headers, payload
):
    client, _ = agent_client

    response = client.request(
        method,
        path,
        headers=headers,
        json=payload,
        follow_redirects=False,
    )

    _assert_error(
        response,
        status=404,
        code="not_found",
        message="Resource not found",
    )
    assert "location" not in response.headers


def test_attachment_returns_raw_bytes_with_safe_download_headers(
    gateway_client, tmp_path
):
    client, _, gateway = gateway_client
    untrusted_output_path = tmp_path / "server-must-not-write-here"
    gateway.attachment.return_value = AttachmentData(
        conversation_id=321,
        attachment_id=701,
        filename='..\\ＦＯＯ／résumé"\r\nInjected.txt',
        data=b"synthetic-image-bytes",
        claimed_mime_type="image/png",
        size=21,
    )

    response = client.get(
        "/agent/v1/conversations/321/attachments/701",
        params={"output": str(untrusted_output_path)},
        headers=READ_HEADERS,
    )

    assert response.status_code == 200
    assert response.content == b"synthetic-image-bytes"
    assert response.headers["cache-control"] == "no-store"
    assert REQUEST_ID_PATTERN.fullmatch(response.headers["x-request-id"])
    assert response.headers["x-supa-api-version"] == API_VERSION
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    disposition = response.headers["content-disposition"]
    assert disposition.startswith('attachment; filename="')
    assert "; filename*=UTF-8''" in disposition
    assert disposition.count('"') == 2
    assert "\\" not in disposition
    assert "／" not in disposition
    assert "\r" not in disposition
    assert "\n" not in disposition
    assert "%0D" not in disposition.upper()
    assert "%0A" not in disposition.upper()
    assert not untrusted_output_path.exists()
    gateway.attachment.assert_awaited_once()
    assert gateway.attachment.await_args.args == (321, 701)
    assert gateway.attachment.await_args.kwargs["identity"].conversations is None


def test_attachment_success_reuses_the_boundary_request_id(
    gateway_client, monkeypatch
):
    client, _, gateway = gateway_client
    generated_ids: list[str] = []

    def next_request_id():
        request_id = f"req_{len(generated_ids) + 1:032x}"
        generated_ids.append(request_id)
        return request_id

    monkeypatch.setattr(agent_api, "_new_request_id", next_request_id)
    gateway.attachment.return_value = AttachmentData(
        conversation_id=321,
        attachment_id=701,
        filename="screen.png",
        data=b"image",
        claimed_mime_type="image/png",
        size=5,
    )

    response = client.get(
        "/agent/v1/conversations/321/attachments/701",
        headers=READ_HEADERS,
    )

    assert response.status_code == 200
    assert generated_ids == [f"req_{1:032x}"]
    assert response.headers["x-request-id"] == generated_ids[0]
    assert response.headers["x-supa-api-version"] == API_VERSION


def test_attachment_error_keeps_one_json_envelope_request_id(
    gateway_client, monkeypatch
):
    client, _, gateway = gateway_client
    generated_ids: list[str] = []

    def next_request_id():
        request_id = f"req_{len(generated_ids) + 1:032x}"
        generated_ids.append(request_id)
        return request_id

    monkeypatch.setattr(agent_api, "_new_request_id", next_request_id)
    gateway.attachment.side_effect = ConversationGatewayError(
        "private attachment detail",
        code="attachment_not_found",
        status=404,
    )

    response = client.get(
        "/agent/v1/conversations/321/attachments/701",
        headers=READ_HEADERS,
    )

    body = _assert_error(
        response,
        status=404,
        code="attachment_not_found",
        message="Attachment not found",
    )
    assert generated_ids == [body["request_id"]]
    assert "x-request-id" not in response.headers
    assert "x-supa-api-version" not in response.headers


def test_attachment_rejects_untrusted_content_type(gateway_client):
    client, _, gateway = gateway_client
    gateway.attachment.return_value = AttachmentData(
        conversation_id=321,
        attachment_id=701,
        filename="report.html",
        data=b"<script>not executed</script>",
        claimed_mime_type="text/html\r\nX-Injected: yes",
        size=29,
    )

    response = client.get(
        "/agent/v1/conversations/321/attachments/701",
        headers=READ_HEADERS,
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert "x-injected" not in response.headers


@pytest.mark.parametrize(
    ("failure", "status", "code", "message", "retryable"),
    [
        (
            ConversationGatewayError(
                "private gateway detail token=secret",
                code="invalid_cursor",
                status=400,
            ),
            400,
            "invalid_cursor",
            "The conversation cursor is invalid.",
            False,
        ),
        (
            HelpScoutUpstreamError(
                "private 404 body with token=secret",
                status=404,
                code="upstream_http_error",
                details={"endpoint": "/private?access_token=secret"},
            ),
            404,
            "not_found",
            "Resource not found",
            False,
        ),
        (
            HelpScoutAuthError(
                "private auth body",
                status=401,
                code="upstream_auth_error",
                details={"endpoint": "/private"},
            ),
            502,
            "upstream_auth_error",
            "Upstream authentication failed",
            False,
        ),
        (
            HelpScoutAuthError(
                "private auth body",
                status=403,
                code="upstream_auth_error",
            ),
            502,
            "upstream_auth_error",
            "Upstream authentication failed",
            False,
        ),
        (
            HelpScoutAuthError(
                "private OAuth endpoint body",
                status=404,
                code="oauth_token_error",
            ),
            502,
            "upstream_auth_error",
            "Upstream authentication failed",
            False,
        ),
        (
            HelpScoutAuthError(
                "private OAuth precondition body",
                status=412,
                code="oauth_token_error",
            ),
            502,
            "upstream_auth_error",
            "Upstream authentication failed",
            False,
        ),
        (
            HelpScoutAuthError(
                "private OAuth rate-limit body",
                status=429,
                code="oauth_token_error",
                retryable=True,
            ),
            429,
            "upstream_auth_error",
            "Upstream authentication failed",
            True,
        ),
        (
            HelpScoutUpstreamError(
                "private lock body",
                status=412,
                code="upstream_http_error",
            ),
            412,
            "conversation_locked",
            "Conversation is locked upstream",
            False,
        ),
        (
            HelpScoutTransportError(
                "private timeout detail",
                code="request_timeout",
                retryable=True,
            ),
            504,
            "upstream_timeout",
            "Upstream service timed out",
            True,
        ),
        (
            HelpScoutTransportError(
                "private network detail",
                code="network_error",
                retryable=True,
            ),
            502,
            "upstream_error",
            "Upstream service failed",
            True,
        ),
        (
            HelpScoutUpstreamError(
                "private rate-limit body",
                status=429,
                code="upstream_http_error",
                retryable=True,
            ),
            429,
            "rate_limited",
            "Too many requests",
            True,
        ),
        (
            HelpScoutUpstreamError(
                "private unavailable body",
                status=503,
                code="upstream_http_error",
                retryable=True,
            ),
            503,
            "upstream_error",
            "Upstream service failed",
            True,
        ),
        (
            HelpScoutAuthError(
                "private OAuth service failure",
                status=503,
                code="oauth_token_error",
                retryable=True,
            ),
            503,
            "upstream_auth_error",
            "Upstream authentication failed",
            True,
        ),
        (
            HelpScoutProtocolError(
                "private malformed body",
                code="invalid_json",
                details={"endpoint": "/private"},
            ),
            502,
            "upstream_error",
            "Upstream service failed",
            False,
        ),
    ],
)
def test_typed_gateway_failures_use_safe_canonical_agent_envelopes(
    gateway_client, failure, status, code, message, retryable
):
    client, _, gateway = gateway_client
    gateway.get.side_effect = failure

    response = client.get("/agent/v1/conversations/321", headers=READ_HEADERS)

    _assert_error(
        response,
        status=status,
        code=code,
        message=message,
        retryable=retryable,
    )
    assert "private" not in response.text
    assert "secret" not in response.text
    assert "access_token" not in response.text
    if isinstance(failure, HelpScoutAuthError):
        assert "www-authenticate" not in response.headers
    gateway.get.assert_awaited_once_with(321)


def test_revoked_read_token_cannot_reach_conversation_gateway(gateway_client):
    client, registry_path, gateway = gateway_client
    write_token_records(
        registry_path,
        [
            _token_record(
                "synthetic-no-read-token",
                token_id="token-no-read-456",
                name="publisher",
                scopes=("send",),
            )
        ],
    )

    response = client.get("/agent/v1/conversations/321", headers=READ_HEADERS)

    _assert_error(
        response,
        status=401,
        code="authentication_failed",
        message="Authentication failed",
    )
    gateway.get.assert_not_awaited()


def test_json_routes_declare_pydantic_response_models():
    expected_routes = {
        ("GET", "/agent/v1/conversations"),
        ("GET", "/agent/v1/conversations/{conversation_id}"),
        ("GET", "/agent/v1/conversations/{conversation_id}/draft"),
        ("PUT", "/agent/v1/conversations/{conversation_id}/draft"),
        ("POST", "/agent/v1/conversations/{conversation_id}/draft/send"),
        ("POST", "/agent/v1/conversations/{conversation_id}/replies"),
        ("POST", "/agent/v1/conversations/{conversation_id}/notes"),
        ("GET", "/agent/v1/operations/{idempotency_key}"),
    }
    matching = {
        (method, route.path): route
        for route in app.router.routes
        for method in getattr(route, "methods", set())
        if (method, getattr(route, "path", None)) in expected_routes
    }

    assert set(matching) == expected_routes
    assert all(route.response_model is not None for route in matching.values())


def test_all_agent_response_models_are_strict_and_extra_forbid():
    response_models = (
        agent_api.AgentSuccessEnvelope,
        agent_api.PersonResponse,
        agent_api.ThreadAuthorResponse,
        agent_api.ThreadSourceResponse,
        agent_api.AttachmentMetadataResponse,
        agent_api.ThreadResponse,
        agent_api.DraftResponse,
        agent_api.MailboxResponse,
        agent_api.ConversationTimestampsResponse,
        agent_api.ConversationResponse,
        agent_api.ConversationListResponse,
        agent_api.DraftReadResponse,
        agent_api.DraftWriteResponse,
        agent_api.SendWriteResponse,
        agent_api.OperationResponse,
    )

    for response_model in response_models:
        assert response_model.model_config["strict"] is True
        assert response_model.model_config["extra"] == "forbid"

    request_models = (
        agent_api.DraftWriteRequest,
        agent_api.DraftSendRequest,
        agent_api.ReplySendRequest,
    )
    for request_model in request_models:
        assert request_model.model_config["strict"] is True
        assert request_model.model_config["extra"] == "forbid"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("id", True),
        lambda value: value.__setitem__("was_merged", 1),
        lambda value: value["threads"][0].__setitem__("draft", 0),
        lambda value: value["threads"][0]["attachments"][0].__setitem__(
            "size", "2048"
        ),
        lambda value: value["threads"][0]["attachments"][0].__setitem__(
            "is_image", 1
        ),
        lambda value: value.__setitem__("latest_actionable_thread_id", 0),
    ],
)
def test_invalid_normalized_types_fail_safely_instead_of_being_coerced(
    gateway_client, mutate
):
    client, _, gateway = gateway_client
    malformed = copy.deepcopy(_normalized_conversation())
    mutate(malformed)
    malformed["subject"] = "private invalid normalized data"
    gateway.get.return_value = malformed

    response = client.get("/agent/v1/conversations/321", headers=READ_HEADERS)

    _assert_error(
        response,
        status=500,
        code="internal_error",
        message="Internal server error",
        retryable=True,
    )
    assert "private invalid normalized data" not in response.text


def test_invalid_normalized_value_is_not_logged(gateway_client, caplog):
    client, _, gateway = gateway_client
    malformed = copy.deepcopy(_normalized_conversation())
    private_value = "private-size-value-that-must-not-be-logged"
    malformed["threads"][0]["attachments"][0]["size"] = private_value
    gateway.get.return_value = malformed

    with caplog.at_level(logging.ERROR, logger="api.agent"):
        response = client.get(
            "/agent/v1/conversations/321",
            headers=READ_HEADERS,
        )

    assert response.status_code == 500
    assert private_value not in response.text
    assert private_value not in caplog.text


@pytest.fixture
def isolated_production_gateway(monkeypatch):
    cache_clear = getattr(agent_api.get_conversation_gateway, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    monkeypatch.setattr(agent_api, "_conversation_gateway", None, raising=False)
    yield
    cache_clear = getattr(agent_api.get_conversation_gateway, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


@pytest.mark.asyncio
async def test_closing_unused_production_gateway_does_not_create_client(
    isolated_production_gateway, monkeypatch
):
    client_factory = Mock()
    monkeypatch.setattr(agent_api, "HelpScoutClient", client_factory)

    close_gateway = getattr(agent_api, "close_conversation_gateway", None)
    assert close_gateway is not None
    await close_gateway()

    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_production_gateway_is_shared_and_closed_once(
    isolated_production_gateway, monkeypatch
):
    client = SimpleNamespace(close=AsyncMock())
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(agent_api, "HelpScoutClient", client_factory)

    first = agent_api.get_conversation_gateway()
    second = agent_api.get_conversation_gateway()
    assert first is second
    assert first.db_path == agent_api.get_supa_db_path()
    assert first.post_send_bookkeeper is agent_api._post_send_learning_callback

    await agent_api.close_conversation_gateway()

    client_factory.assert_called_once()
    client.close.assert_awaited_once_with()


def test_production_learning_callback_is_lazy_and_uses_existing_processor(
    monkeypatch
):
    processor = SimpleNamespace(index_sent_reply=Mock(return_value=True))
    monkeypatch.setattr(
        "services.processor.get_processor",
        Mock(return_value=processor),
    )
    log_ticket = Mock()
    monkeypatch.setattr(dashboard_api, "log_ticket", log_ticket)

    result = agent_api._post_send_learning_callback(
        conversation_id=321,
        thread_id=990,
        subject="Subject",
        customer_message="Customer question",
        reply_html="<p>Answer</p>",
        request_id="req_learning_callback",
    )

    assert result is True
    processor.index_sent_reply.assert_called_once_with(
        conversation_id=321,
        subject="Subject",
        customer_message="Customer question",
        reply_text="<p>Answer</p>",
        theme_tag=None,
    )
    log_ticket.assert_not_called()


@pytest.mark.asyncio
async def test_repeated_production_gateway_close_is_idempotent(
    isolated_production_gateway, monkeypatch
):
    client = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(agent_api, "HelpScoutClient", lambda **_kwargs: client)
    agent_api.get_conversation_gateway()

    await agent_api.close_conversation_gateway()
    await agent_api.close_conversation_gateway()

    client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_production_gateway_recreates_after_close(
    isolated_production_gateway, monkeypatch
):
    clients = [
        SimpleNamespace(close=AsyncMock()),
        SimpleNamespace(close=AsyncMock()),
    ]
    client_factory = Mock(side_effect=clients)
    monkeypatch.setattr(agent_api, "HelpScoutClient", client_factory)

    first = agent_api.get_conversation_gateway()
    await agent_api.close_conversation_gateway()
    second = agent_api.get_conversation_gateway()

    assert first is not second
    assert first.client is clients[0]
    assert second.client is clients[1]
    clients[0].close.assert_awaited_once_with()
    clients[1].close.assert_not_awaited()


@pytest.mark.parametrize("path", ["draft/send", "replies", "draft/schedule"])
@pytest.mark.parametrize("code,given_thread_id", [
    ("wrong_expected_thread", 60),
    ("conversation_changed", 39),
])
def test_expected_thread_failures_expose_safe_ids_and_actionable_message(
    gateway_client, monkeypatch, path, code, given_thread_id
):
    client, _, gateway = gateway_client
    details = {"latest_actionable_thread_id": 40, "given_thread_id": given_thread_id}
    if code == "wrong_expected_thread":
        details["draft_thread_id"] = 60
        message = (
            "--expected-thread must be the latest actionable thread "
            "(the customer's latest message), not the draft's thread."
        )
    else:
        details["thread_id"] = 40
        message = "The latest actionable thread is now 40, not 39."
    failure = ConversationGatewayError(
        "private body and auth=secret", code=code, status=409,
        details={**details, "payload": "private body", "api_key": "secret"},
    )
    if path == "draft/schedule":
        monkeypatch.setattr(agent_api.scheduled_sends, "schedule_send", AsyncMock(side_effect=failure))
    elif path == "draft/send":
        gateway.send_draft.side_effect = failure
    else:
        gateway.send_reply.side_effect = failure
    payload = {
        "expected_latest_thread_id": given_thread_id,
        "idempotency_key": "wrong-thread-precondition",
    }
    if path == "draft/send":
        payload["expected_version"] = "b" * 64
    else:
        payload.update({"body": "Hello", "body_format": "text"})

    response = client.post(
        f"/agent/v1/conversations/321/{path}", headers=SEND_HEADERS, json=payload,
    )

    assert response.status_code == 409
    assert response.json()["error"] == {
        "code": code, "message": message, "retryable": False, "details": details,
    }
    assert "private" not in response.text
    assert "secret" not in response.text


def test_read_only_get_sees_body_free_booked_reply_without_send_authority(gateway_client, monkeypatch):
    client, _, gateway = gateway_client
    gateway.get.return_value = _normalized_conversation()
    row={"status":"queued","send_at":"2026-10-05T15:00:00Z","hold_reason":None,
         "body":"PRIVATE DRAFT","cc":["private@example.com"],"token_name":"private-owner"}
    monkeypatch.setattr(agent_api.scheduled_sends,"latest_for_conversation",lambda *_:row)
    response=client.get("/agent/v1/conversations/123",headers=LIMITED_HEADERS)
    assert response.status_code==200
    assert response.json()["result"]["scheduled_send"]=={key:row[key] for key in ("status","send_at","hold_reason")}
    assert "PRIVATE DRAFT" not in response.text and "private@example.com" not in response.text
    assert client.get("/agent/v1/conversations/123/draft/schedule",headers=LIMITED_HEADERS).status_code==403
    assert client.post("/agent/v1/conversations/123/draft/schedule",headers=LIMITED_HEADERS,json={}).status_code==403


def test_read_only_get_fails_closed_when_booking_observation_is_unavailable(gateway_client, monkeypatch):
    client, _, gateway=gateway_client
    gateway.get.return_value=_normalized_conversation()
    def failed(*_):raise OSError("private storage unavailable")
    monkeypatch.setattr(agent_api.scheduled_sends,"latest_for_conversation",failed)
    response=client.get("/agent/v1/conversations/123",headers=LIMITED_HEADERS)
    assert response.status_code==503
    assert response.json()["error"]["code"]=="schedule_observation_unavailable"
    assert "private storage" not in response.text


def test_read_rate_limit_envelope_exposes_only_safe_retry_delay():
    import json
    from services.helpscout import HelpScoutUpstreamError
    exc = HelpScoutUpstreamError('SECRET', status=429, retryable=True,
                                details={'retry_after_seconds':37,'endpoint':'SECRET'})
    response = agent_api._read_error_response(SimpleNamespace(state=SimpleNamespace(agent_request_id='request')), exc)
    error = json.loads(response.body)['error']
    assert error['details'] == {'retry_after_seconds':37}
    assert 'SECRET' not in response.body.decode()


def test_mailbox_summary_skips_vendor_and_requires_unrestricted_read(agent_client, monkeypatch):
    client, registry = agent_client
    monkeypatch.delenv("SUPA_MAILBOX_ORIGIN", raising=False)
    summary = AsyncMock(return_value={"writer": "paused", "issues": {"Agent event": 3}})
    gateway = Mock(return_value=SimpleNamespace(client=SimpleNamespace(native_operations=summary)))
    monkeypatch.setattr(agent_api, "get_conversation_gateway", gateway)
    path = "/agent/v1/mailbox/operations"
    write_token_records(registry, [{**_token_record(LIMITED_TOKEN, token_id="restricted", name="restricted"),
                                    "conversations": [321]}, _token_record(READ_TOKEN)])
    assert client.get(path, headers=LIMITED_HEADERS).status_code == 403
    assert client.get(path, headers=READ_HEADERS).json()["result"] == {"provider": "vendor"}
    gateway.assert_not_called()
    monkeypatch.setenv("SUPA_MAILBOX_ORIGIN", "https://mailbox.woodland.example")
    assert client.get(path, headers=READ_HEADERS).json()["result"] == {
        "provider": "native", "writer": "paused", "issues": {"Agent event": 3}}
    summary.side_effect = ValueError("private data")
    response = client.get(path, headers=READ_HEADERS)
    assert response.status_code == 503 and "private data" not in response.text


def test_native_operations_accepts_only_a_fixed_body_free_summary(monkeypatch):
    monkeypatch.setattr("services.installation.OPERATIONS_ORIGIN", "https://mailbox.woodland.example")
    import httpx
    from services.helpscout import HelpScoutClient
    async def run():
        client = HelpScoutClient(None, None, environ={
            "SUPA_MAILBOX_ORIGIN": "https://mailbox.woodland.example",
            "SUPA_MAILBOX_CLIENT_ID": "fixture", "SUPA_MAILBOX_CLIENT_SECRET": "fixture"})
        try:
            async def read(payload):
                method = AsyncMock(return_value=httpx.Response(200, json=payload))
                monkeypatch.setattr(client, "_read_bounded", method)
                result = await client.native_operations()
                method.assert_awaited_once_with("/operations", max_bytes=8192)
                return result
            assert await read({"writer": "live", "issues": {}}) == {"writer": "live", "issues": {}}
            for issues in ({"Private body": 4}, {"Agent event": True}, {"Acknowledgment": -1}, []):
                with pytest.raises(ValueError):
                    await read({"writer": "live", "issues": issues})
            with pytest.raises(ValueError):
                await read({"writer": "live", "issues": {}, "body": "private"})
            client.BASE_URL = "https://api.helpscout.net/v2"
            with pytest.raises(ValueError):
                await client.native_operations()
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize(("method", "path", "options", "upstream", "params", "question"), [
    ("GET", "/desk?p=2", {}, "/desk", {"p": 2}, None),
    ("GET", "/mailbox/insights?months=3", {}, "/insights", {"months": 3}, None),
    ("GET", "/conversations/321/precedents", {}, "/precedents", {"conversation": 321}, None),
    ("POST", "/precedents", {"json": {"q": "a private question", "tag": "example"}}, "/precedents", {},
     {"q": "a private question", "tag": "example"}),
])
def test_mailbox_reports_require_unrestricted_read(gateway_client, method, path, options, upstream, params, question):
    client, registry, gateway = gateway_client
    write_token_records(registry, [_token_record(READ_TOKEN),
        {**_token_record(LIMITED_TOKEN, token_id="case", name="case"), "conversations": [321]},
        _token_record(SEND_TOKEN, token_id="writer", name="writer", scopes=("send",))])
    for headers, code in ((LIMITED_HEADERS, "conversation_not_allowed"), (SEND_HEADERS, "insufficient_scope")):
        refused = client.request(method, "/agent/v1" + path, headers=headers, **options)
        assert refused.status_code == 403 and refused.json()["error"]["code"] == code
    gateway.client.native_report.assert_not_called()
    gateway.client.native_report.return_value = {"example": 1}
    result = client.request(method, "/agent/v1" + path, headers=READ_HEADERS, **options)
    assert result.json()["result"] == {"example": 1}
    gateway.client.native_report.assert_awaited_once_with(upstream, params, question=question)


def test_brief_checks_both_ids_and_limits_case_context(gateway_client):
    client, registry, gateway = gateway_client
    write_token_records(registry, [_token_record(READ_TOKEN),
        {**_token_record(LIMITED_TOKEN, token_id="case", name="case"), "conversations": [321]}])
    path = "/agent/v1/conversations/321/brief"
    for url in (path.replace("321", "987"), path + "?precedents=1"):
        assert client.get(url, headers=LIMITED_HEADERS).json()["error"]["code"] == "conversation_not_allowed"
    read = gateway.client.native_report
    read.assert_not_called()
    own = {"id": 321, "messages": [{"text": "current conversation"}], "customer": {"name": "Reader"}}
    private = {**own, "customer": {"name": "Reader", "recent": ["private"], "emails": ["private"],
                                  "first_contact": "private", "conversations": 7}, "precedents": ["private"],
               "future_field": "private"}
    read.return_value = private
    assert client.get(path, headers=LIMITED_HEADERS).json()["result"] == own
    assert client.get(path + "?max_chars=1000&quotes=true&precedents=2", headers=READ_HEADERS).json()["result"] == private
    read.assert_awaited_with("/conversations/321/brief", {"max_chars": 1000, "quotes": 1, "precedents": 2})
    read.return_value = {**private, "id": 987}
    assert client.get(path, headers=LIMITED_HEADERS).json()["error"]["code"] == "conversation_not_allowed"


@pytest.mark.asyncio
async def test_native_report_transport_keeps_private_query_out_of_urls_and_errors(monkeypatch, caplog):
    import httpx
    from services.helpscout import HelpScoutClient, HelpScoutError
    client = HelpScoutClient(None, None, environ={"SUPA_MAILBOX_ORIGIN": "https://mail.example.test",
        "SUPA_MAILBOX_CLIENT_ID": "fixture", "SUPA_MAILBOX_CLIENT_SECRET": "fixture"})
    await client._http.aclose()
    question = {"q": "private lookup phrase", "tag": "private tag"}
    def handler(request):
        assert request.url.path == "/v2/precedents" and not request.url.query
        assert request.method == "POST" and json.loads(request.content) == question
        return httpx.Response(503, text="private upstream response")
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(client, "_ensure_read_token", AsyncMock(return_value="fixture"))
    try:
        with caplog.at_level("INFO"), pytest.raises(HelpScoutError) as failure:
            await client.native_report("/precedents", {}, question=question)
        assert failure.value.status == 503
        for text in (*question.values(), "private upstream response"):
            assert text not in caplog.text and text not in str(failure.value.details)
    finally:
        await client.close()


def test_report_bounds_and_upstream_failures_use_safe_envelopes(gateway_client):
    client, _, gateway = gateway_client
    for path in ("/desk?p=0", "/mailbox/insights?months=4", "/conversations/321/brief?max_chars=999",
                 "/conversations/321/brief?precedents=11"):
        assert client.get("/agent/v1" + path, headers=READ_HEADERS).status_code == 422
    gateway.client.native_report.assert_not_called()
    gateway.client.native_report.side_effect = HelpScoutProtocolError(
        "private mailbox content", code="invalid_json", details={"body": "private mailbox content"})
    for path in ("/desk", "/conversations/321/brief"):
        response = client.get("/agent/v1" + path, headers=READ_HEADERS)
        assert response.status_code == 502 and "private mailbox content" not in response.text
        assert response.json()["request_id"] and response.headers["cache-control"] == "no-store"
