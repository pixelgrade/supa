import base64
import hashlib
import hmac
import json
import logging
import sqlite3
from types import SimpleNamespace

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from db.database import init_db
from services.agent_events import AgentEventStorageError
from api.webhooks import process_in_background
from api.webhooks import handle_customer_reply
from main import app
from tests.lease_recording import RecordingLeaseFactory


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_webhook_journal(tmp_path, monkeypatch):
    db_path = str(tmp_path / "webhook-events.db")
    init_db(db_path)
    monkeypatch.setattr("api.webhooks.DEFAULT_DB_PATH", db_path)
    monkeypatch.delenv("HELPSCOUT_WEBHOOK_SECRET", raising=False)
    return db_path


def _payload(ticket_id=8765):
    return {
        "id": ticket_id,
        "subject": "Need help with café typography",
        "primaryCustomer": {"first": "Reader"},
        "tags": [],
        "createdAt": "2026-09-25T12:00:00Z",
        "updatedAt": "2026-09-25T12:01:00Z",
        "_embedded": {
            "threads": [
                {
                    "id": ticket_id + 100,
                    "type": "customer",
                    "body": "The café heading is wrong ☕",
                    "createdAt": "2026-09-25T12:01:00Z",
                    "createdBy": {"type": "customer"},
                    "source": {"type": "email"},
                }
            ]
        },
    }


@pytest.mark.asyncio
async def test_customer_reply_tag_removal_uses_guarded_conversation_lease(monkeypatch):
    payload = _payload(8766)
    payload["tags"] = [{"tag": "supa-processed"}]
    remove_tag = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "api.webhooks.get_processor",
        lambda: SimpleNamespace(helpscout=SimpleNamespace(remove_tag=remove_tag)),
    )
    leases = RecordingLeaseFactory()
    monkeypatch.setattr("api.webhooks.conversation_lease", leases)
    background_tasks = SimpleNamespace(add_task=lambda *_args, **_kwargs: None)

    response = await handle_customer_reply(payload, background_tasks)

    assert response.status == "queued"
    leases.assert_guarded_count(8766, 1)
    remove_tag.assert_awaited_once()


def _signature(secret, body):
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


@pytest.mark.parametrize("supplied_signature", [None, "not-a-valid-signature"])
def test_signed_webhook_rejects_missing_or_invalid_signature_before_json(
    monkeypatch,
    supplied_signature,
):
    monkeypatch.setenv("HELPSCOUT_WEBHOOK_SECRET", "webhook-test-secret")
    headers = {
        "Content-Type": "application/json",
        "X-HelpScout-Event": "convo.created",
    }
    if supplied_signature is not None:
        headers["X-HelpScout-Signature"] = supplied_signature

    with patch("api.webhooks.append_event") as append, patch(
        "api.webhooks.process_in_background", new_callable=AsyncMock
    ) as process:
        response = client.post(
            "/webhook/helpscout",
            content=b"this is not json",
            headers=headers,
        )

    assert response.status_code == 401
    append.assert_not_called()
    process.assert_not_awaited()


def test_signed_webhook_authenticates_exact_unicode_body_bytes(monkeypatch):
    secret = "webhook-test-secret"
    monkeypatch.setenv("HELPSCOUT_WEBHOOK_SECRET", secret)
    payload = _payload()
    raw_body = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-HelpScout-Event": "convo.created",
        "X-HelpScout-Signature": _signature(secret, raw_body),
    }

    with patch(
        "api.webhooks.process_in_background", new_callable=AsyncMock
    ) as process:
        accepted = client.post(
            "/webhook/helpscout",
            content=raw_body,
            headers=headers,
        )
        escaped_body = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        rejected = client.post(
            "/webhook/helpscout",
            content=escaped_body,
            headers=headers,
        )

    assert accepted.status_code == 200
    assert accepted.json()["status"] == "queued"
    assert rejected.status_code == 401
    process.assert_awaited_once()


def test_webhook_logs_never_include_secret_signature_or_body(monkeypatch, caplog):
    secret = "private-webhook-secret-never-log"
    payload = _payload()
    payload["_embedded"]["threads"][0]["body"] = "private-body-never-log"
    raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = _signature(secret, raw_body)
    monkeypatch.setenv("HELPSCOUT_WEBHOOK_SECRET", secret)
    caplog.set_level(logging.INFO, logger="api.webhooks")

    with patch("api.webhooks.process_in_background", new_callable=AsyncMock):
        response = client.post(
            "/webhook/helpscout",
            content=raw_body,
            headers={
                "Content-Type": "application/json",
                "X-HelpScout-Event": "convo.created",
                "X-HelpScout-Signature": signature,
            },
        )

    assert response.status_code == 200
    webhook_logs = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "api.webhooks"
    )
    assert secret not in webhook_logs
    assert signature not in webhook_logs
    assert "private-body-never-log" not in webhook_logs


def test_malformed_json_is_rejected_without_journal_or_routing():
    with patch("api.webhooks.append_event") as append, patch(
        "api.webhooks.process_in_background", new_callable=AsyncMock
    ) as process:
        response = client.post(
            "/webhook/helpscout",
            content=b"[]",
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 400
    append.assert_not_called()
    process.assert_not_awaited()


def test_journal_failure_returns_503_before_any_routing():
    with patch(
        "api.webhooks.append_event",
        side_effect=AgentEventStorageError("journal unavailable"),
    ) as append, patch("api.webhooks.extensions.intercept_webhook") as intercept, patch(
        "api.webhooks.is_supa_ignored"
    ) as ignored, patch(
        "api.webhooks.process_in_background", new_callable=AsyncMock
    ) as process, patch(
        "api.webhooks.handle_agent_reply", new_callable=AsyncMock
    ) as learning:
        response = client.post(
            "/webhook/helpscout",
            json=_payload(),
            headers={"X-HelpScout-Event": "convo.created"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "The event journal is unavailable."}
    append.assert_called_once()
    intercept.assert_not_called()
    ignored.assert_not_called()
    process.assert_not_awaited()
    learning.assert_not_awaited()


def test_webhook_retries_append_one_deduplicated_event(isolated_webhook_journal):
    payload = _payload()
    with patch(
        "api.webhooks.process_in_background", new_callable=AsyncMock
    ) as process:
        first = client.post(
            "/webhook/helpscout",
            json=payload,
            headers={"X-HelpScout-Event": "convo.created"},
        )
        retry = client.post(
            "/webhook/helpscout",
            json=payload,
            headers={"X-HelpScout-Event": "convo.created"},
        )

    assert first.status_code == retry.status_code == 200
    with sqlite3.connect(isolated_webhook_journal) as db:
        assert db.execute("SELECT COUNT(*) FROM agent_events").fetchone()[0] == 1
    assert process.await_count == 2


@pytest.mark.parametrize(
    ("event_type", "normalized_type"),
    (
        ("convo.assigned", "helpscout.conversation.assigned"),
        ("convo.status", "helpscout.conversation.status.changed"),
        ("convo.tags", "helpscout.conversation.tags.changed"),
    ),
)
def test_metadata_events_are_journal_only(event_type, normalized_type):
    payload = _payload()
    payload["createdBy"] = {"id": 99, "type": "system_user"}
    payload["source"] = {"type": "api"}
    with patch("api.webhooks.append_event") as append, patch(
        "api.webhooks.extensions.intercept_webhook"
    ) as intercept, patch(
        "api.webhooks.process_in_background", new_callable=AsyncMock
    ) as process, patch(
        "api.webhooks.handle_agent_reply", new_callable=AsyncMock
    ) as learning:
        response = client.post(
            "/webhook/helpscout",
            json=payload,
            headers={"X-HelpScout-Event": event_type},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "journaled",
        "reason": "Metadata event",
        "ticket_id": 8765,
    }
    assert append.call_args.args[1].type == normalized_type
    intercept.assert_not_called()
    process.assert_not_awaited()
    learning.assert_not_awaited()


def test_support_and_learning_routes_run_after_journal_append():
    order = []

    def append_side_effect(*_args):
        order.append("journal")

    async def support_side_effect(*_args):
        order.append("support")

    async def learning_side_effect(*_args):
        order.append("learning")
        return {"status": "learned", "ticket_id": 8765}

    with patch("api.webhooks.append_event", side_effect=append_side_effect), patch(
        "api.webhooks.process_in_background",
        new_callable=AsyncMock,
        side_effect=support_side_effect,
    ), patch(
        "api.webhooks.handle_agent_reply",
        new_callable=AsyncMock,
        side_effect=learning_side_effect,
    ):
        support = client.post(
            "/webhook/helpscout",
            json=_payload(),
            headers={"X-HelpScout-Event": "convo.created"},
        )
        learned = client.post(
            "/webhook/helpscout",
            json=_payload(),
            headers={"X-HelpScout-Event": "convo.agent.reply.created"},
        )

    assert support.status_code == learned.status_code == 200
    assert order == ["journal", "support", "journal", "learning"]


@pytest.mark.parametrize(
    ("event_type", "expected_trigger"),
    (
        ("convo.customer.reply.created", "customer_reply"),
        ("convo.note.created", "note_trigger"),
    ),
)
def test_customer_and_note_work_are_scheduled_after_journal(
    event_type,
    expected_trigger,
):
    order = []
    payload = _payload()
    if event_type == "convo.note.created":
        payload["_embedded"]["threads"].append(
            {
                "id": 9999,
                "type": "note",
                "body": "Please investigate",
                "createdAt": "2026-09-25T12:02:00Z",
                "createdBy": {"type": "user"},
                "source": {"type": "web"},
            }
        )

    def append_side_effect(*_args):
        order.append("journal")

    async def process_side_effect(_payload, trigger):
        order.append(trigger)

    with patch("api.webhooks.append_event", side_effect=append_side_effect), patch(
        "api.webhooks.process_in_background",
        new_callable=AsyncMock,
        side_effect=process_side_effect,
    ) as process:
        response = client.post(
            "/webhook/helpscout",
            json=payload,
            headers={"X-HelpScout-Event": event_type},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert order == ["journal", expected_trigger]
    process.assert_awaited_once()


@patch("api.webhooks.process_in_background", new_callable=AsyncMock)
def test_webhook_accepts_customer_conversation_from_api_source(mock_process):
    """Customer-authored conversations can legitimately arrive with API source metadata."""
    payload = {
        "id": 123,
        "subject": "Test",
        "primaryCustomer": {"first": "Reader"},
        "tags": [],
        "_embedded": {
            "threads": [{
                "type": "customer",
                "body": "Hello",
                "createdBy": {"type": "customer"},
                "source": {"type": "api"},
            }]
        }
    }

    response = client.post("/webhook/helpscout", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["ticket_id"] == 123


@patch('api.webhooks.process_in_background', new_callable=AsyncMock)
def test_webhook_accepts_valid_ticket(mock_process):
    """Should queue valid support ticket for processing."""
    payload = {
        "id": 456,
        "subject": "Help needed",
        "primaryCustomer": {"first": "Reader", "last": "Oak"},
        "tags": [{"tag": "field-journal"}],
        "_embedded": {
            "threads": [{
                "type": "customer",
                "body": "<p>Need help with header</p>",
                "source": {"type": "email"}
            }]
        }
    }

    response = client.post("/webhook/helpscout", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["ticket_id"] == 456


def test_webhook_skips_empty_message():
    """Should skip if no customer message."""
    payload = {
        "id": 789,
        "subject": "Empty",
        "primaryCustomer": {"first": "Reader"},
        "tags": [],
        "_embedded": {
            "threads": []
        }
    }

    response = client.post("/webhook/helpscout", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "skipped"


@patch("api.webhooks.process_in_background", new_callable=AsyncMock)
def test_note_webhook_uses_latest_note_instead_of_older_api_note(mock_process):
    payload = {
        "id": 321,
        "subject": "Manual trigger",
        "primaryCustomer": {"first": "Reader"},
        "_embedded": {
            "threads": [
                {
                    "type": "note",
                    "body": "<p>Older automated note</p>",
                    "createdAt": "2026-03-28T09:00:00Z",
                    "source": {"type": "api"},
                },
                {
                    "type": "customer",
                    "body": "<p>Please help</p>",
                    "createdBy": {"type": "customer"},
                },
                {
                    "type": "note",
                    "body": "<p>Please answer with the workaround</p>",
                    "createdAt": "2026-03-29T10:00:00Z",
                    "source": {"type": "web"},
                },
            ]
        }
    }

    response = client.post(
        "/webhook/helpscout",
        json=payload,
        headers={"X-HelpScout-Event": "convo.note.created"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"


@patch("api.webhooks.process_in_background", new_callable=AsyncMock)
def test_note_webhook_skips_duplicate_processed_note(mock_process, monkeypatch):
    monkeypatch.setattr(
        "api.webhooks.get_ticket_log_by_helpscout_id",
        lambda *_args, **_kwargs: {
            "helpscout_id": "321",
            "created_at": "2026-04-06 06:39:15",
            "status": "processed",
            "trigger": "note_trigger",
        },
    )

    payload = {
        "id": 321,
        "subject": "Manual trigger",
        "primaryCustomer": {"first": "Reader"},
        "_embedded": {
            "threads": [
                {
                    "type": "note",
                    "body": "<p>Please answer with the workaround</p>",
                    "createdAt": "2026-04-06T05:55:40Z",
                    "source": {"type": "mobile"},
                },
                {
                    "type": "customer",
                    "body": "<p>Please help</p>",
                    "createdBy": {"type": "customer"},
                },
            ]
        }
    }

    response = client.post(
        "/webhook/helpscout",
        json=payload,
        headers={"X-HelpScout-Event": "convo.note.created"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert "already processed" in response.json()["reason"].lower()
    mock_process.assert_not_called()


@pytest.mark.asyncio
async def test_process_in_background_logs_delivery_outcome(monkeypatch):
    seen = {}
    logged = []

    class FakeProcessor:
        async def process_ticket(self, payload, trigger="new_conversation"):
            seen["payload"] = payload
            seen["trigger"] = trigger
            return {
                "ticket_id": 456,
                "classification": "support",
                "delivery_mode": "auto_send",
                "response_posted": True,
                "confidence_status": "accepted",
            }

    monkeypatch.setattr("api.webhooks.get_processor", lambda: FakeProcessor())
    monkeypatch.setattr("api.webhooks.logger.info", lambda message: logged.append(message))

    payload = {"id": 456, "subject": "Help needed"}
    await process_in_background(payload, trigger="customer_reply")

    assert seen == {"payload": payload, "trigger": "customer_reply"}
    assert any("ticket=456" in message for message in logged)
    assert any("status=processed" in message for message in logged)
    assert any("delivery_mode=auto_send" in message for message in logged)


@pytest.mark.asyncio
async def test_process_in_background_logs_error_status_and_reason(monkeypatch):
    logged = []

    class FakeProcessor:
        async def process_ticket(self, payload, trigger="new_conversation"):
            return {
                "ticket_id": payload["id"],
                "status": "error",
                "classification": "support",
                "delivery_mode": None,
                "response_posted": False,
                "confidence_status": None,
                "error": "boom",
            }

    monkeypatch.setattr("api.webhooks.get_processor", lambda: FakeProcessor())
    monkeypatch.setattr("api.webhooks.logger.info", lambda message: logged.append(message))

    await process_in_background({"id": 999}, trigger="new_conversation")

    assert any("status=error" in message for message in logged)
    assert any("error=boom" in message for message in logged)


def test_agent_reply_from_webhook_threads_skips_drafts():
    from api.webhooks import agent_reply_and_customer_message

    threads = [
        {"id": 3, "type": "message", "state": "draft", "draft": True, "body": "<p>Unsent</p>", "createdBy": {"type": "user"}},
        {"id": 2, "type": "message", "state": "published", "body": "<p>Sent</p>", "createdBy": {"type": "user"}},
        {"id": 1, "type": "customer", "state": "published", "body": "<p>Help</p>", "createdBy": {"type": "customer"}},
    ]
    assert agent_reply_and_customer_message(threads) == ("<p>Sent</p>", "<p>Help</p>")
