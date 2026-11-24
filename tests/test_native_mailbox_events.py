import base64
from contextlib import closing
import hashlib
import hmac
import json
import sqlite3
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api import webhooks
from db.database import init_db
from services import native_mailbox_events as jobs


@pytest.fixture
def native(tmp_path, monkeypatch):
    path = tmp_path / "supa.db"
    init_db(str(path))
    monkeypatch.setenv("SUPA_DB_PATH", str(path))
    monkeypatch.setenv("SUPA_MAILBOX_ORIGIN", "https://inbox.example")
    monkeypatch.setenv("SUPA_MAILBOX_WEBHOOK_SECRET", "test-only")
    application = FastAPI()
    application.include_router(webhooks.router)
    return path, TestClient(application)


PAYLOAD = {"id": 42, "subject": "Café", "tags": [], "_embedded": {"threads": [
    {"id": 99, "type": "customer", "body": "Café ☕", "createdBy": {"type": "customer"},
     "createdAt": "2026-11-01T01:00:00Z", "source": {"type": "email"}}
]}}
BODY = json.dumps(PAYLOAD, ensure_ascii=False).encode()


def signed(body=BODY, event="convo.created", key="17"):
    secret = b"test-only"
    return {"Content-Type": "application/json", "X-HelpScout-Event": event, "X-Inbox-Delivery": key,
            "X-HelpScout-Signature": base64.b64encode(hmac.digest(secret, body, "sha1")).decode(),
            "X-Inbox-Signature": hmac.new(secret, f"{key}\n{event}\n".encode() + body, hashlib.sha256).hexdigest()}


@pytest.mark.asyncio
async def test_signed_http_acceptance_journals_and_dispatches_exactly_once(native, monkeypatch):
    path, client = native
    handler = AsyncMock()
    monkeypatch.setattr(webhooks, "process_in_background", handler)
    monkeypatch.setattr(webhooks, "manual_route", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(webhooks.extensions, "intercept_webhook", lambda *args: None)
    def post(raw=BODY, headers=None):
        return client.post("/webhook/helpscout", headers=headers or signed(raw), content=raw)

    assert post().json()["status"] == post().json()["status"] == "queued"
    for field, value in (("X-Inbox-Delivery", "18"), ("X-HelpScout-Event", "convo.note.created")):
        assert post(headers=signed() | {field: value}).status_code == 401
    assert post(BODY + b" ", signed()).status_code == 401
    assert post(BODY + b" ").status_code == 409
    assert post(headers=signed(event="convo.note.created")).status_code == 409
    assert post(headers=signed(event="convo.invalid")).status_code == 400
    assert post(b"{}").status_code == 400
    assert post(headers={"X-HelpScout-Event": "convo.created"}).json()["status"] == "skipped"
    handler.assert_not_awaited()
    assert await webhooks.process_native_delivery(path)
    handler.assert_awaited_once_with(PAYLOAD, "new_conversation")
    assert not await webhooks.process_native_delivery(path)
    assert post().json()["status"] == "done"
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT count(*) FROM agent_events").fetchone()[0] == 1
        assert db.execute("SELECT payload FROM native_mailbox_jobs").fetchall() == [(None,)]
    monkeypatch.delenv("SUPA_MAILBOX_ORIGIN")
    assert post().status_code == 409


@pytest.mark.asyncio
async def test_failed_worker_and_interrupted_attempt_need_reviewed_resolution(native, monkeypatch):
    path, client = native
    assert client.post("/webhook/helpscout", headers=signed(), content=BODY).status_code == 200
    monkeypatch.setattr(webhooks, "route_mailbox_event", AsyncMock(side_effect=RuntimeError))
    assert await webhooks.process_native_delivery(path)
    assert not await webhooks.process_native_delivery(path)
    row, = jobs.review_list(path)
    assert row["reason"] == "RuntimeError"
    digest = hashlib.sha256(BODY).hexdigest()
    assert jobs.reconcile(path, 17, digest, 1, "retry", "audit-1") == "queued"
    assert jobs.reconcile(path, 17, digest, 1, "retry", "audit-1") == "queued"
    assert jobs.claim(path)["payload"] == BODY
    jobs.recover(path)
    second, = jobs.review_list(path)
    assert second["attempt"] == 2
    assert second["reason"] == "interrupted_during_processing"
    with pytest.raises(jobs.DeliveryConflict):
        jobs.reconcile(path, 17, digest, 1, "finish", "audit-1")
    with pytest.raises(jobs.DeliveryConflict):
        jobs.reconcile(path, 17, "wrong digest", 2, "retry", "audit-2")
    assert jobs.reconcile(path, 17, digest, 2, "finish", "audit-2") == "done"
    assert not await webhooks.process_native_delivery(path)
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT payload FROM native_mailbox_jobs").fetchall() == [(None,)]
        assert db.execute("SELECT action,evidence FROM native_mailbox_resolutions ORDER BY attempt").fetchall() == [
            ("retry", "audit-1"), ("finish", "audit-2")]


def test_journal_failure_rolls_back_native_acceptance(native, monkeypatch):
    path, client = native
    def fail(*args):
        raise sqlite3.OperationalError("unavailable")
    monkeypatch.setattr(jobs, "append_event_in_transaction", fail)
    assert client.post("/webhook/helpscout", headers=signed(), content=BODY).status_code == 503
    assert jobs.counts(path) == {}


@pytest.mark.asyncio
async def test_native_worker_health_shutdown_and_restart_recovery(native, monkeypatch):
    import asyncio
    import main

    path, client = native
    accepted = client.post("/webhook/helpscout", headers=signed(), content=BODY)
    assert accepted.json()["status"] == "queued"
    entered = asyncio.Event()
    async def active_handler(*args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(webhooks, "route_mailbox_event", active_handler)
    task = asyncio.create_task(webhooks.native_worker(path))
    monkeypatch.setattr(main.app.state, "native_mailbox_task", task, raising=False)
    await asyncio.wait_for(entered.wait(), timeout=3)
    assert await main.health() == {"status": "ok", "mailbox_jobs": {"processing": 1}}
    await main.shutdown_recovery_tasks()
    assert task.cancelled()
    assert (await main.health()).status_code == 503
    restarted = asyncio.create_task(webhooks.native_worker(path))
    try:
        await asyncio.sleep(0)
        assert jobs.counts(path) == {"review": 1}
    finally:
        restarted.cancel()
        await asyncio.gather(restarted, return_exceptions=True)
