from datetime import datetime, timezone

import pytest

from unittest.mock import AsyncMock

from db.database import (
    init_db,
    insert_pending_send,
    fetch_scheduled_pending_sends_by_conversation,
    get_db,
)
from services.autosend_scheduler import (
    AutoSendConfig,
    load_config,
    pick_due_at,
    sha256_of_html,
    evaluate_cancel_reason,
    process_due_row,
    run_worker_loop,
    schedule_autosend_worker,
)
from tests.lease_recording import RecordingLeaseFactory


def test_load_config_defaults(monkeypatch):
    for var in (
        "SUPA_AUTOSEND_DELAY_MIN_SECONDS",
        "SUPA_AUTOSEND_DELAY_MAX_SECONDS",
        "SUPA_AUTOSEND_WORKER_TICK_SECONDS",
        "SUPA_AUTOSEND_DELAY_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = load_config()
    assert cfg.delay_min_seconds == 300
    assert cfg.delay_max_seconds == 1200
    assert cfg.tick_seconds == 30
    assert cfg.enabled is True


def test_load_config_overrides(monkeypatch):
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_MIN_SECONDS", "60")
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_MAX_SECONDS", "90")
    monkeypatch.setenv("SUPA_AUTOSEND_WORKER_TICK_SECONDS", "5")
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_ENABLED", "false")

    cfg = load_config()
    assert cfg.delay_min_seconds == 60
    assert cfg.delay_max_seconds == 90
    assert cfg.tick_seconds == 5
    assert cfg.enabled is False


def test_pick_due_at_within_window():
    import random

    rng = random.Random(42)
    now = datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc)
    cfg = AutoSendConfig(delay_min_seconds=300, delay_max_seconds=1200, tick_seconds=30, enabled=True)

    for _ in range(20):
        due = pick_due_at(now=now, config=cfg, rng=rng)
        delta = (datetime.fromisoformat(due) - now).total_seconds()
        assert 300 <= delta <= 1200


def test_sha256_of_html_is_deterministic():
    assert sha256_of_html("<p>hi</p>") == sha256_of_html("<p>hi</p>")
    assert sha256_of_html("<p>hi</p>") != sha256_of_html("<p>HI</p>")


def _thread(thread_id, thread_type, created_at, body="x", source_type="email"):
    return {
        "id": thread_id,
        "type": thread_type,
        "body": body,
        "createdAt": created_at,
        "source": {"type": source_type},
        "createdBy": {"type": "customer" if thread_type == "customer" else "user"},
    }


def _row(**overrides):
    base = {
        "id": 1,
        "conversation_id": "100",
        "draft_thread_id": "500",
        "response_html_sha": sha256_of_html("<p>body</p>"),
        "queued_at": "2026-04-20T10:00:00+00:00",
        "due_at": "2026-04-20T10:10:00+00:00",
        "last_known_customer_thread_id": "200",
    }
    base.update(overrides)
    return base


def test_evaluate_cancel_reason_clean():
    row = _row()
    threads = [
        _thread("200", "customer", "2026-04-20T09:55:00+00:00"),
        {"id": "500", "type": "message", "state": "draft", "body": "<p>body</p>",
         "createdAt": "2026-04-20T09:59:00+00:00", "source": {"type": "api"}},
    ]
    assert evaluate_cancel_reason(row=row, threads=threads) is None


def test_evaluate_cancel_reason_customer_replied():
    row = _row()
    threads = [
        _thread("200", "customer", "2026-04-20T09:55:00+00:00"),
        _thread("201", "customer", "2026-04-20T10:05:00+00:00"),
        {"id": "500", "type": "message", "state": "draft", "body": "<p>body</p>",
         "createdAt": "2026-04-20T09:59:00+00:00", "source": {"type": "api"}},
    ]
    assert evaluate_cancel_reason(row=row, threads=threads) == "customer_replied"


def test_evaluate_cancel_reason_operator_sent():
    row = _row()
    threads = [
        _thread("200", "customer", "2026-04-20T09:55:00+00:00"),
        {"id": "700", "type": "message", "body": "<p>sent by hand</p>",
         "createdAt": "2026-04-20T10:05:00+00:00", "createdBy": {"type": "user"},
         "source": {"type": "web"}},
        {"id": "500", "type": "message", "state": "draft", "body": "<p>body</p>",
         "createdAt": "2026-04-20T09:59:00+00:00", "source": {"type": "api"}},
    ]
    assert evaluate_cancel_reason(row=row, threads=threads) == "operator_sent"


def test_evaluate_cancel_reason_note_added():
    row = _row()
    threads = [
        _thread("200", "customer", "2026-04-20T09:55:00+00:00"),
        {"id": "800", "type": "note", "body": "<p>hmm</p>",
         "createdAt": "2026-04-20T10:02:00+00:00", "createdBy": {"type": "user"},
         "source": {"type": "web"}},
        {"id": "500", "type": "message", "state": "draft", "body": "<p>body</p>",
         "createdAt": "2026-04-20T09:59:00+00:00", "source": {"type": "api"}},
    ]
    assert evaluate_cancel_reason(row=row, threads=threads) == "note_added"


def test_evaluate_cancel_reason_ignores_supa_decision_note():
    """Supa's own api-sourced decision notes (added right after queueing) must
    not cancel the pending_send; only human notes should trigger note_added."""
    row = _row()
    threads = [
        _thread("200", "customer", "2026-04-20T09:55:00+00:00"),
        {"id": "800", "type": "note",
         "body": "<p>Supa decision note: auto-send scheduled.</p>",
         "createdAt": "2026-04-20T10:02:00+00:00",
         "createdBy": {"type": "user"},
         "source": {"type": "api"}},
        {"id": "500", "type": "message", "state": "draft", "body": "<p>body</p>",
         "createdAt": "2026-04-20T09:59:00+00:00", "source": {"type": "api"}},
    ]
    assert evaluate_cancel_reason(row=row, threads=threads) is None


def test_evaluate_cancel_reason_ignores_supa_auto_improvement_note():
    row = _row()
    threads = [
        _thread("200", "customer", "2026-04-20T09:55:00+00:00"),
        {"id": "801", "type": "note",
         "body": "<p>Supa auto-improvement note: gap recorded.</p>",
         "createdAt": "2026-04-20T10:02:00+00:00",
         "createdBy": {"type": "user"},
         "source": {"type": "api"}},
        {"id": "500", "type": "message", "state": "draft", "body": "<p>body</p>",
         "createdAt": "2026-04-20T09:59:00+00:00", "source": {"type": "api"}},
    ]
    assert evaluate_cancel_reason(row=row, threads=threads) is None


def test_evaluate_cancel_reason_note_added_for_human_api_note():
    """A human-authored note (even with api source — e.g., mobile app) still cancels."""
    row = _row()
    threads = [
        _thread("200", "customer", "2026-04-20T09:55:00+00:00"),
        {"id": "802", "type": "note",
         "body": "<p>Remember to mention the refund policy.</p>",
         "createdAt": "2026-04-20T10:02:00+00:00",
         "createdBy": {"type": "user"},
         "source": {"type": "api"}},
        {"id": "500", "type": "message", "state": "draft", "body": "<p>body</p>",
         "createdAt": "2026-04-20T09:59:00+00:00", "source": {"type": "api"}},
    ]
    assert evaluate_cancel_reason(row=row, threads=threads) == "note_added"


def test_evaluate_cancel_reason_draft_deleted():
    row = _row()
    threads = [
        _thread("200", "customer", "2026-04-20T09:55:00+00:00"),
    ]
    assert evaluate_cancel_reason(row=row, threads=threads) == "draft_deleted"


def test_evaluate_cancel_reason_draft_modified():
    row = _row()
    threads = [
        _thread("200", "customer", "2026-04-20T09:55:00+00:00"),
        {"id": "500", "type": "message", "state": "draft", "body": "<p>DIFFERENT</p>",
         "createdAt": "2026-04-20T09:59:00+00:00", "source": {"type": "api"}},
    ]
    assert evaluate_cancel_reason(row=row, threads=threads) == "draft_modified"


def test_evaluate_cancel_reason_extra_draft():
    row = _row()
    threads = [
        _thread("200", "customer", "2026-04-20T09:55:00+00:00"),
        {"id": "500", "type": "message", "state": "draft", "body": "<p>body</p>",
         "createdAt": "2026-04-20T09:59:00+00:00", "source": {"type": "api"}},
        {"id": "600", "type": "message", "state": "draft", "body": "<p>another</p>",
         "createdAt": "2026-04-20T10:05:00+00:00", "source": {"type": "api"}},
    ]
    assert evaluate_cancel_reason(row=row, threads=threads) == "extra_draft"


class FakeHelpScout:
    def __init__(self, conversation: dict):
        self.conversation = conversation
        self.send_reply = AsyncMock(return_value=True)
        self.ensure_conversation_status = AsyncMock(return_value={"verified": True, "conversation": conversation})
        self.add_tags = AsyncMock(return_value=True)
        self.remove_tag = AsyncMock(return_value=True)
        self.delete_draft = AsyncMock(return_value=True)

    async def get_conversation(self, conversation_id):
        return self.conversation

    async def get_complete_conversation(self, conversation_id):
        return self.conversation


def _seed_scheduled_row(tmp_path, **overrides):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    kwargs = {
        "db_path": db_path,
        "conversation_id": "100",
        "draft_thread_id": "500",
        "response_html_sha": sha256_of_html("<p>body</p>"),
        "due_at": "2020-01-01T00:00:00+00:00",
        "last_known_customer_thread_id": "200",
        "tags_on_promote": ["supa-auto-sent"],
        "confidence_score": 10,
        "confidence_provider": "claude",
    }
    kwargs.update(overrides)
    row_id = insert_pending_send(**kwargs)
    return db_path, row_id


def _clean_conversation():
    return {
        "id": 100,
        "primaryCustomer": {"id": 55},
        "tags": [
            {"id": 1, "tag": "supa-processed"},
            {"id": 2, "tag": "supa-pending-send"},
        ],
        "_embedded": {
            "threads": [
                {"id": "200", "type": "customer", "body": "<p>hi</p>",
                 "createdAt": "2019-12-31T23:00:00+00:00",
                 "createdBy": {"type": "customer"}, "source": {"type": "email"}},
                {"id": "500", "type": "message", "state": "draft", "body": "<p>body</p>",
                 "createdAt": "2019-12-31T23:30:00+00:00", "source": {"type": "api"}},
            ]
        },
    }


def _normalization_case(tmp_path, state="cancelled"):
    from db.database import log_ticket, update_pending_send_state
    intended, persisted = "<p>Project &raquo; Content Builder</p>", "<p>Project » Content Builder</p>"
    db_path, row_id = _seed_scheduled_row(tmp_path, response_html_sha=sha256_of_html(intended))
    log_ticket(db_path=db_path, helpscout_id="100", classification="support", status="processed",
               response_posted=True, delivery_mode="auto_send", confidence_score=8,
               confidence_status="accept", draft_response_html=intended)
    if state == "cancelled":
        update_pending_send_state(db_path=db_path, row_id=row_id, state="cancelled", cancel_reason="draft_modified")
    convo = _clean_conversation()
    convo["status"] = "active"
    convo["_embedded"]["threads"][1]["body"] = persisted
    return db_path, row_id, FakeHelpScout(convo), persisted


@pytest.mark.parametrize("state", ["scheduled", "cancelled"])
async def test_normalization_reconciliation_repins_approved_body_then_normal_worker_sends(tmp_path, monkeypatch, state):
    from services.autosend_scheduler import recover_normalized_autosends
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "live")
    db_path, row_id, helpscout, persisted = _normalization_case(tmp_path, state)
    with get_db(db_path) as db:
        before = dict(db.execute("SELECT * FROM pending_sends WHERE id=?", (row_id,)).fetchone())

    assert await recover_normalized_autosends(db_path=db_path, helpscout=helpscout, config=load_config()) == [row_id]

    helpscout.send_reply.assert_not_awaited()
    with get_db(db_path) as db:
        row = dict(db.execute("SELECT * FROM pending_sends WHERE id=?", (row_id,)).fetchone())
    assert row["state"] == "scheduled"
    assert row["response_html_sha"] == sha256_of_html(persisted)
    assert row["queued_at"] == before["queued_at"] and row["due_at"] == before["due_at"]
    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())
    helpscout.send_reply.assert_awaited_once_with(100, persisted, customer_id=55)


@pytest.mark.parametrize("change", ["body", "attribute", "customer", "note", "closed", "owner",
                                    "newer_row", "audit", "verdict", "attempt", "ignore", "disabled", "draft_only"])
async def test_normalization_recovery_keeps_real_edits_and_delivery_holds(tmp_path, monkeypatch, change):
    from services.autosend_scheduler import recover_normalized_autosends
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "draft_only" if change == "draft_only" else "live")
    db_path, row_id, helpscout, _ = _normalization_case(tmp_path)
    convo = helpscout.conversation
    threads = convo["_embedded"]["threads"]
    if change == "body": threads[1]["body"] = "<p>Changed instruction</p>"
    if change == "attribute": threads[1]["body"] = '<p style="display:none">Project » Content Builder</p>'
    if change == "customer": threads.append(_thread(201, "customer", "2099-01-01T00:00:00+00:00"))
    if change == "note": threads.append(_thread(202, "note", "2099-01-01T00:00:00+00:00", source_type="web"))
    if change == "closed": convo["status"] = "closed"
    if change == "owner": threads[1]["source"]["type"] = "web"
    if change == "ignore": convo["tags"].append({"tag": "supa-ignore"})
    with get_db(db_path) as db:
        if change == "audit": db.execute("UPDATE tickets SET draft_response_html='different'")
        if change == "verdict": db.execute("UPDATE tickets SET confidence_status='retry'")
        if change == "attempt": db.execute("UPDATE pending_sends SET attempts=1")
        db.commit()
    if change == "newer_row":
        insert_pending_send(db_path=db_path, conversation_id="100", draft_thread_id="501", response_html_sha="new",
                            due_at="2099-01-01T00:00:00Z", last_known_customer_thread_id="200", tags_on_promote=[],
                            confidence_score=8, confidence_provider="claude")
    cfg = AutoSendConfig(300, 1200, 30, change != "disabled")

    assert await recover_normalized_autosends(db_path=db_path, helpscout=helpscout, config=cfg) == []

    with get_db(db_path) as db:
        assert db.execute("SELECT state FROM pending_sends WHERE id=?", (row_id,)).fetchone()["state"] == "cancelled"
    helpscout.send_reply.assert_not_awaited()


async def test_normalization_recovery_cas_rejects_a_newer_queue_row(tmp_path, monkeypatch):
    from services.autosend_scheduler import recover_normalized_autosends
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "live")
    db_path, row_id, helpscout, _ = _normalization_case(tmp_path)
    async def race(_):
        insert_pending_send(db_path=db_path, conversation_id="100", draft_thread_id="501", response_html_sha="new",
                            due_at="2099-01-01T00:00:00Z", last_known_customer_thread_id="200", tags_on_promote=[],
                            confidence_score=8, confidence_provider="claude")
        return helpscout.conversation
    helpscout.get_complete_conversation = race

    assert await recover_normalized_autosends(db_path=db_path, helpscout=helpscout, config=load_config()) == []

    with get_db(db_path) as db:
        assert db.execute("SELECT state FROM pending_sends WHERE id=?", (row_id,)).fetchone()["state"] == "cancelled"


async def test_closed_conversation_cannot_publish_a_due_auto_reply(tmp_path):
    db_path, row_id = _seed_scheduled_row(tmp_path)
    convo = _clean_conversation()
    convo["status"] = "closed"
    helpscout = FakeHelpScout(convo)

    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())

    helpscout.send_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_due_row_clean_promotes(tmp_path, monkeypatch):
    db_path, row_id = _seed_scheduled_row(tmp_path)
    helpscout = FakeHelpScout(_clean_conversation())
    leases = RecordingLeaseFactory()
    monkeypatch.setattr("services.autosend_scheduler.conversation_lease", leases)

    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())

    helpscout.send_reply.assert_awaited_once_with(100, "<p>body</p>", customer_id=55)
    helpscout.ensure_conversation_status.assert_awaited_once()
    helpscout.add_tags.assert_awaited_once()
    args, _ = helpscout.add_tags.call_args
    assert "supa-auto-sent" in args[1]
    helpscout.remove_tag.assert_awaited_once()
    helpscout.delete_draft.assert_awaited_once_with(100, 500)

    db = get_db(db_path)
    row = db.execute("SELECT state, cancel_reason FROM pending_sends WHERE id=?", (row_id,)).fetchone()
    db.close()
    assert row["state"] == "sent"
    assert row["cancel_reason"] is None
    assert [event[0] for event in leases.events].count("enter") == 1
    leases.assert_guarded_count(100, 5)


@pytest.mark.parametrize("already_sent", [False, True])
async def test_ignore_tag_holds_delayed_send_and_cleanup_until_removed(tmp_path, already_sent):
    db_path, row_id = _seed_scheduled_row(tmp_path)
    conversation = _clean_conversation()
    conversation["tags"].append({"tag": "supa-ignore"})
    if already_sent:
        conversation["_embedded"]["threads"].append({
            "id": "600", "type": "message", "state": "published", "body": "<p>body</p>",
            "createdAt": datetime.now(timezone.utc).isoformat(), "createdBy": {"type": "user"},
            "source": {"type": "api"}})
    helpscout = FakeHelpScout(conversation)
    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())
    for method in (helpscout.send_reply, helpscout.ensure_conversation_status, helpscout.add_tags,
                   helpscout.remove_tag, helpscout.delete_draft):
        method.assert_not_awaited()
    with get_db(db_path) as db:
        assert db.execute("SELECT state FROM pending_sends WHERE id=?", (row_id,)).fetchone()["state"] == "scheduled"
    conversation["tags"] = [t for t in conversation["tags"] if t["tag"] != "supa-ignore"]
    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())
    if already_sent:
        helpscout.send_reply.assert_not_awaited()
    else:
        helpscout.send_reply.assert_awaited_once()
    helpscout.delete_draft.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_due_row_customer_reply_cancels(tmp_path, monkeypatch):
    db_path, row_id = _seed_scheduled_row(tmp_path)
    convo = _clean_conversation()
    convo["_embedded"]["threads"].append({
        "id": "201", "type": "customer", "body": "<p>wait</p>",
        "createdAt": "2099-01-01T00:00:00+00:00",
        "createdBy": {"type": "customer"}, "source": {"type": "email"},
    })
    helpscout = FakeHelpScout(convo)
    leases = RecordingLeaseFactory()
    monkeypatch.setattr("services.autosend_scheduler.conversation_lease", leases)

    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())

    helpscout.send_reply.assert_not_awaited()
    helpscout.remove_tag.assert_awaited_once()

    db = get_db(db_path)
    row = db.execute("SELECT state, cancel_reason FROM pending_sends WHERE id=?", (row_id,)).fetchone()
    db.close()
    assert row["state"] == "cancelled"
    assert row["cancel_reason"] == "customer_replied"
    leases.assert_guarded_count(100, 1)


@pytest.mark.asyncio
async def test_process_due_row_transient_send_failure_retries(tmp_path):
    db_path, row_id = _seed_scheduled_row(tmp_path)
    helpscout = FakeHelpScout(_clean_conversation())
    helpscout.send_reply = AsyncMock(return_value=False)

    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())

    db = get_db(db_path)
    row = db.execute(
        "SELECT state, attempts, due_at, last_error FROM pending_sends WHERE id=?",
        (row_id,),
    ).fetchone()
    db.close()
    assert row["state"] == "scheduled"
    assert row["attempts"] == 1
    assert row["last_error"] is not None
    assert row["due_at"] > "2020-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_process_due_row_gives_up_after_max_attempts(tmp_path):
    db_path, row_id = _seed_scheduled_row(tmp_path)
    conn = get_db(db_path)
    conn.execute("UPDATE pending_sends SET attempts = 4 WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()

    helpscout = FakeHelpScout(_clean_conversation())
    helpscout.send_reply = AsyncMock(return_value=False)

    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())

    db = get_db(db_path)
    row = db.execute("SELECT state FROM pending_sends WHERE id=?", (row_id,)).fetchone()
    db.close()
    assert row["state"] == "failed"


@pytest.mark.asyncio
async def test_process_due_row_idempotent_already_sent(tmp_path):
    db_path, row_id = _seed_scheduled_row(tmp_path)
    convo = _clean_conversation()
    convo["_embedded"]["threads"] = [
        convo["_embedded"]["threads"][0],
        {"id": "900", "type": "message", "state": "published", "body": "<p>body</p>",
         "createdAt": datetime.now(timezone.utc).isoformat(),
         "createdBy": {"type": "user"}, "source": {"type": "api"}},
    ]
    helpscout = FakeHelpScout(convo)

    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())

    helpscout.send_reply.assert_not_awaited()
    db = get_db(db_path)
    row = db.execute("SELECT state FROM pending_sends WHERE id=?", (row_id,)).fetchone()
    db.close()
    assert row["state"] == "sent"


@pytest.mark.asyncio
async def test_run_worker_loop_runs_one_pass_and_exits(tmp_path, monkeypatch):
    db_path, row_id = _seed_scheduled_row(tmp_path)
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_ENABLED", "true")

    helpscout = FakeHelpScout(_clean_conversation())

    async def fake_sleep(_):
        return

    await run_worker_loop(
        db_path=db_path,
        helpscout=helpscout,
        config=load_config(),
        iterations=1,
        sleep_func=fake_sleep,
    )

    db = get_db(db_path)
    row = db.execute("SELECT state FROM pending_sends WHERE id=?", (row_id,)).fetchone()
    db.close()
    assert row["state"] == "sent"


@pytest.mark.asyncio
async def test_run_worker_loop_skips_when_disabled(tmp_path, monkeypatch):
    db_path, row_id = _seed_scheduled_row(tmp_path)
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_ENABLED", "false")

    helpscout = FakeHelpScout(_clean_conversation())

    async def fake_sleep(_):
        return

    await run_worker_loop(
        db_path=db_path,
        helpscout=helpscout,
        config=load_config(),
        iterations=1,
        sleep_func=fake_sleep,
    )

    helpscout.send_reply.assert_not_awaited()
    db = get_db(db_path)
    row = db.execute("SELECT state FROM pending_sends WHERE id=?", (row_id,)).fetchone()
    db.close()
    assert row["state"] == "scheduled"


@pytest.mark.asyncio
async def test_schedule_autosend_worker_returns_task(tmp_path, monkeypatch):
    import asyncio
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    monkeypatch.setenv("SUPA_AUTOSEND_WORKER_TICK_SECONDS", "3600")

    task = schedule_autosend_worker(helpscout=FakeHelpScout(_clean_conversation()), db_path=db_path)
    assert isinstance(task, asyncio.Task)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
