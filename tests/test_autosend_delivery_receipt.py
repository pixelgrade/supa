"""Automatic delivery records bind the acknowledged send to verified published bytes."""

import copy
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from db.database import get_db, init_db, insert_pending_send
from services.autosend_scheduler import load_config, process_due_row, sha256_of_html


BODY = "<p>Here is the verified answer.</p>"


def _seed(tmp_path):
    db_path = str(tmp_path / "receipts.db")
    init_db(db_path)
    row_id = insert_pending_send(
        db_path=db_path, conversation_id="100", draft_thread_id="500",
        response_html_sha=sha256_of_html(BODY), due_at="2020-01-01T00:00:00Z",
        last_known_customer_thread_id="200", tags_on_promote=["supa-auto-sent"],
        confidence_score=8, confidence_provider="claude",
    )
    return db_path, row_id


def _row(db_path, row_id):
    with get_db(db_path) as conn:
        return dict(conn.execute("SELECT * FROM pending_sends WHERE id=?", (row_id,)).fetchone())


class ReceiptHelpScout:
    def __init__(self):
        self.send_calls = 0
        self.readback_error = False
        self.cleanup_error = False
        self.status_result = {"verified": True}
        self.add_tags_result = True
        self.remove_tag_result = True
        self.delete_result = True
        self.delete_calls = 0
        self.status_calls = 0
        self.published_overrides = {}
        self.published_at = datetime.now(timezone.utc).isoformat()
        self.conversation = {
            "id": 100, "status": "active", "primaryCustomer": {"id": 55},
            "tags": [{"tag": "supa-pending-send"}],
            "_embedded": {"threads": [
                {"id": "200", "type": "customer", "state": "published", "body": "Question",
                 "createdAt": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()},
                {"id": "500", "type": "message", "state": "draft", "body": BODY,
                 "createdAt": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                 "source": {"type": "api"}},
            ]},
        }

    def publish(self):
        self.conversation["_embedded"]["threads"].append({
            "id": "700", "type": "message", "state": "published", "body": BODY,
            "createdAt": self.published_at, "source": {"type": "api"},
            "createdBy": {"type": "user"}, **self.published_overrides,
        })

    async def get_conversation(self, conversation_id):
        return copy.deepcopy(self.conversation)

    async def get_complete_conversation(self, conversation_id):
        if self.readback_error:
            raise RuntimeError("readback unavailable")
        return copy.deepcopy(self.conversation)

    async def send_reply(self, conversation_id, body, customer_id=None):
        self.send_calls += 1
        assert body == BODY
        self.publish()
        return True

    async def ensure_conversation_status(self, *args):
        self.status_calls += 1
        if self.cleanup_error:
            raise RuntimeError("post-send status update failed")
        if self.status_result and self.status_result.get("verified"):
            self.conversation["status"] = "pending"
        return self.status_result

    async def add_tags(self, *args):
        if self.add_tags_result:
            current = [t["tag"] for t in self.conversation["tags"]]
            self.conversation["tags"] = [{"tag": tag} for tag in dict.fromkeys(current + args[1])]
        return self.add_tags_result

    async def remove_tag(self, *args):
        if self.remove_tag_result:
            self.conversation["tags"] = [t for t in self.conversation["tags"] if t["tag"] != args[1]]
        return self.remove_tag_result

    async def delete_draft(self, *args):
        self.delete_calls += 1
        if self.delete_result:
            for thread in self.conversation["_embedded"]["threads"]:
                if str(thread["id"]) == str(args[1]):
                    thread["body"] = "<p>Manual review required before sending a reply.</p>"
        return self.delete_result


async def test_successful_autosend_records_verified_published_thread_and_time(tmp_path):
    db_path, row_id = _seed(tmp_path)
    helpscout = ReceiptHelpScout()

    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())

    row = _row(db_path, row_id)
    assert row["state"] == "sent"
    assert row["published_thread_id"] == "700"
    assert row["sent_at"] == helpscout.published_at
    assert row["delivery_verified_at"]
    assert row["delivery_receipt_error"] is None
    assert helpscout.send_calls == 1


async def test_already_sent_reconciliation_records_receipt_without_sending_again(tmp_path):
    db_path, row_id = _seed(tmp_path)
    helpscout = ReceiptHelpScout()
    helpscout.publish()

    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())

    row = _row(db_path, row_id)
    assert row["state"] == "sent"
    assert row["published_thread_id"] == "700"
    assert row["sent_at"] == helpscout.published_at
    assert helpscout.send_calls == 0


async def test_old_identical_reply_is_not_delivery_for_a_new_question(tmp_path):
    db_path, row_id = _seed(tmp_path)
    helpscout = ReceiptHelpScout()
    helpscout.published_at = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    helpscout.publish()
    helpscout.conversation["_embedded"]["threads"][-1]["id"] = "600"
    helpscout.published_at = datetime.now(timezone.utc).isoformat()
    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())
    assert helpscout.send_calls == 1
    assert _row(db_path, row_id)["published_thread_id"] == "700"


async def test_acknowledged_send_readback_failure_is_terminal_and_reconciles_without_resend(tmp_path):
    db_path, row_id = _seed(tmp_path)
    helpscout = ReceiptHelpScout()
    helpscout.readback_error = True

    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())
    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())

    row = _row(db_path, row_id)
    assert row["state"] == "sent"
    assert row["published_thread_id"] is None
    assert row["sent_at"] is None
    assert row["delivery_verified_at"] is None
    assert row["delivery_receipt_error"] == "delivery_readback_failed"
    assert helpscout.send_calls == 1

    from services.autosend_scheduler import reconcile_unverified_autosend_receipts

    helpscout.readback_error = False
    with get_db(db_path) as conn:
        conn.execute("UPDATE pending_sends SET updated_at=? WHERE id=?", (
            (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(), row_id,
        ))
        conn.commit()
    assert await reconcile_unverified_autosend_receipts(db_path=db_path, helpscout=helpscout) == [row_id]
    row = _row(db_path, row_id)
    assert row["published_thread_id"] == "700"
    assert row["delivery_receipt_error"] is None
    assert helpscout.send_calls == 1


@pytest.mark.parametrize("overrides", [
    {"body": "<p>Different answer</p>"},
    {"state": "draft"},
    {"state": "hidden"},
    {"state": None},
    {"source": {"type": "web"}},
    {"id": None},
    {"createdAt": "2019-01-01T00:00:00Z"},
])
async def test_wrong_or_nonpublished_body_never_creates_a_verified_receipt(tmp_path, overrides):
    db_path, row_id = _seed(tmp_path)
    helpscout = ReceiptHelpScout()
    helpscout.published_overrides = overrides

    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())

    row = _row(db_path, row_id)
    assert row["state"] == "sent"
    assert row["published_thread_id"] is None
    assert row["sent_at"] is None
    assert row["delivery_verified_at"] is None
    assert row["delivery_receipt_error"] == "published_reply_not_verified"
    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())
    assert helpscout.send_calls == 1


async def test_cleanup_failure_after_acknowledged_send_cannot_reschedule_delivery(tmp_path):
    db_path, row_id = _seed(tmp_path)
    helpscout = ReceiptHelpScout()
    helpscout.cleanup_error = True

    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())
    assert _row(db_path, row_id)["state"] == "sent"
    assert _row(db_path, row_id)["cleanup_error"] == "cleanup_status_failed"
    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())

    assert _row(db_path, row_id)["state"] == "sent"
    assert helpscout.send_calls == 1


def _make_cleanup_due(db_path, row_id):
    with get_db(db_path) as conn:
        conn.execute("UPDATE pending_sends SET cleanup_updated_at=? WHERE id=?", (
            (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(), row_id,
        ))
        conn.commit()


async def test_cleanup_failure_reconciles_without_resending_or_losing_verified_receipt(tmp_path):
    db_path, row_id = _seed(tmp_path)
    helpscout = ReceiptHelpScout()
    helpscout.delete_result = False

    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())
    before = _row(db_path, row_id)
    assert before["state"] == "sent"
    assert before["published_thread_id"] == "700"
    assert before["cleanup_error"] == "cleanup_delete_draft_failed"
    assert helpscout.conversation["_embedded"]["threads"][1]["body"] == BODY

    from services.autosend_scheduler import reconcile_autosend_cleanup

    helpscout.delete_result = True
    _make_cleanup_due(db_path, row_id)
    assert await reconcile_autosend_cleanup(db_path=db_path, helpscout=helpscout) == [row_id]
    after = _row(db_path, row_id)
    assert after["cleanup_error"] is None
    assert after["published_thread_id"] == before["published_thread_id"]
    assert after["delivery_verified_at"] == before["delivery_verified_at"]
    assert helpscout.conversation["_embedded"]["threads"][1]["body"] != BODY
    assert "supa-pending-send" not in [t["tag"] for t in helpscout.conversation["tags"]]
    assert helpscout.send_calls == 1


@pytest.mark.parametrize("change,error", [
    ({"body": "<p>Operator's revised draft</p>"}, "cleanup_draft_modified"),
    ({"source": {"type": "web"}}, "cleanup_draft_not_api_owned"),
])
async def test_cleanup_preserves_changed_or_human_owned_drafts(tmp_path, change, error):
    db_path, row_id = _seed(tmp_path)
    helpscout = ReceiptHelpScout()
    helpscout.delete_result = False
    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())
    draft = helpscout.conversation["_embedded"]["threads"][1]
    draft.update(change)
    protected = copy.deepcopy(draft)
    delete_calls = helpscout.delete_calls

    from services.autosend_scheduler import reconcile_autosend_cleanup

    helpscout.delete_result = True
    _make_cleanup_due(db_path, row_id)
    assert await reconcile_autosend_cleanup(db_path=db_path, helpscout=helpscout) == []
    assert _row(db_path, row_id)["cleanup_error"] == error
    from db.database import fetch_pending_autosend_cleanup

    assert fetch_pending_autosend_cleanup(db_path=db_path, before="2099-01-01") == []
    assert draft == protected
    assert helpscout.delete_calls == delete_calls
    assert helpscout.send_calls == 1


@pytest.mark.parametrize("setting,value,error", [
    ("status_result", {"verified": False}, "cleanup_status_failed"),
    ("status_result", False, "cleanup_status_failed"),
    ("add_tags_result", False, "cleanup_add_tags_failed"),
    ("remove_tag_result", False, "cleanup_remove_tag_failed"),
    ("delete_result", False, "cleanup_delete_draft_failed"),
])
async def test_false_cleanup_outcomes_remain_explicitly_incomplete(tmp_path, setting, value, error):
    db_path, row_id = _seed(tmp_path)
    helpscout = ReceiptHelpScout()
    setattr(helpscout, setting, value)

    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())

    row = _row(db_path, row_id)
    assert row["state"] == "sent"
    assert row["published_thread_id"] == "700"
    assert row["cleanup_error"] == error
    assert helpscout.send_calls == 1


async def test_old_cleanup_does_not_change_new_customer_status(tmp_path):
    db_path, row_id = _seed(tmp_path)
    helpscout = ReceiptHelpScout()
    helpscout.delete_result = False
    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())
    helpscout.conversation["_embedded"]["threads"].append({
        "id": "800", "type": "customer", "state": "published", "body": "A new question",
        "createdAt": (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
    })
    helpscout.conversation["status"] = "active"
    status_calls = helpscout.status_calls
    helpscout.delete_result = True

    from services.autosend_scheduler import reconcile_autosend_cleanup

    _make_cleanup_due(db_path, row_id)
    assert await reconcile_autosend_cleanup(db_path=db_path, helpscout=helpscout) == [row_id]
    assert helpscout.conversation["status"] == "active"
    assert helpscout.status_calls == status_calls
    assert helpscout.send_calls == 1


async def test_old_cleanup_keeps_a_newer_scheduled_sends_pending_marker(tmp_path):
    db_path, row_id = _seed(tmp_path)
    helpscout = ReceiptHelpScout()
    helpscout.delete_result = False
    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())
    insert_pending_send(
        db_path=db_path, conversation_id="100", draft_thread_id="501",
        response_html_sha=sha256_of_html("<p>A later answer.</p>"), due_at="2099-01-01T00:00:00Z",
        last_known_customer_thread_id="800", tags_on_promote=["supa-auto-sent"],
        confidence_score=8, confidence_provider="claude",
    )
    helpscout.delete_result = True

    from services.autosend_scheduler import reconcile_autosend_cleanup

    _make_cleanup_due(db_path, row_id)
    assert await reconcile_autosend_cleanup(db_path=db_path, helpscout=helpscout) == []
    assert _row(db_path, row_id)["cleanup_error"] == "cleanup_newer_send_pending"
    assert "supa-pending-send" in [t["tag"] for t in helpscout.conversation["tags"]]
    assert helpscout.send_calls == 1


async def test_cancelled_cleanup_stays_pending_and_reconciles_without_resending(tmp_path):
    import asyncio

    db_path, row_id = _seed(tmp_path)
    helpscout = ReceiptHelpScout()
    original_delete = helpscout.delete_draft

    async def cancelled_delete(*args):
        raise asyncio.CancelledError

    helpscout.delete_draft = cancelled_delete
    with pytest.raises(asyncio.CancelledError):
        await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())
    row = _row(db_path, row_id)
    assert row["state"] == "sent"
    assert row["published_thread_id"] == "700"
    assert row["cleanup_error"] == "cleanup_delete_draft_failed"

    from services.autosend_scheduler import reconcile_autosend_cleanup

    helpscout.delete_draft = original_delete
    _make_cleanup_due(db_path, row_id)
    assert await reconcile_autosend_cleanup(db_path=db_path, helpscout=helpscout) == [row_id]
    assert helpscout.send_calls == 1


async def test_cancelled_readback_after_acknowledged_send_is_never_due_again(tmp_path):
    import asyncio

    db_path, row_id = _seed(tmp_path)
    helpscout = ReceiptHelpScout()

    async def cancelled_readback(_):
        raise asyncio.CancelledError

    helpscout.get_complete_conversation = cancelled_readback
    with pytest.raises(asyncio.CancelledError):
        await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())

    row = _row(db_path, row_id)
    assert row["state"] == "sent"
    assert row["delivery_receipt_error"] == "awaiting_delivery_readback"
    assert row["cleanup_error"] == "cleanup_pending"
    await process_due_row(db_path=db_path, row_id=row_id, helpscout=helpscout, config=load_config())
    assert helpscout.send_calls == 1


def test_additive_migration_does_not_fabricate_historical_delivery_receipts(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE pending_sends (
            id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
            draft_thread_id TEXT NOT NULL, response_html_sha TEXT NOT NULL,
            queued_at TEXT NOT NULL, due_at TEXT NOT NULL, last_known_customer_thread_id TEXT,
            tags_on_promote TEXT NOT NULL DEFAULT '[]', confidence_score INTEGER,
            confidence_provider TEXT, state TEXT NOT NULL DEFAULT 'scheduled', cancel_reason TEXT,
            attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT, updated_at TEXT NOT NULL
        )""")
        conn.execute("""INSERT INTO pending_sends
            (conversation_id,draft_thread_id,response_html_sha,queued_at,due_at,state,updated_at)
            VALUES ('100','500','old','2019-01-01','2019-01-01','sent','2019-01-01')""")
    init_db(db_path)

    row = _row(db_path, 1)
    assert row["state"] == "sent"
    assert row["published_thread_id"] is None
    assert row["sent_at"] is None
    assert row["delivery_verified_at"] is None
    assert row["delivery_receipt_error"] is None
    assert row["cleanup_error"] is None
    assert row["cleanup_updated_at"] is None

    from db.database import fetch_unverified_autosend_receipts

    assert fetch_unverified_autosend_receipts(db_path=db_path, before="2099-01-01") == []
