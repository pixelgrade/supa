"""Integration tests for the processor auto-send -> pending_sends scheduling."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from db.database import (
    init_db,
    fetch_scheduled_pending_sends_by_conversation,
    insert_pending_send,
    get_db,
)
from services.autosend_scheduler import sha256_of_html


def _build_conversation_after_draft(draft_thread_id: str, draft_body: str, existing_tags=None):
    return {
        "id": 100,
        "tags": existing_tags or ["supa-processed"],
        "_embedded": {
            "threads": [
                {"id": "200", "type": "customer", "body": "<p>Need help</p>",
                 "createdAt": "2026-04-19T09:00:00+00:00",
                 "createdBy": {"type": "customer"}, "source": {"type": "email"}},
                {"id": draft_thread_id, "type": "message", "state": "draft",
                 "body": draft_body, "createdAt": "2026-04-20T09:59:30+00:00",
                 "source": {"type": "api"}},
            ]
        },
    }


@pytest.mark.asyncio
async def test_autosend_branch_writes_draft_and_schedules_row(tmp_path, monkeypatch):
    from services.processor import process_auto_send

    db_path = str(tmp_path / "supa.db")
    init_db(db_path)
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_ENABLED", "true")

    response_html = "<p>Opening entry</p><p>The next chapter is ready.</p><p>Appendix<br>Reading list</p>"
    draft_thread_id = "500"

    helpscout = AsyncMock()
    helpscout.get_conversation = AsyncMock(
        side_effect=[
            _build_conversation_after_draft("old", ""),
            _build_conversation_after_draft(draft_thread_id, response_html),
        ]
    )
    helpscout.add_draft = AsyncMock(return_value=True)
    helpscout.send_reply = AsyncMock()
    helpscout.ensure_conversation_status = AsyncMock()

    tags_to_add = ["supa-processed"]
    result = await process_auto_send(
        helpscout=helpscout,
        conversation_id=100,
        customer_id=55,
        response_html=response_html,
        tags_to_add=tags_to_add,
        existing_tags=["supa-processed"],
        confidence_score=10,
        confidence_provider="claude",
        db_path=db_path,
    )

    helpscout.send_reply.assert_not_awaited()
    helpscout.ensure_conversation_status.assert_not_awaited()
    helpscout.add_draft.assert_awaited_once_with(100, response_html, 55)

    rows = fetch_scheduled_pending_sends_by_conversation(db_path=db_path, conversation_id="100")
    assert len(rows) == 1
    row = rows[0]
    assert row["draft_thread_id"] == "500"
    assert row["response_html_sha"] == sha256_of_html(response_html)
    assert row["tags_on_promote"] == ["supa-auto-sent"]

    assert "supa-pending-send" in tags_to_add
    assert "supa-auto-sent" not in tags_to_add

    assert result["response_posted"] is True


@pytest.mark.asyncio
async def test_autosend_kill_switch_falls_back_to_immediate_send(tmp_path, monkeypatch):
    from services.processor import process_auto_send

    db_path = str(tmp_path / "supa.db")
    init_db(db_path)
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_ENABLED", "false")

    response_html = "<p>body</p>"
    helpscout = AsyncMock()
    helpscout.send_reply = AsyncMock(return_value=True)
    helpscout.ensure_conversation_status = AsyncMock(return_value={"verified": True, "conversation": {}})
    helpscout.get_conversation = AsyncMock(return_value=_build_conversation_after_draft("old", ""))
    helpscout.add_draft = AsyncMock()

    tags_to_add = ["supa-processed"]
    await process_auto_send(
        helpscout=helpscout, conversation_id=100, customer_id=55,
        response_html=response_html, tags_to_add=tags_to_add,
        existing_tags=["supa-processed"],
        confidence_score=10, confidence_provider="claude", db_path=db_path,
    )

    helpscout.send_reply.assert_awaited_once()
    helpscout.add_draft.assert_not_awaited()
    rows = fetch_scheduled_pending_sends_by_conversation(db_path=db_path, conversation_id="100")
    assert rows == []
    assert "supa-auto-sent" in tags_to_add


@pytest.mark.asyncio
async def test_autosend_supersedes_prior_scheduled_row(tmp_path, monkeypatch):
    from services.processor import process_auto_send

    db_path = str(tmp_path / "supa.db")
    init_db(db_path)
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_ENABLED", "true")

    prior_id = insert_pending_send(
        db_path=db_path, conversation_id="100", draft_thread_id="499",
        response_html_sha="old", due_at="2099-01-01T00:00:00+00:00",
        last_known_customer_thread_id="200",
        tags_on_promote=["supa-auto-sent"], confidence_score=10,
        confidence_provider="claude",
    )

    response_html = "<p>new</p>"
    helpscout = AsyncMock()
    helpscout.get_conversation = AsyncMock(
        side_effect=[
            _build_conversation_after_draft("old", ""),
            _build_conversation_after_draft("500", response_html),
        ]
    )
    helpscout.add_draft = AsyncMock(return_value=True)

    await process_auto_send(
        helpscout=helpscout, conversation_id=100, customer_id=55,
        response_html=response_html, tags_to_add=[], existing_tags=[],
        confidence_score=10, confidence_provider="claude", db_path=db_path,
    )

    db = get_db(db_path)
    prior = db.execute(
        "SELECT state, cancel_reason FROM pending_sends WHERE id=?", (prior_id,)
    ).fetchone()
    db.close()
    assert prior["state"] == "cancelled"
    assert prior["cancel_reason"] == "superseded"

    scheduled = fetch_scheduled_pending_sends_by_conversation(db_path=db_path, conversation_id="100")
    assert len(scheduled) == 1
    assert scheduled[0]["draft_thread_id"] == "500"


@pytest.mark.parametrize("persisted,allowed", [
    ("<p>Project » Content Builder</p>", True),
    ("<p>A changed instruction</p>", False),
])
async def test_queue_pins_verified_readback_instead_of_pre_normalization_html(tmp_path, persisted, allowed):
    from services.processor import process_auto_send
    from services.autosend_scheduler import AutoSendConfig
    db_path = str(tmp_path / "supa.db")
    init_db(db_path)
    intended = "<p>Project &raquo; Content Builder</p>"
    helpscout = AsyncMock()
    helpscout.get_conversation.side_effect = [
        _build_conversation_after_draft("old", ""),
        _build_conversation_after_draft("500", persisted),
    ]
    helpscout.add_draft.return_value = True
    kwargs = dict(helpscout=helpscout, conversation_id=100, customer_id=55, response_html=intended,
                  tags_to_add=[], existing_tags=[], confidence_score=8, confidence_provider="claude",
                  db_path=db_path, config=AutoSendConfig(300, 1200, 30, True))

    if allowed:
        await process_auto_send(**kwargs)
        row = fetch_scheduled_pending_sends_by_conversation(db_path=db_path, conversation_id="100")[0]
        assert row["response_html_sha"] == sha256_of_html(persisted)
    else:
        with pytest.raises(RuntimeError):
            await process_auto_send(**kwargs)
        assert fetch_scheduled_pending_sends_by_conversation(db_path=db_path, conversation_id="100") == []


@pytest.mark.asyncio
async def test_end_to_end_queue_then_worker_promotes(tmp_path, monkeypatch):
    """Processor queues an auto-send; a past-due worker tick promotes it."""
    from services.processor import process_auto_send
    from services.autosend_scheduler import run_worker_loop, load_config

    db_path = str(tmp_path / "supa.db")
    init_db(db_path)
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_ENABLED", "true")
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_MIN_SECONDS", "0")
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_MAX_SECONDS", "0")

    response_html = "<p>Chapter complete.</p><p>Appendix<br>Reading list</p>"
    draft_thread_id = "500"

    convo_state = {
        "id": 100,
        "primaryCustomer": {"id": 55},
        "tags": ["supa-processed"],
        "_embedded": {"threads": [
            {"id": "200", "type": "customer", "body": "<p>help</p>",
             "createdAt": "2026-04-19T09:00:00+00:00",
             "createdBy": {"type": "customer"}, "source": {"type": "email"}},
        ]},
    }

    async def fake_get_conversation(_conversation_id):
        import copy
        return copy.deepcopy(convo_state)

    async def fake_add_draft(_conversation_id, body, _customer):
        convo_state["_embedded"]["threads"].append({
            "id": draft_thread_id, "type": "message", "state": "draft",
            "body": body, "createdAt": "2026-04-20T09:59:30+00:00",
            "source": {"type": "api"},
        })
        return True

    send_called = {"n": 0}
    async def fake_send_reply(_conversation_id, _body, customer_id=None):
        send_called["n"] += 1
        assert customer_id == 55
        return True

    async def fake_ensure_status(_conversation_id, _status):
        return {"verified": True, "conversation": {}}

    async def fake_add_tags(_conversation_id, _tags, _existing):
        return True

    async def fake_remove_tag(_conversation_id, _tag, _existing):
        return True

    async def fake_delete_draft(_conversation_id, _thread_id):
        return True

    helpscout = AsyncMock()
    helpscout.get_conversation = fake_get_conversation
    helpscout.add_draft = fake_add_draft
    helpscout.send_reply = fake_send_reply
    helpscout.ensure_conversation_status = fake_ensure_status
    helpscout.add_tags = fake_add_tags
    helpscout.remove_tag = fake_remove_tag
    helpscout.delete_draft = fake_delete_draft

    tags_to_add = ["supa-processed"]
    await process_auto_send(
        helpscout=helpscout, conversation_id=100, customer_id=55,
        response_html=response_html, tags_to_add=tags_to_add,
        existing_tags=["supa-processed"],
        confidence_score=10, confidence_provider="claude", db_path=db_path,
    )

    rows = fetch_scheduled_pending_sends_by_conversation(db_path=db_path, conversation_id="100")
    assert len(rows) == 1

    async def fake_sleep(_):
        return
    await run_worker_loop(
        db_path=db_path, helpscout=helpscout, config=load_config(),
        iterations=1, sleep_func=fake_sleep,
    )

    assert send_called["n"] == 1
    db = get_db(db_path)
    row = db.execute("SELECT state FROM pending_sends WHERE conversation_id='100'").fetchone()
    db.close()
    assert row["state"] == "sent"
