from copy import deepcopy
from datetime import datetime, timedelta, timezone

from tests.test_processor_support_agent_gate import (
    _make_processor, _pipeline_returns, _auto_send_result,
)
from tests.test_processor_confidence_loop import _ticket_payload, _support_result


def payload(thread_id=1):
    data = _ticket_payload()
    data["_embedded"]["threads"][0].update(id=thread_id, createdAt="2026-04-03T09:00:00Z")
    return data


async def test_held_run_cannot_be_rerolled_into_an_automatic_send(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "live")
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_ENABLED", "false")
    processor, hs = _make_processor(monkeypatch, tmp_path)
    _pipeline_returns(monkeypatch, _support_result(delivery_mode="draft", confidence_score=7))
    await processor.process_ticket(payload())
    _pipeline_returns(monkeypatch, _auto_send_result())
    result = await processor.process_ticket(payload(), trigger="manual_run", delivery_ceiling="policy")
    assert result["delivery_mode"] == "draft"
    assert hs.sent_replies == []
    assert "rerun" in result["decision_note"].lower()
    # A different customer message retains the normal 8/10 gate.
    newer = payload(2)
    newer["_embedded"]["threads"][0]["createdAt"] = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
    hs.conversation_payloads[987] = deepcopy(newer)
    _pipeline_returns(monkeypatch, _auto_send_result())
    result = await processor.process_ticket(newer)
    assert result["delivery_mode"] == "auto_send"


async def test_new_customer_reply_during_investigation_never_gets_stale_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "live")
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_ENABLED", "false")
    processor, hs = _make_processor(monkeypatch, tmp_path)
    hs.conversation_payloads[987] = payload(2)
    _pipeline_returns(monkeypatch, _auto_send_result())
    result = await processor.process_ticket(payload(1))
    assert result["delivery_mode"] == "draft"
    assert hs.sent_replies == []
    assert "changed" in result["decision_note"].lower()
    assert hs.drafts == []
