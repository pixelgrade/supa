"""Delivery gates of the single support-agent path in services/processor.py.

There is one drafting path (the support pipeline with the agent's Opus -> Sonnet -> Codex
chain). These tests pin the safety gates it must keep: draft_only by default, auto-send only
at 8-10 with no blockers, only when the global mode is live and the customer is waiting, one
run per conversation, and no support work on manually excluded conversations.
"""
from unittest.mock import AsyncMock

import pytest

from agents import support_agent
from agents.draft_confidence import DraftConfidenceResult
from agents.support_agent import SupportInvestigationResult
from tests.test_processor_confidence_loop import (
    FakeHelpScout,
    FakeVectorStore,
    _support_result,
    _ticket_payload,
)

_MODE_ENV = ("SUPA_SUPPORT_MODE", "SUPA_SKILL_AGENT_MODE", "SUPA_SKILL_AGENT_ENABLED")


@pytest.fixture(autouse=True)
def _clean_agent_state(monkeypatch):
    for name in _MODE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_ENABLED", "false")
    monkeypatch.delenv("SUPA_SUPPORT_AUTOSEND_MIN_SCORE", raising=False)
    support_agent.reset_state()
    yield
    support_agent.reset_state()


def _make_processor(monkeypatch, tmp_path, fake_helpscout=None):
    from services.processor import TicketProcessor

    fhs = fake_helpscout or FakeHelpScout()
    fvs = FakeVectorStore(str(tmp_path / "vectors.db"))
    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fhs)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fvs)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr("services.processor.find_similar_tickets", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_customer_context", lambda _: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_: "")
    monkeypatch.setattr("services.processor.search_web", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_web_results", lambda _: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_: [])
    return TicketProcessor(auto_index=False), fhs


def _pipeline_returns(monkeypatch, result, captured=None):
    def fake_pipeline(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return result

    monkeypatch.setattr("services.processor.run_support_pipeline", fake_pipeline)


def _auto_send_result():
    return _support_result(
        delivery_mode="auto_send",
        confidence_score=9,
        decision_note="Supa decision note:\nMode: auto_send\nConfidence: 9/10",
    )


def _agent_replied_last(helpscout):
    helpscout.conversation_payloads[987] = {
        "id": 987,
        "_embedded": {"threads": [
            {"id": 1, "type": "customer", "createdAt": "2026-09-01T10:00:00Z",
             "createdBy": {"type": "customer"}},
            {"id": 2, "type": "message", "createdAt": "2026-09-01T11:00:00Z",
             "createdBy": {"type": "user"}},
        ]},
    }


# --- draft_only ceiling -------------------------------------------------------


async def test_default_mode_is_draft_only_and_never_sends(monkeypatch, tmp_path):
    _pipeline_returns(monkeypatch, _auto_send_result())
    processor, helpscout = _make_processor(monkeypatch, tmp_path)

    result = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert helpscout.sent_replies == []
    assert len(helpscout.drafts) == 1
    assert result["delivery_mode"] == "draft"
    note = helpscout.notes[0][1]
    assert "Mode: draft (auto-send suppressed by draft_only mode)" in note
    assert "Mode: auto_send" not in note


async def test_legacy_draft_only_setting_holds_without_the_old_enabled_switch(monkeypatch, tmp_path):
    """Production's preserved .env keeps SUPA_SKILL_AGENT_MODE=draft_only; it must keep
    holding every send now that the skill agent (and its ENABLED switch) is gone."""
    monkeypatch.setenv("SUPA_SKILL_AGENT_MODE", "draft_only")
    monkeypatch.setenv("SUPA_SKILL_AGENT_ENABLED", "false")
    _pipeline_returns(monkeypatch, _auto_send_result())
    processor, helpscout = _make_processor(monkeypatch, tmp_path)

    result = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert helpscout.sent_replies == []
    assert result["delivery_mode"] == "draft"


async def test_live_mode_with_a_waiting_customer_auto_sends(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "live")
    _pipeline_returns(monkeypatch, _auto_send_result())
    processor, helpscout = _make_processor(monkeypatch, tmp_path)

    result = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert result["delivery_mode"] == "auto_send"
    assert len(helpscout.sent_replies) == 1
    assert helpscout.drafts == []


async def test_live_mode_never_auto_sends_when_the_customer_is_not_waiting(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "live")
    _pipeline_returns(monkeypatch, _auto_send_result())
    helpscout = FakeHelpScout()
    _agent_replied_last(helpscout)
    processor, helpscout = _make_processor(monkeypatch, tmp_path, helpscout)

    result = await processor.process_ticket(_ticket_payload(), trigger="note_trigger")

    assert helpscout.sent_replies == []
    assert len(helpscout.drafts) == 1
    assert result["delivery_mode"] == "draft"
    assert "the customer is not waiting" in helpscout.notes[0][1]


# --- the 8/10 gate through the real pipeline ---------------------------------


def _real_pipeline(
    monkeypatch,
    *,
    score,
    hard_blockers=None,
    verdict="accept",
    reply_html="<p>Hey, Reader!</p><p>Verified answer.</p><p>Thank you,<br>Support</p>",
):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: SupportInvestigationResult(
            reply_html=reply_html,
            confidence_score=score,
            decision_summary="Verified.",
            evidence_summary="Docs",
            provider_name="claude",
            model="opus",
        ),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: DraftConfidenceResult(
            score=score,
            verdict=verdict,
            summary="Judge summary",
            hard_blockers=list(hard_blockers or []),
            provider="claude",
        ),
    )


@pytest.mark.parametrize("score,sent", [(7, False), (8, True), (9, True), (10, True)])
async def test_eight_and_above_auto_send(monkeypatch, tmp_path, score, sent):
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "live")
    _real_pipeline(monkeypatch, score=score)
    processor, helpscout = _make_processor(monkeypatch, tmp_path)

    result = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert bool(helpscout.sent_replies) is sent
    assert result["delivery_mode"] == ("auto_send" if sent else "draft")


async def test_a_lowered_threshold_setting_cannot_open_auto_send_below_eight(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "live")
    monkeypatch.setenv("SUPA_SUPPORT_AUTOSEND_MIN_SCORE", "5")
    _real_pipeline(monkeypatch, score=7)
    processor, helpscout = _make_processor(monkeypatch, tmp_path)

    result = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert helpscout.sent_replies == []
    assert result["delivery_mode"] == "draft"


async def test_a_raised_threshold_setting_keeps_an_eight_as_a_draft(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "live")
    monkeypatch.setenv("SUPA_SUPPORT_AUTOSEND_MIN_SCORE", "9")
    _real_pipeline(monkeypatch, score=8)
    processor, helpscout = _make_processor(monkeypatch, tmp_path)

    result = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert helpscout.sent_replies == []
    assert result["delivery_mode"] == "draft"


async def test_a_refund_offer_in_an_eight_requires_review(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "live")
    _real_pipeline(
        monkeypatch,
        score=8,
        reply_html="<p>Hey, Reader!</p><p>If neither path works, I'm happy to process a refund.</p><p>Thank you,<br>Support</p>",
    )
    processor, helpscout = _make_processor(monkeypatch, tmp_path)

    result = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert helpscout.sent_replies == []
    assert result["delivery_mode"] == "draft"
    assert "money_decision" in helpscout.notes[0][1]


async def test_a_hard_blocker_keeps_a_ten_as_a_manual_review_draft(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "live")
    _real_pipeline(
        monkeypatch, score=10, hard_blockers=["missing_critical_evidence"], verdict="manual_review"
    )
    processor, helpscout = _make_processor(monkeypatch, tmp_path)

    result = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert helpscout.sent_replies == []
    assert result["delivery_mode"] == "draft"
    assert "supa-manual-review" in helpscout.tag_updates[-1][1]
    assert "Blocked by:" in helpscout.notes[0][1]


# --- one path, one run, routing skips -----------------------------------------


async def test_drafting_has_no_second_provider_path(monkeypatch, tmp_path):
    captured = {}
    _pipeline_returns(monkeypatch, _support_result(delivery_mode="draft", confidence_score=7), captured)
    monkeypatch.setenv("SUPA_SUPPORT_AGENT_TIMEOUT", "321")
    processor, _ = _make_processor(monkeypatch, tmp_path)

    await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert "investigator_provider" not in captured
    assert captured["investigator_timeout"] == 321


async def test_claude_outage_routes_the_judge_to_codex(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setenv("SUPA_SUPPORT_EVALUATOR", "claude")
    monkeypatch.setattr("services.processor.should_skip_claude", lambda: True)
    _pipeline_returns(monkeypatch, _support_result(delivery_mode="draft", confidence_score=7), captured)
    processor, _ = _make_processor(monkeypatch, tmp_path)

    await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert captured["evaluator_provider"] == "codex"


async def test_a_concurrent_duplicate_run_is_skipped_without_touching_the_ticket(monkeypatch, tmp_path):
    pipeline = AsyncMock()
    monkeypatch.setattr("services.processor.run_support_pipeline", pipeline)
    processor, helpscout = _make_processor(monkeypatch, tmp_path)

    with support_agent.claim_conversation(987):
        result = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert result["status"] == "skipped"
    assert result["handled_by"] == "support_agent_inflight"
    pipeline.assert_not_called()
    assert helpscout.drafts == helpscout.notes == helpscout.tag_updates == []


async def test_supa_ignore_skips_the_support_agent(monkeypatch, tmp_path):
    pipeline = AsyncMock()
    monkeypatch.setattr("services.processor.run_support_pipeline", pipeline)
    processor, helpscout = _make_processor(monkeypatch, tmp_path)
    payload = _ticket_payload()
    payload["tags"].append({"tag": "supa-ignore"})

    result = await processor.process_ticket(payload, trigger="new_conversation")

    assert result["status"] == "skipped"
    pipeline.assert_not_called()
    assert helpscout.drafts == []
