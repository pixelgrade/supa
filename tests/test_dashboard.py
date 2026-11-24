import os
import asyncio
import copy
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from agents.draft_confidence import DraftConfidenceResult
from agents.support_agent import SupportInvestigationResult
from db.database import get_ticket_log_by_helpscout_id, init_db
from main import app
from services.support_pipeline import SupportPipelineResult
from services.agent_auth import AgentIdentity
from services.conversation_gateway import ConversationGateway, ConversationGatewayError
from tests.lease_recording import RecordingLeaseFactory


client = TestClient(app)


@patch("api.dashboard.get_claude_warning")
@patch("api.dashboard.get_processor")
def test_queue_includes_claude_warning(mock_get_processor, mock_get_claude_warning, monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)

    processor = Mock()
    processor.helpscout = SimpleNamespace(
        get_conversations_with_drafts=AsyncMock(return_value=[]),
        get_follow_ups=AsyncMock(return_value=[]),
    )
    processor.vector_store = SimpleNamespace(
        count_by_source=Mock(return_value={"docs": 10, "examples": 2, "internal": 1})
    )

    mock_get_processor.return_value = processor
    mock_get_claude_warning.return_value = {
        "code": "claude_credit_exhausted",
        "message": "Claude is unavailable and Supa is sending tickets to manual review.",
        "detail": "Credit balance is too low.",
        "occurred_at": "2026-03-27T18:00:00+00:00",
    }

    response = client.get("/dashboard/queue")

    assert response.status_code == 200
    body = response.json()
    assert body["warning"]["code"] == "claude_credit_exhausted"
    assert body["warning"]["detail"] == "Credit balance is too low."


def _support_result(*, delivery_mode="auto_send", confidence_score=9):
    investigation = SupportInvestigationResult(
        reply_html="<p>Agentic reply</p>",
        confidence_score=confidence_score,
        decision_summary="Investigated with repo evidence",
        evidence_summary="KB + repo evidence",
        investigation_steps=["Checked KB", "Checked GitHub repo"],
        risk_flags=["used_repo_evidence"],
        provider_name="codex",
        requires_manual_review=False,
        sources=["https://github.com/woodland-example/trail-atlas"],
    )
    evaluation = DraftConfidenceResult(
        score=confidence_score,
        verdict="accept" if delivery_mode == "auto_send" else "retry",
        summary="Safe to send" if delivery_mode == "auto_send" else "Save as draft",
        provider="claude",
    )
    return SupportPipelineResult(
        delivery_mode=delivery_mode,
        confidence_score=confidence_score,
        confidence_verdict=evaluation.verdict,
        reply_html=investigation.reply_html,
        decision_summary=investigation.decision_summary,
        evidence_summary=investigation.evidence_summary,
        investigation_steps=investigation.investigation_steps,
        risk_flags=investigation.risk_flags,
        hard_blockers=[] if delivery_mode == "auto_send" else ["missing_critical_evidence"],
        decision_note=f"Supa decision note:\nMode: {delivery_mode}",
        manual_review=delivery_mode != "auto_send",
        attempts=2 if delivery_mode != "auto_send" else 1,
        investigator_provider="codex",
        evaluator_provider="claude",
        investigator_result=investigation,
        evaluator_result=evaluation,
        trace=[{"attempt": 1, "score": confidence_score, "delivery_mode": delivery_mode}],
    )


@patch("api.dashboard.get_processor")
@patch("api.dashboard.run_support_pipeline", create=True)
@patch("api.dashboard.get_recent_semantic_failures", create=True)
@patch("api.dashboard.get_repo_evidence_context")
@patch("api.dashboard.format_past_tickets_context")
@patch("api.dashboard.find_similar_tickets")
@patch("api.dashboard.format_customer_context")
@patch("api.dashboard.fetch_customer_data")
@patch("api.dashboard.generate_draft_with_confidence", new_callable=AsyncMock, create=True)
@patch("api.dashboard.generate_subject")
@patch("api.dashboard.classify_ticket")
def test_test_ticket_returns_agentic_support_metadata(
    mock_classify_ticket,
    mock_generate_subject,
    mock_generate_draft_with_confidence,
    mock_fetch_customer_data,
    mock_format_customer_context,
    mock_find_similar_tickets,
    mock_format_past_tickets_context,
    mock_get_repo_evidence_context,
    mock_get_recent_semantic_failures,
    mock_run_support_pipeline,
    mock_get_processor,
    monkeypatch,
):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    mock_classify_ticket.return_value = "support"
    mock_generate_subject.return_value = "Better subject"
    mock_fetch_customer_data.return_value = {"plan": "pro"}
    mock_format_customer_context.return_value = "Customer context"
    mock_find_similar_tickets.return_value = [{"id": 12, "subject": "Similar ticket"}]
    mock_format_past_tickets_context.return_value = "Past ticket context"
    mock_get_repo_evidence_context.return_value = "Repo evidence context"
    mock_get_recent_semantic_failures.return_value = []
    mock_run_support_pipeline.return_value = _support_result(delivery_mode="draft", confidence_score=8)
    mock_generate_draft_with_confidence.return_value = SimpleNamespace(
        final_html="<p>Legacy reply</p>",
        final_score=8,
        final_status="manual_review",
        attempts=1,
        summary="Legacy summary",
        trace=[],
        provider="claude",
    )

    processor = Mock()
    processor.helpscout = SimpleNamespace(
        get_conversation=AsyncMock(
            return_value={
                "id": 601,
                "subject": "Original subject",
                "primaryCustomer": {
                    "id": 42,
                    "email": "Mica@example.com",
                    "first": "Mica",
                    "last": "Fern",
                },
                "tags": [{"tag": "trail-atlas"}],
                "_embedded": {
                    "threads": [
                        {
                            "type": "customer",
                            "body": '<p>Does the journal index include an appendix entry?</p>',
                            "createdBy": {"type": "customer"},
                        }
                    ]
                },
            }
        )
    )
    processor.vector_store = SimpleNamespace(
        search=Mock(
            return_value=[
                {
                    "title": "Map article",
                    "url": "https://woodland.example/docs/map",
                    "content": "Use both Map Panel and Google services",
                    "score": 0.82,
                }
            ]
        )
    )
    mock_get_processor.return_value = processor

    response = client.post("/dashboard/test-ticket", json={"conversation_id": 601})

    assert response.status_code == 200
    body = response.json()
    assert body["classification"] == "support"
    assert body["response_html"] == "<p>Agentic reply</p>"
    assert body["delivery_mode"] == "draft"
    assert body["confidence_score"] == 8
    assert body["investigator_provider"] == "codex"
    assert body["evaluator_provider"] == "claude"
    assert body["decision_note"] == "Supa decision note:\nMode: draft"
    assert body["evidence_summary"] == "KB + repo evidence"
    assert body["investigation_steps"] == ["Checked KB", "Checked GitHub repo"]
    assert body["risk_flags"] == ["used_repo_evidence"]
    assert body["sources"] == ["https://github.com/woodland-example/trail-atlas"]


@patch("api.dashboard.get_processor")
@patch("api.dashboard.run_support_pipeline", create=True)
@patch("api.dashboard.get_recent_semantic_failures", create=True)
@patch("api.dashboard.get_repo_evidence_context")
@patch("api.dashboard.format_past_tickets_context")
@patch("api.dashboard.find_similar_tickets")
@patch("api.dashboard.format_customer_context")
@patch("api.dashboard.fetch_customer_data")
@patch("api.dashboard.generate_subject")
@patch("api.dashboard.classify_ticket")
async def test_test_ticket_runs_pipeline_off_event_loop(
    mock_classify_ticket,
    mock_generate_subject,
    mock_fetch_customer_data,
    mock_format_customer_context,
    mock_find_similar_tickets,
    mock_format_past_tickets_context,
    mock_get_repo_evidence_context,
    mock_get_recent_semantic_failures,
    mock_run_support_pipeline,
    mock_get_processor,
    monkeypatch,
):
    """run_support_pipeline blocks for minutes (subprocess investigation/evaluation).
    It MUST be offloaded with asyncio.to_thread so the single uvicorn worker can still
    answer /health; otherwise the runtime watchdog times out and kills the app.
    Regression guard for that watchdog-restart loop on /dashboard/test-ticket.
    """
    # Import inside the test so pytest does not collect the handler/model as tests.
    from api.dashboard import test_ticket as handler, TestTicketRequest

    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    mock_classify_ticket.return_value = "support"
    mock_generate_subject.return_value = "Better subject"
    mock_fetch_customer_data.return_value = {"plan": "pro"}
    mock_format_customer_context.return_value = "Customer context"
    mock_find_similar_tickets.return_value = []
    mock_format_past_tickets_context.return_value = "Past ticket context"
    mock_get_repo_evidence_context.return_value = "Repo evidence context"
    mock_get_recent_semantic_failures.return_value = []

    loop_thread_id = threading.get_ident()
    recorded = {}

    def fake_pipeline(**kwargs):
        recorded["thread_id"] = threading.get_ident()
        return _support_result(delivery_mode="draft", confidence_score=8)

    mock_run_support_pipeline.side_effect = fake_pipeline

    processor = Mock()
    processor.helpscout = SimpleNamespace(
        get_conversation=AsyncMock(
            return_value={
                "id": 601,
                "subject": "Original subject",
                "primaryCustomer": {"id": 42, "email": "x@example.com", "first": "X", "last": "Y"},
                "tags": [{"tag": "trail-atlas"}],
                "_embedded": {"threads": [{"type": "customer", "body": "<p>hi</p>", "createdBy": {"type": "customer"}}]},
            }
        )
    )
    processor.vector_store = SimpleNamespace(search=Mock(return_value=[]))
    mock_get_processor.return_value = processor

    await handler(TestTicketRequest(conversation_id=601), _=True)

    assert recorded.get("thread_id") is not None, "run_support_pipeline was never called"
    assert recorded["thread_id"] != loop_thread_id, (
        "run_support_pipeline ran on the event-loop thread; it must be offloaded via "
        "asyncio.to_thread so /health stays responsive during a long investigation"
    )


@patch("api.dashboard.get_processor")
def test_post_cached_persists_agentic_fields_and_posts_draft(mock_get_processor, monkeypatch, tmp_path):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.setattr("api.dashboard.DB_PATH", str(tmp_path / "supa.db"))
    init_db(str(tmp_path / "supa.db"))

    helpscout = SimpleNamespace(
        update_subject=AsyncMock(return_value=True),
        add_draft=AsyncMock(return_value=True),
        add_tags=AsyncMock(return_value=True),
        add_note=AsyncMock(return_value=True),
    )
    mock_get_processor.return_value = SimpleNamespace(helpscout=helpscout)

    response = client.post(
        "/dashboard/post-cached",
        json={
            "conversation_id": 601,
            "customer_id": 42,
            "response_html": "<p>Agentic reply</p>",
            "new_subject": "Better subject",
            "theme_tag": "trail-atlas",
            "existing_tags": ["trail-atlas"],
            "confidence_score": 8,
            "confidence_status": "retry",
            "confidence_attempts": 2,
            "confidence_summary": "Save as draft",
            "confidence_trace": [{"attempt": 1, "score": 8}],
            "confidence_provider": "claude",
            "evidence_summary": "KB + repo evidence",
            "delivery_mode": "draft",
            "investigator_provider": "codex",
            "evaluator_provider": "claude",
            "decision_note": "Supa decision note:\nMode: draft",
            "investigation_steps": ["Checked KB", "Checked GitHub repo"],
            "risk_flags": ["used_repo_evidence"],
            "sources": ["https://github.com/woodland-example/trail-atlas"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    helpscout.update_subject.assert_awaited_once_with(601, "Better subject")
    helpscout.add_draft.assert_awaited_once_with(601, "<p>Agentic reply</p>", customer_id=42)
    helpscout.add_note.assert_awaited_once_with(601, "Supa decision note:\nMode: draft")

    row = get_ticket_log_by_helpscout_id(str(tmp_path / "supa.db"), "601")
    assert row is not None
    assert row["delivery_mode"] == "draft"
    assert row["investigator_provider"] == "codex"
    assert row["evaluator_provider"] == "claude"
    assert row["decision_note"] == "Supa decision note:\nMode: draft"
    assert row["confidence_score"] == 8


@patch("api.dashboard.get_processor")
def test_post_cached_returns_502_when_draft_save_fails(mock_get_processor, monkeypatch, tmp_path):
    """If HelpScout rejects the draft (e.g. 412 conversation-locked), add_draft
    returns False; post-cached must surface that as a non-200 rather than
    reporting success — otherwise the skill-agent reports OUTCOME: drafted for a
    draft that never landed."""
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.setattr("api.dashboard.DB_PATH", str(tmp_path / "supa.db"))
    init_db(str(tmp_path / "supa.db"))

    helpscout = SimpleNamespace(
        update_subject=AsyncMock(return_value=True),
        add_draft=AsyncMock(return_value=False),   # HelpScout write failed
        send_reply=AsyncMock(return_value=True),
        add_tags=AsyncMock(return_value=True),
        add_note=AsyncMock(return_value=True),
    )
    mock_get_processor.return_value = SimpleNamespace(helpscout=helpscout)

    response = client.post(
        "/dashboard/post-cached",
        json={"conversation_id": 601, "customer_id": 42,
              "response_html": "<p>Agentic reply</p>", "send": False},
    )

    assert response.status_code == 502
    # must NOT log the ticket as posted when the write failed
    row = get_ticket_log_by_helpscout_id(str(tmp_path / "supa.db"), "601")
    assert row is None or not row.get("response_posted")


@patch("api.dashboard.get_processor")
def test_post_cached_returns_502_when_send_fails(mock_get_processor, monkeypatch, tmp_path):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.setattr("api.dashboard.DB_PATH", str(tmp_path / "supa.db"))
    init_db(str(tmp_path / "supa.db"))

    helpscout = SimpleNamespace(
        update_subject=AsyncMock(return_value=True),
        add_draft=AsyncMock(return_value=True),
        send_reply=AsyncMock(return_value=False),  # HelpScout send failed
        add_tags=AsyncMock(return_value=True),
        add_note=AsyncMock(return_value=True),
    )
    mock_get_processor.return_value = SimpleNamespace(helpscout=helpscout)

    response = client.post(
        "/dashboard/post-cached",
        json={"conversation_id": 601, "customer_id": 42,
              "response_html": "<p>Agentic reply</p>", "send": True},
    )

    assert response.status_code == 502


@patch("api.dashboard.get_processor")
def test_send_reply_sets_pending_and_applies_auto_sent_tag(mock_get_processor, monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    session_db_path = os.environ["SUPA_DB_PATH"]
    init_db(session_db_path)

    helpscout = SimpleNamespace(
        get_conversation=AsyncMock(
            return_value={
                "id": 601,
                "subject": "Original subject",
                "primaryCustomer": {"id": 42, "first": "Jane", "last": "Doe"},
                "tags": [{"tag": "trail-atlas"}],
                "_embedded": {
                    "threads": [
                        {
                            "type": "customer",
                            "body": "<p>I need help.</p>",
                            "createdBy": {"type": "customer"},
                        }
                    ]
                },
            }
        ),
        delete_draft=AsyncMock(return_value=True),
        update_subject=AsyncMock(return_value=True),
        send_reply=AsyncMock(return_value=True),
        ensure_conversation_status=AsyncMock(
            return_value={
                "verified": True,
                "conversation": {
                    "id": 601,
                    "status": "pending",
                    "tags": [{"tag": "trail-atlas"}, {"tag": "supa-processed"}, {"tag": "supa-auto-sent"}],
                    "_embedded": {"threads": []},
                },
            }
        ),
        add_tags=AsyncMock(return_value=True),
    )
    processor = SimpleNamespace(helpscout=helpscout, index_sent_reply=Mock())
    mock_get_processor.return_value = processor

    response = client.post(
        "/dashboard/send-reply",
        json={
            "conversation_id": 601,
            "reply_html": "<p>Agentic reply</p>",
            "draft_thread_id": 999,
            "new_subject": "Better subject",
            "theme_tag": "trail-atlas",
            "existing_tags": ["trail-atlas"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "sent": True, "learned": True}
    helpscout.delete_draft.assert_awaited_once_with(601, 999)
    helpscout.update_subject.assert_awaited_once_with(601, "Better subject")
    helpscout.send_reply.assert_awaited_once_with(
        601,
        "<p>Agentic reply</p>",
        status="pending",
        customer_id=42,
    )
    helpscout.ensure_conversation_status.assert_awaited_once_with(601, "pending")
    helpscout.add_tags.assert_awaited_once_with(
        601,
        ["supa-processed", "supa-auto-sent"],
        ["trail-atlas"],
    )
    row = get_ticket_log_by_helpscout_id(session_db_path, "601")
    assert row is not None
    assert row["trigger"] == "dashboard"
    assert row["delivery_mode"] == "auto_send"


@patch("api.dashboard.get_processor")
def test_send_reply_deletes_all_drafts_logs_ticket_and_removes_manual_review_tag(mock_get_processor, monkeypatch, tmp_path):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.setattr("api.dashboard.DB_PATH", str(tmp_path / "supa.db"))
    init_db(str(tmp_path / "supa.db"))

    original_conversation = {
        "id": 601,
        "subject": "Original subject",
        "primaryCustomer": {"id": 42, "first": "Jane", "last": "Doe"},
        "tags": [{"tag": "trail-atlas"}, {"tag": "supa-manual-review"}],
        "_embedded": {
            "threads": [
                {
                    "id": 998,
                    "type": "message",
                    "state": "draft",
                    "body": "<p>Older draft</p>",
                    "createdAt": "2026-04-09T10:00:00Z",
                },
                {
                    "id": 999,
                    "type": "message",
                    "state": "draft",
                    "body": "<p>Newest draft</p>",
                    "createdAt": "2026-04-09T11:00:00Z",
                },
                {
                    "type": "customer",
                    "body": "<p>I need help.</p>",
                    "createdBy": {"type": "customer"},
                },
            ]
        },
    }
    verified_conversation = {
        "id": 601,
        "status": "pending",
        "subject": "Better subject",
        "tags": [{"tag": "trail-atlas"}, {"tag": "supa-processed"}, {"tag": "supa-auto-sent"}],
        "_embedded": {"threads": []},
    }
    helpscout = SimpleNamespace(
        get_conversation=AsyncMock(return_value=original_conversation),
        delete_draft=AsyncMock(return_value=True),
        update_subject=AsyncMock(return_value=True),
        send_reply=AsyncMock(return_value=True),
        ensure_conversation_status=AsyncMock(return_value={"verified": True, "conversation": verified_conversation}),
        add_tags=AsyncMock(return_value=True),
    )
    processor = SimpleNamespace(helpscout=helpscout, index_sent_reply=Mock())
    mock_get_processor.return_value = processor

    response = client.post(
        "/dashboard/send-reply",
        json={
            "conversation_id": 601,
            "reply_html": "<p>Agentic reply</p>",
            "new_subject": "Better subject",
            "theme_tag": "trail-atlas",
            "existing_tags": ["trail-atlas", "supa-manual-review"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "sent": True, "learned": True}
    assert [awaited.args for awaited in helpscout.delete_draft.await_args_list] == [
        (601, 998),
        (601, 999),
    ]
    helpscout.add_tags.assert_awaited_once_with(
        601,
        ["supa-processed", "supa-auto-sent"],
        ["trail-atlas"],
    )

    row = get_ticket_log_by_helpscout_id(str(tmp_path / "supa.db"), "601")
    assert row is not None
    assert row["status"] == "processed"
    assert row["response_posted"] == 1
    assert row["delivery_mode"] == "auto_send"
    assert row["trigger"] == "dashboard"


@patch("api.dashboard.get_processor")
def test_post_cached_send_true_sends_reply_and_logs(mock_get_processor, monkeypatch, tmp_path):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.setattr("api.dashboard.DB_PATH", str(tmp_path / "supa.db"))
    init_db(str(tmp_path / "supa.db"))

    helpscout = SimpleNamespace(
        update_subject=AsyncMock(return_value=True),
        add_draft=AsyncMock(return_value=True),
        send_reply=AsyncMock(return_value=True),
        add_tags=AsyncMock(return_value=True),
        add_note=AsyncMock(return_value=True),
    )
    mock_get_processor.return_value = SimpleNamespace(helpscout=helpscout)

    response = client.post(
        "/dashboard/post-cached",
        json={
            "conversation_id": 602,
            "customer_id": 7,
            "response_html": "<p>Auto-send reply</p>",
            "existing_tags": [],
            "confidence_score": 10,
            "confidence_status": "accept",
            "delivery_mode": "auto_send",
            "send": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["posted"] is True
    helpscout.send_reply.assert_awaited_once_with(602, "<p>Auto-send reply</p>", customer_id=7)
    helpscout.add_draft.assert_not_awaited()
    row = get_ticket_log_by_helpscout_id(str(tmp_path / "supa.db"), "602")
    assert row is not None
    assert row["delivery_mode"] == "auto_send"
    assert row["response_posted"] == 1


@patch("api.dashboard.get_processor")
def test_post_cached_send_false_drafts_not_sends(mock_get_processor, monkeypatch, tmp_path):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.setattr("api.dashboard.DB_PATH", str(tmp_path / "supa.db"))
    init_db(str(tmp_path / "supa.db"))

    helpscout = SimpleNamespace(
        update_subject=AsyncMock(return_value=True),
        add_draft=AsyncMock(return_value=True),
        send_reply=AsyncMock(return_value=True),
        add_tags=AsyncMock(return_value=True),
        add_note=AsyncMock(return_value=True),
    )
    mock_get_processor.return_value = SimpleNamespace(helpscout=helpscout)

    response = client.post(
        "/dashboard/post-cached",
        json={
            "conversation_id": 603,
            "customer_id": 7,
            "response_html": "<p>Draft reply</p>",
            "existing_tags": [],
            "confidence_score": 7,
            "confidence_status": "retry",
            "delivery_mode": "draft",
        },
    )

    assert response.status_code == 200
    helpscout.add_draft.assert_awaited_once_with(603, "<p>Draft reply</p>", customer_id=7)
    helpscout.send_reply.assert_not_awaited()


@patch("api.dashboard.get_processor")
def test_dashboard_post_cached_mutations_use_one_guarded_conversation_lease(
    mock_get_processor, monkeypatch, tmp_path
):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.setattr("api.dashboard.DB_PATH", str(tmp_path / "supa.db"))
    init_db(str(tmp_path / "supa.db"))
    leases = RecordingLeaseFactory()
    monkeypatch.setattr("api.dashboard.conversation_lease", leases)
    helpscout = SimpleNamespace(
        get_conversation=AsyncMock(return_value={"tags": [{"tag": "fresh-tag"}]}),
        update_subject=AsyncMock(return_value=True),
        add_draft=AsyncMock(return_value=True),
        send_reply=AsyncMock(return_value=True),
        add_tags=AsyncMock(return_value=True),
        add_note=AsyncMock(return_value=True),
    )
    mock_get_processor.return_value = SimpleNamespace(helpscout=helpscout)

    response = client.post(
        "/dashboard/post-cached",
        json={
            "conversation_id": 604,
            "customer_id": 7,
            "response_html": "<p>Draft reply</p>",
            "new_subject": "Fresh subject",
            "existing_tags": [],
            "decision_note": "Supa decision note",
        },
    )

    assert response.status_code == 200
    assert [event[0] for event in leases.events].count("enter") == 1
    leases.assert_guarded_count(604, 4)
    assert helpscout.add_tags.await_args.args[2] == ["fresh-tag"]


@patch("api.dashboard.get_processor")
def test_dashboard_send_reply_refetch_and_mutations_share_guarded_lease(
    mock_get_processor, monkeypatch, tmp_path
):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.setattr("api.dashboard.DB_PATH", str(tmp_path / "supa.db"))
    init_db(str(tmp_path / "supa.db"))
    leases = RecordingLeaseFactory()
    monkeypatch.setattr("api.dashboard.conversation_lease", leases)
    helpscout = SimpleNamespace(
        get_conversation=AsyncMock(return_value={
            "id": 605,
            "subject": "Current",
            "primaryCustomer": {"id": 7},
            "tags": [],
            "_embedded": {"threads": []},
        }),
        delete_draft=AsyncMock(return_value=True),
        update_subject=AsyncMock(return_value=True),
        send_reply=AsyncMock(return_value=True),
        ensure_conversation_status=AsyncMock(return_value={"verified": True}),
        add_tags=AsyncMock(return_value=True),
    )
    mock_get_processor.return_value = SimpleNamespace(
        helpscout=helpscout, index_sent_reply=Mock()
    )

    response = client.post(
        "/dashboard/send-reply",
        json={
            "conversation_id": 605,
            "reply_html": "<p>Reply</p>",
            "draft_thread_id": 44,
        },
    )

    assert response.status_code == 200
    assert leases.events[0][0] == "enter"
    assert helpscout.get_conversation.await_count == 1
    leases.assert_guarded_count(605, 4)


@pytest.mark.asyncio
async def test_dashboard_draft_and_agent_gateway_serialize_on_same_conversation(
    monkeypatch, tmp_path
):
    from api.dashboard import PostCachedRequest, post_cached

    db_path = str(tmp_path / "shared-writes.db")
    init_db(db_path)
    monkeypatch.setattr("api.dashboard.DB_PATH", db_path)
    dashboard_started = asyncio.Event()
    release_dashboard = asyncio.Event()

    class SharedClient:
        def __init__(self):
            self.conversation = {
                "id": 606,
                "subject": "Question",
                "status": "active",
                "tags": [],
                "primaryCustomer": {"id": 7},
                "_embedded": {"threads": []},
            }
            self.gateway_reads = 0
            self.gateway_creates = 0

        async def add_draft(self, conversation_id, text, customer_id=None):
            dashboard_started.set()
            await release_dashboard.wait()
            self.conversation["_embedded"]["threads"].append({
                "id": 501,
                "type": "message",
                "state": "draft",
                "draft": True,
                "body": text,
                "createdAt": "2026-09-25T12:00:00Z",
                "createdBy": {"type": "user", "id": 9},
                "source": {"type": "api"},
            })
            return True

        async def add_tags(self, *_args, **_kwargs):
            return True

        async def get_complete_conversation(self, _conversation_id):
            self.gateway_reads += 1
            return copy.deepcopy(self.conversation)

        async def add_draft_result(self, *_args, **_kwargs):
            self.gateway_creates += 1
            raise AssertionError("agent write must observe the dashboard draft")

    shared = SharedClient()
    monkeypatch.setattr(
        "api.dashboard.get_processor",
        lambda: SimpleNamespace(helpscout=shared),
    )
    dashboard_task = asyncio.create_task(post_cached(PostCachedRequest(
        conversation_id=606,
        customer_id=7,
        response_html="<p>Dashboard draft</p>",
    ), _=True))
    await dashboard_started.wait()

    gateway = ConversationGateway(shared, db_path=db_path)
    agent_task = asyncio.create_task(gateway.save_draft(
        606,
        body="Agent draft",
        body_format="text",
        expected_version=None,
        idempotency_key="dashboard-concurrency-agent",
        identity=AgentIdentity(
            token_id="writer-dashboard-race",
            name="Writer",
            scopes=frozenset({"read", "draft"}),
        ),
        request_id="request-dashboard-concurrency",
    ))
    await asyncio.sleep(0)
    assert shared.gateway_reads == 0

    release_dashboard.set()
    assert (await dashboard_task)["posted"] is True
    with pytest.raises(ConversationGatewayError) as conflict:
        await agent_task
    assert conflict.value.code == "draft_conflict"
    assert shared.gateway_reads >= 1
    assert shared.gateway_creates == 0


# --- auto-send-ready surfacing ----------------------------------------

from api.dashboard import _is_auto_send_ready  # noqa: E402


def test_auto_send_ready_flags_a_withheld_high_confidence_draft():
    """An accepted draft awaiting delivery stays visible to the operator."""
    assert _is_auto_send_ready({"confidence_score": 9, "confidence_status": "accept"}) is True
    assert _is_auto_send_ready({"confidence_score": 10, "confidence_status": ""}) is True


def test_auto_send_ready_excludes_drafts_that_did_not_clear_the_bar(monkeypatch):
    monkeypatch.delenv("SUPA_SUPPORT_AUTOSEND_MIN_SCORE", raising=False)
    assert _is_auto_send_ready({"confidence_score": 8, "confidence_status": "accept"}) is True  # bar is 8
    assert _is_auto_send_ready({"confidence_score": 7, "confidence_status": "accept"}) is False
    assert _is_auto_send_ready({"confidence_score": 9, "confidence_status": "manual_review"}) is False
    assert _is_auto_send_ready({"confidence_score": 9, "confidence_status": "retry"}) is False
    assert _is_auto_send_ready({"confidence_score": None, "confidence_status": "accept"}) is False
    assert _is_auto_send_ready({}) is False


def test_auto_send_ready_excludes_high_scored_draft_with_blockers():
    assert _is_auto_send_ready({
        "confidence_score": 10,
        "confidence_status": "accept",
        "decision_note": "Supa decision note:\nMode: draft\nBlocked by: manual_review, account_action_unverified",
    }) is False
