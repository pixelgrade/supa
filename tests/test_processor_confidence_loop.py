import pytest
from unittest.mock import AsyncMock

from agents.draft_confidence import DraftConfidenceResult
from agents.support_agent import SupportInvestigationResult
from db.database import init_db
from db.database import get_ticket_log_by_helpscout_id, log_ticket
from services.processor import TicketProcessor
from services.processor import generate_draft_with_confidence
from services.processor import _has_unsafe_visible_draft_blocker, _should_preserve_existing_draft
from services.draft_confidence_loop import run_draft_confidence_loop
from services.support_pipeline import SupportPipelineResult
from tests.lease_recording import RecordingLeaseFactory


def test_manual_recovery_keeps_existing_draft_when_equal_score_rerun_is_placeholder():
    existing = {
        "confidence_score": 0,
        "draft_response_html": "<p>Useful draft awaiting human review.</p>",
        "response_posted": 1,
        "delivery_mode": "draft",
    }
    assert _should_preserve_existing_draft(
        trigger="manual_run",
        existing_row=existing,
        response_html="<p>Manual review required before sending a reply.</p>",
        confidence_score=0,
        risk_flags=["empty_draft"],
        hard_blockers=["manual_review"],
    ) is True
    assert _should_preserve_existing_draft(
        trigger="manual_run",
        existing_row=existing,
        response_html="<p>New substantive draft.</p>",
        confidence_score=0,
        risk_flags=[],
        hard_blockers=[],
    ) is False


def test_unverified_completed_account_action_is_not_left_as_visible_draft():
    assert _has_unsafe_visible_draft_blocker(["unverified_account_action_claim"]) is True


@pytest.fixture(autouse=True)
def _disable_autosend_delay_by_default(monkeypatch):
    """These tests validate the legacy immediate-send path. Scheduled-send
    behavior is covered by tests/test_autosend_processor_integration.py."""
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_ENABLED", "false")


@pytest.fixture(autouse=True)
def _live_support_mode(monkeypatch):
    """These tests cover live-mode delivery. Production runs draft_only (the default);
    that ceiling is covered by tests/test_processor_support_agent_gate.py."""
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "live")
    from agents import support_agent
    support_agent.reset_state()


@pytest.mark.asyncio
async def test_confidence_loop_retries_until_target_score():
    scores = iter([6, 9])
    rewrite_calls = []
    investigate_calls = []

    def evaluate_fn(*, draft_html, **_kwargs):
        score = next(scores)
        return DraftConfidenceResult(
            score=score,
            verdict="retry" if score < 8 else "accept",
            summary=f"score {score}",
            investigation_actions=["expand_kb_search"] if score < 8 else [],
            rewrite_brief="Make it more specific" if score < 8 else "",
        )

    async def investigate_fn(*, evaluation, attempt, context):
        investigate_calls.append((evaluation.score, attempt, context["knowledge_context"]))
        return {"additional_context": f"extra context {attempt}", "knowledge_context": context["knowledge_context"] + " more"}

    async def rewrite_fn(*, original_draft, rewrite_brief, additional_context, **_kwargs):
        rewrite_calls.append((original_draft, rewrite_brief, additional_context))
        return "<p>retry draft</p>"

    result = await run_draft_confidence_loop(
        initial_html="<p>initial draft</p>",
        initial_context={"knowledge_context": "base"},
        evaluate_fn=evaluate_fn,
        investigate_fn=investigate_fn,
        rewrite_fn=rewrite_fn,
        retry_threshold=8,
        target_score=9,
        max_attempts=3,
    )

    assert result.final_score == 9
    assert result.final_status == "accepted"
    assert result.attempts == 2
    assert result.final_html == "<p>retry draft</p>"
    assert len(rewrite_calls) == 1
    assert investigate_calls == [(6, 1, "base")]


@pytest.mark.asyncio
async def test_confidence_loop_stops_at_manual_review_after_max_attempts():
    scores = iter([6, 7, 7])

    def evaluate_fn(*, draft_html, **_kwargs):
        score = next(scores)
        return DraftConfidenceResult(
            score=score,
            verdict="retry",
            summary=f"score {score}",
            investigation_actions=["expand_kb_search"],
            rewrite_brief=f"rewrite for {score}",
        )

    async def investigate_fn(*, evaluation, attempt, context):
        return {"additional_context": f"ctx {attempt}", "knowledge_context": context["knowledge_context"]}

    async def rewrite_fn(*, original_draft, rewrite_brief, additional_context, **_kwargs):
        return f"<p>{rewrite_brief}|{additional_context}</p>"

    result = await run_draft_confidence_loop(
        initial_html="<p>initial draft</p>",
        initial_context={"knowledge_context": "base"},
        evaluate_fn=evaluate_fn,
        investigate_fn=investigate_fn,
        rewrite_fn=rewrite_fn,
        retry_threshold=8,
        target_score=9,
        max_attempts=3,
    )

    assert result.final_status == "manual_review"
    assert result.final_score == 7
    assert result.attempts == 3
    assert result.final_html == "<p>rewrite for 6|ctx 1</p>" or result.final_html == "<p>rewrite for 7|ctx 2</p>"


@pytest.mark.asyncio
async def test_generate_draft_with_confidence_returns_loop_result(monkeypatch):
    async def fake_generate_response(**_kwargs):
        return "<p>initial</p>"

    async def fake_rewrite_response(**_kwargs):
        return "<p>rewritten</p>"

    def fake_evaluate_draft(**_kwargs):
        return DraftConfidenceResult(
            score=9,
            verdict="accept",
            summary="ready",
            investigation_actions=[],
            rewrite_brief="",
        )

    async def fake_run_loop(**kwargs):
        evaluate_fn = kwargs["evaluate_fn"]
        evaluation = evaluate_fn(draft_html=kwargs["initial_html"], **kwargs["initial_context"])
        return type("LoopResult", (), {
            "final_html": kwargs["initial_html"],
            "final_score": evaluation.score,
            "final_status": "accepted",
            "attempts": 1,
            "summary": evaluation.summary,
            "trace": [{"attempt": 1, "score": evaluation.score}],
            "provider": evaluation.provider,
        })()

    monkeypatch.setattr("services.processor.generate_response", fake_generate_response)
    monkeypatch.setattr("services.processor.rewrite_response", fake_rewrite_response)
    monkeypatch.setattr("services.processor.evaluate_draft", fake_evaluate_draft)
    monkeypatch.setattr("services.processor.run_draft_confidence_loop", fake_run_loop)

    result = await generate_draft_with_confidence(
        customer_name="Reader",
        subject="Journal display",
        message="Street is not recognized",
        theme_tag="field-journal",
        knowledge_context="context",
        past_tickets_context="past",
        customer_context="customer",
    )

    assert result.final_html == "<p>initial</p>"
    assert result.final_score == 9
    assert result.final_status == "accepted"
    assert result.summary == "ready"


@pytest.mark.asyncio
async def test_confidence_loop_applies_guard_before_accepting():
    scores = iter([9, 9])

    def evaluate_fn(*, draft_html, **_kwargs):
        score = next(scores)
        return DraftConfidenceResult(
            score=score,
            verdict="accept",
            summary=f"score {score}",
            investigation_actions=[],
            rewrite_brief="",
        )

    def guard_fn(*, draft_html, evaluation, **_kwargs):
        if "initial" in draft_html:
            return DraftConfidenceResult(
                score=7,
                verdict="retry",
                summary="guarded",
                investigation_actions=["expand_past_ticket_search"],
                rewrite_brief="fix contradiction",
            )
        return evaluation

    async def investigate_fn(*, evaluation, attempt, context):
        return {"additional_context": "more evidence", "knowledge_context": context["knowledge_context"]}

    async def rewrite_fn(*, original_draft, rewrite_brief, additional_context, **_kwargs):
        return "<p>rewritten</p>"

    result = await run_draft_confidence_loop(
        initial_html="<p>initial</p>",
        initial_context={"knowledge_context": "base evidence"},
        evaluate_fn=evaluate_fn,
        investigate_fn=investigate_fn,
        rewrite_fn=rewrite_fn,
        guard_fn=guard_fn,
        retry_threshold=8,
        target_score=9,
        max_attempts=3,
    )

    assert result.final_status == "accepted"
    assert result.attempts == 2
    assert result.final_html == "<p>rewritten</p>"
    assert result.trace[0]["score"] == 7


class FakeHelpScout:
    def __init__(self):
        self.closed = []
        self.status_updates = []
        self.tag_updates = []
        self.subject_updates = []
        self.sent_replies = []
        self.drafts = []
        self.notes = []
        self.deleted_drafts = []
        self.updated_drafts = []
        self.fetched_conversations = []
        self.conversation_payloads = {}
        self.fail_reply = False
        self.fail_status = False
        self.fail_draft = False
        self.fail_on_note = False
        self.fail_on_tags = False
        self.delete_draft_failures = set()
        self.update_draft_failures = set()

    async def close_conversation(self, conversation_id):
        self.closed.append(conversation_id)
        return True

    async def update_subject(self, conversation_id, new_subject):
        self.subject_updates.append((conversation_id, new_subject))
        return True

    async def set_conversation_status(self, conversation_id, status):
        self.status_updates.append((conversation_id, status))
        if self.fail_status:
            return False
        return True

    async def ensure_conversation_status(self, conversation_id, status):
        ok = await self.set_conversation_status(conversation_id, status)
        return {
            "verified": ok,
            "conversation": {"id": conversation_id, "status": status if ok else "active"},
        }

    async def send_reply(self, conversation_id, text, status="pending", customer_id=None):
        self.sent_replies.append((conversation_id, text, status, customer_id))
        if self.fail_reply:
            return False
        return True

    async def get_conversation(self, conversation_id):
        self.fetched_conversations.append(conversation_id)
        payload = self.conversation_payloads.get(conversation_id)
        if payload is None:
            # A customer is waiting: their message is the newest thread.
            return {
                "id": conversation_id,
                "_embedded": {"threads": [{
                    "id": 1,
                    "type": "customer",
                    "createdAt": "2026-04-03T09:00:00Z",
                    "createdBy": {"type": "customer"},
                }]},
            }
        return payload

    async def delete_draft(self, conversation_id, thread_id):
        self.deleted_drafts.append((conversation_id, thread_id))
        if thread_id in self.delete_draft_failures:
            raise RuntimeError(f"delete failed for {thread_id}")
        return True

    async def update_draft(self, conversation_id, thread_id, text):
        self.updated_drafts.append((conversation_id, thread_id, text))
        if thread_id in self.update_draft_failures:
            raise RuntimeError(f"update failed for {thread_id}")
        return True

    async def add_draft(self, conversation_id, text, customer_id=None):
        self.drafts.append((conversation_id, text, customer_id))
        if self.fail_draft:
            return False
        return True

    async def add_note(self, conversation_id, text):
        if self.fail_on_note:
            raise RuntimeError("note write failed")
        self.notes.append((conversation_id, text))
        return True

    async def add_decision(self, conversation_id, text, record=None):
        # The owned mailbox keeps the record; the text is the same note as before.
        self.decision_records = getattr(self, "decision_records", []) + [record]
        return await self.add_note(conversation_id, text)

    async def add_tags(self, conversation_id, tags, existing_tags=None):
        if self.fail_on_tags:
            raise RuntimeError("tag write failed")
        self.tag_updates.append((conversation_id, list(tags), list(existing_tags or [])))
        return True


class FakeVectorStore:
    def __init__(self, _path):
        self.search_results = []

    def search(self, query, theme_filter=None, limit=3):
        return list(self.search_results)


def _support_result(
    *,
    delivery_mode: str,
    confidence_score: int,
    decision_note: str | None = None,
    investigator_provider: str = "codex",
    evaluator_provider: str = "claude",
    evidence_summary: str = "Merged evidence summary",
    investigation_steps: list[str] | None = None,
    risk_flags: list[str] | None = None,
    hard_blockers: list[str] | None = None,
) -> SupportPipelineResult:
    investigation = SupportInvestigationResult(
        reply_html="<p>Agentic reply</p>",
        confidence_score=confidence_score,
        decision_summary="Investigation summary",
        evidence_summary=evidence_summary,
        investigation_steps=investigation_steps or ["Checked docs", "Checked repo"],
        risk_flags=risk_flags or [],
        provider_name=investigator_provider,
        requires_manual_review=False,
        sources=["https://woodland.example/docs/example"],
    )
    evaluation = DraftConfidenceResult(
        score=confidence_score,
        verdict="accept" if delivery_mode == "auto_send" else "retry",
        summary="Evaluator summary",
        missing_information=[],
        investigation_actions=[],
        rewrite_brief="",
        risk_flags=risk_flags or [],
        hard_blockers=hard_blockers or [],
        provider=evaluator_provider,
    )
    return SupportPipelineResult(
        delivery_mode=delivery_mode,
        confidence_score=confidence_score,
        confidence_verdict=evaluation.verdict,
        reply_html=investigation.reply_html,
        decision_summary=investigation.decision_summary,
        evidence_summary=evidence_summary,
        investigation_steps=list(investigation.investigation_steps),
        risk_flags=list(risk_flags or []),
        hard_blockers=list(hard_blockers or []),
        decision_note=decision_note,
        manual_review=False,
        attempts=1,
        investigator_provider=investigator_provider,
        evaluator_provider=evaluator_provider,
        investigator_result=investigation,
        evaluator_result=evaluation,
        trace=[{"attempt": 1, "score": confidence_score, "delivery_mode": delivery_mode}],
    )


def _ticket_payload():
    return {
        "id": 987,
        "subject": "Original subject",
        "primaryCustomer": {
            "id": 33,
            "email": "reader@woodland.example",
            "first": "Reader",
            "last": "Oak",
        },
        "tags": [{"tag": "field-journal"}],
        "_embedded": {
            "threads": [
                {
                    "id": 1,
                    "createdAt": "2026-04-03T09:00:00Z",
                    "type": "customer",
                    "body": "<p>My index is missing its page numbers.</p>",
                    "source": {"type": "email"},
                    "createdBy": {"type": "customer"},
                },
                {
                    "type": "note",
                    "body": "<p>Please verify setup first.</p>",
                    "createdAt": "2026-04-03T10:00:00Z",
                    "source": {"type": "web"},
                },
            ]
        },
    }


@pytest.mark.asyncio
async def test_process_ticket_routes_support_through_support_pipeline_auto_sends_and_logs_audit_fields(
    monkeypatch, tmp_path
):
    fake_helpscout = FakeHelpScout()
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))
    fake_vector_store.search_results = [
        {
            "title": "KB article",
            "url": "https://woodland.example/docs/kb",
            "content": "Knowledge base evidence",
            "score": 0.91,
        }
    ]
    captured = {}

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Rewritten subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value={"plan": "pro"}))
    monkeypatch.setattr(
        "services.processor.find_similar_tickets",
        AsyncMock(return_value=[{"id": 42}]),
    )
    monkeypatch.setattr("services.processor.format_customer_context", lambda _data: "Customer context")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _tickets: "Past ticket context")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_kwargs: "Repo evidence context")
    monkeypatch.setattr("services.processor.search_web", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_web_results", lambda _results: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_kwargs: [])

    async def old_path_should_not_run(**_kwargs):
        raise AssertionError("legacy generate_draft_with_confidence path should not run")

    monkeypatch.setattr("services.processor.generate_draft_with_confidence", old_path_should_not_run)

    def fake_run_support_pipeline(*, context, evaluator_provider, target_score, **_kwargs):
        captured["context"] = context
        captured["kwargs"] = _kwargs
        captured["evaluator_provider"] = evaluator_provider
        captured["target_score"] = target_score
        return _support_result(
            delivery_mode="auto_send",
            confidence_score=9,
            decision_note="Supa decision note:\nMode: auto_send",
            investigator_provider="claude",
            evaluator_provider=evaluator_provider,
            evidence_summary="KB + repo + past tickets",
            investigation_steps=["Checked KB", "Checked repo"],
            risk_flags=["used_repo_evidence"],
        )

    monkeypatch.setattr("services.processor.run_support_pipeline", fake_run_support_pipeline, raising=False)
    monkeypatch.setenv("SUPA_SUPPORT_INVESTIGATOR", "codex")
    monkeypatch.setenv("SUPA_SUPPORT_EVALUATOR", "claude")
    monkeypatch.setenv("SUPA_SUPPORT_AUTOSEND_MIN_SCORE", "9")

    processor = TicketProcessor(auto_index=False)
    result = await processor.process_ticket(_ticket_payload(), trigger="customer_reply")

    assert result["classification"] == "support"
    assert result["delivery_mode"] == "auto_send"
    assert result["response_posted"] is True
    assert fake_helpscout.sent_replies == [(987, "<p>Agentic reply</p>", "pending", 33)]
    assert fake_helpscout.status_updates == [(987, "pending")]
    assert fake_helpscout.drafts == []
    assert fake_helpscout.notes == [(987, "Supa decision note:\nMode: auto_send")]
    # The same decision travels as data for the owned mailbox's ledger.
    (record,) = fake_helpscout.decision_records
    assert (record["kind"], record["actor"], record["data"]["mode"], record["data"]["text"]) == (
        "decision", "supa", "auto_send", "Supa decision note:\nMode: auto_send")
    assert fake_helpscout.tag_updates == [(987, ["supa-processed", "supa-auto-sent"], ["field-journal"])]
    assert "investigator_provider" not in captured["kwargs"]  # the agent owns its model chain
    assert captured["evaluator_provider"] == "claude"
    assert captured["target_score"] == 9
    assert captured["context"].subject == "Rewritten subject"
    assert captured["context"].knowledge_context.startswith("**KB article**")
    assert "Repo evidence context" in captured["context"].knowledge_context
    assert captured["context"].past_tickets_context == "Past ticket context"
    assert captured["context"].customer_context == "Customer context"
    assert captured["context"].internal_note == "Please verify setup first."

    row = get_ticket_log_by_helpscout_id(
        db_path=str(tmp_path / "supa.db"),
        helpscout_id="987",
    )
    assert row is not None
    assert row["delivery_mode"] == "auto_send"
    assert row["investigator_provider"] == "claude"
    assert row["evaluator_provider"] == "claude"
    assert row["decision_note"] == "Supa decision note:\nMode: auto_send"
    assert row["confidence_score"] == 9


@pytest.mark.asyncio
async def test_process_ticket_delayed_auto_send_note_includes_scheduled_send_time(
    monkeypatch, tmp_path
):
    fake_helpscout = FakeHelpScout()
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))
    fake_vector_store.search_results = [
        {
            "title": "KB article",
            "url": "https://woodland.example/docs/kb",
            "content": "Knowledge base evidence",
            "score": 0.91,
        }
    ]
    fake_helpscout.conversation_payloads[987] = {
        "id": 987,
        "primaryCustomer": {"id": 33},
        "tags": [{"tag": "field-journal"}],
        "_embedded": {
            "threads": [
                {
                    "id": 1,
                    "type": "customer",
                    "body": "<p>My index is missing its page numbers.</p>",
                    "createdAt": "2026-04-03T09:00:00Z",
                    "createdBy": {"type": "customer"},
                    "source": {"type": "email"},
                },
            ],
        },
    }

    async def add_draft_and_expose_thread(conversation_id, text, customer_id=None):
        fake_helpscout.drafts.append((conversation_id, text, customer_id))
        fake_helpscout.conversation_payloads[conversation_id]["_embedded"]["threads"].append(
            {
                "id": "500",
                "type": "message",
                "state": "draft",
                "body": text,
                "createdAt": "2026-04-03T10:00:00Z",
                "createdBy": {"type": "user"},
                "source": {"type": "api"},
            }
        )
        return True

    monkeypatch.setattr(fake_helpscout, "add_draft", add_draft_and_expose_thread)
    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Rewritten subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr("services.processor.find_similar_tickets", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_customer_context", lambda _data: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _tickets: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_kwargs: "")
    monkeypatch.setattr("services.processor.search_web", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_web_results", lambda _results: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_kwargs: [])
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_ENABLED", "true")
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_MIN_SECONDS", "900")
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_MAX_SECONDS", "900")

    def fake_run_support_pipeline(**_kwargs):
        return _support_result(
            delivery_mode="auto_send",
            confidence_score=9,
            decision_note="Supa decision note:\nMode: auto_send",
        )

    monkeypatch.setattr("services.processor.run_support_pipeline", fake_run_support_pipeline, raising=False)

    processor = TicketProcessor(auto_index=False)
    result = await processor.process_ticket(_ticket_payload(), trigger="customer_reply")

    assert result["delivery_mode"] == "auto_send"
    assert result["response_posted"] is True
    assert fake_helpscout.sent_replies == []
    assert fake_helpscout.drafts == [(987, "<p>Agentic reply</p>", 33)]
    assert len(fake_helpscout.notes) == 1
    note = fake_helpscout.notes[0][1]
    assert "Mode: auto_send" in note
    assert "Scheduled send:" in note
    assert "UTC" in note

    row = get_ticket_log_by_helpscout_id(
        db_path=str(tmp_path / "supa.db"),
        helpscout_id="987",
    )
    assert row is not None
    assert "Scheduled send:" in row["decision_note"]


@pytest.mark.asyncio
async def test_process_ticket_uses_draft_for_lower_confidence_and_keeps_web_fallback(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))
    fake_vector_store.search_results = [
        {
            "title": "Thin KB article",
            "url": "https://woodland.example/docs/thin",
            "content": "Thin evidence",
            "score": 0.42,
        }
    ]
    captured = {}

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Original subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "services.processor.find_similar_tickets",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr("services.processor.format_customer_context", lambda _data: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _tickets: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_kwargs: "Repo evidence context")
    monkeypatch.setattr(
        "services.processor.search_web",
        AsyncMock(return_value=[{"title": "Web hit"}]),
    )
    monkeypatch.setattr("services.processor.format_web_results", lambda _results: "Web fallback context")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_kwargs: [])

    async def old_path_should_not_run(**_kwargs):
        raise AssertionError("legacy generate_draft_with_confidence path should not run")

    monkeypatch.setattr("services.processor.generate_draft_with_confidence", old_path_should_not_run)

    def fake_run_support_pipeline(*, context, evaluator_provider, target_score, **_kwargs):
        captured["context"] = context
        return _support_result(
            delivery_mode="draft",
            confidence_score=7,
            decision_note="Supa decision note:\nMode: draft",
            investigator_provider="claude",
            evaluator_provider=evaluator_provider,
            evidence_summary="KB + repo + web",
            hard_blockers=["missing_critical_evidence"],
        )

    monkeypatch.setattr("services.processor.run_support_pipeline", fake_run_support_pipeline, raising=False)

    processor = TicketProcessor(auto_index=False)
    result = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert result["classification"] == "support"
    assert result["delivery_mode"] == "draft"
    assert fake_helpscout.sent_replies == []
    assert fake_helpscout.drafts == [(987, "<p>Agentic reply</p>", 33)]
    assert fake_helpscout.notes == [(987, "Supa decision note:\nMode: draft")]
    assert "Web fallback context" in captured["context"].knowledge_context
    assert "Repo evidence context" in captured["context"].knowledge_context

    row = get_ticket_log_by_helpscout_id(
        db_path=str(tmp_path / "supa.db"),
        helpscout_id="987",
    )
    assert row is not None
    assert row["delivery_mode"] == "draft"
    assert row["confidence_score"] == 7
    assert row["decision_note"] == "Supa decision note:\nMode: draft"


@pytest.mark.asyncio
async def test_process_ticket_logs_codex_as_final_evaluator_after_pipeline_fallback(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Original subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr("services.processor.find_similar_tickets", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_customer_context", lambda _data: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _tickets: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_kwargs: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_kwargs: [])

    captured = {}

    def fake_run_support_pipeline(*, evaluator_provider, **_kwargs):
        captured["requested_evaluator_provider"] = evaluator_provider
        return _support_result(
            delivery_mode="auto_send",
            confidence_score=9,
            decision_note="Supa decision note:\nMode: auto_send",
            investigator_provider="codex",
            evaluator_provider="codex",
        )

    monkeypatch.setattr("services.processor.run_support_pipeline", fake_run_support_pipeline, raising=False)
    monkeypatch.setenv("SUPA_SUPPORT_EVALUATOR", "claude")

    processor = TicketProcessor(auto_index=False)
    outcome = await processor.process_ticket(_ticket_payload(), trigger="customer_reply")

    assert outcome["delivery_mode"] == "auto_send"
    assert captured["requested_evaluator_provider"] == "claude"

    row = get_ticket_log_by_helpscout_id(
        db_path=str(tmp_path / "supa.db"),
        helpscout_id="987",
    )
    assert row is not None
    assert row["evaluator_provider"] == "codex"


@pytest.mark.asyncio
async def test_process_ticket_sanitizes_investigator_links_before_auto_send(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))
    fake_vector_store.search_results = [
        {
            "title": "KB article",
            "url": "https://woodland.example/docs/kb",
            "content": "Knowledge base evidence",
            "score": 0.91,
        }
    ]

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Original subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr("services.processor.find_similar_tickets", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_customer_context", lambda _data: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _tickets: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_kwargs: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_kwargs: [])

    result = _support_result(delivery_mode="auto_send", confidence_score=9)
    result.reply_html = '<p>Use <a href="https://evil.example.com/fix">this guide</a>.</p>'
    result.investigator_result.reply_html = result.reply_html

    monkeypatch.setattr("services.processor.run_support_pipeline", lambda **_kwargs: result, raising=False)

    processor = TicketProcessor(auto_index=False)
    await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert fake_helpscout.sent_replies == [(987, "<p>Use this guide.</p>", "pending", 33)]


@pytest.mark.asyncio
async def test_process_ticket_creates_manual_review_placeholder_draft_when_pipeline_reply_is_empty(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Original subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr("services.processor.find_similar_tickets", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_customer_context", lambda _data: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _tickets: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_kwargs: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_kwargs: [])

    result = _support_result(delivery_mode="draft", confidence_score=6, decision_note="Supa decision note:\nMode: draft")
    result.reply_html = ""
    result.investigator_result.reply_html = ""
    result.manual_review = True

    monkeypatch.setattr("services.processor.run_support_pipeline", lambda **_kwargs: result, raising=False)

    processor = TicketProcessor(auto_index=False)
    outcome = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert outcome["delivery_mode"] == "draft"
    assert fake_helpscout.drafts == [(987, "<p>Manual review required before sending a reply.</p>", 33)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "existing_score"),
    [("note_trigger", 7), ("manual_run", 0)],
)
async def test_rerun_preserves_existing_draft_when_new_run_degrades(
    monkeypatch, tmp_path, trigger, existing_score
):
    fake_helpscout = FakeHelpScout()
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Original subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr("services.processor.find_similar_tickets", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_customer_context", lambda _data: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _tickets: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_kwargs: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_kwargs: [])

    init_db(str(tmp_path / "supa.db"))
    log_ticket(
        db_path=str(tmp_path / "supa.db"),
        helpscout_id="987",
        subject="Original subject",
        customer_name="Reader",
        classification="support",
        status="processed",
        response_posted=True,
        trigger="note_trigger",
        confidence_score=existing_score,
        confidence_status="manual_review",
        confidence_attempts=1,
        confidence_summary="Useful draft kept for manual review.",
        draft_response_html="<p>Existing better draft</p>",
        delivery_mode="draft",
        investigator_provider="codex",
        evaluator_provider="claude",
        decision_note="Supa decision note:\nMode: draft\nConfidence: 7/10",
    )

    result = _support_result(
        delivery_mode="draft",
        confidence_score=0,
        decision_note="Supa decision note:\nMode: draft\nConfidence: 0/10",
        risk_flags=["investigator_failure", "empty_draft"],
        hard_blockers=["manual_review", "malformed_provider_output"],
    )
    result.reply_html = ""
    result.investigator_result.reply_html = ""
    result.investigator_result.sources = []
    result.investigator_result.risk_flags = ["investigator_failure", "empty_draft"]
    result.investigator_result.requires_manual_review = True
    result.evaluator_result.score = 0
    result.evaluator_result.verdict = "manual_review"
    result.evaluator_result.risk_flags = ["investigator_failure", "empty_draft"]
    result.evaluator_result.hard_blockers = ["manual_review", "malformed_provider_output"]
    result.manual_review = True

    monkeypatch.setattr("services.processor.run_support_pipeline", lambda **_kwargs: result, raising=False)

    processor = TicketProcessor(auto_index=False)
    outcome = await processor.process_ticket(_ticket_payload(), trigger=trigger)

    assert outcome["delivery_mode"] == "draft"
    assert outcome["confidence_score"] == existing_score
    assert outcome["response_posted"] is True
    assert fake_helpscout.drafts == []
    assert fake_helpscout.notes == []
    assert fake_helpscout.tag_updates == []

    row = get_ticket_log_by_helpscout_id(
        db_path=str(tmp_path / "supa.db"),
        helpscout_id="987",
    )
    assert row is not None
    assert row["confidence_score"] == existing_score
    assert row["draft_response_html"] == "<p>Existing better draft</p>"


@pytest.mark.asyncio
async def test_note_trigger_skips_draft_write_when_newer_auto_send_is_already_logged(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Original subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr("services.processor.find_similar_tickets", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_customer_context", lambda _data: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _tickets: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_kwargs: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_kwargs: [])

    init_db(str(tmp_path / "supa.db"))
    log_ticket(
        db_path=str(tmp_path / "supa.db"),
        helpscout_id="987",
        subject="Original subject",
        customer_name="Reader",
        classification="support",
        status="processed",
        response_posted=True,
        trigger="dashboard",
        confidence_score=10,
        confidence_status="accept",
        confidence_attempts=1,
        confidence_summary="Operator already sent the final reply.",
        draft_response_html="<p>Final sent reply</p>",
        delivery_mode="auto_send",
        investigator_provider="manual",
        evaluator_provider="manual",
        decision_note="Supa decision note:\nMode: auto_send\nConfidence: 10/10",
    )

    monkeypatch.setattr(
        "services.processor.run_support_pipeline",
        lambda **_kwargs: _support_result(
            delivery_mode="draft",
            confidence_score=7,
            decision_note="Supa decision note:\nMode: draft",
        ),
        raising=False,
    )

    payload = _ticket_payload()
    payload["_embedded"]["threads"].insert(
        0,
        {
            "type": "note",
            "body": "<p>Please answer now</p>",
            "createdAt": "2025-04-09T10:00:00Z",
            "createdBy": {"type": "user"},
            "source": {"type": "web"},
        },
    )

    processor = TicketProcessor(auto_index=False)
    outcome = await processor.process_ticket(payload, trigger="note_trigger")

    assert outcome["delivery_mode"] == "auto_send"
    assert outcome["response_posted"] is True
    assert fake_helpscout.drafts == []
    assert fake_helpscout.notes == []
    assert fake_helpscout.tag_updates == []

    row = get_ticket_log_by_helpscout_id(
        db_path=str(tmp_path / "supa.db"),
        helpscout_id="987",
    )
    assert row is not None
    assert row["delivery_mode"] == "auto_send"
    assert row["draft_response_html"] == "<p>Final sent reply</p>"


@pytest.mark.asyncio
async def test_process_ticket_marks_auto_send_failure_as_error_and_skips_follow_up_writes(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    fake_helpscout.fail_reply = True
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Original subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr("services.processor.find_similar_tickets", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_customer_context", lambda _data: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _tickets: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_kwargs: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_kwargs: [])
    monkeypatch.setattr(
        "services.processor.run_support_pipeline",
        lambda **_kwargs: _support_result(
            delivery_mode="auto_send",
            confidence_score=9,
            decision_note="Supa decision note:\nMode: auto_send",
        ),
        raising=False,
    )

    processor = TicketProcessor(auto_index=False)
    outcome = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert outcome["status"] == "error"
    assert outcome["response_posted"] is False
    assert outcome["error"] == "HelpScout auto-send failed"
    assert fake_helpscout.sent_replies == [(987, "<p>Agentic reply</p>", "pending", 33)]
    assert fake_helpscout.notes == []
    assert fake_helpscout.tag_updates == []

    row = get_ticket_log_by_helpscout_id(
        db_path=str(tmp_path / "supa.db"),
        helpscout_id="987",
    )
    assert row is not None
    assert row["status"] == "error"
    assert row["response_posted"] == 0
    assert row["error_message"] == "HelpScout auto-send failed"


@pytest.mark.asyncio
async def test_process_ticket_marks_pending_status_failure_as_error_and_skips_follow_up_writes(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    fake_helpscout.fail_status = True
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Original subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr("services.processor.find_similar_tickets", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_customer_context", lambda _data: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _tickets: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_kwargs: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_kwargs: [])
    monkeypatch.setattr(
        "services.processor.run_support_pipeline",
        lambda **_kwargs: _support_result(
            delivery_mode="auto_send",
            confidence_score=9,
            decision_note="Supa decision note:\nMode: auto_send",
        ),
        raising=False,
    )

    processor = TicketProcessor(auto_index=False)
    outcome = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert outcome["status"] == "error"
    assert outcome["response_posted"] is False
    assert outcome["error"] == "HelpScout pending status update failed"
    assert fake_helpscout.sent_replies == [(987, "<p>Agentic reply</p>", "pending", 33)]
    assert fake_helpscout.status_updates == [(987, "pending")]
    assert fake_helpscout.notes == []
    assert fake_helpscout.tag_updates == []

    row = get_ticket_log_by_helpscout_id(
        db_path=str(tmp_path / "supa.db"),
        helpscout_id="987",
    )
    assert row is not None
    assert row["status"] == "error"
    assert row["response_posted"] == 0
    assert row["error_message"] == "HelpScout pending status update failed"


@pytest.mark.asyncio
async def test_process_ticket_keeps_response_posted_true_when_note_write_fails_after_send(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    fake_helpscout.fail_on_note = True
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Original subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr("services.processor.find_similar_tickets", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_customer_context", lambda _data: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _tickets: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_kwargs: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_kwargs: [])
    monkeypatch.setattr(
        "services.processor.run_support_pipeline",
        lambda **_kwargs: _support_result(
            delivery_mode="auto_send",
            confidence_score=9,
            decision_note="Supa decision note:\nMode: auto_send",
        ),
        raising=False,
    )

    processor = TicketProcessor(auto_index=False)
    outcome = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert outcome["status"] == "warning"
    assert outcome["response_posted"] is True
    assert fake_helpscout.sent_replies == [(987, "<p>Agentic reply</p>", "pending", 33)]

    row = get_ticket_log_by_helpscout_id(
        db_path=str(tmp_path / "supa.db"),
        helpscout_id="987",
    )
    assert row is not None
    assert row["status"] == "warning"
    assert row["response_posted"] == 1


@pytest.mark.asyncio
async def test_process_ticket_offloads_support_pipeline_from_event_loop(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))
    fake_vector_store.search_results = []
    captured = {}

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Original subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr("services.processor.find_similar_tickets", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_customer_context", lambda _data: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _tickets: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_kwargs: "")
    monkeypatch.setattr("services.processor.search_web", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_web_results", lambda _results: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_kwargs: [])

    def fake_run_support_pipeline(**kwargs):
        captured["pipeline_kwargs"] = kwargs
        return _support_result(delivery_mode="draft", confidence_score=7)

    async def fake_to_thread(func, /, *args, **kwargs):
        captured["to_thread_func"] = func
        captured["to_thread_args"] = args
        captured["to_thread_kwargs"] = kwargs
        return func(*args, **kwargs)

    monkeypatch.setattr("services.processor.run_support_pipeline", fake_run_support_pipeline, raising=False)
    monkeypatch.setattr("services.processor.asyncio.to_thread", fake_to_thread)

    processor = TicketProcessor(auto_index=False)
    result = await processor.process_ticket(_ticket_payload(), trigger="note_trigger")

    assert result["delivery_mode"] == "draft"
    assert captured["to_thread_func"] is fake_run_support_pipeline
    assert captured["to_thread_args"] == ()
    assert captured["to_thread_kwargs"]["context"].subject == "Original subject"


@pytest.mark.asyncio
async def test_process_ticket_preserves_non_support_autoclose_path(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    leases = RecordingLeaseFactory()

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.conversation_lease", leases)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: FakeVectorStore(_path))
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="spam"))
    monkeypatch.setattr(
        "services.processor.generate_subject",
        lambda **_kwargs: pytest.fail("generate_subject should not run for non-support"),
    )
    monkeypatch.setattr(
        "services.processor.generate_draft_with_confidence",
        lambda **_kwargs: pytest.fail("legacy support draft path should not run"),
    )
    monkeypatch.setattr(
        "services.processor.run_support_pipeline",
        lambda **_kwargs: pytest.fail("support pipeline should not run for non-support"),
        raising=False,
    )

    processor = TicketProcessor(auto_index=False)
    result = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert result["classification"] == "spam"
    assert result["delivery_mode"] is None
    assert fake_helpscout.closed == [987]
    assert fake_helpscout.sent_replies == []
    assert fake_helpscout.drafts == []
    assert fake_helpscout.notes == []
    assert fake_helpscout.tag_updates == [(987, ["supa-processed", "spam"], ["field-journal"])]

    row = get_ticket_log_by_helpscout_id(
        db_path=str(tmp_path / "supa.db"),
        helpscout_id="987",
    )
    assert row is not None
    assert row["classification"] == "spam"
    assert row["response_posted"] == 0
    leases.assert_mutation_guarded(987, "close_conversation")
    leases.assert_mutation_guarded(987, "add_tags")


@pytest.mark.asyncio
async def test_purge_prior_drafts_updates_each_id_to_placeholder(monkeypatch, tmp_path):
    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    init_db(str(tmp_path / "supa.db"))
    fake_helpscout = FakeHelpScout()
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", FakeVectorStore)
    processor = TicketProcessor(auto_index=False)

    await processor._purge_prior_drafts(987, [1001, 1002, 1003])

    assert fake_helpscout.updated_drafts == [
        (987, 1001, "<p>Manual review required before sending a reply.</p>"),
        (987, 1002, "<p>Manual review required before sending a reply.</p>"),
        (987, 1003, "<p>Manual review required before sending a reply.</p>"),
    ]


@pytest.mark.asyncio
async def test_purge_prior_drafts_noop_on_empty(monkeypatch, tmp_path):
    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    init_db(str(tmp_path / "supa.db"))
    fake_helpscout = FakeHelpScout()
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", FakeVectorStore)
    processor = TicketProcessor(auto_index=False)

    await processor._purge_prior_drafts(987, [])

    assert fake_helpscout.updated_drafts == []


@pytest.mark.asyncio
async def test_purge_prior_drafts_tolerates_delete_failures(monkeypatch, tmp_path):
    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    init_db(str(tmp_path / "supa.db"))
    fake_helpscout = FakeHelpScout()
    fake_helpscout.update_draft_failures = {1002}
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", FakeVectorStore)
    processor = TicketProcessor(auto_index=False)

    await processor._purge_prior_drafts(987, [1001, 1002, 1003])

    assert fake_helpscout.updated_drafts == [
        (987, 1001, "<p>Manual review required before sending a reply.</p>"),
        (987, 1002, "<p>Manual review required before sending a reply.</p>"),
        (987, 1003, "<p>Manual review required before sending a reply.</p>"),
    ]


def _seed_draft_threads_payload(conversation_id: int, draft_specs: list[tuple[int, str]]) -> dict:
    # Draft stacks belong to the same customer message passed into process_ticket;
    # the delivery gate re-reads that exact message before any write.
    threads = [_ticket_payload()["_embedded"]["threads"][0]]
    for thread_id, created_at in draft_specs:
        threads.append(
            {
                "id": thread_id,
                "type": "message",
                "state": "draft",
                "createdAt": created_at,
                "body": f"<p>stacked {thread_id}</p>",
                "source": {"type": "api"},
            }
        )
    return {"id": conversation_id, "_embedded": {"threads": threads}}


def _patch_draft_path_dependencies(monkeypatch, *, delivery_mode: str, confidence_score: int, risk_flags=None):
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Original subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr("services.processor.find_similar_tickets", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_customer_context", lambda _data: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda _tickets: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **_kwargs: "")
    monkeypatch.setattr("services.processor.search_web", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_web_results", lambda _results: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **_kwargs: [])

    async def old_path_should_not_run(**_kwargs):
        raise AssertionError("legacy generate_draft_with_confidence path should not run")

    monkeypatch.setattr("services.processor.generate_draft_with_confidence", old_path_should_not_run)

    def fake_pipeline(**_kwargs):
        return _support_result(
            delivery_mode=delivery_mode,
            confidence_score=confidence_score,
            decision_note="Supa decision note:\nMode: draft",
            risk_flags=risk_flags or [],
        )

    monkeypatch.setattr("services.processor.run_support_pipeline", fake_pipeline, raising=False)


@pytest.mark.asyncio
async def test_per_run_draft_ceiling_suppresses_live_autosend(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))
    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    _patch_draft_path_dependencies(monkeypatch, delivery_mode="auto_send", confidence_score=9)

    processor = TicketProcessor(auto_index=False)
    result = await processor.process_ticket(
        _ticket_payload(), trigger="manual_run", delivery_ceiling="draft"
    )

    assert result["delivery_mode"] == "draft"
    assert fake_helpscout.sent_replies == []
    assert len(fake_helpscout.drafts) == 1
    assert "per-run draft ceiling" in result["decision_note"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("blocker", "reply_html"),
    [
        (
            "unverified_account_action_claim",
            "<p>I’ve canceled the automatic renewal for your subscription.</p>",
        ),
    ],
)
async def test_process_ticket_replaces_unsafe_blocked_reply_with_placeholder(
    monkeypatch, tmp_path, blocker, reply_html
):
    fake_helpscout = FakeHelpScout()
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    _patch_draft_path_dependencies(monkeypatch, delivery_mode="draft", confidence_score=6)

    def fake_pipeline(**_kwargs):
        result = _support_result(
            delivery_mode="draft",
            confidence_score=6,
            hard_blockers=[blocker],
        )
        result.reply_html = reply_html
        result.investigator_result.reply_html = result.reply_html
        return result

    monkeypatch.setattr("services.processor.run_support_pipeline", fake_pipeline, raising=False)

    processor = TicketProcessor(auto_index=False)
    await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert fake_helpscout.drafts == [
        (987, "<p>Manual review required before sending a reply.</p>", 33)
    ]


@pytest.mark.asyncio
async def test_process_ticket_add_draft_path_deletes_prior_drafts(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    fake_helpscout.conversation_payloads[987] = _seed_draft_threads_payload(
        987,
        [
            (5001, "2026-04-18T10:00:00Z"),
            (5002, "2026-04-18T11:00:00Z"),
            (5003, "2026-04-18T12:00:00Z"),
        ],
    )
    # A draft Support typed in Help Scout and an already blanked leftover are left alone.
    fake_helpscout.conversation_payloads[987]["_embedded"]["threads"] += [
        {"id": 5004, "type": "message", "state": "draft", "createdAt": "2026-04-18T12:30:00Z",
         "body": "<p>Support typing</p>", "source": {"type": "web"}},
        {"id": 5005, "type": "message", "state": "draft", "createdAt": "2026-04-18T09:00:00Z",
         "body": "<p>Manual review required before sending a reply.</p>", "source": {"type": "api"}},
    ]
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))
    leases = RecordingLeaseFactory()

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.conversation_lease", leases)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    _patch_draft_path_dependencies(monkeypatch, delivery_mode="draft", confidence_score=7)

    processor = TicketProcessor(auto_index=False)
    await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert len(fake_helpscout.drafts) == 1
    assert fake_helpscout.updated_drafts == [
        (987, 5001, "<p>Manual review required before sending a reply.</p>"),
        (987, 5002, "<p>Manual review required before sending a reply.</p>"),
        (987, 5003, "<p>Manual review required before sending a reply.</p>"),
    ]
    assert [event[0] for event in leases.events].count("enter") == 1
    for mutation in ("add_draft", "update_draft", "add_decision", "add_tags"):
        leases.assert_mutation_guarded(987, mutation)


@pytest.mark.asyncio
async def test_process_ticket_add_draft_path_skips_purge_when_add_fails(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    fake_helpscout.fail_draft = True
    fake_helpscout.conversation_payloads[987] = _seed_draft_threads_payload(
        987,
        [(5001, "2026-04-18T10:00:00Z")],
    )
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    _patch_draft_path_dependencies(monkeypatch, delivery_mode="draft", confidence_score=7)

    processor = TicketProcessor(auto_index=False)
    result = await processor.process_ticket(_ticket_payload(), trigger="new_conversation")

    assert fake_helpscout.updated_drafts == []
    assert result["error"]


@pytest.mark.asyncio
async def test_process_ticket_preserve_path_sweeps_stragglers_keeps_newest(monkeypatch, tmp_path):
    fake_helpscout = FakeHelpScout()
    fake_helpscout.conversation_payloads[987] = _seed_draft_threads_payload(
        987,
        [
            (5001, "2026-04-18T10:00:00Z"),
            (5002, "2026-04-18T11:00:00Z"),
            (5003, "2026-04-18T12:00:00Z"),
        ],
    )
    fake_vector_store = FakeVectorStore(str(tmp_path / "vectors.db"))

    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    init_db(str(tmp_path / "supa.db"))
    log_ticket(
        db_path=str(tmp_path / "supa.db"),
        helpscout_id="987",
        subject="Original subject",
        customer_name="Reader",
        classification="support",
        status="processed",
        response_posted=True,
        trigger="note_trigger",
        confidence_score=9,
        confidence_status="accepted",
        confidence_attempts=1,
        confidence_summary="earlier good draft",
        confidence_trace="[]",
        confidence_provider="claude",
        draft_response_html="<p>keeper</p>",
        evidence_summary="",
        delivery_mode="draft",
    )
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **_kwargs: fake_helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda _path: fake_vector_store)
    _patch_draft_path_dependencies(
        monkeypatch,
        delivery_mode="draft",
        confidence_score=3,
        risk_flags=["investigator_failure"],
    )

    processor = TicketProcessor(auto_index=False)
    await processor.process_ticket(_ticket_payload(), trigger="note_trigger")

    assert fake_helpscout.drafts == []
    assert fake_helpscout.updated_drafts == [
        (987, 5001, "<p>Manual review required before sending a reply.</p>"),
        (987, 5002, "<p>Manual review required before sending a reply.</p>"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', [False, True])
async def test_guarded_recovery_checks_fresh_complete_timeline_before_any_write(monkeypatch, tmp_path, changed):
    from tests.test_support_recovery import fixture
    from services.support_recovery import recovery_metadata
    raw, row = fixture()
    raw['id'] = 987
    expected = recovery_metadata(raw, row)
    fake = FakeHelpScout()
    import copy
    fresh = copy.deepcopy(raw)
    if changed:
        fresh['_embedded']['threads'][-1]['body'] = '<p>Human edit while investigating</p>'
    readback = copy.deepcopy(fresh)
    readback['_embedded']['threads'][-1].update(id=3, body=_support_result(delivery_mode='auto_send', confidence_score=9, decision_note='').reply_html)
    fake.get_complete_conversation = AsyncMock(side_effect=[fresh, fresh, readback])
    monkeypatch.setattr('services.processor.DB_DIR', tmp_path)
    init_db(str(tmp_path/'supa.db'))
    log_ticket(str(tmp_path/'supa.db'), helpscout_id='987', **row)
    monkeypatch.setattr('services.processor.HelpScoutClient', lambda **kw: fake)
    monkeypatch.setattr('services.processor.VectorStore', lambda path: FakeVectorStore(str(tmp_path/'vectors.db')))
    _patch_draft_path_dependencies(monkeypatch, delivery_mode='auto_send', confidence_score=9)
    # Even if the classifier would close this ticket, recovery is support-only.
    classify = AsyncMock(return_value='spam')
    monkeypatch.setattr('services.processor.classify_ticket', classify)
    processor = TicketProcessor(auto_index=False)
    result = await processor.process_ticket(_ticket_payload(), trigger='manual_run', delivery_ceiling='draft', recovery_guard=expected)
    classify.assert_not_awaited()
    assert fake.sent_replies == []
    assert fake.closed == []
    if changed:
        assert result['status'] == 'recovery_held'
        assert fake.drafts == fake.notes == fake.subject_updates == fake.tag_updates == fake.updated_drafts == []
    else:
        assert result['response_posted'] is True
        assert len(fake.drafts) == 1


@pytest.mark.asyncio
async def test_recovery_cleanup_preserves_draft_edited_after_new_draft_save(monkeypatch, tmp_path):
    from tests.test_support_recovery import fixture
    from services.conversation_gateway import draft_version
    raw, row = fixture()
    original = draft_version(raw['_embedded']['threads'][-1])
    raw['_embedded']['threads'][-1]['body'] = 'Human edit'
    fake = FakeHelpScout()
    fake.get_complete_conversation = AsyncMock(return_value=raw)
    monkeypatch.setattr('services.processor.DB_DIR', tmp_path)
    monkeypatch.setattr('services.processor.HelpScoutClient',lambda **kw:fake)
    monkeypatch.setattr('services.processor.VectorStore',lambda path:FakeVectorStore(str(tmp_path/'vectors.db')))
    processor = TicketProcessor(auto_index=False)
    await processor._purge_prior_drafts(123,[2],expected_versions={2:original})
    assert fake.updated_drafts == []
