import json

import pytest

from agents.draft_confidence import DraftConfidenceResult
from agents.support_agent import SupportInvestigationResult
from services.support_pipeline import SupportPipelineContext, run_support_pipeline


def _investigation(
    *,
    reply_html: str = "<p>Reply</p>",
    confidence_score: int = 8,
    decision_summary: str = "Investigation summary",
    evidence_summary: str = "Docs and repo evidence",
    investigation_steps: list[str] | None = None,
    risk_flags: list[str] | None = None,
    provider_name: str = "codex",
    requires_manual_review: bool = False,
    sources: list[str] | None = None,
) -> SupportInvestigationResult:
    return SupportInvestigationResult(
        reply_html=reply_html,
        confidence_score=confidence_score,
        decision_summary=decision_summary,
        evidence_summary=evidence_summary,
        investigation_steps=investigation_steps or ["Checked docs"],
        risk_flags=risk_flags or [],
        provider_name=provider_name,
        requires_manual_review=requires_manual_review,
        sources=sources or ["https://woodland.example/docs/example"],
    )


def _evaluation(
    *,
    score: int,
    verdict: str = "accept",
    summary: str = "Evaluator summary",
    missing_information: list[str] | None = None,
    investigation_actions: list[str] | None = None,
    rewrite_brief: str = "",
    risk_flags: list[str] | None = None,
    hard_blockers: list[str] | None = None,
    provider: str = "claude",
) -> DraftConfidenceResult:
    return DraftConfidenceResult(
        score=score,
        verdict=verdict,
        summary=summary,
        missing_information=missing_information or [],
        investigation_actions=investigation_actions or [],
        rewrite_brief=rewrite_brief,
        risk_flags=risk_flags or [],
        hard_blockers=hard_blockers or [],
        provider=provider,
    )


def _context(**kwargs) -> SupportPipelineContext:
    base = dict(
        customer_name="Reader",
        subject="Journal display",
        message="My index is missing its page numbers.",
        theme_tag="field-journal",
        knowledge_context="KB context",
        past_tickets_context="Past ticket context",
    )
    base.update(kwargs)
    return SupportPipelineContext(**base)


def test_support_pipeline_auto_sends_when_score_is_nine_and_no_blockers(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=9),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=9, verdict="accept", provider="claude"),
    )

    result = run_support_pipeline(context=_context())

    assert result.delivery_mode == "auto_send"
    assert result.confidence_score == 9
    assert result.hard_blockers == []
    assert result.decision_note is not None
    assert "Mode: auto_send" in result.decision_note


def test_support_pipeline_saves_draft_when_score_is_below_eight(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=7),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=7, verdict="accept"),
    )

    result = run_support_pipeline(context=_context(), max_attempts=1)

    assert result.delivery_mode == "draft"
    assert result.confidence_score == 7
    assert result.decision_note is not None
    assert "Mode: draft" in result.decision_note
    assert "auto-send needs 8/10" in result.decision_note


def test_support_pipeline_auto_sends_a_clean_eight(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=8),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=8, verdict="accept"),
    )

    result = run_support_pipeline(context=_context(), max_attempts=1)

    assert result.delivery_mode == "auto_send"
    assert result.confidence_score == 8
    assert result.hard_blockers == []


@pytest.mark.parametrize(
    "message,reply_html",
    [
        ("I bought the wrong theme, can I get a refund?", "<p>Here is how the theme works.</p>"),
        ("Bitte um Rückerstattung, das Theme passt nicht.", "<p>Danke!</p>"),
        ("Quiero un reembolso.", "<p>Gracias.</p>"),
        ("Do you offer a discount for a second site?", "<p>Plus covers one site.</p>"),
        ("The header is too tall.", "<p>I'm happy to process a refund for Trail Atlas.</p>"),
        ("The header is too tall.", "<p>I can send you a free license for Trail Atlas.</p>"),
        ("The header is too tall.", "<p>We can waive the renewal this year.</p>"),
        ("I was charged twice for Plus.", "<p>Thanks.</p>"),
        ("Ich möchte eine Erstattung.", "<p>Danke.</p>"),
        ("Quiero la devolución del dinero.", "<p>Gracias.</p>"),
        ("Vreau banii înapoi.", "<p>Mulțumesc.</p>"),
        ("Ik wil mijn geld terug.", "<p>Bedankt.</p>"),
        ("Proszę o zwrot środków.", "<p>Dziękuję.</p>"),
        ("Est-ce possible d'avoir une remise ?", "<p>Merci.</p>"),
        ("The header is too tall.", "<p>I've credited your account with an extra month.</p>"),
        ("The header is too tall.", "<p>Here is 30% off your renewal.</p>"),
        ("The header is too tall.", "<p>Ich kann Ihnen das Geld zurück überweisen, eine Rückerstattung ist möglich.</p>"),
    ],
)
def test_money_requests_and_offers_never_auto_send(monkeypatch, message, reply_html):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=10, reply_html=reply_html),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=10, verdict="accept"),
    )

    result = run_support_pipeline(context=_context(message=message), max_attempts=1)

    assert result.delivery_mode == "draft"
    assert "money_decision" in result.hard_blockers


def test_a_refund_asked_earlier_in_the_thread_still_holds_the_reply(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=9, reply_html="<p>I'm checking it now.</p>"),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=9, verdict="accept"),
    )
    history = json.dumps([
        {"role": "customer", "message": "The theme doesn't fit, I'd like a refund."},
        {"role": "support", "message": "Let me look."},
    ])

    result = run_support_pipeline(
        context=_context(message="Any update?", conversation_history=history), max_attempts=1
    )

    assert "money_decision" in result.hard_blockers
    assert result.delivery_mode == "draft"


def test_an_eight_the_judge_did_not_accept_stays_a_draft(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=8),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=8, verdict="retry", summary="Needs a sharper answer."),
    )

    result = run_support_pipeline(context=_context(), max_attempts=1)

    assert result.hard_blockers == []
    assert result.delivery_mode == "draft"


def test_support_pipeline_manual_review_verdict_always_forces_draft(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=10),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=10, verdict="manual_review"),
    )

    result = run_support_pipeline(context=_context(), max_attempts=1)

    assert result.delivery_mode == "draft"
    assert "manual_review" in result.hard_blockers


def test_support_pipeline_retries_soft_manual_review_for_tone_only(monkeypatch):
    calls = []
    eval_count = {"value": 0}

    def fake_investigator(request, *_, **__):
        calls.append(request)
        return _investigation(reply_html=f"<p>Draft {len(calls)}</p>")

    def fake_evaluator(*_, **__):
        eval_count["value"] += 1
        if eval_count["value"] == 1:
            return _evaluation(
                score=7,
                verdict="manual_review",
                summary="Tone is too tutorial-like for this operator note.",
                rewrite_brief="Make the reply shorter and more direct.",
                risk_flags=["tone_mismatch_with_internal_operator_note", "reply_length_disproportionate_to_question_complexity"],
            )
        return _evaluation(score=8, verdict="accept", summary="Now concise enough")

    monkeypatch.setattr("services.support_pipeline.draft_reply", fake_investigator)
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", fake_evaluator)

    result = run_support_pipeline(
        context=_context(internal_note="Customer is aggressive; avoid hand-holding."),
        max_attempts=2,
        target_score=8,
    )

    assert len(calls) == 2
    assert result.attempts == 2
    assert result.confidence_score == 8
    assert result.delivery_mode == "auto_send"
    assert "manual_review" not in result.hard_blockers
    assert "Evaluator feedback" in calls[1].internal_note
    assert "Tone is too tutorial-like" in calls[1].internal_note


def test_support_pipeline_does_not_retry_fatal_manual_review(monkeypatch):
    calls = []

    def fake_investigator(request, *_, **__):
        calls.append(request)
        return _investigation(reply_html=f"<p>Draft {len(calls)}</p>")

    monkeypatch.setattr("services.support_pipeline.draft_reply", fake_investigator)
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(
            score=7,
            verdict="manual_review",
            summary="The draft makes an unsupported product claim.",
            hard_blockers=["missing_critical_evidence"],
            risk_flags=["unsupported_claim"],
        ),
    )

    result = run_support_pipeline(context=_context(), max_attempts=2)

    assert len(calls) == 1
    assert result.attempts == 1
    assert result.delivery_mode == "draft"
    assert "missing_critical_evidence" in result.hard_blockers


def test_support_pipeline_keeps_best_retry_result_when_later_attempt_degrades(monkeypatch):
    calls = []
    eval_count = {"value": 0}

    def fake_investigator(request, *_, **__):
        calls.append(request)
        if len(calls) == 1:
            return _investigation(reply_html="<p>First draft</p>")
        return _investigation(
            reply_html="",
            confidence_score=0,
            decision_summary="Support investigator failed: timed out",
            risk_flags=["investigator_failure"],
            requires_manual_review=True,
            sources=[],
        )

    def fake_evaluator(*_, **kwargs):
        eval_count["value"] += 1
        if eval_count["value"] == 1:
            return _evaluation(
                score=7,
                verdict="manual_review",
                summary="Tone is too tutorial-like for this operator note.",
                rewrite_brief="Make the reply shorter and more direct.",
                risk_flags=[
                    "tone_mismatch_with_internal_operator_note",
                    "reply_length_disproportionate_to_question_complexity",
                ],
            )
        return _evaluation(
            score=0,
            verdict="manual_review",
            summary="No draft was generated.",
            hard_blockers=["malformed_provider_output"],
            risk_flags=["investigator_failure"],
        )

    monkeypatch.setattr("services.support_pipeline.draft_reply", fake_investigator)
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", fake_evaluator)

    result = run_support_pipeline(
        context=_context(internal_note="Customer is aggressive; avoid hand-holding."),
        max_attempts=2,
    )

    # 2 drafting calls (the model fallback happens inside draft_reply): attempt 1, then
    # attempt 2 which fails. Still 2 scored attempts; the best (attempt 1's draft) is kept.
    assert len(calls) == 2
    assert result.attempts == 2
    assert result.confidence_score == 7
    assert result.reply_html == "<p>First draft</p>"
    assert result.delivery_mode == "draft"
    assert "malformed_provider_output" not in result.hard_blockers


def test_support_pipeline_hard_blockers_override_high_score(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(
            confidence_score=10,
            risk_flags=["contradictory_evidence"],
        ),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=10, verdict="accept"),
    )

    result = run_support_pipeline(context=_context(), max_attempts=1)

    assert result.delivery_mode == "draft"
    assert "contradiction" in " ".join(result.hard_blockers).lower()


def test_support_pipeline_builds_compact_decision_note_for_meaningful_actions(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=10),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=10, verdict="accept", summary="Ready to send"),
    )

    result = run_support_pipeline(context=_context())

    assert result.decision_note is not None
    assert result.decision_note.startswith("Supa decision note:")
    assert "Confidence: 10/10" in result.decision_note
    assert "Provider: investigator=codex, evaluator=claude" in result.decision_note
    assert len(result.decision_note.splitlines()) <= 7


def test_support_pipeline_decision_note_explains_why_draft_was_not_auto_sent(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(
            confidence_score=6,
            evidence_summary=(
                "Customer confirmed category blocks pull random images from listings; "
                "the proposed Field Journal filter hook is inferred and not verified."
            ),
            risk_flags=["inferred_filter_hook"],
            requires_manual_review=True,
            sources=[
                "Customer message (latest reply)",
                "Past ticket: Intermittent empty background-image URL",
            ],
        ),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(
            score=6,
            verdict="retry",
            summary=(
                "The diagnosis is plausible, but the primary code workaround leads with "
                "a fabricated filter hook that may not exist in Field Journal."
            ),
            missing_information=["Verified Field Journal hook name or source code path"],
            investigation_actions=["check_theme_compatibility"],
            rewrite_brief="Lead with verified diagnostics before any code workaround.",
        ),
    )

    result = run_support_pipeline(context=_context(), max_attempts=1)

    assert result.delivery_mode == "draft"
    assert result.decision_note is not None
    assert (
        "Why: The diagnosis is plausible, but the primary code workaround leads with "
        "a fabricated filter hook"
    ) in result.decision_note
    assert "Blocked by: manual_review" in result.decision_note
    assert "Missing: Verified Field Journal hook name or source code path" in result.decision_note
    assert "Next: check_theme_compatibility; Lead with verified diagnostics before any code workaround." in result.decision_note
    assert "Evidence: Customer confirmed category blocks pull random images from listings" in result.decision_note


def test_support_pipeline_keeps_investigator_and_evaluator_providers_separate(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(provider_name="codex"),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=9, verdict="accept", provider="openclaw"),
    )

    result = run_support_pipeline(
        context=_context(),
        evaluator_provider="openclaw",
    )

    assert result.investigator_provider == "codex"
    assert result.evaluator_provider == "openclaw"
    assert result.investigator_provider != result.evaluator_provider


def test_support_pipeline_falls_back_to_codex_when_claude_evaluator_fails(monkeypatch):
    evaluator_calls = []

    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=9, provider_name="codex"),
    )

    def fake_evaluator(*_, **kwargs):
        evaluator_calls.append(kwargs["provider"])
        if kwargs["provider"] == "claude":
            return _evaluation(
                score=0,
                verdict="manual_review",
                summary="Confidence evaluator failed: Claude CLI failed",
                risk_flags=["evaluator_failure"],
                hard_blockers=["malformed_provider_output"],
                provider="claude",
            )
        return _evaluation(score=9, verdict="accept", summary="Codex evaluator recovered", provider="codex")

    monkeypatch.setattr("services.support_pipeline.evaluate_draft", fake_evaluator)

    result = run_support_pipeline(context=_context(), evaluator_provider="claude", max_attempts=1)

    assert evaluator_calls == ["claude", "codex"]
    assert result.delivery_mode == "auto_send"
    assert result.confidence_score == 9
    assert result.evaluator_provider == "codex"
    assert result.hard_blockers == []
    assert result.trace[0]["evaluator_provider"] == "codex"


def test_support_pipeline_keeps_manual_review_when_codex_fallback_also_fails(monkeypatch):
    evaluator_calls = []

    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=9, provider_name="codex"),
    )

    def fake_evaluator(*_, **kwargs):
        evaluator_calls.append(kwargs["provider"])
        return _evaluation(
            score=0,
            verdict="manual_review",
            summary=f"Confidence evaluator failed: {kwargs['provider']} failed",
            risk_flags=["evaluator_failure"],
            hard_blockers=["malformed_provider_output"],
            provider=kwargs["provider"],
        )

    monkeypatch.setattr("services.support_pipeline.evaluate_draft", fake_evaluator)

    result = run_support_pipeline(context=_context(), evaluator_provider="claude", max_attempts=1)

    assert evaluator_calls == ["claude", "codex"]
    assert result.delivery_mode == "draft"
    assert result.confidence_score == 0
    assert result.evaluator_provider == "codex"
    assert "manual_review" in result.hard_blockers
    assert "malformed_provider_output" in result.hard_blockers


def test_support_pipeline_evaluates_sanitized_reply_html(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(
            reply_html='<p>Choose <strong>Library\x1aJournal\x1aIndex</strong> or see <a href="https://untrusted.example/index">the index</a>.</p>',
            confidence_score=9,
        ),
    )

    def fake_evaluator(*_, **kwargs):
        captured["draft_html"] = kwargs["draft_html"]
        return _evaluation(score=9, verdict="accept", provider="claude")

    monkeypatch.setattr("services.support_pipeline.evaluate_draft", fake_evaluator)

    result = run_support_pipeline(context=_context(knowledge_context="", past_tickets_context=""), max_attempts=1)

    assert "\x1a" not in captured["draft_html"]
    assert "Library → Journal → Index" in captured["draft_html"]
    assert '<a href=' not in captured["draft_html"]
    assert result.reply_html == captured["draft_html"]


def test_support_pipeline_retries_once_and_sends_evaluator_feedback_to_investigator(monkeypatch):
    calls = []
    eval_count = {"value": 0}
    evaluator_internal_notes = []

    def fake_investigator(request, *_, **__):
        calls.append(request)
        return _investigation(reply_html=f"<p>Draft {len(calls)}</p>")

    def fake_evaluator(*_, **kwargs):
        evaluator_internal_notes.append(kwargs.get("internal_note"))
        eval_count["value"] += 1
        if eval_count["value"] == 1:
            return _evaluation(
                score=7,
                verdict="retry",
                summary="Missing setup validation details",
                missing_information=["Exact plugin version"],
                investigation_actions=["expand_kb_search"],
                rewrite_brief="Add exact setup preconditions.",
            )
        return _evaluation(score=9, verdict="accept", summary="Now sufficient")

    monkeypatch.setattr("services.support_pipeline.draft_reply", fake_investigator)
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", fake_evaluator)

    result = run_support_pipeline(
        context=_context(internal_note="Original operator context."),
        max_attempts=2,
    )

    assert result.delivery_mode == "auto_send"
    assert result.attempts == 2
    assert len(calls) == 2
    assert calls[0].internal_note == "Original operator context."
    assert "Original operator context." in calls[1].internal_note
    assert "Evaluator feedback" in calls[1].internal_note
    assert "Missing setup validation details" in calls[1].internal_note
    assert "Exact plugin version" in calls[1].internal_note
    assert evaluator_internal_notes == [
        "Original operator context.",
        "Original operator context.",
    ]


def test_support_pipeline_flags_malformed_provider_output_as_hard_blocker(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(
            confidence_score=0,
            decision_summary="Support investigator failed: invalid JSON",
            risk_flags=["investigator_failure"],
            requires_manual_review=True,
            reply_html="",
        ),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=10, verdict="accept"),
    )

    result = run_support_pipeline(context=_context(), max_attempts=1)

    assert result.delivery_mode == "draft"
    assert any("malformed_provider_output" in blocker for blocker in result.hard_blockers)


def test_support_pipeline_blocks_unverified_claim_that_cancellation_is_complete(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(
            reply_html="<p>I’ve canceled the automatic renewal for your subscription.</p>",
            confidence_score=10,
            risk_flags=["account_action_required", "cancellation_not_verified"],
            requires_manual_review=True,
        ),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=10, verdict="accept"),
    )
    result = run_support_pipeline(context=_context(), max_attempts=1)
    assert result.delivery_mode == "draft"
    assert "unverified_account_action_claim" in result.hard_blockers


def test_support_pipeline_allows_future_cancellation_commitment_as_draft(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(
            reply_html="<p>I’ll cancel the renewal and confirm once it is complete.</p>",
            confidence_score=7,
            risk_flags=["account_action_required", "cancellation_not_verified"],
            requires_manual_review=True,
        ),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=7, verdict="manual_review"),
    )
    result = run_support_pipeline(context=_context(), max_attempts=1)
    assert "unverified_account_action_claim" not in result.hard_blockers


def test_judge_accepted_diagnostic_question_is_not_blocked_by_unknown_cause(monkeypatch):
    monkeypatch.setattr("services.support_pipeline.draft_reply", lambda *a, **kw: _investigation(
        reply_html="<p>The cause is still unknown. Please send the console error from an existing project.</p>",
        risk_flags=["cause_on_existing_projects_unverified"],
    ))
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", lambda **kw: _evaluation(
        score=8, missing_information=["Cannot verify which site has the error until the customer replies"],
    ))

    result = run_support_pipeline(context=_context(), max_attempts=1)

    assert result.delivery_mode == "auto_send"
    assert result.hard_blockers == []


def test_explicit_critical_evidence_risk_still_holds_accepted_draft(monkeypatch):
    monkeypatch.setattr("services.support_pipeline.draft_reply", lambda *a, **kw: _investigation(
        risk_flags=["missing_critical_evidence"],
    ))
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", lambda **kw: _evaluation(score=9))

    result = run_support_pipeline(context=_context(), max_attempts=1)

    assert result.delivery_mode == "draft"
    assert "missing_critical_evidence" in result.hard_blockers


@pytest.mark.parametrize("summary", [
    "This corrects the earlier conflicting explanation using verified source.",
    "No contradiction was found in the source.",
])
def test_accepted_correction_is_not_blocked_by_summary_wording(monkeypatch, summary):
    monkeypatch.setattr("services.support_pipeline.draft_reply", lambda *a, **kw: _investigation(
        decision_summary=summary,
    ))
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", lambda **kw: _evaluation(score=9))

    result = run_support_pipeline(context=_context(), max_attempts=1)

    assert result.delivery_mode == "auto_send"
    assert result.hard_blockers == []


def test_support_pipeline_honors_explicit_evaluator_hard_blocker_names(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=10),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(
            score=10,
            verdict="accept",
            hard_blockers=["malformed_provider_output", "missing_critical_evidence"],
        ),
    )

    result = run_support_pipeline(context=_context(), max_attempts=1)

    assert result.delivery_mode == "draft"
    assert "malformed_provider_output" in result.hard_blockers
    assert "missing_critical_evidence" in result.hard_blockers


def test_support_pipeline_normalizes_evaluator_hard_blocker_aliases(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=10),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(
            score=10,
            verdict="accept",
            hard_blockers=["malformed output", "contradiction"],
        ),
    )

    result = run_support_pipeline(context=_context(), max_attempts=1)

    assert result.delivery_mode == "draft"
    assert "malformed_provider_output" in result.hard_blockers
    assert "contradiction_detected" in result.hard_blockers


def test_support_pipeline_honors_custom_auto_send_threshold(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=8),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=8, verdict="accept"),
    )

    result = run_support_pipeline(context=_context(), target_score=8, max_attempts=1)

    assert result.delivery_mode == "auto_send"
    assert result.confidence_score == 8


def test_pipeline_drafts_through_the_support_agent_once_per_attempt(monkeypatch):
    """The model fallback lives inside the support agent (draft_reply); the pipeline makes
    one drafting call per attempt and passes the agent's per-model timeout through."""
    calls = []

    def fake_draft_reply(request, *, timeout=None, **_):
        calls.append(timeout)
        return _investigation(confidence_score=9, provider_name="claude")

    monkeypatch.setattr("services.support_pipeline.draft_reply", fake_draft_reply)
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=9, verdict="accept", provider="claude"),
    )

    result = run_support_pipeline(context=_context(), max_attempts=1, investigator_timeout=77)

    assert calls == [77]
    assert result.investigator_provider == "claude"
    assert result.delivery_mode == "auto_send"


def test_pipeline_keeps_manual_review_when_every_drafting_model_fails(monkeypatch):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(
            reply_html="", confidence_score=0, provider_name="codex",
            risk_flags=["investigator_failure"], requires_manual_review=True,
        ),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=0, verdict="manual_review", provider="claude"),
    )

    result = run_support_pipeline(context=_context(), max_attempts=1)

    assert result.delivery_mode == "draft"
    assert result.manual_review is True


# --- unreviewed screenshots blocker ------------------------------------

from services.support_pipeline import _has_unreviewed_screenshots


class _Investigation:
    def __init__(self, evidence="", reply="", steps=None, decision=""):
        self.evidence_summary = evidence
        self.decision_summary = decision
        self.investigation_steps = steps or []
        self.reply_html = reply


def _ctx_with_attachments(attachments):
    return SupportPipelineContext(
        customer_name="Reader",
        subject="Index formatting",
        message="Please see my screenshots",
        attachments=attachments,
    )


REAL_SHOT = {"filename": "chapter-index.png", "mime_type": "image/png",
             "size": 825, "is_image": True, "is_blank": False}
BLANK_SHOT = {"filename": "blank.png", "mime_type": "image/png",
              "size": 133, "is_image": True, "is_blank": True}


def test_diagnosing_around_unopened_screenshots_blocks():
    """Source text alone does not establish that an attachment was inspected."""
    investigation = _Investigation(
        evidence="Repo evidence (field-journal): index layout",
        reply="<p>Rebuild the index.</p>",
    )

    assert _has_unreviewed_screenshots(
        context=_ctx_with_attachments([REAL_SHOT]), investigation=investigation
    ) is True


def test_submitted_screenshot_pixels_clear_the_blocker():
    investigation = _Investigation(evidence="The screenshots show duplicate chapter entries.")
    investigation.image_input_count = 1
    ctx = _ctx_with_attachments([REAL_SHOT])
    assert _has_unreviewed_screenshots(context=ctx, investigation=investigation) is False


def test_blank_artifacts_alone_do_not_block():
    """There is nothing to review in a 133-byte placeholder — the right move is telling
    the reader that an image is blank, not blocking."""
    investigation = _Investigation(reply="<p>Answer without mentioning images.</p>")

    assert _has_unreviewed_screenshots(
        context=_ctx_with_attachments([BLANK_SHOT]), investigation=investigation
    ) is False


def test_no_attachments_never_blocks():
    assert _has_unreviewed_screenshots(
        context=_ctx_with_attachments([]), investigation=_Investigation()
    ) is False


# Corrections still pass through the ordinary judge and holds.

from agents.support_agent import DeepPassConfig  # noqa: E402

_NO_DEEP = DeepPassConfig(enabled=False, effort="high", timeout=60)


def _run_correction(monkeypatch, *, flags, verdict="accept", score=9):
    monkeypatch.setattr(
        "services.support_pipeline.draft_reply",
        lambda *_, **__: _investigation(confidence_score=9, requires_manual_review=True, risk_flags=flags),
    )
    monkeypatch.setattr(
        "services.support_pipeline.evaluate_draft",
        lambda *_, **__: _evaluation(score=score, verdict=verdict),
    )
    return run_support_pipeline(context=_context(), max_attempts=1, deep_pass_config=_NO_DEEP)


def test_a_judged_correction_of_an_earlier_reply_auto_sends(monkeypatch):
    result = _run_correction(monkeypatch, flags=["previous_diagnosis_retracted", "corrects_earlier_reply"])
    assert result.delivery_mode == "auto_send"
    assert "manual_review" not in result.hard_blockers


@pytest.mark.parametrize("flags, verdict, score", [
    # Any other self-flag keeps the drafter's hold.
    (["previous_diagnosis_retracted", "test_build_shared_by_us_needs_operator_followup"], "accept", 9),
    # No stated reason: the hold stands.
    ([], "accept", 9),
    # The judge did not accept it.
    (["previous_diagnosis_retracted"], "manual_review", 9),
    # Below the bar it never sends anyway.
    (["previous_diagnosis_retracted"], "accept", 7),
])
def test_other_holds_still_keep_a_correction_as_a_draft(monkeypatch, flags, verdict, score):
    result = _run_correction(monkeypatch, flags=flags, verdict=verdict, score=score)
    assert result.delivery_mode == "draft"


def test_judge_prompt_contains_the_history_and_the_selected_drafting_rules():
    from agents.draft_confidence import build_confidence_prompt
    from agents.support_agent import load_rules

    history = "Reader: The first chapter already has an index."
    text = build_confidence_prompt(customer_name="Reader", subject="Index", message="What about the appendix?",
        draft_html="<p>Each chapter has its own index.</p>", conversation_history=history)
    assert history in text and load_rules() in text


def test_pipeline_forwards_the_same_history_to_the_independent_judge(monkeypatch):
    history = "Reader: I have already indexed the first chapter."
    captured = []
    monkeypatch.setattr("services.support_pipeline.draft_reply", lambda *args, **kwargs: _investigation())
    def judge(**kwargs):
        captured.append(kwargs["conversation_history"])
        return _evaluation(score=9)
    monkeypatch.setattr("services.support_pipeline.evaluate_draft", judge)
    run_support_pipeline(context=_context(conversation_history=history), max_attempts=1)
    assert captured == [history]
