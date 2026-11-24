from agents.draft_confidence import DraftConfidenceResult
from agents.support_agent import SupportInvestigationResult
from services.support_pipeline import SupportPipelineContext, _collect_hard_blockers


def test_present_image_cannot_be_dismissed_as_missing_by_mentioning_it():
    context = SupportPipelineContext(customer_name="Reader", subject="Error", message="Page shown below",
        attachments=[{"filename": "image.png", "is_image": True, "is_blank": False, "is_signature": False}])
    investigation = SupportInvestigationResult(
        reply_html="<p>Your screenshot didn't come through, only a small image that looks like a signature logo.</p>",
        evidence_summary="Screenshot attachment checked.", decision_summary="Answer", confidence_score=8)
    blockers = _collect_hard_blockers(context=context, investigation=investigation,
        evaluation=DraftConfidenceResult(score=8, verdict="accept", summary="Accepted"))
    assert "image_receipt_contradiction" in blockers


def test_signature_and_transport_artifacts_do_not_force_image_review():
    context = SupportPipelineContext(customer_name="Reader", subject="Error", message="Missing screenshot",
        attachments=[{"is_image": True, "is_blank": True}, {"is_image": False, "is_signature": True}])
    investigation = SupportInvestigationResult(reply_html="<p>Your screenshot didn't come through.</p>", confidence_score=8, decision_summary="", evidence_summary="")
    blockers = _collect_hard_blockers(context=context, investigation=investigation,
        evaluation=DraftConfidenceResult(score=8, verdict="accept", summary="Accepted"))
    assert "image_receipt_contradiction" not in blockers


def test_duplicate_conversation_flag_is_a_delivery_hold():
    context = SupportPipelineContext(customer_name="Reader", subject="Question", message="Same question")
    investigation = SupportInvestigationResult(reply_html="<p>Second answer.</p>", confidence_score=8, decision_summary="", evidence_summary="",
        risk_flags=["duplicate_conversation_same_customer"])
    blockers = _collect_hard_blockers(context=context, investigation=investigation,
        evaluation=DraftConfidenceResult(score=8, verdict="accept", summary="Accepted"))
    assert "duplicate_conversation_same_customer" in blockers


def test_source_bound_duplicate_and_known_data_loss_flags_hold_even_at_nine():
    context = SupportPipelineContext(customer_name="Reader", subject="Question", message="Same question", duplicate_conversation_ids=[20])
    investigation = SupportInvestigationResult(reply_html="<p>Edit the live project.</p>", confidence_score=9,
        decision_summary="", evidence_summary="", risk_flags=["known_data_loss_risk"])
    blockers = _collect_hard_blockers(context=context, investigation=investigation,
        evaluation=DraftConfidenceResult(score=9, verdict="accept", summary="Accepted"))
    assert {"duplicate_conversation_same_customer", "known_data_loss_risk"} <= set(blockers)
