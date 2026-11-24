import json
import uuid
from dataclasses import dataclass

from agents.semantic_review import SemanticReviewResult, review_sent_reply
from db.database import get_ticket_log_by_helpscout_id, log_semantic_failure
from services.conversation_locks import conversation_lease


@dataclass
class SemanticFailureLearningOutcome:
    recorded: bool
    note_added: bool
    review: SemanticReviewResult | None = None
    failure_id: int | None = None


def _deserialize_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value.strip()]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [str(parsed).strip()]
    return [str(value).strip()]


def _build_compat_confidence_summary(ticket_log: dict[str, object]) -> str:
    return str(ticket_log.get("confidence_summary") or ticket_log.get("decision_note") or "").strip()


def _build_compat_evidence_summary(ticket_log: dict[str, object]) -> str:
    parts: list[str] = []
    evidence_summary = str(ticket_log.get("evidence_summary") or "").strip()
    if evidence_summary:
        parts.append(evidence_summary)

    sources = _deserialize_string_list(ticket_log.get("sources"))
    if sources:
        parts.append(f"Sources: {', '.join(sources[:3])}")

    investigation_steps = _deserialize_string_list(ticket_log.get("investigation_steps"))
    if investigation_steps:
        parts.append(f"Steps: {'; '.join(investigation_steps[:3])}")

    risk_flags = _deserialize_string_list(ticket_log.get("risk_flags"))
    if risk_flags:
        parts.append(f"Risk flags: {', '.join(risk_flags[:3])}")

    return " | ".join(parts)


def build_auto_improvement_note(review: SemanticReviewResult) -> str:
    from services.helpscout import recorded

    return recorded(
        "Supa auto-improvement note: the final sent reply materially changed the meaning "
        "of Supa's draft, so this ticket was logged as a learning signal for future drafts.\n\n"
        f"Detected change: {review.change_type}\n"
        f"Summary: {review.summary}\n"
        "No action needed here.",
        {"kind": "learning", "actor": "supa", "title": "The sent reply changed the draft's meaning",
         "data": {"change": str(review.change_type or ""), "why": str(review.summary or "")}},
    )


async def _get_conversation_status(helpscout_client, conversation_id: int) -> str:
    get_conversation = getattr(helpscout_client, "get_conversation", None)
    if not callable(get_conversation):
        return ""

    try:
        conversation = await get_conversation(conversation_id)
    except Exception:
        return ""

    return str((conversation or {}).get("status") or "").strip().lower()


async def add_auto_improvement_note_preserving_pending_status(
    *,
    helpscout_client,
    conversation_id: int,
    note_text: str,
    db_path: str | None = None,
) -> bool:
    owner = f"semantic-failure-note:{conversation_id}:{uuid.uuid4().hex}"
    async with conversation_lease(db_path, conversation_id, owner=owner) as lease:
        original_status = await _get_conversation_status(helpscout_client, conversation_id)

        note_added = await lease.run_guarded(
            helpscout_client.add_note,
            conversation_id,
            note_text,
        )
        if not note_added:
            return False

        if original_status == "pending":
            current_status = await _get_conversation_status(
                helpscout_client, conversation_id
            )
            if current_status != "pending":
                ensure_status = getattr(
                    helpscout_client, "ensure_conversation_status", None
                )
                if callable(ensure_status):
                    await lease.run_guarded(
                        ensure_status, conversation_id, "pending"
                    )
                else:
                    set_status = getattr(
                        helpscout_client, "set_conversation_status", None
                    )
                    if callable(set_status):
                        await lease.run_guarded(
                            set_status, conversation_id, "pending"
                        )

        return True


async def learn_from_sent_reply(
    *,
    conversation_id: int,
    subject: str,
    customer_message: str,
    final_reply_html: str,
    theme_tag: str,
    helpscout_client,
    db_path: str | None = None,
    provider: str = "claude",
) -> SemanticFailureLearningOutcome:
    ticket_log = get_ticket_log_by_helpscout_id(db_path=db_path, helpscout_id=str(conversation_id))
    if not ticket_log:
        return SemanticFailureLearningOutcome(recorded=False, note_added=False)

    supa_draft_html = ticket_log.get("draft_response_html") or ticket_log.get("reply_html") or ""
    if not supa_draft_html:
        return SemanticFailureLearningOutcome(recorded=False, note_added=False)

    confidence_summary = _build_compat_confidence_summary(ticket_log)
    evidence_summary = _build_compat_evidence_summary(ticket_log)

    review = review_sent_reply(
        subject=subject,
        customer_message=customer_message,
        supa_draft_html=supa_draft_html,
        final_reply_html=final_reply_html,
        theme_tag=theme_tag,
        confidence_summary=confidence_summary,
        evidence_summary=evidence_summary,
        provider=provider,
    )
    if review.same_meaning:
        return SemanticFailureLearningOutcome(recorded=False, note_added=False, review=review)

    failure_id = log_semantic_failure(
        db_path=db_path,
        helpscout_id=str(conversation_id),
        subject=subject,
        theme_tag=theme_tag,
        customer_message=customer_message,
        draft_html=supa_draft_html,
        final_reply_html=final_reply_html,
        change_type=review.change_type,
        summary=review.summary,
        wrong_claim=review.wrong_claim,
        correct_claim=review.correct_claim,
        confidence_score=ticket_log.get("confidence_score"),
        confidence_status=ticket_log.get("confidence_status") or "",
        confidence_provider=ticket_log.get("confidence_provider") or ticket_log.get("evaluator_provider"),
        confidence_trace=ticket_log.get("confidence_trace"),
        evidence_summary=evidence_summary or ticket_log.get("evidence_summary"),
    )

    note_added = await add_auto_improvement_note_preserving_pending_status(
        helpscout_client=helpscout_client,
        conversation_id=conversation_id,
        note_text=build_auto_improvement_note(review),
        db_path=db_path,
    )
    return SemanticFailureLearningOutcome(
        recorded=True,
        note_added=bool(note_added),
        review=review,
        failure_id=failure_id,
    )
