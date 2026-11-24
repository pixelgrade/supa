import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from services.helpscout import (
    HelpScoutClient,
    get_draft_thread_ids,
    get_supa_owned_draft_thread_ids,
    get_latest_note_thread,
    get_latest_customer_thread,
    is_supa_ignored,
    material_customer_images,
    parse_webhook_payload,
    strip_html,
)
from services.company_pack import PACK_DIR, product_settings, wording

from services.knowledge import get_all_knowledge
from services.vectorstore import VectorStore
from services.crawler import crawl_theme_docs
from services.themes import VALID_SLUGS, is_lt_theme
from services.past_tickets import extract_keywords, find_similar_tickets, format_past_tickets_context
from services.websearch import search_web, format_web_results, build_search_query
from services.customer import fetch_customer_data, format_customer_context
from services.theme_history import theme_hint_for_ticket
from services.support_context import current_product_context, related_customer_context, duplicate_question_ids
from services.support_attempts import register_attempt
from services.repo_evidence import detect_repo_contradiction, get_repo_evidence_context
from services.support_pipeline import SupportPipelineContext, decision_record, run_support_pipeline
from services.deep_pass import DeepEvidence, deep_actions, other_customer_emails
from services.theme_requirements import get_classic_theme_requirements_context
from agents.classifier import classify_ticket, confirm_wrong_address, generate_subject
from services import mailbox_provider
from agents.claude_cli import should_skip_claude
from agents.draft_confidence import DraftConfidenceResult, evaluate_draft
from agents.responder import extract_context_urls, finalize_response_html, generate_response, rewrite_response
from db.database import (
    fetch_scheduled_pending_sends_by_conversation,
    get_recent_semantic_failures,
    get_ticket_log_by_helpscout_id,
    init_db,
    insert_pending_send,
    log_ticket,
    update_pending_send_state,
)
from services.autosend_scheduler import (
    AutoSendConfig,
    load_config as load_autosend_config,
    pick_due_at,
    sha256_of_html,
)
from services.draft_confidence_loop import DraftLoopResult, run_draft_confidence_loop
from services.runtime_paths import get_supa_db_path
from services.email_html import email_html_equivalent
from services.conversation_locks import ConversationLockError, conversation_lease
from agents import support_agent

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_DIR = PACK_DIR
DB_DIR = BASE_DIR / "db"
DEFAULT_DB_DIR = DB_DIR


def _get_audit_db_path() -> str:
    """Resolve the audit DB while preserving explicit DB_DIR test overrides."""
    if DB_DIR != DEFAULT_DB_DIR:
        return str(DB_DIR / "supa.db")
    return get_supa_db_path()


@asynccontextmanager
async def _conversation_mutation_lease(
    *,
    db_path: str,
    conversation_id: int,
    owner_scope: str,
    owned_lease=None,
):
    """Reuse an outer lease or acquire one immediately around a write sequence."""

    if owned_lease is not None:
        await owned_lease.verify_owned()
        yield owned_lease
        return
    owner = f"{owner_scope}:{conversation_id}:{uuid.uuid4().hex}"
    async with conversation_lease(db_path, conversation_id, owner=owner) as lease:
        yield lease


def _get_support_provider_env(name: str, default: str) -> str:
    value = (os.getenv(name, default) or "").strip().lower()
    return value or default


def _get_support_autosend_threshold() -> int:
    return support_agent.autosend_threshold()


def _manual_review_placeholder_html() -> str:
    return "<p>Manual review required before sending a reply.</p>"


def _is_placeholder_manual_review_html(html: str) -> bool:
    return strip_html(html or "") == strip_html(_manual_review_placeholder_html())


def _has_unsafe_visible_draft_blocker(hard_blockers: list[str] | None) -> bool:
    package_guard = product_settings().get("package_guard", {})
    unsafe = {"unverified_account_action_claim"}
    if package_guard:
        unsafe.add(package_guard["blocker"])
    return bool(unsafe.intersection(hard_blockers or ()))


def _should_preserve_existing_draft(
    *,
    trigger: str,
    existing_row: dict[str, Any] | None,
    response_html: str,
    confidence_score: int | None,
    risk_flags: list[str] | None,
    hard_blockers: list[str] | None,
) -> bool:
    if trigger not in {"note_trigger", "manual_run"} or not existing_row:
        return False

    existing_score = existing_row.get("confidence_score")
    if existing_score is None:
        return False

    existing_html = existing_row.get("draft_response_html") or ""
    if not existing_row.get("response_posted"):
        return False
    if (existing_row.get("delivery_mode") or "") != "draft":
        return False
    if not strip_html(existing_html) or _is_placeholder_manual_review_html(existing_html):
        return False

    current_score = int(confidence_score or 0)
    current_flags = set(risk_flags or [])
    current_blockers = set(hard_blockers or [])
    degraded = (
        not strip_html(response_html or "")
        or _is_placeholder_manual_review_html(response_html or "")
        or "investigator_failure" in current_flags
        or "empty_draft" in current_flags
        or "malformed_provider_output" in current_blockers
    )

    if not degraded:
        return False
    if int(existing_score) > current_score:
        return True
    return (
        trigger == "manual_run"
        and int(existing_score) == current_score
        and (not strip_html(response_html or "") or _is_placeholder_manual_review_html(response_html))
    )


def _restore_result_from_existing_draft(result: Dict[str, Any], existing_row: dict[str, Any]) -> Dict[str, Any]:
    result["delivery_mode"] = existing_row.get("delivery_mode") or result.get("delivery_mode")
    result["confidence_score"] = existing_row.get("confidence_score")
    result["confidence_status"] = existing_row.get("confidence_status")
    result["confidence_attempts"] = existing_row.get("confidence_attempts")
    result["confidence_summary"] = existing_row.get("confidence_summary")
    result["decision_note"] = existing_row.get("decision_note")
    result["response_posted"] = bool(existing_row.get("response_posted"))
    return result


def _parse_db_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _format_scheduled_send_time(value: str) -> str:
    parsed = _parse_iso_timestamp(value)
    if parsed is None:
        return value
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _append_delayed_autosend_schedule(decision_note: str | None, due_at: str | None) -> str | None:
    if not decision_note or not due_at:
        return decision_note
    if "Scheduled send:" in decision_note:
        return decision_note
    scheduled_time = _format_scheduled_send_time(due_at)
    return (
        f"{decision_note.rstrip()}\n"
        f"Scheduled send: {scheduled_time}\n"
        "Auto-send will cancel if the customer or an agent changes the conversation first."
    )


def _note_trigger_is_stale_after_newer_send(
    *,
    payload: dict[str, Any],
    trigger: str,
    existing_row: dict[str, Any] | None,
) -> bool:
    if trigger != "note_trigger" or not existing_row:
        return False
    if not existing_row.get("response_posted"):
        return False
    if (existing_row.get("delivery_mode") or "") != "auto_send":
        return False

    note_thread = get_latest_note_thread(payload.get("_embedded", {}).get("threads", []))
    note_created_at = _parse_iso_timestamp((note_thread or {}).get("createdAt"))
    processed_at = _parse_db_timestamp(existing_row.get("created_at"))
    if note_created_at is None or processed_at is None:
        return False

    return processed_at >= note_created_at


def build_evidence_summary(
    *,
    knowledge_context: str = "",
    past_tickets_context: str = "",
    customer_context: str = "",
    internal_note: str = "",
    repo_context: str = "",
) -> str:
    parts = []
    if knowledge_context:
        parts.append(f"Knowledge context:\n{knowledge_context.strip()}")
    if repo_context:
        parts.append(f"Repo evidence:\n{repo_context.strip()}")
    if past_tickets_context:
        parts.append(f"Past tickets context:\n{past_tickets_context.strip()}")
    if customer_context:
        parts.append(f"Customer context:\n{customer_context.strip()}")
    if internal_note:
        parts.append(f"Internal note:\n{internal_note.strip()}")
    return "\n\n".join(part for part in parts if part).strip()


def merge_context_sections(*sections: str) -> str:
    return "\n\n---\n\n".join(section.strip() for section in sections if section and section.strip()).strip()


def _extract_quoted_labels(text: str) -> list[str]:
    labels = []
    for match in re.finditer(r'"([^"\n]{2,80})"', text or ""):
        label = match.group(1).strip()
        if re.search(r"[A-Za-z]", label):
            labels.append(label)
    return labels


def _keyword_overlap(left: str, right: str) -> int:
    left_keywords = set(extract_keywords(left, max_keywords=8))
    right_keywords = set(extract_keywords(right, max_keywords=8))
    return len(left_keywords & right_keywords)


def _cap_evaluation_for_retry(evaluation: DraftConfidenceResult, reason: str) -> DraftConfidenceResult:
    if evaluation.score <= 7:
        return evaluation

    actions = list(dict.fromkeys((evaluation.investigation_actions or []) + ["expand_past_ticket_search"]))
    risk_flags = list(dict.fromkeys((evaluation.risk_flags or []) + ["semantic_failure_guard"]))
    summary = (evaluation.summary or "").strip()
    if summary:
        summary = f"{summary} Guard: {reason}"
    else:
        summary = f"Guard: {reason}"

    return replace(
        evaluation,
        score=7,
        verdict="retry",
        summary=summary,
        investigation_actions=actions,
        rewrite_brief=evaluation.rewrite_brief or reason,
        risk_flags=risk_flags,
    )


def apply_confidence_guard(
    *,
    draft_html: str,
    evaluation: DraftConfidenceResult,
    theme_tag: str,
    evidence_summary: str,
    repo_context: str = "",
    db_path: str | None = None,
    autosend_min: int | None = None,
) -> DraftConfidenceResult:
    draft_text = strip_html(draft_html or "")
    # The repo-contradiction and setting-label checks guard every score that could auto-send.
    autosend_min = _get_support_autosend_threshold() if autosend_min is None else autosend_min
    evidence_text = (evidence_summary or "").lower()

    if theme_tag:
        for failure in get_recent_semantic_failures(db_path=db_path, theme_tag=theme_tag, limit=20):
            wrong_claim = failure.get("wrong_claim") or ""
            if wrong_claim and _keyword_overlap(draft_text, wrong_claim) >= 2:
                return _cap_evaluation_for_retry(
                    evaluation,
                    "draft matches a known semantic failure for this theme",
                )

    if evaluation.score >= autosend_min:
        repo_contradiction = detect_repo_contradiction(
            draft_html=draft_html,
            repo_context=repo_context or evidence_summary,
        )
        if repo_contradiction:
            return _cap_evaluation_for_retry(evaluation, repo_contradiction)

        labels = _extract_quoted_labels(draft_text)
        missing = [label for label in labels if label.lower() not in evidence_text]
        if missing:
            joined = ", ".join(missing)
            return _cap_evaluation_for_retry(
                evaluation,
                f'draft references setting labels missing from the evidence: {joined}',
            )

    return evaluation


async def generate_draft_with_confidence(
    *,
    customer_name: str,
    subject: str,
    message: str,
    theme_tag: str,
    knowledge_context: str,
    past_tickets_context: str = "",
    internal_note: str = "",
    is_classic_theme: bool = False,
    conversation_history: list | None = None,
    allowed_urls: set | None = None,
    customer_context: str = "",
    repo_context: str = "",
    confidence_provider: str | None = None,
    investigate_more=None,
) -> DraftLoopResult:
    """Generate a draft, then evaluate and improve it through the confidence loop."""

    provider = (confidence_provider or os.getenv("SUPA_CONFIDENCE_PROVIDER", "claude")).strip().lower()

    combined_knowledge_context = merge_context_sections(knowledge_context, repo_context)

    initial_html = await generate_response(
        customer_name=customer_name,
        subject=subject,
        message=message,
        theme_tag=theme_tag,
        knowledge_context=combined_knowledge_context,
        past_tickets_context=past_tickets_context,
        internal_note=internal_note,
        is_classic_theme=is_classic_theme,
        conversation_history=conversation_history,
        allowed_urls=allowed_urls,
        customer_context=customer_context,
    )

    if not initial_html:
        return DraftLoopResult(
            final_html=None,
            final_score=0,
            final_status="manual_review",
            attempts=1,
            summary="Initial draft generation failed",
            trace=[],
            provider=provider,
        )

    async def default_investigate_more(**_kwargs):
        return {}

    investigate_callback = investigate_more or default_investigate_more

    def evaluate_fn(*, draft_html, **context):
        return evaluate_draft(
            customer_name=customer_name,
            subject=subject,
            message=message,
            draft_html=draft_html,
            theme_tag=theme_tag,
            knowledge_context=merge_context_sections(
                context.get("knowledge_context", knowledge_context),
                context.get("repo_context", repo_context),
            ),
            past_tickets_context=context.get("past_tickets_context", past_tickets_context),
            customer_context=context.get("customer_context", customer_context),
            internal_note=internal_note,
            provider=provider,
        )

    async def rewrite_fn(*, original_draft, rewrite_brief, additional_context, **context):
        return await rewrite_response(
            customer_name=customer_name,
            subject=subject,
            message=message,
            original_draft=original_draft,
            rewrite_brief=rewrite_brief,
            additional_context=additional_context,
            theme_tag=theme_tag,
            knowledge_context=merge_context_sections(
                context.get("knowledge_context", knowledge_context),
                context.get("repo_context", repo_context),
            ),
            past_tickets_context=context.get("past_tickets_context", past_tickets_context),
            internal_note=internal_note,
            allowed_urls=context.get("allowed_urls", allowed_urls),
        )

    def guard_fn(*, draft_html, evaluation, **context):
        evidence_summary = build_evidence_summary(
            knowledge_context=context.get("knowledge_context", knowledge_context),
            repo_context=context.get("repo_context", repo_context),
            past_tickets_context=context.get("past_tickets_context", past_tickets_context),
            customer_context=context.get("customer_context", customer_context),
            internal_note=internal_note,
        )
        return apply_confidence_guard(
            draft_html=draft_html,
            evaluation=evaluation,
            theme_tag=theme_tag,
            evidence_summary=evidence_summary,
            repo_context=context.get("repo_context", repo_context),
            db_path=_get_audit_db_path(),
        )

    return await run_draft_confidence_loop(
        initial_html=initial_html,
        initial_context={
            "knowledge_context": knowledge_context,
            "repo_context": repo_context,
            "past_tickets_context": past_tickets_context,
            "customer_context": customer_context,
            "allowed_urls": allowed_urls or extract_context_urls(combined_knowledge_context, past_tickets_context),
        },
        evaluate_fn=evaluate_fn,
        investigate_fn=investigate_callback,
        rewrite_fn=rewrite_fn,
        guard_fn=guard_fn,
    )


async def investigate_ticket_for_confidence(
    *,
    processor: "TicketProcessor",
    ticket_id: int,
    subject: str,
    message: str,
    theme_tag: str = "",
    customer_email: str = "",
    evaluation,
    attempt: int,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Expand the evidence pool when a draft scores below the retry threshold."""

    actions = set(evaluation.investigation_actions or [])
    updated_context: Dict[str, Any] = {}
    notes: list[str] = []
    query_bits = [subject, message]
    query_bits.extend(evaluation.missing_information or [])
    expanded_query = " ".join(bit for bit in query_bits if bit).strip()

    if "expand_kb_search" in actions or "check_theme_compatibility" in actions:
        knowledge_results = processor.vector_store.search(
            query=expanded_query or f"{subject} {message}",
            theme_filter=theme_tag if theme_tag else None,
            limit=5,
        )
        if knowledge_results:
            kb_context = "\n\n---\n\n".join([
                f"**{r.get('title', 'Document')}**\n{'URL: ' + r['url'] if r.get('url') else '(no URL available)'}\n{r['content'][:700]}..."
                for r in knowledge_results
            ])
            updated_context["knowledge_context"] = kb_context
            notes.append("Expanded knowledge base search with more results.")

    if "expand_past_ticket_search" in actions:
        similar_tickets = await find_similar_tickets(
            helpscout_client=processor.helpscout,
            theme_tag=theme_tag,
            subject=subject,
            message=expanded_query or message,
            exclude_id=ticket_id,
            limit=5,
        )
        past_tickets_context = format_past_tickets_context(similar_tickets)
        if past_tickets_context:
            updated_context["past_tickets_context"] = past_tickets_context
            notes.append("Expanded past ticket search.")

    if "force_web_search" in actions or "check_theme_compatibility" in actions:
        search_query = build_search_query(
            subject=subject,
            message=f"{message}\n{' '.join(evaluation.missing_information or [])}".strip(),
            theme_tag=theme_tag,
        )
        web_results = await search_web(search_query, limit=5)
        web_context = format_web_results(web_results)
        if web_context:
            merged_knowledge = updated_context.get("knowledge_context") or context.get("knowledge_context", "")
            updated_context["knowledge_context"] = f"{merged_knowledge}\n\n{web_context}".strip() if merged_knowledge else web_context
            notes.append("Forced web search for missing evidence.")

    if "refresh_customer_context" in actions and customer_email:
        customer_data = await fetch_customer_data(customer_email)
        customer_context = format_customer_context(customer_data) if customer_data else ""
        if customer_context:
            updated_context["customer_context"] = customer_context
            notes.append("Refreshed customer context.")

    if "request_openclaw_second_opinion" in actions:
        notes.append("OpenClaw second-opinion requested but not yet enabled in the automatic investigator.")

    knowledge_context = updated_context.get("knowledge_context", context.get("knowledge_context", ""))
    past_tickets_context = updated_context.get("past_tickets_context", context.get("past_tickets_context", ""))
    updated_context["allowed_urls"] = extract_context_urls(knowledge_context, past_tickets_context)
    updated_context["additional_context"] = "\n".join(notes + ([f"Attempt {attempt} evaluator summary: {evaluation.summary}"] if evaluation.summary else []))

    return updated_context


DEEP_EVIDENCE_TIMEOUT = 240  # seconds for the deep pass's evidence collection
DEEP_IMAGE_MAX_BYTES = 8 * 1024 * 1024
_DEEP_IMAGE_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


async def collect_deep_evidence(
    *,
    processor: "TicketProcessor",
    ticket: Dict[str, Any],
    evaluation: DraftConfidenceResult,
    workdir: Path,
) -> DeepEvidence:
    """Evidence for the deep pass: the judge's actions plus the baseline searches, orders for
    every other address the customer names, and their screenshots saved into ``workdir``."""
    customer_email = ticket.get("customer_email", "") or ""
    expanded = await investigate_ticket_for_confidence(
        processor=processor,
        ticket_id=ticket["id"],
        subject=ticket["subject"],
        message=ticket["message"],
        theme_tag=ticket.get("theme_tag") or "",
        customer_email=customer_email,
        evaluation=replace(evaluation, investigation_actions=deep_actions(evaluation)),
        attempt=2,
        context={},
    )
    notes = [
        line for line in (expanded.get("additional_context") or "").splitlines()
        if line and not line.startswith("Attempt ")
    ]
    customer_blocks = [expanded.get("customer_context") or ""]
    # fetch_customer_data returns None both for "no account" and for a failed or unconfigured
    # lookup, so an empty result is reported as exactly that, never as "no purchase".
    empty_lookups = [] if customer_blocks[0] or not customer_email else [customer_email]
    for email in other_customer_emails(ticket["message"], customer_email):
        data = await fetch_customer_data(email)
        block = format_customer_context(data) if data else ""
        if block:
            customer_blocks.append(
                f"Lookup for {email}, an address the customer named in the message:\n{block}"
            )
            notes.append(f"Looked up a named address ({email}).")
        else:
            empty_lookups.append(email)
    if empty_lookups:
        customer_blocks.append(
            wording("account_unavailable").format(emails=", ".join(empty_lookups))
        )
        notes.append(f"Customer lookup returned nothing for {len(empty_lookups)} address(es).")

    image_paths: list[Path] = []
    for index, attachment in enumerate(material_customer_images(ticket.get("attachments") or []), start=1):
        if not attachment.get("id") and not attachment.get("url"):
            continue
        try:
            if attachment.get("id"):
                data, _mime = await processor.helpscout.get_attachment_data(
                    ticket["id"], int(attachment["id"]), max_bytes=DEEP_IMAGE_MAX_BYTES
                )
            else:  # pasted into the message body (Help Scout inline CDN)
                data = await processor.helpscout.get_inline_image_data(
                    attachment["url"], max_bytes=DEEP_IMAGE_MAX_BYTES
                )
        except Exception as exc:
            notes.append(f"Screenshot {attachment.get('filename') or index} could not be downloaded: {exc}")
            continue
        suffix = _DEEP_IMAGE_SUFFIXES.get(attachment.get("mime_type") or "", ".png")
        path = workdir / f"screenshot-{index}{suffix}"
        path.write_bytes(data)
        image_paths.append(path)
    if image_paths:
        notes.append(f"Downloaded {len(image_paths)} screenshot(s).")

    return DeepEvidence(
        knowledge_context=expanded.get("knowledge_context") or "",
        past_tickets_context=expanded.get("past_tickets_context") or "",
        customer_context="\n\n".join(block for block in customer_blocks if block),
        image_paths=image_paths,
        notes=notes,
    )


def deep_evidence_collector(*, processor: "TicketProcessor", ticket: Dict[str, Any]):
    """A sync collector for the pipeline's worker thread. Call it on the event loop: the
    collection is async (Help Scout, the WP customer endpoint) and runs back on that loop."""
    loop = asyncio.get_running_loop()

    def collect(evaluation: DraftConfidenceResult, workdir: Path) -> DeepEvidence:
        return asyncio.run_coroutine_threadsafe(
            collect_deep_evidence(processor=processor, ticket=ticket, evaluation=evaluation, workdir=workdir),
            loop,
        ).result(timeout=DEEP_EVIDENCE_TIMEOUT)

    return collect


async def process_auto_send(
    *,
    helpscout,
    conversation_id: int,
    customer_id: Optional[int],
    response_html: str,
    tags_to_add: list,
    existing_tags: list,
    confidence_score: Optional[int],
    confidence_provider: Optional[str],
    db_path: str,
    config: Optional[AutoSendConfig] = None,
    now: Optional[datetime] = None,
    owned_lease=None,
    expected_customer_thread_id=None,
) -> dict:
    """Execute the auto-send delivery path.

    If SUPA_AUTOSEND_DELAY_ENABLED is true (default): write a HelpScout draft,
    queue a pending_sends row, add 'supa-pending-send' to tags_to_add.
    If false: fall through to immediate send+status (today's behavior).

    Mutates tags_to_add in place. Returns a dict with 'response_posted': True.
    """
    cfg = config or load_autosend_config()
    async with _conversation_mutation_lease(
        db_path=db_path,
        conversation_id=conversation_id,
        owner_scope="processor-auto-send",
        owned_lease=owned_lease,
    ) as lease:
        prior_conv = await helpscout.get_conversation(conversation_id)
        prior_customer = get_latest_customer_thread(prior_conv.get("_embedded", {}).get("threads", []))
        if expected_customer_thread_id is None:
            expected_customer_thread_id = (prior_customer or {}).get("id")
        if str((prior_customer or {}).get("id")) != str(expected_customer_thread_id):
            raise RuntimeError("customer message changed before auto-send")
        if not cfg.enabled:
            if "supa-auto-sent" not in (existing_tags or []) and "supa-auto-sent" not in tags_to_add:
                tags_to_add.append("supa-auto-sent")
            sent = await lease.run_guarded(
                helpscout.send_reply,
                conversation_id,
                response_html,
                customer_id=customer_id,
            )
            if not sent:
                raise RuntimeError("HelpScout auto-send failed")
            pending_result = await lease.run_guarded(
                helpscout.ensure_conversation_status,
                conversation_id,
                "pending",
            )
            if not pending_result.get("verified"):
                raise RuntimeError("HelpScout pending status update failed")
            return {"response_posted": True, "autosend_delayed": False}

        prior_draft_ids = set(
            str(tid) for tid in get_draft_thread_ids(
                prior_conv.get("_embedded", {}).get("threads", []) or []
            )
        )

        saved = await lease.run_guarded(
            helpscout.add_draft,
            conversation_id,
            response_html,
            customer_id,
        )
        if not saved:
            raise RuntimeError("HelpScout draft save failed")

        post_conv = await helpscout.get_conversation(conversation_id)
        post_threads = post_conv.get("_embedded", {}).get("threads", []) or []
        post_draft_ids = [str(tid) for tid in get_draft_thread_ids(post_threads)]
        new_draft_ids = [tid for tid in post_draft_ids if tid not in prior_draft_ids]
        if not new_draft_ids:
            raise RuntimeError("Could not identify newly-added draft thread for scheduling")
        new_draft_thread_id = new_draft_ids[-1]
        persisted = next(t for t in post_threads if str(t.get("id")) == new_draft_thread_id)
        persisted_html = persisted.get("body") or ""
        from services.helpscout import ensure_email_paragraph_spacing
        if (new_draft_thread_id not in {str(tid) for tid in get_supa_owned_draft_thread_ids(post_threads)}
                or not (email_html_equivalent(response_html, persisted_html)
                        or email_html_equivalent(ensure_email_paragraph_spacing(response_html), persisted_html))):
            raise RuntimeError("HelpScout draft readback differs from the approved reply")

        post_customer = get_latest_customer_thread(post_threads)
        if str((post_customer or {}).get("id")) != str(expected_customer_thread_id):
            # Delete only this run's verified unchanged new draft, never a newer
            # customer's or operator's work. It was never admitted to the queue.
            await lease.run_guarded(helpscout.delete_draft, conversation_id, int(new_draft_thread_id))
            raise RuntimeError("customer message changed while saving auto-send draft")

        prior_scheduled = fetch_scheduled_pending_sends_by_conversation(
            db_path=db_path, conversation_id=str(conversation_id),
        )
        for prior in prior_scheduled:
            update_pending_send_state(
                db_path=db_path, row_id=prior["id"],
                state="cancelled", cancel_reason="superseded",
            )

        due_at = pick_due_at(now=now or datetime.now(timezone.utc), config=cfg)
        insert_pending_send(
            db_path=db_path,
            conversation_id=str(conversation_id),
            draft_thread_id=new_draft_thread_id,
            response_html_sha=sha256_of_html(persisted_html),
            due_at=due_at,
            last_known_customer_thread_id=str(expected_customer_thread_id),
            tags_on_promote=["supa-auto-sent"],
            confidence_score=confidence_score,
            confidence_provider=confidence_provider,
        )

        if "supa-pending-send" not in (existing_tags or []) and "supa-pending-send" not in tags_to_add:
            tags_to_add.append("supa-pending-send")

        return {
            "response_posted": True,
            "autosend_delayed": True,
            "due_at": due_at,
            "draft_thread_id": new_draft_thread_id,
        }


class TicketProcessor:
    """Process incoming HelpScout tickets."""

    def __init__(self, auto_index: bool = True):
        self.helpscout = HelpScoutClient(
            client_id=os.getenv("HELPSCOUT_CLIENT_ID", ""),
            client_secret=os.getenv("HELPSCOUT_CLIENT_SECRET", "")
        )

        # Initialize vector store
        DB_DIR.mkdir(exist_ok=True)
        self.vector_store = VectorStore(str(DB_DIR / "vectors.db"))

        # Initialize database
        init_db(_get_audit_db_path())

        # Auto-index for app runtime, but allow explicit rebuild flows to skip it.
        if auto_index:
            self._auto_index_if_needed()

    async def _handle_wrong_address(self, ticket, tags_to_add, result, *, payload, delivery_ceiling, trigger):
        """Send the mailbox's wrong-address reply, or say why not.

        Returns the finished result, or None when this is not a first message that a second
        reading also calls wrong-address; the ticket is then handled as support.

        The reply goes out only when all of these hold on a fresh read under the lease:
        the conversation is one customer message and nothing from us, and that message is
        the one classified; a second, independent reading says wrong-address with at least
        SUPA_WRONG_ADDRESS_MIN_CONFIDENCE (default 0.90); the mailbox's own intake check
        tagged it `possible-wrong-address` (Supa never writes that tag); this run may
        deliver, support delivery is live and the owned mailbox is selected. Anything less
        leaves the conversation open with Supa's own hold tags and the reason in the audit.
        """
        def read(conversation):
            raw = conversation.get("tags")
            tags = ticket["tags"]
            if isinstance(raw, list):
                tags = [
                    tag.get("tag") if isinstance(tag, dict) else tag
                    for tag in raw
                    if isinstance(tag, str) or (isinstance(tag, dict) and isinstance(tag.get("tag"), str))
                ]
            threads = [t for t in ((conversation.get("_embedded") or {}).get("threads") or []) if isinstance(t, dict)]
            customers = [t for t in threads if t.get("type") == "customer"]
            ours = [t for t in threads if t.get("type") in ("message", "reply", "forwardparent", "forwardchild")]
            return tags, {str(tag).lower() for tag in tags}, customers, ours

        classified = get_latest_customer_thread((payload.get("_embedded") or {}).get("threads") or [])
        first = await self.helpscout.get_conversation(ticket["id"])
        if is_supa_ignored(first):
            result.update(status="skipped", reason="supa-ignore: manual handling")
            return result
        _, named, customers, ours = read(first)

        if "wrong-address" in named:
            # Already answered. A thank-you reopens the conversation; close it again, send nothing,
            # and leave the audit row of the original send as it is.
            if first.get("status") == "active":
                async with _conversation_mutation_lease(
                    db_path=_get_audit_db_path(), conversation_id=ticket["id"], owner_scope="processor-triage",
                ) as lease:
                    again = await self.helpscout.get_conversation(ticket["id"])
                    if not is_supa_ignored(again) and "wrong-address" in read(again)[1]:
                        if not await lease.run_guarded(self.helpscout.close_conversation, ticket["id"]):
                            raise RuntimeError("HelpScout wrong-address close failed")
            result.update(
                classification="wrong_address",
                decision_note="Already answered with the wrong-address reply; nothing sent.",
            )
            return result

        def first_message_only(customers, ours):
            return (
                len(customers) == 1 and not ours and classified is not None
                and customers[0].get("id") is not None and customers[0].get("id") == classified.get("id")
            )

        if not first_message_only(customers, ours):
            return None  # A follow-up, or something we already wrote into: ordinary support.

        is_wrong, confidence, reason = await confirm_wrong_address(
            subject=ticket["subject"], message=ticket["message"]
        )
        if is_wrong is False:
            return None
        try:
            threshold = float(os.getenv("SUPA_WRONG_ADDRESS_MIN_CONFIDENCE", "") or 0.90)
        except ValueError:
            threshold = 0.90
        if not 0.5 <= threshold <= 1.0:
            threshold = 0.90
        live = support_agent.load_config().mode == "live"
        native = mailbox_provider.native_origin() is not None

        async with _conversation_mutation_lease(
            db_path=_get_audit_db_path(),
            conversation_id=ticket["id"],
            owner_scope="processor-triage",
        ) as lease:
            fresh = await self.helpscout.get_conversation(ticket["id"])
            if is_supa_ignored(fresh):
                result.update(status="skipped", reason="supa-ignore: manual handling")
                return result
            fresh_tags, named, customers, ours = read(fresh)
            blockers = [
                text for held, text in (
                    (is_wrong is None, "the second reading was unavailable"),
                    (is_wrong is not None and confidence < threshold,
                     f"second reading at {confidence:.2f}, below {threshold:.2f}"),
                    ("possible-wrong-address" not in named, "the mailbox intake check did not flag it"),
                    (not first_message_only(customers, ours), "the conversation changed while it was read"),
                    ("wrong-address" in named, "it was already answered"),
                    (fresh.get("status") not in (None, "active"), "the conversation is no longer active"),
                    (delivery_ceiling != "policy", "this run may not deliver"),
                    (not live, "support delivery is not live"),
                    (not native, "the owned mailbox is not selected"),
                ) if held
            ]
            if blockers:
                # Supa's own hold tags. The mailbox's `possible-wrong-address` is never written
                # here, so a hold cannot satisfy the intake gate on a later pass.
                tagged = await lease.run_guarded(
                    self.helpscout.add_tags, ticket["id"],
                    [*tags_to_add, "supa-wrong-address-held", "supa-triage-review"], fresh_tags,
                )
                if not tagged:
                    raise RuntimeError("HelpScout wrong-address hold tag update failed")
                result.update(
                    classification="wrong_address",
                    delivery_mode="wrong_address_held",
                    decision_note=("Wrong-address reply held: " + "; ".join(blockers) + ".")[:400],
                )
            else:
                sent = await lease.run_guarded(
                    self.helpscout.run_wrong_address_workflow_result, ticket["id"]
                )
                if not getattr(sent, "ok", False):
                    raise RuntimeError(
                        "Wrong-address workflow failed: " + str(getattr(sent, "detail", None) or "unknown")
                    )
                # The mailbox has delivered the reply. Record that before anything else can fail,
                # so a later error is a warning about a sent reply, never a silent retry.
                result.update(
                    classification="wrong_address",
                    response_posted=True,
                    delivery_mode="wrong_address_reply",
                    decision_note=f"Wrong-address reply sent (second reading {confidence:.2f})."[:400],
                )
                # The workflow changed the conversation; let the tag write read the current list.
                tagged = await lease.run_guarded(
                    self.helpscout.add_tags, ticket["id"], [*tags_to_add, "wrong-address"], None
                )
                if not tagged:
                    raise RuntimeError("Wrong-address reply sent, but the tag update failed")

        log_ticket(
            db_path=_get_audit_db_path(),
            helpscout_id=str(ticket["id"]),
            subject=ticket["subject"],
            customer_name=ticket["customer_name"],
            classification="wrong_address",
            status="processed",
            response_posted=bool(result.get("response_posted")),
            trigger=trigger,
            delivery_mode=result.get("delivery_mode"),
            decision_note=result.get("decision_note"),
        )
        return result

    async def process_ticket(
        self,
        payload: Dict[str, Any],
        trigger: str = "new_conversation",
        *,
        delivery_ceiling: str = "policy",
        recovery_guard: dict | None = None,
    ) -> Dict[str, Any]:
        """Run the support agent on one ticket: classify, investigate/draft, deliver.

        One run per conversation at a time: a concurrent duplicate (e.g. a repeated
        webhook) is skipped instead of redrafting over the in-flight run.

        Args:
            payload: HelpScout webhook payload
            trigger: What triggered processing - "new_conversation" or "note_trigger"
        """
        if recovery_guard is not None and delivery_ceiling != "draft":
            raise ValueError("Recovery must be draft-only")
        if delivery_ceiling not in {"draft", "policy"}:
            raise ValueError("Invalid delivery ceiling")
        with support_agent.claim_conversation(payload.get("id")) as claimed:
            if not claimed:
                return {
                    "ticket_id": payload.get("id"),
                    "status": "skipped",
                    "reason": "support agent already running for this conversation",
                    "handled_by": "support_agent_inflight",
                    "response_posted": False,
                }
            return await self._process_ticket(
                payload, trigger, delivery_ceiling=delivery_ceiling, recovery_guard=recovery_guard
            )

    async def _process_ticket(
        self,
        payload: Dict[str, Any],
        trigger: str,
        *,
        delivery_ceiling: str,
        recovery_guard: dict | None = None,
    ) -> Dict[str, Any]:
        ticket = parse_webhook_payload(payload)
        if recovery_guard:
            from services.support_recovery import human_note_versions
            recovery_guard = dict(recovery_guard, baseline_human_notes=human_note_versions(
                payload.get("_embedded", {}).get("threads", [])
            ))

        result = {
            "ticket_id": ticket["id"],
            "status": "processed",
            "classification": None,
            "response_posted": False,
            "delivery_mode": None,
            "confidence_score": None,
            "confidence_status": None,
            "confidence_attempts": None,
            "confidence_summary": None,
            "decision_note": None,
            "error": None,
        }
        if is_supa_ignored(payload):
            result.update(status="skipped", reason="supa-ignore: manual handling")
            return result

        existing_ticket_log = get_ticket_log_by_helpscout_id(
            db_path=_get_audit_db_path(),
            helpscout_id=str(ticket["id"]),
        )

        try:
            # Build tags to add
            tags_to_add = ["supa-processed"]

            # Add theme tag if detected and not already present
            if ticket["theme_tag"] and ticket["theme_tag"] not in ticket["tags"]:
                tags_to_add.append(ticket["theme_tag"])

            # For note triggers, skip classification - go straight to response generation
            if trigger == "note_trigger" or recovery_guard is not None:
                classification = "support"
                result["classification"] = classification
                if _note_trigger_is_stale_after_newer_send(
                    payload=payload,
                    trigger=trigger,
                    existing_row=existing_ticket_log,
                ):
                    return _restore_result_from_existing_draft(result, existing_ticket_log)
            else:
                # Step 1: Classify
                classification = await classify_ticket(
                    subject=ticket["subject"],
                    message=ticket["message"]
                )
                result["classification"] = classification

                # A person who meant to write to another business gets the owner's
                # wrong-address reply when every reading agrees; a doubtful one is drafted
                # like any support ticket.
                if classification == "wrong_address":
                    outcome = await self._handle_wrong_address(
                        ticket, tags_to_add, result,
                        payload=payload, delivery_ceiling=delivery_ceiling, trigger=trigger,
                    )
                    if outcome is not None:
                        return outcome
                    classification = "support"
                    result["classification"] = classification

                # Step 2: Handle non-support tickets (auto-close)
                if classification in ["admin", "marketing", "spam"]:
                    tags_to_add.append(classification)
                    async with _conversation_mutation_lease(
                        db_path=_get_audit_db_path(),
                        conversation_id=ticket["id"],
                        owner_scope="processor-triage",
                    ) as lease:
                        fresh_conversation = await self.helpscout.get_conversation(
                            ticket["id"]
                        )
                        if is_supa_ignored(fresh_conversation):
                            result.update(
                                status="skipped",
                                reason="supa-ignore: manual handling",
                            )
                            return result
                        raw_tags = fresh_conversation.get("tags")
                        fresh_tags = ticket["tags"]
                        if isinstance(raw_tags, list):
                            fresh_tags = [
                                tag.get("tag") if isinstance(tag, dict) else tag
                                for tag in raw_tags
                                if isinstance(tag, str)
                                or (
                                    isinstance(tag, dict)
                                    and isinstance(tag.get("tag"), str)
                                )
                            ]
                        closed = await lease.run_guarded(
                            self.helpscout.close_conversation,
                            ticket["id"],
                        )
                        if not closed:
                            raise RuntimeError("HelpScout non-support close failed")
                        tagged = await lease.run_guarded(
                            self.helpscout.add_tags,
                            ticket["id"],
                            tags_to_add,
                            fresh_tags,
                        )
                        if not tagged:
                            raise RuntimeError("HelpScout non-support tag update failed")

                    log_ticket(
                        db_path=_get_audit_db_path(),
                        helpscout_id=str(ticket["id"]),
                        subject=ticket["subject"],
                        customer_name=ticket["customer_name"],
                        classification=classification,
                        status="processed",
                        response_posted=False
                    )

                    return result

            # A durable admission binds delivery to one customer message. Re-running
            # a held message improves the draft but cannot reopen automatic delivery.
            input_customer = get_latest_customer_thread(payload.get("_embedded", {}).get("threads", []))
            message_attempt = await asyncio.to_thread(register_attempt, _get_audit_db_path(), ticket["id"], input_customer)

            # Step 2b: Generate better subject for searchability
            new_subject = await generate_subject(
                subject=ticket["subject"],
                message=ticket["message"],
                theme_tag=ticket["theme_tag"]
            )
            subject_changed = new_subject != ticket["subject"]
            if subject_changed:
                ticket["subject"] = new_subject  # Use new subject for logging

            # Step 3: Search knowledge base (sync) + customer data & past tickets (async, in parallel)
            customer_task = asyncio.create_task(
                fetch_customer_data(ticket.get("customer_email", ""))
            )
            related_task = asyncio.create_task(related_customer_context(
                helpscout=self.helpscout, customer_id=ticket.get("customer_id"),
                customer_email=ticket.get("customer_email", ""), exclude_id=ticket["id"]))
            # Step 3a: no theme in the ticket -> the customer's licences and earlier tickets.
            # Without a theme there is no repo to check and every code claim stays unbacked.
            # The ticket is not tagged with it: it is a hypothesis the reply must confirm.
            theme_hint = await theme_hint_for_ticket(
                ticket=ticket, helpscout=self.helpscout, customer_data_task=customer_task
            )
            # The hint opens the theme's repos and labels the prompts; it never narrows the KB
            # or past-ticket searches below, which a stale 2020 tag would point at the wrong theme.
            evidence_theme = ticket["theme_tag"] or (theme_hint.slug if theme_hint else "")
            similar_task = asyncio.create_task(
                find_similar_tickets(
                    helpscout_client=self.helpscout,
                    theme_tag=ticket["theme_tag"],
                    subject=ticket["subject"],
                    message=ticket["message"],
                    exclude_id=ticket["id"],
                    limit=2
                )
            )

            # KB search is sync, runs while async tasks are in flight
            knowledge_results = self.vector_store.search(
                query=f"{ticket['subject']} {ticket['message']}",
                theme_filter=ticket["theme_tag"] if ticket["theme_tag"] else None,
                limit=3
            )

            knowledge_context = "\n\n---\n\n".join([
                f"**{r.get('title', 'Document')}**\n{'URL: ' + r['url'] if r.get('url') else '(no URL available)'}\n{r['content'][:500]}..."
                for r in knowledge_results
            ]) if knowledge_results else ""

            customer_data, similar_tickets, related_context = await asyncio.gather(customer_task, similar_task, related_task)
            customer_context = format_customer_context(customer_data) if customer_data else ""
            customer_context = merge_context_sections(customer_context, related_context)
            if theme_hint:
                customer_context = merge_context_sections(theme_hint.context_block(), customer_context)
            past_tickets_context = format_past_tickets_context(similar_tickets)
            conversation_history_context = json.dumps(ticket.get("conversation_history", []))
            requirements_context = get_classic_theme_requirements_context(
                # Adds the (inferred) theme's requirements article; unlike the KB filter it hides nothing.
                theme_tag=evidence_theme,
                subject=ticket["subject"],
                message=ticket["message"],
                internal_note=ticket["internal_note"],
                conversation_history=conversation_history_context,
                docs_dir=KNOWLEDGE_DIR / "docs",
            )
            repo_context = get_repo_evidence_context(
                theme_tag=evidence_theme,
                subject=ticket["subject"],
                message=ticket["message"],
            )
            knowledge_context = merge_context_sections(requirements_context, knowledge_context)
            knowledge_context = merge_context_sections(current_product_context(
                subject=ticket["subject"], message=ticket["message"],
                knowledge_dir=KNOWLEDGE_DIR / "internal"), knowledge_context)
            combined_knowledge_context = merge_context_sections(knowledge_context, repo_context)

            # Step 3c: Web search fallback when internal sources don't have a clear answer
            # Triggers when: no strong KB match (< 0.65) AND no similar past tickets
            web_context = ""
            top_kb_score = knowledge_results[0]["score"] if knowledge_results else 0
            has_strong_kb_match = top_kb_score >= 0.65
            has_past_ticket_answer = len(similar_tickets) > 0

            if not has_strong_kb_match and not has_past_ticket_answer:
                search_query = build_search_query(
                    subject=ticket["subject"],
                    message=ticket["message"],
                    theme_tag=ticket["theme_tag"]
                )
                web_results = await search_web(search_query, limit=3)
                web_context = format_web_results(web_results)
                if web_context:
                    knowledge_context = f"{knowledge_context}\n\n{web_context}" if knowledge_context else web_context
                    combined_knowledge_context = merge_context_sections(knowledge_context, repo_context)

            semantic_failure_matches = get_recent_semantic_failures(
                db_path=_get_audit_db_path(),
                theme_tag=ticket["theme_tag"] or None,
                limit=20,
            )

            # Step 4: Run the support pipeline
            first_name = ticket["customer_name"].split()[0] if ticket["customer_name"] else "there"
            support_context = SupportPipelineContext(
                customer_name=first_name,
                subject=ticket["subject"],
                message=ticket["message"],
                theme_tag=evidence_theme,
                theme_source=theme_hint.source if theme_hint and not ticket["theme_tag"] else "",
                knowledge_context=combined_knowledge_context,
                past_tickets_context=past_tickets_context,
                customer_context=customer_context,
                internal_note=ticket["internal_note"],
                conversation_history=conversation_history_context,
                semantic_failure_matches=semantic_failure_matches,
                attachments=ticket.get("attachments") or [],
                duplicate_conversation_ids=duplicate_question_ids(ticket["message"], related_context, conversation_id=ticket["id"]),
            )
            evaluator_provider = _get_support_provider_env("SUPA_SUPPORT_EVALUATOR", "claude")
            autosend_threshold = _get_support_autosend_threshold()
            agent_config = support_agent.load_config()

            # Claude quota/login/timeout failures are remembered briefly: score with Codex
            # instead of repeating the failed call. The drafting step handles its own
            # Opus -> Sonnet -> Codex fallback inside the support agent.
            if should_skip_claude() and evaluator_provider == "claude":
                evaluator_provider = "codex"

            deep_evidence = deep_evidence_collector(processor=self, ticket=ticket)
            async with support_agent.drafting_capacity(agent_config):
                pipeline_result = await asyncio.to_thread(
                    run_support_pipeline,
                    context=support_context,
                    evaluator_provider=evaluator_provider,
                    target_score=autosend_threshold,
                    investigator_timeout=agent_config.timeout,
                    deep_evidence=deep_evidence,
                )
            if pipeline_result.delivery_mode == "auto_send":
                # Ceiling and mode first; the customer-waiting rule is checked on the
                # fresh conversation inside the delivery lease below.
                hold_line = support_agent.delivery_hold_line(
                    delivery_ceiling=delivery_ceiling,
                    mode=agent_config.mode,
                    customer_is_waiting=True,
                )
                if hold_line:
                    support_agent.downgrade_to_draft(pipeline_result, hold_line)
            if message_attempt != 1:
                support_agent.downgrade_to_draft(
                    pipeline_result,
                    "Mode: draft (rerun requires review; earlier holds cannot be cleared by another model run)"
                    if message_attempt else "Mode: draft (customer message identity is unknown)",
                )
                hold = "rerun_requires_review" if message_attempt else "customer_identity_unknown"
                if hold not in pipeline_result.hard_blockers:
                    pipeline_result.hard_blockers.append(hold)
                pipeline_result.manual_review = True
            pipeline_result.trace.append({"customer_thread_id": (input_customer or {}).get("id"), "message_attempt": message_attempt,
                "hard_blockers": list(pipeline_result.hard_blockers), "manual_review": pipeline_result.manual_review})
            response_html = finalize_response_html(
                pipeline_result.reply_html or "",
                combined_knowledge_context,
                past_tickets_context,
                allowed_urls=extract_context_urls(
                    combined_knowledge_context,
                    past_tickets_context,
                    extra_urls=pipeline_result.investigator_result.sources,
                ),
            )
            if not response_html and pipeline_result.delivery_mode == "draft":
                response_html = _manual_review_placeholder_html()
            if _has_unsafe_visible_draft_blocker(pipeline_result.hard_blockers):
                response_html = _manual_review_placeholder_html()
            result["delivery_mode"] = pipeline_result.delivery_mode
            result["confidence_score"] = pipeline_result.confidence_score
            result["confidence_status"] = pipeline_result.confidence_verdict
            result["confidence_attempts"] = pipeline_result.attempts
            result["confidence_summary"] = pipeline_result.evaluator_result.summary
            result["decision_note"] = pipeline_result.decision_note
            result["hard_blockers"] = list(pipeline_result.hard_blockers)
            result["risk_flags"] = list(pipeline_result.risk_flags)
            result["recovery_reason"] = (
                "evaluator_failure" if "evaluator_failure" in pipeline_result.risk_flags else
                "missing_evidence" if {"missing_critical_evidence", "missing_repo_evidence", "unread_attachments"} & set(pipeline_result.hard_blockers) else None
            )

            async with _conversation_mutation_lease(
                db_path=_get_audit_db_path(),
                conversation_id=ticket["id"],
                owner_scope="processor-delivery",
            ) as lease:
                fetch = self.helpscout.get_complete_conversation if recovery_guard else self.helpscout.get_conversation
                fresh_conversation = await fetch(ticket["id"])
                if recovery_guard is not None:
                    from services.support_recovery import guard_matches
                    if not guard_matches(fresh_conversation, existing_ticket_log, recovery_guard):
                        result.update(status="recovery_held", reason="draft_or_obligation_changed")
                        return result
                if is_supa_ignored(fresh_conversation):
                    result.update(
                        status="skipped",
                        reason="supa-ignore: manual handling",
                    )
                    return result
                raw_tags = fresh_conversation.get("tags")
                fresh_tags = ticket["tags"]
                if isinstance(raw_tags, list):
                    fresh_tags = [
                        tag.get("tag") if isinstance(tag, dict) else tag
                        for tag in raw_tags
                        if isinstance(tag, str)
                        or (
                            isinstance(tag, dict)
                            and isinstance(tag.get("tag"), str)
                        )
                    ]
                fresh_customer = get_latest_customer_thread(fresh_conversation.get("_embedded", {}).get("threads", []))
                if str((fresh_customer or {}).get("id")) != str((input_customer or {}).get("id")):
                    support_agent.downgrade_to_draft(pipeline_result, "Mode: draft (customer message changed during investigation)")
                    result["delivery_mode"] = pipeline_result.delivery_mode
                    result["decision_note"] = pipeline_result.decision_note
                    result.update(status="skipped", reason="customer_message_changed", response_posted=False)
                    return result
                if pipeline_result.delivery_mode == "auto_send" and not support_agent.customer_waiting(
                    fresh_conversation.get("_embedded", {}).get("threads", []) or []
                ):
                    # Never auto-send on top of a reply the customer has not answered yet.
                    support_agent.downgrade_to_draft(
                        pipeline_result,
                        support_agent.delivery_hold_line(
                            delivery_ceiling=delivery_ceiling,
                            mode=agent_config.mode,
                            customer_is_waiting=False,
                        ),
                    )
                    result["delivery_mode"] = pipeline_result.delivery_mode
                    result["decision_note"] = pipeline_result.decision_note
                if subject_changed:
                    await lease.run_guarded(
                        self.helpscout.update_subject,
                        ticket["id"],
                        new_subject,
                    )

                if _should_preserve_existing_draft(
                    trigger=trigger,
                    existing_row=existing_ticket_log,
                    response_html=response_html,
                    confidence_score=pipeline_result.confidence_score,
                    risk_flags=pipeline_result.risk_flags,
                    hard_blockers=pipeline_result.hard_blockers,
                ):
                    if recovery_guard:
                        result.update(status="recovery_held", reason="no_improved_draft", response_posted=False)
                        return result
                    preserve_draft_ids = get_draft_thread_ids(
                        fresh_conversation.get("_embedded", {}).get("threads", [])
                    )
                    if preserve_draft_ids:
                        kept_id = preserve_draft_ids[-1]
                        stragglers = [tid for tid in preserve_draft_ids if tid != kept_id]
                        await self._purge_prior_drafts(
                            ticket["id"], stragglers, owned_lease=lease
                        )
                    return _restore_result_from_existing_draft(result, existing_ticket_log)

                if response_html:
                    if pipeline_result.manual_review and "supa-manual-review" not in tags_to_add:
                        tags_to_add.append("supa-manual-review")

                    if pipeline_result.delivery_mode == "auto_send":
                        auto_send_result = await process_auto_send(
                            helpscout=self.helpscout,
                            conversation_id=ticket["id"],
                            customer_id=ticket.get("customer_id"),
                            response_html=response_html,
                            tags_to_add=tags_to_add,
                            existing_tags=fresh_tags,
                            confidence_score=pipeline_result.confidence_score,
                            confidence_provider=pipeline_result.evaluator_provider,
                            db_path=_get_audit_db_path(),
                            owned_lease=lease,
                            expected_customer_thread_id=(input_customer or {}).get("id"),
                        )
                        pipeline_result.decision_note = _append_delayed_autosend_schedule(
                            pipeline_result.decision_note,
                            auto_send_result.get("due_at"),
                        )
                        result["decision_note"] = pipeline_result.decision_note
                    else:
                        # Only delete a draft owned by this service.
                        prior_draft_thread_ids = get_supa_owned_draft_thread_ids(
                            fresh_conversation.get("_embedded", {}).get("threads", [])
                        )
                        saved = await lease.run_guarded(
                            self.helpscout.add_draft,
                            ticket["id"],
                            response_html,
                            ticket.get("customer_id"),
                        )
                        if not saved:
                            raise RuntimeError("HelpScout draft save failed")
                        cleanup_kwargs = {}
                        if recovery_guard:
                            from services.conversation_gateway import draft_version
                            cleanup_kwargs["expected_versions"] = {
                                thread["id"]: draft_version(thread)
                                for thread in fresh_conversation.get("_embedded", {}).get("threads", [])
                                if thread.get("id") in prior_draft_thread_ids
                            }
                        await self._purge_prior_drafts(
                            ticket["id"], prior_draft_thread_ids, owned_lease=lease,
                            **cleanup_kwargs,
                        )

                    result["response_posted"] = True

                if pipeline_result.decision_note:
                    try:
                        record = decision_record(pipeline_result, pipeline_result.decision_note)
                    except Exception:  # The note is the same decision in prose; never lose it to a formatting fault.
                        logger.exception("Decision record could not be built for %s", ticket["id"])
                        record = None
                    note_added = await lease.run_guarded(
                        self.helpscout.add_decision,
                        ticket["id"],
                        pipeline_result.decision_note,
                        record,
                    )
                    if not note_added:
                        raise RuntimeError("HelpScout note write failed")

                tags_added = await lease.run_guarded(
                    self.helpscout.add_tags,
                    ticket["id"],
                    tags_to_add,
                    fresh_tags,
                )
                if not tags_added:
                    raise RuntimeError("HelpScout tag update failed")
                if recovery_guard:
                    from services.support_recovery import verified_replacement
                    readback = await self.helpscout.get_complete_conversation(ticket["id"])
                    receipt = verified_replacement(readback, recovery_guard, response_html)
                    if receipt is None:
                        raise RuntimeError("Recovery draft readback failed")
                    result["recovery_receipt"] = receipt
                    result["status"] = "recovery_needs_evidence" if result["recovery_reason"] else (
                        "draft_ready" if not pipeline_result.hard_blockers and not pipeline_result.manual_review
                        and pipeline_result.confidence_score >= autosend_threshold else "recovery_human_review"
                    )

            log_ticket(
                db_path=_get_audit_db_path(),
                helpscout_id=str(ticket["id"]),
                subject=ticket["subject"],
                customer_name=ticket["customer_name"],
                classification=classification,
                status=result["status"],
                response_posted=result["response_posted"],
                trigger=trigger,
                confidence_score=pipeline_result.confidence_score,
                confidence_status=pipeline_result.confidence_verdict,
                confidence_attempts=pipeline_result.attempts,
                confidence_summary=pipeline_result.evaluator_result.summary,
                confidence_trace=json.dumps(pipeline_result.trace),
                confidence_provider=pipeline_result.evaluator_provider,
                draft_response_html=response_html,
                evidence_summary=pipeline_result.evidence_summary,
                delivery_mode=pipeline_result.delivery_mode,
                investigator_provider=pipeline_result.investigator_provider,
                evaluator_provider=pipeline_result.evaluator_provider,
                decision_note=pipeline_result.decision_note,
                investigation_steps=pipeline_result.investigation_steps,
                risk_flags=pipeline_result.risk_flags,
                sources=pipeline_result.investigator_result.sources,
            )

        except Exception as e:
            result["status"] = "warning" if result.get("response_posted") else "error"
            result["error"] = str(e)

            log_ticket(
                db_path=_get_audit_db_path(),
                helpscout_id=str(ticket["id"]),
                subject=ticket.get("subject", ""),
                customer_name=ticket.get("customer_name", ""),
                classification=result.get("classification", ""),
                status=result["status"],
                response_posted=bool(result.get("response_posted")),
                error_message=str(e),
                trigger=trigger,
                confidence_score=result.get("confidence_score"),
                confidence_status=result.get("confidence_status"),
                confidence_attempts=result.get("confidence_attempts"),
                confidence_summary=result.get("confidence_summary"),
                decision_note=result.get("decision_note"),
                delivery_mode=result.get("delivery_mode"),
            )

        return result

    async def _purge_prior_drafts(
        self,
        conversation_id: int,
        prior_draft_thread_ids: list[int],
        *,
        owned_lease=None,
        expected_versions: dict | None = None,
    ) -> None:
        """Best-effort delete of stacked drafts. Failures are logged, not raised."""
        if not prior_draft_thread_ids:
            return

        purged = 0
        async with _conversation_mutation_lease(
            db_path=_get_audit_db_path(),
            conversation_id=conversation_id,
            owner_scope="processor-draft-cleanup",
            owned_lease=owned_lease,
        ) as lease:
            for thread_id in prior_draft_thread_ids:
                try:
                    if expected_versions is not None:
                        from services.conversation_gateway import draft_version
                        latest = await self.helpscout.get_complete_conversation(conversation_id)
                        threads = latest.get("_embedded", {}).get("threads", [])
                        current = next((t for t in threads if t.get("id") == thread_id), None)
                        if (current is None or thread_id not in get_supa_owned_draft_thread_ids(threads)
                                or draft_version(current) != expected_versions.get(thread_id)):
                            continue
                    updated = await lease.run_guarded(
                        self.helpscout.update_draft,
                        conversation_id,
                        thread_id,
                        _manual_review_placeholder_html(),
                    )
                    if updated:
                        purged += 1
                except ConversationLockError:
                    raise
                except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                    logger.warning(
                        "update_draft failed for conversation=%s thread=%s: %s",
                        conversation_id,
                        thread_id,
                        exc,
                    )

        logger.info(
            "Neutralized %s of %s prior drafts for conversation %s",
            purged,
            len(prior_draft_thread_ids),
            conversation_id,
        )

    def _auto_index_if_needed(self):
        """Auto-index docs and knowledge if source files are newer than DB."""
        vectors_db = DB_DIR / "vectors.db"
        db_mtime = vectors_db.stat().st_mtime if vectors_db.exists() else 0

        # Check if any docs or knowledge files are newer
        needs_reindex = False

        # Check docs directory
        docs_dir = KNOWLEDGE_DIR / "docs"
        if docs_dir.exists():
            for f in docs_dir.glob("*.json"):
                if f.stat().st_mtime > db_mtime:
                    needs_reindex = True
                    break

        # Check examples directory
        examples_dir = KNOWLEDGE_DIR / "examples"
        if examples_dir.exists():
            for f in examples_dir.glob("*.md"):
                if f.stat().st_mtime > db_mtime:
                    needs_reindex = True
                    break

        if needs_reindex or self.vector_store.count() == 0:
            print("Auto-indexing knowledge base...")
            self.index_knowledge()
            self.index_docs()
            print(f"Indexed {self.vector_store.count()} documents total")

    def index_knowledge(self) -> int:
        """Index all knowledge files (examples and internal notes)."""
        docs = get_all_knowledge(KNOWLEDGE_DIR)
        # Clear only examples/internal sources, keep crawled docs
        self.vector_store.clear(source="examples")
        self.vector_store.clear(source="internal")
        added = self.vector_store.add_documents(docs)
        return added

    def index_docs(self) -> int:
        """Index documentation from pre-crawled JSON files."""
        import json

        docs_dir = KNOWLEDGE_DIR / "docs"
        if not docs_dir.exists():
            print(f"No docs directory found at {docs_dir}")
            return 0

        # Clear previously indexed docs
        self.vector_store.clear(source="docs")

        total_added = 0
        for json_file in sorted(docs_dir.glob("*.json")):
            with open(json_file) as f:
                docs = json.load(f)
            added = self.vector_store.add_documents(docs)
            print(f"  Indexed {added} docs from {json_file.name}")
            total_added += added

        print(f"Total: {total_added} docs indexed")
        return total_added

    def index_sent_reply(
        self,
        conversation_id: int,
        subject: str,
        customer_message: str,
        reply_text: str,
        theme_tag: str = None
    ) -> bool:
        """
        Index a sent reply for future learning.

        Stores the Q&A pair in the vector store so similar questions
        can reference this successful answer.
        """
        import re

        # Strip HTML tags from reply for cleaner indexing
        clean_reply = re.sub(r'<[^>]+>', ' ', reply_text)
        clean_reply = re.sub(r'\s+', ' ', clean_reply).strip()

        # Build searchable content combining question and answer
        content = f"""Question: {subject}

Customer message: {customer_message[:500]}

Successful reply:
{clean_reply[:1000]}"""

        doc = {
            "id": f"sent_reply_{conversation_id}",
            "title": f"Past answer: {subject[:100]}",
            "content": content,
            "theme": theme_tag or "all",
            "source": "sent_replies",
            "metadata": {
                "conversation_id": conversation_id,
                "type": "learned_answer"
            }
        }

        try:
            added = self.vector_store.add_documents([doc])
            print(f"Indexed sent reply for conversation {conversation_id}")
            return added > 0
        except Exception as e:
            print(f"Error indexing sent reply: {e}")
            return False

    async def crawl_docs(self, theme_slugs: list[str] = None) -> int:
        """Crawl and index documentation from the configured site."""
        if theme_slugs is None:
            theme_slugs = product_settings()["default_crawl_products"]

        # Clear previously crawled docs
        self.vector_store.clear(source="docs")

        total_added = 0
        for slug in theme_slugs:
            if slug not in VALID_SLUGS:
                print(f"Skipping invalid theme: {slug}")
                continue

            print(f"Crawling {slug}...")
            docs = await crawl_theme_docs(slug)
            added = self.vector_store.add_documents(docs)
            print(f"  Indexed {added} pages")
            total_added += added

        return total_added

    def get_knowledge_count(self) -> int:
        """Get count of indexed documents."""
        return self.vector_store.count()

    async def close(self):
        """Cleanup resources."""
        await self.helpscout.close()


# Global processor instance
_processor: Optional[TicketProcessor] = None


def get_processor() -> TicketProcessor:
    global _processor
    if _processor is None:
        _processor = TicketProcessor()
    return _processor
