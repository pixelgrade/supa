import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from pydantic import BaseModel

from services import extensions, mailbox_provider, native_mailbox_events
from db.database import get_ticket_log_by_helpscout_id
from services.agent_events import (
    AgentEventStorageError,
    append_event,
    normalize_webhook_event,
)
from services.conversation_locks import conversation_lease
from services.conversation_policy import manual_route
from services.helpscout import (
    get_latest_note_thread,
    is_supa_ignored,
    parse_webhook_payload,
)
from services.processor import get_processor
from services.runtime_paths import get_supa_db_path
from services.semantic_failure_learning import learn_from_sent_reply

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhook", tags=["webhooks"])
DEFAULT_DB_PATH = get_supa_db_path()
METADATA_ONLY_EVENTS = frozenset({"convo.assigned", "convo.status", "convo.tags"})


NATIVE_EVENTS = METADATA_ONLY_EVENTS | {
    "convo.created", "convo.note.created", "convo.customer.reply.created", "convo.agent.reply.created",
}


class WebhookResponse(BaseModel):
    status: str
    reason: Optional[str] = None
    ticket_id: Optional[int] = None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _parse_db_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _already_processed_after_timestamp(ticket_id: int, event_created_at: str | None) -> bool:
    event_timestamp = _parse_timestamp(event_created_at)
    if event_timestamp is None:
        return False

    row = get_ticket_log_by_helpscout_id(DEFAULT_DB_PATH, str(ticket_id))
    if not row:
        return False

    status = (row.get("status") or "").lower()
    if status not in {"processed", "warning"}:
        return False

    processed_at = _parse_db_timestamp(row.get("created_at"))
    if processed_at is None:
        return False

    return processed_at >= event_timestamp


async def process_in_background(payload: Dict[str, Any], trigger: str = "new_conversation"):
    """Background task to process ticket."""
    processor = get_processor()
    result = await processor.process_ticket(payload, trigger=trigger)
    logger.info(
        "Background processing finished: "
        f"ticket={result.get('ticket_id')} "
        f"trigger={trigger} "
        f"status={result.get('status', 'processed')} "
        f"classification={result.get('classification')} "
        f"delivery_mode={result.get('delivery_mode')} "
        f"response_posted={result.get('response_posted')} "
        f"confidence_status={result.get('confidence_status')} "
        f"error={result.get('error')}"
    )
    if mailbox_provider.native_origin() and result.get("status") in {"warning", "error"}:
        raise RuntimeError("Support processing requires readback")


@router.post("/helpscout", response_model=WebhookResponse)
async def helpscout_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_helpscout_event: Optional[str] = Header(None),
    x_helpscout_signature: Optional[str] = Header(None),
):
    """Receive HelpScout webhook events.

    Handles:
    - convo.created: New conversation
    - convo.customer.reply.created: Customer replied (removes supa-processed tag)
    - convo.note.created: Note added (manual trigger)
    - convo.agent.reply.created: Agent sent reply (learning)
    """

    raw_body = await request.body()
    native = bool(mailbox_provider.native_origin())
    key = request.headers.get("x-inbox-delivery")
    if native and key is None:
        # Fence delayed vendor callbacks after moving the one live writer.
        return WebhookResponse(status="skipped", reason="Previous mailbox provider")
    if key is not None and not native:
        raise HTTPException(409, "Native mailbox is not selected")
    webhook_secret = os.getenv("SUPA_MAILBOX_WEBHOOK_SECRET" if native else "HELPSCOUT_WEBHOOK_SECRET", "")
    if native:
        if not webhook_secret or not re.fullmatch(r"[1-9][0-9]{0,17}", key or ""):
            raise HTTPException(401, "Invalid native delivery")
        expected = native_mailbox_events.signature(webhook_secret, key, x_helpscout_event or "", raw_body)
        supplied = request.headers.get("x-inbox-signature", "")
        if not re.fullmatch(r"[0-9a-f]{64}", supplied) or not hmac.compare_digest(expected, supplied):
            raise HTTPException(401, "Invalid native delivery signature")
    if webhook_secret:
        expected_signature = base64.b64encode(
            hmac.new(
                webhook_secret.encode("utf-8"),
                raw_body,
                hashlib.sha1,
            ).digest()
        ).decode("ascii")
        supplied_signature = x_helpscout_signature or ""
        try:
            signature_valid = hmac.compare_digest(
                expected_signature.encode("ascii"),
                supplied_signature.encode("ascii"),
            )
        except UnicodeEncodeError:
            signature_valid = False
        if not signature_valid:
            raise HTTPException(
                status_code=401,
                detail="The Help Scout webhook signature is invalid.",
            )

    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError):
        raise HTTPException(
            status_code=400,
            detail="The Help Scout webhook payload is invalid.",
        ) from None
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="The Help Scout webhook payload is invalid.",
        )

    event_type = x_helpscout_event or ""
    ticket_id = payload.get("id", "unknown")

    if native and (event_type not in NATIVE_EVENTS or not payload.get("id")):
        raise HTTPException(400, "Invalid native conversation event")

    # A conversation ID is required both by legacy routing and by the durable
    # journal. Preserve the existing terminal response for an absent ID.
    if not payload.get("id"):
        logger.warning("Webhook payload missing ticket ID, skipping")
        return WebhookResponse(status="skipped", reason="No ticket ID in payload")

    try:
        event = normalize_webhook_event(event_type, payload)
        if native:
            state = native_mailbox_events.accept(
                get_supa_db_path(), int(key), event_type, int(event.conversation_id), raw_body, event
            )
            return WebhookResponse(status=state, ticket_id=int(event.conversation_id))
        append_event(DEFAULT_DB_PATH, event)
    except native_mailbox_events.DeliveryConflict:
        raise HTTPException(409, "Conflicting native delivery") from None
    except (sqlite3.Error, AgentEventStorageError):
        logger.error("Help Scout webhook journal is unavailable")
        raise HTTPException(
            status_code=503,
            detail="The event journal is unavailable.",
        ) from None
    except (RecursionError, TypeError, UnicodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The Help Scout webhook payload is invalid.",
        ) from None

    ticket_id = int(event.conversation_id)
    return await route_mailbox_event(event_type, ticket_id, payload, background_tasks)


async def process_native_delivery(path):
    row = native_mailbox_events.claim(path)
    if row is None:
        return False
    try:
        tasks = BackgroundTasks()
        await route_mailbox_event(row["event"], row["conversation"], json.loads(row["payload"]), tasks)
        await tasks()
    except Exception as exc:
        native_mailbox_events.finish(path, row["id"], "review", type(exc).__name__)
        logger.error("Native mailbox delivery %s needs reconciliation", row["id"])
    else:
        native_mailbox_events.finish(path, row["id"], "done")
    return True


async def native_worker(path):
    recover = True
    while True:
        try:
            if recover:
                native_mailbox_events.recover(path)
                recover = False
            if not await process_native_delivery(path):
                await asyncio.sleep(1)
        except sqlite3.OperationalError:
            # A failed completion write may follow an external action. Hold it before claiming more.
            recover = True
            logger.warning("Native mailbox ledger unavailable, retrying")
            await asyncio.sleep(5)


async def route_mailbox_event(event_type, ticket_id, payload, background_tasks):
    logger.info("Webhook received: event=%s, ticket=%s", event_type, ticket_id)

    # Status/tag/assignment hooks can be caused by Supa itself. They remain
    # durable signals but must never recursively start a support workflow.
    if event_type.strip().lower() in METADATA_ONLY_EVENTS:
        return WebhookResponse(
            status="journaled",
            reason="Metadata event",
            ticket_id=ticket_id,
        )

    route, case_id = manual_route(payload, ignored_check=is_supa_ignored)
    if route == "supa_ignore":
        return WebhookResponse(
            status="skipped",
            reason="supa-ignore: manual handling",
            ticket_id=ticket_id,
        )

    response = extensions.intercept_webhook(event_type, ticket_id, payload, background_tasks, route, case_id)
    if response is not None:
        return response

    # Handle note-added event (manual trigger for draft generation)
    if event_type == "convo.note.created":
        logger.info(f"Processing note trigger for ticket {ticket_id}")
        return await handle_note_added(payload, background_tasks)

    # Handle customer reply - remove supa-processed tag so it gets re-processed
    if event_type == "convo.customer.reply.created":
        logger.info(f"Processing customer reply for ticket {ticket_id}")
        return await handle_customer_reply(payload, background_tasks)

    # Handle agent reply - index for learning
    if event_type == "convo.agent.reply.created":
        logger.info(f"Processing agent reply for learning: ticket {ticket_id}")
        return await handle_agent_reply(payload)

    # Handle new conversation
    logger.info(f"Processing new conversation for ticket {ticket_id}")
    return await handle_new_conversation(payload, background_tasks)


async def handle_customer_reply(payload: Dict[str, Any], background_tasks: BackgroundTasks) -> WebhookResponse:
    """Handle customer reply - remove supa-processed tag and process."""

    ticket = parse_webhook_payload(payload)

    # Remove supa-processed tag if present (customer replied, needs new processing)
    if "supa-processed" in ticket["tags"]:
        processor = get_processor()
        owner = f"webhook-customer-reply:{ticket['id']}:{uuid.uuid4().hex}"
        async with conversation_lease(DEFAULT_DB_PATH, ticket["id"], owner=owner) as lease:
            current_tags = list(ticket["tags"])
            get_conversation = getattr(processor.helpscout, "get_conversation", None)
            if callable(get_conversation):
                current = await get_conversation(ticket["id"])
                raw_tags = current.get("tags") if isinstance(current, dict) else None
                if isinstance(raw_tags, list):
                    current_tags = [
                        tag.get("tag") if isinstance(tag, dict) else tag
                        for tag in raw_tags
                        if isinstance(tag, str)
                        or (
                            isinstance(tag, dict)
                            and isinstance(tag.get("tag"), str)
                        )
                    ]
            if "supa-processed" in current_tags:
                await lease.run_guarded(
                    processor.helpscout.remove_tag,
                    ticket["id"],
                    "supa-processed",
                    current_tags,
                )

    # Skip if no message content
    if not ticket["message"]:
        return WebhookResponse(
            status="skipped",
            reason="No customer message found",
            ticket_id=ticket["id"]
        )

    # Process in background
    background_tasks.add_task(process_in_background, payload, "customer_reply")

    return WebhookResponse(
        status="queued",
        reason="Customer replied",
        ticket_id=ticket["id"]
    )


async def handle_note_added(payload: Dict[str, Any], background_tasks: BackgroundTasks) -> WebhookResponse:
    """Handle note-added webhook - triggers draft generation."""

    ticket = parse_webhook_payload(payload)
    threads = payload.get("_embedded", {}).get("threads", [])

    logger.info(f"Note webhook: ticket={ticket['id']}, threads={len(threads)}")

    # Find the note in threads (may not be at index 0)
    recent_note = get_latest_note_thread(threads)

    if not recent_note:
        logger.warning(f"No note found in payload for ticket {ticket['id']}, thread types: {[t.get('type') for t in threads]}")
        return WebhookResponse(
            status="skipped",
            reason="No note found in payload",
            ticket_id=ticket["id"]
        )

    # Skip if note was created by API (our own notes) to avoid loops
    note_source = recent_note.get("source", {}).get("type", "")
    logger.info(f"Note source: {note_source}")

    if note_source == "api":
        logger.info(f"Skipping API note for ticket {ticket['id']}")
        return WebhookResponse(
            status="skipped",
            reason="Note from API - avoiding loop",
            ticket_id=ticket["id"]
        )

    if _already_processed_after_timestamp(ticket["id"], recent_note.get("createdAt")):
        logger.info(f"Skipping duplicate note trigger for ticket {ticket['id']}")
        return WebhookResponse(
            status="skipped",
            reason="Note already processed",
            ticket_id=ticket["id"],
        )

    # Note: We intentionally ignore supa-processed tag here
    # Manual note trigger should always work (for follow-up responses)

    logger.info(f"Queueing note trigger processing for ticket {ticket['id']}")
    # Process in background - note content becomes internal_note for instructions
    background_tasks.add_task(process_in_background, payload, "note_trigger")

    return WebhookResponse(
        status="queued",
        reason="Triggered by note",
        ticket_id=ticket["id"]
    )


def agent_reply_and_customer_message(threads: list) -> tuple:
    """The newest sent agent reply and customer message in a webhook payload (newest first).

    Unsent drafts and blanked leftovers are skipped: learning from them would index a reply
    the customer never received (2026-10-02).
    """
    agent_reply = None
    customer_message = None
    for thread in threads:
        if thread.get("state") == "draft" or thread.get("draft") is True:
            continue
        if thread.get("state") not in (None, "published"):
            continue
        thread_type = thread.get("type", "")
        created_by = (thread.get("createdBy") or {}).get("type", "")
        if thread_type == "message" and created_by == "user":
            if not agent_reply:
                agent_reply = thread.get("body", "")
        elif thread_type == "customer" or (thread_type == "message" and created_by == "customer"):
            if not customer_message:
                customer_message = thread.get("body", "")
    return agent_reply, customer_message


async def handle_agent_reply(payload: Dict[str, Any]) -> WebhookResponse:
    """Handle agent reply webhook - index for learning.

    This captures ALL replies sent through HelpScout (not just via Supa dashboard),
    allowing the system to learn from every response.
    """
    ticket = parse_webhook_payload(payload)
    threads = payload.get("_embedded", {}).get("threads", [])

    agent_reply, customer_message = agent_reply_and_customer_message(threads)

    # A reply a mailbox workflow sent is fixed wording (for example the wrong-address reply, or
    # the mailbox's one-time note to a site owner), not an answer to learn from.
    sent = [
        t for t in threads
        if isinstance(t, dict) and t.get("type") == "message" and t.get("state") in (None, "published")
    ]
    newest = max(sent, key=lambda t: str(t.get("createdAt") or ""), default=None)
    tags = {str(tag).lower() for tag in ticket.get("tags") or []}
    if tags & {"wrong-address", "non-support", "site-mail-notice"} or (
        newest and (newest.get("source") or {}).get("type") in ("manual-workflow", "workflows")
    ):
        return WebhookResponse(
            status="skipped",
            reason="Workflow reply, not a learned answer",
            ticket_id=ticket["id"],
        )

    if not agent_reply:
        logger.info(f"No agent reply found in payload for ticket {ticket['id']}")
        return WebhookResponse(
            status="skipped",
            reason="No agent reply found",
            ticket_id=ticket["id"]
        )

    # Get theme tag if available
    theme_tag = ticket.get("theme_tag", "")

    # Index for learning
    processor = get_processor()
    processor.index_sent_reply(
        conversation_id=ticket["id"],
        subject=ticket.get("subject", ""),
        customer_message=customer_message or ticket.get("message", ""),
        reply_text=agent_reply,
        theme_tag=theme_tag
    )

    await learn_from_sent_reply(
        conversation_id=ticket["id"],
        subject=ticket.get("subject", ""),
        customer_message=customer_message or ticket.get("message", ""),
        final_reply_html=agent_reply,
        theme_tag=theme_tag,
        helpscout_client=processor.helpscout,
    )

    logger.info(f"Indexed agent reply for learning: ticket {ticket['id']}")

    return WebhookResponse(
        status="learned",
        reason="Indexed agent reply for future reference",
        ticket_id=ticket["id"]
    )


async def handle_new_conversation(payload: Dict[str, Any], background_tasks: BackgroundTasks) -> WebhookResponse:
    """Handle new conversation or customer reply webhook."""

    ticket = parse_webhook_payload(payload)

    # Skip if no message content
    if not ticket["message"]:
        return WebhookResponse(
            status="skipped",
            reason="No customer message found",
            ticket_id=ticket["id"]
        )

    # Process in background
    background_tasks.add_task(process_in_background, payload, "new_conversation")

    return WebhookResponse(
        status="queued",
        ticket_id=ticket["id"]
    )
