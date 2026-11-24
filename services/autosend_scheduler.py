"""Background scheduler for humanize-autosend-timing.

Auto-sends are queued as HelpScout drafts + pending_sends rows; this module
owns the delay config, change-detection logic, and worker loop that promotes
or cancels due rows.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from services.helpscout import _is_draft_thread, _is_manual_review_placeholder, _is_supa_system_note, is_supa_ignored
from services.conversation_locks import ConversationLockError, conversation_lease
from services.runtime_paths import get_supa_db_path
from services.email_html import email_html_equivalent

logger = logging.getLogger(__name__)

DEFAULT_DELAY_MIN_SECONDS = 300
DEFAULT_DELAY_MAX_SECONDS = 1200
DEFAULT_TICK_SECONDS = 30
SKEW_GRACE_SECONDS = 2


@dataclass(frozen=True)
class AutoSendConfig:
    delay_min_seconds: int
    delay_max_seconds: int
    tick_seconds: int
    enabled: bool


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def load_config() -> AutoSendConfig:
    return AutoSendConfig(
        delay_min_seconds=_env_int("SUPA_AUTOSEND_DELAY_MIN_SECONDS", DEFAULT_DELAY_MIN_SECONDS),
        delay_max_seconds=_env_int("SUPA_AUTOSEND_DELAY_MAX_SECONDS", DEFAULT_DELAY_MAX_SECONDS),
        tick_seconds=_env_int("SUPA_AUTOSEND_WORKER_TICK_SECONDS", DEFAULT_TICK_SECONDS),
        enabled=_env_bool("SUPA_AUTOSEND_DELAY_ENABLED", True),
    )


def pick_due_at(
    *,
    now: datetime,
    config: AutoSendConfig,
    rng: Optional[random.Random] = None,
) -> str:
    """Pick a uniformly-random due_at ISO string in [now+min, now+max]."""
    picker = rng or random
    delta = picker.randint(config.delay_min_seconds, config.delay_max_seconds)
    return (now + timedelta(seconds=delta)).isoformat()


def sha256_of_html(html: str) -> str:
    """Stable hash of reply HTML for change-detection."""
    return hashlib.sha256((html or "").encode("utf-8")).hexdigest()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _is_agent_message_thread(thread: dict) -> bool:
    if thread.get("type") != "message":
        return False
    if _is_draft_thread(thread):
        return False
    created_by_type = (thread.get("createdBy") or {}).get("type")
    return created_by_type in ("user", "team")


def _is_note_thread(thread: dict) -> bool:
    if thread.get("type") != "note":
        return False
    # Supa's own decision/auto-improvement notes are added right after queueing
    # and would otherwise self-cancel the pending send as "note_added".
    return not _is_supa_system_note(thread.get("body", ""))


def _normalize_tag_names(tags: Iterable[Any]) -> list[str]:
    normalized: list[str] = []
    for tag in tags or []:
        if isinstance(tag, dict):
            value = tag.get("tag")
        else:
            value = tag
        if isinstance(value, str) and value:
            normalized.append(value)
    return normalized


def evaluate_cancel_reason(
    *,
    row: dict,
    threads: Iterable[dict],
) -> Optional[str]:
    """Return the first cancel reason that fires for this row, or None if clean.

    Precedence: customer_replied -> operator_sent -> note_added -> draft_deleted
    -> draft_modified -> extra_draft.
    """
    threads = list(threads or [])
    queued_at = _parse_iso(row.get("queued_at"))
    if queued_at is not None:
        queued_at = queued_at - timedelta(seconds=SKEW_GRACE_SECONDS)

    draft_thread_id = str(row.get("draft_thread_id"))
    response_hash = row.get("response_html_sha")
    last_known = row.get("last_known_customer_thread_id")

    for t in threads:
        if t.get("type") != "customer":
            continue
        t_id = str(t.get("id"))
        if last_known is not None and t_id == str(last_known):
            continue
        created = _parse_iso(t.get("createdAt"))
        if queued_at is None or (created is not None and created > queued_at):
            return "customer_replied"

    for t in threads:
        if not _is_agent_message_thread(t):
            continue
        created = _parse_iso(t.get("createdAt"))
        if queued_at is not None and created is not None and created > queued_at:
            return "operator_sent"

    for t in threads:
        if not _is_note_thread(t):
            continue
        created = _parse_iso(t.get("createdAt"))
        if queued_at is not None and created is not None and created > queued_at:
            return "note_added"

    matching_draft = None
    for t in threads:
        if _is_draft_thread(t) and str(t.get("id")) == draft_thread_id:
            matching_draft = t
            break
    if matching_draft is None:
        return "draft_deleted"

    if sha256_of_html(matching_draft.get("body") or "") != response_hash:
        return "draft_modified"

    for t in threads:
        if not _is_draft_thread(t):
            continue
        if str(t.get("id")) == draft_thread_id:
            continue
        created = _parse_iso(t.get("createdAt"))
        if queued_at is not None and created is not None and created > queued_at:
            return "extra_draft"

    return None


MAX_RETRY_ATTEMPTS = 5
BASE_RETRY_SECONDS = 60
MAX_RETRY_SECONDS = 600


def _retry_delay_seconds(attempts: int) -> int:
    """60s * 2^attempts, capped at 10 min."""
    return min(BASE_RETRY_SECONDS * (2 ** attempts), MAX_RETRY_SECONDS)


def _find_sent_thread_with_hash(threads: list, response_hash: str) -> Optional[dict]:
    """Look for an API-sourced agent message thread whose body hash matches.

    Requires source.type == 'api' so we don't mistake a coincidental manual
    send with the same content for our own promote.
    """
    for t in threads:
        if not _is_agent_message_thread(t):
            continue
        if (t.get("source") or {}).get("type") != "api":
            continue
        if sha256_of_html(t.get("body") or "") == response_hash:
            return t
    return None


def _verified_delivery_thread(*, threads: list[dict], row: dict) -> Optional[dict]:
    """A receipt needs one explicitly published API reply after this queue entry.

    A matching old closing, draft, hidden message or ambiguous duplicate is not proof
    of this delivery. Missing metadata remains an unverified send rather than guessed.
    """
    queued_at = _parse_iso(row.get("queued_at"))
    if queued_at is None or queued_at.tzinfo is None:
        return None
    matches = []
    for thread in threads:
        if (not thread.get("id") or thread.get("state") != "published"
                or not _is_agent_message_thread(thread)
                or (thread.get("source") or {}).get("type") != "api"
                or str(thread.get("id")) == str(row.get("draft_thread_id"))
                or sha256_of_html(thread.get("body") or "") != row["response_html_sha"]):
            continue
        created_at = _parse_iso(thread.get("createdAt"))
        if (created_at is not None and created_at.tzinfo is not None
                and created_at >= queued_at - timedelta(seconds=SKEW_GRACE_SECONDS)):
            matches.append(thread)
    return matches[0] if len(matches) == 1 else None


async def _record_delivery_readback(*, db_path: str, row: dict, helpscout, lease) -> bool:
    """Verify a terminal send without any possibility of sending it again."""
    from db.database import record_pending_send_delivery

    try:
        conversation = await helpscout.get_complete_conversation(int(row["conversation_id"]))
        if not isinstance(conversation, dict) or str(conversation.get("id")) != str(row["conversation_id"]):
            thread = None
        else:
            thread = _verified_delivery_thread(
                threads=conversation.get("_embedded", {}).get("threads", []) or [], row=row,
            )
        await lease.verify_owned()
    except asyncio.CancelledError:
        raise
    except Exception:
        record_pending_send_delivery(
            db_path=db_path, row_id=row["id"], receipt_error="delivery_readback_failed",
        )
        logger.exception("autosend: published delivery readback failed for row %s", row["id"])
        return False
    if thread is None:
        record_pending_send_delivery(
            db_path=db_path, row_id=row["id"], receipt_error="published_reply_not_verified",
        )
        return False
    return record_pending_send_delivery(
        db_path=db_path, row_id=row["id"], published_thread_id=str(thread["id"]),
        sent_at=thread["createdAt"],
    )


async def _cleanup_acknowledged_send(*, db_path: str, row: dict, helpscout, lease) -> bool:
    """Finish only housekeeping after publication; never modify a changed/human draft."""
    from db.database import fetch_scheduled_pending_sends_by_conversation, record_pending_send_cleanup

    def diagnostic(error):
        record_pending_send_cleanup(db_path=db_path, row_id=row["id"], error=error)

    diagnostic("cleanup_readback_failed")
    try:
        conversation_id = int(row["conversation_id"])
        conversation = await helpscout.get_complete_conversation(conversation_id)
        if not isinstance(conversation, dict) or str(conversation.get("id")) != str(conversation_id):
            return False
        if is_supa_ignored(conversation):
            diagnostic("cleanup_manual_conversation")
            return False
        threads = conversation.get("_embedded", {}).get("threads", []) or []
        draft = next((thread for thread in threads if _is_draft_thread(thread)
                      and str(thread.get("id")) == str(row["draft_thread_id"])), None)
        if draft:
            if (draft.get("source") or {}).get("type") != "api":
                diagnostic("cleanup_draft_not_api_owned")
                return False
            placeholder = _is_manual_review_placeholder(draft.get("body") or "")
            if not placeholder and sha256_of_html(draft.get("body") or "") != row["response_html_sha"]:
                diagnostic("cleanup_draft_modified")
                return False
            if not placeholder:
                diagnostic("cleanup_delete_draft_failed")
                if not await lease.run_guarded(helpscout.delete_draft, conversation_id, int(row["draft_thread_id"])):
                    return False

        # A later queued reply owns its status/tags. The old exact draft was safe to
        # neutralize, but removing that newer reply's pending marker would hide it.
        pending = fetch_scheduled_pending_sends_by_conversation(
            db_path=db_path, conversation_id=row["conversation_id"],
        )
        if any(queued["id"] > row["id"] for queued in pending):
            diagnostic("cleanup_newer_send_pending")
            return False
        queued_at = _parse_iso(row.get("queued_at"))
        newer_customer = any(
            thread.get("type") == "customer"
            and str(thread.get("id")) != str(row.get("last_known_customer_thread_id"))
            and (not queued_at or not (created := _parse_iso(thread.get("createdAt")))
                 or created >= queued_at - timedelta(seconds=SKEW_GRACE_SECONDS))
            for thread in threads
        )
        if not newer_customer and conversation.get("status") not in {"closed", "spam"}:
            diagnostic("cleanup_status_failed")
            status = await lease.run_guarded(helpscout.ensure_conversation_status, conversation_id, "pending")
            if not isinstance(status, dict) or status.get("verified") is not True:
                return False
        existing_tags = _normalize_tag_names(conversation.get("tags", []) or [])
        diagnostic("cleanup_add_tags_failed")
        if not await lease.run_guarded(helpscout.add_tags, conversation_id, row["tags_on_promote"], existing_tags):
            return False
        diagnostic("cleanup_remove_tag_failed")
        if not await lease.run_guarded(
            helpscout.remove_tag, conversation_id, "supa-pending-send",
            list(dict.fromkeys(existing_tags + row["tags_on_promote"])),
        ):
            return False
        diagnostic(None)
        return True
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("autosend: post-send cleanup remains incomplete for row %s", row["id"])
        return False


async def _promote(
    *,
    helpscout,
    conversation_id: str,
    row: dict,
    response_html: str,
    customer_id: int | None,
    existing_tags: list,
    db_path: str,
    lease,
) -> None:
    """Send the draft content, update status, swap tags, delete draft, mark sent.

    The caller already verified the draft against the approved hash. After the
    API acknowledges the send, a fresh complete read records the published receipt.
    """
    from db.database import record_pending_send_delivery, update_pending_send_state, update_pending_send_retry

    sent = await lease.run_guarded(
        helpscout.send_reply,
        int(conversation_id),
        response_html,
        customer_id=customer_id,
    )
    if not sent:
        attempts = row["attempts"] + 1
        if attempts >= MAX_RETRY_ATTEMPTS:
            update_pending_send_state(
                db_path=db_path, row_id=row["id"], state="failed",
                cancel_reason="send_failed",
            )
            logger.warning("autosend: row %s failed after %s attempts", row["id"], attempts)
            return

        new_due = (datetime.now(timezone.utc) + timedelta(seconds=_retry_delay_seconds(attempts))).isoformat()
        update_pending_send_retry(
            db_path=db_path, row_id=row["id"],
            new_due_at=new_due, last_error="send_reply returned False",
        )
        return

    # The API acknowledged publication. Fence the row before any later readback,
    # status/tag cleanup, lease failure or cancellation can return it to a due queue.
    record_pending_send_delivery(db_path=db_path, row_id=row["id"])
    await _record_delivery_readback(db_path=db_path, row=row, helpscout=helpscout, lease=lease)
    await _cleanup_acknowledged_send(db_path=db_path, row=row, helpscout=helpscout, lease=lease)


async def _cancel(
    *,
    helpscout,
    conversation_id: str,
    existing_tags: list,
    row_id: int,
    reason: str,
    db_path: str,
    lease,
) -> None:
    from db.database import update_pending_send_state

    try:
        await lease.run_guarded(
            helpscout.remove_tag,
            int(conversation_id),
            "supa-pending-send",
            existing_tags or [],
        )
    except ConversationLockError:
        raise
    except Exception:
        logger.exception("autosend: failed to remove supa-pending-send tag on cancel")
    update_pending_send_state(db_path=db_path, row_id=row_id, state="cancelled", cancel_reason=reason)


async def _process_due_row_with_lease(
    *,
    db_path: str,
    row_id: int,
    row: dict,
    helpscout,
    lease,
) -> None:
    """Refetch and evaluate a scheduled send while its conversation is leased."""

    from db.database import record_pending_send_delivery, update_pending_send_state, update_pending_send_retry

    conversation_id = row["conversation_id"]
    try:
        convo = await helpscout.get_conversation(int(conversation_id))
    except Exception as exc:
        attempts = row["attempts"] + 1
        if attempts >= MAX_RETRY_ATTEMPTS:
            update_pending_send_state(
                db_path=db_path,
                row_id=row_id,
                state="failed",
                cancel_reason="poll_failed",
            )
            return
        new_due = (
            datetime.now(timezone.utc)
            + timedelta(seconds=_retry_delay_seconds(attempts))
        ).isoformat()
        update_pending_send_retry(
            db_path=db_path,
            row_id=row_id,
            new_due_at=new_due,
            last_error=f"get_conversation: {exc!r}",
        )
        return

    if is_supa_ignored(convo):
        return

    threads = convo.get("_embedded", {}).get("threads", []) or []
    existing_tags = _normalize_tag_names(convo.get("tags", []) or [])
    draft_present = any(
        _is_draft_thread(t) and str(t.get("id")) == str(row["draft_thread_id"])
        for t in threads
    )
    already_sent = _verified_delivery_thread(threads=threads, row=row)

    if already_sent is not None:
        record_pending_send_delivery(db_path=db_path, row_id=row_id)
        await _record_delivery_readback(db_path=db_path, row=row, helpscout=helpscout, lease=lease)
        await _cleanup_acknowledged_send(db_path=db_path, row=row, helpscout=helpscout, lease=lease)
        return

    if convo.get("status") in {"closed", "spam"}:
        await _cancel(helpscout=helpscout, conversation_id=conversation_id, existing_tags=existing_tags,
                      row_id=row_id, reason="conversation_closed", db_path=db_path, lease=lease)
        return

    if not draft_present:
        await _cancel(
            helpscout=helpscout,
            conversation_id=conversation_id,
            existing_tags=existing_tags,
            row_id=row_id,
            reason="draft_deleted",
            db_path=db_path,
            lease=lease,
        )
        return

    reason = evaluate_cancel_reason(row=row, threads=threads)
    if reason is not None:
        await _cancel(
            helpscout=helpscout,
            conversation_id=conversation_id,
            existing_tags=existing_tags,
            row_id=row_id,
            reason=reason,
            db_path=db_path,
            lease=lease,
        )
        return

    draft_thread = next(
        (
            t
            for t in threads
            if _is_draft_thread(t)
            and str(t.get("id")) == str(row["draft_thread_id"])
        ),
        None,
    )
    response_html = (draft_thread or {}).get("body") or ""
    customer = convo.get("primaryCustomer") or convo.get("createdBy") or {}
    await _promote(
        helpscout=helpscout,
        conversation_id=conversation_id,
        row=row,
        response_html=response_html,
        customer_id=customer.get("id"),
        existing_tags=existing_tags,
        db_path=db_path,
        lease=lease,
    )


async def process_due_row(
    *,
    db_path: str,
    row_id: int,
    helpscout,
    config: AutoSendConfig,
) -> None:
    """Process a single due pending_sends row.

    Fetches the conversation once; handles cancel, idempotent-already-sent, or
    promote with transient-failure retry.
    """
    from db.database import get_db, _row_to_pending_send

    conn = get_db(db_path)
    try:
        raw = conn.execute("SELECT * FROM pending_sends WHERE id = ?", (row_id,)).fetchone()
    finally:
        conn.close()
    if raw is None or raw["state"] != "scheduled":
        return

    row = _row_to_pending_send(raw)
    conversation_id = row["conversation_id"]
    owner = f"autosend-scheduler:{conversation_id}:{uuid.uuid4().hex}"
    async with conversation_lease(
        db_path, int(conversation_id), owner=owner
    ) as lease:
        await _process_due_row_with_lease(
            db_path=db_path,
            row_id=row_id,
            row=row,
            helpscout=helpscout,
            lease=lease,
        )


async def run_worker_loop(
    *,
    db_path: str,
    helpscout,
    config: AutoSendConfig,
    iterations: Optional[int] = None,
    sleep_func=asyncio.sleep,
) -> None:
    """Run the worker loop: every tick_seconds, process all due rows.

    If config.enabled is False, the loop sleeps and does nothing (kill switch).
    Cancellation-safe via asyncio.CancelledError propagation.
    """
    from db.database import fetch_due_pending_sends

    try:
        await recover_normalized_autosends(db_path=db_path, helpscout=helpscout, config=config)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("autosend: normalization reconciliation failed")

    cycle = 0
    while iterations is None or cycle < iterations:
        try:
            if config.enabled:
                await reconcile_unverified_autosend_receipts(db_path=db_path, helpscout=helpscout)
                await reconcile_autosend_cleanup(db_path=db_path, helpscout=helpscout)
                now_iso = datetime.now(timezone.utc).isoformat()
                due = fetch_due_pending_sends(db_path=db_path, now_iso=now_iso)
                for row in due:
                    try:
                        await process_due_row(
                            db_path=db_path,
                            row_id=row["id"],
                            helpscout=helpscout,
                            config=config,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception("autosend: process_due_row crashed for row %s", row["id"])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("autosend: worker loop tick failed")

        cycle += 1
        if iterations is not None and cycle >= iterations:
            break

        await sleep_func(config.tick_seconds)


async def reconcile_unverified_autosend_receipts(*, db_path: str, helpscout) -> list[int]:
    """Retry readback for at most ten due receipts; each failed row waits five minutes.

    Historical sent rows have no diagnostic and are deliberately excluded. This path
    never calls a customer mutation, promotes a draft or requeues a delivery.
    """
    from db.database import fetch_unverified_autosend_receipts

    before = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    verified = []
    for row in fetch_unverified_autosend_receipts(db_path=db_path, before=before):
        try:
            async with conversation_lease(
                db_path, int(row["conversation_id"]), owner=f"autosend-receipt:{uuid.uuid4().hex}",
            ) as lease:
                if await _record_delivery_readback(db_path=db_path, row=row, helpscout=helpscout, lease=lease):
                    verified.append(row["id"])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("autosend: delivery receipt reconciliation failed for row %s", row["id"])
    return verified


async def reconcile_autosend_cleanup(*, db_path: str, helpscout) -> list[int]:
    """Retry at most ten due cleanup jobs under fresh leases, without a send path."""
    from db.database import fetch_pending_autosend_cleanup

    before = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    cleaned = []
    for row in fetch_pending_autosend_cleanup(db_path=db_path, before=before):
        try:
            async with conversation_lease(
                db_path, int(row["conversation_id"]), owner=f"autosend-cleanup:{uuid.uuid4().hex}",
            ) as lease:
                if await _cleanup_acknowledged_send(db_path=db_path, row=row, helpscout=helpscout, lease=lease):
                    cleaned.append(row["id"])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("autosend: cleanup reconciliation remains incomplete for row %s", row["id"])
    return cleaned


async def recover_normalized_autosends(*, db_path: str, helpscout, config: AutoSendConfig) -> list[int]:
    """Startup-only bounded repair of equivalent HTML rejected by legacy raw request hashes.

    Requeue only the latest, unattempted, still-accepted auto-send obligation. Read complete
    history under its lease and retain every ordinary cancellation gate. Publishing remains
    the normal due-row worker's job, under the original queued_at/due_at.
    """
    from agents.support_agent import autosend_threshold, load_config as support_config
    from db.database import normalization_recovery_candidates, repin_normalized_send
    from services.helpscout import get_latest_actionable_thread, get_supa_owned_draft_thread_ids

    if not config.enabled or support_config().mode != "live":
        return []
    minimum = autosend_threshold()
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    recovered = []
    for row in normalization_recovery_candidates(db_path=db_path, since=since, min_score=minimum):
        expected = row["expected_html"] or ""
        if not expected or sha256_of_html(expected) != row["response_html_sha"]:
            continue
        try:
            async with conversation_lease(db_path, int(row["conversation_id"]),
                                          owner=f"autosend-normalization:{uuid.uuid4().hex}") as lease:
                convo = await helpscout.get_complete_conversation(int(row["conversation_id"]))
                if convo.get("status") != "active" or is_supa_ignored(convo):
                    continue
                threads = convo.get("_embedded", {}).get("threads", []) or []
                latest = get_latest_actionable_thread(threads)
                if (not latest or latest.get("type") != "customer"
                        or str(latest.get("id")) != str(row["last_known_customer_thread_id"])):
                    continue
                owned = {str(i) for i in get_supa_owned_draft_thread_ids(threads)}
                draft = next((t for t in threads if str(t.get("id")) == str(row["draft_thread_id"])), None)
                if not draft or str(row["draft_thread_id"]) not in owned:
                    continue
                persisted = draft.get("body") or ""
                persisted_hash = sha256_of_html(persisted)
                if persisted_hash == row["response_html_sha"] or not email_html_equivalent(expected, persisted):
                    continue
                verified_row = {**row, "response_html_sha": persisted_hash}
                if evaluate_cancel_reason(row=verified_row, threads=threads) is not None:
                    continue
                await lease.verify_owned()
                if repin_normalized_send(db_path=db_path, row=row, persisted_hash=persisted_hash, min_score=minimum):
                    recovered.append(row["id"])
                    logger.info("autosend: recovered verified HTML normalization row %s", row["id"])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("autosend: could not reconcile normalization for row %s", row["id"])
    return recovered


def schedule_autosend_worker(*, helpscout, db_path: Optional[str] = None) -> asyncio.Task:
    """Create the asyncio.Task that FastAPI startup attaches to app.state."""
    resolved_db_path = db_path or get_supa_db_path()
    config = load_config()
    return asyncio.create_task(
        run_worker_loop(
            db_path=resolved_db_path,
            helpscout=helpscout,
            config=config,
        )
    )
