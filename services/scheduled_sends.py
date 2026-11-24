"""Scheduled agent replies: `supa draft schedule` books a reply for later.

Replies from case agents go out 30-60 minutes after the
customer's email, at a random moment in that window, any hour of the day. An agent books
the reply with `supa draft schedule`; this module stores it and the daemon's worker sends
it when it is due.

What is stored: the reply body itself (`--body-file`, preferred) or a snapshot of the
saved draft (`--expected-version` without a body). Help Scout's API cannot publish a
draft, so both are sent through the reply path of the gateway (`send_reply`), which is
what `draft send` also uses: same body normalization and paragraph spacing, same
idempotency ledger, conversation lease, Pending status, draft neutralization, tags,
learning callback and `supa.reply.sent` event.

At send time the worker re-reads the token (revoked token, lost `send` scope or a
conversation no longer on its list hold the send) and, under the conversation lease on
the same preflight read the send uses, holds the send when:

- the conversation is closed or spam (`conversation_closed`, `conversation_spam`);
- the customer wrote again (`customer_replied`);
- someone published a reply since it was booked (`agent_replied`);
- a person added a note since it was booked (`note_added`; API notes and Supa's
  decision notes do not count);
- the live draft changed, appeared or disappeared (`draft_changed`).

Upstream failures retry with backoff a few times, then the row fails. Each attempt has
its own ledger key, recorded before the call, so a restart never sends twice: a key whose
outcome is unknown fails the row instead of retrying.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from db.database import get_db, init_db
from services.agent_auth import (
    CONVERSATION_NOT_ALLOWED_CODE,
    AgentIdentity,
    current_identity,
)
from services.agent_events import append_supa_event
from services.agent_operations import (
    AgentOperationError,
    AgentOperationNotFoundError,
    get_operation_locally,
)
from services.conversation_gateway import (
    ConversationGateway,
    ConversationGatewayError,
    expected_thread_error,
    normalize_cc,
    normalize_conversation,
    normalize_draft_body,
)
from services.conversation_locks import ConversationLockError
from services.helpscout import (
    _is_agent_message_thread,
    _is_customer_thread,
    _is_draft_thread,
    _is_supa_system_note,
    _parse_thread_created_at,
)
from services.runtime_paths import get_supa_db_path

logger = logging.getLogger(__name__)

DEFAULT_AFTER_CUSTOMER_MINUTES = (30, 60)
LATE_DELAY_MINUTES = (2, 8)
MIN_LEAD = timedelta(minutes=1)
MAX_AHEAD = timedelta(days=7)
MAX_WINDOW_MINUTES = 24 * 60
MAX_ATTEMPTS = 4
BASE_RETRY_SECONDS = 60
MAX_RETRY_SECONDS = 600
CLAIM_TTL = timedelta(minutes=10)
DEFAULT_TICK_SECONDS = 30
STATUSES = ("queued", "sent", "held", "cancelled", "failed")
HOLD_REASONS = (
    "conversation_closed",
    "conversation_spam",
    "customer_replied",
    "agent_replied",
    "note_added",
    "draft_changed",
    "conversation_changed",
    "token_revoked",
    "scope_revoked",
    CONVERSATION_NOT_ALLOWED_CODE,
)
# A failed ledger attempt with one of these codes decided nothing about the message;
# the next attempt re-reads the conversation and decides again.
_REEVALUATE_CODES = frozenset({"scheduled_send_held", "conversation_changed"})

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse(value: Any) -> datetime | None:
    return _parse_thread_created_at(value)


def _error(code: str, status: int, message: str, *, retryable: bool = False,
           details: dict[str, Any] | None = None) -> ConversationGatewayError:
    return ConversationGatewayError(
        message, code=code, status=status, retryable=retryable, details=details
    )


# --- send_at -------------------------------------------------------------------------


def compute_send_at(
    *,
    now: datetime,
    customer_message_at: datetime | None,
    after_customer: tuple[int, int] = DEFAULT_AFTER_CUSTOMER_MINUTES,
    send_at: datetime | None = None,
    rng: random.Random | None = None,
) -> datetime:
    """When a booked reply goes out.

    An explicit `send_at` is used as given. Otherwise the reply goes out a uniformly
    random [MIN, MAX] minutes after the customer's latest message; if that moment has
    already passed (or there is no customer message), a random 2-8 minutes from now.
    Never earlier than one minute from now.
    """
    picker = rng or random
    floor = now + MIN_LEAD
    if send_at is not None:
        return max(send_at, floor)
    low, high = after_customer
    if customer_message_at is not None:
        candidate = customer_message_at + timedelta(
            seconds=picker.randint(low * 60, high * 60)
        )
        if candidate > now:
            return max(candidate, floor)
    late_low, late_high = LATE_DELAY_MINUTES
    return max(now + timedelta(seconds=picker.randint(late_low * 60, late_high * 60)), floor)


def latest_customer_message_at(conversation: dict[str, Any]) -> datetime | None:
    """Created time of the customer's latest published message in a normalized conversation."""
    latest: datetime | None = None
    for thread in conversation.get("threads") or []:
        if not isinstance(thread, dict) or thread.get("draft") or thread.get("is_internal_note"):
            continue
        author = thread.get("author") if isinstance(thread.get("author"), dict) else {}
        if author.get("role") != "customer" or thread.get("state") not in (None, "published"):
            continue
        created = _parse(thread.get("created_at"))
        if created is not None and (latest is None or created > latest):
            latest = created
    return latest


# --- storage -------------------------------------------------------------------------


def _connect(db_path: str) -> sqlite3.Connection:
    return get_db(db_path)


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def public_record(row: dict[str, Any]) -> dict[str, Any]:
    """The agent-visible view of one booking; the body is never returned."""
    return {
        "schedule_id": row["id"],
        "conversation_id": row["conversation_id"],
        "requested_conversation_id": row["requested_conversation_id"],
        "status": row["status"],
        "hold_reason": row["hold_reason"],
        "send_at": row["send_at"],
        "source": row["source"],
        "draft_thread_id": row["draft_thread_id"],
        "draft_version": row["draft_version"],
        "expected_latest_thread_id": row["expected_latest_thread_id"],
        "customer_message_at": row["customer_message_at"],
        "body_sha256": row["body_sha256"],
        "cc": _row_cc(row),
        "token_name": row["token_name"],
        "attempts": row["attempts"],
        "next_attempt_at": row["next_attempt_at"],
        "last_error": row["last_error"],
        "sent_thread_id": row["sent_thread_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
    }


def _row_cc(row: dict[str, Any]) -> list[str]:
    """The CC list stored with a booking ([] for rows booked without one)."""
    try:
        value = json.loads(row.get("cc_json") or "[]")
    except (TypeError, ValueError):
        return []
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def get_row(db_path: str, row_id: int) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        return _row_dict(conn.execute("SELECT * FROM scheduled_sends WHERE id = ?", (row_id,)).fetchone())
    finally:
        conn.close()


def latest_for_conversation(db_path: str, conversation_id: int) -> dict[str, Any] | None:
    """The newest booking for a conversation (by requested or resolved id), any status."""
    conn = _connect(db_path)
    try:
        return _row_dict(conn.execute(
            "SELECT * FROM scheduled_sends WHERE conversation_id = ? OR requested_conversation_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (conversation_id, conversation_id),
        ).fetchone())
    finally:
        conn.close()


def _update(conn: sqlite3.Connection, row_id: int, now: datetime, **fields: Any) -> None:
    fields["updated_at"] = _iso(now)
    assignments = ", ".join(f"{name} = ?" for name in fields)
    conn.execute(f"UPDATE scheduled_sends SET {assignments} WHERE id = ?", (*fields.values(), row_id))


def _request_hash(request: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _claim_is_live(row: dict[str, Any], now: datetime) -> bool:
    claimed = _parse(row.get("claimed_at"))
    return claimed is not None and claimed > now - CLAIM_TTL


# --- booking -------------------------------------------------------------------------


async def schedule_send(
    *,
    gateway: ConversationGateway,
    db_path: str,
    identity: AgentIdentity,
    conversation_id: int,
    expected_latest_thread_id: int,
    idempotency_key: str,
    body: str | None = None,
    body_format: str | None = None,
    expected_version: str | None = None,
    send_at: datetime | None = None,
    after_customer: tuple[int, int] | None = None,
    clock: Clock = _utc_now,
    rng: random.Random | None = None,
    cc: list[str] | None = None,
) -> dict[str, Any]:
    """Book a reply; replaces the conversation's queued booking, if any.

    `cc` is checked now (`gateway.check_cc`) and again by the gateway at send time.
    Booking the saved draft also keeps the draft's own CC.
    """
    if not identity.allows_conversation(conversation_id):
        raise _error(CONVERSATION_NOT_ALLOWED_CODE, 403, "Conversation not allowed")
    requested_cc = normalize_cc(cc)
    if (body is None) != (body_format is None):
        raise _error("invalid_schedule", 422, "body and body_format go together")
    if body is None and expected_version is None:
        raise _error("invalid_schedule", 422, "A body or the saved draft's version is required")
    if send_at is not None and after_customer is not None:
        raise _error("invalid_schedule", 422, "send_at and after_customer exclude each other")
    window = after_customer or DEFAULT_AFTER_CUSTOMER_MINUTES
    if not (0 <= window[0] <= window[1] <= MAX_WINDOW_MINUTES):
        raise _error("invalid_schedule", 422, "after_customer must be MIN-MAX minutes")
    now = clock()
    if send_at is not None:
        if send_at.tzinfo is None or send_at.utcoffset() is None:
            raise _error("invalid_send_at", 422, "send_at needs a timezone")
        if send_at > now + MAX_AHEAD:
            raise _error("invalid_send_at", 422, "send_at is more than 7 days ahead")

    request = {
        "conversation_id": conversation_id,
        "body": body,
        "body_format": body_format,
        "expected_version": expected_version,
        "expected_latest_thread_id": expected_latest_thread_id,
        "send_at": _iso(send_at) if send_at is not None else None,
        "after_customer": list(after_customer) if after_customer is not None else None,
    }
    if requested_cc:
        # Only when given, so bookings without CC keep their idempotency hash.
        request["cc"] = requested_cc
    request_hash = _request_hash(request)
    replay = _replay(db_path, identity, idempotency_key, request_hash)
    if replay is not None:
        return replay

    conversation = await gateway.get(conversation_id)
    resolved_id = conversation.get("resolved_id") or conversation.get("id")
    if not identity.allows_conversation(resolved_id) or not isinstance(resolved_id, int):
        raise _error(CONVERSATION_NOT_ALLOWED_CODE, 403, "Conversation not allowed")
    if conversation.get("status") in {"closed", "spam"}:
        raise _error("conversation_not_open", 409, "The conversation is closed or spam")
    thread_error = expected_thread_error(conversation, expected_latest_thread_id)
    if thread_error is not None:
        raise thread_error
    draft = conversation.get("draft") if isinstance(conversation.get("draft"), dict) else None
    draft_version = draft.get("version") if draft else None
    draft_thread_id = draft.get("thread_id") if draft else None
    if expected_version is not None and draft_version != expected_version:
        raise _error(
            "draft_conflict", 409, "The draft changed after it was read",
            details={"current_version": draft_version} if draft_version else {},
        )
    if body is None:
        body, body_format, source = draft.get("body_html") or "", "html", "draft"
        effective_cc = normalize_cc([*(draft.get("cc") or []), *requested_cc])
    else:
        source = "body"
        effective_cc = requested_cc
    normalized = normalize_draft_body(body, body_format)  # raises invalid_body / body_too_large
    await gateway.check_cc(conversation, effective_cc)
    customer_at = latest_customer_message_at(conversation)
    due = compute_send_at(
        now=now, customer_message_at=customer_at, after_customer=window, send_at=send_at, rng=rng,
    )

    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT * FROM scheduled_sends WHERE token_id = ? AND idempotency_key = ?",
            (identity.token_id, idempotency_key),
        ).fetchone()
        if existing is not None:
            conn.rollback()
            return _replay_row(dict(existing), request_hash)
        queued = conn.execute(
            "SELECT * FROM scheduled_sends WHERE conversation_id = ? AND status = 'queued'",
            (resolved_id,),
        ).fetchone()
        if queued is not None:
            if _claim_is_live(dict(queued), now):
                conn.rollback()
                raise _error("schedule_sending", 409, "The booked reply is being sent now", retryable=True)
            _update(conn, queued["id"], now, status="cancelled", hold_reason="replaced",
                    completed_at=_iso(now))
        cursor = conn.execute(
            """INSERT INTO scheduled_sends (
                conversation_id, requested_conversation_id, source, body, body_format, body_sha256,
                draft_thread_id, draft_version, expected_latest_thread_id, customer_message_at,
                send_at, status, token_id, token_name, idempotency_key, request_hash,
                created_at, updated_at, cc_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)""",
            (
                resolved_id, conversation_id, source, body, body_format,
                hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                draft_thread_id if isinstance(draft_thread_id, int) else None, draft_version,
                expected_latest_thread_id, _iso(customer_at) if customer_at else None,
                _iso(due), identity.token_id, identity.name, idempotency_key, request_hash,
                _iso(now), _iso(now), json.dumps(effective_cc) if effective_cc else None,
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()
    return public_record(get_row(db_path, row_id))


def _replay_row(row: dict[str, Any], request_hash: str) -> dict[str, Any]:
    if row["request_hash"] != request_hash:
        raise _error("idempotency_conflict", 409, "Idempotency key conflict")
    return public_record(row)


def _replay(db_path: str, identity: AgentIdentity, key: str, request_hash: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM scheduled_sends WHERE token_id = ? AND idempotency_key = ?",
            (identity.token_id, key),
        ).fetchone()
    finally:
        conn.close()
    return _replay_row(dict(row), request_hash) if row is not None else None


def cancel_send(*, db_path: str, identity: AgentIdentity, conversation_id: int,
                clock: Clock = _utc_now) -> dict[str, Any]:
    """Cancel the conversation's queued booking."""
    if not identity.allows_conversation(conversation_id):
        raise _error(CONVERSATION_NOT_ALLOWED_CODE, 403, "Conversation not allowed")
    now = clock()
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM scheduled_sends WHERE (conversation_id = ? OR requested_conversation_id = ?) "
            "AND status = 'queued' ORDER BY id DESC LIMIT 1",
            (conversation_id, conversation_id),
        ).fetchone()
        if row is None:
            conn.rollback()
            raise _error("schedule_not_found", 404, "No queued reply for this conversation")
        if not identity.allows_conversation(row["conversation_id"]):
            conn.rollback()
            raise _error(CONVERSATION_NOT_ALLOWED_CODE, 403, "Conversation not allowed")
        if _claim_is_live(dict(row), now):
            conn.rollback()
            raise _error("schedule_sending", 409, "The booked reply is being sent now", retryable=True)
        _update(conn, row["id"], now, status="cancelled", hold_reason="cancelled_by_agent",
                completed_at=_iso(now))
        conn.commit()
        row_id = row["id"]
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()
    return public_record(get_row(db_path, row_id))


# --- send-time checks ----------------------------------------------------------------


def hold_reason_for(row: dict[str, Any], raw: dict[str, Any]) -> str | None:
    """Why a booked reply must not go out now, read from the live conversation; else None."""
    status = raw.get("status")
    if status == "closed":
        return "conversation_closed"
    if status == "spam":
        return "conversation_spam"
    embedded = raw.get("_embedded") if isinstance(raw.get("_embedded"), dict) else {}
    threads = [t for t in embedded.get("threads") or [] if isinstance(t, dict)]
    booked_at = _parse(row["created_at"])
    expected = str(row["expected_latest_thread_id"])

    def after_booking(thread: dict[str, Any]) -> bool:
        created = _parse(thread.get("createdAt"))
        return booked_at is None or created is None or created > booked_at

    published = [t for t in threads if not _is_draft_thread(t) and t.get("state") in (None, "published")]
    latest = ConversationGateway._latest_actionable_thread(raw)
    if any(_is_customer_thread(t) and t.get("type") != "note" and str(t.get("id")) != expected
           and after_booking(t) for t in published):
        return "customer_replied"
    if latest is not None and str(latest.get("id")) != expected and _is_customer_thread(latest):
        return "customer_replied"
    if any(_is_agent_message_thread(t) and str(t.get("id")) != expected and after_booking(t)
           for t in published):
        return "agent_replied"
    if latest is not None and str(latest.get("id")) != expected:
        return "agent_replied"
    for thread in threads:
        if thread.get("type") != "note" or not after_booking(thread):
            continue
        author = thread.get("createdBy") if isinstance(thread.get("createdBy"), dict) else {}
        source = thread.get("source") if isinstance(thread.get("source"), dict) else {}
        if (author.get("type") == "user" and source.get("type") != "api"
                and not _is_supa_system_note(thread.get("body") or "")):
            return "note_added"
    live = normalize_conversation(raw).get("draft")
    live_version = live.get("version") if isinstance(live, dict) else None
    if live_version != row["draft_version"]:
        return "draft_changed"
    return None


# --- sender --------------------------------------------------------------------------


def _retry_delay(attempts: int) -> timedelta:
    return timedelta(seconds=min(BASE_RETRY_SECONDS * (2 ** max(0, attempts - 1)), MAX_RETRY_SECONDS))


def _emit(db_path: str, row: dict[str, Any], status: str, reason: str | None) -> None:
    try:
        append_supa_event(
            db_path,
            type=f"supa.scheduled_send.{status}",
            dedupe_key=f"scheduled-send:{row['id']}:{status}",
            conversation_id=row["conversation_id"],
            summary={"result": status, "status": reason or status, "token_name": row["token_name"]},
        )
    except Exception:
        logger.exception("scheduled send %s: event append failed", row["id"])


def _finish(db_path: str, row: dict[str, Any], now: datetime, status: str,
            reason: str | None = None, **fields: Any) -> str:
    conn = _connect(db_path)
    try:
        _update(conn, row["id"], now, status=status, hold_reason=reason, claimed_at=None,
                completed_at=_iso(now), **fields)
        conn.commit()
    finally:
        conn.close()
    if status in {"held", "failed"}:
        _emit(db_path, row, status, reason)
    logger.info("scheduled send %s for %s: %s %s", row["id"], row["conversation_id"], status, reason or "")
    return status if reason is None else f"{status}:{reason}"


def _retry_or_fail(db_path: str, row: dict[str, Any], now: datetime, attempts: int, code: str) -> str:
    if attempts >= MAX_ATTEMPTS:
        return _finish(db_path, row, now, "failed", code, last_error=code)
    conn = _connect(db_path)
    try:
        _update(conn, row["id"], now, claimed_at=None, last_error=code,
                next_attempt_at=_iso(now + _retry_delay(attempts)))
        conn.commit()
    finally:
        conn.close()
    return f"retry:{code}"


def _identity_hold(row: dict[str, Any], identity: AgentIdentity | None) -> str | None:
    if identity is None:
        return "token_revoked"
    if "send" not in identity.scopes:
        return "scope_revoked"
    if not (identity.allows_conversation(row["conversation_id"])
            and identity.allows_conversation(row["requested_conversation_id"])):
        return CONVERSATION_NOT_ALLOWED_CODE
    return None


async def process_row(
    row_id: int,
    *,
    gateway: ConversationGateway,
    db_path: str,
    clock: Clock | None = None,
    identity_lookup: Callable[[str], AgentIdentity | None] | None = None,
) -> str:
    """Send, hold, retry or fail one due booking. Returns a short outcome label."""
    clock = clock or _utc_now
    identity_lookup = identity_lookup or current_identity
    now = clock()
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = _row_dict(conn.execute("SELECT * FROM scheduled_sends WHERE id = ?", (row_id,)).fetchone())
        if row is None or row["status"] != "queued":
            conn.rollback()
            return "skipped"
        if _claim_is_live(row, now):
            conn.rollback()
            return "busy"
        _update(conn, row_id, now, claimed_at=_iso(now))
        conn.commit()
    finally:
        conn.close()

    try:
        identity = identity_lookup(row["token_id"])
        if reason := _identity_hold(row, identity):
            return _finish(db_path, row, now, "held", reason)

        attempts = row["attempts"]
        key = row["send_operation_key"]
        if key:
            # A previous attempt reached the ledger: its outcome decides, never a blind resend.
            try:
                record = get_operation_locally(idempotency_key=key, db_path=db_path)
            except AgentOperationNotFoundError:
                record = None  # Never reserved: the same key is still safe to use.
            if record is not None:
                if record.state in {"succeeded", "sent_with_warning"}:
                    thread_id = record.upstream_thread_id
                    return _finish(db_path, row, now, "sent",
                                   sent_thread_id=int(thread_id) if thread_id else None)
                if record.state in {"reserved", "in_progress"}:
                    conn = _connect(db_path)
                    try:
                        _update(conn, row_id, now, claimed_at=None)
                        conn.commit()
                    finally:
                        conn.close()
                    return "busy"
                if record.state == "outcome_unknown":
                    return _finish(db_path, row, now, "failed", "outcome_unknown",
                                   last_error="outcome_unknown")
                error = record.error or {}
                code = str(error.get("code") or "send_failed")
                if code not in _REEVALUATE_CODES and not error.get("retryable"):
                    return _finish(db_path, row, now, "failed", code, last_error=code)
                if code not in _REEVALUATE_CODES and attempts >= MAX_ATTEMPTS:
                    return _finish(db_path, row, now, "failed", code, last_error=code)
                key = None
        if not key:
            attempts += 1
            key = f"scheduled-send:{row_id}:{attempts}"
            conn = _connect(db_path)
            try:
                _update(conn, row_id, now, attempts=attempts, send_operation_key=key)
                conn.commit()
            finally:
                conn.close()

        try:
            result = await gateway.send_reply(
                row["conversation_id"],
                body=row["body"],
                body_format=row["body_format"],
                expected_latest_thread_id=row["expected_latest_thread_id"],
                idempotency_key=key,
                identity=identity,
                request_id=f"sched_{row_id}_{attempts}",
                send_guard=lambda raw: hold_reason_for(row, raw),
                cc=_row_cc(row) or None,
            )
        except (ConversationGatewayError, ConversationLockError, AgentOperationError) as exc:
            code = getattr(exc, "code", "send_failed")
            if code == "scheduled_send_held":
                reason = exc.details.get("hold_reason") or "held"
                return _finish(db_path, row, now, "held", reason)
            if code in {"conversation_changed", CONVERSATION_NOT_ALLOWED_CODE}:
                return _finish(db_path, row, now, "held", code)
            if code in {"outcome_unknown", "upstream_outcome_unknown"}:
                return _finish(db_path, row, now, "failed", "outcome_unknown", last_error=code)
            if getattr(exc, "retryable", False) or code == "operation_in_progress":
                return _retry_or_fail(db_path, row, now, attempts, code)
            return _finish(db_path, row, now, "failed", code, last_error=code)
        return _finish(db_path, row, now, "sent", sent_thread_id=result.get("thread_id"),
                       last_error=None)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("scheduled send %s crashed", row_id)
        return _retry_or_fail(db_path, row, now, row["attempts"] + 1, type(exc).__name__.lower()[:60])


def release_claims(db_path: str) -> None:
    """At daemon start no send is in flight: drop claims a stopped process left behind.

    Safe because the ledger key of an interrupted attempt stays on the row and decides
    the next tick (sent, or failed as outcome_unknown), never a blind resend.
    """
    conn = _connect(db_path)
    try:
        conn.execute("UPDATE scheduled_sends SET claimed_at = NULL WHERE status = 'queued'")
        conn.commit()
    finally:
        conn.close()


def due_row_ids(db_path: str, now: datetime) -> list[int]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id FROM scheduled_sends WHERE status = 'queued' "
            "AND COALESCE(next_attempt_at, send_at) <= ? ORDER BY send_at, id",
            (_iso(now),),
        ).fetchall()
    finally:
        conn.close()
    return [row["id"] for row in rows]


async def tick(*, gateway: ConversationGateway, db_path: str, clock: Clock | None = None,
               identity_lookup: Callable[[str], AgentIdentity | None] | None = None) -> list[tuple[int, str]]:
    """One pass over due bookings. Queued rows survive restarts; overdue rows go out now."""
    clock = clock or _utc_now
    outcomes = []
    for row_id in due_row_ids(db_path, clock()):
        try:
            outcomes.append((row_id, await process_row(
                row_id, gateway=gateway, db_path=db_path, clock=clock, identity_lookup=identity_lookup,
            )))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("scheduled send %s: tick failed", row_id)
    return outcomes


def _tick_seconds() -> int:
    try:
        return max(5, int(os.environ.get("SUPA_SCHEDULED_SEND_TICK_SECONDS") or DEFAULT_TICK_SECONDS))
    except ValueError:
        return DEFAULT_TICK_SECONDS


def _enabled() -> bool:
    raw = os.environ.get("SUPA_SCHEDULED_SEND_ENABLED")
    return raw is None or raw.strip().lower() not in {"0", "false", "no", "off", ""}


async def run_worker_loop(*, db_path: str, gateway_factory: Callable[[], ConversationGateway],
                          iterations: int | None = None, sleep_func=asyncio.sleep) -> None:
    """Every tick, send the due bookings (same loop shape as the autosend worker)."""
    init_db(db_path)
    release_claims(db_path)
    cycle = 0
    while iterations is None or cycle < iterations:
        try:
            if _enabled() and due_row_ids(db_path, _utc_now()):
                await tick(gateway=gateway_factory(), db_path=db_path)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("scheduled send worker tick failed")
        cycle += 1
        if iterations is not None and cycle >= iterations:
            break
        await sleep_func(_tick_seconds())


def schedule_worker(*, gateway_factory: Callable[[], ConversationGateway],
                    db_path: str | None = None) -> asyncio.Task:
    """The asyncio task FastAPI startup attaches to app.state."""
    return asyncio.create_task(
        run_worker_loop(db_path=db_path or get_supa_db_path(), gateway_factory=gateway_factory)
    )
