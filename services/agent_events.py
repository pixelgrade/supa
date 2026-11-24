"""Durable, compact event journal records for agent-facing pings.

The journal deliberately stores metadata only.  Message bodies, previews,
customer details, credentials, and arbitrary upstream payload fields are not
copied into ``summary_json``.  Consumers use an event as a signal to fetch the
current conversation through the normal authenticated read API.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Iterator, Mapping

from services.runtime_paths import get_supa_db_path


Clock = Callable[[], datetime]
PathValue = str | os.PathLike[str]
Sleep = Callable[[float], Awaitable[None]]

MAX_DEDUPE_INPUT_CHARS = 4_096
MAX_EVENT_TYPE_CHARS = 200
MAX_IDENTIFIER_CHARS = 200
MAX_SUMMARY_JSON_BYTES = 4_096
MAX_SUMMARY_TEXT_CHARS = 200
EVENT_LEASE_SECONDS = 60
ACK_TOKEN_BYTES = 32
DEFAULT_EVENT_RETENTION_DAYS = 30
DEFAULT_EVENT_RETENTION_INTERVAL_SECONDS = 21_600

_EVENT_TYPE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,199}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_NUMERIC_ID_RE = re.compile(r"^[1-9][0-9]{0,18}$")
_DEDUPE_KEY_RE = re.compile(r"^(?:helpscout|supa):[0-9a-f]{64}$")
_ACK_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43,512}$")
_TOKEN_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_BEARER_VALUE_RE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_SECRET_VALUE_RE = re.compile(
    r"\b(?:sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]{12,})\b"
)
_SUPA_TOKEN_VALUE_RE = re.compile(r"[A-Za-z0-9_-]{43,}")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:authorization|api[_-]?key|access[_-]?token|secret|password)"
    r"\s*[:=]\s*.*$"
)

_HELP_SCOUT_EVENT_TYPES = {
    "": "helpscout.conversation.created",
    "convo.created": "helpscout.conversation.created",
    "convo.customer.reply.created": "helpscout.customer.replied",
    "convo.note.created": "helpscout.note.created",
    "convo.agent.reply.created": "helpscout.agent.replied",
    "convo.assigned": "helpscout.conversation.assigned",
    "convo.status": "helpscout.conversation.status.changed",
    "convo.tags": "helpscout.conversation.tags.changed",
}
_ACTOR_TYPES = frozenset({"customer", "user", "system_user", "team", "system"})
_SUMMARY_CODE_KEYS = frozenset(
    {
        "delivery_mode",
        "operation_name",
        "result",
        "source_type",
        "state",
        "status",
        "thread_state",
        "thread_type",
    }
)
_SUMMARY_IDENTIFIER_KEYS = frozenset(
    {"operation_id", "request_id", "run_id", "token_id"}
)
_SUMMARY_NUMERIC_ID_KEYS = frozenset(
    {"conversation_id", "conversation_number", "mailbox_id", "thread_id"}
)
_SUMMARY_COUNT_KEYS = frozenset(
    {"attachment_count", "confidence_score", "tag_count"}
)
_SUMMARY_TEXT_KEYS = frozenset({"token_name"})
_SUMMARY_KEYS = (
    _SUMMARY_CODE_KEYS
    | _SUMMARY_IDENTIFIER_KEYS
    | _SUMMARY_NUMERIC_ID_KEYS
    | _SUMMARY_COUNT_KEYS
    | _SUMMARY_TEXT_KEYS
)


logger = logging.getLogger(__name__)


class AgentEventStorageError(RuntimeError):
    """The durable event journal could not be read or updated safely."""

    code = "event_storage_error"
    retryable = True
    status = 503
    exit_code = 6
    details: dict[str, Any] = {}


class AgentEventProtocolError(RuntimeError):
    """Base class carrying the future Agent API event error contract."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool,
        status: int,
        exit_code: int,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status = status
        self.exit_code = exit_code
        self.details = dict(details or {})


class AgentEventAckError(AgentEventProtocolError):
    """An acknowledgement does not own the named active or last lease."""

    def __init__(self) -> None:
        super().__init__(
            "The event acknowledgement is invalid or expired.",
            code="event_ack_invalid",
            retryable=False,
            status=409,
            exit_code=5,
        )


class AgentEventCursorExpiredError(AgentEventProtocolError):
    """Retention removed an event that the consumer had not acknowledged."""

    def __init__(
        self,
        *,
        oldest_retained_event_id: int | None,
        current_head_event_id: int,
    ) -> None:
        super().__init__(
            "The event cursor is older than the retained event journal.",
            code="cursor_expired",
            retryable=False,
            status=409,
            exit_code=5,
            details={
                "oldest_retained_event_id": oldest_retained_event_id,
                "current_head_event_id": current_head_event_id,
            },
        )


class AgentEventStartError(AgentEventProtocolError):
    """An explicit event start points beyond the current journal head."""

    def __init__(self, event_id: int, current_head_event_id: int) -> None:
        super().__init__(
            "The requested event start is beyond the current journal head.",
            code="event_start_invalid",
            retryable=False,
            status=400,
            exit_code=2,
            details={
                "event_id": event_id,
                "current_head_event_id": current_head_event_id,
            },
        )


@dataclass(frozen=True)
class AgentEvent:
    """A normalized journal event before it receives its monotonic database ID."""

    type: str
    conversation_id: str | None
    thread_id: str | None
    run_id: str | None
    source: str
    actor_type: str | None
    occurred_at: datetime
    received_at: datetime
    dedupe_key: str
    summary: Mapping[str, Any] = field(default_factory=dict)


TokenBytes = Callable[[int], bytes]
_ACK_TOKEN_CACHE: dict[
    tuple[str, str, str, int, str], tuple[str, datetime]
] = {}
_ACK_TOKEN_CACHE_LOCK = threading.Lock()


def _path_value(db_path: PathValue | None) -> str:
    return get_supa_db_path() if db_path is None else os.fspath(Path(db_path))


def _connect(db_path: PathValue | None) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(
            _path_value(db_path),
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection
    except sqlite3.Error as error:
        raise AgentEventStorageError("The event journal is unavailable.") from error


@contextmanager
def _write_transaction(db_path: PathValue | None) -> Iterator[sqlite3.Connection]:
    connection = _connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except AgentEventStorageError:
        try:
            connection.rollback()
        except sqlite3.Error:
            pass
        raise
    except sqlite3.Error as error:
        try:
            connection.rollback()
        except sqlite3.Error:
            pass
        raise AgentEventStorageError("The event journal could not be updated.") from error
    except BaseException:
        try:
            connection.rollback()
        except sqlite3.Error:
            pass
        raise
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass


def _aware_utc(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a timezone-aware datetime")
    try:
        offset = value.utcoffset()
        if offset is None:
            raise ValueError(f"{field_name} must be a timezone-aware datetime")
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field_name} must be a valid datetime") from error


def _now(clock: Clock | None) -> datetime:
    return _aware_utc(
        datetime.now(timezone.utc) if clock is None else clock(),
        field_name="clock value",
    )


def _timestamp(value: datetime) -> str:
    normalized = _aware_utc(value, field_name="timestamp")
    # Fixed-width UTC timestamps preserve chronological ordering under SQLite's
    # indexed TEXT comparisons, including values either side of a microsecond
    # retention boundary.
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _payload_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return _aware_utc(parsed, field_name="payload timestamp")
    except (OverflowError, ValueError):
        return None


def _numeric_id(value: Any, *, required: bool, field_name: str) -> str | None:
    if isinstance(value, bool):
        normalized = None
    elif isinstance(value, int):
        normalized = str(value)
    elif isinstance(value, str):
        normalized = value.strip()
    else:
        normalized = None
    if normalized is not None and _NUMERIC_ID_RE.fullmatch(normalized):
        return normalized
    if required:
        raise ValueError(f"{field_name} must be a positive numeric ID")
    return None


def _contains_secret(value: str) -> bool:
    return any(
        pattern.search(value)
        for pattern in (
            _BEARER_VALUE_RE,
            _SECRET_VALUE_RE,
            _SECRET_ASSIGNMENT_RE,
            _SUPA_TOKEN_VALUE_RE,
        )
    )


def _identifier(value: Any, *, required: bool, field_name: str) -> str | None:
    normalized = value.strip() if isinstance(value, str) else None
    if (
        normalized
        and len(normalized) <= MAX_IDENTIFIER_CHARS
        and _IDENTIFIER_RE.fullmatch(normalized)
        and not _contains_secret(normalized)
    ):
        return normalized
    if required:
        raise ValueError(f"{field_name} must be a safe identifier")
    return None


def _event_type(value: Any, *, prefix: str | None = None) -> str:
    if not isinstance(value, str):
        raise ValueError("event type must be a string")
    normalized = value.strip().lower()
    if (
        not normalized
        or len(normalized) > MAX_EVENT_TYPE_CHARS
        or not _EVENT_TYPE_RE.fullmatch(normalized)
        or _contains_secret(normalized)
        or (prefix is not None and not normalized.startswith(prefix))
    ):
        raise ValueError("event type is invalid")
    return normalized


def _normalized_help_scout_type(raw_event_type: str) -> str:
    if not isinstance(raw_event_type, str):
        raise ValueError("Help Scout event type must be a string")
    raw = raw_event_type.strip().lower()
    known = _HELP_SCOUT_EVENT_TYPES.get(raw)
    if known is not None:
        return known
    if not _EVENT_TYPE_RE.fullmatch(raw):
        raise ValueError("Help Scout event type is invalid")
    if raw.startswith("convo."):
        return _event_type(
            f"helpscout.conversation.{raw.removeprefix('convo.')}",
            prefix="helpscout.",
        )
    return _event_type(f"helpscout.{raw}", prefix="helpscout.")


def _threads(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    embedded = payload.get("_embedded")
    if not isinstance(embedded, Mapping):
        return []
    raw_threads = embedded.get("threads")
    if not isinstance(raw_threads, list):
        return []
    return [thread for thread in raw_threads if isinstance(thread, Mapping)]


def _created_by_type(thread: Mapping[str, Any]) -> str | None:
    created_by = thread.get("createdBy")
    if not isinstance(created_by, Mapping):
        return None
    actor_type = created_by.get("type")
    return actor_type if actor_type in _ACTOR_TYPES else None


def _is_customer_thread(thread: Mapping[str, Any]) -> bool:
    return thread.get("type") == "customer" or _created_by_type(thread) == "customer"


def _is_agent_thread(thread: Mapping[str, Any]) -> bool:
    return (
        thread.get("type") == "message"
        and thread.get("state") != "draft"
        and _created_by_type(thread) in {"user", "system_user"}
    )


def _latest_thread(candidates: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not candidates:
        return None
    timestamped: list[tuple[datetime, int, int, Mapping[str, Any]]] = []
    for position, thread in enumerate(candidates):
        created_at = _payload_timestamp(thread.get("createdAt"))
        if created_at is None:
            continue
        raw_id = _numeric_id(thread.get("id"), required=False, field_name="thread ID")
        numeric_id = int(raw_id) if raw_id is not None else -1
        timestamped.append((created_at, numeric_id, -position, thread))
    if timestamped:
        return max(timestamped, key=lambda item: item[:3])[3]
    # Help Scout normally returns embedded threads newest-first.  Keeping the
    # first matching item is the only deterministic choice without timestamps.
    return candidates[0]


def _affected_thread(
    raw_event_type: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    candidates = _threads(payload)
    if raw_event_type in {"", "convo.created", "convo.customer.reply.created"}:
        return _latest_thread([thread for thread in candidates if _is_customer_thread(thread)])
    if raw_event_type == "convo.note.created":
        return _latest_thread(
            [thread for thread in candidates if thread.get("type") == "note"]
        )
    if raw_event_type == "convo.agent.reply.created":
        return _latest_thread([thread for thread in candidates if _is_agent_thread(thread)])
    return None


def _canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    try:
        canonical = json.dumps(
            payload,
            allow_nan=False,
            check_circular=True,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (RecursionError, TypeError, UnicodeError, ValueError) as error:
        raise ValueError("webhook payload is not canonical JSON") from error
    return hashlib.sha256(canonical).hexdigest()


def _digest_key(prefix: str, parts: list[str]) -> str:
    canonical = json.dumps(
        parts,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return f"{prefix}:{hashlib.sha256(canonical).hexdigest()}"


def _safe_code(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if (
        not normalized
        or len(normalized) > MAX_SUMMARY_TEXT_CHARS
        or not _EVENT_TYPE_RE.fullmatch(normalized)
        or _contains_secret(normalized)
    ):
        return None
    return normalized


def _safe_summary_identifier(value: Any) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    return _identifier(value, required=False, field_name="summary identifier")


def _safe_summary_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return None
    redacted = _BEARER_VALUE_RE.sub("Bearer [redacted]", value)
    redacted = _SECRET_VALUE_RE.sub("[redacted]", redacted)
    redacted = _SECRET_ASSIGNMENT_RE.sub("[redacted]", redacted)
    redacted = _SUPA_TOKEN_VALUE_RE.sub("[redacted]", redacted).strip()
    if not redacted:
        return None
    if len(redacted) > MAX_SUMMARY_TEXT_CHARS:
        redacted = redacted[: MAX_SUMMARY_TEXT_CHARS - 3] + "..."
    return redacted


def _sanitize_summary(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("event summary must be a mapping")
    sanitized: dict[str, Any] = {}
    for key in sorted(key for key in value if isinstance(key, str)):
        if key not in _SUMMARY_KEYS:
            continue
        raw = value[key]
        if key in _SUMMARY_CODE_KEYS:
            clean = _safe_code(raw)
        elif key in _SUMMARY_IDENTIFIER_KEYS:
            clean = _safe_summary_identifier(raw)
        elif key in _SUMMARY_NUMERIC_ID_KEYS:
            clean = _numeric_id(raw, required=False, field_name=key)
        elif key in _SUMMARY_COUNT_KEYS:
            clean = (
                raw
                if isinstance(raw, int)
                and not isinstance(raw, bool)
                and 0 <= raw <= 1_000_000
                else None
            )
        else:
            clean = _safe_summary_text(raw)
        if clean is not None:
            sanitized[key] = clean
    try:
        encoded = json.dumps(
            sanitized,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (RecursionError, TypeError, UnicodeError, ValueError) as error:
        raise ValueError("event summary is invalid") from error
    if len(encoded) > MAX_SUMMARY_JSON_BYTES:
        raise ValueError("event summary is too large")
    return sanitized


def _help_scout_summary(
    payload: Mapping[str, Any],
    thread: Mapping[str, Any] | None,
    event_type: str | None = None,
) -> dict[str, Any]:
    mailbox_id = payload.get("mailboxId")
    if mailbox_id is None:
        mailbox = payload.get("mailbox")
        if isinstance(mailbox, Mapping):
            mailbox_id = mailbox.get("id")
    summary: dict[str, Any] = {
        "conversation_number": payload.get("number"),
        "mailbox_id": mailbox_id,
        "status": payload.get("status"),
    }
    tags = payload.get("tags")
    if isinstance(tags, list):
        summary["tag_count"] = len(tags)
    if thread is not None:
        summary["thread_type"] = thread.get("type")
        summary["thread_state"] = thread.get("state")
        if event_type == "helpscout.note.created":
            # Lets consumers tell API-written notes from people's own notes.
            source = thread.get("source")
            if isinstance(source, Mapping):
                summary["source_type"] = source.get("type")
        embedded = thread.get("_embedded")
        if isinstance(embedded, Mapping) and isinstance(embedded.get("attachments"), list):
            summary["attachment_count"] = len(embedded["attachments"])
    return _sanitize_summary(summary)


def normalize_webhook_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    clock: Clock | None = None,
) -> AgentEvent:
    """Normalize one full Help Scout conversation webhook into a compact event."""

    if not isinstance(payload, dict):
        raise ValueError("webhook payload must be an object")
    raw_event_type = event_type.strip().lower() if isinstance(event_type, str) else event_type
    normalized_type = _normalized_help_scout_type(event_type)
    conversation_id = _numeric_id(
        payload.get("id"),
        required=True,
        field_name="conversation ID",
    )
    received_at = _now(clock)
    thread = _affected_thread(raw_event_type, payload)
    thread_id = (
        _numeric_id(thread.get("id"), required=False, field_name="thread ID")
        if thread is not None
        else None
    )

    identity_time = None
    if thread is not None:
        identity_time = _payload_timestamp(thread.get("createdAt"))
    elif raw_event_type in {"", "convo.created"}:
        identity_time = _payload_timestamp(payload.get("createdAt"))
    else:
        identity_time = _payload_timestamp(payload.get("updatedAt"))

    occurred_at = identity_time
    if occurred_at is None:
        occurred_at = _payload_timestamp(payload.get("updatedAt"))
    if occurred_at is None:
        occurred_at = _payload_timestamp(payload.get("createdAt"))
    if occurred_at is None:
        occurred_at = received_at

    identity_parts = [normalized_type, conversation_id]
    if thread_id is not None:
        identity_parts.extend(
            [
                thread_id,
                _timestamp(identity_time) if identity_time is not None else "",
            ]
        )
    else:
        identity_parts.extend(
            [
                _timestamp(identity_time) if identity_time is not None else "",
                _canonical_payload_hash(payload),
            ]
        )

    actor_type = _created_by_type(thread) if thread is not None else None
    if actor_type is None:
        created_by = payload.get("createdBy")
        if isinstance(created_by, Mapping) and created_by.get("type") in _ACTOR_TYPES:
            actor_type = created_by["type"]

    return AgentEvent(
        type=normalized_type,
        conversation_id=conversation_id,
        thread_id=thread_id,
        run_id=None,
        source="helpscout",
        actor_type=actor_type,
        occurred_at=occurred_at,
        received_at=received_at,
        dedupe_key=_digest_key("helpscout", identity_parts),
        summary=_help_scout_summary(payload, thread, normalized_type),
    )


def _validated_event(event: AgentEvent) -> tuple[Any, ...]:
    if not isinstance(event, AgentEvent):
        raise ValueError("event must be an AgentEvent")
    source = event.source
    if source not in {"helpscout", "supa"}:
        raise ValueError("event source is invalid")
    normalized_type = _event_type(event.type, prefix=f"{source}.")
    conversation_id = (
        _numeric_id(event.conversation_id, required=True, field_name="conversation ID")
        if event.conversation_id is not None
        else None
    )
    thread_id = (
        _numeric_id(event.thread_id, required=True, field_name="thread ID")
        if event.thread_id is not None
        else None
    )
    run_id = (
        _identifier(event.run_id, required=True, field_name="run ID")
        if event.run_id is not None
        else None
    )
    if event.actor_type is not None and event.actor_type not in _ACTOR_TYPES:
        raise ValueError("event actor type is invalid")
    if not isinstance(event.dedupe_key, str) or not _DEDUPE_KEY_RE.fullmatch(event.dedupe_key):
        raise ValueError("event dedupe key is invalid")
    summary = _sanitize_summary(event.summary)
    summary_json = json.dumps(
        summary,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        normalized_type,
        conversation_id,
        thread_id,
        run_id,
        source,
        event.actor_type,
        _timestamp(event.occurred_at),
        _timestamp(event.received_at),
        event.dedupe_key,
        summary_json,
    )


def _stored_optional_numeric_id(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = _numeric_id(value, required=True, field_name=field_name)
    if value != normalized:
        raise ValueError(f"stored {field_name} is not canonical")
    return normalized


def _stored_optional_identifier(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = _identifier(value, required=True, field_name=field_name)
    if value != normalized:
        raise ValueError(f"stored {field_name} is not canonical")
    return normalized


def _stored_timestamp(value: Any, *, field_name: str) -> str:
    parsed = _payload_timestamp(value)
    if parsed is None:
        raise ValueError(f"stored {field_name} is invalid")
    normalized = _timestamp(parsed)
    if value != normalized:
        raise ValueError(f"stored {field_name} is not canonical")
    return normalized


def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        summary = json.loads(row["summary_json"])
        if not isinstance(summary, dict) or _sanitize_summary(summary) != summary:
            raise ValueError("stored summary is invalid")
        event_id = row["id"]
        if not isinstance(event_id, int) or isinstance(event_id, bool) or event_id <= 0:
            raise ValueError("stored event ID is invalid")
        source = row["source"]
        if source not in {"helpscout", "supa"}:
            raise ValueError("stored source is invalid")
        event_type = _event_type(row["event_type"], prefix=f"{source}.")
        if row["event_type"] != event_type:
            raise ValueError("stored event type is not canonical")
        conversation_id = _stored_optional_numeric_id(
            row["conversation_id"],
            field_name="conversation ID",
        )
        thread_id = _stored_optional_numeric_id(
            row["thread_id"],
            field_name="thread ID",
        )
        run_id = _stored_optional_identifier(row["run_id"], field_name="run ID")
        actor_type = row["actor_type"]
        if actor_type is not None and actor_type not in _ACTOR_TYPES:
            raise ValueError("stored actor type is invalid")
        dedupe_key = row["dedupe_key"]
        if (
            not isinstance(dedupe_key, str)
            or not _DEDUPE_KEY_RE.fullmatch(dedupe_key)
            or not dedupe_key.startswith(f"{source}:")
        ):
            raise ValueError("stored dedupe key is invalid")
        occurred_at = _stored_timestamp(row["occurred_at"], field_name="occurred_at")
        received_at = _stored_timestamp(row["received_at"], field_name="received_at")
    except (
        IndexError,
        KeyError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as error:
        raise AgentEventStorageError("The event journal contains invalid data.") from error
    return {
        "event_id": event_id,
        "type": event_type,
        "conversation_id": conversation_id,
        "thread_id": thread_id,
        "run_id": run_id,
        "source": source,
        "actor_type": actor_type,
        "occurred_at": occurred_at,
        "received_at": received_at,
        "summary": summary,
    }


def _protocol_identifier(value: Any, *, field_name: str) -> str:
    normalized = _identifier(value, required=True, field_name=field_name)
    assert normalized is not None
    return normalized


def _consumer_name(value: Any) -> str:
    return _protocol_identifier(value, field_name="consumer name")


def _audit_name(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    try:
        normalized.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError(f"{field_name} is invalid") from error
    if (
        not normalized
        or len(normalized) > MAX_SUMMARY_TEXT_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
        or _contains_secret(normalized)
    ):
        raise ValueError(f"{field_name} is invalid")
    return normalized


def _event_id(value: Any, *, field_name: str = "event ID") -> int:
    normalized = _numeric_id(value, required=True, field_name=field_name)
    assert normalized is not None
    return int(normalized)


def _parse_start(value: str | int) -> tuple[str, int | None]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"now", "oldest"}:
            return normalized, None
    return "event_id", _event_id(value, field_name="start event ID")


def _journal_bounds(connection: sqlite3.Connection) -> tuple[int | None, int]:
    row = connection.execute(
        "SELECT MIN(id) AS oldest_id FROM agent_events"
    ).fetchone()
    oldest = row["oldest_id"]
    if oldest is not None and (
        not isinstance(oldest, int) or isinstance(oldest, bool) or oldest <= 0
    ):
        raise AgentEventStorageError("The event journal contains an invalid event ID.")
    sequence_row = connection.execute(
        "SELECT seq FROM sqlite_sequence WHERE name = 'agent_events'"
    ).fetchone()
    head = sequence_row["seq"] if sequence_row is not None else 0
    if not isinstance(head, int) or isinstance(head, bool) or head < 0:
        raise AgentEventStorageError("The event journal head is invalid.")
    return oldest, head


def _cursor_expired_error(
    connection: sqlite3.Connection,
) -> AgentEventCursorExpiredError:
    oldest, head = _journal_bounds(connection)
    return AgentEventCursorExpiredError(
        oldest_retained_event_id=oldest,
        current_head_event_id=head,
    )


def _resolve_start(
    connection: sqlite3.Connection,
    start: str | int,
) -> tuple[str, int]:
    mode, explicit_event_id = _parse_start(start)
    oldest, head = _journal_bounds(connection)
    if mode == "now":
        return mode, head
    if mode == "oldest":
        return mode, oldest - 1 if oldest is not None else head

    assert explicit_event_id is not None
    retained = connection.execute(
        "SELECT 1 FROM agent_events WHERE id = ?",
        (explicit_event_id,),
    ).fetchone()
    if retained is not None:
        return mode, explicit_event_id - 1
    if explicit_event_id <= head:
        raise _cursor_expired_error(connection)
    raise AgentEventStartError(explicit_event_id, head)


def _stored_consumer_timestamp(value: Any, *, field_name: str) -> datetime:
    normalized = _stored_timestamp(value, field_name=field_name)
    parsed = _payload_timestamp(normalized)
    assert parsed is not None
    return parsed


def _optional_positive_stored_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise AgentEventStorageError(f"The stored {field_name} is invalid.")
    return value


def _optional_token_hash(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _TOKEN_HASH_RE.fullmatch(value):
        raise AgentEventStorageError(f"The stored {field_name} is invalid.")
    return value


def _validated_consumer_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        row_id = row["id"]
        if not isinstance(row_id, int) or isinstance(row_id, bool) or row_id <= 0:
            raise ValueError("consumer row ID is invalid")
        token_id = _protocol_identifier(row["token_id"], field_name="token ID")
        consumer_name = _consumer_name(row["consumer_name"])
        if token_id != row["token_id"] or consumer_name != row["consumer_name"]:
            raise ValueError("consumer identity is not canonical")
        acknowledged_event_id = row["acknowledged_event_id"]
        if (
            not isinstance(acknowledged_event_id, int)
            or isinstance(acknowledged_event_id, bool)
            or acknowledged_event_id < 0
        ):
            raise ValueError("consumer cursor is invalid")
        leased_event_id = _optional_positive_stored_int(
            row["leased_event_id"],
            field_name="leased event ID",
        )
        lease_token_hash = _optional_token_hash(
            row["lease_token_hash"],
            field_name="lease token hash",
        )
        raw_expiry = row["lease_expires_at"]
        lease_expires_at = (
            _stored_consumer_timestamp(raw_expiry, field_name="lease expiry")
            if raw_expiry is not None
            else None
        )
        if (leased_event_id is None) != (lease_expires_at is None) or (
            leased_event_id is None
        ) != (lease_token_hash is None):
            raise ValueError("consumer lease fields are inconsistent")
        if leased_event_id is not None and leased_event_id <= acknowledged_event_id:
            raise ValueError("consumer lease does not follow its cursor")
        last_ack_event_id = _optional_positive_stored_int(
            row["last_ack_event_id"],
            field_name="last acknowledged event ID",
        )
        last_ack_token_hash = _optional_token_hash(
            row["last_ack_token_hash"],
            field_name="last acknowledgement token hash",
        )
        if (last_ack_event_id is None) != (last_ack_token_hash is None):
            raise ValueError("last acknowledgement fields are inconsistent")
        if (
            last_ack_event_id is not None
            and last_ack_event_id > acknowledged_event_id
        ):
            raise ValueError("last acknowledgement is beyond the cursor")
        start_mode = row["start_mode"]
        if start_mode not in {"now", "oldest", "event_id", "cursor_expired"}:
            raise ValueError("consumer start mode is invalid")
        if start_mode == "cursor_expired" and leased_event_id is not None:
            raise ValueError("expired cursor retains an active lease")
        created_at = _stored_consumer_timestamp(
            row["created_at"],
            field_name="consumer created_at",
        )
        updated_at = _stored_consumer_timestamp(
            row["updated_at"],
            field_name="consumer updated_at",
        )
    except (IndexError, KeyError, TypeError, UnicodeError, ValueError) as error:
        raise AgentEventStorageError(
            "The event consumer store contains invalid data."
        ) from error
    return {
        "id": row_id,
        "token_id": token_id,
        "consumer_name": consumer_name,
        "acknowledged_event_id": acknowledged_event_id,
        "leased_event_id": leased_event_id,
        "lease_expires_at": lease_expires_at,
        "lease_token_hash": lease_token_hash,
        "last_ack_event_id": last_ack_event_id,
        "last_ack_token_hash": last_ack_token_hash,
        "start_mode": start_mode,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _select_consumer(
    connection: sqlite3.Connection,
    *,
    token_id: str,
    consumer_name: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT * FROM agent_event_consumers
        WHERE token_id = ? AND consumer_name = ?
        """,
        (token_id, consumer_name),
    ).fetchone()


def _ack_cache_key(
    db_path: PathValue | None,
    *,
    token_id: str,
    consumer_name: str,
    event_id: int,
    token_hash: str,
) -> tuple[str, str, str, int, str]:
    return (
        os.path.abspath(_path_value(db_path)),
        token_id,
        consumer_name,
        event_id,
        token_hash,
    )


def _cache_ack_token(
    db_path: PathValue | None,
    *,
    token_id: str,
    consumer_name: str,
    event_id: int,
    token_hash: str,
    ack_token: str,
    expires_at: datetime,
) -> None:
    key = _ack_cache_key(
        db_path,
        token_id=token_id,
        consumer_name=consumer_name,
        event_id=event_id,
        token_hash=token_hash,
    )
    with _ACK_TOKEN_CACHE_LOCK:
        _ACK_TOKEN_CACHE[key] = (ack_token, expires_at)


def _cached_ack_token(
    db_path: PathValue | None,
    *,
    token_id: str,
    consumer_name: str,
    event_id: int,
    token_hash: str,
    now: datetime,
) -> str | None:
    key = _ack_cache_key(
        db_path,
        token_id=token_id,
        consumer_name=consumer_name,
        event_id=event_id,
        token_hash=token_hash,
    )
    with _ACK_TOKEN_CACHE_LOCK:
        cached = _ACK_TOKEN_CACHE.get(key)
        if cached is None:
            return None
        ack_token, expires_at = cached
        if expires_at <= now:
            _ACK_TOKEN_CACHE.pop(key, None)
            return None
        return ack_token


def _clear_cached_ack_tokens(
    db_path: PathValue | None,
    *,
    token_id: str,
    consumer_name: str,
) -> None:
    scope = (os.path.abspath(_path_value(db_path)), token_id, consumer_name)
    with _ACK_TOKEN_CACHE_LOCK:
        for key in tuple(_ACK_TOKEN_CACHE):
            if key[:3] == scope:
                _ACK_TOKEN_CACHE.pop(key, None)


def _new_ack_token(token_bytes: TokenBytes | None) -> tuple[str, str]:
    source = secrets.token_bytes if token_bytes is None else token_bytes
    raw = source(ACK_TOKEN_BYTES)
    if not isinstance(raw, bytes) or len(raw) < ACK_TOKEN_BYTES:
        raise ValueError("ack token source must return at least 32 bytes")
    token = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    return token, hashlib.sha256(token.encode("ascii")).hexdigest()


def _ack_token_hash(value: Any) -> str:
    if not isinstance(value, str) or not _ACK_TOKEN_RE.fullmatch(value):
        raise AgentEventAckError()
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _lease_result(
    row: sqlite3.Row,
    *,
    ack_token: str,
    lease_expires_at: datetime,
) -> dict[str, Any]:
    return {
        **_event_from_row(row),
        "ack_token": ack_token,
        "lease_expires_at": _timestamp(lease_expires_at),
    }


def append_event(db_path: PathValue | None, event: AgentEvent) -> dict[str, Any]:
    """Append one event or return the existing row for the same dedupe key."""

    with _write_transaction(db_path) as connection:
        return append_event_in_transaction(connection, event)


def append_event_in_transaction(
    connection: sqlite3.Connection, event: AgentEvent
) -> dict[str, Any]:
    """Append an event within a caller-owned SQLite transaction."""
    values = _validated_event(event)
    connection.execute(
            """
            INSERT INTO agent_events (
                event_type, conversation_id, thread_id, run_id, source,
                actor_type, occurred_at, received_at, dedupe_key, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dedupe_key) DO NOTHING
            """,
            values,
        )
    row = connection.execute(
        "SELECT * FROM agent_events WHERE dedupe_key = ?",
        (event.dedupe_key,),
    ).fetchone()
    if row is None:
        raise AgentEventStorageError("The event journal did not retain the event.")
    return _event_from_row(row)


def append_supa_event(
    db_path: PathValue | None,
    *,
    type: str,
    dedupe_key: str,
    conversation_id: int | str | None = None,
    thread_id: int | str | None = None,
    run_id: str | None = None,
    actor_type: str = "system",
    occurred_at: datetime | None = None,
    summary: Mapping[str, Any] | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Normalize and append one Supa-side outcome event."""

    return append_event(
        db_path,
        _supa_event(
            type=type, dedupe_key=dedupe_key,
            conversation_id=conversation_id, thread_id=thread_id,
            run_id=run_id, actor_type=actor_type,
            occurred_at=occurred_at, summary=summary, clock=clock,
        ),
    )


def append_supa_event_in_transaction(
    connection: sqlite3.Connection,
    *,
    type: str,
    dedupe_key: str,
    conversation_id: int | str | None = None,
    run_id: str | None = None,
    summary: Mapping[str, Any] | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Atomically persist an outcome ping with a caller's state transition."""
    return append_event_in_transaction(
        connection,
        _supa_event(
            type=type, dedupe_key=dedupe_key,
            conversation_id=conversation_id, run_id=run_id,
            summary=summary, clock=clock,
        ),
    )


def _supa_event(
    *,
    type: str,
    dedupe_key: str,
    conversation_id: int | str | None = None,
    thread_id: int | str | None = None,
    run_id: str | None = None,
    actor_type: str = "system",
    occurred_at: datetime | None = None,
    summary: Mapping[str, Any] | None = None,
    clock: Clock | None = None,
) -> AgentEvent:

    normalized_type = _event_type(type, prefix="supa.")
    normalized_conversation_id = (
        _numeric_id(conversation_id, required=True, field_name="conversation ID")
        if conversation_id is not None
        else None
    )
    normalized_thread_id = (
        _numeric_id(thread_id, required=True, field_name="thread ID")
        if thread_id is not None
        else None
    )
    normalized_run_id = (
        _identifier(run_id, required=True, field_name="run ID")
        if run_id is not None
        else None
    )
    if (
        not isinstance(dedupe_key, str)
        or not dedupe_key
        or len(dedupe_key) > MAX_DEDUPE_INPUT_CHARS
    ):
        raise ValueError("dedupe_key is invalid")
    try:
        dedupe_key.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError("dedupe_key is invalid") from error
    received_at = _now(clock)
    occurred = received_at if occurred_at is None else _aware_utc(
        occurred_at,
        field_name="occurred_at",
    )
    durable_dedupe_key = _digest_key(
        "supa",
        [
            normalized_type,
            normalized_conversation_id or "",
            normalized_thread_id or "",
            normalized_run_id or "",
            dedupe_key,
        ],
    )
    return AgentEvent(
            type=normalized_type,
            conversation_id=normalized_conversation_id,
            thread_id=normalized_thread_id,
            run_id=normalized_run_id,
            source="supa",
            actor_type=actor_type,
            occurred_at=occurred,
            received_at=received_at,
            dedupe_key=durable_dedupe_key,
            summary=_sanitize_summary(summary),
        )


def _allowed_conversation_texts(
    conversations: Iterable[int] | None,
) -> tuple[str, ...] | None:
    """Normalize a token's conversation allowlist to the stored text ids."""
    if conversations is None:
        return None
    allowed: set[str] = set()
    for conversation_id in conversations:
        if isinstance(conversation_id, bool) or not isinstance(conversation_id, int):
            raise ValueError("conversation allowlist is invalid")
        if conversation_id <= 0:
            raise ValueError("conversation allowlist is invalid")
        allowed.add(str(conversation_id))
    return tuple(sorted(allowed))


def lease_next_event(
    db_path: PathValue | None,
    *,
    token_id: str,
    consumer_name: str,
    start: str | int = "now",
    clock: Clock | None = None,
    token_bytes: TokenBytes | None = None,
    conversations: Iterable[int] | None = None,
) -> dict[str, Any] | None:
    """Lease the consumer's next event without holding a connection while idle.

    With ``conversations`` (a restricted token's allowlist) only events whose
    conversation id is listed are leased; events without a conversation id and
    events for any other conversation are skipped. The consumer cursor still
    advances past them on acknowledgement.
    """

    normalized_token_id = _protocol_identifier(token_id, field_name="token ID")
    allowed_conversations = _allowed_conversation_texts(conversations)
    normalized_consumer_name = _consumer_name(consumer_name)
    # Parse every call so malformed API input is rejected consistently, but the
    # selected start only applies when the token-scoped consumer is first made.
    _parse_start(start)
    now = _now(clock)
    now_text = _timestamp(now)

    with _write_transaction(db_path) as connection:
        row = _select_consumer(
            connection,
            token_id=normalized_token_id,
            consumer_name=normalized_consumer_name,
        )
        if row is None:
            start_mode, cursor = _resolve_start(connection, start)
            connection.execute(
                """
                INSERT INTO agent_event_consumers (
                    token_id, consumer_name, acknowledged_event_id,
                    start_mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_token_id,
                    normalized_consumer_name,
                    cursor,
                    start_mode,
                    now_text,
                    now_text,
                ),
            )
            row = _select_consumer(
                connection,
                token_id=normalized_token_id,
                consumer_name=normalized_consumer_name,
            )
            if row is None:
                raise AgentEventStorageError(
                    "The event consumer could not be initialized."
                )

        consumer = _validated_consumer_row(row)
        if consumer["start_mode"] == "cursor_expired":
            raise _cursor_expired_error(connection)

        leased_event_id = consumer["leased_event_id"]
        if leased_event_id is not None:
            event_row = connection.execute(
                "SELECT * FROM agent_events WHERE id = ?",
                (leased_event_id,),
            ).fetchone()
            if event_row is None:
                connection.execute(
                    """
                    UPDATE agent_event_consumers
                    SET start_mode = 'cursor_expired', leased_event_id = NULL,
                        lease_expires_at = NULL, lease_token_hash = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now_text, consumer["id"]),
                )
                _clear_cached_ack_tokens(
                    db_path,
                    token_id=normalized_token_id,
                    consumer_name=normalized_consumer_name,
                )
                raise _cursor_expired_error(connection)

            assert consumer["lease_expires_at"] is not None
            assert consumer["lease_token_hash"] is not None
            if (
                allowed_conversations is not None
                and event_row["conversation_id"] not in allowed_conversations
            ):
                # The allowlist narrowed after this lease; never hand it out.
                # Replace the lease with the next allowed event below.
                event_row = None
                leased_event_id = None
            elif consumer["lease_expires_at"] > now:
                ack_token = _cached_ack_token(
                    db_path,
                    token_id=normalized_token_id,
                    consumer_name=normalized_consumer_name,
                    event_id=leased_event_id,
                    token_hash=consumer["lease_token_hash"],
                    now=now,
                )
                if ack_token is None:
                    # A restart loses the deliberately process-local raw
                    # capability. Keep the outstanding event, but rotate its
                    # persisted hash below so recovery never waits for expiry.
                    pass
                else:
                    return _lease_result(
                        event_row,
                        ack_token=ack_token,
                        lease_expires_at=consumer["lease_expires_at"],
                    )
        if leased_event_id is None:
            if allowed_conversations is None:
                event_row = connection.execute(
                    """
                    SELECT * FROM agent_events
                    WHERE id > ?
                    ORDER BY id
                    LIMIT 1
                    """,
                    (consumer["acknowledged_event_id"],),
                ).fetchone()
            elif not allowed_conversations:
                event_row = None
            else:
                placeholders = ", ".join("?" for _ in allowed_conversations)
                event_row = connection.execute(
                    f"""
                    SELECT * FROM agent_events
                    WHERE id > ? AND conversation_id IN ({placeholders})
                    ORDER BY id
                    LIMIT 1
                    """,
                    (consumer["acknowledged_event_id"], *allowed_conversations),
                ).fetchone()
            if event_row is None:
                if consumer["leased_event_id"] is not None:
                    # Drop the disallowed lease so it cannot be acknowledged.
                    connection.execute(
                        """
                        UPDATE agent_event_consumers
                        SET leased_event_id = NULL, lease_expires_at = NULL,
                            lease_token_hash = NULL, updated_at = ?
                        WHERE id = ?
                        """,
                        (now_text, consumer["id"]),
                    )
                    _clear_cached_ack_tokens(
                        db_path,
                        token_id=normalized_token_id,
                        consumer_name=normalized_consumer_name,
                    )
                return None
            leased_event_id = event_row["id"]

        ack_token, token_hash = _new_ack_token(token_bytes)
        try:
            lease_expires_at = now + timedelta(seconds=EVENT_LEASE_SECONDS)
        except OverflowError as error:
            raise ValueError("clock value cannot represent a complete lease") from error
        lease_expiry_text = _timestamp(lease_expires_at)
        updated = connection.execute(
            """
            UPDATE agent_event_consumers
            SET leased_event_id = ?, lease_expires_at = ?,
                lease_token_hash = ?, updated_at = ?
            WHERE id = ? AND acknowledged_event_id = ?
              AND leased_event_id IS ? AND lease_token_hash IS ?
            """,
            (
                leased_event_id,
                lease_expiry_text,
                token_hash,
                now_text,
                consumer["id"],
                consumer["acknowledged_event_id"],
                consumer["leased_event_id"],
                consumer["lease_token_hash"],
            ),
        )
        if updated.rowcount != 1:
            raise AgentEventStorageError("The event lease could not be persisted.")
        _clear_cached_ack_tokens(
            db_path,
            token_id=normalized_token_id,
            consumer_name=normalized_consumer_name,
        )
        _cache_ack_token(
            db_path,
            token_id=normalized_token_id,
            consumer_name=normalized_consumer_name,
            event_id=leased_event_id,
            token_hash=token_hash,
            ack_token=ack_token,
            expires_at=lease_expires_at,
        )
        return _lease_result(
            event_row,
            ack_token=ack_token,
            lease_expires_at=lease_expires_at,
        )


def acknowledge_event(
    db_path: PathValue | None,
    *,
    token_id: str,
    consumer_name: str,
    event_id: int | str,
    ack_token: str,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Acknowledge exactly the event and capability owned by one consumer."""

    normalized_token_id = _protocol_identifier(token_id, field_name="token ID")
    normalized_consumer_name = _consumer_name(consumer_name)
    normalized_event_id = _event_id(event_id)
    supplied_hash = _ack_token_hash(ack_token)
    now = _now(clock)
    now_text = _timestamp(now)

    with _write_transaction(db_path) as connection:
        row = _select_consumer(
            connection,
            token_id=normalized_token_id,
            consumer_name=normalized_consumer_name,
        )
        if row is None:
            raise AgentEventAckError()
        consumer = _validated_consumer_row(row)

        last_event_id = consumer["last_ack_event_id"]
        last_hash = consumer["last_ack_token_hash"]
        if (
            last_event_id == normalized_event_id
            and last_hash is not None
            and hmac.compare_digest(last_hash, supplied_hash)
        ):
            return {
                "acknowledged": True,
                "event_id": normalized_event_id,
                "replayed": True,
            }

        lease_hash = consumer["lease_token_hash"]
        if (
            consumer["start_mode"] == "cursor_expired"
            or consumer["leased_event_id"] != normalized_event_id
            or lease_hash is None
            or consumer["lease_expires_at"] is None
            or consumer["lease_expires_at"] <= now
            or not hmac.compare_digest(lease_hash, supplied_hash)
        ):
            raise AgentEventAckError()

        updated = connection.execute(
            """
            UPDATE agent_event_consumers
            SET acknowledged_event_id = ?, leased_event_id = NULL,
                lease_expires_at = NULL, lease_token_hash = NULL,
                last_ack_event_id = ?, last_ack_token_hash = ?, updated_at = ?
            WHERE id = ? AND leased_event_id = ? AND lease_token_hash = ?
            """,
            (
                normalized_event_id,
                normalized_event_id,
                supplied_hash,
                now_text,
                consumer["id"],
                normalized_event_id,
                lease_hash,
            ),
        )
        if updated.rowcount != 1:
            raise AgentEventAckError()
        _clear_cached_ack_tokens(
            db_path,
            token_id=normalized_token_id,
            consumer_name=normalized_consumer_name,
        )
        return {
            "acknowledged": True,
            "event_id": normalized_event_id,
            "replayed": False,
        }


def _insert_reset_audit(
    connection: sqlite3.Connection,
    *,
    token_id: str,
    token_name: str,
    consumer_name: str,
    request_id: str,
    from_cursor: int | None,
    to_cursor: int,
    start_mode: str,
    timestamp: str,
) -> None:
    request = {
        "consumer_name": consumer_name,
        "start_mode": start_mode,
        "to_acknowledged_event_id": to_cursor,
    }
    request_hash = hashlib.sha256(
        json.dumps(
            request,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    metadata = {
        "consumer_name": consumer_name,
        "from_acknowledged_event_id": from_cursor,
        "start_mode": start_mode,
        "to_acknowledged_event_id": to_cursor,
    }
    connection.execute(
        """
        INSERT INTO agent_audit (
            idempotency_key, token_id, token_name, scope, operation_name,
            target_id, request_id, request_hash, result, metadata_json,
            created_at
        ) VALUES (NULL, ?, ?, 'events', 'events.reset', ?, ?, ?,
                  'succeeded', ?, ?)
        """,
        (
            token_id,
            token_name,
            consumer_name,
            request_id,
            request_hash,
            json.dumps(
                metadata,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            timestamp,
        ),
    )


def reset_event_consumer(
    db_path: PathValue | None,
    *,
    token_id: str,
    token_name: str,
    consumer_name: str,
    to: str | int,
    request_id: str,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Explicitly move a token-scoped cursor and audit the destructive choice."""

    normalized_token_id = _protocol_identifier(token_id, field_name="token ID")
    normalized_token_name = _audit_name(token_name, field_name="token name")
    normalized_consumer_name = _consumer_name(consumer_name)
    normalized_request_id = _protocol_identifier(request_id, field_name="request ID")
    _parse_start(to)
    now = _now(clock)
    now_text = _timestamp(now)

    with _write_transaction(db_path) as connection:
        start_mode, cursor = _resolve_start(connection, to)
        row = _select_consumer(
            connection,
            token_id=normalized_token_id,
            consumer_name=normalized_consumer_name,
        )
        if row is None:
            from_cursor = None
            connection.execute(
                """
                INSERT INTO agent_event_consumers (
                    token_id, consumer_name, acknowledged_event_id,
                    start_mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_token_id,
                    normalized_consumer_name,
                    cursor,
                    start_mode,
                    now_text,
                    now_text,
                ),
            )
        else:
            consumer = _validated_consumer_row(row)
            from_cursor = consumer["acknowledged_event_id"]
            updated = connection.execute(
                """
                UPDATE agent_event_consumers
                SET acknowledged_event_id = ?, leased_event_id = NULL,
                    lease_expires_at = NULL, lease_token_hash = NULL,
                    last_ack_event_id = NULL, last_ack_token_hash = NULL,
                    start_mode = ?, updated_at = ?
                WHERE id = ?
                """,
                (cursor, start_mode, now_text, consumer["id"]),
            )
            if updated.rowcount != 1:
                raise AgentEventStorageError("The event cursor could not be reset.")

        _insert_reset_audit(
            connection,
            token_id=normalized_token_id,
            token_name=normalized_token_name,
            consumer_name=normalized_consumer_name,
            request_id=normalized_request_id,
            from_cursor=from_cursor,
            to_cursor=cursor,
            start_mode=start_mode,
            timestamp=now_text,
        )
        _clear_cached_ack_tokens(
            db_path,
            token_id=normalized_token_id,
            consumer_name=normalized_consumer_name,
        )
        return {
            "acknowledged_event_id": cursor,
            "consumer_name": normalized_consumer_name,
            "start_mode": start_mode,
        }


def prune_events(
    db_path: PathValue | None,
    *,
    before: datetime,
    clock: Clock | None = None,
) -> int:
    """Delete journal rows received before the caller's retention cutoff."""

    cutoff = _timestamp(_aware_utc(before, field_name="before"))
    timestamp = _timestamp(_now(clock))
    expired_consumers: list[tuple[str, str]] = []
    with _write_transaction(db_path) as connection:
        expired_consumers = [
            (row["token_id"], row["consumer_name"])
            for row in connection.execute(
                """
                SELECT token_id, consumer_name
                FROM agent_event_consumers AS consumer
                WHERE consumer.start_mode != 'cursor_expired'
                  AND EXISTS (
                      SELECT 1
                      FROM agent_events AS event
                      WHERE event.received_at < ?
                        AND event.id > consumer.acknowledged_event_id
                  )
                """,
                (cutoff,),
            ).fetchall()
        ]
        connection.execute(
            """
            UPDATE agent_event_consumers AS consumer
            SET start_mode = 'cursor_expired', leased_event_id = NULL,
                lease_expires_at = NULL, lease_token_hash = NULL,
                updated_at = ?
            WHERE consumer.start_mode != 'cursor_expired'
              AND EXISTS (
                  SELECT 1
                  FROM agent_events AS event
                  WHERE event.received_at < ?
                    AND event.id > consumer.acknowledged_event_id
              )
            """,
            (timestamp, cutoff),
        )
        cursor = connection.execute(
            "DELETE FROM agent_events WHERE received_at < ?",
            (cutoff,),
        )
        deleted = cursor.rowcount
    for token_id, consumer_name in expired_consumers:
        _clear_cached_ack_tokens(
            db_path,
            token_id=token_id,
            consumer_name=consumer_name,
        )
    return deleted


def _retention_configuration(
    retention_days: int,
    interval_seconds: float,
) -> tuple[int, float]:
    if (
        not isinstance(retention_days, int)
        or isinstance(retention_days, bool)
        or not 1 <= retention_days <= 3_650
    ):
        raise ValueError("retention_days must be between 1 and 3650")
    if (
        not isinstance(interval_seconds, (int, float))
        or isinstance(interval_seconds, bool)
        or not math.isfinite(interval_seconds)
        or not 1 <= interval_seconds <= 604_800
    ):
        raise ValueError("interval_seconds must be between 1 and 604800")
    return retention_days, float(interval_seconds)


async def run_event_retention_worker(
    db_path: PathValue | None,
    *,
    retention_days: int = DEFAULT_EVENT_RETENTION_DAYS,
    interval_seconds: float = DEFAULT_EVENT_RETENTION_INTERVAL_SECONDS,
    clock: Clock | None = None,
    sleep: Sleep = asyncio.sleep,
    prune: Callable[..., int] | None = None,
) -> None:
    """Prune once immediately, then periodically without holding SQLite while idle."""

    days, interval = _retention_configuration(retention_days, interval_seconds)
    selected_path = _path_value(db_path)
    pruner = prune_events if prune is None else prune
    while True:
        try:
            now = _now(clock)
            cutoff = now - timedelta(days=days)
            await asyncio.to_thread(
                pruner,
                selected_path,
                before=cutoff,
                clock=lambda now=now: now,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(
                "Event retention prune failed exception_type=%s",
                type(error).__name__,
            )
        await sleep(interval)


def schedule_event_retention_worker(
    db_path: PathValue | None,
    *,
    retention_days: int = DEFAULT_EVENT_RETENTION_DAYS,
    interval_seconds: float = DEFAULT_EVENT_RETENTION_INTERVAL_SECONDS,
    clock: Clock | None = None,
    sleep: Sleep = asyncio.sleep,
) -> asyncio.Task[None]:
    """Create the single application-owned retention worker task."""

    _retention_configuration(retention_days, interval_seconds)
    return asyncio.create_task(
        run_event_retention_worker(
            db_path,
            retention_days=retention_days,
            interval_seconds=interval_seconds,
            clock=clock,
            sleep=sleep,
        ),
        name="agent-event-retention",
    )
