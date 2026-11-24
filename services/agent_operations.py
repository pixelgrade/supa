"""Durable idempotency records and append-only audit entries for agent writes.

The operation ledger deliberately separates request identity from request data:
only a canonical SHA-256 request hash and a small sanitized metadata summary are
persisted.  Every state change uses ``BEGIN IMMEDIATE`` so separate processes
cannot both win a reservation or transition the same operation concurrently.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping

from services.agent_auth import AgentIdentity
from services.runtime_paths import get_supa_db_path


Clock = Callable[[], datetime]
PathValue = str | os.PathLike[str]
OperationState = Literal[
    "reserved",
    "in_progress",
    "succeeded",
    "sent_with_warning",
    "failed",
    "outcome_unknown",
]

# Five minutes is longer than ordinary Help Scout request timeouts while still
# surfacing abandoned work promptly after a restart. Operators may tune this
# through SUPA_AGENT_OPERATION_STALE_SECONDS without changing code.
DEFAULT_STALE_AFTER_SECONDS = 300.0
MIN_STALE_AFTER_SECONDS = 1.0
MAX_STALE_AFTER_SECONDS = 86_400.0
STALE_AFTER_ENV = "SUPA_AGENT_OPERATION_STALE_SECONDS"

MAX_PERSISTED_JSON_BYTES = 16 * 1024
MAX_IDEMPOTENCY_KEY_LENGTH = 200
MAX_NAME_LENGTH = 200
MAX_TARGET_LENGTH = 200
MAX_REQUEST_ID_LENGTH = 200
MAX_REASON_LENGTH = 1_000
MAX_PERSISTED_STRING_CHARS = 4_096
MAX_CANONICAL_JSON_DEPTH = 32
MAX_CANONICAL_JSON_ITEMS = 10_000
MAX_CANONICAL_JSON_INTEGER = (1 << 63) - 1

_ACTIVE_STATES = frozenset({"reserved", "in_progress"})
_TERMINAL_STATES = frozenset({"succeeded", "sent_with_warning", "failed"})
_FINISH_OUTCOMES = frozenset(
    {"succeeded", "sent_with_warning", "failed", "outcome_unknown"}
)
_OPERATION_ROW_COLUMNS = frozenset(
    {
        "id",
        "idempotency_key",
        "token_id",
        "token_name",
        "operation_name",
        "target_id",
        "request_hash",
        "state",
        "request_metadata_json",
        "upstream_thread_id",
        "response_json",
        "error_json",
        "created_at",
        "updated_at",
        "upstream_started_at",
        "completed_at",
    }
)
_SAFE_PERSISTED_KEYS = frozenset(
    {
        "action",
        "attempted_at",
        "body_format",
        "body_sha256",
        "cc",
        "code",
        "completed_at",
        "conversation_id",
        "conversation_status",
        "created",
        "created_at",
        "current_version",
        "details",
        "draft_thread_id",
        "expected_predecessor_thread_id",
        "expected_thread_id",
        "expected_version",
        "failed",
        "given_thread_id",
        "latest_actionable_thread_id",
        "new_version",
        "normalized_body_hash",
        "operation_name",
        "original_bytes",
        "reason",
        "refused_cc",
        "request_hash",
        "resolution",
        "resource_id",
        "result",
        "retryable",
        "sent",
        "sha256",
        "source",
        "startup_recovery",
        "state",
        "status",
        "thread_id",
        "truncated",
        "updated",
        "updated_at",
        "version",
        "warning_codes",
        "warnings",
    }
)
_SHA256_FIELDS = frozenset(
    {
        "body_sha256",
        "normalized_body_hash",
        "request_hash",
        "sha256",
    }
)
_VERSION_FIELDS = frozenset(
    {"current_version", "expected_version", "new_version", "version"}
)
_NUMERIC_ID_FIELDS = frozenset(
    {
        "conversation_id",
        "draft_thread_id",
        "expected_predecessor_thread_id",
        "expected_thread_id",
        "given_thread_id",
        "latest_actionable_thread_id",
        "resource_id",
        "thread_id",
    }
)
_BOOLEAN_FIELDS = frozenset(
    {
        "created",
        "failed",
        "retryable",
        "sent",
        "startup_recovery",
        "truncated",
        "updated",
    }
)
_TIMESTAMP_FIELDS = frozenset({"attempted_at", "completed_at", "created_at", "updated_at"})
_CODE_FIELDS = frozenset({"code", "operation_name", "result"})
_BEARER_VALUE_RE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_SECRET_VALUE_RE = re.compile(r"\b(?:sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]{12,})\b")
_SUPA_TOKEN_VALUE_RE = re.compile(r"[A-Za-z0-9_-]{43,}")
_SAFE_CODE_VALUE_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,199}$")
# CC addresses the gateway already validated (lowercased plain addresses).
_EMAIL_LIST_FIELDS = frozenset({"cc", "refused_cc"})
_EMAIL_VALUE_RE = re.compile(r"^[a-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}@[a-z0-9.-]{1,253}$")
_SHA256_VALUE_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_NUMERIC_ID_VALUE_RE = re.compile(r"^[1-9][0-9]{0,18}$")
_SAFE_ENUM_VALUES = {
    "action": frozenset({"added", "created", "updated", "sent", "resolved", "discarded"}),
    "body_format": frozenset({"html", "text"}),
    "conversation_status": frozenset({"active", "closed", "pending", "spam"}),
    "resolution": frozenset({"automatic", "manual", "reconciled"}),
    "state": frozenset(
        {
            "reserved",
            "in_progress",
            "succeeded",
            "sent_with_warning",
            "failed",
            "outcome_unknown",
        }
    ),
    "status": frozenset(
        {
            "active",
            "added",
            "added_with_warning",
            "closed",
            "created",
            "discarded",
            "failed",
            "ok",
            "pending",
            "saved",
            "sent",
            "sent_with_warning",
            "spam",
            "succeeded",
            "unknown",
            "updated",
            "updated_with_warning",
            "warning",
        }
    ),
    "source": frozenset({"local_resolution_recovery", "startup_recovery"}),
}
_DRAFT_ATTEMPT_METADATA_KEYS = frozenset(
    {
        "action",
        "conversation_id",
        "expected_thread_id",
        "normalized_body_hash",
    }
)
_SEND_ATTEMPT_METADATA_KEYS = frozenset(
    {
        "action",
        "conversation_id",
        "expected_predecessor_thread_id",
        "normalized_body_hash",
    }
)
_NOTE_ATTEMPT_METADATA_KEYS = frozenset(
    {
        "action",
        "conversation_id",
        "conversation_status",
        "normalized_body_hash",
    }
)
_NOTE_UPDATE_ATTEMPT_METADATA_KEYS = frozenset(
    {
        "action",
        "conversation_id",
        "conversation_status",
        "expected_thread_id",
        "normalized_body_hash",
    }
)
_DROP = object()


class AgentOperationError(Exception):
    """Base class carrying the stable API/CLI error contract."""

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


class AgentOperationConflictError(AgentOperationError):
    """An idempotency key already names a different request or owner."""

    def __init__(self, idempotency_key: str) -> None:
        super().__init__(
            "The idempotency key is already associated with a different request.",
            code="idempotency_conflict",
            retryable=False,
            status=409,
            exit_code=5,
            details={"idempotency_key": idempotency_key},
        )


class AgentOperationNotFoundError(AgentOperationError):
    """The operation is absent or not owned by the requesting token."""

    def __init__(self, idempotency_key: str) -> None:
        super().__init__(
            "Operation not found.",
            code="operation_not_found",
            retryable=False,
            status=404,
            exit_code=4,
            details={"idempotency_key": idempotency_key},
        )


class AgentOperationStateError(AgentOperationError):
    """The requested transition is not legal from the durable current state."""

    def __init__(
        self,
        idempotency_key: str,
        *,
        current_state: str | None = None,
        requested_state: str | None = None,
    ) -> None:
        details: dict[str, Any] = {"idempotency_key": idempotency_key}
        if current_state is not None:
            details["current_state"] = current_state
        if requested_state is not None:
            details["requested_state"] = requested_state
        super().__init__(
            "The operation cannot make that state transition.",
            code="operation_state_conflict",
            retryable=False,
            status=409,
            exit_code=5,
            details=details,
        )


class AgentOperationStorageError(AgentOperationError):
    """SQLite or stored operation data could not be handled safely."""

    def __init__(self) -> None:
        super().__init__(
            "The operation store could not be read or updated safely.",
            code="operation_storage_error",
            retryable=True,
            status=503,
            exit_code=6,
        )


class AgentOperationDataError(AgentOperationError):
    """Stored operation data is malformed and cannot improve through retry."""

    def __init__(self) -> None:
        super().__init__(
            "The operation store contains invalid data.",
            code="operation_data_error",
            retryable=False,
            status=500,
            exit_code=6,
        )


@dataclass(frozen=True)
class OperationRecord:
    idempotency_key: str
    token_id: str
    token_name: str
    operation_name: str
    target_id: str
    request_hash: str
    state: OperationState
    request_metadata: dict[str, Any]
    upstream_thread_id: str | None
    response: dict[str, Any] | None
    error: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    upstream_started_at: datetime | None
    completed_at: datetime | None

    def to_public_dict(self) -> dict[str, Any]:
        """Return the owner-safe operation status exposed by API/CLI reads."""

        return {
            "idempotency_key": self.idempotency_key,
            "operation_name": self.operation_name,
            "target_id": self.target_id,
            "request_hash": self.request_hash,
            "state": self.state,
            "request_metadata": self.request_metadata,
            "upstream_thread_id": self.upstream_thread_id,
            "response": self.response,
            "error": self.error,
            "created_at": _timestamp(self.created_at),
            "updated_at": _timestamp(self.updated_at),
            "upstream_started_at": (
                _timestamp(self.upstream_started_at)
                if self.upstream_started_at is not None
                else None
            ),
            "completed_at": (
                _timestamp(self.completed_at)
                if self.completed_at is not None
                else None
            ),
        }


@dataclass(frozen=True)
class OperationReservation:
    operation: OperationRecord
    replayed: bool


@dataclass(frozen=True)
class RecoverySummary:
    examined: int
    failed_before_attempt: int
    outcome_unknown: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _now(clock: Clock | None) -> datetime:
    value = (clock or _utc_now)()
    if not isinstance(value, datetime):
        raise TypeError("clock must return a datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise ValueError("clock returned an unsupported datetime") from error


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise AgentOperationDataError()
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError) as error:
        raise AgentOperationDataError() from error


def _optional_timestamp(value: Any) -> datetime | None:
    return None if value is None else _parse_timestamp(value)


def _stored_optional_numeric_id(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _sanitize_numeric_id(value)
    if normalized is _DROP:
        raise AgentOperationDataError()
    assert isinstance(normalized, str)
    return normalized


def _stored_text(value: Any, *, maximum: int) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise AgentOperationDataError()
    if len(value) > maximum or _SUPA_TOKEN_VALUE_RE.search(value):
        raise AgentOperationDataError()
    try:
        _valid_unicode_scalar_string(value)
    except ValueError as error:
        raise AgentOperationDataError() from error
    return value


def _input_optional_numeric_id(name: str, value: str | int | None) -> str | None:
    if value is None:
        return None
    normalized = _sanitize_numeric_id(value)
    if normalized is _DROP:
        raise ValueError(f"{name} must be a positive numeric ID")
    assert isinstance(normalized, str)
    return normalized


def _bounded_text(name: str, value: Any, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    normalized = value.strip()
    try:
        _valid_unicode_scalar_string(normalized)
    except ValueError as error:
        raise ValueError(f"{name} must contain valid Unicode") from error
    if len(normalized) > maximum:
        raise ValueError(f"{name} is too long")
    if _SUPA_TOKEN_VALUE_RE.search(normalized):
        raise ValueError(f"{name} must not contain a bearer token")
    return normalized


def _valid_unicode_scalar_string(value: str) -> None:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError("invalid Unicode scalar string") from error


def _validate_json_document(value: Any) -> None:
    """Validate one finite, bounded document made only of exact JSON types."""

    seen: set[int] = set()
    item_count = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal item_count
        item_count += 1
        if item_count > MAX_CANONICAL_JSON_ITEMS:
            raise ValueError("JSON document has too many items")
        if depth > MAX_CANONICAL_JSON_DEPTH:
            raise ValueError("JSON document is too deeply nested")

        item_type = type(item)
        if item is None or item_type is bool:
            return
        if item_type is int:
            if abs(item) > MAX_CANONICAL_JSON_INTEGER:
                raise ValueError("JSON integer is outside the supported range")
            return
        if item_type is float:
            if not math.isfinite(item):
                raise ValueError("JSON number must be finite")
            return
        if item_type is str:
            _valid_unicode_scalar_string(item)
            return
        if item_type not in {dict, list}:
            raise ValueError("unsupported JSON value type")

        identity = id(item)
        if identity in seen:
            raise ValueError("cyclic JSON document")
        seen.add(identity)
        try:
            if item_type is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise ValueError("JSON object keys must be strings")
                    item_count += 1
                    if item_count > MAX_CANONICAL_JSON_ITEMS:
                        raise ValueError("JSON document has too many items")
                    _valid_unicode_scalar_string(key)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
        finally:
            seen.remove(identity)

    visit(value, 0)


def canonical_request_json(request: Any) -> str:
    """Serialize a request deterministically for idempotency hashing."""

    try:
        _validate_json_document(request)
        serialized = json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        serialized.encode("utf-8", errors="strict")
        return serialized
    except (
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as error:
        raise ValueError("request must be valid JSON") from error


def hash_request(request: Any) -> str:
    try:
        canonical = canonical_request_json(request)
        return hashlib.sha256(canonical.encode("utf-8", errors="strict")).hexdigest()
    except (RecursionError, UnicodeError, ValueError) as error:
        raise ValueError("request must be valid JSON") from error


def _safe_persisted_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return normalized in _SAFE_PERSISTED_KEYS


def _sanitize_string(value: str) -> str:
    try:
        _valid_unicode_scalar_string(value)
    except ValueError:
        return "[redacted-invalid-unicode]"
    redacted = _BEARER_VALUE_RE.sub("Bearer [redacted]", value)
    redacted = _SECRET_VALUE_RE.sub("[redacted-token]", redacted)
    redacted = _SUPA_TOKEN_VALUE_RE.sub("[redacted-token]", redacted)
    if len(redacted) > MAX_PERSISTED_STRING_CHARS:
        digest = hashlib.sha256(redacted.encode("utf-8")).hexdigest()
        return f"[truncated sha256={digest}]"
    return redacted


def _sanitize_timestamp(value: Any) -> str | object:
    if type(value) is not str:
        return _DROP
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return _DROP
        return _timestamp(parsed.astimezone(timezone.utc))
    except (OverflowError, TypeError, ValueError):
        return _DROP


def _sanitize_numeric_id(value: Any) -> str | object:
    if type(value) is int:
        normalized = str(value)
    elif type(value) is str:
        normalized = value
    else:
        return _DROP
    if not _NUMERIC_ID_VALUE_RE.fullmatch(normalized):
        return _DROP
    if int(normalized) > MAX_CANONICAL_JSON_INTEGER:
        return _DROP
    return normalized


def _sanitize_code(value: Any) -> str | object:
    if type(value) is not str:
        return _DROP
    sanitized = _sanitize_string(value)
    if not _SAFE_CODE_VALUE_RE.fullmatch(sanitized):
        return _DROP
    return sanitized


def _sanitize_value(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    if depth > 8:
        return _DROP
    if key in _SHA256_FIELDS | _VERSION_FIELDS:
        if key == "expected_version" and value is None:
            return None
        if type(value) is not str or not _SHA256_VALUE_RE.fullmatch(value):
            return _DROP
        return value.lower()
    if key in _NUMERIC_ID_FIELDS:
        if key in {
            "expected_predecessor_thread_id", "expected_thread_id", "latest_actionable_thread_id"
        } and value is None:
            return None
        return _sanitize_numeric_id(value)
    if key in _BOOLEAN_FIELDS:
        return value if type(value) is bool else _DROP
    if key in _TIMESTAMP_FIELDS:
        return _sanitize_timestamp(value)
    if key in _CODE_FIELDS:
        return _sanitize_code(value)
    if key == "status" and type(value) is int:
        return value if 100 <= value <= 599 else _DROP
    if key in _SAFE_ENUM_VALUES:
        if type(value) is not str:
            return _DROP
        sanitized = _sanitize_string(value)
        return sanitized if sanitized in _SAFE_ENUM_VALUES[key] else _DROP
    if key == "reason":
        if type(value) is not str:
            return _DROP
        return _sanitize_string(value)
    if key == "original_bytes":
        return value if type(value) is int and value >= 0 else _DROP
    if key in _EMAIL_LIST_FIELDS:
        if type(value) is not list:
            return _DROP
        return [
            item
            for item in value[:10]
            if type(item) is str and _EMAIL_VALUE_RE.fullmatch(item)
        ]
    if key in {"warning_codes", "warnings"}:
        if type(value) is not list:
            return _DROP
        items: list[Any] = []
        for item in value[:100]:
            if type(item) is dict:
                sanitized_item = _sanitize_value(item, key="details", depth=depth + 1)
            else:
                sanitized_item = _sanitize_code(item)
            if sanitized_item is not _DROP:
                items.append(sanitized_item)
        return items
    if key == "details" or (key is None and type(value) is dict):
        if type(value) is not dict:
            return _DROP
        sanitized: dict[str, Any] = {}
        for index, (raw_key, raw_value) in enumerate(value.items()):
            if index >= 100:
                break
            if type(raw_key) is not str or not _safe_persisted_key(raw_key):
                continue
            normalized_key = raw_key.casefold().replace("-", "_")
            sanitized_value = _sanitize_value(
                raw_value,
                key=normalized_key,
                depth=depth + 1,
            )
            if sanitized_value is not _DROP:
                sanitized[normalized_key] = sanitized_value
        return sanitized
    return _DROP


def _bounded_sanitized_mapping(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("persisted operation data must be a JSON object")
    try:
        _validate_json_document(value)
    except (RecursionError, UnicodeError, ValueError) as error:
        raise ValueError("persisted operation data must be valid JSON") from error
    sanitized = _sanitize_value(value)
    assert isinstance(sanitized, dict)
    encoded = canonical_request_json(sanitized).encode("utf-8")
    if len(encoded) <= MAX_PERSISTED_JSON_BYTES:
        return sanitized

    summary: dict[str, Any] = {
        "truncated": True,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "original_bytes": len(encoded),
    }
    for key in ("code", "status", "state", "thread_id", "resolution"):
        candidate = sanitized.get(key)
        if candidate is None or isinstance(candidate, (bool, int, float, str)):
            if candidate is not None:
                summary[key] = candidate
    for key in (
        "body_sha256",
        "expected_predecessor_thread_id",
        "expected_thread_id",
        "expected_version",
        "normalized_body_hash",
    ):
        candidate = sanitized.get(key)
        if candidate is None or isinstance(candidate, (bool, int, float, str)):
            if candidate is not None:
                summary[key] = candidate
    return summary


def _validated_attempt_metadata(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the reconciliation facts persisted at the submission boundary."""

    if value is None:
        return {}
    if type(value) is not dict:
        raise ValueError("attempt_metadata has invalid fields")
    fields = frozenset(value)
    if fields not in {
        _DRAFT_ATTEMPT_METADATA_KEYS,
        _SEND_ATTEMPT_METADATA_KEYS,
        _NOTE_ATTEMPT_METADATA_KEYS,
        _NOTE_UPDATE_ATTEMPT_METADATA_KEYS,
    }:
        raise ValueError("attempt_metadata has invalid fields")
    sanitized = _bounded_sanitized_mapping(value)
    if sanitized is None or set(sanitized) != fields:
        raise ValueError("attempt_metadata contains invalid values")
    action = sanitized.get("action")
    if fields == _NOTE_ATTEMPT_METADATA_KEYS:
        if action != "added" or sanitized.get("conversation_status") is None:
            raise ValueError("note attempt_metadata is invalid")
        return sanitized
    if fields == _NOTE_UPDATE_ATTEMPT_METADATA_KEYS:
        if (
            action != "updated"
            or sanitized.get("conversation_status") is None
            or sanitized.get("expected_thread_id") is None
        ):
            raise ValueError("note update attempt_metadata is invalid")
        return sanitized
    if fields == _SEND_ATTEMPT_METADATA_KEYS:
        if action != "sent" or sanitized.get("expected_predecessor_thread_id") is None:
            raise ValueError("send attempt_metadata is invalid")
        return sanitized
    expected_thread_id = sanitized.get("expected_thread_id")
    if action not in {"created", "updated"}:
        raise ValueError("attempt_metadata action is invalid")
    if action == "created" and expected_thread_id is not None:
        raise ValueError("a draft create cannot have an expected thread id")
    if action == "updated" and expected_thread_id is None:
        raise ValueError("a draft update requires an expected thread id")
    return sanitized


def _json_text(value: Mapping[str, Any] | None) -> str | None:
    sanitized = _bounded_sanitized_mapping(value)
    return None if sanitized is None else canonical_request_json(sanitized)


def _parse_json_mapping(value: Any, *, nullable: bool) -> dict[str, Any] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise AgentOperationDataError()
    try:
        parsed = json.loads(value)
        _validate_json_document(parsed)
    except (
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as error:
        raise AgentOperationDataError() from error
    if not isinstance(parsed, dict):
        raise AgentOperationDataError()
    try:
        sanitized = _bounded_sanitized_mapping(parsed)
    except (RecursionError, UnicodeError, ValueError) as error:
        raise AgentOperationDataError() from error
    assert sanitized is not None
    return sanitized


def _path_value(db_path: PathValue | None) -> str:
    return get_supa_db_path() if db_path is None else os.fspath(Path(db_path))


def _connect(db_path: PathValue | None) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(
            _path_value(db_path), timeout=30.0, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection
    except sqlite3.Error as error:
        raise AgentOperationStorageError() from error


@contextmanager
def _write_transaction(db_path: PathValue | None) -> Iterator[sqlite3.Connection]:
    connection = _connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except AgentOperationError:
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
        raise AgentOperationStorageError() from error
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


def _record_from_row(row: sqlite3.Row) -> OperationRecord:
    state = row["state"]
    if state not in _ACTIVE_STATES | _TERMINAL_STATES | {"outcome_unknown"}:
        raise AgentOperationDataError()
    request_hash = row["request_hash"]
    if (
        not isinstance(request_hash, str)
        or len(request_hash) != 64
        or any(character not in "0123456789abcdef" for character in request_hash)
    ):
        raise AgentOperationDataError()
    metadata = _parse_json_mapping(row["request_metadata_json"], nullable=False)
    assert metadata is not None
    return OperationRecord(
        idempotency_key=_stored_text(
            row["idempotency_key"], maximum=MAX_IDEMPOTENCY_KEY_LENGTH
        ),
        token_id=_stored_text(row["token_id"], maximum=MAX_NAME_LENGTH),
        token_name=_stored_text(row["token_name"], maximum=MAX_NAME_LENGTH),
        operation_name=_stored_text(
            row["operation_name"], maximum=MAX_NAME_LENGTH
        ),
        target_id=_stored_text(row["target_id"], maximum=MAX_TARGET_LENGTH),
        request_hash=request_hash,
        state=state,
        request_metadata=metadata,
        upstream_thread_id=_stored_optional_numeric_id(row["upstream_thread_id"]),
        response=_parse_json_mapping(row["response_json"], nullable=True),
        error=_parse_json_mapping(row["error_json"], nullable=True),
        created_at=_parse_timestamp(row["created_at"]),
        updated_at=_parse_timestamp(row["updated_at"]),
        upstream_started_at=_optional_timestamp(row["upstream_started_at"]),
        completed_at=_optional_timestamp(row["completed_at"]),
    )


def _select_operation(
    connection: sqlite3.Connection, idempotency_key: str
) -> sqlite3.Row | None:
    row = connection.execute(
        "SELECT * FROM agent_operations WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()
    return None if row is None else _validated_operation_row(row)


def _validated_operation_row(row: sqlite3.Row) -> sqlite3.Row:
    """Reject incompatible row shapes before any named-column access."""

    try:
        columns = frozenset(row.keys())
    except (AttributeError, TypeError) as error:
        raise AgentOperationDataError() from error
    if not _OPERATION_ROW_COLUMNS.issubset(columns):
        raise AgentOperationDataError()
    return row


def _owned_row(
    connection: sqlite3.Connection,
    idempotency_key: str,
    identity: AgentIdentity,
) -> sqlite3.Row:
    row = _select_operation(connection, idempotency_key)
    if row is None or row["token_id"] != identity.token_id:
        raise AgentOperationNotFoundError(idempotency_key)
    _record_from_row(row)
    return row


def _insert_audit(
    connection: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    scope: str,
    request_id: str,
    result: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO agent_audit (
            idempotency_key, token_id, token_name, scope, operation_name,
            target_id, request_id, request_hash, result, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["idempotency_key"],
            row["token_id"],
            row["token_name"],
            _bounded_text("scope", scope, MAX_NAME_LENGTH),
            row["operation_name"],
            row["target_id"],
            _bounded_text("request_id", request_id, MAX_REQUEST_ID_LENGTH),
            row["request_hash"],
            result,
            _json_text(metadata or {}) or "{}",
        ),
    )


def reserve_operation(
    *,
    idempotency_key: str,
    identity: AgentIdentity,
    scope: str,
    operation_name: str,
    target_id: str | int,
    request_id: str,
    request: Any,
    request_metadata: Mapping[str, Any] | None = None,
    db_path: PathValue | None = None,
    clock: Clock | None = None,
) -> OperationReservation:
    """Atomically reserve a key, or return its exact same-request replay."""

    key = _bounded_text(
        "idempotency_key", idempotency_key, MAX_IDEMPOTENCY_KEY_LENGTH
    )
    token_id = _bounded_text("token_id", identity.token_id, MAX_NAME_LENGTH)
    token_name = _bounded_text("token_name", identity.name, MAX_NAME_LENGTH)
    operation = _bounded_text("operation_name", operation_name, MAX_NAME_LENGTH)
    target = _bounded_text("target_id", str(target_id), MAX_TARGET_LENGTH)
    request_digest = hash_request(request)
    metadata_text = _json_text(request_metadata or {}) or "{}"
    timestamp = _timestamp(_now(clock))

    with _write_transaction(db_path) as connection:
        existing = _select_operation(connection, key)
        if existing is not None:
            if (
                existing["request_hash"] != request_digest
                or existing["token_id"] != token_id
                or existing["operation_name"] != operation
                or existing["target_id"] != target
            ):
                raise AgentOperationConflictError(key)
            return OperationReservation(_record_from_row(existing), replayed=True)

        connection.execute(
            """
            INSERT INTO agent_operations (
                idempotency_key, token_id, token_name, operation_name,
                target_id, request_hash, state, request_metadata_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'reserved', ?, ?, ?)
            """,
            (
                key,
                token_id,
                token_name,
                operation,
                target,
                request_digest,
                metadata_text,
                timestamp,
                timestamp,
            ),
        )
        inserted = _select_operation(connection, key)
        assert inserted is not None
        _insert_audit(
            connection,
            row=inserted,
            scope=scope,
            request_id=request_id,
            result="reserved",
            metadata={"state": "reserved"},
        )
        return OperationReservation(_record_from_row(inserted), replayed=False)


def get_operation(
    *,
    idempotency_key: str,
    identity: AgentIdentity,
    db_path: PathValue | None = None,
) -> OperationRecord:
    """Read an operation only when it belongs to the authenticated token."""

    key = _bounded_text(
        "idempotency_key", idempotency_key, MAX_IDEMPOTENCY_KEY_LENGTH
    )
    connection = _connect(db_path)
    try:
        row = _owned_row(connection, key, identity)
        return _record_from_row(row)
    except AgentOperationError:
        raise
    except sqlite3.Error as error:
        raise AgentOperationStorageError() from error
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass


def get_operation_locally(
    *, idempotency_key: str, db_path: PathValue | None = None
) -> OperationRecord:
    """Read an operation for local maintenance without a bearer identity."""

    key = _bounded_text(
        "idempotency_key", idempotency_key, MAX_IDEMPOTENCY_KEY_LENGTH
    )
    connection = _connect(db_path)
    try:
        row = _select_operation(connection, key)
        if row is None:
            raise AgentOperationNotFoundError(key)
        return _record_from_row(row)
    except AgentOperationError:
        raise
    except sqlite3.Error as error:
        raise AgentOperationStorageError() from error
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass


def begin_operation(
    *,
    idempotency_key: str,
    identity: AgentIdentity,
    scope: str,
    request_id: str,
    db_path: PathValue | None = None,
    clock: Clock | None = None,
) -> OperationRecord:
    """Apply the sole normal non-terminal transition: reserved -> in_progress."""

    key = _bounded_text(
        "idempotency_key", idempotency_key, MAX_IDEMPOTENCY_KEY_LENGTH
    )
    timestamp = _timestamp(_now(clock))
    with _write_transaction(db_path) as connection:
        row = _owned_row(connection, key, identity)
        if row["state"] != "reserved":
            raise AgentOperationStateError(
                key,
                current_state=row["state"],
                requested_state="in_progress",
            )
        connection.execute(
            "UPDATE agent_operations SET state = 'in_progress', updated_at = ? "
            "WHERE idempotency_key = ? AND state = 'reserved'",
            (timestamp, key),
        )
        updated = _select_operation(connection, key)
        assert updated is not None
        _insert_audit(
            connection,
            row=updated,
            scope=scope,
            request_id=request_id,
            result="in_progress",
            metadata={"state": "in_progress"},
        )
        return _record_from_row(updated)


def mark_upstream_started(
    *,
    idempotency_key: str,
    identity: AgentIdentity,
    scope: str,
    request_id: str,
    attempt_metadata: Mapping[str, Any] | None = None,
    db_path: PathValue | None = None,
    clock: Clock | None = None,
) -> OperationRecord:
    """Durably mark the one allowed submission attempt immediately before I/O."""

    key = _bounded_text(
        "idempotency_key", idempotency_key, MAX_IDEMPOTENCY_KEY_LENGTH
    )
    metadata = _validated_attempt_metadata(attempt_metadata)
    timestamp = _timestamp(_now(clock))
    with _write_transaction(db_path) as connection:
        row = _owned_row(connection, key, identity)
        if row["state"] != "in_progress" or row["upstream_started_at"] is not None:
            raise AgentOperationStateError(
                key,
                current_state=row["state"],
                requested_state="upstream_started",
            )
        if metadata:
            operation_name = row["operation_name"]
            action = metadata.get("action")
            schema_matches = (
                action in {"created", "updated"}
                and operation_name == "draft.save"
            ) or (
                action == "sent"
                and operation_name in {"draft.send", "reply.send"}
            ) or (action == "added" and operation_name == "note.add") or (
                action == "updated"
                and operation_name == "note.update"
                and frozenset(metadata) == _NOTE_UPDATE_ATTEMPT_METADATA_KEYS
            )
            if not schema_matches:
                raise ValueError("attempt_metadata does not match the operation")
        stored_metadata = _parse_json_mapping(
            row["request_metadata_json"],
            nullable=False,
        )
        assert stored_metadata is not None
        stored_metadata.update(metadata)
        stored_metadata["attempted_at"] = timestamp
        serialized_metadata = _json_text(stored_metadata)
        persisted_metadata = _parse_json_mapping(
            serialized_metadata,
            nullable=False,
        )
        assert persisted_metadata is not None
        required_attempt_facts = {"attempted_at": timestamp, **metadata}
        if any(
            persisted_metadata.get(field) != expected
            or field not in persisted_metadata
            for field, expected in required_attempt_facts.items()
        ):
            raise ValueError("attempt metadata could not be persisted without loss")
        connection.execute(
            "UPDATE agent_operations "
            "SET upstream_started_at = ?, request_metadata_json = ?, updated_at = ? "
            "WHERE idempotency_key = ? AND state = 'in_progress' "
            "AND upstream_started_at IS NULL",
            (timestamp, serialized_metadata, timestamp, key),
        )
        updated = _select_operation(connection, key)
        assert updated is not None
        _insert_audit(
            connection,
            row=updated,
            scope=scope,
            request_id=request_id,
            result="upstream_started",
            metadata={"attempted_at": timestamp, **metadata},
        )
        return _record_from_row(updated)


def finish_operation(
    *,
    idempotency_key: str,
    identity: AgentIdentity,
    scope: str,
    request_id: str,
    outcome: str,
    upstream_thread_id: str | int | None = None,
    response: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
    db_path: PathValue | None = None,
    clock: Clock | None = None,
) -> OperationRecord:
    """Finish an in-progress operation with one documented durable outcome."""

    key = _bounded_text(
        "idempotency_key", idempotency_key, MAX_IDEMPOTENCY_KEY_LENGTH
    )
    if outcome not in _FINISH_OUTCOMES:
        raise AgentOperationStateError(key, requested_state=outcome)
    thread_id = _input_optional_numeric_id("upstream_thread_id", upstream_thread_id)
    response_text = _json_text(response)
    error_text = _json_text(error)
    timestamp = _timestamp(_now(clock))

    with _write_transaction(db_path) as connection:
        row = _owned_row(connection, key, identity)
        if row["state"] != "in_progress":
            raise AgentOperationStateError(
                key,
                current_state=row["state"],
                requested_state=outcome,
            )
        # A successful, warning, or uncertain upstream result is impossible
        # without the durable pre-submission marker. A local pre-I/O failure is
        # the only terminal result permitted without it.
        if outcome != "failed" and row["upstream_started_at"] is None:
            raise AgentOperationStateError(
                key,
                current_state=row["state"],
                requested_state=outcome,
            )
        completed_at = None if outcome == "outcome_unknown" else timestamp
        connection.execute(
            """
            UPDATE agent_operations
            SET state = ?, upstream_thread_id = COALESCE(?, upstream_thread_id),
                response_json = ?, error_json = ?, updated_at = ?, completed_at = ?
            WHERE idempotency_key = ? AND state = 'in_progress'
            """,
            (
                outcome,
                thread_id,
                response_text,
                error_text,
                timestamp,
                completed_at,
                key,
            ),
        )
        updated = _select_operation(connection, key)
        assert updated is not None
        _insert_audit(
            connection,
            row=updated,
            scope=scope,
            request_id=request_id,
            result=outcome,
            metadata={"state": outcome, "thread_id": thread_id},
        )
        return _record_from_row(updated)


def _stale_after_seconds(value: float | None) -> float:
    raw: Any = os.getenv(STALE_AFTER_ENV, str(DEFAULT_STALE_AFTER_SECONDS)) if value is None else value
    if isinstance(raw, bool):
        raise ValueError("stale_after_seconds must be a finite number")
    try:
        normalized = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("stale_after_seconds must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError("stale_after_seconds must be finite")
    if normalized < MIN_STALE_AFTER_SECONDS or normalized > MAX_STALE_AFTER_SECONDS:
        raise ValueError(
            f"stale_after_seconds must be between {MIN_STALE_AFTER_SECONDS:g} "
            f"and {MAX_STALE_AFTER_SECONDS:g} seconds"
        )
    return normalized


def _is_stale(row: sqlite3.Row, *, now: datetime, stale_after: float) -> bool:
    try:
        cutoff = now - timedelta(seconds=stale_after)
    except OverflowError as error:
        raise ValueError("stale_after_seconds is outside the datetime range") from error
    return _parse_timestamp(row["updated_at"]) <= cutoff


def _recover_row(
    connection: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    timestamp: str,
    request_id: str,
    source: Literal["local_resolution_recovery", "startup_recovery"],
) -> Literal["failed", "outcome_unknown"]:
    if row["state"] == "reserved" and row["upstream_started_at"] is None:
        outcome: Literal["failed", "outcome_unknown"] = "failed"
        error = {
            "code": "interrupted_before_attempt",
            "retryable": False,
        }
        completed_at: str | None = timestamp
        audit_result = "failed"
    else:
        outcome = "outcome_unknown"
        error = {
            "code": "interrupted_with_uncertain_outcome",
            "retryable": False,
        }
        completed_at = None
        audit_result = "outcome_unknown"
    connection.execute(
        "UPDATE agent_operations "
        "SET state = ?, error_json = ?, updated_at = ?, completed_at = ? "
        "WHERE id = ? AND state IN ('reserved', 'in_progress')",
        (outcome, _json_text(error), timestamp, completed_at, row["id"]),
    )
    updated = _select_operation(connection, row["idempotency_key"])
    assert updated is not None
    _insert_audit(
        connection,
        row=updated,
        scope="maintenance",
        request_id=request_id,
        result=audit_result,
        metadata={"reason": error["code"], "source": source},
    )
    return outcome


def recover_stale_operations(
    *,
    db_path: PathValue | None = None,
    stale_after_seconds: float | None = None,
    clock: Clock | None = None,
) -> RecoverySummary:
    """Fail or quarantine abandoned active rows without retrying any write."""

    now = _now(clock)
    stale_after = _stale_after_seconds(stale_after_seconds)
    timestamp = _timestamp(now)
    examined = failed = unknown = 0
    with _write_transaction(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM agent_operations "
            "WHERE state IN ('reserved', 'in_progress') ORDER BY id"
        ).fetchall()
        validated_rows = [
            (_validated_operation_row(row), _record_from_row(row)) for row in rows
        ]
        try:
            cutoff = now - timedelta(seconds=stale_after)
        except OverflowError as error:
            raise ValueError(
                "stale_after_seconds is outside the datetime range"
            ) from error
        for row, record in validated_rows:
            has_upstream_start = record.upstream_started_at is not None
            if not has_upstream_start and record.updated_at > cutoff:
                continue
            examined += 1
            outcome = _recover_row(
                connection,
                row=row,
                timestamp=timestamp,
                request_id="startup-recovery",
                source="startup_recovery",
            )
            if outcome == "failed":
                failed += 1
            else:
                unknown += 1
    return RecoverySummary(
        examined=examined,
        failed_before_attempt=failed,
        outcome_unknown=unknown,
    )


def resolve_operation_locally(
    *,
    idempotency_key: str,
    outcome: str,
    reason: str,
    upstream_thread_id: str | int | None = None,
    db_path: PathValue | None = None,
    stale_after_seconds: float | None = None,
    clock: Clock | None = None,
) -> OperationRecord:
    """Resolve an unknown (or stale, first quarantined) row from local tooling."""

    key = _bounded_text(
        "idempotency_key", idempotency_key, MAX_IDEMPOTENCY_KEY_LENGTH
    )
    if outcome not in {"succeeded", "failed"}:
        raise ValueError("outcome must be succeeded or failed")
    resolution_reason = _bounded_text("reason", reason, MAX_REASON_LENGTH)
    thread_id = _input_optional_numeric_id("upstream_thread_id", upstream_thread_id)
    now = _now(clock)
    stale_after = _stale_after_seconds(stale_after_seconds)
    timestamp = _timestamp(now)

    with _write_transaction(db_path) as connection:
        row = _select_operation(connection, key)
        if row is None:
            raise AgentOperationNotFoundError(key)
        _record_from_row(row)

        if row["state"] in _ACTIVE_STATES:
            if not _is_stale(row, now=now, stale_after=stale_after):
                raise AgentOperationStateError(
                    key,
                    current_state=row["state"],
                    requested_state=outcome,
                )
            if row["state"] == "reserved" and row["upstream_started_at"] is None:
                # Recovery can prove this write was never attempted. It may be
                # locally confirmed failed, but never fabricated as succeeded.
                if outcome != "failed":
                    raise AgentOperationStateError(
                        key,
                        current_state=row["state"],
                        requested_state=outcome,
                    )
                _recover_row(
                    connection,
                    row=row,
                    timestamp=timestamp,
                    request_id="local-resolution-recovery",
                    source="local_resolution_recovery",
                )
                recovered = _select_operation(connection, key)
                assert recovered is not None
                _insert_audit(
                    connection,
                    row=recovered,
                    scope="maintenance",
                    request_id="local-resolution",
                    result="resolved_failed",
                    metadata={"reason": resolution_reason, "thread_id": thread_id},
                )
                return _record_from_row(recovered)

            _recover_row(
                connection,
                row=row,
                timestamp=timestamp,
                request_id="local-resolution-recovery",
                source="local_resolution_recovery",
            )
            row = _select_operation(connection, key)
            assert row is not None

        if row["state"] != "outcome_unknown":
            raise AgentOperationStateError(
                key,
                current_state=row["state"],
                requested_state=outcome,
            )

        stored_thread_id = _stored_optional_numeric_id(row["upstream_thread_id"])
        effective_thread_id = thread_id or stored_thread_id
        if outcome == "succeeded" and effective_thread_id is None:
            raise AgentOperationStateError(
                key,
                current_state=row["state"],
                requested_state=outcome,
            )

        if outcome == "succeeded":
            response = {
                "resolution": "manual",
                "reason": resolution_reason,
            }
            error = None
        else:
            response = None
            error = {
                "code": "manually_resolved_failed",
                "reason": resolution_reason,
                "retryable": False,
            }
        connection.execute(
            """
            UPDATE agent_operations
            SET state = ?, upstream_thread_id = COALESCE(?, upstream_thread_id),
                response_json = ?, error_json = ?, updated_at = ?, completed_at = ?
            WHERE id = ? AND state = 'outcome_unknown'
            """,
            (
                outcome,
                effective_thread_id,
                _json_text(response),
                _json_text(error),
                timestamp,
                timestamp,
                row["id"],
            ),
        )
        updated = _select_operation(connection, key)
        assert updated is not None
        _insert_audit(
            connection,
            row=updated,
            scope="maintenance",
            request_id="local-resolution",
            result=f"resolved_{outcome}",
            metadata={
                "reason": resolution_reason,
                "thread_id": effective_thread_id,
            },
        )
        return _record_from_row(updated)
