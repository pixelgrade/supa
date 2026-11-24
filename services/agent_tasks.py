"""Durable repository for owner-scoped named task runs and progress."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from services import extensions
from services.agent_events import append_supa_event_in_transaction
from services.runtime_paths import get_supa_db_path
from services.conversation_policy import manual_route


PathValue = str | os.PathLike[str]
Clock = Callable[[], datetime]
UUIDFactory = Callable[[], uuid.UUID]
RunStatus = Literal["queued", "running", "completed", "failed", "interrupted"]

MAX_RUN_JSON_BYTES = 1024 * 1024
MAX_RUN_JSON_DEPTH = 32
MAX_RUN_JSON_ITEMS = 10_000
MAX_IDENTIFIER_LENGTH = 200
MAX_TEXT_LENGTH = 1024 * 1024
MAX_RESOURCE_ID = (1 << 63) - 1

_RUN_STATUSES = frozenset({"queued", "running", "completed", "failed", "interrupted"})
_TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted"})
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_TASK_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,199}$")
_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9._~-]{1,200}$")
_STAGE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,199}$")
_NUMERIC_ID_RE = re.compile(r"^[1-9][0-9]{0,18}$")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[^\s,;<>\"']+")
_API_SECRET_RE = re.compile(r"\b(?:sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]{12,})\b")
_SUPA_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{43}(?![A-Za-z0-9_-])")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|password|secret)(\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;<>]+)"
)
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "client_secret",
        "cookie",
        "credentials",
        "password",
        "secret",
        "set_cookie",
        "token",
        "access_token",
        "refresh_token",
    }
)
_NUMERIC_ID_KEYS = frozenset(
    {
        "attachment_id",
        "conversation_id",
        "conversation_number",
        "mailbox_id",
        "thread_id",
    }
)
_RUN_ROW_COLUMNS = frozenset(
    {
        "run_id",
        "idempotency_key",
        "token_id",
        "token_name",
        "task_name",
        "conversation_id",
        "status",
        "arguments_json",
        "scope_snapshot_json",
        "delivery_ceiling",
        "result_json",
        "error_json",
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
    }
)
_EVENT_ROW_COLUMNS = frozenset(
    {"id", "run_id", "sequence", "stage", "progress_json", "created_at"}
)


class AgentRunError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool,
        status: int,
        exit_code: int = 6,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status = status
        self.exit_code = exit_code


class AgentRunStorageError(AgentRunError):
    def __init__(self) -> None:
        super().__init__(
            "The task run store is unavailable.",
            code="run_storage_error",
            retryable=True,
            status=503,
        )


class AgentRunDataError(AgentRunError):
    def __init__(self) -> None:
        super().__init__(
            "The task run store contains invalid data.",
            code="run_data_error",
            retryable=False,
            status=500,
        )


class AgentRunNotFoundError(AgentRunError):
    def __init__(self, run_id: str) -> None:
        super().__init__(
            "Task run not found.",
            code="run_not_found",
            retryable=False,
            status=404,
            exit_code=4,
        )
        self.run_id = run_id


class AgentRunStateError(AgentRunError):
    def __init__(self) -> None:
        super().__init__(
            "Task run state transition is invalid.",
            code="run_state_conflict",
            retryable=False,
            status=409,
            exit_code=5,
        )


class AgentTaskRequestError(AgentRunError):
    def __init__(self, code: str) -> None:
        messages = {
            "unknown_task": "Unknown task.",
            "invalid_task_arguments": "Task arguments are invalid.",
            "insufficient_scope": "Insufficient scope.",
            "idempotency_conflict": "Idempotency key was used for a different task request.",
        }
        super().__init__(
            messages[code], code=code, retryable=False,
            status=403 if code == "insufficient_scope" else (409 if code == "idempotency_conflict" else 400),
            exit_code=3 if code == "insufficient_scope" else (5 if code == "idempotency_conflict" else 2),
        )


class _TaskArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    conversation_id: int | str

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: int | str) -> str:
        return _conversation_id(value)


class _ProcessArguments(_TaskArguments):
    allow_send: bool = False


class _RecoveryArguments(_TaskArguments):
    expected_thread_id: str = Field(pattern=r"^[1-9][0-9]*$")
    expected_draft_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_conversation_version: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TaskDefinition:
    required_scopes: frozenset[str]
    arguments_model: type[_TaskArguments]
    runner_name: str


TASKS: dict[str, TaskDefinition] = {
    "ticket.recover": TaskDefinition(
        required_scopes=frozenset({"task", "read", "draft"}),
        arguments_model=_RecoveryArguments,
        runner_name="process",
    ),
    "ticket.evaluate": TaskDefinition(
        required_scopes=frozenset({"task", "read"}),
        arguments_model=_TaskArguments,
        runner_name="evaluate",
    ),
    **{name: TaskDefinition(task.required_scopes, _TaskArguments, task.runner_name)
       for name, task in extensions.agent_tasks().items()},
    "ticket.process": TaskDefinition(
        required_scopes=frozenset({"task", "read", "draft"}),
        arguments_model=_ProcessArguments,
        runner_name="process",
    ),
}


@dataclass(frozen=True)
class PreparedTaskRequest:
    task_name: str
    arguments: dict[str, Any]
    scopes: tuple[str, ...]
    delivery_ceiling: Literal["draft", "policy"]


def prepare_task_request(
    task_name: str,
    arguments: Mapping[str, Any],
    *,
    scopes: Iterable[str],
) -> PreparedTaskRequest:
    definition = TASKS.get(task_name) if isinstance(task_name, str) else None
    if definition is None:
        raise AgentTaskRequestError("unknown_task")
    _, normalized_scopes = _scope_text(scopes)
    if not definition.required_scopes.issubset(normalized_scopes):
        raise AgentTaskRequestError("insufficient_scope")
    try:
        parsed = definition.arguments_model.model_validate(arguments)
    except (TypeError, ValueError, ValidationError) as error:
        raise AgentTaskRequestError("invalid_task_arguments") from error
    normalized_arguments = parsed.model_dump()
    allow_send = normalized_arguments.get("allow_send", False)
    if allow_send and "send" not in normalized_scopes:
        raise AgentTaskRequestError("insufficient_scope")
    return PreparedTaskRequest(
        task_name=task_name,
        arguments=normalized_arguments,
        scopes=normalized_scopes,
        delivery_ceiling="policy" if allow_send else "draft",
    )


async def run_evaluate_task(repository: AgentTaskRepository, run_id: str) -> None:
    """Execute one queued evaluation using the dashboard's read-only pipeline."""
    claimed = repository.claim_run(run_id)
    if claimed is None:
        return
    if claimed.task_name != "ticket.evaluate":
        repository.fail_run(
            run_id,
            error={"code": "wrong_task", "message": "Run is not an evaluation task."},
        )
        return
    try:
        repository.append_progress(
            run_id, stage="fetch", progress={"message": "Fetching conversation"}
        )
        # Import lazily so the dashboard and named task share the same dry-run
        # implementation without a module-level API/repository import cycle.
        from api.dashboard import evaluate_ticket

        repository.append_progress(
            run_id, stage="evaluate", progress={"message": "Evaluating ticket"}
        )
        result = await evaluate_ticket(int(claimed.conversation_id))
        repository.append_progress(
            run_id, stage="completed", progress={"message": "Evaluation complete"}
        )
        repository.complete_run(run_id, result=result.model_dump(mode="json"))
    except asyncio.CancelledError:
        repository.interrupt_run(
            run_id, error={"code": "run_interrupted", "message": "Run interrupted."}
        )
        raise
    except Exception:
        repository.fail_run(
            run_id,
            error={"code": "evaluation_failed", "message": "Ticket evaluation failed."},
        )


async def run_process_task(
    repository: AgentTaskRepository,
    run_id: str,
    *,
    processor: Any = None,
) -> None:
    """Process one allowlisted task with the persisted delivery ceiling."""
    claimed = repository.claim_run(run_id)
    if claimed is None:
        return
    if claimed.task_name not in {"ticket.process", "ticket.recover"}:
        repository.fail_run(
            run_id, error={"code": "wrong_task", "message": "Run is not a process task."}
        )
        return
    if (
        claimed.delivery_ceiling not in {"draft", "policy"}
        or not {"task", "read", "draft"}.issubset(claimed.scopes)
        or (claimed.delivery_ceiling == "policy" and "send" not in claimed.scopes)
    ):
        repository.fail_run(
            run_id, error={"code": "invalid_run_authority", "message": "Run authority is invalid."}
        )
        return
    try:
        from services.processor import get_processor

        active_processor = processor if processor is not None else get_processor()
        repository.append_progress(
            run_id, stage="fetch", progress={"message": "Fetching conversation"}
        )
        fetch = (active_processor.helpscout.get_complete_conversation
                 if claimed.task_name == "ticket.recover" else active_processor.helpscout.get_conversation)
        payload = await fetch(int(claimed.conversation_id))
        if not isinstance(payload, dict) or str(payload.get("id")) != claimed.conversation_id:
            raise ValueError("Fetched conversation ID mismatch")
        route, _case_id = manual_route(payload)
        if route is not None:
            repository.append_progress(
                run_id, stage="manual", progress={"message": "Manual conversation"}
            )
            repository.complete_run(
                run_id,
                result={"status": "manual_conversation", "reason": route, "retryable": False,
                        **({"case_id": _case_id, "owner_admission_required": True} if _case_id else {})},
            )
            return
        repository.append_progress(
            run_id, stage="process", progress={"message": "Processing ticket"}
        )
        recovery_guard = None
        if claimed.task_name == "ticket.recover":
            from services.support_recovery import guard_matches
            from db.database import get_ticket_log_by_helpscout_id
            recovery_guard = claimed.arguments
            row = get_ticket_log_by_helpscout_id(repository.db_path, claimed.conversation_id)
            if claimed.delivery_ceiling != "draft" or not guard_matches(payload, row, recovery_guard):
                repository.complete_run(run_id, result={"status": "recovery_held", "reason": "draft_or_obligation_changed"})
                return
            # Count all admitted attempts, including crashes, against this customer obligation.
            connection = _connect(repository.db_path)
            try:
                attempts = connection.execute(
                    "SELECT COUNT(*) FROM agent_runs WHERE task_name='ticket.recover' AND conversation_id=? "
                    "AND status != 'queued' AND json_extract(arguments_json, '$.expected_thread_id')=? "
                    "AND (result_json IS NULL OR COALESCE(json_extract(result_json, '$.reason'), '') "
                    "NOT IN ('draft_or_obligation_changed', 'recovery_exhausted'))",
                    (claimed.conversation_id, recovery_guard["expected_thread_id"]),
                ).fetchone()[0]
            finally:
                connection.close()
            if attempts > 2:
                repository.complete_run(run_id, result={"status": "recovery_held", "reason": "recovery_exhausted"})
                return
        kwargs = {"recovery_guard": recovery_guard} if recovery_guard else {}
        result = await active_processor.process_ticket(
            payload, trigger="manual_run", delivery_ceiling=claimed.delivery_ceiling, **kwargs
        )
        if not isinstance(result, dict) or result.get("status") == "error":
            repository.fail_run(
                run_id, error={"code": "processing_failed", "message": "Ticket processing failed."}
            )
            return
        repository.append_progress(
            run_id, stage="completed", progress={"message": "Processing complete"}
        )
        repository.complete_run(run_id, result=result)
    except asyncio.CancelledError:
        repository.interrupt_run(
            run_id, error={"code": "run_interrupted", "message": "Run interrupted."}
        )
        raise
    except Exception:
        repository.fail_run(
            run_id, error={"code": "processing_failed", "message": "Ticket processing failed."}
        )


@dataclass(frozen=True)
class AgentRunRecord:
    run_id: str
    idempotency_key: str | None
    token_id: str
    token_name: str
    task_name: str
    conversation_id: str
    status: RunStatus
    arguments: dict[str, Any]
    scopes: tuple[str, ...]
    delivery_ceiling: str | None
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_name": self.task_name,
            "conversation_id": self.conversation_id,
            "status": self.status,
            "arguments": self.arguments,
            "scopes": list(self.scopes),
            "delivery_ceiling": self.delivery_ceiling,
            "result": self.result,
            "error": self.error,
            "created_at": _timestamp(self.created_at),
            "updated_at": _timestamp(self.updated_at),
            "started_at": _optional_timestamp_text(self.started_at),
            "completed_at": _optional_timestamp_text(self.completed_at),
        }


@dataclass(frozen=True)
class AgentRunProgress:
    run_id: str
    sequence: int
    stage: str
    progress: dict[str, Any]
    created_at: datetime

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sequence": self.sequence,
            "stage": self.stage,
            "progress": self.progress,
            "created_at": _timestamp(self.created_at),
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("clock must return a timezone-aware datetime")
    try:
        if value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise ValueError("clock must return a valid timezone-aware datetime") from error


def _timestamp(value: datetime) -> str:
    return _aware_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _optional_timestamp_text(value: datetime | None) -> str | None:
    return None if value is None else _timestamp(value)


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        raise AgentRunDataError()
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        normalized = _timestamp(parsed)
    except (OverflowError, TypeError, ValueError) as error:
        raise AgentRunDataError() from error
    if value != normalized:
        raise AgentRunDataError()
    return parsed.astimezone(timezone.utc)


def _parse_optional_timestamp(value: Any) -> datetime | None:
    return None if value is None else _parse_timestamp(value)


def _bounded_text(value: Any, *, maximum: int = MAX_IDENTIFIER_LENGTH) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError("text value is invalid")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError("text value is invalid") from error
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("text value is invalid")
    return value


def _identifier(value: Any) -> str:
    normalized = _bounded_text(value)
    if _IDENTIFIER_RE.fullmatch(normalized) is None:
        raise ValueError("identifier is invalid")
    return normalized


def _task_name(value: Any) -> str:
    normalized = _bounded_text(value)
    if _TASK_NAME_RE.fullmatch(normalized) is None:
        raise ValueError("task name is invalid")
    return normalized


def _conversation_id(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("conversation ID is invalid")
    normalized = str(value)
    if _NUMERIC_ID_RE.fullmatch(normalized) is None or int(normalized) > MAX_RESOURCE_ID:
        raise ValueError("conversation ID is invalid")
    return normalized


def _idempotency_key(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _IDEMPOTENCY_KEY_RE.fullmatch(value) is None:
        raise ValueError("idempotency key is invalid")
    return value


def _redact_text(value: str) -> str:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError("JSON text contains invalid Unicode") from error
    if len(value) > MAX_TEXT_LENGTH:
        raise ValueError("JSON text exceeds the task run limit")
    value = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        value,
    )
    value = _BEARER_RE.sub("[REDACTED]", value)
    value = _API_SECRET_RE.sub("[REDACTED]", value)
    return _SUPA_TOKEN_RE.sub("[REDACTED]", value)


def _normalized_sensitive_key(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _numeric_json_id(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("numeric ID is invalid")
    normalized = str(value)
    if _NUMERIC_ID_RE.fullmatch(normalized) is None or int(normalized) > MAX_RESOURCE_ID:
        raise ValueError("numeric ID is invalid")
    return normalized


def _sanitize_json_value(
    value: Any,
    *,
    key: str | None,
    depth: int,
    seen: set[int],
    counter: list[int],
) -> Any:
    counter[0] += 1
    if counter[0] > MAX_RUN_JSON_ITEMS or depth > MAX_RUN_JSON_DEPTH:
        raise ValueError("task run JSON exceeds structural limits")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_RESOURCE_ID:
            raise ValueError("task run JSON integer is out of range")
        return _numeric_json_id(value) if key in _NUMERIC_ID_KEYS else value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("task run JSON contains a non-finite number")
        return value
    if isinstance(value, str):
        return _numeric_json_id(value) if key in _NUMERIC_ID_KEYS else _redact_text(value)
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen:
            raise ValueError("task run JSON contains a cycle")
        seen.add(marker)
        sanitized: dict[str, Any] = {}
        try:
            for raw_key, item in value.items():
                if not isinstance(raw_key, str):
                    raise ValueError("task run JSON keys must be strings")
                clean_key = _redact_text(raw_key)
                if (
                    not clean_key
                    or len(clean_key) > MAX_IDENTIFIER_LENGTH
                    or _normalized_sensitive_key(clean_key) in _SENSITIVE_KEYS
                ):
                    continue
                sanitized[clean_key] = _sanitize_json_value(
                    item,
                    key=clean_key,
                    depth=depth + 1,
                    seen=seen,
                    counter=counter,
                )
        finally:
            seen.remove(marker)
        return sanitized
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in seen:
            raise ValueError("task run JSON contains a cycle")
        seen.add(marker)
        try:
            return [
                _sanitize_json_value(
                    item,
                    key=None,
                    depth=depth + 1,
                    seen=seen,
                    counter=counter,
                )
                for item in value
            ]
        finally:
            seen.remove(marker)
    raise ValueError("task run JSON contains an unsupported value")


def _json_text(value: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("task run JSON must be an object")
    sanitized = _sanitize_json_value(
        value,
        key=None,
        depth=0,
        seen=set(),
        counter=[0],
    )
    assert isinstance(sanitized, dict)
    try:
        encoded = json.dumps(
            sanitized,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as error:
        raise ValueError("task run JSON is invalid") from error
    if len(encoded.encode("utf-8")) > MAX_RUN_JSON_BYTES:
        raise ValueError("task run JSON exceeds the task run limit")
    return encoded, sanitized


def _parse_json_mapping(value: Any, *, nullable: bool) -> dict[str, Any] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or len(value.encode("utf-8", errors="ignore")) > MAX_RUN_JSON_BYTES:
        raise AgentRunDataError()
    try:
        parsed = json.loads(value)
        encoded, sanitized = _json_text(parsed)
    except (RecursionError, TypeError, UnicodeError, ValueError) as error:
        raise AgentRunDataError() from error
    if encoded != value:
        raise AgentRunDataError()
    return sanitized


def _scope_text(scopes: Iterable[str]) -> tuple[str, tuple[str, ...]]:
    if isinstance(scopes, (str, bytes)):
        raise ValueError("scope snapshot is invalid")
    normalized = tuple(sorted({_identifier(scope) for scope in scopes}))
    return json.dumps(list(normalized), separators=(",", ":")), normalized


def _parse_scopes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or len(value) > MAX_RUN_JSON_BYTES:
        raise AgentRunDataError()
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as error:
        raise AgentRunDataError() from error
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise AgentRunDataError()
    try:
        encoded, normalized = _scope_text(parsed)
    except ValueError as error:
        raise AgentRunDataError() from error
    if encoded != value or list(normalized) != parsed:
        raise AgentRunDataError()
    return normalized


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
        raise AgentRunStorageError() from error


@contextmanager
def _write_transaction(db_path: PathValue | None) -> Iterator[sqlite3.Connection]:
    connection = _connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except AgentRunError:
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
        raise AgentRunStorageError() from error
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


def _validated_row(row: sqlite3.Row, required: frozenset[str]) -> sqlite3.Row:
    try:
        columns = frozenset(row.keys())
    except (AttributeError, TypeError) as error:
        raise AgentRunDataError() from error
    if not required.issubset(columns):
        raise AgentRunDataError()
    return row


def _stored_text(value: Any, *, validator: Callable[[Any], str]) -> str:
    try:
        normalized = validator(value)
    except ValueError as error:
        raise AgentRunDataError() from error
    if normalized != value:
        raise AgentRunDataError()
    return normalized


def _stored_run_id(value: Any) -> str:
    if not isinstance(value, str):
        raise AgentRunDataError()
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise AgentRunDataError() from error
    if parsed.version != 4 or str(parsed) != value:
        raise AgentRunDataError()
    return value


def _record_from_row(row: sqlite3.Row) -> AgentRunRecord:
    row = _validated_row(row, _RUN_ROW_COLUMNS)
    status = row["status"]
    if status not in _RUN_STATUSES:
        raise AgentRunDataError()
    delivery_ceiling = row["delivery_ceiling"]
    if delivery_ceiling not in {None, "draft", "policy"}:
        raise AgentRunDataError()
    idempotency_key = row["idempotency_key"]
    try:
        normalized_idempotency_key = _idempotency_key(idempotency_key)
    except ValueError as error:
        raise AgentRunDataError() from error
    created_at = _parse_timestamp(row["created_at"])
    updated_at = _parse_timestamp(row["updated_at"])
    started_at = _parse_optional_timestamp(row["started_at"])
    completed_at = _parse_optional_timestamp(row["completed_at"])
    arguments = _parse_json_mapping(row["arguments_json"], nullable=False)
    result = _parse_json_mapping(row["result_json"], nullable=True)
    error = _parse_json_mapping(row["error_json"], nullable=True)
    assert arguments is not None
    if status == "queued":
        consistent = started_at is None and completed_at is None and result is None and error is None
    elif status == "running":
        consistent = started_at is not None and completed_at is None and result is None and error is None
    elif status == "completed":
        consistent = started_at is not None and completed_at is not None and result is not None and error is None
    else:
        consistent = started_at is not None and completed_at is not None and result is None and error is not None
    if not consistent or updated_at < created_at:
        raise AgentRunDataError()
    return AgentRunRecord(
        run_id=_stored_run_id(row["run_id"]),
        idempotency_key=normalized_idempotency_key,
        token_id=_stored_text(row["token_id"], validator=_identifier),
        token_name=_stored_text(
            row["token_name"],
            validator=lambda value: _bounded_text(value, maximum=MAX_IDENTIFIER_LENGTH),
        ),
        task_name=_stored_text(row["task_name"], validator=_task_name),
        conversation_id=_stored_text(row["conversation_id"], validator=_conversation_id),
        status=status,
        arguments=arguments,
        scopes=_parse_scopes(row["scope_snapshot_json"]),
        delivery_ceiling=delivery_ceiling,
        result=result,
        error=error,
        created_at=created_at,
        updated_at=updated_at,
        started_at=started_at,
        completed_at=completed_at,
    )


def _progress_from_row(row: sqlite3.Row) -> AgentRunProgress:
    row = _validated_row(row, _EVENT_ROW_COLUMNS)
    sequence = row["sequence"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 0:
        raise AgentRunDataError()
    return AgentRunProgress(
        run_id=_stored_run_id(row["run_id"]),
        sequence=sequence,
        stage=_stored_text(row["stage"], validator=lambda value: _stage(value)),
        progress=_parse_json_mapping(row["progress_json"], nullable=False) or {},
        created_at=_parse_timestamp(row["created_at"]),
    )


def _stage(value: Any) -> str:
    normalized = _bounded_text(value)
    if _STAGE_RE.fullmatch(normalized) is None:
        raise ValueError("progress stage is invalid")
    return normalized


class AgentTaskRepository:
    """SQLite repository with guarded run transitions and owner-scoped reads."""

    def __init__(
        self,
        db_path: PathValue | None = None,
        *,
        clock: Clock | None = None,
        uuid_factory: UUIDFactory | None = None,
    ) -> None:
        self.db_path = db_path
        self._clock = clock or _utc_now
        self._uuid_factory = uuid_factory or uuid.uuid4

    def _now(self) -> datetime:
        return _aware_utc(self._clock())

    def create_task_run(
        self,
        *,
        token_id: str,
        token_name: str,
        task_name: str,
        arguments: Mapping[str, Any],
        scopes: Iterable[str],
        idempotency_key: str,
    ) -> AgentRunRecord:
        prepared = prepare_task_request(task_name, arguments, scopes=scopes)
        return self.create_run(
            token_id=token_id,
            token_name=token_name,
            task_name=prepared.task_name,
            conversation_id=prepared.arguments["conversation_id"],
            arguments=prepared.arguments,
            scopes=prepared.scopes,
            delivery_ceiling=prepared.delivery_ceiling,
            idempotency_key=idempotency_key,
        )

    def create_run(
        self,
        *,
        token_id: str,
        token_name: str,
        task_name: str,
        conversation_id: int | str,
        arguments: Mapping[str, Any] | None = None,
        scopes: Iterable[str] = (),
        delivery_ceiling: str | None = None,
        idempotency_key: str | None = None,
    ) -> AgentRunRecord:
        normalized_token_id = _identifier(token_id)
        normalized_token_name = _bounded_text(token_name)
        normalized_task_name = _task_name(task_name)
        normalized_conversation_id = _conversation_id(conversation_id)
        normalized_idempotency_key = _idempotency_key(idempotency_key)
        if delivery_ceiling not in {None, "draft", "policy"}:
            raise ValueError("delivery ceiling is invalid")
        arguments_text, _ = _json_text(arguments or {})
        scopes_text, _ = _scope_text(scopes)
        generated = self._uuid_factory()
        if not isinstance(generated, uuid.UUID) or generated.version != 4:
            raise ValueError("UUID factory must return a UUID4 value")
        run_id = str(generated)
        timestamp = _timestamp(self._now())
        with _write_transaction(self.db_path) as connection:
            if normalized_idempotency_key is not None:
                existing = connection.execute(
                    "SELECT * FROM agent_runs WHERE token_id = ? AND idempotency_key = ?",
                    (normalized_token_id, normalized_idempotency_key),
                ).fetchone()
                if existing is not None:
                    record = _record_from_row(existing)
                    if (
                        record.task_name != normalized_task_name
                        or record.conversation_id != normalized_conversation_id
                        or record.arguments != _json_text(arguments or {})[1]
                    ):
                        raise AgentTaskRequestError("idempotency_conflict")
                    return record
            if normalized_task_name == "ticket.recover":
                # Serialize reservation with insertion, across keys and token owners.
                # Replay above keeps its original identity; duplicates expose no other run.
                active = connection.execute(
                    "SELECT 1 FROM agent_runs WHERE task_name='ticket.recover' "
                    "AND conversation_id=? AND status IN ('queued', 'running') "
                    "AND json_extract(arguments_json, '$.expected_thread_id')=? LIMIT 1",
                    (normalized_conversation_id, (arguments or {}).get("expected_thread_id")),
                ).fetchone()
                if active is not None:
                    raise AgentRunError(
                        "Recovery is already queued or running for this customer thread.",
                        code="recovery_inflight", retryable=True, status=409, exit_code=5,
                    )
            connection.execute(
                """
                INSERT INTO agent_runs (
                    run_id, idempotency_key, token_id, token_name, task_name,
                    conversation_id, status, arguments_json,
                    scope_snapshot_json, delivery_ceiling, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    normalized_idempotency_key,
                    normalized_token_id,
                    normalized_token_name,
                    normalized_task_name,
                    normalized_conversation_id,
                    arguments_text,
                    scopes_text,
                    delivery_ceiling,
                    timestamp,
                    timestamp,
                ),
            )
            return self._select_required(connection, run_id)

    def get_run(self, run_id: str, *, token_id: str) -> AgentRunRecord:
        normalized_run_id = self._input_run_id(run_id)
        normalized_token_id = _identifier(token_id)
        connection = _connect(self.db_path)
        try:
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?", (normalized_run_id,)
            ).fetchone()
            if row is None:
                raise AgentRunNotFoundError(normalized_run_id)
            record = _record_from_row(row)
            if record.token_id != normalized_token_id:
                raise AgentRunNotFoundError(normalized_run_id)
            return record
        except sqlite3.Error as error:
            raise AgentRunStorageError() from error
        finally:
            connection.close()

    def get_run_unscoped(self, run_id: str) -> AgentRunRecord:
        """Internal-only run lookup for the scheduler; never expose through API."""
        normalized_run_id = self._input_run_id(run_id)
        connection = _connect(self.db_path)
        try:
            return self._select_required(connection, normalized_run_id)
        except sqlite3.Error as error:
            raise AgentRunStorageError() from error
        finally:
            connection.close()

    def list_queued_runs(self) -> list[AgentRunRecord]:
        connection = _connect(self.db_path)
        try:
            rows = connection.execute(
                """
                SELECT * FROM agent_runs
                WHERE status = 'queued'
                ORDER BY created_at, run_id
                """
            ).fetchall()
            return [_record_from_row(row) for row in rows]
        except sqlite3.Error as error:
            raise AgentRunStorageError() from error
        finally:
            connection.close()

    def claim_run(self, run_id: str) -> AgentRunRecord | None:
        normalized_run_id = self._input_run_id(run_id)
        timestamp = _timestamp(self._now())
        with _write_transaction(self.db_path) as connection:
            record = self._select_required(connection, normalized_run_id)
            if record.status != "queued":
                return None
            updated = connection.execute(
                """
                UPDATE agent_runs
                SET status = 'running', started_at = ?, updated_at = ?
                WHERE run_id = ? AND status = 'queued'
                """,
                (timestamp, timestamp, normalized_run_id),
            )
            if updated.rowcount != 1:
                return None
            return self._select_required(connection, normalized_run_id)

    def append_progress(
        self,
        run_id: str,
        *,
        stage: str,
        progress: Mapping[str, Any],
    ) -> AgentRunProgress:
        normalized_run_id = self._input_run_id(run_id)
        normalized_stage = _stage(stage)
        progress_text, _ = _json_text(progress)
        timestamp = _timestamp(self._now())
        with _write_transaction(self.db_path) as connection:
            record = self._select_required(connection, normalized_run_id)
            if record.status != "running":
                raise AgentRunStateError()
            rows = connection.execute(
                """
                SELECT * FROM agent_run_events
                WHERE run_id = ?
                ORDER BY sequence
                """,
                (normalized_run_id,),
            ).fetchall()
            prior = [_progress_from_row(row) for row in rows]
            if [event.sequence for event in prior] != list(range(1, len(prior) + 1)):
                raise AgentRunDataError()
            sequence = len(prior) + 1
            connection.execute(
                """
                INSERT INTO agent_run_events (
                    run_id, sequence, stage, progress_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (normalized_run_id, sequence, normalized_stage, progress_text, timestamp),
            )
            connection.execute(
                "UPDATE agent_runs SET updated_at = ? WHERE run_id = ? AND status = 'running'",
                (timestamp, normalized_run_id),
            )
            row = connection.execute(
                "SELECT * FROM agent_run_events WHERE run_id = ? AND sequence = ?",
                (normalized_run_id, sequence),
            ).fetchone()
            if row is None:
                raise AgentRunStorageError()
            return _progress_from_row(row)

    def list_run_events(
        self,
        run_id: str,
        *,
        token_id: str,
        after_sequence: int = 0,
    ) -> list[AgentRunProgress]:
        normalized_run_id = self._input_run_id(run_id)
        normalized_token_id = _identifier(token_id)
        if (
            not isinstance(after_sequence, int)
            or isinstance(after_sequence, bool)
            or after_sequence < 0
        ):
            raise ValueError("after sequence is invalid")
        connection = _connect(self.db_path)
        try:
            record = self._select_required(connection, normalized_run_id)
            if record.token_id != normalized_token_id:
                raise AgentRunNotFoundError(normalized_run_id)
            rows = connection.execute(
                """
                SELECT * FROM agent_run_events
                WHERE run_id = ? AND sequence > ?
                ORDER BY sequence
                """,
                (normalized_run_id, after_sequence),
            ).fetchall()
            events = [_progress_from_row(row) for row in rows]
            if events and events[0].sequence != after_sequence + 1:
                raise AgentRunDataError()
            if any(
                event.sequence != previous.sequence + 1
                for previous, event in zip(events, events[1:])
            ):
                raise AgentRunDataError()
            return events
        except sqlite3.Error as error:
            raise AgentRunStorageError() from error
        finally:
            connection.close()

    def complete_run(
        self, run_id: str, *, result: Mapping[str, Any]
    ) -> AgentRunRecord:
        return self._finish_run(run_id, status="completed", payload=result)

    def fail_run(
        self, run_id: str, *, error: Mapping[str, Any]
    ) -> AgentRunRecord:
        return self._finish_run(run_id, status="failed", payload=error)

    def interrupt_run(
        self, run_id: str, *, error: Mapping[str, Any]
    ) -> AgentRunRecord:
        return self._finish_run(run_id, status="interrupted", payload=error)

    def interrupt_stale_running(self) -> int:
        error_text, _ = _json_text(
            {
                "code": "run_interrupted",
                "message": "Run interrupted by service restart",
            }
        )
        timestamp = _timestamp(self._now())
        with _write_transaction(self.db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM agent_runs WHERE status = 'running' ORDER BY run_id"
            ).fetchall()
            records = [_record_from_row(row) for row in rows]
            for record in records:
                updated = connection.execute(
                    """
                    UPDATE agent_runs
                    SET status = 'interrupted', error_json = ?,
                        updated_at = ?, completed_at = ?
                    WHERE run_id = ? AND status = 'running'
                    """,
                    (error_text, timestamp, timestamp, record.run_id),
                )
                if updated.rowcount != 1:
                    raise AgentRunStateError()
                append_supa_event_in_transaction(
                    connection,
                    type="supa.run.failed",
                    dedupe_key=f"run:{record.run_id}:interrupted",
                    conversation_id=record.conversation_id,
                    run_id=record.run_id,
                    summary={"status": "interrupted"},
                    clock=self._clock,
                )
            return len(records)

    def _finish_run(
        self,
        run_id: str,
        *,
        status: Literal["completed", "failed", "interrupted"],
        payload: Mapping[str, Any],
    ) -> AgentRunRecord:
        normalized_run_id = self._input_run_id(run_id)
        payload_text, _ = _json_text(payload)
        timestamp = _timestamp(self._now())
        result_text = payload_text if status == "completed" else None
        error_text = None if status == "completed" else payload_text
        with _write_transaction(self.db_path) as connection:
            record = self._select_required(connection, normalized_run_id)
            if record.status != "running":
                raise AgentRunStateError()
            updated = connection.execute(
                """
                UPDATE agent_runs
                SET status = ?, result_json = ?, error_json = ?,
                    updated_at = ?, completed_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (
                    status,
                    result_text,
                    error_text,
                    timestamp,
                    timestamp,
                    normalized_run_id,
                ),
            )
            if updated.rowcount != 1:
                raise AgentRunStateError()
            append_supa_event_in_transaction(
                connection,
                type="supa.run.completed" if status == "completed" else "supa.run.failed",
                dedupe_key=f"run:{normalized_run_id}:{status}",
                conversation_id=record.conversation_id,
                run_id=normalized_run_id,
                summary={"status": status},
                clock=self._clock,
            )
            return self._select_required(connection, normalized_run_id)

    def _select_required(
        self, connection: sqlite3.Connection, run_id: str
    ) -> AgentRunRecord:
        row = connection.execute(
            "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise AgentRunNotFoundError(run_id)
        return _record_from_row(row)

    @staticmethod
    def _input_run_id(value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("run ID is invalid")
        try:
            parsed = uuid.UUID(value)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("run ID is invalid") from error
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError("run ID is invalid")
        return value


__all__ = [
    "MAX_RUN_JSON_BYTES",
    "AgentRunDataError",
    "AgentRunError",
    "AgentRunNotFoundError",
    "AgentRunProgress",
    "AgentRunRecord",
    "AgentRunStateError",
    "AgentRunStorageError",
    "AgentTaskRepository",
]
