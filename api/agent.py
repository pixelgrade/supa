"""Versioned, token-authenticated API surface for remote support agents."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import tomllib
import unicodedata
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Condition, Lock
from typing import Annotated, Any, Callable, Coroutine, Generic, Literal, TypeVar
from urllib.parse import quote
from uuid import uuid4

from services.company_pack import wording

from fastapi import APIRouter, Depends, Path as FastAPIPath, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from db.database import init_db
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_serializer,
)
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from services.agent_auth import (
    CONVERSATION_NOT_ALLOWED_CODE,
    AgentIdentity,
    normalize_conversation_id,
    require_scopes,
)
from services.agent_tasks import (
    TASKS, AgentRunError, AgentTaskRepository, run_evaluate_task, run_process_task,
)
from services.agent_events import (
    AgentEventProtocolError,
    AgentEventStorageError,
    acknowledge_event,
    lease_next_event,
    reset_event_consumer,
)
from services.agent_operations import (
    MAX_IDEMPOTENCY_KEY_LENGTH,
    AgentOperationError,
    get_operation,
)
from services.conversation_locks import ConversationLockError
from services.conversation_gateway import (
    AttachmentData,
    ConversationGateway,
    ConversationGatewayError,
)
from services.helpscout import (
    HelpScoutAttachmentTooLargeError,
    HelpScoutAuthError,
    HelpScoutClient,
    HelpScoutError,
    HelpScoutTransportError,
)
from services.runtime_paths import get_supa_db_path
from services import extensions, scheduled_sends


logger = logging.getLogger(__name__)
API_VERSION = "v1"
NO_STORE_HEADERS = {"Cache-Control": "no-store"}
PROJECT_METADATA_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"
AGENT_RESPONSE_MAX_BUFFERED_BYTES = 32 * 1024 * 1024
AGENT_RESPONSE_MAX_BUFFERED_MESSAGES = 10_000
MAX_RESOURCE_ID = (1 << 63) - 1
MAX_WRITE_BODY_BYTES = 1024 * 1024
EVENT_POLL_RECHECK_SECONDS = 1.0
AVAILABLE_READ_COMMANDS = (
    "attachment.download",
    "capabilities",
    "conversation.get",
    "conversation.create",
    "conversation.discard",
    "conversation.list",
    "conversation.search",
    "conversation.state",
    "request.add",
    "request.done",
    "draft.get",
    "draft.save",
    "draft.schedule",
    "draft.schedule.cancel",
    "draft.schedule.get",
    "draft.send",
    "health",
    "mailbox.operations",
    "note.add",
    "note.update",
    "operation.get",
    "reply.send",
    *extensions.agent_commands(),
    "desk",
    "conversation.brief",
    "conversation.precedents",
    "mailbox.insights",
)
AVAILABLE_EVENT_COMMANDS = ("events.ack", "events.next", "events.reset")
AVAILABLE_TASK_COMMANDS = ("ticket.evaluate", "ticket.process", "ticket.recover", *extensions.agent_tasks(), "run.get", "run.watch")
PLANNED_READ_COMMANDS: tuple[str, ...] = ()
_SAFE_ATTACHMENT_MIME_RE = re.compile(
    r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$"
)
_RFC5987_ATTR_CHAR_SAFE = "!#$&+-.^_`|~"
_GATEWAY_ERROR_MESSAGES = {
    "attachment_not_found": "Attachment not found",
    "conversation_not_allowed": "Token is not allowed to access this conversation",
    "invalid_conversation": "Upstream service failed",
    "invalid_cursor": "The conversation cursor is invalid.",
    "invalid_limit": "Invalid conversation list limit",
    "invalid_upstream_pagination": "Upstream service failed",
}
_WRITE_ERROR_MESSAGES = {
    "cc_blocked": "A cc address is on the do-not-contact list",
    "cc_check_failed": "The do-not-contact list could not be checked",
    "cc_not_allowed": (
        "A cc address was not given by the customer in this conversation"
    ),
    "invalid_cc": "Invalid cc: at most 2 plain email addresses",
    "conversation_not_discardable": "Only unsent draft conversations can be discarded",
    "insufficient_scope": "Required scope is missing",
    "invalid_subject": "Invalid conversation subject",
    "body_too_large": "The message body is too large",
    "conversation_busy": "Conversation is busy",
    "conversation_changed": "The conversation changed after it was read",
    "conversation_lock_lost": "The conversation lock was lost",
    "conversation_lock_storage_error": "The conversation lock store is unavailable",
    "conversation_locked": "Conversation is locked upstream",
    "conversation_not_allowed": "Token is not allowed to access this conversation",
    "conversation_not_open": "The conversation is closed or spam",
    "conversation_resolution_changed": "The conversation changed while locking",
    "draft_conflict": "The draft changed after it was read",
    "external_write_conflict": "The upstream write could not be verified",
    "idempotency_conflict": "Idempotency key conflict",
    "invalid_body": "Invalid message body",
    "invalid_body_format": "Invalid body format",
    "invalid_conversation": "Invalid conversation",
    "invalid_draft_version": "Invalid draft version",
    "invalid_note_version": "Invalid note version",
    "invalid_operation_result": "Stored operation result is invalid",
    "invalid_schedule": "Invalid schedule request",
    "invalid_send_at": "Invalid send time",
    "invalid_thread_id": "Invalid thread id",
    "network_error": "Upstream service is unavailable",
    "note_conflict": "The note changed after it was read",
    "note_not_editable": "Only notes written through the API can be edited",
    "note_not_found": "No internal note with that thread id",
    "oauth_network_error": "Upstream authentication service is unavailable",
    "oauth_request_timeout": "Upstream authentication timed out",
    "operation_data_error": "Stored operation data is invalid",
    "operation_in_progress": "Operation is already in progress",
    "operation_not_found": "Operation not found",
    "operation_state_conflict": "Operation state conflict",
    "operation_storage_error": "Operation store is unavailable",
    "outcome_unknown": "Upstream outcome is unknown",
    "request_timeout": "Upstream service timed out",
    "schedule_not_found": "No queued reply for this conversation",
    "schedule_sending": "The booked reply is being sent now",
    "upstream_auth_error": "Upstream authentication failed",
    "upstream_not_found": "Resource not found",
    "upstream_outcome_unknown": "Upstream outcome is unknown",
    "upstream_rate_limited": "Too many requests",
    "upstream_read_failed": "Upstream read failed",
    "upstream_rejected": "Upstream rejected the request",
    "upstream_server_error": "Upstream service failed",
    "wrong_expected_thread": (
        "--expected-thread must be the latest actionable thread "
        "(the customer's latest message), not the draft's thread."
    ),
}
_SAFE_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_EMAIL_RE = re.compile(r"^[a-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}@[a-z0-9.-]{1,253}$")
_TOKEN_LIKE_RE = re.compile(r"[A-Za-z0-9_-]{43,}")
_EVENT_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_EVENT_ACK_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43,512}$")
_EVENT_ERROR_MESSAGES = {
    "cursor_expired": "Event cursor expired",
    "event_ack_invalid": "Event acknowledgement is invalid or expired",
    "event_start_invalid": "Event start is invalid",
    "event_storage_error": "Event store is unavailable",
}


ResultT = TypeVar("ResultT")


class StrictResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TaskStartRequest(StrictRequestModel):
    task_name: str = Field(min_length=1, max_length=200)
    arguments: dict[str, Any]
    idempotency_key: str = Field(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)


def _validate_unicode_scalar_text(value: str, *, field_name: str) -> str:
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError(f"{field_name} contains invalid Unicode") from error
    if field_name == "body" and len(encoded) > MAX_WRITE_BODY_BYTES:
        raise ValueError("body is too large")
    return value


def _validate_idempotency_key(value: str) -> str:
    _validate_unicode_scalar_text(value, field_name="idempotency_key")
    if not value.strip() or _TOKEN_LIKE_RE.search(value):
        raise ValueError("invalid idempotency key")
    return value


class ConversationCreateRequest(StrictRequestModel):
    mailbox_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    customer_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    subject: str = Field(min_length=1, max_length=255)
    body: str = Field(min_length=1)
    body_format: Literal["html", "text"] = "text"
    draft: bool = True
    idempotency_key: str = Field(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)

    @field_validator("body", "subject")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_unicode_scalar_text(value, field_name="body")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_idempotency_key(value)


class ConversationDiscardRequest(StrictRequestModel):
    expected_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    idempotency_key: str = Field(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_idempotency_key(value)


class DraftWriteRequest(StrictRequestModel):
    body: str = Field(min_length=1)
    body_format: Literal["html", "text"]
    expected_version: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    idempotency_key: str = Field(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: str) -> str:
        return _validate_unicode_scalar_text(value, field_name="body")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_idempotency_key(value)


# CC on agent sends: at most two addresses the customer gave in the conversation. The
# gateway (normalize_cc) de-duplicates case-insensitively and then enforces the limit
# (`invalid_cc`); this bound only caps the payload.
CcList = list[Annotated[str, Field(min_length=3, max_length=320)]]


class DraftSendRequest(StrictRequestModel):
    expected_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_latest_thread_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    cc: CcList | None = Field(default=None, max_length=10)
    idempotency_key: str = Field(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_idempotency_key(value)


class ReplySendRequest(StrictRequestModel):
    body: str = Field(min_length=1)
    body_format: Literal["html", "text"]
    expected_latest_thread_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    cc: CcList | None = Field(default=None, max_length=10)
    idempotency_key: str = Field(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: str) -> str:
        return _validate_unicode_scalar_text(value, field_name="body")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_idempotency_key(value)


class DraftScheduleRequest(StrictRequestModel):
    """Book a reply: a body (preferred) or the saved draft at `expected_version`."""

    body: str | None = Field(default=None, min_length=1)
    body_format: Literal["html", "text"] | None = None
    expected_version: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_latest_thread_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    send_at: str | None = Field(default=None, min_length=1, max_length=64)
    after_customer_min_minutes: int | None = Field(default=None, ge=0, le=1440)
    after_customer_max_minutes: int | None = Field(default=None, ge=0, le=1440)
    cc: CcList | None = Field(default=None, max_length=10)
    idempotency_key: str = Field(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: str | None) -> str | None:
        return None if value is None else _validate_unicode_scalar_text(value, field_name="body")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_idempotency_key(value)


from supa_cli.installation import waiting_on_choices

_waiting_on_values = waiting_on_choices()
WaitingOnValue = Literal[_waiting_on_values]
_stages = extensions.stages()
StageValue = Literal[_stages] if _stages else str


class StateChanges(StrictRequestModel):
    """Typed conversation state; the mailbox keeps the tags older readers still use in step."""

    # Support conversations only (a campaign conversation is steered by stage, waiting state and
    # reminders). Only the owner can hand a conversation back to the agents.
    handling: Literal["person"] | None = None
    stage: StageValue | None = None
    waiting_on: WaitingOnValue | None = None
    reminders: Literal["on", "off"] | None = None
    has_site: bool | None = None
    paid: bool | None = None


class StateChangeRequest(StrictRequestModel):
    changes: StateChanges
    reason: str = Field(default="", max_length=1000)


class OwnerRequestAdd(StrictRequestModel):
    """One thing an agent needs from the mailbox owner, in a sentence or two."""

    text: str = Field(min_length=3, max_length=2000)
    idempotency_key: str = Field(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_unicode_scalar_text(value, field_name="text")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_idempotency_key(value)


class NoteAddRequest(StrictRequestModel):
    body: str = Field(min_length=1)
    body_format: Literal["html", "text"]
    idempotency_key: str = Field(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: str) -> str:
        return _validate_unicode_scalar_text(value, field_name="body")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_idempotency_key(value)


class NoteUpdateRequest(StrictRequestModel):
    body: str = Field(min_length=1)
    body_format: Literal["html", "text"]
    expected_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    idempotency_key: str = Field(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: str) -> str:
        return _validate_unicode_scalar_text(value, field_name="body")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_idempotency_key(value)


EventCursor = Literal["now", "oldest"] | Annotated[
    int,
    Field(ge=1, le=MAX_RESOURCE_ID),
]


def _validate_event_consumer(value: str) -> str:
    _validate_unicode_scalar_text(value, field_name="consumer")
    if _EVENT_IDENTIFIER_RE.fullmatch(value) is None or _TOKEN_LIKE_RE.search(value):
        raise ValueError("invalid event consumer")
    return value


class EventNextRequest(StrictRequestModel):
    consumer: str = Field(min_length=1, max_length=200)
    start: EventCursor = "now"
    wait: int = Field(default=0, ge=0, le=55)

    @field_validator("consumer")
    @classmethod
    def validate_consumer(cls, value: str) -> str:
        return _validate_event_consumer(value)


class EventAckRequest(StrictRequestModel):
    consumer: str = Field(min_length=1, max_length=200)
    ack_token: str = Field(min_length=43, max_length=512)

    @field_validator("consumer")
    @classmethod
    def validate_consumer(cls, value: str) -> str:
        return _validate_event_consumer(value)

    @field_validator("ack_token")
    @classmethod
    def validate_ack_token(cls, value: str) -> str:
        if _EVENT_ACK_TOKEN_RE.fullmatch(value) is None:
            raise ValueError("invalid acknowledgement token")
        return value


class EventResetRequest(StrictRequestModel):
    consumer: str = Field(min_length=1, max_length=200)
    to: EventCursor

    @field_validator("consumer")
    @classmethod
    def validate_consumer(cls, value: str) -> str:
        return _validate_event_consumer(value)


class AgentSuccessEnvelope(StrictResponseModel, Generic[ResultT]):
    """Versioned success contract used by generated OpenAPI and runtime validation."""

    api_version: Literal["v1"]
    request_id: str
    result: ResultT


class PersonResponse(StrictResponseModel):
    id: int | str | None
    type: str | None
    name: str | None
    email: str | None


class ThreadAuthorResponse(PersonResponse):
    role: Literal["customer", "agent", "system_user", "unknown"]


class ThreadSourceResponse(StrictResponseModel):
    type: str | None
    via: str | None


class AttachmentMetadataResponse(StrictResponseModel):
    id: int | str | None
    filename: str
    mime_type: str
    size: int = Field(ge=0)
    is_image: bool
    is_blank_transport_artifact: bool


class ThreadResponse(StrictResponseModel):
    id: int | str | None
    type: str | None
    state: str | None
    draft: bool
    is_internal_note: bool
    # True only for a draft created before a published agent reply (a leftover, not pending).
    superseded_by_reply: bool = False
    author: ThreadAuthorResponse
    source: ThreadSourceResponse
    body_html: str
    body_text: str
    created_at: str | None
    updated_at: str | None
    opened_at: str | None
    attachments: list[AttachmentMetadataResponse]
    # Internal notes only: the precondition for `note update`.
    version: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # The thread's To/CC addresses, only when it has any (e.g. our reply with --cc).
    to: list[str] | None = None
    cc: list[str] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_version(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if data.get("version") is None:
            data.pop("version", None)
        for header in ("to", "cc"):
            if not data.get(header):
                data.pop(header, None)
        return data


class DraftResponse(StrictResponseModel):
    thread_id: int | str | None
    type: str | None
    state: str | None
    body_html: str
    body_text: str
    created_at: str | None
    updated_at: str | None
    version: str = Field(pattern=r"^[0-9a-f]{64}$")
    # The draft's stored CC (sent with it by `draft send`), only when it has any.
    cc: list[str] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_cc(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if not data.get("cc"):
            data.pop("cc", None)
        return data


class MailboxResponse(StrictResponseModel):
    id: int | str | None
    name: str | None


class ConversationTimestampsResponse(StrictResponseModel):
    created_at: str | None
    updated_at: str | None
    user_updated_at: str | None
    closed_at: str | None
    customer_waiting_since: str | None


class ConversationResponse(StrictResponseModel):
    id: int | str | None
    requested_id: int | str | None
    resolved_id: int | str | None
    was_merged: bool
    number: int | str | None
    subject: str
    status: str | None
    tags: list[str]
    mailbox: MailboxResponse
    assignee: PersonResponse | None
    customer: PersonResponse | None
    timestamps: ConversationTimestampsResponse
    help_scout_url: str | None
    support_recovery: dict[str, Any] | None = None
    waiting_on: Literal["customer", "agent", "unknown"]
    latest_actionable_thread_id: int | str | None
    threads: list[ThreadResponse]
    draft: DraftResponse | None
    version: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_serializer(mode="wrap")
    def _omit_absent_recovery(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if data.get("support_recovery") is None:
            data.pop("support_recovery", None)
        return data

    @field_validator("latest_actionable_thread_id")
    @classmethod
    def validate_latest_actionable_thread_id(
        cls, value: int | str | None
    ) -> int | str | None:
        if value is None:
            return None
        if type(value) is int and 1 <= value <= MAX_RESOURCE_ID:
            return value
        if type(value) is str and value.isascii() and value.isdecimal():
            if 1 <= int(value) <= MAX_RESOURCE_ID:
                return value
        raise ValueError("invalid latest actionable thread id")


class ConversationListResponse(StrictResponseModel):
    conversations: list[ConversationResponse]
    next_cursor: str | None


class DraftReadResponse(StrictResponseModel):
    draft: DraftResponse | None
    latest_actionable_thread_id: int | str | None


class ConversationCreateResponse(StrictResponseModel):
    action: Literal["created"]
    conversation_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    status: Literal["created", "sent"]
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ConversationDiscardResponse(StrictResponseModel):
    action: Literal["discarded"]
    conversation_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    status: Literal["discarded"]


class DraftWriteResponse(StrictResponseModel):
    action: Literal["created", "updated"]
    conversation_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    thread_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    version: str = Field(pattern=r"^[0-9a-f]{64}$")


class SendWriteResponse(StrictResponseModel):
    action: Literal["sent"]
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conversation_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    status: Literal["sent", "sent_with_warning"]
    thread_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    warnings: list[str] = Field(max_length=100)
    # The CC addresses that went out with the reply, only when there were any.
    cc: list[str] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_cc(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if not data.get("cc"):
            data.pop("cc", None)
        return data

    @field_validator("warnings")
    @classmethod
    def validate_warning_codes(cls, value: list[str]) -> list[str]:
        if any(_SAFE_ERROR_CODE_RE.fullmatch(item) is None for item in value):
            raise ValueError("invalid warning code")
        return value


class NoteWriteResponse(StrictResponseModel):
    action: Literal["added"]
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conversation_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    conversation_status: Literal["active", "closed", "pending", "spam"]
    status: Literal["added", "added_with_warning"]
    thread_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    warnings: list[str] = Field(max_length=100)

    @field_validator("warnings")
    @classmethod
    def validate_warning_codes(cls, value: list[str]) -> list[str]:
        if any(_SAFE_ERROR_CODE_RE.fullmatch(item) is None for item in value):
            raise ValueError("invalid warning code")
        return value


class NoteUpdateResponse(StrictResponseModel):
    action: Literal["updated"]
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conversation_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    conversation_status: Literal["active", "closed", "pending", "spam"]
    status: Literal["updated", "updated_with_warning"]
    thread_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    # None only when the readback could not show the new body (then status warns).
    version: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    warnings: list[str] = Field(max_length=100)

    @field_validator("warnings")
    @classmethod
    def validate_warning_codes(cls, value: list[str]) -> list[str]:
        if any(_SAFE_ERROR_CODE_RE.fullmatch(item) is None for item in value):
            raise ValueError("invalid warning code")
        return value


class ScheduledSendResponse(StrictResponseModel):
    schedule_id: int = Field(ge=1)
    conversation_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    requested_conversation_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    status: Literal["queued", "sent", "held", "cancelled", "failed"]
    hold_reason: str | None
    send_at: str
    source: Literal["body", "draft"]
    draft_thread_id: int | None
    draft_version: str | None
    expected_latest_thread_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    customer_message_at: str | None
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cc: list[str] = Field(default_factory=list, max_length=2)
    token_name: str
    attempts: int = Field(ge=0)
    next_attempt_at: str | None
    last_error: str | None
    sent_thread_id: int | None
    created_at: str
    updated_at: str
    completed_at: str | None


class ScheduledSendReadResponse(StrictResponseModel):
    scheduled_send: ScheduledSendResponse | None


class ScheduledSendObservation(StrictResponseModel):
    """Body-free timing/hold facts for read identities; no scheduling authority."""
    status: Literal["queued", "sent", "held", "cancelled", "failed"]
    hold_reason: str | None
    send_at: str


class ConversationGetResponse(ConversationResponse):
    # Send identities retain their existing detailed record; readers see timing/hold facts only.
    scheduled_send: ScheduledSendResponse | ScheduledSendObservation | None = None


class OperationResponse(StrictResponseModel):
    idempotency_key: str = Field(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)
    operation_name: str
    target_id: str
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: Literal[
        "reserved",
        "in_progress",
        "succeeded",
        "sent_with_warning",
        "failed",
        "outcome_unknown",
    ]
    request_metadata: dict[str, Any]
    upstream_thread_id: str | None
    response: dict[str, Any] | None
    error: dict[str, Any] | None
    created_at: str
    updated_at: str
    upstream_started_at: str | None
    completed_at: str | None

    @field_validator("upstream_thread_id")
    @classmethod
    def validate_upstream_thread_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value.isascii() and value.isdecimal() and 1 <= int(value) <= MAX_RESOURCE_ID:
            return value
        raise ValueError("invalid upstream thread id")


class EventResponse(StrictResponseModel):
    event_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    type: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,199}$")
    conversation_id: str | None
    thread_id: str | None
    run_id: str | None
    source: Literal["helpscout", "supa"]
    actor_type: Literal["customer", "user", "system_user", "team", "system"] | None
    occurred_at: str
    received_at: str
    summary: dict[str, Any]
    ack_token: str = Field(pattern=r"^[A-Za-z0-9_-]{43,512}$")
    lease_expires_at: str


class EventNextResponse(StrictResponseModel):
    event: EventResponse | None


class EventAckResponse(StrictResponseModel):
    acknowledged: Literal[True]
    event_id: int = Field(ge=1, le=MAX_RESOURCE_ID)
    replayed: bool


class EventResetResponse(StrictResponseModel):
    acknowledged_event_id: int = Field(ge=0, le=MAX_RESOURCE_ID)
    consumer_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
    start_mode: Literal["now", "oldest", "event_id"]


ConversationId = Annotated[
    int,
    FastAPIPath(ge=1, le=MAX_RESOURCE_ID),
]
AttachmentId = Annotated[
    int,
    FastAPIPath(ge=1, le=MAX_RESOURCE_ID),
]
EventId = Annotated[
    int,
    FastAPIPath(ge=1, le=MAX_RESOURCE_ID),
]
ThreadId = Annotated[
    int,
    FastAPIPath(ge=1, le=MAX_RESOURCE_ID),
]


_conversation_gateway: ConversationGateway | None = None
_conversation_gateway_lock = Lock()


class _EventWaitNotifier:
    """Cross-thread wakeup generation used between bounded SQLite rechecks."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._generation = 0

    def snapshot(self) -> int:
        with self._condition:
            return self._generation

    def notify(self) -> None:
        with self._condition:
            self._generation += 1
            self._condition.notify_all()

    def _wait(self, generation: int, timeout: float) -> None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._generation != generation,
                timeout=timeout,
            )

    async def wait(self, generation: int, timeout: float) -> None:
        await asyncio.to_thread(self._wait, generation, timeout)


_event_wait_notifier = _EventWaitNotifier()
_event_poll_monotonic = time.monotonic


def _event_clock() -> datetime:
    return datetime.now(timezone.utc)


async def _wait_for_event_notification(generation: int, timeout: float) -> None:
    await _event_wait_notifier.wait(generation, timeout)


def notify_agent_event_waiters() -> None:
    """Wake in-process pollers; periodic rechecks also observe other processes."""
    _event_wait_notifier.notify()


def _event_feed_enabled() -> bool:
    return bool(os.getenv("HELPSCOUT_WEBHOOK_SECRET", ""))


def _available_agent_commands() -> list[str]:
    commands = list(AVAILABLE_READ_COMMANDS)
    commands.extend(AVAILABLE_TASK_COMMANDS)
    if _event_feed_enabled():
        commands.extend(AVAILABLE_EVENT_COMMANDS)
    return sorted(commands)


def _server_version() -> str:
    """Read installed project metadata, with a source-tree fallback."""
    try:
        return version("supa")
    except PackageNotFoundError:
        try:
            with PROJECT_METADATA_PATH.open("rb") as handle:
                project_version = tomllib.load(handle)["project"]["version"]
        except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
            return "unknown"
        if isinstance(project_version, str) and project_version.strip():
            return project_version.strip()
        return "unknown"


SERVER_VERSION = _server_version()


def _new_request_id() -> str:
    return f"req_{uuid4().hex}"


def agent_request_id(request: Request) -> str:
    request_id = getattr(request.state, "agent_request_id", None)
    if request_id is None:
        request_id = _new_request_id()
        request.state.agent_request_id = request_id
    return request_id


def success_response(request: Request, result: Any) -> JSONResponse:
    return JSONResponse(
        {
            "api_version": API_VERSION,
            "request_id": agent_request_id(request),
            "result": result,
        },
        headers=NO_STORE_HEADERS,
    )


def modeled_success_response(
    request: Request,
    result: Any,
    result_model: type[BaseModel],
) -> JSONResponse:
    """Validate an operation result before serializing the stable envelope."""
    try:
        validated = result_model.model_validate(result)
    except ValidationError:
        request_id = agent_request_id(request)
        logger.error(
            "Agent API response validation failed request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        return error_response(
            request_id=request_id,
            status_code=500,
            code="internal_error",
            message="Internal server error",
            retryable=True,
        )
    return success_response(request, validated.model_dump(mode="json"))


def error_response(
    *,
    request_id: str,
    status_code: int,
    code: str,
    message: str,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    response_headers = dict(headers or {})
    response_headers.update(NO_STORE_HEADERS)
    return JSONResponse(
        {
            "api_version": API_VERSION,
            "request_id": request_id,
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "details": details or {},
            },
        },
        status_code=status_code,
        headers=response_headers,
    )


def _http_error_contract(status_code: int) -> tuple[str, str, bool]:
    contracts = {
        400: ("invalid_request", "Invalid request", False),
        401: ("authentication_failed", "Authentication failed", False),
        403: ("insufficient_scope", "Insufficient scope", False),
        404: ("not_found", "Resource not found", False),
        405: ("method_not_allowed", "Method not allowed", False),
        409: ("conflict", "Request conflicts with current state", False),
        412: ("precondition_failed", "Precondition failed", False),
        422: ("invalid_request", "Invalid request", False),
        429: ("rate_limited", "Too many requests", True),
        502: ("upstream_error", "Upstream service failed", True),
        503: ("service_unavailable", "Service unavailable", True),
        504: ("upstream_timeout", "Upstream service timed out", True),
    }
    if status_code >= 500:
        return contracts.get(
            status_code, ("internal_error", "Internal server error", True)
        )
    return contracts.get(status_code, ("request_failed", "Request failed", False))


def _read_error_response(request: Request, exc: Exception) -> JSONResponse:
    """Map typed read failures without returning upstream text, bodies, or URLs."""
    request_id = agent_request_id(request)
    if isinstance(exc, ConversationGatewayError):
        status = exc.status if 400 <= exc.status <= 599 else 500
        return error_response(
            request_id=request_id,
            status_code=status,
            code=exc.code,
            message=_GATEWAY_ERROR_MESSAGES.get(
                exc.code,
                "Conversation read failed",
            ),
            retryable=exc.retryable,
        )

    if not isinstance(exc, HelpScoutError):  # pragma: no cover - caller narrows
        raise exc

    upstream_status = exc.status
    if isinstance(exc, HelpScoutAuthError):
        if upstream_status == 429 or (
            upstream_status is not None and 500 <= upstream_status <= 599
        ):
            status = upstream_status
        else:
            status = 502
        code = "upstream_auth_error"
        message = "Upstream authentication failed"
    elif upstream_status == 404:
        status, code, message = 404, "not_found", "Resource not found"
    elif upstream_status == 412:
        status, code, message = (
            412,
            "conversation_locked",
            "Conversation is locked upstream",
        )
    elif upstream_status == 429:
        status, code, message = 429, "rate_limited", "Too many requests"
    elif isinstance(exc, HelpScoutAttachmentTooLargeError):
        status, code, message = (
            413,
            "attachment_too_large",
            "Attachment exceeds the download limit",
        )
    elif isinstance(exc, HelpScoutTransportError) and exc.code == "request_timeout":
        status, code, message = (
            504,
            "upstream_timeout",
            "Upstream service timed out",
        )
    elif upstream_status is not None and 500 <= upstream_status <= 599:
        status, code, message = upstream_status, "upstream_error", "Upstream service failed"
    elif upstream_status is not None and 400 <= upstream_status <= 499:
        status, code, message = upstream_status, "upstream_error", "Upstream service failed"
    else:
        status, code, message = 502, "upstream_error", "Upstream service failed"

    details = {}
    delay = (exc.details or {}).get("retry_after_seconds")
    if code == "rate_limited" and type(delay) is int and 0 <= delay <= 120:
        details["retry_after_seconds"] = delay

    return error_response(
        request_id=request_id,
        status_code=status,
        code=code,
        message=message,
        retryable=exc.retryable,
        details=details,
    )


def _safe_write_error_details(exc: Exception) -> dict[str, Any]:
    """Return only documented scalar conflict details, never request/upstream text."""
    raw = getattr(exc, "details", None)
    if not isinstance(raw, dict):
        return {}
    details: dict[str, Any] = {}
    current_version = raw.get("current_version")
    if isinstance(current_version, str) and _SHA256_RE.fullmatch(current_version):
        details["current_version"] = current_version
    for key in (
        "thread_id", "latest_actionable_thread_id", "given_thread_id", "draft_thread_id"
    ):
        thread_id = raw.get(key)
        if (
            isinstance(thread_id, int)
            and not isinstance(thread_id, bool)
            and 1 <= thread_id <= MAX_RESOURCE_ID
        ):
            details[key] = thread_id
        elif key == "latest_actionable_thread_id" and key in raw and thread_id is None:
            details[key] = None
    refused_cc = raw.get("refused_cc")
    if isinstance(refused_cc, list):
        details["refused_cc"] = [
            item
            for item in refused_cc[:10]
            if isinstance(item, str) and _SAFE_EMAIL_RE.fullmatch(item)
        ]
    return details


def _write_error_response(request: Request, exc: Exception) -> JSONResponse:
    """Map typed mutation/ledger/lease failures without echoing their text."""
    if not isinstance(
        exc,
        (ConversationGatewayError, AgentOperationError, ConversationLockError),
    ):  # pragma: no cover - callers narrow this helper
        raise exc
    request_id = agent_request_id(request)
    status_value = getattr(exc, "status", 500)
    status = (
        status_value
        if isinstance(status_value, int) and 400 <= status_value <= 599
        else 500
    )
    code_value = getattr(exc, "code", "internal_error")
    code = (
        code_value
        if isinstance(code_value, str) and _SAFE_ERROR_CODE_RE.fullmatch(code_value)
        else "internal_error"
    )
    details = _safe_write_error_details(exc)
    if (
        code == "conversation_changed"
        and "latest_actionable_thread_id" in details
        and "given_thread_id" in details
    ):
        message = (
            f"The latest actionable thread is now {details['latest_actionable_thread_id']}, "
            f"not {details['given_thread_id']}."
        )
    elif code in _WRITE_ERROR_MESSAGES:
        message = _WRITE_ERROR_MESSAGES[code]
    elif status >= 500:
        message = "Upstream service failed"
    else:
        message = "Write request failed"
    return error_response(
        request_id=request_id,
        status_code=status,
        code=code,
        message=message,
        retryable=getattr(exc, "retryable", False) is True,
        details=details,
    )


def _safe_event_error_details(exc: Exception) -> dict[str, Any]:
    """Allow only documented event/cursor integers out of protocol errors."""
    raw = getattr(exc, "details", None)
    if not isinstance(raw, dict):
        return {}
    details: dict[str, Any] = {}
    for key in ("current_head_event_id", "event_id"):
        value = raw.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= MAX_RESOURCE_ID
        ):
            details[key] = value
    oldest = raw.get("oldest_retained_event_id")
    if oldest is None and "oldest_retained_event_id" in raw:
        details["oldest_retained_event_id"] = None
    elif (
        isinstance(oldest, int)
        and not isinstance(oldest, bool)
        and 1 <= oldest <= MAX_RESOURCE_ID
    ):
        details["oldest_retained_event_id"] = oldest
    return details


def _event_error_response(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, (AgentEventProtocolError, AgentEventStorageError)):
        raise exc  # pragma: no cover - callers narrow this helper
    status_value = getattr(exc, "status", 500)
    status = (
        status_value
        if isinstance(status_value, int) and 400 <= status_value <= 599
        else 500
    )
    code_value = getattr(exc, "code", "event_storage_error")
    code = (
        code_value
        if isinstance(code_value, str) and code_value in _EVENT_ERROR_MESSAGES
        else "event_storage_error"
    )
    return error_response(
        request_id=agent_request_id(request),
        status_code=status,
        code=code,
        message=_EVENT_ERROR_MESSAGES[code],
        retryable=getattr(exc, "retryable", False) is True,
        details=_safe_event_error_details(exc),
    )


def _event_feed_unavailable_response(request: Request) -> JSONResponse:
    return error_response(
        request_id=agent_request_id(request),
        status_code=503,
        code="event_feed_unavailable",
        message="Event feed is unavailable",
        retryable=False,
    )


def _unexpected_write_error_response(request: Request, exc: Exception) -> JSONResponse:
    """Log only the exception class and request metadata at the public write boundary."""
    request_id = agent_request_id(request)
    logger.error(
        "Agent API write failed request_id=%s method=%s path=%s exception_type=%s",
        request_id,
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    return error_response(
        request_id=request_id,
        status_code=500,
        code="internal_error",
        message="Internal server error",
        retryable=True,
    )


def _unexpected_event_error_response(request: Request, exc: Exception) -> JSONResponse:
    request_id = agent_request_id(request)
    logger.error(
        "Agent API event request failed request_id=%s method=%s path=%s exception_type=%s",
        request_id,
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    return error_response(
        request_id=request_id,
        status_code=500,
        code="internal_error",
        message="Internal server error",
        retryable=True,
    )


def _invalid_event_request_response(request: Request) -> JSONResponse:
    return error_response(
        request_id=agent_request_id(request),
        status_code=422,
        code="invalid_request",
        message="Invalid request",
    )


def _post_send_learning_callback(
    *,
    conversation_id: int,
    thread_id: int,
    subject: str,
    customer_message: str,
    reply_html: str,
    request_id: str,
) -> bool:
    """Lazy adapter for existing learning; the gateway offloads this sync callback."""
    del thread_id, request_id
    from services.processor import get_processor

    return get_processor().index_sent_reply(
        conversation_id=conversation_id,
        subject=subject,
        customer_message=customer_message,
        reply_text=reply_html,
        theme_tag=None,
    )


def get_conversation_gateway() -> ConversationGateway:
    """Build the production gateway; FastAPI dependency overrides keep tests isolated."""
    global _conversation_gateway
    with _conversation_gateway_lock:
        if _conversation_gateway is None:
            _conversation_gateway = ConversationGateway(
                HelpScoutClient(
                    client_id=os.getenv("HELPSCOUT_CLIENT_ID", ""),
                    client_secret=os.getenv("HELPSCOUT_CLIENT_SECRET", ""),
                ),
                db_path=get_supa_db_path(),
                post_send_bookkeeper=_post_send_learning_callback,
            )
        return _conversation_gateway


async def close_conversation_gateway() -> None:
    """Close and forget the shared Agent API Help Scout client, when created."""
    global _conversation_gateway
    with _conversation_gateway_lock:
        gateway = _conversation_gateway
        _conversation_gateway = None
    if gateway is not None:
        await gateway.client.close()


def _safe_attachment_media_type(claimed: str | None) -> str:
    if (
        not isinstance(claimed, str)
        or len(claimed) > 255
        or "\r" in claimed
        or "\n" in claimed
    ):
        return "application/octet-stream"
    media_type = claimed.split(";", 1)[0].strip().lower()
    if not _SAFE_ATTACHMENT_MIME_RE.fullmatch(media_type):
        return "application/octet-stream"
    return media_type


def _safe_attachment_filename(filename: str, attachment_id: int) -> tuple[str, str]:
    """Return an ASCII fallback and an RFC 5987 value for Content-Disposition."""
    value = unicodedata.normalize("NFKC", filename) if isinstance(filename, str) else ""
    cleaned = "".join(
        "_"
        if (
            character in {'/', '\\', '"'}
            or unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        )
        else character
        for character in value
    )
    cleaned = cleaned.strip(" .")[:180]
    if not cleaned:
        cleaned = f"attachment-{attachment_id}"

    ascii_fallback = "".join(
        character
        if 0x20 <= ord(character) < 0x7F and character not in {'/', '\\', '"'}
        else "_"
        for character in cleaned
    ).strip(" .")
    ascii_fallback = ascii_fallback[:120] or f"attachment-{attachment_id}"
    encoded = quote(cleaned, safe=_RFC5987_ATTR_CHAR_SAFE, encoding="utf-8")
    return ascii_fallback, encoded


def _attachment_response(request: Request, attachment: AttachmentData) -> Response:
    fallback, encoded = _safe_attachment_filename(
        attachment.filename,
        attachment.attachment_id,
    )
    return Response(
        content=attachment.data,
        media_type=_safe_attachment_media_type(attachment.claimed_mime_type),
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": (
                f'attachment; filename="{fallback}"; filename*=UTF-8\'\'{encoded}'
            ),
            "X-Request-ID": agent_request_id(request),
            "X-Supa-API-Version": API_VERSION,
            "X-Content-Type-Options": "nosniff",
        },
    )


def _is_agent_path(path: str) -> bool:
    return path == "/agent/v1" or path.startswith("/agent/v1/")


def _preserved_error_headers(response_start: Message) -> dict[str, str]:
    preserved_names = {b"allow", b"www-authenticate", b"retry-after"}
    return {
        name.decode("latin-1"): value.decode("latin-1")
        for name, value in response_start.get("headers", [])
        if name.lower() in preserved_names
    }


class AgentResponseTooLarge(Exception):
    """The bounded Agent API response exceeded a middleware safety limit."""


def _message_buffered_bytes(message: Message) -> int:
    total = 0
    body = message.get("body", b"")
    if isinstance(body, (bytes, bytearray, memoryview)):
        total += len(body)
    for name, value in message.get("headers", []):
        total += len(name) + len(value)
    return total


def _response_start(messages: list[Message]) -> Message | None:
    return next(
        (
            message
            for message in messages
            if message["type"] == "http.response.start"
        ),
        None,
    )


def _has_terminal_body(messages: list[Message]) -> bool:
    return any(
        message["type"] == "http.response.body"
        and not message.get("more_body", False)
        for message in messages
    )


def _contains_response_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, AgentResponseTooLarge):
        return True
    return any(
        _contains_response_limit_error(nested)
        for nested in getattr(exc, "exceptions", ())
    )


def _is_canonical_error_envelope(
    messages: list[Message],
    *,
    request_id: str,
) -> bool:
    body = b"".join(
        bytes(message.get("body", b""))
        for message in messages
        if message["type"] == "http.response.body"
    )
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(payload, dict) or set(payload) != {
        "api_version",
        "request_id",
        "error",
    }:
        return False
    error = payload.get("error")
    return (
        payload.get("api_version") == API_VERSION
        and payload.get("request_id") == request_id
        and isinstance(error, dict)
        and set(error) == {"code", "message", "retryable", "details"}
        and isinstance(error.get("code"), str)
        and bool(error["code"])
        and isinstance(error.get("message"), str)
        and bool(error["message"])
        and type(error.get("retryable")) is bool
        and isinstance(error.get("details"), dict)
    )


async def _send_http_response(
    response: Response,
    *,
    scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    if scope.get("method", "GET").upper() != "HEAD":
        await response(scope, receive, send)
        return

    async def send_without_body(message: Message) -> None:
        if message["type"] == "http.response.body":
            message = {**message, "body": b""}
        await send(message)

    await response(scope, receive, send_without_body)


async def _replay_http_messages(
    messages: list[Message],
    *,
    scope: Scope,
    send: Send,
) -> None:
    is_head = scope.get("method", "GET").upper() == "HEAD"
    for message in messages:
        if is_head and message["type"] == "http.response.body":
            message = {**message, "body": b""}
        await send(message)


async def _emit_buffered_agent_response(
    messages: list[Message],
    *,
    request_id: str,
    scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    response_start = _response_start(messages)
    if response_start is None:  # pragma: no cover - guarded before this helper
        raise RuntimeError("Agent API returned no response start")

    status_code = response_start["status"]
    if 300 <= status_code < 400:
        code, message, retryable = _http_error_contract(404)
        response = error_response(
            request_id=request_id,
            status_code=404,
            code=code,
            message=message,
            retryable=retryable,
            headers=_preserved_error_headers(response_start),
        )
        await _send_http_response(
            response,
            scope=scope,
            receive=receive,
            send=send,
        )
        return
    if status_code >= 400 and not _is_canonical_error_envelope(
        messages,
        request_id=request_id,
    ):
        code, message, retryable = _http_error_contract(status_code)
        response = error_response(
            request_id=request_id,
            status_code=status_code,
            code=code,
            message=message,
            retryable=retryable,
            headers=_preserved_error_headers(response_start),
        )
        await _send_http_response(
            response,
            scope=scope,
            receive=receive,
            send=send,
        )
        return

    response_start["headers"] = [
        (name, value)
        for name, value in response_start.get("headers", [])
        if name.lower() != b"cache-control"
    ] + [(b"cache-control", b"no-store")]
    await _replay_http_messages(messages, scope=scope, send=send)


class AgentAPIBoundaryMiddleware:
    """Buffer bounded Agent API responses so failures never leak partial output.

    Agent v1 responses are contractually bounded. Attachment endpoints enforce a byte
    limit, so buffering the full agent response is safe. Non-agent responses bypass this
    middleware without buffering, preserving the dashboard and webhook streaming behavior.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_buffered_bytes: int = AGENT_RESPONSE_MAX_BUFFERED_BYTES,
        max_buffered_messages: int = AGENT_RESPONSE_MAX_BUFFERED_MESSAGES,
    ):
        if max_buffered_bytes <= 0 or max_buffered_messages <= 0:
            raise ValueError("Agent API response buffer limits must be positive")
        self.app = app
        self.max_buffered_bytes = max_buffered_bytes
        self.max_buffered_messages = max_buffered_messages

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _is_agent_path(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        request_id = agent_request_id(request)
        messages: list[Message] = []
        buffered_bytes = 0
        response_limit_error: AgentResponseTooLarge | None = None

        async def buffer_message(message: Message) -> None:
            nonlocal buffered_bytes, response_limit_error
            if response_limit_error is not None:
                raise response_limit_error
            next_message_count = len(messages) + 1
            next_buffered_bytes = buffered_bytes + _message_buffered_bytes(message)
            if (
                next_message_count > self.max_buffered_messages
                or next_buffered_bytes > self.max_buffered_bytes
            ):
                response_limit_error = AgentResponseTooLarge(
                    "Agent API response exceeded its buffer limit"
                )
                raise response_limit_error
            messages.append(dict(message))
            buffered_bytes = next_buffered_bytes

        try:
            await self.app(scope, receive, buffer_message)
            if _response_start(messages) is None or not _has_terminal_body(messages):
                raise RuntimeError("Agent API returned an incomplete ASGI response")
        except Exception as exc:
            if _contains_response_limit_error(exc):
                logger.exception(
                    "Agent API response buffer limit exceeded request_id=%s method=%s path=%s",
                    request_id,
                    scope.get("method", ""),
                    scope.get("path", ""),
                )
                response = error_response(
                    request_id=request_id,
                    status_code=500,
                    code="response_too_large",
                    message="Agent API response exceeded its buffer limit",
                    retryable=False,
                )
                await _send_http_response(
                    response,
                    scope=scope,
                    receive=receive,
                    send=send,
                )
                return
            if _response_start(messages) is not None and _has_terminal_body(messages):
                logger.exception(
                    "Agent API post-completion exception request_id=%s method=%s path=%s",
                    request_id,
                    scope.get("method", ""),
                    scope.get("path", ""),
                )
                await _emit_buffered_agent_response(
                    messages,
                    request_id=request_id,
                    scope=scope,
                    receive=receive,
                    send=send,
                )
                return
            logger.exception(
                "Unhandled Agent API exception request_id=%s method=%s path=%s",
                request_id,
                scope.get("method", ""),
                scope.get("path", ""),
            )
            code, message, retryable = _http_error_contract(500)
            response = error_response(
                request_id=request_id,
                status_code=500,
                code=code,
                message=message,
                retryable=retryable,
            )
            await _send_http_response(
                response,
                scope=scope,
                receive=receive,
                send=send,
            )
            return

        await _emit_buffered_agent_response(
            messages,
            request_id=request_id,
            scope=scope,
            receive=receive,
            send=send,
        )


class AgentAPIRoute(APIRoute):
    """Apply the Agent API contract without changing any non-agent route."""

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        route_handler = super().get_route_handler()

        async def contracted_handler(request: Request) -> Response:
            request_id = agent_request_id(request)
            request.state.agent_route_matched = True
            try:
                response = await route_handler(request)
            except RequestValidationError:
                return error_response(
                    request_id=request_id,
                    status_code=422,
                    code="invalid_request",
                    message="Invalid request",
                )
            except HTTPException as exc:
                code, message, retryable = _http_error_contract(exc.status_code)
                return error_response(
                    request_id=request_id,
                    status_code=exc.status_code,
                    code=code,
                    message=message,
                    retryable=retryable,
                    headers=exc.headers,
                )
            response.headers["Cache-Control"] = "no-store"
            return response

        return contracted_handler


router = APIRouter(
    prefix="/agent/v1",
    tags=["agent"],
    route_class=AgentAPIRoute,
)


_active_agent_run_tasks: set[asyncio.Task] = set()
_active_agent_run_ids: set[str] = set()


def schedule_agent_run(run_id: str) -> None:
    """Keep a strong task reference; duplicate schedules are safe and ignored."""
    if run_id in _active_agent_run_ids:
        return
    _active_agent_run_ids.add(run_id)

    async def execute() -> None:
        repository = AgentTaskRepository(get_supa_db_path())
        try:
            # The repository claim is the cross-process execution gate.
            record = repository.get_run_unscoped(run_id)
            if record.task_name == "ticket.evaluate":
                await run_evaluate_task(repository, run_id)
            elif record.task_name in {"ticket.process", "ticket.recover"}:
                await run_process_task(repository, run_id)
            elif record.task_name in extensions.agent_tasks():
                await extensions.agent_tasks()[record.task_name].run(repository, run_id)
        except Exception:
            logger.exception("Agent run execution failed: %s", run_id)

    task = asyncio.create_task(execute(), name=f"agent-run:{run_id}")
    _active_agent_run_tasks.add(task)

    def finished(done: asyncio.Task) -> None:
        _active_agent_run_tasks.discard(done)
        _active_agent_run_ids.discard(run_id)
        try:
            done.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Agent run task crashed: %s", run_id)

    task.add_done_callback(finished)


def recover_agent_runs(db_path: str) -> tuple[int, int]:
    """Interrupt orphaned in-flight runs and schedule durable queued work."""
    init_db(db_path)
    repository = AgentTaskRepository(db_path)
    interrupted = repository.interrupt_stale_running()
    queued = repository.list_queued_runs()
    for run in queued:
        schedule_agent_run(run.run_id)
    return interrupted, len(queued)


async def close_agent_run_tasks() -> None:
    """Cancel and await in-process runs so each can persist interruption."""
    tasks = tuple(_active_agent_run_tasks)
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _task_error_response(request: Request, error: AgentRunError) -> JSONResponse:
    return error_response(
        request_id=agent_request_id(request), status_code=error.status,
        code=error.code, message=str(error), retryable=error.retryable,
    )


def _conversation_not_allowed_response(request: Request) -> JSONResponse:
    """The one refusal for every conversation outside a token's allowlist.

    It never names the conversation or says whether it exists.
    """
    return error_response(
        request_id=agent_request_id(request),
        status_code=403,
        code=CONVERSATION_NOT_ALLOWED_CODE,
        message=_WRITE_ERROR_MESSAGES[CONVERSATION_NOT_ALLOWED_CODE],
        retryable=False,
    )


def _resolved_conversation_allowed(identity: AgentIdentity, result: Any) -> bool:
    """A read result is visible only if its resolved (post-merge) id is allowed."""
    if not identity.is_conversation_restricted:
        return True
    if not isinstance(result, dict):
        return False
    resolved = result.get("resolved_id")
    if resolved is None:
        resolved = result.get("id")
    return identity.allows_conversation(resolved) and identity.allows_conversation(
        result.get("id")
    )


def _task_target_allowed(identity: AgentIdentity, arguments: Any) -> bool:
    """Every named task targets one conversation; a restricted token needs it listed."""
    if not identity.is_conversation_restricted:
        return True
    if not isinstance(arguments, dict):
        return False
    return identity.allows_conversation(
        normalize_conversation_id(arguments.get("conversation_id"))
    )


@router.post("/tasks")
async def start_agent_task(
    request: Request,
    payload: TaskStartRequest,
    identity: AgentIdentity = Depends(require_scopes("task")),
) -> JSONResponse:
    if not _task_target_allowed(identity, payload.arguments):
        return _conversation_not_allowed_response(request)
    try:
        repository = AgentTaskRepository(get_supa_db_path())
        run = repository.create_task_run(
            token_id=identity.token_id, token_name=identity.name,
            task_name=payload.task_name, arguments=payload.arguments,
            scopes=identity.scopes, idempotency_key=payload.idempotency_key,
        )
        if run.status == "queued":
            schedule_agent_run(run.run_id)
    except AgentRunError as error:
        return _task_error_response(request, error)
    except ValueError:
        return error_response(
            request_id=agent_request_id(request), status_code=422,
            code="invalid_request", message="Invalid request",
        )
    return success_response(request, run.to_public_dict())


@router.get("/runs/{run_id}")
async def get_agent_run(
    request: Request, run_id: str,
    identity: AgentIdentity = Depends(require_scopes("task")),
) -> JSONResponse:
    try:
        run = AgentTaskRepository(get_supa_db_path()).get_run(
            run_id, token_id=identity.token_id
        )
    except AgentRunError as error:
        return _task_error_response(request, error)
    except ValueError:
        return error_response(
            request_id=agent_request_id(request), status_code=422,
            code="invalid_request", message="Invalid request",
        )
    return success_response(request, run.to_public_dict())


@router.get("/runs/{run_id}/events")
async def get_agent_run_events(
    request: Request, run_id: str,
    after: Annotated[int, Query(ge=0)] = 0,
    wait: Annotated[int, Query(ge=0, le=55)] = 0,
    identity: AgentIdentity = Depends(require_scopes("task")),
) -> JSONResponse:
    repository = AgentTaskRepository(get_supa_db_path())
    deadline = time.monotonic() + wait
    try:
        while True:
            run = repository.get_run(run_id, token_id=identity.token_id)
            events = repository.list_run_events(
                run_id, token_id=identity.token_id, after_sequence=after
            )
            if events or run.status in {"completed", "failed", "interrupted"} or time.monotonic() >= deadline:
                break
            await asyncio.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    except AgentRunError as error:
        return _task_error_response(request, error)
    except ValueError:
        return error_response(
            request_id=agent_request_id(request), status_code=422,
            code="invalid_request", message="Invalid request",
        )
    return success_response(request, {
        "run": run.to_public_dict(),
        "events": [event.to_public_dict() for event in events],
    })


@router.get("/health")
def health(
    request: Request,
    _identity: AgentIdentity = Depends(require_scopes("read")),
) -> JSONResponse:
    return success_response(
        request,
        {"status": "ok", "server_version": SERVER_VERSION},
    )


@router.get("/mailbox/operations")
async def mailbox_operations(request: Request, identity: AgentIdentity = Depends(require_scopes("read"))) -> JSONResponse:
    if identity.is_conversation_restricted:
        return _conversation_not_allowed_response(request)
    from services import mailbox_provider
    if mailbox_provider.native_origin() is None:
        return success_response(request, {"provider": "vendor"})
    try:
        summary = await get_conversation_gateway().client.native_operations()
    except (HelpScoutError, ValueError):
        return error_response(request_id=agent_request_id(request), status_code=503,
                              code="mailbox_status_unavailable", message="Mailbox status unavailable", retryable=True)
    return success_response(request, {"provider": "native", **summary})


async def _mailbox_report(request, identity, gateway, path, params, *, question=None):
    if identity.is_conversation_restricted:
        return _conversation_not_allowed_response(request)
    try:
        result = await gateway.client.native_report(path, params, question=question)
    except HelpScoutError as exc:
        return _read_error_response(request, exc)
    return success_response(request, result)


@router.get("/desk")
async def review_desk(request: Request, p: Annotated[int, Query(ge=1, le=1_000_000)] = 1,
                      identity: AgentIdentity = Depends(require_scopes("read")),
                      gateway: ConversationGateway = Depends(get_conversation_gateway)):
    return await _mailbox_report(request, identity, gateway, "/desk", {"p": p})


@router.get("/mailbox/insights")
async def mailbox_insights(request: Request, months: int = 12,
                           identity: AgentIdentity = Depends(require_scopes("read")),
                           gateway: ConversationGateway = Depends(get_conversation_gateway)):
    if months not in (3, 6, 12, 24):
        return error_response(request_id=agent_request_id(request), status_code=422,
                              code="invalid_request", message="Choose 3, 6, 12 or 24 months.")
    return await _mailbox_report(request, identity, gateway, "/insights", {"months": months})


class PrecedentQuestion(StrictRequestModel):
    q: str = Field(min_length=1, max_length=32000)
    tag: str = Field(default="", max_length=80)


@router.post("/precedents")
async def question_precedents(request: Request, question: PrecedentQuestion,
                              identity: AgentIdentity = Depends(require_scopes("read")),
                              gateway: ConversationGateway = Depends(get_conversation_gateway)):
    return await _mailbox_report(request, identity, gateway, "/precedents", {}, question=question.model_dump())


@router.get("/conversations/{conversation_id}/precedents")
async def conversation_precedents(request: Request, conversation_id: ConversationId,
                                  identity: AgentIdentity = Depends(require_scopes("read")),
                                  gateway: ConversationGateway = Depends(get_conversation_gateway)):
    return await _mailbox_report(request, identity, gateway, "/precedents", {"conversation": conversation_id})


@router.get("/conversations/{conversation_id}/brief")
async def conversation_brief(request: Request, conversation_id: ConversationId,
                             max_chars: Annotated[int, Query(ge=1000, le=200000)] = 24000,
                             precedents: Annotated[int, Query(ge=0, le=10)] = 0, quotes: bool = False,
                             identity: AgentIdentity = Depends(require_scopes("read")),
                             gateway: ConversationGateway = Depends(get_conversation_gateway)):
    if not identity.allows_conversation(conversation_id) or (identity.is_conversation_restricted and precedents):
        return _conversation_not_allowed_response(request)
    try:
        result = await gateway.client.native_report(f"/conversations/{conversation_id}/brief",
            {"max_chars": max_chars, "quotes": int(quotes), "precedents": precedents})
    except HelpScoutError as exc:
        return _read_error_response(request, exc)
    if identity.is_conversation_restricted:
        if not _resolved_conversation_allowed(identity, result):
            return _conversation_not_allowed_response(request)
        # Opt in to fields from this exchange. History, account-wide email addresses and
        # future report sections need an unrestricted identity.
        customer = result.get("customer") or {}
        result = {k: result[k] for k in ("id", "number", "subject", "status", "state", "tags", "assignee",
                  "created", "modified", "waiting_since", "snoozed_until", "messages", "omitted_messages",
                  "draft", "intake", "links") if k in result}
        result["customer"] = {k: customer[k] for k in ("id", "name", "email") if k in customer}
    return success_response(request, result)


extensions.install_agent_routes(router)


@router.get("/capabilities")
def capabilities(
    request: Request,
    identity: AgentIdentity = Depends(require_scopes("read")),
) -> JSONResponse:
    return success_response(
        request,
        {
            "api_version": API_VERSION,
            "server_version": SERVER_VERSION,
            "identity": {
                "name": identity.name,
                "token_id": identity.token_id,
            },
            "scopes": sorted(identity.scopes),
            # null = unrestricted; a list = the only conversations this token
            # may read, write, receive events for, or start tasks on.
            "conversations": (
                sorted(identity.conversations)
                if identity.conversations is not None
                else None
            ),
            "commands": {
                "available": _available_agent_commands(),
                "planned": list(PLANNED_READ_COMMANDS),
            },
            "tasks": [
                {"name": name, "required_scopes": sorted(definition.required_scopes)}
                for name, definition in TASKS.items()
            ],
        },
    )


@router.post("/conversations", response_model=AgentSuccessEnvelope[ConversationCreateResponse])
async def post_new_conversation(
    request: Request, payload: ConversationCreateRequest,
    identity: AgentIdentity = Depends(require_scopes("create")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    from services.conversation_lifecycle import create_conversation
    try:
        result = await create_conversation(
            gateway, **payload.model_dump(), identity=identity,
            request_id=agent_request_id(request),
        )
    except (ConversationGatewayError, AgentOperationError, ConversationLockError) as exc:
        return _write_error_response(request, exc)
    except Exception as exc:
        return _unexpected_write_error_response(request, exc)
    return modeled_success_response(request, result, ConversationCreateResponse)


@router.post("/conversations/{conversation_id}/discard",
             response_model=AgentSuccessEnvelope[ConversationDiscardResponse])
async def discard_unsent_conversation(
    request: Request, payload: ConversationDiscardRequest, conversation_id: ConversationId,
    identity: AgentIdentity = Depends(require_scopes("delete")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    from services.conversation_lifecycle import discard_conversation
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    try:
        result = await discard_conversation(
            gateway, conversation_id, **payload.model_dump(), identity=identity,
            request_id=agent_request_id(request),
        )
    except (ConversationGatewayError, AgentOperationError, ConversationLockError) as exc:
        return _write_error_response(request, exc)
    except Exception as exc:
        return _unexpected_write_error_response(request, exc)
    return modeled_success_response(request, result, ConversationDiscardResponse)


@router.get(
    "/conversations",
    response_model=AgentSuccessEnvelope[ConversationListResponse],
)
async def list_conversations(
    request: Request,
    status: Annotated[str, Query(min_length=1, max_length=64)] = "all",
    tag: Annotated[str | None, Query(min_length=1, max_length=255)] = None,
    query: Annotated[str | None, Query(min_length=1, max_length=2_000)] = None,
    cursor: Annotated[str | None, Query(min_length=1, max_length=256)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    identity: AgentIdentity = Depends(require_scopes("read")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    try:
        result = await gateway.list(
            status=status,
            tag=tag,
            query=query,
            cursor=cursor,
            limit=limit,
        )
    except (ConversationGatewayError, HelpScoutError) as exc:
        return _read_error_response(request, exc)
    if identity.is_conversation_restricted:
        # Filter rather than refuse: the upstream page is fetched as usual and
        # only allowed conversations are returned. A page may therefore hold
        # fewer than `limit` items while next_cursor still advances.
        if not isinstance(result, dict) or not isinstance(
            result.get("conversations"), list
        ):
            return modeled_success_response(request, result, ConversationListResponse)
        result = {
            **result,
            "conversations": [
                conversation
                for conversation in result["conversations"]
                if _resolved_conversation_allowed(identity, conversation)
            ],
        }
    return modeled_success_response(request, result, ConversationListResponse)


@router.get(
    "/conversations/{conversation_id}",
    response_model=AgentSuccessEnvelope[ConversationGetResponse],
)
async def get_conversation(
    request: Request,
    conversation_id: ConversationId,
    identity: AgentIdentity = Depends(require_scopes("read")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    try:
        result = await gateway.get(conversation_id)
    except (ConversationGatewayError, HelpScoutError) as exc:
        return _read_error_response(request, exc)
    if not _resolved_conversation_allowed(identity, result):
        return _conversation_not_allowed_response(request)
    if isinstance(result, dict):
        if "send" in identity.scopes:
            result = {**result, "scheduled_send": _scheduled_send_summary(result)}
        else:
            try:
                cid = result.get("resolved_id") or result.get("id")
                row = scheduled_sends.latest_for_conversation(get_supa_db_path(), cid)
                observation = {key: row[key] for key in ("status", "send_at", "hold_reason")} if row is not None else None
                result = {**result, "scheduled_send": observation}
            except Exception:
                return error_response(request_id=agent_request_id(request), status_code=503,
                    code="schedule_observation_unavailable", message="Booking observation unavailable", retryable=True)
    return modeled_success_response(request, result, ConversationGetResponse)


def _scheduled_send_summary(conversation: dict[str, Any]) -> dict[str, Any] | None:
    """The conversation's newest booked reply; a local read that never fails the get."""
    conversation_id = conversation.get("resolved_id") or conversation.get("id")
    if not isinstance(conversation_id, int):
        return None
    try:
        row = scheduled_sends.latest_for_conversation(get_supa_db_path(), conversation_id)
    except Exception:
        logger.warning("Scheduled send lookup failed for conversation %s", conversation_id)
        return None
    return scheduled_sends.public_record(row) if row is not None else None


@router.get(
    "/conversations/{conversation_id}/draft",
    response_model=AgentSuccessEnvelope[DraftReadResponse],
)
async def get_conversation_draft(
    request: Request,
    conversation_id: ConversationId,
    identity: AgentIdentity = Depends(require_scopes("read")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    try:
        conversation = await gateway.get(conversation_id)
    except (ConversationGatewayError, HelpScoutError) as exc:
        return _read_error_response(request, exc)
    if not _resolved_conversation_allowed(identity, conversation):
        return _conversation_not_allowed_response(request)
    return modeled_success_response(
        request,
        {
            "draft": conversation.get("draft"),
            "latest_actionable_thread_id": conversation.get("latest_actionable_thread_id"),
        },
        DraftReadResponse,
    )


@router.put(
    "/conversations/{conversation_id}/draft",
    response_model=AgentSuccessEnvelope[DraftWriteResponse],
)
async def put_conversation_draft(
    request: Request,
    payload: DraftWriteRequest,
    conversation_id: ConversationId,
    identity: AgentIdentity = Depends(require_scopes("draft")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    try:
        result = await gateway.save_draft(
            conversation_id,
            body=payload.body,
            body_format=payload.body_format,
            expected_version=payload.expected_version,
            idempotency_key=payload.idempotency_key,
            identity=identity,
            request_id=agent_request_id(request),
        )
    except (ConversationGatewayError, AgentOperationError, ConversationLockError) as exc:
        return _write_error_response(request, exc)
    except Exception as exc:
        return _unexpected_write_error_response(request, exc)
    return modeled_success_response(request, result, DraftWriteResponse)


@router.post(
    "/conversations/{conversation_id}/draft/send",
    response_model=AgentSuccessEnvelope[SendWriteResponse],
)
async def send_conversation_draft(
    request: Request,
    payload: DraftSendRequest,
    conversation_id: ConversationId,
    identity: AgentIdentity = Depends(require_scopes("send")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    try:
        result = await gateway.send_draft(
            conversation_id,
            expected_version=payload.expected_version,
            expected_latest_thread_id=payload.expected_latest_thread_id,
            idempotency_key=payload.idempotency_key,
            identity=identity,
            request_id=agent_request_id(request),
            cc=payload.cc,
        )
    except (ConversationGatewayError, AgentOperationError, ConversationLockError) as exc:
        return _write_error_response(request, exc)
    except Exception as exc:
        return _unexpected_write_error_response(request, exc)
    return modeled_success_response(request, result, SendWriteResponse)


def _parse_send_at(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ConversationGatewayError(
            "send_at is not an ISO 8601 time.", code="invalid_send_at", status=422
        ) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ConversationGatewayError(
            "send_at needs a timezone.", code="invalid_send_at", status=422
        )
    return parsed


@router.post(
    "/conversations/{conversation_id}/draft/schedule",
    response_model=AgentSuccessEnvelope[ScheduledSendResponse],
)
async def schedule_conversation_reply(
    request: Request,
    payload: DraftScheduleRequest,
    conversation_id: ConversationId,
    identity: AgentIdentity = Depends(require_scopes("send")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    """Book a reply for later; the daemon's worker sends it (services/scheduled_sends.py)."""
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    window = None
    if (payload.after_customer_min_minutes is None) != (payload.after_customer_max_minutes is None):
        return error_response(
            request_id=agent_request_id(request), status_code=422,
            code="invalid_schedule", message=_WRITE_ERROR_MESSAGES["invalid_schedule"],
        )
    if payload.after_customer_min_minutes is not None:
        window = (payload.after_customer_min_minutes, payload.after_customer_max_minutes)
    try:
        result = await scheduled_sends.schedule_send(
            gateway=gateway,
            db_path=get_supa_db_path(),
            identity=identity,
            conversation_id=conversation_id,
            expected_latest_thread_id=payload.expected_latest_thread_id,
            idempotency_key=payload.idempotency_key,
            body=payload.body,
            body_format=payload.body_format,
            expected_version=payload.expected_version,
            send_at=_parse_send_at(payload.send_at),
            after_customer=window,
            cc=payload.cc,
        )
    except ConversationGatewayError as exc:
        return _write_error_response(request, exc)
    except HelpScoutError as exc:
        return _read_error_response(request, exc)
    except Exception as exc:
        return _unexpected_write_error_response(request, exc)
    return modeled_success_response(request, result, ScheduledSendResponse)


@router.get(
    "/conversations/{conversation_id}/draft/schedule",
    response_model=AgentSuccessEnvelope[ScheduledSendReadResponse],
)
async def get_conversation_schedule(
    request: Request,
    conversation_id: ConversationId,
    identity: AgentIdentity = Depends(require_scopes("send")),
) -> JSONResponse:
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    try:
        row = await asyncio.to_thread(
            scheduled_sends.latest_for_conversation, get_supa_db_path(), conversation_id
        )
    except Exception as exc:
        return _unexpected_write_error_response(request, exc)
    if row is not None and not identity.allows_conversation(row["conversation_id"]):
        return _conversation_not_allowed_response(request)
    return modeled_success_response(
        request,
        {"scheduled_send": scheduled_sends.public_record(row) if row else None},
        ScheduledSendReadResponse,
    )


@router.post(
    "/conversations/{conversation_id}/draft/schedule/cancel",
    response_model=AgentSuccessEnvelope[ScheduledSendResponse],
)
async def cancel_conversation_schedule(
    request: Request,
    conversation_id: ConversationId,
    identity: AgentIdentity = Depends(require_scopes("send")),
) -> JSONResponse:
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    try:
        result = await asyncio.to_thread(
            lambda: scheduled_sends.cancel_send(
                db_path=get_supa_db_path(), identity=identity, conversation_id=conversation_id
            )
        )
    except ConversationGatewayError as exc:
        return _write_error_response(request, exc)
    except Exception as exc:
        return _unexpected_write_error_response(request, exc)
    return modeled_success_response(request, result, ScheduledSendResponse)


@router.post(
    "/conversations/{conversation_id}/replies",
    response_model=AgentSuccessEnvelope[SendWriteResponse],
)
async def post_conversation_reply(
    request: Request,
    payload: ReplySendRequest,
    conversation_id: ConversationId,
    identity: AgentIdentity = Depends(require_scopes("send")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    try:
        result = await gateway.send_reply(
            conversation_id,
            body=payload.body,
            body_format=payload.body_format,
            expected_latest_thread_id=payload.expected_latest_thread_id,
            idempotency_key=payload.idempotency_key,
            identity=identity,
            request_id=agent_request_id(request),
            cc=payload.cc,
        )
    except (ConversationGatewayError, AgentOperationError, ConversationLockError) as exc:
        return _write_error_response(request, exc)
    except Exception as exc:
        return _unexpected_write_error_response(request, exc)
    return modeled_success_response(request, result, SendWriteResponse)


@router.get("/conversations/{conversation_id}/state")
async def get_conversation_state(
    request: Request,
    conversation_id: ConversationId,
    identity: AgentIdentity = Depends(require_scopes("read")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    """The conversation's state and what agents and people did on it, as data."""
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    ledger = await gateway.client.ledger(conversation_id)
    if ledger is None:
        return error_response(
            request_id=agent_request_id(request), status_code=503, code="state_unavailable",
            message="The mailbox did not return the conversation state.", retryable=True,
        )
    if not ledger:
        return error_response(
            request_id=agent_request_id(request), status_code=404, code="not_found",
            message="The mailbox has no such conversation.",
        )
    # A merged conversation answers under another id; a per-case token sees only its own.
    if not identity.allows_conversation(ledger.get("conversation")):
        return _conversation_not_allowed_response(request)
    return success_response(request, ledger)


# Changes that can lead to an email to the customer (Supa's reminders act on them).
_STATE_NEEDS_SEND = (("waiting_on", "customer"), ("reminders", "on"))


@router.post("/conversations/{conversation_id}/state")
async def post_conversation_state(
    request: Request,
    payload: StateChangeRequest,
    conversation_id: ConversationId,
    identity: AgentIdentity = Depends(require_scopes("note")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    """Change the stage, who is waited on, the reminders, or take a support conversation off Supa.

    Replaces asking a person to change tags. Setting the same state twice changes nothing.
    """
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    changes = payload.changes.model_dump(exclude_none=True)
    if not changes:
        return error_response(
            request_id=agent_request_id(request), status_code=400, code="no_changes",
            message="Name at least one state field to change.",
        )
    if "send" not in identity.scopes and any(changes.get(field) == value for field, value in _STATE_NEEDS_SEND):
        return error_response(
            request_id=agent_request_id(request), status_code=403, code="insufficient_scope",
            message="Waiting on the customer and reminders on can lead to an email; they need the send scope.",
        )
    try:
        # The mailbox follows a merge to the surviving conversation; refuse before changing it.
        if not await _own_conversation_in_mailbox(identity, gateway, conversation_id):
            return _conversation_not_allowed_response(request)
        status, result = await gateway.client.change_state(
            conversation_id, changes, actor=identity.name, reason=payload.reason,
        )
    except Exception as exc:
        logger.warning("State change failed: %s", type(exc).__name__)
        status, result = None, "The mailbox could not be reached."
    if status == 200:
        logger.info("State of conversation %s changed by %s: %s", conversation_id, identity.name, sorted(changes))
        return success_response(request, {"conversation_id": conversation_id, "changes": changes, "state": result})
    refused = status in (400, 404, 409)
    retryable = status is None or (isinstance(status, int) and status >= 500)
    return error_response(
        request_id=agent_request_id(request),
        status_code=status if refused else 503 if retryable else 502,
        code={400: "invalid_state", 404: "not_found", 409: "state_refused"}.get(status, "state_unavailable"),
        message=str(result),
        retryable=retryable,
    )


async def _own_conversation_in_mailbox(identity: AgentIdentity, gateway: ConversationGateway, conversation_id: int) -> bool:
    """For a per-case token: the mailbox follows a merge, so the surviving conversation must be its own too."""
    if not identity.is_conversation_restricted:
        return True
    ledger = await gateway.client.ledger(conversation_id, 1)
    return bool(ledger) and identity.allows_conversation(ledger.get("conversation"))


def _owner_request_error(request: Request, status: int | None, *, conflict: str = "") -> JSONResponse:
    refused = status in (400, 404, 409)
    retryable = status is None or (isinstance(status, int) and status >= 500)
    message = {
        400: "The mailbox refused the request as invalid.",
        404: "The mailbox has no such conversation or request.",
        409: conflict or "The mailbox refused the request.",
    }.get(status, "The mailbox did not answer; the same idempotency key may be sent again.")
    return error_response(
        request_id=agent_request_id(request),
        status_code=status if refused else 503 if retryable else 502,
        code={400: "invalid_request", 404: "not_found", 409: "request_conflict"}.get(status, "request_unavailable"),
        message=message,
        retryable=retryable,
    )


@router.post("/conversations/{conversation_id}/requests", description=wording("owner_request_description"))
async def post_owner_request(
    request: Request,
    payload: OwnerRequestAdd,
    conversation_id: ConversationId,
    identity: AgentIdentity = Depends(require_scopes("note")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    """Record an idempotent request for the owner."""
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    try:
        if not await _own_conversation_in_mailbox(identity, gateway, conversation_id):
            return _conversation_not_allowed_response(request)
        status, body = await gateway.client.add_request(conversation_id, identity.name, payload.text, payload.idempotency_key)
    except Exception as exc:
        logger.warning("Owner request failed: %s", type(exc).__name__)
        return _owner_request_error(request, None)
    if status != 201 or type(body.get("id")) is not int:
        return _owner_request_error(
            request, status if status != 201 else None,
            conflict="This idempotency key already recorded a request with another text; use a new key.")
    return success_response(request, {
        "conversation_id": conversation_id, "request_id": body["id"],
        "status": body.get("status") or "open", "replayed": bool(body.get("replayed")),
    })


@router.post("/conversations/{conversation_id}/requests/{event_id}/done")
async def post_owner_request_done(
    request: Request,
    conversation_id: ConversationId,
    event_id: ThreadId,
    identity: AgentIdentity = Depends(require_scopes("note")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    """Close a request that got its answer or is no longer needed. A per-case agent closes its own."""
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    try:
        ledger = await gateway.client.ledger(conversation_id, 200)
        if ledger is None:
            return _owner_request_error(request, None)
        if not ledger or not identity.allows_conversation(ledger.get("conversation")):
            return _conversation_not_allowed_response(request) if ledger else _owner_request_error(request, 404)
        asked = next((event for event in ledger.get("events") or []
                      if isinstance(event, dict) and event.get("id") == event_id and event.get("kind") == "request"), None)
        if asked is None:
            return _owner_request_error(request, 404)
        if identity.is_conversation_restricted and asked.get("actor") != identity.name:
            return error_response(
                request_id=agent_request_id(request), status_code=403, code="request_not_yours",
                message="A case agent closes the requests it made; the owner closes the others.",
            )
        result = await gateway.client.close_request(conversation_id, event_id, identity.name)
    except Exception as exc:
        logger.warning("Closing an owner request failed: %s", type(exc).__name__)
        return _owner_request_error(request, None)
    if not result.ok:
        return _owner_request_error(request, None if result.outcome_unknown else result.status_code)
    return success_response(request, {"conversation_id": conversation_id, "request_id": event_id, "status": "done"})


@router.post(
    "/conversations/{conversation_id}/notes",
    response_model=AgentSuccessEnvelope[NoteWriteResponse],
)
async def post_conversation_note(
    request: Request,
    payload: NoteAddRequest,
    conversation_id: ConversationId,
    identity: AgentIdentity = Depends(require_scopes("note")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    """Add an internal note; the conversation status is left unchanged."""
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    try:
        result = await gateway.add_note(
            conversation_id,
            body=payload.body,
            body_format=payload.body_format,
            idempotency_key=payload.idempotency_key,
            identity=identity,
            request_id=agent_request_id(request),
        )
    except (ConversationGatewayError, AgentOperationError, ConversationLockError) as exc:
        return _write_error_response(request, exc)
    except Exception as exc:
        return _unexpected_write_error_response(request, exc)
    return modeled_success_response(request, result, NoteWriteResponse)


@router.put(
    "/conversations/{conversation_id}/notes/{thread_id}",
    response_model=AgentSuccessEnvelope[NoteUpdateResponse],
)
async def put_conversation_note(
    request: Request,
    payload: NoteUpdateRequest,
    conversation_id: ConversationId,
    thread_id: ThreadId,
    identity: AgentIdentity = Depends(require_scopes("note")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> JSONResponse:
    """Replace an API-written internal note at its version; the status is left unchanged."""
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    try:
        result = await gateway.update_note(
            conversation_id,
            thread_id=thread_id,
            body=payload.body,
            body_format=payload.body_format,
            expected_version=payload.expected_version,
            idempotency_key=payload.idempotency_key,
            identity=identity,
            request_id=agent_request_id(request),
        )
    except (ConversationGatewayError, AgentOperationError, ConversationLockError) as exc:
        return _write_error_response(request, exc)
    except Exception as exc:
        return _unexpected_write_error_response(request, exc)
    return modeled_success_response(request, result, NoteUpdateResponse)


async def _poll_next_event(
    *,
    token_id: str,
    consumer: str,
    start: str | int,
    wait_seconds: int,
    conversations: frozenset[int] | None = None,
) -> dict[str, Any] | None:
    """Lease immediately, then recheck after bounded wakeups until the deadline."""
    deadline = _event_poll_monotonic() + wait_seconds
    while True:
        generation = _event_wait_notifier.snapshot()
        event = await asyncio.to_thread(
            lease_next_event,
            get_supa_db_path(),
            token_id=token_id,
            consumer_name=consumer,
            start=start,
            clock=_event_clock,
            conversations=conversations,
        )
        if event is not None:
            return event
        remaining = deadline - _event_poll_monotonic()
        if remaining <= 0:
            return None
        await _wait_for_event_notification(
            generation,
            min(EVENT_POLL_RECHECK_SECONDS, remaining),
        )


@router.post(
    "/events/next",
    response_model=AgentSuccessEnvelope[EventNextResponse],
)
async def next_agent_event(
    request: Request,
    payload: EventNextRequest,
    identity: AgentIdentity = Depends(require_scopes("events")),
) -> JSONResponse:
    if not _event_feed_enabled():
        return _event_feed_unavailable_response(request)
    try:
        event = await _poll_next_event(
            token_id=identity.token_id,
            consumer=payload.consumer,
            start=payload.start,
            wait_seconds=payload.wait,
            conversations=identity.conversations,
        )
    except (AgentEventProtocolError, AgentEventStorageError) as exc:
        return _event_error_response(request, exc)
    except ValueError:
        return _invalid_event_request_response(request)
    except Exception as exc:
        return _unexpected_event_error_response(request, exc)
    return modeled_success_response(
        request,
        {"event": event},
        EventNextResponse,
    )


@router.post(
    "/events/{event_id}/ack",
    response_model=AgentSuccessEnvelope[EventAckResponse],
)
async def ack_agent_event(
    request: Request,
    payload: EventAckRequest,
    event_id: EventId,
    identity: AgentIdentity = Depends(require_scopes("events")),
) -> JSONResponse:
    if not _event_feed_enabled():
        return _event_feed_unavailable_response(request)
    try:
        result = await asyncio.to_thread(
            acknowledge_event,
            get_supa_db_path(),
            token_id=identity.token_id,
            consumer_name=payload.consumer,
            event_id=event_id,
            ack_token=payload.ack_token,
            clock=_event_clock,
        )
    except (AgentEventProtocolError, AgentEventStorageError) as exc:
        return _event_error_response(request, exc)
    except ValueError:
        return _invalid_event_request_response(request)
    except Exception as exc:
        return _unexpected_event_error_response(request, exc)
    return modeled_success_response(request, result, EventAckResponse)


@router.post(
    "/events/reset",
    response_model=AgentSuccessEnvelope[EventResetResponse],
)
async def reset_agent_event_consumer(
    request: Request,
    payload: EventResetRequest,
    identity: AgentIdentity = Depends(require_scopes("events")),
) -> JSONResponse:
    if not _event_feed_enabled():
        return _event_feed_unavailable_response(request)
    try:
        result = await asyncio.to_thread(
            reset_event_consumer,
            get_supa_db_path(),
            token_id=identity.token_id,
            token_name=identity.name,
            consumer_name=payload.consumer,
            to=payload.to,
            request_id=agent_request_id(request),
            clock=_event_clock,
        )
    except (AgentEventProtocolError, AgentEventStorageError) as exc:
        return _event_error_response(request, exc)
    except ValueError:
        return _invalid_event_request_response(request)
    except Exception as exc:
        return _unexpected_event_error_response(request, exc)
    return modeled_success_response(request, result, EventResetResponse)


@router.get(
    "/operations/{idempotency_key}",
    response_model=AgentSuccessEnvelope[OperationResponse],
)
async def get_agent_operation(
    request: Request,
    idempotency_key: Annotated[
        str,
        FastAPIPath(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH),
    ],
    identity: AgentIdentity = Depends(require_scopes()),
) -> JSONResponse:
    try:
        _validate_idempotency_key(idempotency_key)
        record = await asyncio.to_thread(
            get_operation,
            idempotency_key=idempotency_key,
            identity=identity,
            db_path=get_supa_db_path(),
        )
    except AgentOperationError as exc:
        return _write_error_response(request, exc)
    except ValueError:
        return error_response(
            request_id=agent_request_id(request),
            status_code=422,
            code="invalid_request",
            message="Invalid request",
        )
    except Exception as exc:
        return _unexpected_write_error_response(request, exc)
    return modeled_success_response(request, record.to_public_dict(), OperationResponse)


@router.get(
    "/conversations/{conversation_id}/attachments/{attachment_id}",
    response_class=Response,
)
async def download_conversation_attachment(
    request: Request,
    conversation_id: ConversationId,
    attachment_id: AttachmentId,
    identity: AgentIdentity = Depends(require_scopes("read")),
    gateway: ConversationGateway = Depends(get_conversation_gateway),
) -> Response:
    if not identity.allows_conversation(conversation_id):
        return _conversation_not_allowed_response(request)
    try:
        # The gateway re-checks the resolved (post-merge) id before download.
        attachment = await gateway.attachment(
            conversation_id, attachment_id, identity=identity
        )
    except (ConversationGatewayError, HelpScoutError) as exc:
        return _read_error_response(request, exc)
    return _attachment_response(request, attachment)
