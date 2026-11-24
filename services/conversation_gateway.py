"""Normalized, credential-safe Help Scout conversation reads for agent clients."""

from __future__ import annotations

from services import installation

from services import mailbox_provider
from services.reply_text import body_lines, customer_words

import asyncio
import base64
import binascii
import hashlib
import html
import inspect
import json
import logging
import os
import re
import warnings
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from bs4 import (
    BeautifulSoup,
    CData,
    Comment,
    Declaration,
    Doctype,
    MarkupResemblesLocatorWarning,
    ProcessingInstruction,
)

from services.helpscout import (
    BLANK_IMAGE_MAX_BYTES,
    HelpScoutClient,
    HelpScoutError,
    MutationResult,
    _is_agent_message_thread,
    _is_customer_thread,
    _is_draft_thread,
    _normalize_claimed_attachment_mime,
    ensure_email_paragraph_spacing,
    recorded,
    get_latest_actionable_thread,
    get_latest_draft_thread,
    is_draft_superseded_by_reply,
    strip_html,
)
from services.agent_auth import CONVERSATION_NOT_ALLOWED_CODE, AgentIdentity
from services.agent_events import AgentEventStorageError, append_supa_event
from services.agent_operations import (
    OperationRecord,
    OperationReservation,
    begin_operation,
    canonical_request_json,
    finish_operation,
    get_operation,
    mark_upstream_started,
    reserve_operation,
)
from services.conversation_locks import ConversationLockError, conversation_lease


_CURSOR_VERSION = 1
_MAX_CURSOR_LENGTH = 256
_MAX_CURSOR_PAGE = 1_000_000
_DEFAULT_ATTACHMENT_MAX_BYTES = 25 * 1024 * 1024
_CURSOR_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_DRAFT_VERSION_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_SUMMARY_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,39}$")
_SAFE_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_MAX_RESOLUTION_ATTEMPTS = 3
_MAX_DRAFT_BODY_BYTES = 1024 * 1024
_NOTE_CONVERSATION_STATUSES = frozenset({"active", "closed", "pending", "spam"})
# An agent may copy at most two addresses supplied by the customer.
MAX_CC_ADDRESSES = 2
_CC_ADDRESS_RE = re.compile(
    r"^[a-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}@"
    r"(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+$"
)
# Match complete email tokens. Validate each candidate against the address grammar.
_EMAIL_LOCAL_CHARS = r"A-Za-z0-9.!#$%&'*+/=?^_`{|}~-"
_EMAIL_IN_TEXT_RE = re.compile(
    rf"(?<![{_EMAIL_LOCAL_CHARS}])[{_EMAIL_LOCAL_CHARS}]{{1,64}}@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
    r"(?![A-Za-z0-9_-])(?!\.[A-Za-z0-9])"
)
# Never copy our own mailboxes or team: a CC there could loop back into Help Scout.
# SUPA_CC_OWN_DOMAINS (comma-separated) adds domains.
_OWN_CC_DOMAINS = installation.OWN_DOMAINS
_SAFE_DRAFT_TAGS = frozenset(
    {
        "b",
        "a",
        "blockquote",
        "br",
        "code",
        "em",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "i",
        "li",
        "ol",
        "p",
        "pre",
        "strong",
        "u",
        "ul",
    }
)
_DROP_DRAFT_TAGS = frozenset(
    {"embed", "iframe", "math", "object", "script", "style", "svg"}
)
_REMOVED_DRAFT_STRING_TYPES = (
    CData,
    Comment,
    Declaration,
    Doctype,
    ProcessingInstruction,
)
logger = logging.getLogger(__name__)


class ConversationGatewayError(Exception):
    """Typed validation/read error suitable for the versioned Agent API."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool = False,
        status: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status = status
        self.details = dict(details or {})


class InvalidCursorError(ConversationGatewayError):
    """The opaque public cursor was malformed or from an unsupported version."""

    def __init__(self) -> None:
        super().__init__(
            "The conversation cursor is invalid.",
            code="invalid_cursor",
            retryable=False,
            status=400,
        )


@dataclass(frozen=True)
class AttachmentData:
    """Bounded attachment bytes and safe metadata from the complete conversation."""

    conversation_id: int
    attachment_id: int
    filename: str
    data: bytes
    claimed_mime_type: str | None
    size: int


def _safe_draft_href(value: Any) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2048
        or value != value.strip()
        or any(
            character.isspace()
            or ord(character) < 32
            or ord(character) == 127
            for character in value
        )
    ):
        return None
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        return None
    scheme = parsed.scheme.casefold()
    if (
        scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    hostname = parsed.hostname.casefold()
    try:
        canonical_host = (
            f"[{hostname}]"
            if ":" in hostname
            else hostname.encode("idna").decode("ascii")
        )
    except UnicodeError:
        return None
    default_port = 80 if scheme == "http" else 443
    netloc = canonical_host if port in (None, default_port) else f"{canonical_host}:{port}"
    return urlunsplit((scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _draft_soup(markup: str) -> BeautifulSoup:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        return BeautifulSoup(markup, "html.parser")


def _has_disallowed_draft_control(value: str) -> bool:
    return any(
        (ord(character) < 32 and character not in "\t\n\r")
        or 127 <= ord(character) <= 159
        for character in value
    )


def normalize_draft_body(body: str, body_format: str) -> str:
    """Return deterministic, non-executable HTML for one draft body."""

    if type(body_format) is not str or body_format not in {"html", "text"}:
        raise ConversationGatewayError(
            "body_format must be html or text.",
            code="invalid_body_format",
            retryable=False,
            status=400,
        )
    if type(body) is not str or not body.strip():
        raise ConversationGatewayError(
            "The draft body must not be empty.",
            code="invalid_body",
            retryable=False,
            status=400,
        )
    try:
        body_bytes = body.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ConversationGatewayError(
            "The draft body contains invalid Unicode.",
            code="invalid_body",
            retryable=False,
            status=400,
        ) from error
    if len(body_bytes) > _MAX_DRAFT_BODY_BYTES:
        raise ConversationGatewayError(
            "The draft body is too large.",
            code="body_too_large",
            retryable=False,
            status=413,
        )
    if _has_disallowed_draft_control(body):
        raise ConversationGatewayError(
            "The draft body contains disallowed control characters.",
            code="invalid_body",
            retryable=False,
            status=400,
        )

    try:
        if body_format == "text":
            canonical = body.replace("\r\n", "\n").replace("\r", "\n").strip()
            paragraphs = re.split(r"\n[ \t]*\n+", canonical)
            normalized = "".join(
                f"<p>{html.escape(paragraph, quote=True).replace(chr(10), '<br>')}</p>"
                for paragraph in paragraphs
                if paragraph
            )
        else:
            soup = _draft_soup(body.strip())
            for special in soup.find_all(
                string=lambda value: isinstance(value, _REMOVED_DRAFT_STRING_TYPES)
            ):
                special.extract()
            for tag in list(soup.find_all(_DROP_DRAFT_TAGS)):
                tag.decompose()
            for tag in list(soup.find_all(True)):
                if tag.name not in _SAFE_DRAFT_TAGS:
                    tag.unwrap()
                elif tag.name == "a":
                    safe_href = _safe_draft_href(tag.get("href"))
                    if safe_href is None:
                        tag.unwrap()
                    else:
                        tag.attrs = {
                            "href": safe_href,
                            "rel": "noopener noreferrer",
                        }
                else:
                    tag.attrs = {}
            normalized = str(soup).strip()

        normalized = ensure_email_paragraph_spacing(normalized)
        normalized_bytes = normalized.encode("utf-8", errors="strict")
        if _has_disallowed_draft_control(normalized):
            raise ConversationGatewayError(
                "The normalized draft body contains disallowed control characters.",
                code="invalid_body",
                retryable=False,
                status=400,
            )
        if len(normalized_bytes) > _MAX_DRAFT_BODY_BYTES:
            raise ConversationGatewayError(
                "The normalized draft body is too large.",
                code="body_too_large",
                retryable=False,
                status=413,
            )
        if not normalized or not _draft_soup(normalized).get_text(strip=True):
            raise ConversationGatewayError(
                "The draft body must contain visible text.",
                code="invalid_body",
                retryable=False,
                status=400,
            )
    except ConversationGatewayError:
        raise
    except Exception as error:
        raise ConversationGatewayError(
            "The draft body could not be normalized safely.",
            code="invalid_body",
            retryable=False,
            status=400,
        ) from error
    return normalized



def _same_body(upstream: Any, normalized_body: str) -> bool:
    """Match stored HTML after the mailbox normalizes entities and line breaks. Text, attributes and structure must remain equivalent."""
    if not isinstance(upstream, str):
        return False
    if upstream == normalized_body:
        return True
    try:
        return normalize_draft_body(upstream, "html") == normalize_draft_body(normalized_body, "html")
    except ConversationGatewayError:
        return False

def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_id(value: Any) -> int | str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped or None
    return None


def _timestamp(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _thread_id_sort_key(thread: dict[str, Any]) -> tuple[int, int | str]:
    thread_id = _safe_id(thread.get("id"))
    if isinstance(thread_id, int):
        return (0, thread_id)
    return (1, thread_id or "")


def _thread_sort_key(thread: dict[str, Any]) -> tuple[int, str, tuple[int, int | str]]:
    """Chronological ordering with unknown timestamps last and id as the tie-breaker."""
    created_at = _timestamp(thread.get("createdAt") or thread.get("created_at"))
    return (1 if created_at is None else 0, created_at or "", _thread_id_sort_key(thread))


def _selection_order(threads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make existing latest-thread helpers deterministic for timestamp/id ties."""
    return sorted(threads, key=_thread_id_sort_key, reverse=True)


def _normalize_person(value: Any, *, include_role: bool = False) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    person_type = value.get("type") if isinstance(value.get("type"), str) else None
    name = " ".join(
        part.strip()
        for part in (value.get("first"), value.get("last"))
        if isinstance(part, str) and part.strip()
    )
    if not name and isinstance(value.get("name"), str):
        name = value["name"].strip()
    email = value.get("email") if isinstance(value.get("email"), str) else None
    result: dict[str, Any] = {
        "id": _safe_id(value.get("id")),
        "type": person_type,
    }
    if include_role:
        result["role"] = {
            "customer": "customer",
            "user": "agent",
            "system_user": "system_user",
        }.get(person_type, "unknown")
    result["name"] = name or None
    result["email"] = email
    return result


def _normalize_source(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"type": value, "via": None}
    if not isinstance(value, dict):
        return {"type": None, "via": None}
    return {
        "type": value.get("type") if isinstance(value.get("type"), str) else None,
        "via": value.get("via") if isinstance(value.get("via"), str) else None,
    }


def _normalize_mime(value: Any) -> str:
    return _normalize_claimed_attachment_mime(value) or ""


def _normalize_size(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _attachment_list(thread: dict[str, Any]) -> list[dict[str, Any]]:
    embedded = thread.get("_embedded")
    raw_attachments = embedded.get("attachments") if isinstance(embedded, dict) else []
    if not isinstance(raw_attachments, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw in raw_attachments:
        if not isinstance(raw, dict):
            continue
        mime_type = _normalize_mime(raw.get("mimeType") or raw.get("mime_type"))
        size = _normalize_size(raw.get("size"))
        is_image = mime_type.startswith("image/")
        normalized.append(
            {
                "id": _safe_id(raw.get("id")),
                "filename": raw.get("filename")
                if isinstance(raw.get("filename"), str)
                else "",
                "mime_type": mime_type,
                "size": size,
                "is_image": is_image,
                "is_blank_transport_artifact": (
                    is_image and 0 < size <= BLANK_IMAGE_MAX_BYTES
                ),
            }
        )
    return normalized


def _normalized_thread(thread: dict[str, Any]) -> dict[str, Any]:
    body = thread.get("body_html", thread.get("body", ""))
    if not isinstance(body, str):
        body = ""
    created_at = _timestamp(thread.get("created_at") or thread.get("createdAt"))
    updated_at = _timestamp(
        thread.get("updated_at") or thread.get("updatedAt") or thread.get("modifiedAt")
    )
    opened_at = _timestamp(thread.get("opened_at") or thread.get("openedAt"))
    raw_author = thread.get("author") or thread.get("createdBy")
    author = _normalize_person(raw_author, include_role=True)
    if author is None:
        author = {
            "id": None,
            "type": None,
            "role": "unknown",
            "name": None,
            "email": None,
        }
    attachments = (
        thread.get("attachments")
        if isinstance(thread.get("attachments"), list)
        else _attachment_list(thread)
    )
    # Re-normalize already-public attachment records so versioning and output share
    # one allowlist and never accidentally acquire an upstream link field.
    if "attachments" in thread:
        attachment_wrapper = {"_embedded": {"attachments": attachments}}
        attachments = _attachment_list(attachment_wrapper)

    thread_type = thread.get("type") if isinstance(thread.get("type"), str) else None
    normalized = {
        "id": _safe_id(thread.get("id") or thread.get("thread_id")),
        "type": thread_type,
        "state": thread.get("state") if isinstance(thread.get("state"), str) else None,
        "draft": _is_draft_thread(thread),
        "is_internal_note": thread_type == "note",
        "author": author,
        "source": _normalize_source(thread.get("source")),
        "body_html": body,
        "body_text": strip_html(body),
        "created_at": created_at,
        "updated_at": updated_at,
        "opened_at": opened_at,
        "attachments": attachments,
    }
    if thread_type == "note" and not _is_draft_thread(thread):
        # The precondition `note update` checks; additive, so drafts and the
        # conversation version are unchanged.
        normalized["version"] = note_version(thread)
    # Only when present, so threads without them (and every version) are unchanged.
    for header in ("to", "cc"):
        addresses = _thread_addresses(thread, header)
        if addresses:
            normalized[header] = addresses
    return normalized


def _thread_addresses(thread: dict[str, Any], header: str) -> list[str]:
    """The `to`/`cc` addresses Help Scout reports on one thread (strings only, as given)."""
    raw = thread.get(header)
    if not isinstance(raw, list):
        return []
    return [item.strip() for item in raw if isinstance(item, str) and item.strip()]


def _thread_cc(thread: dict[str, Any]) -> list[str]:
    return _thread_addresses(thread, "cc")


def _draft_version_record(thread: dict[str, Any]) -> dict[str, Any]:
    body = thread.get("body_html", thread.get("body", ""))
    if not isinstance(body, str):
        body = ""
    record = {
        "body_html": body,
        "created_at": _timestamp(thread.get("created_at") or thread.get("createdAt")),
        "draft": _is_draft_thread(thread),
        "id": _safe_id(thread.get("id") or thread.get("thread_id")),
        "state": thread.get("state") if isinstance(thread.get("state"), str) else None,
        "type": thread.get("type") if isinstance(thread.get("type"), str) else None,
        "updated_at": _timestamp(
            thread.get("updated_at") or thread.get("updatedAt") or thread.get("modifiedAt")
        ),
    }
    cc = _thread_cc(thread)
    if cc:
        # `draft send` sends the draft's CC, so its version pins it; drafts without
        # CC keep the version they always had.
        record["cc"] = cc
    return record


def draft_version(thread: dict[str, Any]) -> str:
    """Return a canonical SHA-256 version for one live draft."""
    return _canonical_hash(_draft_version_record(thread))


def note_version(thread: dict[str, Any]) -> str:
    """Return a canonical SHA-256 version for one internal note (same record as drafts)."""
    return _canonical_hash(_draft_version_record(thread))


def _conversation_thread_record(thread: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalized_thread(thread)
    return {
        "attachments": normalized["attachments"],
        "author": normalized["author"],
        "body_html": normalized["body_html"],
        "created_at": normalized["created_at"],
        "draft": normalized["draft"],
        "id": normalized["id"],
        "opened_at": normalized["opened_at"],
        "source": normalized["source"],
        "state": normalized["state"],
        "type": normalized["type"],
        "updated_at": normalized["updated_at"],
    }


def conversation_version(threads: list[dict[str, Any]]) -> str:
    """Hash deterministically ordered public thread/state/body metadata."""
    records = [_conversation_thread_record(thread) for thread in threads]
    records.sort(key=_thread_sort_key)
    return _canonical_hash(records)


def _normalize_mailbox(raw: dict[str, Any]) -> dict[str, Any]:
    mailbox = raw.get("mailbox")
    if not isinstance(mailbox, dict):
        mailbox = {}
    return {
        "id": _safe_id(mailbox.get("id") or raw.get("mailboxId")),
        "name": mailbox.get("name") if isinstance(mailbox.get("name"), str) else None,
    }


def _normalize_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    for item in value:
        tag = item.get("tag") if isinstance(item, dict) else item
        if isinstance(tag, str) and tag:
            tags.append(tag)
    return tags


def _customer_waiting_since(value: Any) -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        return _timestamp(value.get("time"))
    return None


def _latest_actionable(threads: list[dict[str, Any]]) -> dict[str, Any] | None:
    actionable = [
        thread
        for thread in _selection_order(threads)
        if (_is_customer_thread(thread) or _is_agent_message_thread(thread))
        and not _is_draft_thread(thread)
        and thread.get("state") in (None, "published")
    ]
    return get_latest_actionable_thread(actionable)


def _waiting_on(threads: list[dict[str, Any]]) -> str:
    latest = _latest_actionable(threads)
    if latest is None:
        return "unknown"
    if _is_customer_thread(latest):
        return "agent"
    if _is_agent_message_thread(latest):
        return "customer"
    return "unknown"


def _live_draft(threads: list[dict[str, Any]]) -> dict[str, Any] | None:
    selection_threads = _selection_order(threads)
    draft = get_latest_draft_thread(selection_threads)
    if draft is None or is_draft_superseded_by_reply(selection_threads, draft):
        return None
    normalized = _normalized_thread(draft)
    live = {
        "thread_id": normalized["id"],
        "type": normalized["type"],
        "state": normalized["state"],
        "body_html": normalized["body_html"],
        "body_text": normalized["body_text"],
        "created_at": normalized["created_at"],
        "updated_at": normalized["updated_at"],
        "version": draft_version(draft),
    }
    if normalized.get("cc"):
        live["cc"] = normalized["cc"]
    return live


def normalize_conversation(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the stable public representation of one complete conversation."""
    embedded = raw.get("_embedded")
    raw_threads = embedded.get("threads") if isinstance(embedded, dict) else []
    threads = (
        [thread for thread in raw_threads if isinstance(thread, dict)]
        if isinstance(raw_threads, list)
        else []
    )
    selection_threads = _selection_order(threads)
    normalized_threads = [
        {
            **_normalized_thread(thread),
            # A draft older than a published agent reply is the leftover Help Scout
            # cannot delete (Supa blanks it after a send); it is not pending.
            "superseded_by_reply": _is_draft_thread(thread)
            and is_draft_superseded_by_reply(selection_threads, thread),
        }
        for thread in threads
    ]
    normalized_threads.sort(key=_thread_sort_key)
    latest_actionable = _latest_actionable(threads)
    waiting_on = _waiting_on(threads)
    latest_actionable_thread_id = (
        _safe_id(latest_actionable.get("id") or latest_actionable.get("thread_id"))
        if isinstance(latest_actionable, dict)
        else None
    )

    resolved_id = _safe_id(raw.get("id"))
    requested_id = _safe_id(raw.get("requested_conversation_id"))
    if requested_id is None:
        requested_id = resolved_id
    primary_customer = raw.get("primaryCustomer") or raw.get("customer") or raw.get("createdBy")

    return {
        "id": resolved_id,
        "requested_id": requested_id,
        "resolved_id": resolved_id,
        "was_merged": requested_id != resolved_id,
        "number": _safe_id(raw.get("number")),
        "subject": raw.get("subject") if isinstance(raw.get("subject"), str) else "",
        "status": raw.get("status") if isinstance(raw.get("status"), str) else None,
        "tags": _normalize_tags(raw.get("tags")),
        "mailbox": _normalize_mailbox(raw),
        "assignee": _normalize_person(raw.get("assignee")),
        "customer": _normalize_person(primary_customer),
        "timestamps": {
            "created_at": _timestamp(raw.get("createdAt")),
            "updated_at": _timestamp(raw.get("updatedAt") or raw.get("modifiedAt")),
            "user_updated_at": _timestamp(raw.get("userUpdatedAt")),
            "closed_at": _timestamp(raw.get("closedAt")),
            # Help Scout can keep customerWaitingSince after our reply went out;
            # once our published reply is the latest actionable thread, nobody waits on us.
            "customer_waiting_since": (
                None
                if waiting_on == "customer"
                else _customer_waiting_since(raw.get("customerWaitingSince"))
            ),
        },
        "help_scout_url": (
            mailbox_provider.conversation_url(resolved_id)
            if resolved_id is not None
            else None
        ),
        "waiting_on": waiting_on,
        "latest_actionable_thread_id": latest_actionable_thread_id,
        "threads": normalized_threads,
        "draft": _live_draft(threads),
        "version": conversation_version(threads),
    }


def expected_thread_error(
    conversation: dict[str, Any], expected_latest_thread_id: int
) -> ConversationGatewayError | None:
    """Check the actionable-thread precondition against a normalized conversation."""
    latest_thread_id = _safe_id(conversation.get("latest_actionable_thread_id"))
    if not isinstance(latest_thread_id, int):
        latest_thread_id = None
    if latest_thread_id == expected_latest_thread_id:
        return None
    details = {
        "latest_actionable_thread_id": latest_thread_id,
        "given_thread_id": expected_latest_thread_id,
    }
    for thread in conversation.get("threads", []):
        if _is_draft_thread(thread) and _safe_id(thread.get("id")) == expected_latest_thread_id:
            return ConversationGatewayError(
                "--expected-thread must be the latest actionable thread "
                "(the customer's latest message), not the draft's thread.",
                code="wrong_expected_thread",
                status=409,
                details={**details, "draft_thread_id": expected_latest_thread_id},
            )
    if isinstance(latest_thread_id, int):
        details["thread_id"] = latest_thread_id
    return ConversationGatewayError(
        f"The latest actionable thread is now {latest_thread_id}, not {expected_latest_thread_id}.",
        code="conversation_changed",
        status=409,
        details=details,
    )


def _invalid_cc(message: str) -> ConversationGatewayError:
    return ConversationGatewayError(message, code="invalid_cc", status=422)


def normalize_cc(value: Any) -> list[str]:
    """Validated CC addresses: lowercased, de-duplicated, at most MAX_CC_ADDRESSES.

    None means no CC. Raises `invalid_cc` (422) for anything that is not a list of
    plain email addresses.
    """
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise _invalid_cc("cc must be a list of email addresses.")
    addresses: list[str] = []
    for item in value:
        if type(item) is not str:
            raise _invalid_cc("cc must be a list of email addresses.")
        address = item.strip().lower()
        if _CC_ADDRESS_RE.fullmatch(address) is None:
            raise _invalid_cc("A cc address is not a valid email address.")
        if address not in addresses:
            addresses.append(address)
    if len(addresses) > MAX_CC_ADDRESSES:
        raise _invalid_cc(f"At most {MAX_CC_ADDRESSES} cc addresses.")
    # Sorted, so the same set of addresses is the same idempotency request.
    return sorted(addresses)


def _addresses_in_text(text: str) -> set[str]:
    found: set[str] = set()
    for match in _EMAIL_IN_TEXT_RE.findall(text):
        # A quote or backtick right before an address is punctuation, not the address.
        candidate = match.lstrip("'`{|}~.").lower()
        if _CC_ADDRESS_RE.fullmatch(candidate):
            found.add(candidate)
    return found


def _customer_text(body_html: str) -> str:
    """Read only the current author’s text when collecting addresses."""
    own = re.split(r"(?i)<blockquote\b", body_html or "", maxsplit=1)[0]
    return customer_words(body_lines(own))


def _own_cc_domains() -> tuple[str, ...]:
    extra = os.environ.get("SUPA_CC_OWN_DOMAINS") or ""
    return _OWN_CC_DOMAINS + tuple(
        domain.strip().lower() for domain in extra.split(",") if domain.strip()
    )


def cc_own_addresses(conversation: dict[str, Any]) -> set[str]:
    """Our team's addresses in this conversation (agent authors, assignee)."""
    own: set[str] = set()
    people = [conversation.get("assignee")]
    for thread in conversation.get("threads") or []:
        if isinstance(thread, dict):
            people.append(thread.get("author"))
    for person in people:
        if not isinstance(person, dict) or not isinstance(person.get("email"), str):
            continue
        if person.get("type") in {"user", "team"} or person.get("role") in {"agent", "system_user"}:
            own.add(person["email"].strip().lower())
    return own


def _is_own_cc_address(address: str, own: set[str]) -> bool:
    domain = address.rsplit("@", 1)[-1]
    return address in own or any(
        domain == own_domain or domain.endswith("." + own_domain)
        for own_domain in _own_cc_domains()
    )


def cc_allowed_addresses(conversation: dict[str, Any]) -> set[str]:
    """Return verified customer addresses for this conversation. Include published customer text and headers. Exclude quoted history, agent replies, drafts, notes and the installation's own mailboxes."""
    allowed: set[str] = set()
    customer = conversation.get("customer")
    if isinstance(customer, dict) and isinstance(customer.get("email"), str):
        allowed.add(customer["email"].strip().lower())
    for thread in conversation.get("threads") or []:
        if not isinstance(thread, dict) or thread.get("draft") or thread.get("is_internal_note"):
            continue
        author = thread.get("author") if isinstance(thread.get("author"), dict) else {}
        if thread.get("type") != "customer" and author.get("role") != "customer":
            continue
        if thread.get("state") not in (None, "published"):
            continue
        if isinstance(author.get("email"), str):
            allowed.add(author["email"].strip().lower())
        for header in ("to", "cc"):
            for address in thread.get(header) or []:
                if isinstance(address, str):
                    allowed.add(address.strip().lower())
        body = thread.get("body_html") if isinstance(thread.get("body_html"), str) else ""
        allowed.update(_addresses_in_text(_customer_text(body)))
    own = cc_own_addresses(conversation)
    return {address for address in allowed if not _is_own_cc_address(address, own)}


def _default_cc_block_check(address: str) -> str | None:
    """Run the configured recipient checks. Storage errors propagate to the gateway."""
    from services.extensions import cc_block_reason
    return cc_block_reason(address)


def _encode_cursor(page: int, offset: int = 0) -> str:
    record = {"page": page, "v": _CURSOR_VERSION}
    if offset:
        record["offset"] = offset
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str | None) -> tuple[int, int]:
    if cursor is None:
        return 1, 0
    if (
        not isinstance(cursor, str)
        or not cursor
        or len(cursor) > _MAX_CURSOR_LENGTH
        or _CURSOR_RE.fullmatch(cursor) is None
    ):
        raise InvalidCursorError()
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
        record = json.loads(decoded.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise InvalidCursorError() from None
    valid_keys = isinstance(record, dict) and set(record) in (
        {"v", "page"},
        {"v", "page", "offset"},
    )
    offset = record.get("offset", 0) if isinstance(record, dict) else 0
    if (
        not isinstance(record, dict)
        or not valid_keys
        or isinstance(record.get("v"), bool)
        or record.get("v") != _CURSOR_VERSION
        or isinstance(record.get("page"), bool)
        or not isinstance(record.get("page"), int)
        or not 1 <= record["page"] <= _MAX_CURSOR_PAGE
        or isinstance(offset, bool)
        or not isinstance(offset, int)
        or not 0 <= offset < 25
        or ("offset" in record and offset == 0)
        or _encode_cursor(record["page"], offset) != cursor
    ):
        raise InvalidCursorError()
    return record["page"], offset


class ConversationGateway:
    """I/O boundary over HelpScoutClient with normalized public results."""

    def __init__(
        self,
        client: HelpScoutClient,
        *,
        attachment_max_bytes: int = _DEFAULT_ATTACHMENT_MAX_BYTES,
        db_path: str | os.PathLike[str] | None = None,
        post_send_bookkeeper: Any | None = None,
        cc_block_check: Any | None = None,
    ) -> None:
        if (
            isinstance(attachment_max_bytes, bool)
            or not isinstance(attachment_max_bytes, int)
            or attachment_max_bytes <= 0
        ):
            raise ValueError("attachment_max_bytes must be a positive integer")
        self.client = client
        self.attachment_max_bytes = attachment_max_bytes
        self.db_path = None if db_path is None else os.fspath(Path(db_path))
        self.post_send_bookkeeper = post_send_bookkeeper
        # address -> do-not-contact reason or None; raising fails the send closed.
        self.cc_block_check = cc_block_check or _default_cc_block_check

    @staticmethod
    async def _durable_thread_call(function: Any, /, **kwargs: Any) -> Any:
        """Do not abandon a ledger transaction when its waiter is cancelled."""

        task = asyncio.create_task(asyncio.to_thread(function, **kwargs))
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancellation = error
        result = task.result()
        if cancellation is not None:
            raise cancellation
        return result

    async def _operation_reserve(self, **kwargs: Any) -> OperationReservation:
        return await self._durable_thread_call(
            reserve_operation,
            db_path=self.db_path,
            **kwargs,
        )

    async def _operation_get(self, **kwargs: Any) -> OperationRecord:
        return await self._durable_thread_call(
            get_operation,
            db_path=self.db_path,
            **kwargs,
        )

    async def _operation_begin(self, **kwargs: Any) -> OperationRecord:
        return await self._durable_thread_call(
            begin_operation,
            db_path=self.db_path,
            **kwargs,
        )

    async def _operation_mark_upstream(self, **kwargs: Any) -> OperationRecord:
        return await self._durable_thread_call(
            mark_upstream_started,
            db_path=self.db_path,
            **kwargs,
        )

    async def _operation_finish(self, **kwargs: Any) -> OperationRecord:
        return await self._durable_thread_call(
            finish_operation,
            db_path=self.db_path,
            **kwargs,
        )

    @staticmethod
    def _event_summary_identifier(value: str, *, prefix: str) -> str:
        """Preserve ordinary IDs and pseudonymize unusual or secret-shaped IDs."""

        if (
            _EVENT_SUMMARY_IDENTIFIER_RE.fullmatch(value) is not None
            and not value.casefold().startswith(("sk-", "sk-ant-"))
        ):
            return value
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
        return f"{prefix}-{digest}"

    async def _emit_operation_event(
        self,
        *,
        operation: OperationRecord,
        request_id: str,
        event_type: str,
        conversation_id: int,
        thread_id: int,
        result: str,
        status: str,
    ) -> None:
        """Append one operation-derived, body-free outcome event."""

        try:
            await self._durable_thread_call(
                append_supa_event,
                db_path=self.db_path,
                type=event_type,
                conversation_id=conversation_id,
                thread_id=thread_id,
                dedupe_key=(
                    f"operation:{operation.token_id}:"
                    f"{operation.idempotency_key}:{event_type}"
                ),
                summary={
                    "operation_id": self._event_summary_identifier(
                        operation.idempotency_key,
                        prefix="op",
                    ),
                    "operation_name": operation.operation_name,
                    "request_id": self._event_summary_identifier(
                        request_id,
                        prefix="req",
                    ),
                    "result": result,
                    "status": status,
                    "token_id": self._event_summary_identifier(
                        operation.token_id,
                        prefix="token",
                    ),
                    "token_name": operation.token_name,
                },
            )
        except AgentEventStorageError as error:
            self._safe_log_exception(
                "outcome_event",
                error,
                conversation_id=conversation_id,
                request_id=request_id,
            )
            raise ConversationGatewayError(
                "The verified write could not be added to the event journal.",
                code="event_storage_error",
                retryable=True,
                status=503,
            ) from error

    @staticmethod
    def _draft_result_from_record(record: OperationRecord) -> dict[str, Any]:
        response = record.response
        if not isinstance(response, dict):
            raise ConversationGatewayError(
                "The saved operation result is invalid.",
                code="invalid_operation_result",
                retryable=False,
                status=500,
            )
        action = response.get("action")
        conversation_id = _safe_id(response.get("conversation_id"))
        thread_id = _safe_id(response.get("thread_id"))
        body_sha256 = response.get("body_sha256")
        version = response.get("version")
        if (
            action not in {"created", "updated"}
            or not isinstance(conversation_id, int)
            or not isinstance(thread_id, int)
            or not isinstance(body_sha256, str)
            or _DRAFT_VERSION_RE.fullmatch(body_sha256) is None
            or not isinstance(version, str)
            or _DRAFT_VERSION_RE.fullmatch(version) is None
        ):
            raise ConversationGatewayError(
                "The saved operation result is invalid.",
                code="invalid_operation_result",
                retryable=False,
                status=500,
            )
        return {
            "action": action,
            "conversation_id": conversation_id,
            "thread_id": thread_id,
            "body_sha256": body_sha256,
            "version": version,
        }

    @staticmethod
    def _raise_replayed_operation(record: OperationRecord) -> None:
        if record.state == "outcome_unknown":
            raise ConversationGatewayError(
                "The earlier draft write has an unknown upstream outcome.",
                code="outcome_unknown",
                retryable=False,
                status=409,
            )
        if record.state in {"reserved", "in_progress"}:
            raise ConversationGatewayError(
                "The draft write is already in progress.",
                code="operation_in_progress",
                retryable=True,
                status=409,
            )
        error = record.error or {}
        code = error.get("code")
        if not isinstance(code, str):
            code = "draft_write_failed"
        status = error.get("status")
        if not isinstance(status, int) or isinstance(status, bool):
            status = 409 if code in {"draft_conflict", "external_write_conflict"} else 502
        retryable = error.get("retryable") is True
        details = {
            key: error[key]
            for key in (
                "current_version",
                "thread_id",
                "latest_actionable_thread_id",
                "given_thread_id",
                "draft_thread_id",
                "refused_cc",
            )
            if key in error
        }
        # The ledger stores ids as text; a replay answers with the original int.
        for key in (
            "thread_id", "latest_actionable_thread_id", "given_thread_id", "draft_thread_id"
        ):
            if key in details:
                details[key] = _safe_id(details[key])
        message = "The earlier draft write failed."
        if code == "wrong_expected_thread":
            message = (
                "--expected-thread must be the latest actionable thread "
                "(the customer's latest message), not the draft's thread."
            )
        elif code == "conversation_changed" and "given_thread_id" in details:
            message = (
                f"The latest actionable thread is now {details.get('latest_actionable_thread_id')}, "
                f"not {details['given_thread_id']}."
            )
        raise ConversationGatewayError(
            message,
            code=code,
            retryable=retryable,
            status=status,
            details=details,
        )

    async def _finish_draft_failure(
        self,
        *,
        idempotency_key: str,
        identity: AgentIdentity,
        request_id: str,
        scope: str = "draft",
        code: str,
        status: int,
        retryable: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        error: dict[str, Any] = {
            "code": code,
            "status": status,
            "retryable": retryable,
        }
        error.update(details or {})
        await self._operation_finish(
            idempotency_key=idempotency_key,
            identity=identity,
            scope=scope,
            request_id=request_id,
            outcome="failed",
            error=error,
        )

    async def _finish_draft_unknown(
        self,
        *,
        idempotency_key: str,
        identity: AgentIdentity,
        request_id: str,
        code: str,
        scope: str = "draft",
    ) -> None:
        await self._operation_finish(
            idempotency_key=idempotency_key,
            identity=identity,
            scope=scope,
            request_id=request_id,
            outcome="outcome_unknown",
            error={"code": code, "retryable": False},
        )

    async def _terminalize_active_draft(
        self,
        *,
        idempotency_key: str,
        identity: AgentIdentity,
        request_id: str,
        scope: str = "draft",
        code: str,
        status: int,
        retryable: bool,
        outcome_uncertain: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Best-effort terminalization after a control-flow interruption."""

        record = await self._operation_get(
            idempotency_key=idempotency_key,
            identity=identity,
        )
        if record.state == "reserved":
            record = await self._operation_begin(
                idempotency_key=idempotency_key,
                identity=identity,
                scope=scope,
                request_id=request_id,
            )
        if record.state != "in_progress":
            return
        if outcome_uncertain:
            await self._finish_draft_unknown(
                idempotency_key=idempotency_key,
                identity=identity,
                request_id=request_id,
                code=code,
                scope=scope,
            )
        else:
            await self._finish_draft_failure(
                idempotency_key=idempotency_key,
                identity=identity,
                request_id=request_id,
                scope=scope,
                code=code,
                status=status,
                retryable=retryable,
                details=details,
            )

    @staticmethod
    def _safe_mutation_error(
        status_code: int | None,
        detail: str | None,
    ) -> tuple[str, int, bool]:
        code = detail if isinstance(detail, str) else "upstream_rejected"
        if status_code is None and code == "oauth_request_timeout":
            return code, 504, True
        if status_code is None and code == "oauth_network_error":
            return code, 503, True
        if status_code in {301, 401, 403}:
            return code, 502, False
        if status_code == 404:
            return code, 404, False
        if status_code == 412:
            return code, 412, False
        if status_code == 429:
            return code, 429, True
        if isinstance(status_code, int) and 500 <= status_code <= 599:
            return code, status_code, True
        return code, 502, False

    @staticmethod
    def _safe_read_error(error: Exception) -> tuple[str, int, bool]:
        if not isinstance(error, HelpScoutError):
            return "upstream_read_failed", 503, True

        raw_code = error.code
        code = (
            raw_code
            if isinstance(raw_code, str)
            and _SAFE_ERROR_CODE_RE.fullmatch(raw_code) is not None
            else "upstream_read_failed"
        )
        status = error.status
        if code == "request_timeout":
            return code, 504, True
        if code == "network_error":
            return code, 503, True
        if status == 404:
            return "upstream_not_found", 404, False
        if status == 412:
            return "conversation_locked", 412, False
        if status == 429:
            return "upstream_rate_limited", 429, True
        if isinstance(status, int) and 500 <= status <= 599:
            return "upstream_server_error", status, True
        return code, 502, False

    @staticmethod
    async def _verify_lease_set(leases: tuple[Any, ...]) -> None:
        for lease in leases:
            await lease.verify_owned()

    @staticmethod
    async def _run_with_lease_guards(
        leases: tuple[Any, ...],
        callback: Any,
    ) -> Any:
        async def invoke(index: int) -> Any:
            if index == len(leases):
                return await callback()

            async def next_guard() -> Any:
                return await invoke(index + 1)

            return await leases[index].run_guarded(next_guard)

        return await invoke(0)

    @asynccontextmanager
    async def _draft_lease_scope(
        self,
        conversation_ids: tuple[int, ...],
        *,
        owner: str,
        idempotency_key: str,
        identity: AgentIdentity,
        request_id: str,
        attempt_state: dict[str, Any],
        scope: str = "draft",
    ):
        """Terminalize an active operation when lease control flow aborts."""

        try:
            async with AsyncExitStack() as stack:
                leases = []
                for conversation_id in sorted(set(conversation_ids)):
                    lease = await stack.enter_async_context(
                        conversation_lease(
                            self.db_path,
                            conversation_id,
                            owner=owner,
                        )
                    )
                    leases.append(lease)
                yield tuple(leases)
        except asyncio.CancelledError:
            try:
                await self._terminalize_active_draft(
                    idempotency_key=idempotency_key,
                    identity=identity,
                    request_id=request_id,
                    scope=scope,
                    code=(
                        "cancelled_after_submission"
                        if attempt_state["callback_started"]
                        else "cancelled_before_attempt"
                    ),
                    status=499,
                    retryable=False,
                    outcome_uncertain=attempt_state["callback_started"],
                )
            except Exception:
                pass
            raise
        except ConversationLockError as error:
            if attempt_state.get("durable_success", False):
                return
            try:
                record = await self._operation_get(
                    idempotency_key=idempotency_key,
                    identity=identity,
                )
                if record.state in {"succeeded", "sent_with_warning"}:
                    return
                await self._terminalize_active_draft(
                    idempotency_key=idempotency_key,
                    identity=identity,
                    request_id=request_id,
                    scope=scope,
                    code=error.code,
                    status=error.status,
                    retryable=error.retryable,
                    outcome_uncertain=attempt_state["callback_started"],
                )
            except Exception:
                pass
            raise

    @staticmethod
    def _conversation_not_allowed() -> ConversationGatewayError:
        return ConversationGatewayError(
            "This token is not allowed to access the conversation.",
            code=CONVERSATION_NOT_ALLOWED_CODE,
            retryable=False,
            status=403,
        )

    @classmethod
    def _require_requested_allowed(cls, identity: Any, conversation_id: Any) -> None:
        """Refuse a restricted token before any reservation, lock or upstream call."""
        allows = getattr(identity, "allows_conversation", None)
        if allows is not None and not allows(conversation_id):
            raise cls._conversation_not_allowed()

    async def _require_resolved_allowed(
        self,
        *,
        identity: AgentIdentity,
        resolved_id: int,
        idempotency_key: str,
        request_id: str,
        scope: str,
    ) -> None:
        """Refuse a merged/aliased id that resolves outside the token's list.

        Runs right after the preflight read, before any mutation; the reserved
        operation is terminalized so a replay returns the same refusal.
        """
        allows = getattr(identity, "allows_conversation", None)
        if allows is None or allows(resolved_id):
            return
        await self._terminalize_active_draft(
            idempotency_key=idempotency_key,
            identity=identity,
            request_id=request_id,
            scope=scope,
            code=CONVERSATION_NOT_ALLOWED_CODE,
            status=403,
            retryable=False,
            outcome_uncertain=False,
        )
        raise self._conversation_not_allowed()

    @staticmethod
    def _resolved_id(raw: dict[str, Any]) -> int:
        resolved_id = _safe_id(raw.get("id")) if isinstance(raw, dict) else None
        if not isinstance(resolved_id, int) or resolved_id <= 0:
            raise ConversationGatewayError(
                "The resolved conversation id is invalid.",
                code="invalid_conversation",
                retryable=False,
                status=502,
            )
        return resolved_id

    @staticmethod
    def _raw_threads(raw: dict[str, Any]) -> list[dict[str, Any]]:
        embedded = raw.get("_embedded")
        threads = embedded.get("threads") if isinstance(embedded, dict) else None
        if not isinstance(threads, list) or any(
            not isinstance(thread, dict) for thread in threads
        ):
            raise ConversationGatewayError(
                "Help Scout returned invalid conversation threads.",
                code="invalid_conversation",
                retryable=False,
                status=502,
            )
        return list(threads)

    @staticmethod
    def _upstream_thread_timestamp(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                return None
            return parsed.astimezone(timezone.utc)
        except (OverflowError, TypeError, ValueError):
            return None

    @staticmethod
    def _verified_draft_result(
        *,
        action: str,
        before: dict[str, Any],
        after: dict[str, Any],
        normalized_body: str,
        resource_id: int | None,
        expected_thread_id: int | None,
        attempted_at: datetime | None,
        require_after_attempt: bool,
    ) -> dict[str, Any] | None:
        conversation_changed = (
            ConversationGateway._resolved_id(after)
            != ConversationGateway._resolved_id(before)
        )
        if conversation_changed:
            return None
        normalized_after = normalize_conversation(after)
        current = normalized_after.get("draft")
        if not isinstance(current, dict):
            return None
        current_thread_id = _safe_id(current.get("thread_id"))
        if (
            not isinstance(current_thread_id, int)
            or not _same_body(current.get("body_html"), normalized_body)
        ):
            return None
        if require_after_attempt:
            if attempted_at is None:
                return None
            raw_timestamp = (
                current.get("created_at")
                if action == "created"
                else current.get("updated_at") or current.get("created_at")
            )
            candidate_at = ConversationGateway._upstream_thread_timestamp(
                raw_timestamp
            )
            if candidate_at is None or candidate_at < attempted_at:
                return None

        if action == "updated":
            if current_thread_id != expected_thread_id:
                return None
        else:
            before_ids = {
                _safe_id(thread.get("id") or thread.get("thread_id"))
                for thread in ConversationGateway._raw_threads(before)
            }
            candidates: list[int] = []
            for thread in ConversationGateway._raw_threads(after):
                thread_id = _safe_id(thread.get("id") or thread.get("thread_id"))
                source = thread.get("source")
                source_type = source.get("type") if isinstance(source, dict) else None
                body = thread.get("body_html", thread.get("body", ""))
                if (
                    isinstance(thread_id, int)
                    and thread_id not in before_ids
                    and _is_draft_thread(thread)
                    and source_type == "api"
                    and _same_body(body, normalized_body)
                ):
                    candidates.append(thread_id)
            if resource_id is not None:
                if current_thread_id != resource_id or resource_id not in candidates:
                    return None
            elif candidates != [current_thread_id]:
                return None

        version = current.get("version")
        if not isinstance(version, str) or _DRAFT_VERSION_RE.fullmatch(version) is None:
            return None
        conversation_id = _safe_id(normalized_after.get("resolved_id"))
        if not isinstance(conversation_id, int):
            return None
        return {
            "action": action,
            "conversation_id": conversation_id,
            "thread_id": current_thread_id,
            "body_sha256": hashlib.sha256(normalized_body.encode("utf-8")).hexdigest(),
            "version": version,
        }

    @staticmethod
    def _safe_log_exception(
        event: str,
        error: BaseException,
        *,
        conversation_id: int,
        request_id: str,
    ) -> None:
        """Log only bounded operational context, never exception/body text."""

        logger.warning(
            "conversation_gateway_failure",
            extra={
                "event": event,
                "exception_type": type(error).__name__[:100],
                "conversation_id": conversation_id,
                "request_id": request_id[:200],
            },
        )

    @staticmethod
    def _latest_actionable_thread(raw: dict[str, Any]) -> dict[str, Any] | None:
        threads = ConversationGateway._raw_threads(raw)
        actionable = [
            thread
            for thread in _selection_order(threads)
            if (_is_customer_thread(thread) or _is_agent_message_thread(thread))
            and not _is_draft_thread(thread)
            and thread.get("state") in (None, "published")
        ]
        return get_latest_actionable_thread(actionable)

    @staticmethod
    def _latest_actionable_id(raw: dict[str, Any]) -> int | None:
        latest = ConversationGateway._latest_actionable_thread(raw)
        thread_id = (
            _safe_id(latest.get("id") or latest.get("thread_id"))
            if isinstance(latest, dict)
            else None
        )
        return thread_id if isinstance(thread_id, int) else None

    @staticmethod
    def _latest_customer_message(raw: dict[str, Any]) -> str:
        customers = [
            thread
            for thread in ConversationGateway._raw_threads(raw)
            if _is_customer_thread(thread) and not _is_draft_thread(thread)
        ]
        if not customers:
            return ""
        latest = get_latest_actionable_thread(_selection_order(customers))
        if latest is None:
            return ""
        body = latest.get("body_html", latest.get("body", ""))
        return strip_html(body) if isinstance(body, str) else ""

    @staticmethod
    def _send_result_from_record(record: OperationRecord) -> dict[str, Any]:
        response = record.response
        if not isinstance(response, dict):
            raise ConversationGatewayError(
                "The send operation result is invalid.",
                code="invalid_operation_result",
                retryable=False,
                status=500,
            )
        action = response.get("action")
        conversation_id = _safe_id(response.get("conversation_id"))
        thread_id = _safe_id(response.get("thread_id"))
        body_sha256 = response.get("body_sha256")
        status = response.get("status")
        warnings_value = response.get("warnings")
        valid_warnings = (
            isinstance(warnings_value, list)
            and all(
                isinstance(item, str)
                and _SAFE_ERROR_CODE_RE.fullmatch(item) is not None
                for item in warnings_value
            )
        )
        if (
            action != "sent"
            or not isinstance(conversation_id, int)
            or not isinstance(thread_id, int)
            or not isinstance(body_sha256, str)
            or _DRAFT_VERSION_RE.fullmatch(body_sha256) is None
            or status not in {"sent", "sent_with_warning"}
            or not valid_warnings
        ):
            raise ConversationGatewayError(
                "The send operation result is invalid.",
                code="invalid_operation_result",
                retryable=False,
                status=500,
            )
        result = {
            "action": "sent",
            "body_sha256": body_sha256,
            "conversation_id": conversation_id,
            "status": status,
            "thread_id": thread_id,
            "warnings": list(warnings_value),
        }
        cc_value = response.get("cc")
        if isinstance(cc_value, list) and cc_value:
            result["cc"] = [item for item in cc_value if isinstance(item, str)]
        return result

    @staticmethod
    def _sent_reply_candidates(
        *,
        before: dict[str, Any],
        after: dict[str, Any],
        normalized_body: str,
        attempted_at: datetime,
        expected_predecessor_thread_id: int,
    ) -> list[int]:
        before_ids = {
            _safe_id(thread.get("id") or thread.get("thread_id"))
            for thread in ConversationGateway._raw_threads(before)
        }
        if ConversationGateway._latest_actionable_id(before) != (
            expected_predecessor_thread_id
        ):
            return []
        # Help Scout commonly truncates ``createdAt`` to whole seconds while
        # our durable marker preserves microseconds. Treat that shared second
        # as post-attempt; the new thread ID/body/source checks still prevent
        # a pre-existing reply from being reconciled.
        attempted_second = attempted_at.replace(microsecond=0)
        actionable = [
            thread
            for thread in ConversationGateway._raw_threads(after)
            if (_is_customer_thread(thread) or _is_agent_message_thread(thread))
            and not _is_draft_thread(thread)
            and thread.get("state") in (None, "published")
        ]
        candidates: list[int] = []
        for thread in actionable:
            thread_id = _safe_id(thread.get("id") or thread.get("thread_id"))
            source = thread.get("source")
            source_type = source.get("type") if isinstance(source, dict) else None
            body = thread.get("body_html", thread.get("body", ""))
            created_at = ConversationGateway._upstream_thread_timestamp(
                thread.get("created_at") or thread.get("createdAt")
            )
            if (
                isinstance(thread_id, int)
                and thread_id not in before_ids
                and _is_agent_message_thread(thread)
                and source_type == "api"
                and _same_body(body, normalized_body)
                and created_at is not None
                and created_at >= attempted_second
            ):
                candidates.append(thread_id)
        return candidates

    @staticmethod
    def _authoritative_resource_id(mutation: MutationResult | None) -> int | None:
        """Return a definitive Help Scout reply id only for a valid 201 result."""

        if mutation is None or not mutation.ok or mutation.status_code != 201:
            return None
        resource_id = mutation.resource_id
        if (
            isinstance(resource_id, bool)
            or not isinstance(resource_id, int)
            or not 0 < resource_id <= (1 << 63) - 1
        ):
            return None
        return resource_id

    @staticmethod
    def _definitive_send_verification_warning(
        *,
        after: dict[str, Any],
        conversation_id: int,
        thread_id: int,
        normalized_body: str,
        attempted_at: datetime,
    ) -> str | None:
        """Check post-state without undoing an authoritative 201/Resource-ID."""

        try:
            if ConversationGateway._resolved_id(after) != conversation_id:
                return "verification_mismatch"
            matches = [
                thread
                for thread in ConversationGateway._raw_threads(after)
                if _safe_id(thread.get("id") or thread.get("thread_id")) == thread_id
            ]
        except Exception:
            return "verification_failed"
        if not matches:
            return "verification_unconfirmed"
        if len(matches) != 1:
            return "verification_mismatch"
        thread = matches[0]
        source = thread.get("source")
        source_type = source.get("type") if isinstance(source, dict) else None
        body = thread.get("body_html", thread.get("body", ""))
        created_at = ConversationGateway._upstream_thread_timestamp(
            thread.get("created_at") or thread.get("createdAt")
        )
        attempted_second = attempted_at.replace(microsecond=0)
        if (
            not _is_agent_message_thread(thread)
            or _is_draft_thread(thread)
            or thread.get("state") not in (None, "published")
            or source_type != "api"
            or not _same_body(body, normalized_body)
            or created_at is None
            or created_at < attempted_second
        ):
            return "verification_mismatch"
        return None

    async def _read_write_preflight(
        self,
        conversation_id: int,
        *,
        idempotency_key: str,
        identity: AgentIdentity,
        request_id: str,
        scope: str,
    ) -> dict[str, Any]:
        try:
            raw = await self.client.get_complete_conversation(conversation_id)
            self._raw_threads(raw)
            self._resolved_id(raw)
            return raw
        except asyncio.CancelledError:
            await self._terminalize_active_draft(
                idempotency_key=idempotency_key,
                identity=identity,
                request_id=request_id,
                scope=scope,
                code="cancelled_before_attempt",
                status=499,
                retryable=False,
                outcome_uncertain=False,
            )
            raise
        except ConversationGatewayError as error:
            await self._terminalize_active_draft(
                idempotency_key=idempotency_key,
                identity=identity,
                request_id=request_id,
                scope=scope,
                code=error.code,
                status=error.status,
                retryable=error.retryable,
                outcome_uncertain=False,
            )
            raise
        except Exception as error:
            self._safe_log_exception(
                "preflight_read",
                error,
                conversation_id=conversation_id,
                request_id=request_id,
            )
            code, status, retryable = self._safe_read_error(error)
            await self._terminalize_active_draft(
                idempotency_key=idempotency_key,
                identity=identity,
                request_id=request_id,
                scope=scope,
                code=code,
                status=status,
                retryable=retryable,
                outcome_uncertain=False,
            )
            raise ConversationGatewayError(
                "The conversation could not be read before the write.",
                code=code,
                retryable=retryable,
                status=status,
            ) from error

    async def _run_post_send_bookkeeping(
        self,
        *,
        before: dict[str, Any],
        conversation_id: int,
        draft_thread_ids: list[int],
        normalized_body: str,
        thread_id: int,
        request_id: str,
        initial_warning: str | None,
        ownership_check: Any,
    ) -> list[str]:
        warnings_found: list[str] = []

        def add_warning(code: str) -> None:
            if code not in warnings_found:
                warnings_found.append(code)

        if initial_warning in {
            "status_unverified",
            "status_update_failed",
            "verification_failed",
            "verification_mismatch",
            "verification_unconfirmed",
        }:
            add_warning(initial_warning)
        for draft_thread_id in draft_thread_ids:
            await ownership_check()
            try:
                result = await self.client.delete_draft_result(
                    conversation_id,
                    draft_thread_id,
                )
                if not isinstance(result, MutationResult) or not result.ok:
                    add_warning("draft_cleanup_failed")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._safe_log_exception(
                    "draft_cleanup",
                    error,
                    conversation_id=conversation_id,
                    request_id=request_id,
                )
                add_warning("draft_cleanup_failed")
        await ownership_check()
        status_result: Any = None
        try:
            status_result = await self.client.ensure_conversation_status(
                conversation_id,
                "pending",
            )
            if not isinstance(status_result, dict) or not status_result.get("verified"):
                add_warning("pending_status_unverified")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._safe_log_exception(
                "pending_status",
                error,
                conversation_id=conversation_id,
                request_id=request_id,
            )
            add_warning("pending_status_unverified")

        normalized_before = normalize_conversation(before)
        fresh_tag_source = (
            status_result.get("conversation")
            if isinstance(status_result, dict)
            and isinstance(status_result.get("conversation"), dict)
            and _safe_id(status_result["conversation"].get("id")) == conversation_id
            else None
        )
        if fresh_tag_source is None:
            await ownership_check()
            try:
                candidate = await self.client.get_complete_conversation(
                    conversation_id
                )
                if (
                    isinstance(candidate, dict)
                    and self._resolved_id(candidate) == conversation_id
                ):
                    fresh_tag_source = candidate
                else:
                    add_warning("tag_refresh_failed")
            except asyncio.CancelledError:
                raise
            except ConversationLockError:
                raise
            except Exception as error:
                self._safe_log_exception(
                    "tag_refresh",
                    error,
                    conversation_id=conversation_id,
                    request_id=request_id,
                )
                add_warning("tag_refresh_failed")
        existing_tags = (
            [
                tag
                for tag in _normalize_tags(fresh_tag_source.get("tags"))
                if tag != "supa-manual-review"
            ]
            if isinstance(fresh_tag_source, dict)
            else []
        )
        await ownership_check()
        if fresh_tag_source is not None:
            try:
                tag_result = await self.client.add_tags_result(
                    conversation_id,
                    ["supa-processed", "supa-auto-sent"],
                    existing_tags,
                )
                if not isinstance(tag_result, MutationResult) or not tag_result.ok:
                    add_warning("tag_update_failed")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._safe_log_exception(
                    "tag_update",
                    error,
                    conversation_id=conversation_id,
                    request_id=request_id,
                )
                add_warning("tag_update_failed")

        if self.post_send_bookkeeper is not None:
            await ownership_check()
            try:
                callback_kwargs = {
                    "conversation_id": conversation_id,
                    "thread_id": thread_id,
                    "subject": normalized_before.get("subject", ""),
                    "customer_message": self._latest_customer_message(before),
                    "reply_html": normalized_body,
                    "request_id": request_id,
                }
                if inspect.iscoroutinefunction(self.post_send_bookkeeper):
                    outcome = self.post_send_bookkeeper(**callback_kwargs)
                else:
                    outcome = await asyncio.to_thread(
                        self.post_send_bookkeeper,
                        **callback_kwargs,
                    )
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                if outcome is False:
                    add_warning("learning_audit_failed")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._safe_log_exception(
                    "learning_audit",
                    error,
                    conversation_id=conversation_id,
                    request_id=request_id,
                )
                add_warning("learning_audit_failed")
        return warnings_found

    async def save_draft(
        self,
        conversation_id: int,
        *,
        body: str,
        body_format: str,
        expected_version: str | None,
        idempotency_key: str,
        identity: AgentIdentity,
        request_id: str,
    ) -> dict[str, Any]:
        """Create or replace the newest live draft exactly once."""

        if (
            isinstance(conversation_id, bool)
            or not isinstance(conversation_id, int)
            or conversation_id <= 0
        ):
            raise ConversationGatewayError(
                "The conversation id must be a positive integer.",
                code="invalid_conversation",
                retryable=False,
                status=400,
            )

        if type(body) is not str:
            raise ConversationGatewayError(
                "The draft body must be a string.",
                code="invalid_body",
                retryable=False,
                status=400,
            )
        if type(body_format) is not str:
            raise ConversationGatewayError(
                "body_format must be html or text.",
                code="invalid_body_format",
                retryable=False,
                status=400,
            )
        if expected_version is not None and type(expected_version) is not str:
            raise ConversationGatewayError(
                "The expected draft version is invalid.",
                code="invalid_draft_version",
                retryable=False,
                status=400,
            )
        try:
            body_bytes = body.encode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ConversationGatewayError(
                "The draft body contains invalid Unicode.",
                code="invalid_body",
                retryable=False,
                status=400,
            ) from error
        try:
            body_format.encode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ConversationGatewayError(
                "The draft body format contains invalid Unicode.",
                code="invalid_body_format",
                retryable=False,
                status=400,
            ) from error
        if expected_version is not None:
            try:
                expected_version.encode("utf-8", errors="strict")
            except UnicodeError as error:
                raise ConversationGatewayError(
                    "The expected draft version contains invalid Unicode.",
                    code="invalid_draft_version",
                    retryable=False,
                    status=400,
                ) from error
        if len(body_bytes) > _MAX_DRAFT_BODY_BYTES:
            raise ConversationGatewayError(
                "The draft body is too large.",
                code="body_too_large",
                retryable=False,
                status=413,
            )

        request = {
            "body": body,
            "body_format": body_format,
            "expected_version": expected_version,
        }
        try:
            canonical_request_json(request)
        except ValueError as error:
            raise ConversationGatewayError(
                "The draft request is not valid JSON.",
                code="invalid_body",
                retryable=False,
                status=400,
            ) from error

        self._require_requested_allowed(identity, conversation_id)
        metadata: dict[str, Any] = {
            "body_format": body_format,
            "conversation_id": conversation_id,
            "expected_version": expected_version,
        }
        # Cancellation cannot safely terminalize here: the durable reservation
        # may be a replay owned by another in-flight waiter. A newly-created
        # reservation is safe to leave reserved for startup recovery.
        reservation = await self._operation_reserve(
            idempotency_key=idempotency_key,
            identity=identity,
            scope="draft",
            operation_name="draft.save",
            target_id=conversation_id,
            request_id=request_id,
            request=request,
            request_metadata=metadata,
        )
        if reservation.replayed:
            if reservation.operation.state == "succeeded":
                replayed = self._draft_result_from_record(reservation.operation)
                await self._emit_operation_event(
                    operation=reservation.operation,
                    request_id=request_id,
                    event_type="supa.draft.saved",
                    conversation_id=replayed["conversation_id"],
                    thread_id=replayed["thread_id"],
                    result=replayed["action"],
                    status="saved",
                )
                return replayed
            self._raise_replayed_operation(reservation.operation)

        owner = hashlib.sha256(
            f"{identity.token_id}\0{idempotency_key}\0{request_id}".encode("utf-8")
        ).hexdigest()
        lock_ids = (conversation_id,)
        for _ in range(_MAX_RESOLUTION_ATTEMPTS):
            attempt_state = {
                "callback_started": False,
                "durable_success": False,
            }
            async with self._draft_lease_scope(
                lock_ids,
                owner=owner,
                idempotency_key=idempotency_key,
                identity=identity,
                request_id=request_id,
                attempt_state=attempt_state,
            ) as leases:
                try:
                    before = await self.client.get_complete_conversation(conversation_id)
                    resolved_id = self._resolved_id(before)
                except asyncio.CancelledError:
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        code="cancelled_before_attempt",
                        status=499,
                        retryable=False,
                        outcome_uncertain=False,
                    )
                    raise
                except ConversationGatewayError as error:
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        code=error.code,
                        status=error.status,
                        retryable=error.retryable,
                        outcome_uncertain=False,
                    )
                    raise
                except Exception as error:
                    code, status, retryable = self._safe_read_error(error)
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        code=code,
                        status=status,
                        retryable=retryable,
                        outcome_uncertain=False,
                    )
                    raise ConversationGatewayError(
                        "The conversation could not be read before saving the draft.",
                        code=code,
                        retryable=retryable,
                        status=status,
                    ) from error
                await self._require_resolved_allowed(
                    identity=identity,
                    resolved_id=resolved_id,
                    idempotency_key=idempotency_key,
                    request_id=request_id,
                    scope="draft",
                )
                required_lock_ids = tuple(sorted({conversation_id, resolved_id}))
                if required_lock_ids != lock_ids:
                    lock_ids = required_lock_ids
                    continue

                try:
                    self._raw_threads(before)
                    normalized_before = normalize_conversation(before)
                except ConversationGatewayError as error:
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        code=error.code,
                        status=error.status,
                        retryable=error.retryable,
                        outcome_uncertain=False,
                    )
                    raise
                except Exception as error:
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        code="invalid_conversation",
                        status=502,
                        retryable=False,
                        outcome_uncertain=False,
                    )
                    raise ConversationGatewayError(
                        "Help Scout returned invalid conversation data.",
                        code="invalid_conversation",
                        status=502,
                        retryable=False,
                    ) from error
                current = normalized_before.get("draft")
                current_version = (
                    current.get("version") if isinstance(current, dict) else None
                )
                current_thread_id = (
                    _safe_id(current.get("thread_id"))
                    if isinstance(current, dict)
                    else None
                )
                if (
                    expected_version is not None
                    and _DRAFT_VERSION_RE.fullmatch(expected_version) is None
                ):
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        code="invalid_draft_version",
                        status=400,
                        retryable=False,
                        outcome_uncertain=False,
                    )
                    raise ConversationGatewayError(
                        "The expected draft version is invalid.",
                        code="invalid_draft_version",
                        retryable=False,
                        status=400,
                    )
                conflict = (
                    (expected_version is None and current is not None)
                    or (
                        expected_version is not None
                        and (
                            current is None
                            or current_version != expected_version
                            or not isinstance(current_thread_id, int)
                        )
                    )
                )
                if conflict:
                    details: dict[str, Any] = {}
                    if isinstance(current_version, str):
                        details["current_version"] = current_version
                    if isinstance(current_thread_id, int):
                        details["thread_id"] = current_thread_id
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        code="draft_conflict",
                        status=409,
                        retryable=False,
                        outcome_uncertain=False,
                        details=details,
                    )
                    raise ConversationGatewayError(
                        "The live draft no longer matches the expected version.",
                        code="draft_conflict",
                        retryable=False,
                        status=409,
                        details=details,
                    )

                try:
                    normalized_body = normalize_draft_body(body, body_format)
                except ConversationGatewayError as error:
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        code=error.code,
                        status=error.status,
                        retryable=error.retryable,
                        outcome_uncertain=False,
                    )
                    raise
                normalized_body_hash = hashlib.sha256(
                    normalized_body.encode("utf-8")
                ).hexdigest()

                action = "created" if current is None else "updated"
                customer = normalized_before.get("customer")
                customer_id = (
                    _safe_id(customer.get("id")) if isinstance(customer, dict) else None
                )
                marker_started = False
                try:
                    await self._verify_lease_set(leases)
                    await self._operation_begin(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        scope="draft",
                        request_id=request_id,
                    )
                    marker_started = True
                    started = await self._operation_mark_upstream(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        scope="draft",
                        request_id=request_id,
                        attempt_metadata={
                            "normalized_body_hash": normalized_body_hash,
                            "action": action,
                            "conversation_id": resolved_id,
                            "expected_thread_id": current_thread_id,
                        },
                    )
                except (asyncio.CancelledError, ConversationLockError):
                    raise
                except Exception:
                    try:
                        await self._terminalize_active_draft(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            code=(
                                "operation_marker_failed"
                                if marker_started
                                else "operation_begin_failed"
                            ),
                            status=500,
                            retryable=False,
                            outcome_uncertain=False,
                        )
                    except Exception:
                        pass
                    raise

                async def submit_draft():
                    attempt_state["callback_started"] = True
                    if action == "created":
                        return await self.client.add_draft_result(
                            resolved_id,
                            normalized_body,
                            customer_id=(
                                customer_id if isinstance(customer_id, int) else None
                            ),
                        )
                    assert isinstance(current_thread_id, int)
                    return await self.client.update_draft_result(
                        resolved_id,
                        current_thread_id,
                        normalized_body,
                    )

                try:
                    mutation = await self._run_with_lease_guards(
                        leases,
                        submit_draft,
                    )
                except asyncio.CancelledError:
                    await self._durable_thread_call(
                        finish_operation,
                        db_path=self.db_path,
                        idempotency_key=idempotency_key,
                        identity=identity,
                        scope="draft",
                        request_id=request_id,
                        outcome=(
                            "outcome_unknown"
                            if attempt_state["callback_started"]
                            else "failed"
                        ),
                        error={
                            "code": (
                                "cancelled_after_submission"
                                if attempt_state["callback_started"]
                                else "cancelled_before_attempt"
                            ),
                            "retryable": False,
                            "status": 499,
                        },
                    )
                    raise
                except ConversationLockError:
                    if not attempt_state["callback_started"]:
                        await self._terminalize_active_draft(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            code="conversation_lock_lost",
                            status=409,
                            retryable=False,
                            outcome_uncertain=False,
                        )
                    raise
                except Exception:
                    mutation = None

                if not isinstance(mutation, MutationResult):
                    mutation = None

                uncertain = mutation is None or mutation.outcome_unknown

                if mutation is not None and not mutation.ok and not mutation.outcome_unknown:
                    code, status, retryable = self._safe_mutation_error(
                        mutation.status_code,
                        mutation.detail,
                    )
                    await self._finish_draft_failure(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        code=code,
                        status=status,
                        retryable=retryable,
                    )
                    raise ConversationGatewayError(
                        "Help Scout rejected the draft write.",
                        code=code,
                        retryable=retryable,
                        status=status,
                    )

                try:
                    after = await self.client.get_complete_conversation(resolved_id)
                except asyncio.CancelledError:
                    await self._finish_draft_unknown(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        code="cancelled_during_verification",
                    )
                    raise
                except Exception:
                    after = None

                missing_create_identity = (
                    action == "created"
                    and mutation is not None
                    and mutation.ok
                    and mutation.resource_id is None
                )
                post_state_valid = isinstance(after, dict)
                try:
                    if not isinstance(after, dict):
                        verified = None
                    else:
                        self._raw_threads(after)
                        self._resolved_id(after)
                        verified = self._verified_draft_result(
                            action=action,
                            before=before,
                            after=after,
                            normalized_body=normalized_body,
                            resource_id=(
                                mutation.resource_id
                                if mutation is not None and action == "created"
                                else None
                            ),
                            expected_thread_id=(
                                current_thread_id
                                if isinstance(current_thread_id, int)
                                else None
                            ),
                            attempted_at=started.upstream_started_at,
                            require_after_attempt=(
                                uncertain or missing_create_identity
                            ),
                        )
                except Exception:
                    post_state_valid = False
                    verified = None
                if verified is None:
                    if uncertain or missing_create_identity or not post_state_valid:
                        await self._finish_draft_unknown(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            code="upstream_outcome_unknown",
                        )
                        raise ConversationGatewayError(
                            "The draft write may have reached Help Scout.",
                            code="outcome_unknown",
                            retryable=False,
                            status=409,
                        )
                    await self._finish_draft_failure(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        code="external_write_conflict",
                        status=409,
                        retryable=False,
                    )
                    raise ConversationGatewayError(
                        "The final Help Scout draft differs from the intended write.",
                        code="external_write_conflict",
                        retryable=False,
                        status=409,
                    )

                completed = await self._operation_finish(
                    idempotency_key=idempotency_key,
                    identity=identity,
                    scope="draft",
                    request_id=request_id,
                    outcome="succeeded",
                    upstream_thread_id=verified["thread_id"],
                    response=verified,
                )
                attempt_state["durable_success"] = True
                await self._emit_operation_event(
                    operation=completed,
                    request_id=request_id,
                    event_type="supa.draft.saved",
                    conversation_id=verified["conversation_id"],
                    thread_id=verified["thread_id"],
                    result=verified["action"],
                    status="saved",
                )
                return verified

        await self._terminalize_active_draft(
            idempotency_key=idempotency_key,
            identity=identity,
            request_id=request_id,
            code="conversation_resolution_changed",
            status=409,
            retryable=True,
            outcome_uncertain=False,
        )
        raise ConversationGatewayError(
            "The resolved conversation changed while acquiring its write lock.",
            code="conversation_resolution_changed",
            retryable=True,
            status=409,
        )

    async def send_draft(
        self,
        conversation_id: int,
        *,
        expected_version: str,
        expected_latest_thread_id: int,
        idempotency_key: str,
        identity: AgentIdentity,
        request_id: str,
        cc: Any = None,
    ) -> dict[str, Any]:
        """Publish exactly the draft/version and predecessor the caller read.

        The draft's own CC (pinned by its version) goes out with it, plus `cc`.
        """

        if type(expected_version) is not str:
            raise ConversationGatewayError(
                "The expected draft version is invalid.",
                code="invalid_draft_version",
                status=400,
            )
        try:
            expected_version.encode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ConversationGatewayError(
                "The expected draft version contains invalid Unicode.",
                code="invalid_draft_version",
                status=400,
            ) from error
        return await self._send_message(
            conversation_id,
            mode="draft",
            body=None,
            body_format=None,
            expected_version=expected_version,
            expected_latest_thread_id=expected_latest_thread_id,
            idempotency_key=idempotency_key,
            identity=identity,
            request_id=request_id,
            cc=cc,
        )

    async def send_reply(
        self,
        conversation_id: int,
        *,
        body: str,
        body_format: str,
        expected_latest_thread_id: int,
        idempotency_key: str,
        identity: AgentIdentity,
        request_id: str,
        send_guard: Any | None = None,
        cc: Any = None,
    ) -> dict[str, Any]:
        """Send one direct reply over the exact latest actionable thread.

        `cc` (at most two addresses) must each be an address the customer gave in
        this conversation and not on do-not-contact (`check_cc`).

        `send_guard`, when given, is called with the raw preflight conversation
        under the conversation lease, before any other precondition. A non-empty
        string return value is a hold reason: the operation fails with
        `scheduled_send_held` and nothing is sent (scheduled sends use it).
        """

        if type(body) is not str:
            raise ConversationGatewayError(
                "The reply body must be a string.",
                code="invalid_body",
                status=400,
            )
        if type(body_format) is not str:
            raise ConversationGatewayError(
                "body_format must be html or text.",
                code="invalid_body_format",
                status=400,
            )
        try:
            body_bytes = body.encode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ConversationGatewayError(
                "The reply body contains invalid Unicode.",
                code="invalid_body",
                status=400,
            ) from error
        try:
            body_format.encode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ConversationGatewayError(
                "The reply body format contains invalid Unicode.",
                code="invalid_body_format",
                status=400,
            ) from error
        if len(body_bytes) > _MAX_DRAFT_BODY_BYTES:
            raise ConversationGatewayError(
                "The reply body is too large.",
                code="body_too_large",
                status=413,
            )
        return await self._send_message(
            conversation_id,
            mode="reply",
            body=body,
            body_format=body_format,
            expected_version=None,
            expected_latest_thread_id=expected_latest_thread_id,
            idempotency_key=idempotency_key,
            identity=identity,
            request_id=request_id,
            send_guard=send_guard,
            cc=cc,
        )

    async def check_cc(self, conversation: dict[str, Any], cc: list[str]) -> None:
        """Validate each requested recipient against the conversation, then apply registered recipient checks. A failed check holds the operation."""
        if not cc:
            return
        if len(cc) > MAX_CC_ADDRESSES:
            raise _invalid_cc(f"At most {MAX_CC_ADDRESSES} cc addresses.")
        allowed = cc_allowed_addresses(conversation)
        refused = [address for address in cc if address not in allowed]
        if refused:
            raise ConversationGatewayError(
                "A cc address was not given by the customer in this conversation.",
                code="cc_not_allowed",
                status=403,
                details={"refused_cc": refused},
            )
        for address in cc:
            try:
                reason = await asyncio.to_thread(self.cc_block_check, address)
            except Exception as error:
                logger.warning("cc do-not-contact check failed: %s", type(error).__name__)
                raise ConversationGatewayError(
                    "The do-not-contact list could not be checked.",
                    code="cc_check_failed",
                    status=503,
                    retryable=True,
                ) from error
            if reason:
                raise ConversationGatewayError(
                    "A cc address is on the do-not-contact list.",
                    code="cc_blocked",
                    status=403,
                    details={"refused_cc": [address]},
                )

    async def _send_message(
        self,
        conversation_id: int,
        *,
        mode: str,
        body: str | None,
        body_format: str | None,
        expected_version: str | None,
        expected_latest_thread_id: int,
        idempotency_key: str,
        identity: AgentIdentity,
        request_id: str,
        send_guard: Any | None = None,
        cc: Any = None,
    ) -> dict[str, Any]:
        if (
            isinstance(conversation_id, bool)
            or not isinstance(conversation_id, int)
            or conversation_id <= 0
        ):
            raise ConversationGatewayError(
                "The conversation id must be a positive integer.",
                code="invalid_conversation",
                status=400,
            )
        if (
            isinstance(expected_latest_thread_id, bool)
            or not isinstance(expected_latest_thread_id, int)
            or expected_latest_thread_id <= 0
        ):
            raise ConversationGatewayError(
                "The expected latest thread id must be a positive integer.",
                code="invalid_thread_id",
                status=400,
            )
        requested_cc = normalize_cc(cc)
        request = (
            {
                "expected_latest_thread_id": expected_latest_thread_id,
                "expected_version": expected_version,
            }
            if mode == "draft"
            else {
                "body": body,
                "body_format": body_format,
                "expected_latest_thread_id": expected_latest_thread_id,
            }
        )
        if requested_cc:
            # Only when given, so requests without CC keep their idempotency hash.
            request["cc"] = requested_cc
        try:
            canonical_request_json(request)
        except ValueError as error:
            raise ConversationGatewayError(
                "The send request is not valid JSON.",
                code="invalid_body",
                status=400,
            ) from error
        self._require_requested_allowed(identity, conversation_id)
        operation_name = "draft.send" if mode == "draft" else "reply.send"
        request_metadata = {
            "body_format": body_format,
            "conversation_id": conversation_id,
            "expected_predecessor_thread_id": expected_latest_thread_id,
            "expected_version": expected_version,
        }
        if requested_cc:
            request_metadata["cc"] = requested_cc
        # As above, a cancelled waiter cannot distinguish a reservation it
        # created from a replay of another request's active row.
        reservation = await self._operation_reserve(
            idempotency_key=idempotency_key,
            identity=identity,
            scope="send",
            operation_name=operation_name,
            target_id=conversation_id,
            request_id=request_id,
            request=request,
            request_metadata=request_metadata,
        )
        if reservation.replayed:
            if reservation.operation.state in {"succeeded", "sent_with_warning"}:
                replayed = self._send_result_from_record(reservation.operation)
                await self._emit_operation_event(
                    operation=reservation.operation,
                    request_id=request_id,
                    event_type="supa.reply.sent",
                    conversation_id=replayed["conversation_id"],
                    thread_id=replayed["thread_id"],
                    result="sent",
                    status=replayed["status"],
                )
                return replayed
            self._raise_replayed_operation(reservation.operation)

        owner = hashlib.sha256(
            f"{identity.token_id}\0{idempotency_key}\0{request_id}".encode("utf-8")
        ).hexdigest()
        lock_ids = (conversation_id,)
        for _ in range(_MAX_RESOLUTION_ATTEMPTS):
            attempt_state: dict[str, Any] = {
                "callback_started": False,
                "durable_success": False,
            }
            async with self._draft_lease_scope(
                lock_ids,
                owner=owner,
                idempotency_key=idempotency_key,
                identity=identity,
                request_id=request_id,
                attempt_state=attempt_state,
                scope="send",
            ) as leases:
                before = await self._read_write_preflight(
                    conversation_id,
                    idempotency_key=idempotency_key,
                    identity=identity,
                    request_id=request_id,
                    scope="send",
                )
                resolved_id = self._resolved_id(before)
                await self._require_resolved_allowed(
                    identity=identity,
                    resolved_id=resolved_id,
                    idempotency_key=idempotency_key,
                    request_id=request_id,
                    scope="send",
                )
                required_lock_ids = tuple(sorted({conversation_id, resolved_id}))
                if required_lock_ids != lock_ids:
                    lock_ids = required_lock_ids
                    continue
                try:
                    normalized_before = normalize_conversation(before)
                except Exception as error:
                    self._safe_log_exception(
                        "preflight_normalization",
                        error,
                        conversation_id=conversation_id,
                        request_id=request_id,
                    )
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="send",
                        code="invalid_conversation",
                        status=502,
                        retryable=False,
                        outcome_uncertain=False,
                    )
                    raise ConversationGatewayError(
                        "Help Scout returned invalid conversation data.",
                        code="invalid_conversation",
                        status=502,
                    ) from error

                if send_guard is not None:
                    hold_reason = send_guard(before)
                    if hold_reason:
                        await self._terminalize_active_draft(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="send",
                            code="scheduled_send_held",
                            status=409,
                            retryable=False,
                            outcome_uncertain=False,
                        )
                        raise ConversationGatewayError(
                            "The scheduled send is held.",
                            code="scheduled_send_held",
                            status=409,
                            details={"hold_reason": str(hold_reason)},
                        )

                thread_error = expected_thread_error(
                    normalized_before, expected_latest_thread_id
                )
                if thread_error is not None:
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="send",
                        code=thread_error.code,
                        status=thread_error.status,
                        retryable=thread_error.retryable,
                        outcome_uncertain=False,
                        details=thread_error.details,
                    )
                    raise thread_error

                current = normalized_before.get("draft")
                if mode == "draft":
                    current_version = (
                        current.get("version") if isinstance(current, dict) else None
                    )
                    if expected_version is None or _DRAFT_VERSION_RE.fullmatch(
                        expected_version
                    ) is None:
                        await self._terminalize_active_draft(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="send",
                            code="invalid_draft_version",
                            status=400,
                            retryable=False,
                            outcome_uncertain=False,
                        )
                        raise ConversationGatewayError(
                            "The expected draft version is invalid.",
                            code="invalid_draft_version",
                            status=400,
                        )
                    if current_version != expected_version:
                        details = {}
                        if isinstance(current_version, str):
                            details["current_version"] = current_version
                        await self._terminalize_active_draft(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="send",
                            code="draft_conflict",
                            status=409,
                            retryable=False,
                            outcome_uncertain=False,
                            details=details,
                        )
                        raise ConversationGatewayError(
                            "The live draft no longer matches the expected version.",
                            code="draft_conflict",
                            status=409,
                            details=details,
                        )
                    raw_body = current.get("body_html")
                    try:
                        normalized_body = normalize_draft_body(raw_body, "html")
                    except ConversationGatewayError as error:
                        await self._terminalize_active_draft(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="send",
                            code=error.code,
                            status=error.status,
                            retryable=error.retryable,
                            outcome_uncertain=False,
                        )
                        raise
                else:
                    assert body is not None and body_format is not None
                    try:
                        normalized_body = normalize_draft_body(body, body_format)
                    except ConversationGatewayError as error:
                        await self._terminalize_active_draft(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="send",
                            code=error.code,
                            status=error.status,
                            retryable=error.retryable,
                            outcome_uncertain=False,
                        )
                        raise

                # The draft's stored CC goes out with it (its version pins it).
                stored_cc = (
                    current.get("cc")
                    if mode == "draft" and isinstance(current, dict)
                    else None
                )
                try:
                    effective_cc = normalize_cc([*(stored_cc or []), *requested_cc])
                    await self.check_cc(normalized_before, effective_cc)
                except ConversationGatewayError as error:
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="send",
                        code=error.code,
                        status=error.status,
                        retryable=error.retryable,
                        outcome_uncertain=False,
                        details=error.details,
                    )
                    raise
                with_cc: dict[str, Any] = (
                    {"cc": list(effective_cc)} if effective_cc else {}
                )

                normalized_body_hash = hashlib.sha256(
                    normalized_body.encode("utf-8")
                ).hexdigest()
                customer = normalized_before.get("customer")
                customer_id = (
                    _safe_id(customer.get("id"))
                    if isinstance(customer, dict)
                    else None
                )
                draft_thread_ids = [
                    thread_id
                    for thread in self._raw_threads(before)
                    if _is_draft_thread(thread)
                    and isinstance(
                        thread_id := _safe_id(
                            thread.get("id") or thread.get("thread_id")
                        ),
                        int,
                    )
                ]
                marker_started = False
                try:
                    await self._verify_lease_set(leases)
                    await self._operation_begin(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        scope="send",
                        request_id=request_id,
                    )
                    marker_started = True
                    started = await self._operation_mark_upstream(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        scope="send",
                        request_id=request_id,
                        attempt_metadata={
                            "normalized_body_hash": normalized_body_hash,
                            "action": "sent",
                            "conversation_id": resolved_id,
                            "expected_predecessor_thread_id": expected_latest_thread_id,
                        },
                    )
                except (asyncio.CancelledError, ConversationLockError):
                    raise
                except Exception as error:
                    self._safe_log_exception(
                        "send_marker",
                        error,
                        conversation_id=resolved_id,
                        request_id=request_id,
                    )
                    try:
                        await self._terminalize_active_draft(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="send",
                            code=(
                                "operation_marker_failed"
                                if marker_started
                                else "operation_begin_failed"
                            ),
                            status=500,
                            retryable=False,
                            outcome_uncertain=False,
                        )
                    except Exception:
                        pass
                    raise

                async def submit_reply() -> Any:
                    attempt_state["callback_started"] = True
                    return await self.client.send_reply_result(
                        resolved_id,
                        normalized_body,
                        status="pending",
                        customer_id=(
                            customer_id if isinstance(customer_id, int) else None
                        ),
                        verify_status=False,
                        **with_cc,
                    )

                try:
                    mutation = await self._run_with_lease_guards(
                        leases,
                        submit_reply,
                    )
                except asyncio.CancelledError:
                    await self._durable_thread_call(
                        finish_operation,
                        db_path=self.db_path,
                        idempotency_key=idempotency_key,
                        identity=identity,
                        scope="send",
                        request_id=request_id,
                        outcome=(
                            "outcome_unknown"
                            if attempt_state["callback_started"]
                            else "failed"
                        ),
                        error={
                            "code": (
                                "cancelled_after_submission"
                                if attempt_state["callback_started"]
                                else "cancelled_before_attempt"
                            ),
                            "retryable": False,
                            "status": 499,
                        },
                    )
                    raise
                except ConversationLockError:
                    if not attempt_state["callback_started"]:
                        await self._terminalize_active_draft(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="send",
                            code="conversation_lock_lost",
                            status=409,
                            retryable=False,
                            outcome_uncertain=False,
                        )
                    raise
                except Exception as error:
                    self._safe_log_exception(
                        "send_submission",
                        error,
                        conversation_id=resolved_id,
                        request_id=request_id,
                    )
                    mutation = None

                if not isinstance(mutation, MutationResult):
                    mutation = None
                if mutation is not None and not mutation.ok and not mutation.outcome_unknown:
                    code, status, retryable = self._safe_mutation_error(
                        mutation.status_code,
                        mutation.detail,
                    )
                    await self._finish_draft_failure(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="send",
                        code=code,
                        status=status,
                        retryable=retryable,
                    )
                    raise ConversationGatewayError(
                        "Help Scout rejected the reply send.",
                        code=code,
                        retryable=retryable,
                        status=status,
                    )

                confirmed_thread_id = self._authoritative_resource_id(mutation)
                verification_warning: str | None = None
                try:
                    after = await self.client.get_complete_conversation(resolved_id)
                except asyncio.CancelledError:
                    if confirmed_thread_id is None:
                        await self._finish_draft_unknown(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="send",
                            code="cancelled_during_verification",
                        )
                    else:
                        response = {
                            "action": "sent",
                            "body_sha256": normalized_body_hash,
                            "conversation_id": resolved_id,
                            "status": "sent_with_warning",
                            "thread_id": confirmed_thread_id,
                            "warnings": ["verification_cancelled"],
                            **with_cc,
                        }
                        completed = await self._operation_finish(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            scope="send",
                            request_id=request_id,
                            outcome="sent_with_warning",
                            upstream_thread_id=confirmed_thread_id,
                            response=response,
                        )
                        attempt_state["durable_success"] = True
                        try:
                            await self._emit_operation_event(
                                operation=completed,
                                request_id=request_id,
                                event_type="supa.reply.sent",
                                conversation_id=resolved_id,
                                thread_id=confirmed_thread_id,
                                result="sent",
                                status="sent_with_warning",
                            )
                        except ConversationGatewayError:
                            # Cancellation stays authoritative; replay repairs the event.
                            pass
                    raise
                except Exception as error:
                    self._safe_log_exception(
                        "send_verification",
                        error,
                        conversation_id=resolved_id,
                        request_id=request_id,
                    )
                    after = None
                    if confirmed_thread_id is not None:
                        verification_warning = "verification_failed"

                if confirmed_thread_id is not None and isinstance(after, dict):
                    verification_warning = self._definitive_send_verification_warning(
                        after=after,
                        conversation_id=resolved_id,
                        thread_id=confirmed_thread_id,
                        normalized_body=normalized_body,
                        attempted_at=started.upstream_started_at,
                    )
                if confirmed_thread_id is None and isinstance(after, dict):
                    try:
                        if self._resolved_id(after) == resolved_id:
                            candidates = self._sent_reply_candidates(
                                before=before,
                                after=after,
                                normalized_body=normalized_body,
                                attempted_at=started.upstream_started_at,
                                expected_predecessor_thread_id=expected_latest_thread_id,
                            )
                            if len(candidates) == 1:
                                confirmed_thread_id = candidates[0]
                    except Exception as error:
                        self._safe_log_exception(
                            "send_reconciliation",
                            error,
                            conversation_id=resolved_id,
                            request_id=request_id,
                        )
                if confirmed_thread_id is None:
                    await self._finish_draft_unknown(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="send",
                        code="upstream_outcome_unknown",
                    )
                    raise ConversationGatewayError(
                        "The reply may have reached Help Scout.",
                        code="outcome_unknown",
                        retryable=False,
                        status=409,
                    )

                if (
                    effective_cc
                    and verification_warning is None
                    and isinstance(after, dict)
                ):
                    sent_thread = next(
                        (
                            thread
                            for thread in self._raw_threads(after)
                            if _safe_id(thread.get("id")) == confirmed_thread_id
                        ),
                        None,
                    )
                    shown_cc = {
                        address.lower()
                        for address in (
                            _thread_cc(sent_thread) if isinstance(sent_thread, dict) else []
                        )
                    }
                    if not set(effective_cc) <= shown_cc:
                        verification_warning = "cc_unverified"

                initial_warning = (
                    verification_warning
                    or (mutation.detail if mutation is not None else None)
                )

                async def run_bookkeeping() -> list[str]:
                    return await self._run_post_send_bookkeeping(
                        before=before,
                        conversation_id=resolved_id,
                        draft_thread_ids=draft_thread_ids,
                        normalized_body=normalized_body,
                        thread_id=confirmed_thread_id,
                        request_id=request_id,
                        initial_warning=initial_warning,
                        ownership_check=lambda: self._verify_lease_set(leases),
                    )

                try:
                    warnings_found = await self._run_with_lease_guards(
                        leases,
                        run_bookkeeping,
                    )
                except asyncio.CancelledError:
                    response = {
                        "action": "sent",
                        "body_sha256": normalized_body_hash,
                        "conversation_id": resolved_id,
                        "status": "sent_with_warning",
                        "thread_id": confirmed_thread_id,
                        "warnings": ["bookkeeping_cancelled"],
                        **with_cc,
                    }
                    completed = await self._operation_finish(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        scope="send",
                        request_id=request_id,
                        outcome="sent_with_warning",
                        upstream_thread_id=confirmed_thread_id,
                        response=response,
                    )
                    attempt_state["durable_success"] = True
                    try:
                        await self._emit_operation_event(
                            operation=completed,
                            request_id=request_id,
                            event_type="supa.reply.sent",
                            conversation_id=resolved_id,
                            thread_id=confirmed_thread_id,
                            result="sent",
                            status="sent_with_warning",
                        )
                    except ConversationGatewayError:
                        # Cancellation stays authoritative; replay repairs the event.
                        pass
                    raise
                except ConversationLockError as error:
                    self._safe_log_exception(
                        "send_bookkeeping_lock",
                        error,
                        conversation_id=resolved_id,
                        request_id=request_id,
                    )
                    warnings_found = [error.code]
                except Exception as error:
                    self._safe_log_exception(
                        "send_bookkeeping",
                        error,
                        conversation_id=resolved_id,
                        request_id=request_id,
                    )
                    warnings_found = ["bookkeeping_failed"]

                outcome = "sent_with_warning" if warnings_found else "succeeded"
                response = {
                    "action": "sent",
                    "body_sha256": normalized_body_hash,
                    "conversation_id": resolved_id,
                    "status": (
                        "sent_with_warning" if warnings_found else "sent"
                    ),
                    "thread_id": confirmed_thread_id,
                    "warnings": warnings_found,
                    **with_cc,
                }
                completed = await self._operation_finish(
                    idempotency_key=idempotency_key,
                    identity=identity,
                    scope="send",
                    request_id=request_id,
                    outcome=outcome,
                    upstream_thread_id=confirmed_thread_id,
                    response=response,
                )
                attempt_state["durable_success"] = True
                await self._emit_operation_event(
                    operation=completed,
                    request_id=request_id,
                    event_type="supa.reply.sent",
                    conversation_id=resolved_id,
                    thread_id=confirmed_thread_id,
                    result="sent",
                    status=response["status"],
                )
                return response

        await self._terminalize_active_draft(
            idempotency_key=idempotency_key,
            identity=identity,
            request_id=request_id,
            scope="send",
            code="conversation_resolution_changed",
            status=409,
            retryable=True,
            outcome_uncertain=False,
        )
        raise ConversationGatewayError(
            "The resolved conversation changed while acquiring its write lock.",
            code="conversation_resolution_changed",
            retryable=True,
            status=409,
        )

    @staticmethod
    def _note_result_from_record(record: OperationRecord) -> dict[str, Any]:
        response = record.response
        if not isinstance(response, dict):
            raise ConversationGatewayError(
                "The note operation result is invalid.",
                code="invalid_operation_result",
                retryable=False,
                status=500,
            )
        conversation_id = _safe_id(response.get("conversation_id"))
        thread_id = _safe_id(response.get("thread_id"))
        body_sha256 = response.get("body_sha256")
        conversation_status = response.get("conversation_status")
        status = response.get("status")
        warnings_value = response.get("warnings")
        valid_warnings = isinstance(warnings_value, list) and all(
            isinstance(item, str) and _SAFE_ERROR_CODE_RE.fullmatch(item) is not None
            for item in warnings_value
        )
        if (
            response.get("action") != "added"
            or not isinstance(conversation_id, int)
            or not isinstance(thread_id, int)
            or not isinstance(body_sha256, str)
            or _DRAFT_VERSION_RE.fullmatch(body_sha256) is None
            or conversation_status not in _NOTE_CONVERSATION_STATUSES
            or status not in {"added", "added_with_warning"}
            or not valid_warnings
        ):
            raise ConversationGatewayError(
                "The note operation result is invalid.",
                code="invalid_operation_result",
                retryable=False,
                status=500,
            )
        return {
            "action": "added",
            "body_sha256": body_sha256,
            "conversation_id": conversation_id,
            "conversation_status": conversation_status,
            "status": status,
            "thread_id": thread_id,
            "warnings": list(warnings_value),
        }

    @staticmethod
    def _is_note_thread(thread: dict[str, Any]) -> bool:
        return thread.get("type") == "note" and not _is_draft_thread(thread)

    @staticmethod
    def _new_note_candidates(
        *,
        before: dict[str, Any],
        after: dict[str, Any],
        normalized_body: str,
        attempted_at: datetime,
    ) -> list[int]:
        """Return new API notes with the exact body created after the attempt."""

        before_ids = {
            _safe_id(thread.get("id") or thread.get("thread_id"))
            for thread in ConversationGateway._raw_threads(before)
        }
        attempted_second = attempted_at.replace(microsecond=0)
        candidates: list[int] = []
        for thread in ConversationGateway._raw_threads(after):
            thread_id = _safe_id(thread.get("id") or thread.get("thread_id"))
            source = thread.get("source")
            source_type = source.get("type") if isinstance(source, dict) else None
            created_at = ConversationGateway._upstream_thread_timestamp(
                thread.get("created_at") or thread.get("createdAt")
            )
            if (
                isinstance(thread_id, int)
                and thread_id not in before_ids
                and ConversationGateway._is_note_thread(thread)
                and source_type == "api"
                and _same_body(thread.get("body_html", thread.get("body", "")), normalized_body)
                and created_at is not None
                and created_at >= attempted_second
            ):
                candidates.append(thread_id)
        return candidates

    @staticmethod
    def _note_verification_warnings(
        *,
        after: dict[str, Any],
        conversation_id: int,
        thread_id: int,
        normalized_body: str,
        conversation_status: str,
    ) -> list[str]:
        """Check the note landed and the conversation status did not move."""

        try:
            if ConversationGateway._resolved_id(after) != conversation_id:
                return ["verification_mismatch"]
            matches = [
                thread
                for thread in ConversationGateway._raw_threads(after)
                if _safe_id(thread.get("id") or thread.get("thread_id")) == thread_id
            ]
        except Exception:
            return ["verification_failed"]
        warnings_found: list[str] = []
        if not matches:
            warnings_found.append("verification_unconfirmed")
        elif len(matches) != 1 or not (
            ConversationGateway._is_note_thread(matches[0])
            and _same_body(
                matches[0].get("body_html", matches[0].get("body", "")),
                normalized_body,
            )
        ):
            warnings_found.append("verification_mismatch")
        if after.get("status") != conversation_status:
            warnings_found.append("status_changed")
        return warnings_found

    async def add_note(
        self,
        conversation_id: int,
        *,
        body: str,
        body_format: str,
        idempotency_key: str,
        identity: AgentIdentity,
        request_id: str,
    ) -> dict[str, Any]:
        """Add one internal note exactly once without changing the status.

        Help Scout reopens a conversation as Active when a note is posted
        without a status, so the note always carries the status read under
        the conversation lease.
        """

        if (
            isinstance(conversation_id, bool)
            or not isinstance(conversation_id, int)
            or conversation_id <= 0
        ):
            raise ConversationGatewayError(
                "The conversation id must be a positive integer.",
                code="invalid_conversation",
                status=400,
            )
        if type(body) is not str:
            raise ConversationGatewayError(
                "The note body must be a string.",
                code="invalid_body",
                status=400,
            )
        if type(body_format) is not str:
            raise ConversationGatewayError(
                "body_format must be html or text.",
                code="invalid_body_format",
                status=400,
            )
        try:
            body_bytes = body.encode("utf-8", errors="strict")
            body_format.encode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ConversationGatewayError(
                "The note contains invalid Unicode.",
                code="invalid_body",
                status=400,
            ) from error
        if len(body_bytes) > _MAX_DRAFT_BODY_BYTES:
            raise ConversationGatewayError(
                "The note body is too large.",
                code="body_too_large",
                status=413,
            )
        request = {"body": body, "body_format": body_format}
        try:
            canonical_request_json(request)
        except ValueError as error:
            raise ConversationGatewayError(
                "The note request is not valid JSON.",
                code="invalid_body",
                status=400,
            ) from error

        self._require_requested_allowed(identity, conversation_id)
        reservation = await self._operation_reserve(
            idempotency_key=idempotency_key,
            identity=identity,
            scope="note",
            operation_name="note.add",
            target_id=conversation_id,
            request_id=request_id,
            request=request,
            request_metadata={
                "body_format": body_format,
                "conversation_id": conversation_id,
            },
        )
        if reservation.replayed:
            if reservation.operation.state in {"succeeded", "sent_with_warning"}:
                replayed = self._note_result_from_record(reservation.operation)
                await self._emit_operation_event(
                    operation=reservation.operation,
                    request_id=request_id,
                    event_type="supa.note.added",
                    conversation_id=replayed["conversation_id"],
                    thread_id=replayed["thread_id"],
                    result="added",
                    status=replayed["conversation_status"],
                )
                return replayed
            self._raise_replayed_operation(reservation.operation)

        owner = hashlib.sha256(
            f"{identity.token_id}\0{idempotency_key}\0{request_id}".encode("utf-8")
        ).hexdigest()
        lock_ids = (conversation_id,)
        for _ in range(_MAX_RESOLUTION_ATTEMPTS):
            attempt_state: dict[str, Any] = {
                "callback_started": False,
                "durable_success": False,
            }
            async with self._draft_lease_scope(
                lock_ids,
                owner=owner,
                idempotency_key=idempotency_key,
                identity=identity,
                request_id=request_id,
                attempt_state=attempt_state,
                scope="note",
            ) as leases:
                before = await self._read_write_preflight(
                    conversation_id,
                    idempotency_key=idempotency_key,
                    identity=identity,
                    request_id=request_id,
                    scope="note",
                )
                resolved_id = self._resolved_id(before)
                await self._require_resolved_allowed(
                    identity=identity,
                    resolved_id=resolved_id,
                    idempotency_key=idempotency_key,
                    request_id=request_id,
                    scope="note",
                )
                required_lock_ids = tuple(sorted({conversation_id, resolved_id}))
                if required_lock_ids != lock_ids:
                    lock_ids = required_lock_ids
                    continue

                conversation_status = before.get("status")
                if conversation_status not in _NOTE_CONVERSATION_STATUSES:
                    # Never post without a status: Help Scout would reopen it.
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="note",
                        code="invalid_conversation",
                        status=502,
                        retryable=False,
                        outcome_uncertain=False,
                    )
                    raise ConversationGatewayError(
                        "Help Scout returned no usable conversation status.",
                        code="invalid_conversation",
                        status=502,
                    )
                try:
                    normalized_body = normalize_draft_body(body, body_format)
                except ConversationGatewayError as error:
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="note",
                        code=error.code,
                        status=error.status,
                        retryable=error.retryable,
                        outcome_uncertain=False,
                    )
                    raise
                normalized_body_hash = hashlib.sha256(
                    normalized_body.encode("utf-8")
                ).hexdigest()

                marker_started = False
                try:
                    await self._verify_lease_set(leases)
                    await self._operation_begin(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        scope="note",
                        request_id=request_id,
                    )
                    marker_started = True
                    started = await self._operation_mark_upstream(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        scope="note",
                        request_id=request_id,
                        attempt_metadata={
                            "action": "added",
                            "conversation_id": resolved_id,
                            "conversation_status": conversation_status,
                            "normalized_body_hash": normalized_body_hash,
                        },
                    )
                except (asyncio.CancelledError, ConversationLockError):
                    raise
                except Exception as error:
                    self._safe_log_exception(
                        "note_marker",
                        error,
                        conversation_id=resolved_id,
                        request_id=request_id,
                    )
                    try:
                        await self._terminalize_active_draft(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="note",
                            code=(
                                "operation_marker_failed"
                                if marker_started
                                else "operation_begin_failed"
                            ),
                            status=500,
                            retryable=False,
                            outcome_uncertain=False,
                        )
                    except Exception:
                        pass
                    raise

                async def submit_note() -> Any:
                    attempt_state["callback_started"] = True
                    return await self.client.add_note_result(
                        resolved_id,
                        # The mailbox records which agent wrote the note.
                        recorded(normalized_body, {"actor": identity.name}),
                        status=conversation_status,
                    )

                try:
                    mutation = await self._run_with_lease_guards(leases, submit_note)
                except asyncio.CancelledError:
                    await self._durable_thread_call(
                        finish_operation,
                        db_path=self.db_path,
                        idempotency_key=idempotency_key,
                        identity=identity,
                        scope="note",
                        request_id=request_id,
                        outcome=(
                            "outcome_unknown"
                            if attempt_state["callback_started"]
                            else "failed"
                        ),
                        error={
                            "code": (
                                "cancelled_after_submission"
                                if attempt_state["callback_started"]
                                else "cancelled_before_attempt"
                            ),
                            "retryable": False,
                            "status": 499,
                        },
                    )
                    raise
                except ConversationLockError:
                    if not attempt_state["callback_started"]:
                        await self._terminalize_active_draft(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="note",
                            code="conversation_lock_lost",
                            status=409,
                            retryable=False,
                            outcome_uncertain=False,
                        )
                    raise
                except Exception as error:
                    self._safe_log_exception(
                        "note_submission",
                        error,
                        conversation_id=resolved_id,
                        request_id=request_id,
                    )
                    mutation = None

                if not isinstance(mutation, MutationResult):
                    mutation = None
                if mutation is not None and not mutation.ok and not mutation.outcome_unknown:
                    code, status, retryable = self._safe_mutation_error(
                        mutation.status_code,
                        mutation.detail,
                    )
                    await self._finish_draft_failure(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="note",
                        code=code,
                        status=status,
                        retryable=retryable,
                    )
                    raise ConversationGatewayError(
                        "Help Scout rejected the note.",
                        code=code,
                        retryable=retryable,
                        status=status,
                    )

                confirmed_thread_id = self._authoritative_resource_id(mutation)
                warnings_found: list[str] = []
                try:
                    after = await self.client.get_complete_conversation(resolved_id)
                except asyncio.CancelledError:
                    if confirmed_thread_id is None:
                        await self._finish_draft_unknown(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="note",
                            code="cancelled_during_verification",
                        )
                    else:
                        response = {
                            "action": "added",
                            "body_sha256": normalized_body_hash,
                            "conversation_id": resolved_id,
                            "conversation_status": conversation_status,
                            "status": "added_with_warning",
                            "thread_id": confirmed_thread_id,
                            "warnings": ["verification_cancelled"],
                        }
                        completed = await self._operation_finish(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            scope="note",
                            request_id=request_id,
                            outcome="sent_with_warning",
                            upstream_thread_id=confirmed_thread_id,
                            response=response,
                        )
                        attempt_state["durable_success"] = True
                        try:
                            await self._emit_operation_event(
                                operation=completed,
                                request_id=request_id,
                                event_type="supa.note.added",
                                conversation_id=resolved_id,
                                thread_id=confirmed_thread_id,
                                result="added",
                                status=conversation_status,
                            )
                        except ConversationGatewayError:
                            # Cancellation stays authoritative; replay repairs the event.
                            pass
                    raise
                except Exception as error:
                    self._safe_log_exception(
                        "note_verification",
                        error,
                        conversation_id=resolved_id,
                        request_id=request_id,
                    )
                    after = None
                    if confirmed_thread_id is not None:
                        warnings_found = ["verification_failed"]

                if confirmed_thread_id is None and isinstance(after, dict):
                    try:
                        if self._resolved_id(after) == resolved_id:
                            candidates = self._new_note_candidates(
                                before=before,
                                after=after,
                                normalized_body=normalized_body,
                                attempted_at=started.upstream_started_at,
                            )
                            if len(candidates) == 1:
                                confirmed_thread_id = candidates[0]
                    except Exception as error:
                        self._safe_log_exception(
                            "note_reconciliation",
                            error,
                            conversation_id=resolved_id,
                            request_id=request_id,
                        )
                if confirmed_thread_id is None:
                    await self._finish_draft_unknown(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="note",
                        code="upstream_outcome_unknown",
                    )
                    raise ConversationGatewayError(
                        "The note may have reached Help Scout.",
                        code="outcome_unknown",
                        retryable=False,
                        status=409,
                    )
                if isinstance(after, dict):
                    warnings_found = self._note_verification_warnings(
                        after=after,
                        conversation_id=resolved_id,
                        thread_id=confirmed_thread_id,
                        normalized_body=normalized_body,
                        conversation_status=conversation_status,
                    )

                response = {
                    "action": "added",
                    "body_sha256": normalized_body_hash,
                    "conversation_id": resolved_id,
                    "conversation_status": conversation_status,
                    "status": "added_with_warning" if warnings_found else "added",
                    "thread_id": confirmed_thread_id,
                    "warnings": warnings_found,
                }
                completed = await self._operation_finish(
                    idempotency_key=idempotency_key,
                    identity=identity,
                    scope="note",
                    request_id=request_id,
                    outcome="sent_with_warning" if warnings_found else "succeeded",
                    upstream_thread_id=confirmed_thread_id,
                    response=response,
                )
                attempt_state["durable_success"] = True
                await self._emit_operation_event(
                    operation=completed,
                    request_id=request_id,
                    event_type="supa.note.added",
                    conversation_id=resolved_id,
                    thread_id=confirmed_thread_id,
                    result="added",
                    status=conversation_status,
                )
                return response

        await self._terminalize_active_draft(
            idempotency_key=idempotency_key,
            identity=identity,
            request_id=request_id,
            scope="note",
            code="conversation_resolution_changed",
            status=409,
            retryable=True,
            outcome_uncertain=False,
        )
        raise ConversationGatewayError(
            "The resolved conversation changed while acquiring its write lock.",
            code="conversation_resolution_changed",
            retryable=True,
            status=409,
        )

    @staticmethod
    def _note_update_result_from_record(record: OperationRecord) -> dict[str, Any]:
        response = record.response
        if not isinstance(response, dict):
            raise ConversationGatewayError(
                "The note operation result is invalid.",
                code="invalid_operation_result",
                retryable=False,
                status=500,
            )
        conversation_id = _safe_id(response.get("conversation_id"))
        thread_id = _safe_id(response.get("thread_id"))
        body_sha256 = response.get("body_sha256")
        conversation_status = response.get("conversation_status")
        status = response.get("status")
        version = response.get("version")
        warnings_value = response.get("warnings")
        valid_warnings = isinstance(warnings_value, list) and all(
            isinstance(item, str) and _SAFE_ERROR_CODE_RE.fullmatch(item) is not None
            for item in warnings_value
        )
        if (
            response.get("action") != "updated"
            or not isinstance(conversation_id, int)
            or not isinstance(thread_id, int)
            or not isinstance(body_sha256, str)
            or _DRAFT_VERSION_RE.fullmatch(body_sha256) is None
            or conversation_status not in _NOTE_CONVERSATION_STATUSES
            or status not in {"updated", "updated_with_warning"}
            or not (
                version is None
                or (isinstance(version, str) and _DRAFT_VERSION_RE.fullmatch(version))
            )
            or (version is None and status == "updated")
            or not valid_warnings
        ):
            raise ConversationGatewayError(
                "The note operation result is invalid.",
                code="invalid_operation_result",
                retryable=False,
                status=500,
            )
        return {
            "action": "updated",
            "body_sha256": body_sha256,
            "conversation_id": conversation_id,
            "conversation_status": conversation_status,
            "status": status,
            "thread_id": thread_id,
            "version": version,
            "warnings": list(warnings_value),
        }

    @staticmethod
    def _editable_note_error(
        raw: dict[str, Any],
        *,
        thread_id: int,
        expected_version: str,
    ) -> ConversationGatewayError | None:
        """Refuse anything but the expected version of a note Supa or an agent wrote.

        Notes a person typed in Help Scout (source ``web``, ``email``...) are never
        edited: only ``source.type == "api"`` notes are Supa's or an agent's own.
        """

        matches = [
            thread
            for thread in ConversationGateway._raw_threads(raw)
            if _safe_id(thread.get("id") or thread.get("thread_id")) == thread_id
        ]
        if len(matches) != 1 or not ConversationGateway._is_note_thread(matches[0]):
            return ConversationGatewayError(
                "The conversation has no internal note with that thread id.",
                code="note_not_found",
                retryable=False,
                status=404,
            )
        note = matches[0]
        if _normalize_source(note.get("source"))["type"] != "api":
            return ConversationGatewayError(
                "Only notes written through the API can be edited.",
                code="note_not_editable",
                retryable=False,
                status=409,
            )
        current_version = note_version(note)
        if current_version != expected_version:
            return ConversationGatewayError(
                "The note no longer matches the expected version.",
                code="note_conflict",
                retryable=False,
                status=409,
                details={"current_version": current_version, "thread_id": thread_id},
            )
        return None

    @staticmethod
    def _updated_note_version(
        after: dict[str, Any],
        *,
        conversation_id: int,
        thread_id: int,
        normalized_body: str,
    ) -> str | None:
        """The edited note's new version when the readback shows the intended body."""

        try:
            if ConversationGateway._resolved_id(after) != conversation_id:
                return None
            matches = [
                thread
                for thread in ConversationGateway._raw_threads(after)
                if _safe_id(thread.get("id") or thread.get("thread_id")) == thread_id
            ]
        except Exception:
            return None
        if (
            len(matches) != 1
            or not ConversationGateway._is_note_thread(matches[0])
            or not _same_body(
                matches[0].get("body_html", matches[0].get("body", "")),
                normalized_body,
            )
        ):
            return None
        return note_version(matches[0])

    async def update_note(
        self,
        conversation_id: int,
        *,
        thread_id: int,
        body: str,
        body_format: str,
        expected_version: str,
        idempotency_key: str,
        identity: AgentIdentity,
        request_id: str,
    ) -> dict[str, Any]:
        """Replace the text of one API-written internal note exactly once.

        Only notes with source type ``api`` (Supa's or an agent's) are editable,
        and only at ``expected_version``. Editing a thread never changes the
        conversation status; the readback checks that it did not.
        """

        for value, code, label in (
            (conversation_id, "invalid_conversation", "conversation"),
            (thread_id, "invalid_thread_id", "thread"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConversationGatewayError(
                    f"The {label} id must be a positive integer.",
                    code=code,
                    status=400,
                )
        if type(body) is not str:
            raise ConversationGatewayError(
                "The note body must be a string.",
                code="invalid_body",
                status=400,
            )
        if type(body_format) is not str:
            raise ConversationGatewayError(
                "body_format must be html or text.",
                code="invalid_body_format",
                status=400,
            )
        if (
            type(expected_version) is not str
            or _DRAFT_VERSION_RE.fullmatch(expected_version) is None
        ):
            raise ConversationGatewayError(
                "The expected note version is invalid.",
                code="invalid_note_version",
                status=400,
            )
        try:
            body_bytes = body.encode("utf-8", errors="strict")
            body_format.encode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ConversationGatewayError(
                "The note contains invalid Unicode.",
                code="invalid_body",
                status=400,
            ) from error
        if len(body_bytes) > _MAX_DRAFT_BODY_BYTES:
            raise ConversationGatewayError(
                "The note body is too large.",
                code="body_too_large",
                status=413,
            )
        request = {
            "body": body,
            "body_format": body_format,
            "expected_version": expected_version,
            "thread_id": thread_id,
        }
        try:
            canonical_request_json(request)
        except ValueError as error:
            raise ConversationGatewayError(
                "The note request is not valid JSON.",
                code="invalid_body",
                status=400,
            ) from error

        self._require_requested_allowed(identity, conversation_id)
        reservation = await self._operation_reserve(
            idempotency_key=idempotency_key,
            identity=identity,
            scope="note",
            operation_name="note.update",
            target_id=conversation_id,
            request_id=request_id,
            request=request,
            request_metadata={
                "body_format": body_format,
                "conversation_id": conversation_id,
                "expected_version": expected_version,
                "thread_id": thread_id,
            },
        )
        if reservation.replayed:
            if reservation.operation.state in {"succeeded", "sent_with_warning"}:
                replayed = self._note_update_result_from_record(reservation.operation)
                await self._emit_operation_event(
                    operation=reservation.operation,
                    request_id=request_id,
                    event_type="supa.note.updated",
                    conversation_id=replayed["conversation_id"],
                    thread_id=replayed["thread_id"],
                    result="updated",
                    status=replayed["conversation_status"],
                )
                return replayed
            self._raise_replayed_operation(reservation.operation)

        owner = hashlib.sha256(
            f"{identity.token_id}\0{idempotency_key}\0{request_id}".encode("utf-8")
        ).hexdigest()
        lock_ids = (conversation_id,)
        for _ in range(_MAX_RESOLUTION_ATTEMPTS):
            attempt_state: dict[str, Any] = {
                "callback_started": False,
                "durable_success": False,
            }
            async with self._draft_lease_scope(
                lock_ids,
                owner=owner,
                idempotency_key=idempotency_key,
                identity=identity,
                request_id=request_id,
                attempt_state=attempt_state,
                scope="note",
            ) as leases:
                before = await self._read_write_preflight(
                    conversation_id,
                    idempotency_key=idempotency_key,
                    identity=identity,
                    request_id=request_id,
                    scope="note",
                )
                resolved_id = self._resolved_id(before)
                await self._require_resolved_allowed(
                    identity=identity,
                    resolved_id=resolved_id,
                    idempotency_key=idempotency_key,
                    request_id=request_id,
                    scope="note",
                )
                required_lock_ids = tuple(sorted({conversation_id, resolved_id}))
                if required_lock_ids != lock_ids:
                    lock_ids = required_lock_ids
                    continue

                conversation_status = before.get("status")
                if conversation_status not in _NOTE_CONVERSATION_STATUSES:
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="note",
                        code="invalid_conversation",
                        status=502,
                        retryable=False,
                        outcome_uncertain=False,
                    )
                    raise ConversationGatewayError(
                        "Help Scout returned no usable conversation status.",
                        code="invalid_conversation",
                        status=502,
                    )
                refusal = self._editable_note_error(
                    before,
                    thread_id=thread_id,
                    expected_version=expected_version,
                )
                if refusal is not None:
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="note",
                        code=refusal.code,
                        status=refusal.status,
                        retryable=refusal.retryable,
                        outcome_uncertain=False,
                        details=refusal.details,
                    )
                    raise refusal
                try:
                    normalized_body = normalize_draft_body(body, body_format)
                except ConversationGatewayError as error:
                    await self._terminalize_active_draft(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="note",
                        code=error.code,
                        status=error.status,
                        retryable=error.retryable,
                        outcome_uncertain=False,
                    )
                    raise
                normalized_body_hash = hashlib.sha256(
                    normalized_body.encode("utf-8")
                ).hexdigest()

                marker_started = False
                try:
                    await self._verify_lease_set(leases)
                    await self._operation_begin(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        scope="note",
                        request_id=request_id,
                    )
                    marker_started = True
                    await self._operation_mark_upstream(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        scope="note",
                        request_id=request_id,
                        attempt_metadata={
                            "action": "updated",
                            "conversation_id": resolved_id,
                            "conversation_status": conversation_status,
                            "expected_thread_id": thread_id,
                            "normalized_body_hash": normalized_body_hash,
                        },
                    )
                except (asyncio.CancelledError, ConversationLockError):
                    raise
                except Exception as error:
                    self._safe_log_exception(
                        "note_update_marker",
                        error,
                        conversation_id=resolved_id,
                        request_id=request_id,
                    )
                    try:
                        await self._terminalize_active_draft(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="note",
                            code=(
                                "operation_marker_failed"
                                if marker_started
                                else "operation_begin_failed"
                            ),
                            status=500,
                            retryable=False,
                            outcome_uncertain=False,
                        )
                    except Exception:
                        pass
                    raise

                async def submit_note_update() -> Any:
                    attempt_state["callback_started"] = True
                    return await self.client.update_note_result(
                        resolved_id,
                        thread_id,
                        normalized_body,
                    )

                try:
                    mutation = await self._run_with_lease_guards(
                        leases, submit_note_update
                    )
                except asyncio.CancelledError:
                    await self._durable_thread_call(
                        finish_operation,
                        db_path=self.db_path,
                        idempotency_key=idempotency_key,
                        identity=identity,
                        scope="note",
                        request_id=request_id,
                        outcome=(
                            "outcome_unknown"
                            if attempt_state["callback_started"]
                            else "failed"
                        ),
                        error={
                            "code": (
                                "cancelled_after_submission"
                                if attempt_state["callback_started"]
                                else "cancelled_before_attempt"
                            ),
                            "retryable": False,
                            "status": 499,
                        },
                    )
                    raise
                except ConversationLockError:
                    if not attempt_state["callback_started"]:
                        await self._terminalize_active_draft(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="note",
                            code="conversation_lock_lost",
                            status=409,
                            retryable=False,
                            outcome_uncertain=False,
                        )
                    raise
                except Exception as error:
                    self._safe_log_exception(
                        "note_update_submission",
                        error,
                        conversation_id=resolved_id,
                        request_id=request_id,
                    )
                    mutation = None

                if not isinstance(mutation, MutationResult):
                    mutation = None
                if mutation is not None and not mutation.ok and not mutation.outcome_unknown:
                    code, status, retryable = self._safe_mutation_error(
                        mutation.status_code,
                        mutation.detail,
                    )
                    await self._finish_draft_failure(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="note",
                        code=code,
                        status=status,
                        retryable=retryable,
                    )
                    raise ConversationGatewayError(
                        "Help Scout rejected the note edit.",
                        code=code,
                        retryable=retryable,
                        status=status,
                    )

                # A 204 confirms the edit; anything else must be confirmed by the readback.
                confirmed = mutation is not None and mutation.ok
                warnings_found: list[str] = []
                try:
                    after = await self.client.get_complete_conversation(resolved_id)
                except asyncio.CancelledError:
                    if not confirmed:
                        await self._finish_draft_unknown(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            request_id=request_id,
                            scope="note",
                            code="cancelled_during_verification",
                        )
                    else:
                        response = {
                            "action": "updated",
                            "body_sha256": normalized_body_hash,
                            "conversation_id": resolved_id,
                            "conversation_status": conversation_status,
                            "status": "updated_with_warning",
                            "thread_id": thread_id,
                            "version": None,
                            "warnings": ["verification_cancelled"],
                        }
                        completed = await self._operation_finish(
                            idempotency_key=idempotency_key,
                            identity=identity,
                            scope="note",
                            request_id=request_id,
                            outcome="sent_with_warning",
                            upstream_thread_id=thread_id,
                            response=response,
                        )
                        attempt_state["durable_success"] = True
                        try:
                            await self._emit_operation_event(
                                operation=completed,
                                request_id=request_id,
                                event_type="supa.note.updated",
                                conversation_id=resolved_id,
                                thread_id=thread_id,
                                result="updated",
                                status=conversation_status,
                            )
                        except ConversationGatewayError:
                            # Cancellation stays authoritative; replay repairs the event.
                            pass
                    raise
                except Exception as error:
                    self._safe_log_exception(
                        "note_update_verification",
                        error,
                        conversation_id=resolved_id,
                        request_id=request_id,
                    )
                    after = None
                    if confirmed:
                        warnings_found = ["verification_failed"]

                new_version = (
                    self._updated_note_version(
                        after,
                        conversation_id=resolved_id,
                        thread_id=thread_id,
                        normalized_body=normalized_body,
                    )
                    if isinstance(after, dict)
                    else None
                )
                if not confirmed and new_version is None:
                    await self._finish_draft_unknown(
                        idempotency_key=idempotency_key,
                        identity=identity,
                        request_id=request_id,
                        scope="note",
                        code="upstream_outcome_unknown",
                    )
                    raise ConversationGatewayError(
                        "The note edit may have reached Help Scout.",
                        code="outcome_unknown",
                        retryable=False,
                        status=409,
                    )
                if isinstance(after, dict):
                    warnings_found = self._note_verification_warnings(
                        after=after,
                        conversation_id=resolved_id,
                        thread_id=thread_id,
                        normalized_body=normalized_body,
                        conversation_status=conversation_status,
                    )

                response = {
                    "action": "updated",
                    "body_sha256": normalized_body_hash,
                    "conversation_id": resolved_id,
                    "conversation_status": conversation_status,
                    "status": "updated_with_warning" if warnings_found else "updated",
                    "thread_id": thread_id,
                    "version": new_version,
                    "warnings": warnings_found,
                }
                completed = await self._operation_finish(
                    idempotency_key=idempotency_key,
                    identity=identity,
                    scope="note",
                    request_id=request_id,
                    outcome="sent_with_warning" if warnings_found else "succeeded",
                    upstream_thread_id=thread_id,
                    response=response,
                )
                attempt_state["durable_success"] = True
                await self._emit_operation_event(
                    operation=completed,
                    request_id=request_id,
                    event_type="supa.note.updated",
                    conversation_id=resolved_id,
                    thread_id=thread_id,
                    result="updated",
                    status=conversation_status,
                )
                return response

        await self._terminalize_active_draft(
            idempotency_key=idempotency_key,
            identity=identity,
            request_id=request_id,
            scope="note",
            code="conversation_resolution_changed",
            status=409,
            retryable=True,
            outcome_uncertain=False,
        )
        raise ConversationGatewayError(
            "The resolved conversation changed while acquiring its write lock.",
            code="conversation_resolution_changed",
            retryable=True,
            status=409,
        )

    async def get(self, conversation_id: int) -> dict[str, Any]:
        raw = await self.client.get_complete_conversation(conversation_id)
        result = normalize_conversation(raw)
        from services.support_recovery import recovery_metadata
        from db.database import get_ticket_log_by_helpscout_id
        row = get_ticket_log_by_helpscout_id(self.db_path, str(raw.get("id")))
        if row:
            result["support_recovery"] = recovery_metadata(raw, row)
        return result

    async def list(
        self,
        *,
        status: str,
        tag: str | None,
        query: str | None,
        cursor: str | None,
        limit: int,
    ) -> dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ConversationGatewayError(
                "The conversation list limit must be between 1 and 100.",
                code="invalid_limit",
            )
        page, offset = _decode_cursor(cursor)
        remaining = limit
        conversations: list[dict[str, Any]] = []
        next_cursor: str | None = None
        # A 100-item request beginning at offset 24 spans at most five fixed pages.
        max_calls = (offset + limit + 24) // 25

        for _ in range(max_calls):
            result = await self.client.list_conversations(
                query=query,
                status=status,
                tag=tag,
                page=page,
                embed="threads",
            )
            raw_conversations = result.get("conversations")
            if not isinstance(raw_conversations, list):
                raw_conversations = []
            pages = result.get("pages")
            has_next_page = (
                isinstance(pages, int)
                and not isinstance(pages, bool)
                and page < pages
                and page < _MAX_CURSOR_PAGE
            )

            if offset and offset >= len(raw_conversations):
                raise InvalidCursorError()
            if (
                len(raw_conversations) > 25
                or any(not isinstance(raw, dict) for raw in raw_conversations)
                or (has_next_page and len(raw_conversations) != 25)
            ):
                raise ConversationGatewayError(
                    "Help Scout returned an invalid fixed conversation page.",
                    code="invalid_upstream_pagination",
                    retryable=False,
                    status=502,
                )

            available = raw_conversations[offset:]
            take = min(remaining, len(available))
            conversations.extend(
                normalize_conversation(raw)
                for raw in available[:take]
                if isinstance(raw, dict)
            )
            remaining -= take
            consumed_offset = offset + take

            if take < len(available):
                next_cursor = _encode_cursor(page, consumed_offset)
                break
            if not has_next_page:
                break
            if remaining == 0:
                next_cursor = _encode_cursor(page + 1)
                break

            page += 1
            offset = 0

        return {"conversations": conversations, "next_cursor": next_cursor}

    async def attachment(
        self,
        conversation_id: int,
        attachment_id: int,
        *,
        identity: AgentIdentity | None = None,
    ) -> AttachmentData:
        self._require_requested_allowed(identity, conversation_id)
        raw = await self.client.get_complete_conversation(conversation_id)
        resolved_id = _safe_id(raw.get("id"))
        if not isinstance(resolved_id, int):
            raise ConversationGatewayError(
                "The resolved conversation id is invalid.",
                code="invalid_conversation",
                status=502,
                retryable=False,
            )
        # A merged id may resolve to a conversation outside the allowlist.
        self._require_requested_allowed(identity, resolved_id)

        metadata: dict[str, Any] | None = None
        embedded = raw.get("_embedded")
        threads = embedded.get("threads") if isinstance(embedded, dict) else []
        if isinstance(threads, list):
            for thread in threads:
                if not isinstance(thread, dict):
                    continue
                thread_embedded = thread.get("_embedded")
                attachments = (
                    thread_embedded.get("attachments")
                    if isinstance(thread_embedded, dict)
                    else []
                )
                if not isinstance(attachments, list):
                    continue
                for attachment in attachments:
                    if (
                        isinstance(attachment, dict)
                        and _safe_id(attachment.get("id")) == attachment_id
                    ):
                        metadata = attachment
                        break
                if metadata is not None:
                    break

        if metadata is None:
            raise ConversationGatewayError(
                "The attachment was not found in this conversation.",
                code="attachment_not_found",
                status=404,
                retryable=False,
            )

        data, downloaded_mime = await self.client.get_attachment_data(
            resolved_id,
            attachment_id,
            max_bytes=self.attachment_max_bytes,
        )
        filename = metadata.get("filename")
        if not isinstance(filename, str) or not filename:
            filename = f"attachment-{attachment_id}"
        claimed_mime_type = downloaded_mime or _normalize_mime(metadata.get("mimeType")) or None
        return AttachmentData(
            conversation_id=resolved_id,
            attachment_id=attachment_id,
            filename=filename,
            data=data,
            claimed_mime_type=claimed_mime_type,
            size=len(data),
        )
