import base64
import binascii
import hashlib
import uuid
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode, urlsplit

import httpx
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from services import installation

from services import mailbox_provider
from services.themes import detect_theme
from services.email_html import ensure_email_paragraph_spacing


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MutationResult:
    ok: bool
    status_code: int | None
    resource_id: int | None = None
    outcome_unknown: bool = False
    detail: str | None = None


class RecordedNote(str):
    """A note's text together with the record it stands for.

    The owned mailbox keeps what agents did as typed records. A writer that knows its facts
    (a reminder's step and score, a responder's reasons) states them here, so the mailbox does
    not have to read them back out of the wording. It is still a string: code and tests that
    treat a note as text keep working, and the vendor mailbox only ever receives the text.
    """

    event: dict | None = None


def recorded(text: str, event: dict | None) -> "RecordedNote":
    note = RecordedNote(text)
    note.event = event
    return note


@dataclass(frozen=True)
class HelpScoutWebhook:
    """Validated webhook metadata returned by the Help Scout Mailbox API."""

    id: int
    url: str
    state: str
    events: tuple[str, ...]
    notification: bool
    payload_version: str
    label: str | None
    mailbox_ids: tuple[int, ...]


class _MutationPreSubmissionError(Exception):
    """Safe internal marker for OAuth failure before a mutation was attempted."""

    def __init__(self, *, status_code: int | None, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class HelpScoutError(Exception):
    """Base error for structured Help Scout read failures."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str = "helpscout_error",
        retryable: bool = False,
        details: Dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable
        self.details = dict(details or {})


class HelpScoutUpstreamError(HelpScoutError):
    """Help Scout returned a non-success HTTP response."""


class HelpScoutProtocolError(HelpScoutError):
    """Help Scout returned an invalid or unsafe response."""


class HelpScoutRedirectError(HelpScoutProtocolError):
    """A Help Scout merge redirect could not be followed safely."""


class HelpScoutAttachmentTooLargeError(HelpScoutError):
    """An attachment exceeded the caller's explicit byte limit."""


class HelpScoutAuthError(HelpScoutError):
    """Help Scout OAuth failed for a read request."""


class HelpScoutTransportError(HelpScoutError):
    """A read request could not reach Help Scout reliably."""


class HelpScoutResponseTooLargeError(HelpScoutProtocolError):
    """A Help Scout response exceeded its bounded read limit."""


_MAX_CONVERSATION_THREADS = 1000
_MAX_CONVERSATION_THREAD_PAGES = 100
_MAX_CONVERSATION_LIST_PAGES = 1_000_000
_HELPSCOUT_CONVERSATION_PAGE_SIZE = 25
_MAX_SAFE_THREAD_ID = (1 << 63) - 1
_READ_JSON_MAX_BYTES = 8 * 1024 * 1024
# Attachment data is base64 inside a small JSON envelope. This allowance covers
# JSON syntax plus optional claimed metadata without making the envelope unbounded.
_ATTACHMENT_JSON_OVERHEAD_BYTES = 4096
_ATTACHMENT_MIME_RE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
_WEBHOOK_EVENT_RE = re.compile(r"^[a-z][a-z0-9.-]{0,99}$")
_WEBHOOK_SECRET_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_MAX_WEBHOOK_LIST_PAGES = 1_000
_MAX_WEBHOOK_RESPONSE_BYTES = 2 * 1024 * 1024


def _mutation_failure_detail(status_code: int) -> str:
    if status_code == 301:
        return "upstream_redirect"
    if status_code in (401, 403):
        return "upstream_auth_error"
    if status_code == 404:
        return "upstream_not_found"
    if status_code == 412:
        return "conversation_locked"
    if status_code == 429:
        return "upstream_rate_limited"
    if status_code >= 500:
        return "upstream_server_error"
    return "upstream_http_error"


def _resource_id_from_response(response: Any) -> tuple[int | None, str | None]:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None, "resource_id_missing"
    try:
        entries = headers.items()
    except AttributeError:
        return None, "resource_id_missing"

    raw_value: Any = None
    found = False
    for raw_name, candidate in entries:
        if isinstance(raw_name, str) and raw_name.casefold() == "resource-id":
            raw_value = candidate
            found = True
            break
    if not found:
        return None, "resource_id_missing"
    if (
        not isinstance(raw_value, str)
        or re.fullmatch(r"[0-9]{1,19}", raw_value) is None
    ):
        return None, "resource_id_invalid"
    resource_id = int(raw_value)
    if not 0 < resource_id <= _MAX_SAFE_THREAD_ID:
        return None, "resource_id_invalid"
    return resource_id, None


def _with_mutation_detail(result: MutationResult, detail: str) -> MutationResult:
    combined = detail if result.detail is None else f"{result.detail}+{detail}"
    return MutationResult(
        ok=result.ok,
        status_code=result.status_code,
        resource_id=result.resource_id,
        outcome_unknown=result.outcome_unknown,
        detail=combined,
    )


class _BoundedReadResponse:
    def __init__(self, status_code: int, headers: httpx.Headers, content: bytes):
        self.status_code = status_code
        self.headers = headers
        self.content = content

    def json(self) -> Any:
        return json.loads(self.content)


def _raise_for_read_status(response: Any, endpoint: str) -> None:
    if 200 <= response.status_code < 300:
        return
    status = response.status_code
    if status in (401, 403):
        raise HelpScoutAuthError(
            f"Help Scout authorization failed with status {status}",
            status=status,
            code="upstream_auth_error",
            retryable=False,
            details={"endpoint": endpoint},
        )
    details = {"endpoint": endpoint}
    raw_delay = (getattr(response, "headers", None) or {}).get("Retry-After", "")
    if status == 429 and isinstance(raw_delay, str) and raw_delay.isascii() and raw_delay.isdecimal() and len(raw_delay) <= 3:
        delay = int(raw_delay)
        if 0 <= delay <= 120:
            details["retry_after_seconds"] = delay
    raise HelpScoutUpstreamError(
        f"Help Scout request failed with status {status}",
        status=status,
        code="upstream_http_error",
        retryable=status == 429 or status >= 500,
        details=details,
    )


def _response_json_object(response: Any, endpoint: str) -> Dict[str, Any]:
    try:
        payload = response.json()
    except Exception as exc:
        raise HelpScoutProtocolError(
            "Help Scout returned invalid JSON",
            code="invalid_json",
            details={"endpoint": endpoint},
        ) from exc
    if not isinstance(payload, dict):
        raise HelpScoutProtocolError(
            "Help Scout returned an invalid JSON object",
            code="invalid_json",
            details={"endpoint": endpoint},
        )
    return payload


def _parse_merge_location(location: str | None, current_id: int, api_url: str = mailbox_provider.VENDOR_API) -> int:
    if not isinstance(location, str) or not location:
        raise HelpScoutRedirectError(
            "Help Scout merge redirect is missing a valid Location",
            status=301,
            code="invalid_merge_redirect",
            details={"conversation_id": current_id},
        )

    try:
        parsed = urlsplit(location)
        port = parsed.port
    except ValueError as exc:
        raise HelpScoutRedirectError(
            "Help Scout merge redirect has an invalid Location",
            status=301,
            code="invalid_merge_redirect",
            details={"conversation_id": current_id},
        ) from exc

    if parsed.netloc:
        expected = urlsplit(api_url)
        is_official_origin = (
            parsed.scheme == "https"
            and parsed.hostname == expected.hostname
            and parsed.username is None
            and parsed.password is None
            and (443 if port is None else port) == (expected.port or 443)
        )
    else:
        is_official_origin = not parsed.scheme

    match = re.fullmatch(r"/v2/conversations/(\d+)", parsed.path)
    if (
        not is_official_origin
        or parsed.query
        or parsed.fragment
        or match is None
    ):
        raise HelpScoutRedirectError(
            "Help Scout merge redirect has an invalid Location",
            status=301,
            code="invalid_merge_redirect",
            details={"conversation_id": current_id},
        )

    resolved_id = int(match.group(1))
    if resolved_id == current_id:
        raise HelpScoutRedirectError(
            "Help Scout merge redirect points to the same conversation",
            status=301,
            code="invalid_merge_redirect",
            details={"conversation_id": current_id},
        )
    return resolved_id


def _thread_page_parts(
    payload: Dict[str, Any],
    endpoint: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    embedded = payload.get("_embedded")
    if not isinstance(embedded, dict) or not isinstance(embedded.get("threads"), list):
        raise HelpScoutProtocolError(
            "Help Scout returned an invalid thread page",
            code="invalid_json",
            details={"endpoint": endpoint},
        )
    threads = embedded["threads"]
    if any(not isinstance(thread, dict) for thread in threads):
        raise HelpScoutProtocolError(
            "Help Scout returned an invalid thread entry",
            code="invalid_json",
            details={"endpoint": endpoint},
        )

    page = payload.get("page", {})
    if page is None:
        page = {}
    if not isinstance(page, dict):
        raise HelpScoutProtocolError(
            "Help Scout returned invalid pagination metadata",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )
    return threads, page


def _optional_page_integer(
    page: Dict[str, Any],
    field: str,
    endpoint: str,
) -> int | None:
    value = page.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HelpScoutProtocolError(
            "Help Scout returned invalid pagination metadata",
            code="invalid_pagination",
            details={"endpoint": endpoint, "field": field},
        )
    return value


def _first_page_state(
    page: Dict[str, Any],
    threads: List[Dict[str, Any]],
    endpoint: str,
) -> Dict[str, int | None | bool]:
    number = _optional_page_integer(page, "number", endpoint)
    total_pages = _optional_page_integer(page, "totalPages", endpoint)
    size = _optional_page_integer(page, "size", endpoint)
    total_elements = _optional_page_integer(page, "totalElements", endpoint)

    zero_sentinel = number == 0 and total_pages == 0 and total_elements == 0
    any_zero_sentinel_field = number == 0 or total_pages == 0 or total_elements == 0
    if zero_sentinel:
        if threads:
            raise HelpScoutProtocolError(
                "Help Scout returned a nonempty zero-page sentinel",
                code="invalid_pagination",
                details={"endpoint": endpoint},
            )
        if size == 0:
            size = None
        return {
            "total_pages": 0,
            "size": size,
            "total_elements": 0,
            "empty_sentinel": True,
        }
    if any_zero_sentinel_field:
        raise HelpScoutProtocolError(
            "Help Scout returned invalid zero-page metadata",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )

    if number not in (None, 1) or size == 0:
        raise HelpScoutProtocolError(
            "Help Scout returned invalid first-page metadata",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )
    if total_elements is not None and total_elements > _MAX_CONVERSATION_THREADS:
        raise HelpScoutProtocolError(
            "Help Scout reported too many conversation threads",
            code="invalid_pagination",
            details={"endpoint": endpoint, "max_threads": _MAX_CONVERSATION_THREADS},
        )

    if total_pages is None and size is not None and total_elements is not None:
        total_pages = (total_elements + size - 1) // size
    if total_pages is None:
        total_pages = 1
    if total_pages > _MAX_CONVERSATION_THREAD_PAGES:
        raise HelpScoutProtocolError(
            "Help Scout reported too many thread pages",
            code="invalid_pagination",
            details={"endpoint": endpoint, "max_pages": _MAX_CONVERSATION_THREAD_PAGES},
        )
    if total_elements is not None and total_pages > total_elements:
        raise HelpScoutProtocolError(
            "Help Scout returned inconsistent pagination totals",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )
    if size is not None and len(threads) > size:
        raise HelpScoutProtocolError(
            "Help Scout returned more entries than the page size",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )
    if size is not None and total_elements is not None:
        expected_pages = (total_elements + size - 1) // size
        if total_pages != expected_pages:
            raise HelpScoutProtocolError(
                "Help Scout returned inconsistent pagination totals",
                code="invalid_pagination",
                details={"endpoint": endpoint},
            )
    if total_elements is not None and len(threads) > total_elements:
        raise HelpScoutProtocolError(
            "Help Scout returned too many entries for its total",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )
    return {
        "total_pages": total_pages,
        "size": size,
        "total_elements": total_elements,
        "empty_sentinel": False,
    }


def _validate_later_page(
    page: Dict[str, Any],
    threads: List[Dict[str, Any]],
    endpoint: str,
    requested_page: int,
    state: Dict[str, int | None | bool],
) -> None:
    number = _optional_page_integer(page, "number", endpoint)
    if number != requested_page:
        code = (
            "repeated_page"
            if number is not None and 0 < number < requested_page
            else "invalid_pagination"
        )
        raise HelpScoutProtocolError(
            "Help Scout returned an unexpected one-based page number",
            code=code,
            details={"endpoint": endpoint},
        )

    for field, state_key in (
        ("size", "size"),
        ("totalPages", "total_pages"),
        ("totalElements", "total_elements"),
    ):
        value = _optional_page_integer(page, field, endpoint)
        expected = state[state_key]
        if value is not None and expected is not None and value != expected:
            raise HelpScoutProtocolError(
                "Help Scout changed pagination metadata during traversal",
                code="invalid_pagination",
                details={"endpoint": endpoint, "field": field},
            )

    size = _optional_page_integer(page, "size", endpoint)
    if size == 0 or (size is not None and len(threads) > size):
        raise HelpScoutProtocolError(
            "Help Scout returned invalid page size metadata",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )
    total_pages = _optional_page_integer(page, "totalPages", endpoint)
    total_elements = _optional_page_integer(page, "totalElements", endpoint)
    if total_pages is not None and (
        total_pages == 0 or total_pages > _MAX_CONVERSATION_THREAD_PAGES
    ):
        raise HelpScoutProtocolError(
            "Help Scout returned invalid page bounds",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )
    if total_elements is not None and (
        total_elements == 0 or total_elements > _MAX_CONVERSATION_THREADS
    ):
        raise HelpScoutProtocolError(
            "Help Scout returned invalid thread totals",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )
    if size is not None and total_pages is not None and total_elements is not None:
        if total_pages != (total_elements + size - 1) // size:
            raise HelpScoutProtocolError(
                "Help Scout returned inconsistent pagination totals",
                code="invalid_pagination",
                details={"endpoint": endpoint},
            )


def _record_thread_ids(
    threads: List[Dict[str, Any]],
    seen_ids: set[int],
    endpoint: str,
) -> None:
    for thread in threads:
        thread_id = thread.get("id")
        if isinstance(thread_id, int) and not isinstance(thread_id, bool):
            normalized_id = thread_id
        elif (
            isinstance(thread_id, str)
            and len(thread_id) <= 19
            and re.fullmatch(r"[0-9]+", thread_id)
        ):
            normalized_id = int(thread_id)
        else:
            normalized_id = 0
        if not 0 < normalized_id <= _MAX_SAFE_THREAD_ID:
            raise HelpScoutProtocolError(
                "Help Scout returned a thread without a usable id",
                code="invalid_thread_id",
                details={"endpoint": endpoint},
            )
        if normalized_id in seen_ids:
            raise HelpScoutProtocolError(
                "Help Scout repeated a thread id",
                code="duplicate_thread",
                details={"endpoint": endpoint, "thread_id": normalized_id},
            )
        seen_ids.add(normalized_id)


def _conversation_list_parts(
    payload: Dict[str, Any],
    endpoint: str,
    *,
    requested_page: int,
) -> tuple[List[Dict[str, Any]], int]:
    """Validate the bounded Help Scout conversation-list response."""
    embedded = payload.get("_embedded")
    if not isinstance(embedded, dict):
        raise HelpScoutProtocolError(
            "Help Scout returned an invalid conversation list",
            code="invalid_json",
            details={"endpoint": endpoint},
        )
    conversations = embedded.get("conversations")
    if not isinstance(conversations, list) or any(
        not isinstance(conversation, dict) for conversation in conversations
    ):
        raise HelpScoutProtocolError(
            "Help Scout returned an invalid conversation list",
            code="invalid_json",
            details={"endpoint": endpoint},
        )

    seen_ids: set[int] = set()
    for conversation in conversations:
        conversation_id = conversation.get("id")
        if isinstance(conversation_id, int) and not isinstance(conversation_id, bool):
            normalized_id = conversation_id
        elif (
            isinstance(conversation_id, str)
            and len(conversation_id) <= 19
            and re.fullmatch(r"[0-9]+", conversation_id)
        ):
            normalized_id = int(conversation_id)
        else:
            normalized_id = 0
        if not 0 < normalized_id <= _MAX_SAFE_THREAD_ID:
            raise HelpScoutProtocolError(
                "Help Scout returned a conversation without a usable id",
                code="invalid_conversation_id",
                details={"endpoint": endpoint},
            )
        if normalized_id in seen_ids:
            raise HelpScoutProtocolError(
                "Help Scout repeated a conversation id",
                code="duplicate_conversation",
                details={"endpoint": endpoint, "conversation_id": normalized_id},
            )
        seen_ids.add(normalized_id)

    page = payload.get("page")
    if not isinstance(page, dict):
        raise HelpScoutProtocolError(
            "Help Scout returned invalid pagination metadata",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )
    total_pages = _optional_page_integer(page, "totalPages", endpoint)
    if total_pages is None or total_pages > _MAX_CONVERSATION_LIST_PAGES:
        raise HelpScoutProtocolError(
            "Help Scout returned invalid page bounds",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )

    number = _optional_page_integer(page, "number", endpoint)
    if total_pages == 0:
        valid_page_number = number in (None, 0, 1) and requested_page == 1
    else:
        valid_page_number = number is None or number == requested_page
    if not valid_page_number:
        raise HelpScoutProtocolError(
            "Help Scout returned an unexpected page number",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )
    size = _optional_page_integer(page, "size", endpoint)
    if size == 0 or (size is not None and len(conversations) > size):
        raise HelpScoutProtocolError(
            "Help Scout returned invalid page size metadata",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )
    total_elements = _optional_page_integer(page, "totalElements", endpoint)
    if total_elements is not None and len(conversations) > total_elements:
        raise HelpScoutProtocolError(
            "Help Scout returned invalid conversation totals",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )

    if total_pages == 0:
        if conversations or (total_elements not in (None, 0)):
            raise HelpScoutProtocolError(
                "Help Scout returned inconsistent empty pagination metadata",
                code="invalid_pagination",
                details={"endpoint": endpoint},
            )
    elif requested_page <= total_pages and not conversations:
        raise HelpScoutProtocolError(
            "Help Scout returned an empty page inside its reported bounds",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )
    elif requested_page > total_pages:
        raise HelpScoutProtocolError(
            "Help Scout returned page bounds below the requested page",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )

    if len(conversations) > _HELPSCOUT_CONVERSATION_PAGE_SIZE or (
        requested_page < total_pages
        and len(conversations) != _HELPSCOUT_CONVERSATION_PAGE_SIZE
    ):
        raise HelpScoutProtocolError(
            "Help Scout returned an invalid fixed conversation page",
            code="invalid_pagination",
            details={"endpoint": endpoint},
        )

    if size is not None and total_elements is not None:
        expected_pages = (total_elements + size - 1) // size if total_elements else 0
        if expected_pages != total_pages:
            raise HelpScoutProtocolError(
                "Help Scout returned inconsistent pagination totals",
                code="invalid_pagination",
                details={"endpoint": endpoint},
            )
    return conversations, total_pages


def _normalize_claimed_attachment_mime(
    value: Any,
    *,
    response_header: bool = False,
) -> str | None:
    if not isinstance(value, str) or len(value) > 255 or "\r" in value or "\n" in value:
        return None
    mime_type = value.split(";", 1)[0].strip().lower()
    if (
        mime_type == "application/hal+json"
        or (response_header and (mime_type == "application/json" or mime_type.endswith("+json")))
        or not _ATTACHMENT_MIME_RE.fullmatch(mime_type)
    ):
        return None
    return mime_type


def is_supa_ignored(payload: Dict[str, Any]) -> bool:
    """Hold tagged and enrolled manual campaigns across every automation entry point."""
    tagged = any(
        (tag.get("tag") if isinstance(tag, dict) else tag) == "supa-ignore"
        for tag in (payload.get("tags") or [])
    )
    if tagged:
        return True
    from services import extensions
    return extensions.manual_hold(payload)


def strip_html(html: str) -> str:
    """Remove HTML tags and clean up whitespace."""
    if not html:
        return ""
    text = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _is_supa_system_note(body: str) -> bool:
    normalized = strip_html(body).lower()
    return normalized.startswith("supa decision note:") or normalized.startswith("supa auto-improvement note:")


def _is_customer_thread(thread: Dict[str, Any]) -> bool:
    created_by = thread.get("createdBy", {})
    if thread.get("type") == "customer":
        return True
    return isinstance(created_by, dict) and created_by.get("type") == "customer"


def _is_agent_message_thread(thread: Dict[str, Any]) -> bool:
    created_by = thread.get("createdBy", {})
    if not isinstance(created_by, dict):
        return False
    return (
        thread.get("type") == "message"
        and created_by.get("type") in {"user", "system_user"}
    )


def _is_draft_thread(thread: Dict[str, Any]) -> bool:
    return thread.get("state") == "draft" or thread.get("draft") is True


def _is_published_thread(thread: Dict[str, Any]) -> bool:
    """A thread the customer can see: not a draft, not hidden or under review.

    Help Scout keeps unsent drafts (and Supa's blanked leftovers) among the threads as
    ``type: "message"``; anything that reads history must leave them out (2026-10-02).
    """
    return not _is_draft_thread(thread) and thread.get("state") in (None, "published")


def _get_latest_thread(threads: List[Dict[str, Any]], predicate) -> Optional[Dict[str, Any]]:
    matching_threads = [thread for thread in threads if predicate(thread)]
    if not matching_threads:
        return None

    threads_with_timestamps = [thread for thread in matching_threads if thread.get("createdAt")]
    if threads_with_timestamps:
        return max(threads_with_timestamps, key=lambda thread: thread.get("createdAt", ""))

    return matching_threads[0]


def extract_conversation_history(
    threads: List[Dict[str, Any]], max_chars_per_message: int = 12000,
    *, max_total_chars: int = 64000,
) -> List[Dict[str, str]]:
    """Extract conversation history from threads (customer and support messages only).

    Returns published messages in chronological order, with a total budget favouring recent
    messages. The old 500-character limit discarded the actual fix after its introduction,
    causing follow-ups to ask the customer to identify code we had already sent them.
    """
    history = []

    # Embedded collections are not reliably ordered. Missing dates keep the old stable
    # fallback order; dated messages are explicitly chronological before applying a budget.
    ordered = sorted(reversed(threads), key=lambda t:
        created.timestamp() if (created := _parse_thread_created_at(t.get("createdAt"))) else float("-inf"))
    for thread in ordered:
        thread_type = thread.get("type")

        # Only include customer messages and agent replies (not notes, lineitems, etc.)
        if thread_type not in ["customer", "message"]:
            continue
        # An unsent draft or a blanked leftover was never said to the customer.
        if not _is_published_thread(thread):
            continue

        created_by = thread.get("createdBy", {})
        is_customer = created_by.get("type") == "customer" or thread_type == "customer"

        body = strip_html(thread.get("body", ""))
        if not body:
            continue

        # Truncate long messages
        if len(body) > max_chars_per_message:
            body = body[:max_chars_per_message] + "..."

        history.append({
            "role": "customer" if is_customer else "support",
            "message": body
        })

    recent = []
    used = 0
    for message in reversed(history):
        size = len(message["message"])
        if used + size > max_total_chars:
            if not recent and max_total_chars > 0:
                recent.append({**message, "message": message["message"][:max_total_chars]})
            break
        recent.append(message)
        used += size
    return list(reversed(recent))


def get_latest_note_thread(threads: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the newest internal note thread when present.

    HelpScout thread ordering is usually newest-first, but not reliably enough to
    depend on `threads[0]`. Prefer `createdAt` when available and fall back to the
    first note encountered.
    """
    return _get_latest_thread(
        threads,
        lambda thread: thread.get("type") == "note" and not _is_supa_system_note(thread.get("body", "")),
    )


def get_latest_draft_thread(threads: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the newest draft thread when multiple drafts exist."""
    return _get_latest_thread(threads, _is_draft_thread)


def is_draft_superseded_by_reply(
    threads: List[Dict[str, Any]],
    draft_thread: Optional[Dict[str, Any]],
) -> bool:
    """Whether an agent reply was actually sent after this draft was created.

    `delete_draft` can only neutralize a draft's text — Help Scout no longer
    accepts DELETE on conversation threads — so a conversation that has been
    replied to keeps its leftover draft thread forever and would otherwise sit
    in the drafts queue permanently.

    Comparing timestamps is what separates that leftover from a draft that still
    needs attention. Deliberately *not* matched on the placeholder body: the same
    "Manual review required before sending a reply." text is what Supa posts when
    it cannot generate a draft at all, and those must stay visible until someone
    replies to them.
    """
    if not draft_thread:
        return False

    draft_created_at = _parse_thread_created_at(draft_thread.get("createdAt"))
    if draft_created_at is None:
        # Without a timestamp we cannot prove the draft is stale — keep it visible.
        return False

    draft_id = _thread_id_for_ordering(draft_thread.get("id"))
    for reply in threads:
        if (
            not _is_agent_message_thread(reply)
            or _is_draft_thread(reply)
            or reply.get("state") not in (None, "published")
        ):
            continue
        reply_created_at = _parse_thread_created_at(reply.get("createdAt"))
        if reply_created_at is None:
            continue
        if reply_created_at > draft_created_at:
            return True
        if reply_created_at != draft_created_at:
            continue

        # Help Scout IDs are monotonic. They provide the deterministic ordering
        # when two threads share the same timestamp, but only when both are valid.
        reply_id = _thread_id_for_ordering(reply.get("id"))
        if draft_id is not None and reply_id is not None and reply_id > draft_id:
            return True

    return False


def _parse_thread_created_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _thread_id_for_ordering(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        normalized = int(value)
    else:
        return None
    return normalized if 0 < normalized <= _MAX_SAFE_THREAD_ID else None


def _is_manual_review_placeholder(html: str) -> bool:
    """Supa's blanked-draft text (the same check as processor._is_placeholder_manual_review_html)."""
    return strip_html(html or "") == "Manual review required before sending a reply."


def get_supa_owned_draft_thread_ids(threads: List[Dict[str, Any]]) -> List[int]:
    """Return the IDs of drafts the service owns. Operator drafts must remain available for review."""
    owned = []
    for thread in threads:
        if not (_is_draft_thread(thread) and thread.get("id")):
            continue
        source = thread.get("source")
        if not isinstance(source, dict) or source.get("type") != "api":
            continue
        if _is_manual_review_placeholder(thread.get("body", "")):
            continue
        owned.append(thread)
    if owned and all(thread.get("createdAt") for thread in owned):
        owned.sort(key=lambda thread: thread.get("createdAt", ""))
    return [thread["id"] for thread in owned]


def agent_replies_from_threads(threads: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Replies an agent actually sent (published messages from a user), in thread order."""
    replies = []
    for thread in threads:
        if thread.get("type") != "message" or not _is_published_thread(thread):
            continue
        if (thread.get("createdBy") or {}).get("type") == "user":
            replies.append({
                "body": strip_html(thread.get("body", "")),
                "created_at": thread.get("createdAt", ""),
            })
    return replies


def get_draft_thread_ids(threads: List[Dict[str, Any]]) -> List[int]:
    """Return all draft thread ids, ordered by creation time when available."""
    draft_threads = [thread for thread in threads if _is_draft_thread(thread) and thread.get("id")]
    if not draft_threads:
        return []

    threads_with_timestamps = [thread for thread in draft_threads if thread.get("createdAt")]
    if len(threads_with_timestamps) == len(draft_threads):
        draft_threads = sorted(draft_threads, key=lambda thread: thread.get("createdAt", ""))

    return [thread["id"] for thread in draft_threads]


def get_latest_customer_thread(threads: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the newest customer-authored thread."""
    return _get_latest_thread(threads, _is_customer_thread)


# Small placeholders can arrive when a mail client omits the original image.
BLANK_IMAGE_MAX_BYTES = 1024


# Inline images may be fetched only from explicitly owned attachment hosts.
HELPSCOUT_INLINE_IMAGE_HOSTS = installation.INLINE_IMAGE_HOSTS
INLINE_ICON_MAX_PX = 64
_SCREENSHOT_NAME_RE = re.compile(
    r"screen\s*-?shot|screen[-_ ]?capture|bildschirm|captur|clipboard|pasted|snip|schermata",
    re.IGNORECASE,
)
_SCREENSHOT_TEXT_RE = re.compile(
    r"screen\s*-?shots?|screengrab|bildschirmfoto|captur[aă] de ecran|capture d.[ée]cran"
    r"|captura de pantalla|schermata|print\s*screen",
    re.IGNORECASE,
)
_IMAGE_SUFFIX_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
_INLINE_SIGNATURE_MARKERS = {
    "helpscout-signature", "gmail_signature", "ex-gmail_signature", "moz-signature",
    "protonmail_signature_block", "signature",
}
_INLINE_QUOTE_MARKERS = {"gmail_quote", "ex-gmail_quote", "yahoo_quoted"}
_POSSIBLE_BRANDING_NAME_RE = re.compile(
    r"signature|(?:^|[-_\s])logo(?:[-_\s.]|$)|_email\.[a-z]+$",
    re.IGNORECASE,
)


def _inline_in_region(tag: Any, markers: set[str], *, quoted: bool = False) -> bool:
    for parent in (tag, *tag.parents):
        if quoted and parent.name == "blockquote":
            return True
        classes = {str(value).lower() for value in parent.get("class", [])}
        if classes & markers or str(parent.get("id", "")).lower() in markers:
            return True
    return False


def is_helpscout_inline_image_url(url: str) -> bool:
    parts = urlsplit(url or "")
    return parts.scheme == "https" and parts.hostname in HELPSCOUT_INLINE_IMAGE_HOSTS and parts.path.startswith("/inline/")


def extract_inline_images(thread: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Images pasted into a customer message body, shaped like attachment metadata."""
    body = thread.get("body") or ""
    soup = BeautifulSoup(body, "html.parser")
    # Joining text nodes also recognizes "See s<img ...><br>creenshots". Do not let an
    # older quoted screenshot or a signature describe evidence in the current message.
    message_text = "".join(
        str(node) for node in soup.find_all(string=True)
        if not _inline_in_region(node.parent, _INLINE_QUOTE_MARKERS, quoted=True)
        and not _inline_in_region(node.parent, _INLINE_SIGNATURE_MARKERS)
    )
    message_mentions_screenshot = bool(_SCREENSHOT_TEXT_RE.search(message_text))
    images: List[Dict[str, Any]] = []
    for tag in soup.find_all("img"):
        if _inline_in_region(tag, _INLINE_QUOTE_MARKERS, quoted=True):
            continue
        src = str(tag.get("src", "")).strip()
        if not is_helpscout_inline_image_url(src) or any(i["url"] == src for i in images):
            continue
        sizes = [int(v) for v in (str(tag.get("width", "")), str(tag.get("height", ""))) if v.isdigit()]
        if sizes and max(sizes) <= INLINE_ICON_MAX_PX:
            continue  # social icons and spacers
        alt = str(tag.get("alt", "")).strip()
        url_filename = urlsplit(src).path.rsplit("/", 1)[-1]
        filename = alt or url_filename
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix not in _IMAGE_SUFFIX_MIME:
            suffix = url_filename.rsplit(".", 1)[-1].lower() if "." in url_filename else ""
        names = f"{alt} {url_filename}"
        screenshot_name = bool(_SCREENSHOT_NAME_RE.search(names))
        # A logo/signature filename alone cannot prove the image belongs to a signature:
        # it may be the very image the customer needs help with. Keep its purpose unknown
        # and its pixels material unless a signature container supplies actual evidence.
        signature = _inline_in_region(tag, _INLINE_SIGNATURE_MARKERS)
        possible_branding = bool(_POSSIBLE_BRANDING_NAME_RE.search(names))
        large_body_image = len(sizes) == 2 and min(sizes) > INLINE_ICON_MAX_PX and max(sizes) >= 200
        screenshot = not signature and (
            screenshot_name or message_mentions_screenshot or (large_body_image and not possible_branding)
        )
        images.append({
            "id": None,
            "url": src,
            "filename": filename,
            "mime_type": _IMAGE_SUFFIX_MIME.get(suffix, "image/png"),
            "size": 0,
            "inline": True,
            "is_signature": signature,
            # Only a likely screenshot is primary evidence; a signature logo must not block.
            "is_image": screenshot,
            "is_blank": False,
        })
    return images


def extract_customer_attachments(threads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attachment metadata from the newest customer message.

    Attachments are primary evidence — a ticket was once answered at 6/10 around five
    unopened screenshots that contained the entire root cause. Surfacing them into the
    pipeline context is what lets the investigator know they exist and the blocker layer
    insist they are looked at.
    """
    thread = get_latest_customer_thread(threads)
    if not thread:
        return []

    attachments: List[Dict[str, Any]] = []
    for attachment in (thread.get("_embedded") or {}).get("attachments") or []:
        mime = (attachment.get("mimeType") or "").lower()
        size = attachment.get("size") or 0
        attachments.append({
            "id": attachment.get("id"),
            "filename": attachment.get("filename") or "",
            "mime_type": mime,
            "size": size,
            "is_image": mime.startswith("image/"),
            "is_blank": mime.startswith("image/") and 0 < size <= BLANK_IMAGE_MAX_BYTES,
        })
    attachments.extend(extract_inline_images(thread))
    return attachments


def material_customer_images(attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Images requiring pixel inspection, including inline images of unknown purpose.

    Only explicit signature evidence or a blank artifact excludes an extracted image;
    failing to recognize its filename as a screenshot cannot make its pixels optional.
    Small icons/trackers and untrusted remote hosts are excluded during extraction.
    """
    return [a for a in (attachments or [])
            if not a.get("is_blank") and not a.get("is_signature")
            and (a.get("is_image") or a.get("inline"))]


def describe_attachments(attachments: List[Dict[str, Any]]) -> str:
    """Human/agent-readable summary of the latest customer message's attachments."""
    if not attachments:
        return ""

    lines = ["Customer attachments on the latest message:"]
    for a in attachments:
        if a.get("inline"):
            if a["is_image"]:
                kind = "pasted into the message"
            elif a.get("is_signature"):
                kind = "pasted into the message, probably a signature logo"
            else:
                kind = "pasted into the message, image purpose not verified; inspect before claiming a screenshot is missing"
            lines.append(f"- {a['filename']} ({kind})")
            continue
        size_note = f"{a['size']} bytes" if a["size"] < 10240 else f"{a['size'] // 1024} KB"
        flag = " — BLANK transport artifact, the real image never arrived" if a["is_blank"] else ""
        lines.append(f"- {a['filename']} ({a['mime_type']}, {size_note}){flag}")

    real_images = material_customer_images(attachments)
    blanks = [a for a in attachments if a["is_blank"]]
    unclassified_inline = [a for a in attachments if a.get("inline") and not a.get("is_signature") and not a["is_image"]]
    if real_images:
        lines.append(
            "These images are primary evidence: open and analyze every one before "
            "diagnosing. Do not answer around them."
        )
    if blanks and not real_images and not unclassified_inline:
        lines.append(
            "Every image is a blank artifact — the customer's screenshots did not come "
            "through. Tell them so, and ask them to attach the images as files instead of "
            "pasting them into the message."
        )
    return "\n".join(lines)


def get_latest_actionable_thread(threads: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the newest non-note, non-draft customer/agent thread."""
    return _get_latest_thread(
        threads,
        lambda thread: thread.get("type") in {"customer", "message"}
        and thread.get("type") != "note"
        and not _is_draft_thread(thread),
    )


def parse_api_conversation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse HelpScout API conversation response into structured data."""
    # Basic info
    ticket_id = data.get("id", "")
    subject = data.get("subject", "")

    # Customer info - API uses different structure
    customer = data.get("primaryCustomer") or data.get("createdBy") or {}
    customer_id = customer.get("id")
    customer_email = customer.get("email", "")
    customer_name = " ".join(filter(None, [
        customer.get("first", ""),
        customer.get("last", "")
    ])) or customer.get("email", "Customer")

    # Tags
    tags = [t.get("tag", "") for t in data.get("tags", [])]

    # Threads from embedded data
    threads = data.get("_embedded", {}).get("threads", [])

    # Find the most recent customer message
    customer_thread = get_latest_customer_thread(threads)
    customer_message = strip_html(customer_thread.get("body", "")) if customer_thread else ""

    # Find internal note (instructions)
    note_thread = get_latest_note_thread(threads)
    internal_note = strip_html(note_thread.get("body", "")) if note_thread else ""

    # Detect theme from tags, subject, and message
    theme_slug = detect_theme(tags=tags, subject=subject, message=customer_message)

    # Extract conversation history
    conversation_history = extract_conversation_history(threads)

    return {
        "id": ticket_id,
        "subject": subject,
        "customer_id": customer_id,
        "customer_email": customer_email,
        "customer_name": customer_name,
        "theme_tag": theme_slug or "",
        "tags": tags,
        "message": customer_message,
        "internal_note": internal_note,
        "attachments": extract_customer_attachments(threads),
        "conversation_history": conversation_history,
        "source_type": "api"
    }


def parse_webhook_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Parse HelpScout webhook payload into structured data."""
    # Basic info
    raw_id = payload.get("id", "")
    ticket_id = int(raw_id) if raw_id and str(raw_id).isdigit() else None
    subject = payload.get("subject", "")

    # Customer info
    customer = payload.get("primaryCustomer", {})
    customer_id = customer.get("id")
    customer_email = customer.get("email", "")
    customer_name = " ".join(filter(None, [
        customer.get("first", ""),
        customer.get("last", "")
    ])) or "Customer"

    # Tags
    tags = [t.get("tag", "") for t in payload.get("tags", [])]

    # Threads
    threads = payload.get("_embedded", {}).get("threads", [])

    # Find customer message
    customer_thread = get_latest_customer_thread(threads)
    message = strip_html(customer_thread.get("body", "")) if customer_thread else ""

    # Find internal note (your instructions)
    note_thread = get_latest_note_thread(threads)
    internal_note = strip_html(note_thread.get("body", "")) if note_thread else ""

    # Source check (to avoid loops)
    source_type = (customer_thread or {}).get("source", {}).get("type", "")

    # Detect theme from tags, subject, and message
    theme_slug = detect_theme(tags=tags, subject=subject, message=message)

    # Extract conversation history
    conversation_history = extract_conversation_history(threads)

    return {
        "id": ticket_id,
        "subject": subject,
        "customer_id": customer_id,
        "customer_email": customer_email,
        "customer_name": customer_name,
        "theme_tag": theme_slug or "",
        "tags": tags,
        "message": message,
        "internal_note": internal_note,
        "attachments": extract_customer_attachments(threads),
        "conversation_history": conversation_history,
        "source_type": source_type
    }


def _positive_webhook_id(value: Any, *, field_name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= _MAX_SAFE_THREAD_ID
    ):
        raise ValueError(f"{field_name} is invalid")
    return value


def _webhook_text(value: Any, *, field_name: str, max_length: int) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ValueError(f"{field_name} is invalid")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError(f"{field_name} is invalid") from error
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} is invalid")
    return value


def _webhook_url(value: Any) -> str:
    url = _webhook_text(value, field_name="webhook URL", max_length=2_048)
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("webhook URL is invalid") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None and not 1 <= port <= 65_535
        or parsed.fragment
    ):
        raise ValueError("webhook URL is invalid")
    return url


def _webhook_events(value: Any) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or len(value) > 100
        or any(
            not isinstance(event, str)
            or _WEBHOOK_EVENT_RE.fullmatch(event) is None
            for event in value
        )
    ):
        raise ValueError("webhook events are invalid")
    events = tuple(value)
    if len(set(events)) != len(events):
        raise ValueError("webhook events are invalid")
    return events


def _webhook_mailbox_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > 10_000:
        raise ValueError("webhook mailbox IDs are invalid")
    mailbox_ids = tuple(
        _positive_webhook_id(item, field_name="webhook mailbox ID") for item in value
    )
    if len(set(mailbox_ids)) != len(mailbox_ids):
        raise ValueError("webhook mailbox IDs are invalid")
    return mailbox_ids


def _webhook_from_payload(value: Any) -> HelpScoutWebhook:
    if not isinstance(value, dict):
        raise HelpScoutProtocolError(
            "Help Scout returned an invalid webhook",
            code="invalid_webhook",
        )
    try:
        webhook_id = _positive_webhook_id(value.get("id"), field_name="webhook ID")
        url = _webhook_url(value.get("url"))
        state = value.get("state")
        if state not in {"enabled", "disabled"}:
            raise ValueError("webhook state is invalid")
        events = _webhook_events(value.get("events"))
        notification = value.get("notification")
        if not isinstance(notification, bool):
            raise ValueError("webhook notification flag is invalid")
        payload_version = value.get("payloadVersion")
        if payload_version not in {"V2", "V3"}:
            raise ValueError("webhook payload version is invalid")
        label_value = value.get("label")
        label = (
            None
            if label_value is None
            else _webhook_text(label_value, field_name="webhook label", max_length=255)
        )
        mailbox_ids = _webhook_mailbox_ids(value.get("mailboxIds", []))
    except (TypeError, UnicodeError, ValueError) as error:
        raise HelpScoutProtocolError(
            "Help Scout returned an invalid webhook",
            code="invalid_webhook",
        ) from error
    return HelpScoutWebhook(
        id=webhook_id,
        url=url,
        state=state,
        events=events,
        notification=notification,
        payload_version=payload_version,
        label=label,
        mailbox_ids=mailbox_ids,
    )


def _webhook_write_payload(
    *,
    url: str,
    events: tuple[str, ...] | List[str],
    secret: str,
    notification: bool,
    payload_version: str,
    label: str | None,
    mailbox_ids: tuple[int, ...] | List[int],
) -> Dict[str, Any]:
    normalized_url = _webhook_url(url)
    normalized_events = _webhook_events(events)
    if not isinstance(secret, str) or _WEBHOOK_SECRET_RE.fullmatch(secret) is None:
        raise ValueError("webhook secret is invalid")
    if not isinstance(notification, bool):
        raise ValueError("webhook notification flag is invalid")
    if payload_version not in {"V2", "V3"}:
        raise ValueError("webhook payload version is invalid")
    normalized_label = (
        None
        if label is None
        else _webhook_text(label, field_name="webhook label", max_length=255)
    )
    normalized_mailboxes = _webhook_mailbox_ids(mailbox_ids)
    payload: Dict[str, Any] = {
        "url": normalized_url,
        "events": list(normalized_events),
        "secret": secret,
        "notification": notification,
        "payloadVersion": payload_version,
        "mailboxIds": list(normalized_mailboxes),
    }
    if normalized_label is not None:
        payload["label"] = normalized_label
    return payload


class HelpScoutClient:
    """HelpScout API client with OAuth2 authentication."""

    BASE_URL = mailbox_provider.VENDOR_API
    TOKEN_URL = BASE_URL + "/oauth2/token"
    DEFAULT_TIMEOUT = httpx.Timeout(20.0, connect=10.0)

    def __init__(self, client_id: str, client_secret: str, *, environ=None):
        self.BASE_URL, self.client_id, self.client_secret = mailbox_provider.connection(
            client_id, client_secret, environ=environ
        )
        self._operations_origin = installation.OPERATIONS_ORIGIN or mailbox_provider.native_origin(environ)
        self.TOKEN_URL = self.BASE_URL + "/oauth2/token"
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._http = httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT)

    async def _ensure_token(self) -> str:
        """Get or refresh access token."""
        if self._access_token and self._token_expires and datetime.now() < self._token_expires:
            return self._access_token

        response = await self._http.post(
            self.TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret
            }
        )
        response.raise_for_status()
        data = response.json()

        self._access_token = data["access_token"]
        self._token_expires = datetime.now() + timedelta(seconds=data.get("expires_in", 7200) - 300)

        return self._access_token

    async def _request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make authenticated request."""
        classify_mutation_auth = kwargs.pop("_classify_mutation_auth", False)
        try:
            token = await self._ensure_token()
        except httpx.TimeoutException as exc:
            if not classify_mutation_auth:
                raise
            raise _MutationPreSubmissionError(
                status_code=None,
                detail="oauth_request_timeout",
            ) from exc
        except httpx.HTTPStatusError as exc:
            if not classify_mutation_auth:
                raise
            raise _MutationPreSubmissionError(
                status_code=exc.response.status_code,
                detail="oauth_token_error",
            ) from exc
        except httpx.RequestError as exc:
            if not classify_mutation_auth:
                raise
            raise _MutationPreSubmissionError(
                status_code=None,
                detail="oauth_network_error",
            ) from exc
        except (OverflowError, ValueError, KeyError, TypeError) as exc:
            if not classify_mutation_auth:
                raise
            raise _MutationPreSubmissionError(
                status_code=200,
                detail="invalid_oauth_token_response",
            ) from exc
        if classify_mutation_auth and (not isinstance(token, str) or not token):
            raise _MutationPreSubmissionError(
                status_code=200,
                detail="invalid_oauth_token_response",
            )
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {token}"
        headers["Content-Type"] = "application/json"

        if classify_mutation_auth:
            request = self._http.build_request(
                method,
                f"{self.BASE_URL}{endpoint}",
                headers=headers,
                **kwargs,
            )
            return await self._http.send(
                request,
                stream=True,
                follow_redirects=False,
            )
        return await self._http.request(
            method,
            f"{self.BASE_URL}{endpoint}",
            headers=headers,
            **kwargs
        )

    async def _ensure_read_token(self, endpoint: str) -> str:
        """Use the existing OAuth flow while typing failures for new read paths only."""
        try:
            token = await self._ensure_token()
        except httpx.TimeoutException as exc:
            raise HelpScoutTransportError(
                "Help Scout OAuth request timed out",
                code="request_timeout",
                retryable=True,
                details={"endpoint": endpoint, "operation": "oauth_token"},
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise HelpScoutAuthError(
                f"Help Scout OAuth failed with status {status}",
                status=status,
                code="oauth_token_error",
                retryable=status == 429 or status >= 500,
                details={"operation": "oauth_token"},
            ) from exc
        except httpx.RequestError as exc:
            raise HelpScoutTransportError(
                "Help Scout OAuth network request failed",
                code="network_error",
                retryable=True,
                details={"endpoint": endpoint, "operation": "oauth_token"},
            ) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise HelpScoutAuthError(
                "Help Scout OAuth returned an invalid token response",
                status=200,
                code="invalid_oauth_token_response",
                details={"operation": "oauth_token"},
            ) from exc

        if not isinstance(token, str) or not token:
            cause = ValueError("invalid access_token field")
            raise HelpScoutAuthError(
                "Help Scout OAuth returned an invalid token response",
                status=200,
                code="invalid_oauth_token_response",
                details={"operation": "oauth_token"},
            ) from cause
        return token

    async def _read_bounded(
        self,
        endpoint: str,
        *,
        max_bytes: int,
        question: dict | None = None,
    ) -> _BoundedReadResponse:
        """Stream an authenticated read; a private precedent query travels in the body."""
        token = await self._ensure_read_token(endpoint)
        request = self._http.build_request(
            "GET" if question is None else "POST",
            f"{self.BASE_URL}{endpoint}",
            json=question,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        response: httpx.Response | None = None
        try:
            response = await self._http.send(
                request,
                stream=True,
                follow_redirects=False,
            )
            headers = httpx.Headers(response.headers)

            # Redirect and error bodies are neither useful nor safe to materialize.
            if not 200 <= response.status_code < 300:
                return _BoundedReadResponse(response.status_code, headers, b"")

            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except (TypeError, ValueError) as exc:
                    raise HelpScoutProtocolError(
                        "Help Scout returned an invalid Content-Length",
                        status=response.status_code,
                        code="invalid_content_length",
                        details={"endpoint": endpoint},
                    ) from exc
                if declared_length < 0:
                    raise HelpScoutProtocolError(
                        "Help Scout returned an invalid Content-Length",
                        status=response.status_code,
                        code="invalid_content_length",
                        details={"endpoint": endpoint},
                    )
                if declared_length > max_bytes:
                    raise HelpScoutResponseTooLargeError(
                        "Help Scout response exceeds the read limit",
                        status=response.status_code,
                        code="response_too_large",
                        details={"endpoint": endpoint, "max_bytes": max_bytes},
                    )

            content = bytearray()
            async for chunk in response.aiter_bytes():
                if len(content) + len(chunk) > max_bytes:
                    raise HelpScoutResponseTooLargeError(
                        "Help Scout response exceeds the read limit",
                        status=response.status_code,
                        code="response_too_large",
                        details={"endpoint": endpoint, "max_bytes": max_bytes},
                    )
                content.extend(chunk)
            return _BoundedReadResponse(response.status_code, headers, bytes(content))
        except HelpScoutError:
            raise
        except httpx.InvalidURL as exc:
            # The request URL is constructed locally; an InvalidURL here can only
            # come from httpx validating an upstream redirect Location.
            raise HelpScoutRedirectError(
                "Help Scout redirect has an invalid Location",
                status=301,
                code="invalid_merge_redirect",
                details={"endpoint": endpoint},
            ) from exc
        except httpx.TimeoutException as exc:
            raise HelpScoutTransportError(
                "Help Scout read request timed out",
                code="request_timeout",
                retryable=True,
                details={"endpoint": endpoint},
            ) from exc
        except httpx.RequestError as exc:
            raise HelpScoutTransportError(
                "Help Scout read network request failed",
                code="network_error",
                retryable=True,
                details={"endpoint": endpoint},
            ) from exc
        finally:
            if response is not None:
                await response.aclose()

    async def _mutation_result(
        self,
        method: str,
        endpoint: str,
        *,
        success_status: int,
        capture_resource_id: bool = False,
        **kwargs,
    ) -> MutationResult:
        """Submit exactly one mutation and classify its response without body access."""
        try:
            response = await self._request(
                method,
                endpoint,
                _classify_mutation_auth=True,
                **kwargs,
            )
        except _MutationPreSubmissionError as exc:
            return MutationResult(
                ok=False,
                status_code=exc.status_code,
                outcome_unknown=False,
                detail=exc.detail,
            )
        except httpx.TimeoutException:
            return MutationResult(
                ok=False,
                status_code=None,
                outcome_unknown=True,
                detail="request_timeout",
            )
        except httpx.RequestError:
            return MutationResult(
                ok=False,
                status_code=None,
                outcome_unknown=True,
                detail="network_error",
            )

        try:
            status_code = getattr(response, "status_code", None)
            if isinstance(status_code, bool) or not isinstance(status_code, int):
                result = MutationResult(
                    ok=False,
                    status_code=None,
                    detail="invalid_response_status",
                )
            elif (self.BASE_URL != mailbox_provider.VENDOR_API and status_code == 504
                  and response.headers.get("X-Inbox-Delivery-State") == "uncertain"):
                result = MutationResult(ok=False, status_code=status_code, outcome_unknown=True,
                                        detail="mail_acceptance_unconfirmed")
            elif status_code != success_status:
                result = MutationResult(
                    ok=False,
                    status_code=status_code,
                    detail=_mutation_failure_detail(status_code),
                )
            elif capture_resource_id:
                resource_id, detail = _resource_id_from_response(response)
                result = MutationResult(
                    ok=True,
                    status_code=status_code,
                    resource_id=resource_id,
                    detail=detail,
                )
            else:
                result = MutationResult(ok=True, status_code=status_code)
        finally:
            close_response = getattr(response, "aclose", None)
            if close_response is not None:
                try:
                    await close_response()
                except Exception:
                    logger.warning(
                        "Help Scout mutation response could not be closed cleanly"
                    )
        return result

    async def run_wrong_address_workflow_result(self, conversation_id: int) -> MutationResult:
        """Run the mailbox's own "Email sent to the wrong address" manual workflow once.

        The workflow holds the owner's wording, closes the conversation and files it; running
        it keeps that text in one place. Only the owned mailbox offers this, and its
        idempotency key makes a retried call return the first result instead of a second email.
        """
        if self.BASE_URL == mailbox_provider.VENDOR_API:
            return MutationResult(ok=False, status_code=None, detail="native_mailbox_required")
        cid = int(conversation_id)
        try:
            response = await self._read_bounded("/workflows", max_bytes=262144)
            _raise_for_read_status(response, "/workflows")
            data = _response_json_object(response, "/workflows")
        except (HelpScoutError, httpx.HTTPError, ValueError) as exc:
            logger.warning("Workflow list unavailable: %s", type(exc).__name__)
            return MutationResult(ok=False, status_code=None, detail="workflow_list_unavailable")
        rules = (data.get("_embedded") or {}).get("workflows")
        matches = [
            rule for rule in (rules if isinstance(rules, list) else [])
            if isinstance(rule, dict) and rule.get("type") == "manual" and rule.get("status") == "active"
            and "wrong address" in str(rule.get("name", "")).lower()
            and type(rule.get("id")) is int and rule["id"] > 0
        ]
        if len(matches) != 1:
            return MutationResult(ok=False, status_code=None, detail="wrong_address_workflow_not_found")
        return await self._mutation_result(
            "POST",
            f"/conversations/{cid}/workflows/{matches[0]['id']}",
            success_status=200,
            headers={"Idempotency-Key": f"supa-wrong-address-{cid}"},
        )

    async def precedents(self, *, conversation_id=None, text="", tag="", limit=3):
        """Closed conversations with a sent answer, ranked by similarity, from the owned mailbox.

        Returns None when the vendor mailbox is selected or the request fails, so callers keep
        their keyword search. One request replaces a search plus one read per candidate.
        """
        if self.BASE_URL == mailbox_provider.VENDOR_API:
            return None
        from urllib.parse import urlencode

        params = {"limit": max(1, min(int(limit), 10))}
        if conversation_id:
            params["conversation"] = int(conversation_id)
        if str(text).strip():
            # Sent with the conversation as well, so an expanded query is searched as written.
            params["q"] = str(text)[:1500]
        if tag:
            params["tag"] = str(tag)[:80]
        endpoint = "/precedents?" + urlencode(params)
        try:
            response = await self._read_bounded(endpoint, max_bytes=262144)
            _raise_for_read_status(response, "/precedents")
            data = _response_json_object(response, "/precedents")
        except (HelpScoutError, httpx.HTTPError, ValueError) as exc:
            logger.warning("Precedent lookup unavailable: %s", type(exc).__name__)
            return None
        found = []
        for item in data.get("precedents") if isinstance(data.get("precedents"), list) else []:
            if not isinstance(item, dict) or type(item.get("id")) is not int:
                continue
            answer = item.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                continue
            found.append({
                "id": item["id"],
                "subject": str(item.get("subject") or "(no subject)")[:300],
                "replies": [{"body": answer, "created_at": str(item.get("answered") or "")}],
                "rating": item.get("rating") if isinstance(item.get("rating"), str) else None,
                "shared_tags": [str(t).lower() for t in item.get("shared_tags") or [] if isinstance(t, str)],
            })
        return found

    async def native_report(self, path: str, params: dict, *, question: dict | None = None) -> dict:
        """Read a desk report from the owned mailbox. Callers supply fixed paths and numeric filters."""
        if self.BASE_URL == mailbox_provider.VENDOR_API:
            raise HelpScoutUpstreamError("Native mailbox unavailable", status=503, code="service_unavailable")
        endpoint = path + ("?" + urlencode(params) if params else "")
        response = await self._read_bounded(endpoint, max_bytes=4 * 1024 * 1024, question=question)
        _raise_for_read_status(response, path)
        return _response_json_object(response, path)

    async def native_operations(self):
        """Return only bounded, enumerated operational counts from the owned gateway."""
        origin = self._operations_origin
        if not origin or self.BASE_URL != origin.rstrip("/") + "/v2":
            raise ValueError("Native mailbox is not selected")
        response = await self._read_bounded("/operations", max_bytes=8192)
        _raise_for_read_status(response, "/operations")
        data = _response_json_object(response, "/operations")
        kinds = {"Outgoing reply", "Acknowledgment", "Agent event", "Incoming email",
                 "Provider notification", "Incoming notification"}
        counts = data.get("issues")
        if (set(data) != {"writer", "issues"} or data["writer"] not in ("standby", "live", "paused")
                or not isinstance(counts, dict) or not set(counts) <= kinds
                or any(type(v) is not int or not 0 <= v <= 2**31-1 for v in counts.values())):
            raise ValueError("Invalid native operations summary")
        return {"writer": data["writer"], "issues": counts}

    async def list_webhooks(self) -> List[HelpScoutWebhook]:
        """Return every webhook as a validated typed record."""

        webhooks: List[HelpScoutWebhook] = []
        seen_ids: set[int] = set()
        page_number = 1
        while True:
            endpoint = "/webhooks" if page_number == 1 else f"/webhooks?page={page_number}"
            response = await self._read_bounded(
                endpoint,
                max_bytes=_MAX_WEBHOOK_RESPONSE_BYTES,
            )
            _raise_for_read_status(response, endpoint)
            payload = _response_json_object(response, endpoint)
            embedded = payload.get("_embedded")
            raw_webhooks = (
                embedded.get("webhooks") if isinstance(embedded, dict) else None
            )
            page = payload.get("page")
            if not isinstance(raw_webhooks, list) or not isinstance(page, dict):
                raise HelpScoutProtocolError(
                    "Help Scout returned an invalid webhook list",
                    code="invalid_webhook_list",
                    details={"endpoint": endpoint},
                )
            total_pages = page.get("totalPages")
            returned_page = page.get("number", page_number)
            if (
                not isinstance(total_pages, int)
                or isinstance(total_pages, bool)
                or not 0 <= total_pages <= _MAX_WEBHOOK_LIST_PAGES
                or not isinstance(returned_page, int)
                or isinstance(returned_page, bool)
                or returned_page != page_number
                or total_pages == 0 and raw_webhooks
                or total_pages > 0 and page_number > total_pages
            ):
                raise HelpScoutProtocolError(
                    "Help Scout returned invalid webhook pagination",
                    code="invalid_pagination",
                    details={"endpoint": endpoint},
                )
            for raw_webhook in raw_webhooks:
                webhook = _webhook_from_payload(raw_webhook)
                if webhook.id in seen_ids:
                    raise HelpScoutProtocolError(
                        "Help Scout repeated a webhook ID",
                        code="duplicate_webhook",
                        details={"endpoint": endpoint},
                    )
                seen_ids.add(webhook.id)
                webhooks.append(webhook)
            if total_pages == 0 or page_number == total_pages:
                return webhooks
            page_number += 1

    async def create_webhook(
        self,
        *,
        url: str,
        events: tuple[str, ...] | List[str],
        secret: str,
        notification: bool,
        payload_version: str,
        label: str | None,
        mailbox_ids: tuple[int, ...] | List[int],
    ) -> MutationResult:
        """Create one webhook without ever reading an upstream response body."""

        payload = _webhook_write_payload(
            url=url,
            events=events,
            secret=secret,
            notification=notification,
            payload_version=payload_version,
            label=label,
            mailbox_ids=mailbox_ids,
        )
        return await self._mutation_result(
            "POST",
            "/webhooks",
            success_status=201,
            capture_resource_id=True,
            json=payload,
        )

    async def update_webhook(
        self,
        webhook_id: int,
        *,
        url: str,
        events: tuple[str, ...] | List[str],
        secret: str,
        notification: bool,
        payload_version: str,
        label: str | None,
        mailbox_ids: tuple[int, ...] | List[int],
    ) -> MutationResult:
        """Replace one webhook while callers explicitly preserve desired fields."""

        normalized_id = _positive_webhook_id(webhook_id, field_name="webhook ID")
        payload = _webhook_write_payload(
            url=url,
            events=events,
            secret=secret,
            notification=notification,
            payload_version=payload_version,
            label=label,
            mailbox_ids=mailbox_ids,
        )
        return await self._mutation_result(
            "PUT",
            f"/webhooks/{normalized_id}",
            success_status=204,
            json=payload,
        )

    async def add_note_result(
        self,
        conversation_id: int,
        text: str,
        status: str | None = None,
    ) -> MutationResult:
        """Add an internal note and return its explicit upstream outcome.

        On the owned mailbox the note says who wrote it (Supa, unless the caller named an
        agent) and, when the caller stated one, the record it stands for.
        """
        payload = {"text": str(text) if isinstance(text, RecordedNote) else text}
        if status:
            payload["status"] = status
        event = self._note_event(text)
        if event:
            result = await self._mutation_result(
                "POST",
                f"/conversations/{conversation_id}/notes",
                success_status=201,
                capture_resource_id=True,
                json={**payload, "event": event},
            )
            # A record the mailbox will not take must never cost the note: a definite refusal
            # of the request as a whole is asked again as plain text.
            if result.ok or result.outcome_unknown or result.status_code != 400:
                return result
            logger.warning("The mailbox refused a note sent with its record; asking again without the record")
        return await self._mutation_result(
            "POST",
            f"/conversations/{conversation_id}/notes",
            success_status=201,
            capture_resource_id=True,
            json=payload,
        )

    def _note_event(self, text) -> dict | None:
        """What the owned mailbox is told about a note; None for the vendor mailbox."""
        if self.BASE_URL == mailbox_provider.VENDOR_API:
            return None
        event = getattr(text, "event", None)
        if not isinstance(event, dict):
            return {"actor": "supa"}
        try:
            json.dumps(event, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError):
            return {"actor": "supa"}
        return event

    async def add_decision(self, conversation_id: int, text: str, record: dict | None = None) -> bool:
        """Record a decision where the reader will find it.

        The owned mailbox keeps it as a typed record in its agent ledger, outside the
        correspondence. Elsewhere, or when that write is refused or cannot be made, it is the
        internal note it always was: a decision is never lost to the record's own faults.
        """
        if record and self.BASE_URL != mailbox_provider.VENDOR_API:
            try:
                # One key for this decision: an uncertain first attempt is repeated under it, so
                # the mailbox stores the record once; another decision gets its own key even
                # when its wording is identical.
                payload = {**record, "operation": f"supa-decision-{uuid.uuid4().hex}"}
                # Encoded here as the HTTP client will encode it, so a fault shows before anything is sent.
                json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                result = None
                for _attempt in range(2):
                    result = await self._mutation_result(
                        "POST",
                        f"/conversations/{conversation_id}/events",
                        success_status=201,
                        json=payload,
                    )
                    if result.ok:
                        return True
                    if not result.outcome_unknown:
                        break
                logger.warning(
                    "Decision record not stored (%s); writing the note instead",
                    "outcome unknown" if result.outcome_unknown else result.detail or result.status_code,
                )
            except Exception:  # A record that cannot be encoded or sent must not take the note with it.
                logger.exception("Decision record failed for %s; writing the note instead", conversation_id)
        return await self.add_note(conversation_id, text)

    async def add_request(self, conversation_id: int, actor: str, text: str, key: str):
        """Ask the mailbox owner something, as an open request he sees under "Needs you".

        Returns (status code or None, the mailbox's answer). The answer of a 201 carries the
        request's id, its present status and whether this key had recorded it before: a request
        that was already closed is reported as closed, not as open again. The same key with
        another text is a 409.
        """
        if self.BASE_URL == mailbox_provider.VENDOR_API:
            return 409, {"message": "Requests live in the owned mailbox."}
        text = " ".join(str(text).split())
        # Unambiguous whatever a name or a key contains.
        digest = hashlib.sha256(json.dumps([int(conversation_id), str(actor), str(key)]).encode()).hexdigest()[:32]
        try:
            response = await self._request(
                "POST",
                f"/conversations/{int(conversation_id)}/events",
                json={"kind": "request", "actor": actor, "title": text[:120], "status": "open",
                      "data": {"action": text[:2000]}, "operation": f"supa-request-{digest}"},
            )
        except Exception as exc:  # Token faults included; the same key may be sent again.
            logger.warning("Owner request not sent: %s", type(exc).__name__)
            return None, {}
        try:
            body = response.json()
        except ValueError:
            body = {}
        return response.status_code, body if isinstance(body, dict) else {}

    async def close_request(self, conversation_id: int, event_id: int, actor: str) -> MutationResult:
        """Mark a request done: the asker got its answer, or no longer needs one."""
        if self.BASE_URL == mailbox_provider.VENDOR_API:
            return MutationResult(ok=False, status_code=None, detail="native_mailbox_required")
        return await self._mutation_result(
            "PATCH",
            f"/conversations/{int(conversation_id)}/events/{int(event_id)}",
            success_status=200,
            json={"status": "done", "actor": actor},
        )

    async def ledger(self, conversation_id: int, limit: int = 20) -> dict | None:
        """State and recent agent records of a conversation in the owned mailbox.

        None when the mailbox could not be asked; an empty dict when it has no such conversation.
        """
        if self.BASE_URL == mailbox_provider.VENDOR_API:
            return None
        endpoint = f"/conversations/{int(conversation_id)}/ledger?limit={max(1, min(int(limit), 200))}"
        try:
            response = await self._read_bounded(endpoint, max_bytes=1_048_576)
            if response.status_code == 404:
                return {}
            _raise_for_read_status(response, "/ledger")
            return _response_json_object(response, "/ledger")
        except (HelpScoutError, httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("Ledger unavailable: %s", type(exc).__name__)
            return None

    async def change_state(self, conversation_id: int, changes: dict, actor: str, reason: str = ""):
        """Change who handles a conversation, its stage, who is waited on or its reminders.

        One request, applied by the mailbox in one transaction. Returns (status code or None,
        the new state or the mailbox's own refusal text).
        """
        if self.BASE_URL == mailbox_provider.VENDOR_API:
            return None, "The conversation state lives in the owned mailbox."
        try:
            response = await self._request(
                "PATCH",
                f"/conversations/{int(conversation_id)}/state",
                json={"changes": changes, "actor": actor, "reason": reason},
            )
        except Exception as exc:  # Token faults included: the caller gets an answer, not a crash.
            logger.warning("State change not sent: %s", type(exc).__name__)
            return None, "The mailbox could not be reached."
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code == 200 and isinstance(body, dict):
            return 200, body
        said = (body.get("message") or body.get("detail")) if isinstance(body, dict) else None
        return response.status_code, str(said or "The mailbox refused the change.")[:300]

    async def update_note_result(
        self,
        conversation_id: int,
        thread_id: int,
        text: str,
    ) -> MutationResult:
        """Replace an internal note's text; sent as given, like ``add_note_result``.

        Help Scout edits a thread's text in place and leaves the conversation
        status alone (no status travels with the PATCH).
        """
        payload = {"op": "replace", "path": "/text", "value": str(text) if isinstance(text, RecordedNote) else text}
        event = self._note_event(text)
        if event and event.get("kind") is not None:
            # The writer restates what the edited note now is; a plain edit leaves the mailbox
            # to read the new wording and keep the original writer's name.
            result = await self._mutation_result(
                "PATCH",
                f"/conversations/{conversation_id}/threads/{thread_id}",
                success_status=204,
                json={**payload, "event": event},
            )
            # As with a new note: a refused record never costs the edit.
            if result.ok or result.outcome_unknown or result.status_code != 400:
                return result
            logger.warning("The mailbox refused an edit sent with its record; asking again without the record")
        return await self._mutation_result(
            "PATCH",
            f"/conversations/{conversation_id}/threads/{thread_id}",
            success_status=204,
            json=payload,
        )

    async def add_note(
        self,
        conversation_id: int,
        text: str,
        status: str | None = None,
    ) -> bool:
        """Compatibility wrapper returning a literal bool.

        Without ``status`` Help Scout reopens the conversation as Active, so a
        note written after a Pending send must pass ``status="pending"``.
        """
        return (await self.add_note_result(conversation_id, text, status)).ok

    async def add_draft_result(
        self,
        conversation_id: int,
        text: str,
        customer_id: int = None,
    ) -> MutationResult:
        """Add a draft reply and return its explicit upstream outcome."""
        payload = {
            "text": ensure_email_paragraph_spacing(text),
            "draft": True
        }
        if customer_id:
            payload["customer"] = {"id": customer_id}

        return await self._mutation_result(
            "POST",
            f"/conversations/{conversation_id}/reply",
            success_status=201,
            capture_resource_id=True,
            json=payload,
        )

    async def add_draft(
        self,
        conversation_id: int,
        text: str,
        customer_id: int = None,
    ) -> bool:
        """Compatibility wrapper returning a literal bool."""
        return (
            await self.add_draft_result(conversation_id, text, customer_id)
        ).ok

    async def send_reply_result(
        self,
        conversation_id: int,
        text: str,
        status: str = "pending",
        customer_id: int | None = None,
        *,
        verify_status: bool = True,
        cc: list[str] | None = None,
    ) -> MutationResult:
        """Send one reply and preserve success across status-warning failures.

        ``cc`` is Help Scout's reply-thread ``cc`` array (email strings); callers
        validate it (the agent gateway allows only addresses the customer gave).

        HelpScout's reply ``status`` field does not reliably move the
        conversation (an already-active conversation can stay active), so after
        a successful send we explicitly patch and verify the status. This is
        best-effort: the customer already has the reply, so a status hiccup must
        not make the send report as failed.
        """
        payload = {
            "text": ensure_email_paragraph_spacing(text),
            "status": status
        }
        if customer_id:
            payload["customer"] = {"id": customer_id}
        if cc:
            payload["cc"] = list(cc)

        result = await self._mutation_result(
            "POST",
            f"/conversations/{conversation_id}/reply",
            success_status=201,
            capture_resource_id=True,
            json=payload,
        )
        if not result.ok:
            return result

        if status and verify_status:
            try:
                verification = await self.ensure_conversation_status(
                    conversation_id,
                    status,
                )
                if not verification.get("verified"):
                    logger.warning(
                        "send_reply: could not verify status '%s' for conversation %s",
                        status, conversation_id,
                    )
                    result = _with_mutation_detail(result, "status_unverified")
            except Exception:
                logger.exception(
                    "send_reply: error setting status '%s' for conversation %s",
                    status, conversation_id,
                )
                result = _with_mutation_detail(result, "status_update_failed")

        return result

    async def send_reply(
        self,
        conversation_id: int,
        text: str,
        status: str = "pending",
        customer_id: int | None = None,
    ) -> bool:
        """Compatibility wrapper returning a literal bool."""
        return (
            await self.send_reply_result(
                conversation_id,
                text,
                status,
                customer_id,
            )
        ).ok

    async def delete_draft_result(
        self,
        conversation_id: int,
        thread_id: int,
    ) -> MutationResult:
        """Neutralize a draft thread.

        Help Scout's current thread endpoint no longer accepts DELETE for
        conversation threads. Replacing the text keeps stale or superseded draft
        content from remaining visible in the queue.
        """
        return await self.update_draft_result(
            conversation_id,
            thread_id,
            "<p>Manual review required before sending a reply.</p>",
        )

    async def delete_draft(self, conversation_id: int, thread_id: int) -> bool:
        """Compatibility wrapper returning a literal bool."""
        return (await self.delete_draft_result(conversation_id, thread_id)).ok

    async def update_draft_result(
        self,
        conversation_id: int,
        thread_id: int,
        text: str,
    ) -> MutationResult:
        """Replace the text for an existing draft thread."""
        return await self._mutation_result(
            "PATCH",
            f"/conversations/{conversation_id}/threads/{thread_id}",
            success_status=204,
            json={"op": "replace", "path": "/text", "value": ensure_email_paragraph_spacing(text)},
        )

    async def update_draft(
        self,
        conversation_id: int,
        thread_id: int,
        text: str,
    ) -> bool:
        """Compatibility wrapper returning a literal bool."""
        return (
            await self.update_draft_result(conversation_id, thread_id, text)
        ).ok

    async def close_conversation_result(
        self,
        conversation_id: int,
    ) -> MutationResult:
        """Close a conversation and return its explicit upstream outcome."""
        return await self.set_conversation_status_result(conversation_id, "closed")

    async def close_conversation(self, conversation_id: int) -> bool:
        """Compatibility wrapper returning a literal bool."""
        return (await self.close_conversation_result(conversation_id)).ok

    async def set_conversation_status_result(
        self,
        conversation_id: int,
        status: str,
    ) -> MutationResult:
        """Set conversation status explicitly."""
        return await self._mutation_result(
            "PATCH",
            f"/conversations/{conversation_id}",
            success_status=204,
            json={"op": "replace", "path": "/status", "value": status}
        )

    async def set_conversation_status(
        self,
        conversation_id: int,
        status: str,
    ) -> bool:
        """Compatibility wrapper returning a literal bool."""
        return (
            await self.set_conversation_status_result(conversation_id, status)
        ).ok

    async def ensure_conversation_status(
        self,
        conversation_id: int,
        status: str,
        max_attempts: int = 2,
    ) -> Dict[str, Any]:
        """Apply the requested status if the conversation still needs the transition."""
        normalized_status = (status or "").strip().lower()
        # Kept in the signature for caller compatibility. Mutations are never
        # retried automatically, regardless of the requested verification count.
        _ = max_attempts
        updated = await self.set_conversation_status(conversation_id, status)
        if not updated:
            return {"verified": False, "conversation": None}

        conversation = await self.get_conversation(conversation_id)
        current_status = (conversation.get("status") or "").strip().lower()
        return {
            "verified": current_status == normalized_status,
            "conversation": conversation,
        }

    async def update_subject_result(
        self,
        conversation_id: int,
        new_subject: str,
    ) -> MutationResult:
        """Update conversation subject."""
        return await self._mutation_result(
            "PATCH",
            f"/conversations/{conversation_id}",
            success_status=204,
            json={"op": "replace", "path": "/subject", "value": new_subject}
        )

    async def update_subject(self, conversation_id: int, new_subject: str) -> bool:
        """Compatibility wrapper returning a literal bool."""
        return (await self.update_subject_result(conversation_id, new_subject)).ok

    async def add_tags_result(
        self,
        conversation_id: int,
        tags: List[str],
        existing_tags: List[str] = None,
    ) -> MutationResult:
        """Add tags to conversation while preserving existing tags.

        Help Scout's tag write replaces the whole list. When the caller does not pass the current tags,
        read them first: without that, the call wiped every other tag (supa-ignore included).
        An explicit empty list still means the conversation has no tags.
        """
        if existing_tags is None:
            conversation = await self.get_conversation(conversation_id)
            existing_tags = [
                t.get("tag") if isinstance(t, dict) else t
                for t in (conversation or {}).get("tags") or []
            ]
            existing_tags = [t for t in existing_tags if t]
        all_tags = list(dict.fromkeys(list(existing_tags) + list(tags)))

        return await self._mutation_result(
            "PUT",
            f"/conversations/{conversation_id}/tags",
            success_status=204,
            json={"tags": all_tags}
        )

    async def add_tags(
        self,
        conversation_id: int,
        tags: List[str],
        existing_tags: List[str] = None,
    ) -> bool:
        """Compatibility wrapper returning a literal bool."""
        return (
            await self.add_tags_result(conversation_id, tags, existing_tags)
        ).ok

    async def remove_tag_result(
        self,
        conversation_id: int,
        tag_to_remove: str,
        existing_tags: List[str],
    ) -> MutationResult:
        """Remove a specific tag while preserving other tags."""
        new_tags = [t for t in existing_tags if t != tag_to_remove]
        return await self._mutation_result(
            "PUT",
            f"/conversations/{conversation_id}/tags",
            success_status=204,
            json={"tags": new_tags}
        )

    async def remove_tag(
        self,
        conversation_id: int,
        tag_to_remove: str,
        existing_tags: List[str],
    ) -> bool:
        """Compatibility wrapper returning a literal bool."""
        return (
            await self.remove_tag_result(
                conversation_id,
                tag_to_remove,
                existing_tags,
            )
        ).ok

    async def create_conversation_result(
        self, *, mailbox_id: int, subject: str, customer_id: int,
        text: str, draft: bool = True,
    ) -> MutationResult:
        """Single-attempt outbound creation with structured ambiguous outcomes."""
        customer = {"id": customer_id}
        return await self._mutation_result(
            "POST", "/conversations", success_status=201, capture_resource_id=True,
            json={"subject": subject, "customer": customer, "mailboxId": mailbox_id,
                  "type": "email", "status": "pending", "tags": ["supa-ignore"],
                  "threads": [{"type": "reply", "customer": customer,
                               "text": text, "draft": draft}]},
        )

    async def delete_conversation_result(self, conversation_id: int) -> MutationResult:
        """Delete the exact whole conversation, not an individual draft thread.

        Help Scout documents DELETE /v2/conversations/{id}, returning 204.
        This is not the UI trash operation; callers must not promise recovery.
        """
        return await self._mutation_result(
            "DELETE", f"/conversations/{conversation_id}", success_status=204,
        )

    async def create_conversation(
        self,
        mailbox_id: int,
        subject: str,
        customer: Dict[str, Any],
        text: str,
        tags: List[str] = None,
        user_id: int = None,
        draft: bool = True,
        status: str = "pending",
        attachments: List[Dict[str, str]] = None,
    ) -> Optional[int]:
        """Create a new outbound conversation with a single reply thread.

        `draft=True` leaves the reply as a draft for a human to send from the HelpScout UI;
        `draft=False` sends the email. `attachments`: files sent with the reply, each
        {"fileName", "mimeType", "data" (base64)} as the Mailbox API takes them on a new
        conversation's thread. Returns the new conversation id, or None on failure.
        """
        thread: Dict[str, Any] = {
            "type": "reply",
            "customer": customer,
            "text": text,
            "draft": bool(draft),
        }
        if user_id:
            thread["user"] = user_id
        if attachments:
            for item in attachments:
                if not (item.get("fileName") and item.get("mimeType") and item.get("data")):
                    raise ValueError("each attachment needs fileName, mimeType and data")
            thread["attachments"] = [
                {"fileName": a["fileName"], "mimeType": a["mimeType"], "data": a["data"]}
                for a in attachments
            ]

        payload: Dict[str, Any] = {
            "subject": subject,
            "customer": customer,
            "mailboxId": mailbox_id,
            "type": "email",
            "status": status,
            "tags": list(tags or []),
            "threads": [thread],
        }

        response = await self._request("POST", "/conversations", json=payload)
        if response.status_code not in (200, 201):
            return None

        headers = getattr(response, "headers", {}) or {}
        resource_id = headers.get("Resource-Id") or headers.get("Resource-ID") or headers.get("resource-id")
        if resource_id and str(resource_id).isdigit():
            return int(resource_id)

        location = headers.get("Location") or ""
        match = re.search(r"/conversations/(\d+)", location)
        if match:
            return int(match.group(1))

        try:
            return response.json().get("id")
        except Exception:
            return None

    async def list_conversations(
        self,
        query: str = None,
        status: str = "all",
        page: int = 1,
        page_size: int = 50,
        embed: str = "threads",
        tag: str = None,
    ) -> Dict[str, Any]:
        """Read one fixed 25-item Help Scout page.

        ``page_size`` remains in the signature for older callers but Help Scout v2
        ignores it, so it is deliberately not sent as an unsupported query parameter.
        Public limit/slicing semantics live in ``ConversationGateway``.
        """
        params = [("status", status), ("page", page)]
        if query:
            params.append(("query", query))
        if tag:
            params.append(("tag", tag))
        if embed:
            params.append(("embed", embed))

        endpoint = f"/conversations?{urlencode(params)}"
        response = await self._read_bounded(endpoint, max_bytes=_READ_JSON_MAX_BYTES)
        _raise_for_read_status(response, endpoint)
        data = _response_json_object(response, endpoint)
        conversations, pages = _conversation_list_parts(
            data,
            endpoint,
            requested_page=page,
        )
        return {
            "conversations": conversations,
            "pages": pages,
        }

    async def get_complete_conversation(self, conversation_id: int) -> Dict[str, Any]:
        """Fetch a conversation and replace its embedded threads with every thread page.

        Help Scout can leave an incomplete thread collection in the conversation embed.
        Merged conversations are represented as a single 301 to another Help Scout
        conversation URL; redirects are handled explicitly so credentials never follow an
        untrusted Location.
        """
        requested_id = conversation_id
        resolved_id = conversation_id
        conversation_endpoint = f"/conversations/{resolved_id}?embed=threads"
        response = await self._read_bounded(
            conversation_endpoint,
            max_bytes=_READ_JSON_MAX_BYTES,
        )

        if response.status_code == 301:
            resolved_id = _parse_merge_location(response.headers.get("Location"), requested_id, self.BASE_URL)
            conversation_endpoint = f"/conversations/{resolved_id}?embed=threads"
            try:
                response = await self._read_bounded(
                    conversation_endpoint,
                    max_bytes=_READ_JSON_MAX_BYTES,
                )
            except HelpScoutRedirectError as exc:
                raise HelpScoutRedirectError(
                    "Help Scout returned more than one merge redirect",
                    status=301,
                    code="merge_redirect_limit",
                    details={
                        "requested_conversation_id": requested_id,
                        "resolved_conversation_id": resolved_id,
                    },
                ) from exc
            if response.status_code == 301:
                raise HelpScoutRedirectError(
                    "Help Scout returned more than one merge redirect",
                    status=301,
                    code="merge_redirect_limit",
                    details={
                        "requested_conversation_id": requested_id,
                        "resolved_conversation_id": resolved_id,
                    },
                )

        _raise_for_read_status(response, conversation_endpoint)
        conversation = _response_json_object(response, conversation_endpoint)

        upstream_id = conversation.get("id")
        if (
            isinstance(upstream_id, bool)
            or not isinstance(upstream_id, (int, str))
            or not str(upstream_id).isdigit()
            or int(upstream_id) != resolved_id
        ):
            raise HelpScoutProtocolError(
                "Help Scout returned a conversation with an unexpected id",
                code="invalid_json",
                details={"endpoint": conversation_endpoint},
            )

        embedded = conversation.get("_embedded")
        if embedded is None:
            embedded = {}
            conversation["_embedded"] = embedded
        elif not isinstance(embedded, dict):
            raise HelpScoutProtocolError(
                "Help Scout returned an invalid conversation embed",
                code="invalid_json",
                details={"endpoint": conversation_endpoint},
            )

        first_endpoint = f"/conversations/{resolved_id}/threads?page=1"
        first_response = await self._read_bounded(
            first_endpoint,
            max_bytes=_READ_JSON_MAX_BYTES,
        )
        _raise_for_read_status(first_response, first_endpoint)
        first_payload = _response_json_object(first_response, first_endpoint)
        first_threads, first_page = _thread_page_parts(first_payload, first_endpoint)
        page_state = _first_page_state(first_page, first_threads, first_endpoint)
        total_pages = int(page_state["total_pages"] or 0)
        seen_thread_ids: set[int] = set()
        _record_thread_ids(first_threads, seen_thread_ids, first_endpoint)
        if len(first_threads) > _MAX_CONVERSATION_THREADS:
            raise HelpScoutProtocolError(
                "Help Scout returned too many conversation threads",
                code="too_many_threads",
                details={"endpoint": first_endpoint, "max_threads": _MAX_CONVERSATION_THREADS},
            )
        seen_page_fingerprints = {repr(first_threads)}
        all_threads = list(first_threads)

        for requested_page in range(2, total_pages + 1):
            endpoint = f"/conversations/{resolved_id}/threads?page={requested_page}"
            page_response = await self._read_bounded(
                endpoint,
                max_bytes=_READ_JSON_MAX_BYTES,
            )
            _raise_for_read_status(page_response, endpoint)
            payload = _response_json_object(page_response, endpoint)
            threads, page = _thread_page_parts(payload, endpoint)
            _validate_later_page(page, threads, endpoint, requested_page, page_state)
            _record_thread_ids(threads, seen_thread_ids, endpoint)
            if len(all_threads) + len(threads) > _MAX_CONVERSATION_THREADS:
                raise HelpScoutProtocolError(
                    "Help Scout returned too many conversation threads",
                    code="too_many_threads",
                    details={"endpoint": endpoint, "max_threads": _MAX_CONVERSATION_THREADS},
                )

            fingerprint = repr(threads)
            if fingerprint in seen_page_fingerprints:
                raise HelpScoutProtocolError(
                    "Help Scout repeated a thread page",
                    code="repeated_page",
                    details={"endpoint": endpoint},
                )
            seen_page_fingerprints.add(fingerprint)
            all_threads.extend(threads)

        total_elements = page_state["total_elements"]
        if total_elements is not None and len(all_threads) != total_elements:
            raise HelpScoutProtocolError(
                "Help Scout thread count does not match pagination metadata",
                code="invalid_pagination",
                details={"endpoint": first_endpoint},
            )

        embedded["threads"] = all_threads
        conversation["id"] = resolved_id
        conversation["requested_conversation_id"] = requested_id
        return conversation

    async def get_inline_image_data(self, url: str, *, max_bytes: int) -> bytes:
        """Bytes of an image pasted into a message body, from Help Scout's inline CDN only."""
        if not is_helpscout_inline_image_url(url):
            raise HelpScoutProtocolError("Not a Help Scout inline image URL", code="invalid_inline_image_url")
        if self.BASE_URL != mailbox_provider.VENDOR_API:
            key = hashlib.sha256(url.encode("utf-8")).hexdigest()
            endpoint = f"/inline/{key}"
            response = await self._read_bounded(endpoint, max_bytes=max_bytes)
            _raise_for_read_status(response, endpoint)
            return response.content
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise HelpScoutAttachmentTooLargeError(
                            "Inline image exceeds the byte limit",
                            code="attachment_too_large",
                            details={"max_bytes": max_bytes},
                        )
                    chunks.append(chunk)
        return b"".join(chunks)

    async def get_attachment_data(
        self,
        conversation_id: int,
        attachment_id: int,
        *,
        max_bytes: int,
    ) -> tuple[bytes, str | None]:
        """Fetch bounded bytes plus optional unverified, claimed MIME metadata."""
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise HelpScoutProtocolError(
                "Attachment byte limit must be a positive integer",
                code="invalid_max_bytes",
            )

        endpoint = (
            f"/conversations/{conversation_id}/attachments/{attachment_id}/data"
        )
        max_encoded_characters = 4 * ((max_bytes + 2) // 3)
        response = await self._read_bounded(
            endpoint,
            max_bytes=max_encoded_characters + _ATTACHMENT_JSON_OVERHEAD_BYTES,
        )
        _raise_for_read_status(response, endpoint)
        payload = _response_json_object(response, endpoint)
        encoded = payload.get("data")
        if not isinstance(encoded, str):
            raise HelpScoutProtocolError(
                "Help Scout returned invalid attachment data",
                code="invalid_attachment_data",
                details={"endpoint": endpoint},
            )

        if len(encoded) > max_encoded_characters:
            raise HelpScoutAttachmentTooLargeError(
                "Help Scout attachment exceeds the byte limit",
                code="attachment_too_large",
                details={
                    "max_bytes": max_bytes,
                    "encoded_characters": len(encoded),
                },
            )

        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
            raise HelpScoutProtocolError(
                "Help Scout returned invalid attachment data",
                code="invalid_attachment_data",
                details={"endpoint": endpoint},
            ) from exc

        if len(decoded) > max_bytes:
            raise HelpScoutAttachmentTooLargeError(
                "Help Scout attachment exceeds the byte limit",
                code="attachment_too_large",
                details={"max_bytes": max_bytes, "decoded_bytes": len(decoded)},
            )

        payload_mime_candidates = (
            payload.get("mimeType"),
            payload.get("mime_type"),
            payload.get("contentType"),
            payload.get("content_type"),
        )
        header_mime_candidates = (
            response.headers.get("X-Attachment-Content-Type"),
            response.headers.get("X-File-Content-Type"),
            response.headers.get("Content-Type"),
        )
        claimed_mime_type = next(
            (
                claimed_mime
                for candidate in payload_mime_candidates
                if (
                    claimed_mime := _normalize_claimed_attachment_mime(candidate)
                ) is not None
            ),
            None,
        )
        if claimed_mime_type is None:
            claimed_mime_type = next(
                (
                    claimed_mime
                    for candidate in header_mime_candidates
                    if (
                        claimed_mime := _normalize_claimed_attachment_mime(
                            candidate,
                            response_header=True,
                        )
                    ) is not None
                ),
                None,
            )
        return decoded, claimed_mime_type

    async def get_conversation(self, conversation_id: int) -> Dict[str, Any]:
        """Fetch a conversation by ID with embedded threads."""
        response = await self._request(
            "GET",
            f"/conversations/{conversation_id}?embed=threads"
        )
        response.raise_for_status()
        return response.json()

    async def get_recent_conversations(
        self,
        *,
        limit: int = 100,
        statuses: tuple[str, ...] = ("active", "pending"),
    ) -> List[Dict[str, Any]]:
        """Fetch recent conversations with embedded threads for reconciliation/recovery."""
        query_parts = [f"status={status}" for status in statuses]
        query_parts.extend([
            "embed=threads",
            "sortField=modifiedAt",
            "sortOrder=desc",
            f"pageSize={limit}",
        ])
        endpoint = f"/conversations?{'&'.join(query_parts)}"
        last_error: httpx.TimeoutException | None = None

        for attempt in range(2):
            try:
                response = await self._request("GET", endpoint)
                if response.status_code != 200:
                    return []

                data = response.json()
                return data.get("_embedded", {}).get("conversations", [])
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt == 0:
                    logger.warning("HelpScout recent conversation fetch timed out; retrying once")
                    continue
                raise

        if last_error is not None:
            raise last_error

        return []

    async def search_conversations(self, query: str, status: str = "all", limit: int = 5) -> List[Dict[str, Any]]:
        """Search mailbox query syntax and return conversation summaries without embedded threads."""
        import urllib.parse
        encoded_query = urllib.parse.quote(query)

        response = await self._request(
            "GET",
            f"/conversations?query={encoded_query}&status={status}&pageSize={limit}"
        )

        if response.status_code != 200:
            return []

        data = response.json()
        conversations = data.get("_embedded", {}).get("conversations", [])
        return conversations

    async def get_conversation_with_replies(self, conversation_id: int) -> Dict[str, Any]:
        """Get conversation with full thread details including agent replies."""
        conv = await self.get_conversation(conversation_id)
        threads = conv.get("_embedded", {}).get("threads", [])

        agent_replies = agent_replies_from_threads(threads)

        return {
            "id": conv.get("id"),
            "subject": conv.get("subject", ""),
            "status": conv.get("status", ""),
            "agent_replies": agent_replies
        }

    async def get_conversations_with_drafts(self, limit: int = 10, include_details: bool = False) -> List[Dict[str, Any]]:
        """Get active/pending conversations that have draft replies.

        If include_details=True, returns full conversation details including
        customer message and draft content.
        """
        # Fetch active and pending conversations
        response = await self._request(
            "GET",
            f"/conversations?status=active&status=pending&embed=threads&pageSize={limit}"
        )

        if response.status_code != 200:
            return []

        data = response.json()
        conversations = data.get("_embedded", {}).get("conversations", [])

        # Filter to those with drafts
        drafts = []
        for conv in conversations:
            threads = conv.get("_embedded", {}).get("threads", [])

            draft_thread = get_latest_draft_thread(threads)
            if not draft_thread:
                continue

            # A neutralized leftover from an already-sent reply is not pending work.
            if is_draft_superseded_by_reply(threads, draft_thread):
                continue

            # Detect theme from tags
            tags = [t.get("tag", "") for t in conv.get("tags", [])]
            theme = detect_theme(tags=tags, subject=conv.get("subject", ""), message="")

            # Customer info
            customer = conv.get("primaryCustomer") or conv.get("createdBy") or {}
            customer_name = " ".join(filter(None, [
                customer.get("first", ""),
                customer.get("last", "")
            ])) or customer.get("email", "Customer")
            customer_email = customer.get("email", "")

            draft_item = {
                "id": conv.get("id"),
                "subject": conv.get("subject", "No subject"),
                "theme": theme or "",
                "url": mailbox_provider.conversation_url(conv.get('id'))
            }

            if include_details:
                # Find customer message (most recent from customer)
                customer_thread = get_latest_customer_thread(threads)
                customer_message = strip_html(customer_thread.get("body", "")) if customer_thread else ""

                draft_item.update({
                    "customer_name": customer_name,
                    "customer_email": customer_email,
                    "customer_message": customer_message,
                    "draft_content": draft_thread.get("body", ""),
                    "draft_thread_id": draft_thread.get("id"),
                    "tags": [t.get("tag", "") for t in conv.get("tags", [])]
                })

            drafts.append(draft_item)

        return drafts

    async def get_follow_ups(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversations where customer replied after support already responded."""
        # Fetch active conversations with threads
        response = await self._request(
            "GET",
            f"/conversations?status=active&embed=threads&pageSize=50"
        )

        if response.status_code != 200:
            return []

        data = response.json()
        conversations = data.get("_embedded", {}).get("conversations", [])

        follow_ups = []
        for conv in conversations:
            threads = conv.get("_embedded", {}).get("threads", [])
            if not threads:
                continue

            latest_actionable_thread = get_latest_actionable_thread(threads)
            has_agent_reply = any(_is_agent_message_thread(thread) for thread in threads if not _is_draft_thread(thread))
            most_recent_is_customer = bool(latest_actionable_thread and _is_customer_thread(latest_actionable_thread))

            if most_recent_is_customer and has_agent_reply:
                tags = [t.get("tag", "") for t in conv.get("tags", [])]
                theme = detect_theme(tags=tags, subject=conv.get("subject", ""), message="")

                follow_ups.append({
                    "id": conv.get("id"),
                    "subject": conv.get("subject", "No subject"),
                    "theme": theme or "",
                    "url": mailbox_provider.conversation_url(conv.get('id'))
                })

                if len(follow_ups) >= limit:
                    break

        return follow_ups

    async def close(self):
        """Close HTTP client."""
        await self._http.aclose()
