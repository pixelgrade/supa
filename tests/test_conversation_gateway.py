import base64
import copy
import hashlib
import json
import warnings
from unittest.mock import AsyncMock

import pytest
from bs4 import MarkupResemblesLocatorWarning

import services.conversation_gateway as gateway_module
from services.conversation_gateway import (
    AttachmentData,
    ConversationGateway,
    ConversationGatewayError,
    InvalidCursorError,
    conversation_version,
    draft_version,
    normalize_draft_body,
    normalize_conversation,
)


def _attachment(
    attachment_id: int,
    filename: str,
    mime_type: str,
    size: int,
) -> dict:
    return {
        "id": attachment_id,
        "filename": filename,
        "mimeType": mime_type,
        "size": size,
        "_links": {
            "web": {
                "href": (
                    "https://api.helpscout.net/v2/conversations/321/"
                    f"attachments/{attachment_id}?access_token=secret"
                )
            }
        },
    }


def _thread(
    thread_id: int,
    *,
    thread_type: str,
    body: str,
    created_at: str | None,
    creator_type: str,
    state: str = "published",
    draft: bool = False,
    source_type: str = "email",
    attachments: list[dict] | None = None,
) -> dict:
    thread = {
        "id": thread_id,
        "type": thread_type,
        "state": state,
        "draft": draft,
        "body": body,
        "createdBy": {
            "id": thread_id + 1000,
            "type": creator_type,
            "first": creator_type.title(),
            "last": "Author",
            "email": f"{creator_type}@example.com",
            "photoUrl": "https://cdn.example/avatar-with-secret-token",
        },
        "source": {"type": source_type, "via": "helpdesk"},
        "_embedded": {"attachments": list(attachments or [])},
        "_links": {
            "self": {
                "href": f"https://api.helpscout.net/v2/threads/{thread_id}?token=secret"
            }
        },
    }
    if created_at is not None:
        thread["createdAt"] = created_at
    return thread


def _raw_conversation() -> dict:
    old_neutralized_draft = _thread(
        21,
        thread_type="message",
        body="<p>Manual review required before sending a reply.</p>",
        created_at="2026-09-20T09:30:00Z",
        creator_type="user",
        state="draft",
        draft=True,
        source_type="api",
    )
    first_agent_reply = _thread(
        30,
        thread_type="message",
        body="<p>We sent the real reply.</p>",
        created_at="2026-09-20T10:00:00Z",
        creator_type="user",
        source_type="api",
    )
    customer_reply = _thread(
        40,
        thread_type="customer",
        body="<p>It is <strong>still</strong> broken.<br>See files.</p>",
        created_at="2026-09-21T10:00:00Z",
        creator_type="customer",
        attachments=[
            _attachment(701, "screen.png", "image/png", 24_000),
            _attachment(702, "image001.png", "image/png", 1024),
            _attachment(703, "notes.pdf", "application/pdf", 1024),
        ],
    )
    # Identical timestamps deliberately arrive out of order. The greater id is newer.
    internal_note = _thread(
        51,
        thread_type="note",
        body="<p>Private operator instruction.</p>",
        created_at="2026-09-21T11:00:00Z",
        creator_type="user",
        source_type="web",
    )
    system_message = _thread(
        50,
        thread_type="lineitem",
        body="<p>Automation changed assignment.</p>",
        created_at="2026-09-21T11:00:00Z",
        creator_type="system_user",
        source_type="api",
    )
    live_draft = _thread(
        60,
        thread_type="message",
        body="<p>Fresh draft<br>second line</p>",
        created_at="2026-09-22T12:00:00Z",
        creator_type="user",
        state="draft",
        draft=True,
        source_type="api",
    )
    missing_timestamp = _thread(
        70,
        thread_type="lineitem",
        body="<p>Timestamp unavailable.</p>",
        created_at=None,
        creator_type="system_user",
        source_type="api",
    )
    return {
        "id": 321,
        "requested_conversation_id": 123,
        "number": 987,
        "subject": "Header issue",
        "status": "pending",
        "mailboxId": 42,
        "mailbox": {"id": 42, "name": "Support", "secret": "drop-me"},
        "assignee": {
            "id": 9,
            "type": "user",
            "first": "Operator",
            "last": "Oak",
            "email": "operator@example.com",
            "photoUrl": "https://cdn.example/private-avatar",
        },
        "primaryCustomer": {
            "id": 8,
            "type": "customer",
            "first": "Mica",
            "last": "Fern",
            "email": "mica@example.com",
            "photoUrl": "https://cdn.example/private-customer-avatar",
        },
        "tags": [
            {"id": 1, "tag": "field-journal", "color": "blue"},
            {"id": 2, "tag": "follow-up", "color": "red"},
        ],
        "createdAt": "2026-09-20T08:00:00Z",
        "updatedAt": "2026-09-22T12:00:00Z",
        "userUpdatedAt": "2026-09-22T11:30:00Z",
        "closedAt": None,
        "customerWaitingSince": {"time": "2026-09-21T10:00:00Z"},
        "_embedded": {
            "threads": [
                live_draft,
                system_message,
                old_neutralized_draft,
                missing_timestamp,
                customer_reply,
                first_agent_reply,
                internal_note,
            ]
        },
        "_links": {
            "self": {
                "href": "https://api.helpscout.net/v2/conversations/321?token=secret"
            },
            "web": {"href": "https://secure.helpscout.net/conversation/321"},
        },
    }


def _cursor(page: int, offset: int = 0) -> str:
    record = {"page": page, "v": 1}
    if offset:
        record["offset"] = offset
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":")
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _raw_conversation_with_id(conversation_id: int) -> dict:
    raw = copy.deepcopy(_raw_conversation())
    raw["id"] = conversation_id
    raw["requested_conversation_id"] = conversation_id
    raw["number"] = 10_000 + conversation_id
    return raw


def _raw_page(first_id: int, count: int = 25) -> list[dict]:
    return [_raw_conversation_with_id(value) for value in range(first_id, first_id + count)]


@pytest.mark.parametrize("body_format", [None, "", "markdown", "HTML", True])
def test_draft_body_requires_exact_html_or_text_format(body_format):
    with pytest.raises(ConversationGatewayError) as raised:
        normalize_draft_body("Hello", body_format)

    assert raised.value.code == "invalid_body_format"


def test_plain_text_draft_is_escaped_before_deterministic_paragraph_conversion():
    body = 'Hello <b> & "quoted"\r\nsecond line\n\nNext paragraph'

    normalized = normalize_draft_body(body, "text")

    assert normalized == (
        "<p>Hello &lt;b&gt; &amp; &quot;quoted&quot;<br>second line</p>"
        "<p>&nbsp;</p><p>Next paragraph</p>"
    )
    assert normalize_draft_body(body, "text") == normalized


def test_html_draft_removes_executable_markup_attributes_and_unsafe_links():
    body = (
        '<p onclick="steal()">Hello <strong>safe</strong>'
        '<script>alert(1)</script><a href="javascript:steal()">link</a></p>'
        '<iframe src="https://private.example"></iframe><p>Second</p>'
    )

    normalized = normalize_draft_body(body, "html")

    assert normalized == (
        "<p>Hello <strong>safe</strong>link</p>"
        "<p>&nbsp;</p><p>Second</p>"
    )
    assert "script" not in normalized.lower()
    assert "onclick" not in normalized.lower()
    assert "javascript:" not in normalized.lower()


def test_html_draft_preserves_only_canonical_safe_http_links():
    body = (
        '<p><a href="HTTPS://Example.COM/docs?q=one&amp;x=two#top" '
        'onclick="steal()" target="_blank">Docs</a> '
        '<a href="https://user:secret@example.com/private">credentials</a> '
        '<a href="data:text/html,bad">data</a></p>'
    )

    normalized = normalize_draft_body(body, "html")

    assert normalized == (
        '<p><a href="https://example.com/docs?q=one&amp;x=two#top" '
        'rel="noopener noreferrer">Docs</a> credentials data</p>'
    )
    assert "secret" not in normalized
    assert "target=" not in normalized


def test_html_draft_removes_declarations_processing_instructions_and_cdata():
    body = (
        '<!DOCTYPE html><?xml-stylesheet href="https://evil.example/style.css"?>'
        '<?agent instruction?><p>Safe</p><![CDATA[hidden executable-like text]]>'
    )

    normalized = normalize_draft_body(body, "html")

    assert normalized == "<p>Safe</p>"


@pytest.mark.parametrize("body_format", ["html", "text"])
@pytest.mark.parametrize(
    "control",
    ["\x00", "\x01", "\x1a", "\x7f", "\x80", "\x9f"],
)
def test_draft_body_rejects_disallowed_controls(body_format, control):
    with pytest.raises(ConversationGatewayError) as raised:
        normalize_draft_body(f"Hello{control}world", body_format)

    assert raised.value.code == "invalid_body"


@pytest.mark.parametrize("entity", ["&#1;", "&#x1a;", "&#127;"])
def test_html_draft_rejects_entities_that_decode_to_disallowed_controls(entity):
    with pytest.raises(ConversationGatewayError) as raised:
        normalize_draft_body(f"<p>Hello{entity}world</p>", "html")

    assert raised.value.code == "invalid_body"


def test_html_draft_maps_parser_failures_to_stable_invalid_body(monkeypatch):
    def fail_parser(_markup):
        raise RecursionError("private pathological parser detail")

    monkeypatch.setattr(gateway_module, "_draft_soup", fail_parser)

    with pytest.raises(ConversationGatewayError) as raised:
        normalize_draft_body("<p>Hello</p>", "html")

    assert raised.value.code == "invalid_body"
    assert "private pathological parser detail" not in str(raised.value)


def test_normalized_draft_body_has_a_byte_bound():
    with pytest.raises(ConversationGatewayError) as raised:
        normalize_draft_body("&" * 250_000, "text")

    assert raised.value.code == "body_too_large"
    assert raised.value.status == 413


def test_plain_url_html_does_not_emit_locator_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalized = normalize_draft_body("https://example.com/docs", "html")

    assert normalized == "https://example.com/docs"
    assert not any(
        issubclass(warning.category, MarkupResemblesLocatorWarning)
        for warning in caught
    )


def test_normalize_conversation_is_complete_deterministic_and_redacted():
    raw = _raw_conversation()

    result = normalize_conversation(raw)

    assert result["id"] == 321
    assert result["requested_id"] == 123
    assert result["resolved_id"] == 321
    assert result["was_merged"] is True
    assert result["number"] == 987
    assert result["subject"] == "Header issue"
    assert result["status"] == "pending"
    assert result["tags"] == ["field-journal", "follow-up"]
    assert result["mailbox"] == {"id": 42, "name": "Support"}
    assert result["assignee"] == {
        "id": 9,
        "type": "user",
        "name": "Operator Oak",
        "email": "operator@example.com",
    }
    assert result["customer"] == {
        "id": 8,
        "type": "customer",
        "name": "Mica Fern",
        "email": "mica@example.com",
    }
    assert result["timestamps"] == {
        "created_at": "2026-09-20T08:00:00Z",
        "updated_at": "2026-09-22T12:00:00Z",
        "user_updated_at": "2026-09-22T11:30:00Z",
        "closed_at": None,
        "customer_waiting_since": "2026-09-21T10:00:00Z",
    }
    assert result["help_scout_url"] == "https://secure.helpscout.net/conversation/321"
    assert result["waiting_on"] == "agent"
    assert result["latest_actionable_thread_id"] == 40
    assert [thread["id"] for thread in result["threads"]] == [21, 30, 40, 50, 51, 60, 70]

    customer = next(thread for thread in result["threads"] if thread["id"] == 40)
    assert customer["body_html"] == raw["_embedded"]["threads"][4]["body"]
    assert customer["body_text"] == "It is still broken. See files."
    assert customer["author"] == {
        "id": 1040,
        "type": "customer",
        "role": "customer",
        "name": "Customer Author",
        "email": "customer@example.com",
    }
    assert customer["source"] == {"type": "email", "via": "helpdesk"}
    assert customer["is_internal_note"] is False
    assert customer["attachments"] == [
        {
            "id": 701,
            "filename": "screen.png",
            "mime_type": "image/png",
            "size": 24_000,
            "is_image": True,
            "is_blank_transport_artifact": False,
        },
        {
            "id": 702,
            "filename": "image001.png",
            "mime_type": "image/png",
            "size": 1024,
            "is_image": True,
            "is_blank_transport_artifact": True,
        },
        {
            "id": 703,
            "filename": "notes.pdf",
            "mime_type": "application/pdf",
            "size": 1024,
            "is_image": False,
            "is_blank_transport_artifact": False,
        },
    ]
    note = next(thread for thread in result["threads"] if thread["id"] == 51)
    assert note["is_internal_note"] is True
    assert note["type"] == "note"
    assert note["author"]["role"] == "agent"
    system = next(thread for thread in result["threads"] if thread["id"] == 50)
    assert system["author"]["role"] == "system_user"

    assert result["draft"]["thread_id"] == 60
    assert result["draft"]["body_html"] == "<p>Fresh draft<br>second line</p>"
    assert result["draft"]["body_text"] == "Fresh draft second line"
    assert result["draft"]["version"] == draft_version(
        next(thread for thread in raw["_embedded"]["threads"] if thread["id"] == 60)
    )
    assert result["version"] == conversation_version(raw["_embedded"]["threads"])
    assert len(result["version"]) == 64
    assert len(result["draft"]["version"]) == 64
    int(result["version"], 16)
    int(result["draft"]["version"], 16)

    encoded = json.dumps(result, sort_keys=True)
    assert "access_token" not in encoded
    assert "token=secret" not in encoded
    assert "photoUrl" not in encoded
    assert "_links" not in encoded
    assert "api.helpscout.net" not in encoded


def test_normalization_order_and_versions_do_not_depend_on_upstream_order_or_links():
    raw = _raw_conversation()
    shuffled = copy.deepcopy(raw)
    shuffled["_embedded"]["threads"].reverse()
    shuffled["_links"]["self"]["href"] = "https://api.helpscout.net/new-secret"
    for index, thread in enumerate(shuffled["_embedded"]["threads"]):
        thread["_links"] = {"unrelated": {"href": f"https://private/{index}"}}

    first = normalize_conversation(raw)
    second = normalize_conversation(shuffled)

    assert first == second
    assert first["version"] == second["version"]


@pytest.mark.parametrize("field,value", [("body", "<p>Changed</p>"), ("state", "hidden")])
def test_conversation_version_changes_for_relevant_thread_mutations(field, value):
    threads = _raw_conversation()["_embedded"]["threads"]
    changed = copy.deepcopy(threads)
    changed[0][field] = value

    assert conversation_version(threads) != conversation_version(changed)


def test_draft_version_is_canonical_and_ignores_unrelated_upstream_links():
    draft = _thread(
        99,
        thread_type="message",
        body="<p>Draft</p>",
        created_at="2026-09-24T12:00:00Z",
        creator_type="user",
        state="draft",
        draft=True,
    )
    changed_link = copy.deepcopy(draft)
    changed_link["_links"] = {"self": {"href": "https://private.example/changed"}}
    changed_body = copy.deepcopy(draft)
    changed_body["body"] = "<p>New body</p>"

    assert draft_version(draft) == draft_version(changed_link)
    assert draft_version(draft) != draft_version(changed_body)
    assert draft_version(draft) == hashlib.sha256(
        json.dumps(
            {
                "body_html": "<p>Draft</p>",
                "created_at": "2026-09-24T12:00:00Z",
                "draft": True,
                "id": 99,
                "state": "draft",
                "type": "message",
                "updated_at": None,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def test_newest_published_agent_message_means_waiting_on_customer():
    raw = _raw_conversation()
    raw["_embedded"]["threads"].append(
        _thread(
            80,
            thread_type="message",
            body="<p>Please try that and let us know.</p>",
            created_at="2026-09-23T12:00:00Z",
            creator_type="user",
            source_type="api",
        )
    )

    assert normalize_conversation(raw)["waiting_on"] == "customer"


def test_newest_published_system_user_message_means_waiting_on_customer():
    raw = _raw_conversation()
    raw["_embedded"]["threads"].append(
        _thread(
            81,
            thread_type="message",
            body="<p>Automated but published Help Scout reply.</p>",
            created_at="2026-09-23T12:00:00Z",
            creator_type="system_user",
            source_type="api",
        )
    )

    result = normalize_conversation(raw)

    assert result["waiting_on"] == "customer"
    assert result["draft"] is None


def test_no_published_actionable_thread_means_waiting_on_unknown():
    raw = _raw_conversation()
    raw["_embedded"]["threads"] = [
        thread
        for thread in raw["_embedded"]["threads"]
        if thread["type"] not in {"customer", "message"} or thread["draft"]
    ]

    assert normalize_conversation(raw)["waiting_on"] == "unknown"


def test_null_created_by_is_ignored_without_crashing_normalization():
    raw = _raw_conversation()
    raw["_embedded"]["threads"].append(
        {
            "id": 82,
            "type": "message",
            "state": "published",
            "createdAt": "2026-09-24T12:00:00Z",
            "createdBy": None,
            "body": "<p>Malformed author metadata.</p>",
        }
    )

    result = normalize_conversation(raw)

    assert result["waiting_on"] == "agent"
    malformed = next(thread for thread in result["threads"] if thread["id"] == 82)
    assert malformed["author"]["role"] == "unknown"


def test_superseded_newest_draft_is_not_exposed_as_live():
    raw = _raw_conversation()
    raw["_embedded"]["threads"].append(
        _thread(
            90,
            thread_type="message",
            body="<p>Published after every draft.</p>",
            created_at="2026-09-23T13:00:00Z",
            creator_type="user",
            source_type="api",
        )
    )

    assert normalize_conversation(raw)["draft"] is None


def test_each_draft_thread_says_whether_a_later_published_reply_superseded_it():
    raw = _raw_conversation()

    result = normalize_conversation(raw)
    by_id = {thread["id"]: thread for thread in result["threads"]}

    # 21 is the neutralized leftover: reply 30 was published after it.
    assert by_id[21]["superseded_by_reply"] is True
    # 60 is the live draft: nothing was published after it.
    assert by_id[60]["superseded_by_reply"] is False
    # Not drafts: never superseded.
    for thread_id in (30, 40, 50, 51, 70):
        assert by_id[thread_id]["superseded_by_reply"] is False
    # Additive: the conversation version ignores the new field.
    assert result["version"] == conversation_version(raw["_embedded"]["threads"])


def test_live_draft_reads_superseded_once_a_reply_is_published_after_it():
    raw = _raw_conversation()
    raw["_embedded"]["threads"].append(
        _thread(
            90,
            thread_type="message",
            body="<p>Published after every draft.</p>",
            created_at="2026-09-23T13:00:00Z",
            creator_type="user",
            source_type="api",
        )
    )

    by_id = {thread["id"]: thread for thread in normalize_conversation(raw)["threads"]}

    assert by_id[60]["superseded_by_reply"] is True
    assert by_id[90]["superseded_by_reply"] is False
    # Earlier published threads are not drafts, so nothing supersedes them.
    assert by_id[30]["superseded_by_reply"] is False
    assert by_id[40]["superseded_by_reply"] is False
    assert by_id[51]["superseded_by_reply"] is False


def test_customer_waiting_since_is_null_once_our_published_reply_is_latest():
    raw = _raw_conversation()
    # Help Scout can keep customerWaitingSince after our reply went out.
    raw["_embedded"]["threads"].append(
        _thread(
            90,
            thread_type="message",
            body="<p>Our answer.</p>",
            created_at="2026-09-23T13:00:00Z",
            creator_type="user",
            source_type="api",
        )
    )

    result = normalize_conversation(raw)

    assert result["waiting_on"] == "customer"
    assert result["timestamps"]["customer_waiting_since"] is None


def test_customer_waiting_since_is_kept_while_the_customer_waits_or_state_is_unknown():
    waiting = normalize_conversation(_raw_conversation())
    assert waiting["waiting_on"] == "agent"
    assert waiting["timestamps"]["customer_waiting_since"] == "2026-09-21T10:00:00Z"

    unknown = normalize_conversation(
        {
            "id": 5,
            "customerWaitingSince": {"time": "2026-09-21T10:00:00Z"},
            "_embedded": {"threads": []},
        }
    )
    assert unknown["waiting_on"] == "unknown"
    assert unknown["timestamps"]["customer_waiting_since"] == "2026-09-21T10:00:00Z"


def test_identical_or_missing_timestamp_ties_are_deterministic_by_thread_id():
    first = _thread(
        100,
        thread_type="customer",
        body="<p>Lower id</p>",
        created_at=None,
        creator_type="customer",
    )
    second = _thread(
        101,
        thread_type="customer",
        body="<p>Higher id</p>",
        created_at=None,
        creator_type="customer",
    )
    raw = {
        "id": 1,
        "requested_conversation_id": 1,
        "_embedded": {"threads": [second, first]},
    }

    result = normalize_conversation(raw)

    assert [thread["id"] for thread in result["threads"]] == [100, 101]
    assert result["waiting_on"] == "agent"
    assert result == normalize_conversation(
        {**raw, "_embedded": {"threads": [first, second]}}
    )


@pytest.mark.asyncio
async def test_gateway_get_fetches_complete_conversation_then_normalizes():
    client = AsyncMock()
    client.get_complete_conversation.return_value = _raw_conversation()
    gateway = ConversationGateway(client)

    result = await gateway.get(123)

    client.get_complete_conversation.assert_awaited_once_with(123)
    assert result == normalize_conversation(_raw_conversation())


@pytest.mark.asyncio
async def test_gateway_list_passes_separate_filters_and_uses_opaque_cursor():
    client = AsyncMock()
    client.list_conversations.return_value = {
        "conversations": _raw_page(26),
        "pages": 3,
    }
    gateway = ConversationGateway(client)

    result = await gateway.list(
        status="pending",
        tag="needs review & follow-up",
        query='subject:"Field Journal"',
        cursor=_cursor(2),
        limit=25,
    )

    client.list_conversations.assert_awaited_once_with(
        query='subject:"Field Journal"',
        status="pending",
        tag="needs review & follow-up",
        page=2,
        embed="threads",
    )
    assert result == {
        "conversations": [normalize_conversation(raw) for raw in _raw_page(26)],
        "next_cursor": _cursor(3),
    }
    assert "page" not in result
    assert "page" not in result["next_cursor"]


@pytest.mark.asyncio
async def test_gateway_limit_one_slices_one_fixed_upstream_page_without_skipping():
    client = AsyncMock()
    client.list_conversations.return_value = {
        "conversations": _raw_page(1),
        "pages": 2,
    }
    gateway = ConversationGateway(client)

    result = await gateway.list(
        status="all", tag=None, query=None, cursor=None, limit=1
    )

    assert [item["id"] for item in result["conversations"]] == [1]
    assert result["next_cursor"] == _cursor(1, 1)
    assert client.list_conversations.await_count == 1


@pytest.mark.asyncio
async def test_gateway_crosses_fixed_pages_from_offset_and_resumes_first_unconsumed():
    client = AsyncMock()

    async def list_page(**kwargs):
        page = kwargs["page"]
        return {"conversations": _raw_page((page - 1) * 25 + 1), "pages": 3}

    client.list_conversations.side_effect = list_page
    gateway = ConversationGateway(client)

    result = await gateway.list(
        status="all", tag=None, query=None, cursor=_cursor(1, 20), limit=10
    )

    assert [item["id"] for item in result["conversations"]] == list(range(21, 31))
    assert result["next_cursor"] == _cursor(2, 5)
    assert [call.kwargs["page"] for call in client.list_conversations.await_args_list] == [1, 2]


@pytest.mark.asyncio
async def test_gateway_exact_page_boundary_returns_next_page_cursor_without_extra_fetch():
    client = AsyncMock()
    client.list_conversations.return_value = {
        "conversations": _raw_page(1),
        "pages": 2,
    }
    gateway = ConversationGateway(client)

    result = await gateway.list(
        status="all", tag=None, query=None, cursor=_cursor(1, 20), limit=5
    )

    assert [item["id"] for item in result["conversations"]] == list(range(21, 26))
    assert result["next_cursor"] == _cursor(2)
    assert client.list_conversations.await_count == 1


@pytest.mark.asyncio
async def test_gateway_short_last_page_returns_no_cursor():
    client = AsyncMock()
    client.list_conversations.return_value = {
        "conversations": _raw_page(26, 5),
        "pages": 2,
    }
    gateway = ConversationGateway(client)

    result = await gateway.list(
        status="all", tag=None, query=None, cursor=_cursor(2), limit=100
    )

    assert [item["id"] for item in result["conversations"]] == list(range(26, 31))
    assert result["next_cursor"] is None
    assert client.list_conversations.await_count == 1


@pytest.mark.asyncio
async def test_gateway_max_limit_fetches_at_most_five_fixed_pages_from_mid_page():
    client = AsyncMock()

    async def list_page(**kwargs):
        page = kwargs["page"]
        return {"conversations": _raw_page((page - 1) * 25 + 1), "pages": 6}

    client.list_conversations.side_effect = list_page
    gateway = ConversationGateway(client)

    result = await gateway.list(
        status="all", tag=None, query=None, cursor=_cursor(1, 24), limit=100
    )

    assert [item["id"] for item in result["conversations"]] == list(range(25, 125))
    assert result["next_cursor"] == _cursor(5, 24)
    assert client.list_conversations.await_count == 5


@pytest.mark.asyncio
async def test_gateway_cursor_sequence_has_no_skips_or_duplicates():
    client = AsyncMock()

    async def list_page(**kwargs):
        page = kwargs["page"]
        if page == 1:
            return {"conversations": _raw_page(1), "pages": 3}
        if page == 2:
            return {"conversations": _raw_page(26), "pages": 3}
        return {"conversations": _raw_page(51, 3), "pages": 3}

    client.list_conversations.side_effect = list_page
    gateway = ConversationGateway(client)
    cursor = None
    seen: list[int] = []

    while True:
        result = await gateway.list(
            status="all", tag=None, query=None, cursor=cursor, limit=7
        )
        seen.extend(item["id"] for item in result["conversations"])
        cursor = result["next_cursor"]
        if cursor is None:
            break

    assert seen == list(range(1, 54))
    assert len(seen) == len(set(seen))


@pytest.mark.asyncio
async def test_gateway_rejects_stale_offset_past_short_page():
    client = AsyncMock()
    client.list_conversations.return_value = {
        "conversations": _raw_page(1, 10),
        "pages": 1,
    }
    gateway = ConversationGateway(client)

    with pytest.raises(InvalidCursorError) as raised:
        await gateway.list(
            status="all", tag=None, query=None, cursor=_cursor(1, 20), limit=1
        )

    assert raised.value.code == "invalid_cursor"


@pytest.mark.asyncio
async def test_gateway_rejects_oversized_fixed_upstream_page():
    client = AsyncMock()
    client.list_conversations.return_value = {
        "conversations": _raw_page(1, 26),
        "pages": 1,
    }
    gateway = ConversationGateway(client)

    with pytest.raises(ConversationGatewayError) as raised:
        await gateway.list(
            status="all", tag=None, query=None, cursor=None, limit=100
        )

    assert raised.value.code == "invalid_upstream_pagination"


@pytest.mark.asyncio
async def test_gateway_list_has_no_next_cursor_on_last_or_invalid_upstream_page_count():
    client = AsyncMock()
    gateway = ConversationGateway(client)
    for pages in (1, 0, None, "invalid"):
        client.list_conversations.return_value = {
            "conversations": [],
            "pages": pages,
        }
        result = await gateway.list(
            status="all", tag=None, query=None, cursor=None, limit=50
        )
        assert result == {"conversations": [], "next_cursor": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.5, "50"])
async def test_gateway_list_rejects_unbounded_or_non_integer_limits(limit):
    gateway = ConversationGateway(AsyncMock())

    with pytest.raises(ConversationGatewayError) as raised:
        await gateway.list(
            status="all", tag=None, query=None, cursor=None, limit=limit
        )

    assert raised.value.code == "invalid_limit"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cursor",
    [
        "not-base64!",
        "e30",  # {}
        base64.urlsafe_b64encode(b'{"v":1,"page":0}').decode().rstrip("="),
        base64.urlsafe_b64encode(b'{"v":2,"page":2}').decode().rstrip("="),
        base64.urlsafe_b64encode(b'{"v":1,"page":2,"extra":true}').decode().rstrip("="),
        base64.urlsafe_b64encode(b'{"offset":0,"page":2,"v":1}').decode().rstrip("="),
        base64.urlsafe_b64encode(b'{"offset":25,"page":2,"v":1}').decode().rstrip("="),
        base64.urlsafe_b64encode(b'{"offset":-1,"page":2,"v":1}').decode().rstrip("="),
        base64.urlsafe_b64encode(b'{"offset":true,"page":2,"v":1}').decode().rstrip("="),
        base64.urlsafe_b64encode(b'{"offset":"1","page":2,"v":1}').decode().rstrip("="),
        base64.urlsafe_b64encode(b'{"page":2,"v":1}').decode(),  # padded/noncanonical
        base64.urlsafe_b64encode(b'{ "page":2, "v":1 }').decode().rstrip("="),
    ],
)
async def test_gateway_list_rejects_malformed_unsupported_or_noncanonical_cursor(cursor):
    gateway = ConversationGateway(AsyncMock())

    with pytest.raises(InvalidCursorError) as raised:
        await gateway.list(
            status="all", tag=None, query=None, cursor=cursor, limit=50
        )

    assert raised.value.code == "invalid_cursor"
    assert raised.value.retryable is False
    assert cursor not in str(raised.value)


@pytest.mark.asyncio
async def test_gateway_attachment_uses_resolved_id_and_returns_safe_metadata():
    raw = _raw_conversation()
    client = AsyncMock()
    client.get_complete_conversation.return_value = raw
    client.get_attachment_data.return_value = (b"png-data", "image/png")
    gateway = ConversationGateway(client, attachment_max_bytes=4 * 1024 * 1024)

    result = await gateway.attachment(123, 701)

    assert result == AttachmentData(
        conversation_id=321,
        attachment_id=701,
        filename="screen.png",
        data=b"png-data",
        claimed_mime_type="image/png",
        size=8,
    )
    client.get_complete_conversation.assert_awaited_once_with(123)
    client.get_attachment_data.assert_awaited_once_with(
        321, 701, max_bytes=4 * 1024 * 1024
    )


@pytest.mark.asyncio
async def test_gateway_attachment_falls_back_to_safe_metadata_mime_and_filename():
    raw = _raw_conversation()
    attachment = raw["_embedded"]["threads"][4]["_embedded"]["attachments"][0]
    attachment["filename"] = ""
    client = AsyncMock()
    client.get_complete_conversation.return_value = raw
    client.get_attachment_data.return_value = (b"data", None)
    gateway = ConversationGateway(client)

    result = await gateway.attachment(123, 701)

    assert result.filename == "attachment-701"
    assert result.claimed_mime_type == "image/png"


@pytest.mark.asyncio
async def test_gateway_attachment_rejects_an_id_not_present_in_complete_conversation():
    client = AsyncMock()
    client.get_complete_conversation.return_value = _raw_conversation()
    gateway = ConversationGateway(client)

    with pytest.raises(ConversationGatewayError) as raised:
        await gateway.attachment(123, 999)

    assert raised.value.code == "attachment_not_found"
    client.get_attachment_data.assert_not_awaited()
