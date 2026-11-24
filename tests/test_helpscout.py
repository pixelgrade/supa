import base64
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
import json

import httpx
import pytest
from types import SimpleNamespace

from services.helpscout import (
    HelpScoutAttachmentTooLargeError,
    HelpScoutAuthError,
    HelpScoutClient,
    HelpScoutProtocolError,
    HelpScoutRedirectError,
    HelpScoutResponseTooLargeError,
    HelpScoutTransportError,
    HelpScoutUpstreamError,
    MutationResult,
    _is_agent_message_thread,
    get_draft_thread_ids,
    is_draft_superseded_by_reply,
    parse_api_conversation,
    parse_webhook_payload,
)
from services.themes import detect_theme


def test_mutation_result_is_frozen_and_has_safe_defaults():
    result = MutationResult(ok=False, status_code=500)

    assert result.resource_id is None
    assert result.outcome_unknown is False
    assert result.detail is None
    with pytest.raises(FrozenInstanceError):
        result.ok = True


async def _client_with_transport(handler, *, authenticated=True) -> HelpScoutClient:
    client = HelpScoutClient(client_id="id", client_secret="secret")
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    if authenticated:
        client._access_token = "test-token"
        client._token_expires = datetime.now() + timedelta(hours=1)
    return client


def _conversation_payload(conversation_id, *, embedded_threads=None):
    return {
        "id": conversation_id,
        "subject": "Complete read",
        "_embedded": {"threads": list(embedded_threads or [])},
    }


def _thread_page(
    threads,
    *,
    number=None,
    total_pages=None,
    size=None,
    total_elements=None,
):
    payload = {"_embedded": {"threads": threads}}
    if any(value is not None for value in (number, total_pages, size, total_elements)):
        payload["page"] = {}
        if number is not None:
            payload["page"]["number"] = number
        if total_pages is not None:
            payload["page"]["totalPages"] = total_pages
        if size is not None:
            payload["page"]["size"] = size
        if total_elements is not None:
            payload["page"]["totalElements"] = total_elements
    return payload


class _TrackingAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.yielded = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.yielded += 1
            yield chunk

    async def aclose(self):
        self.closed = True


class _ExplodingMutationBody(httpx.AsyncByteStream):
    def __init__(self, request):
        self.request = request
        self.read_attempted = False
        self.closed = False

    async def __aiter__(self):
        self.read_attempted = True
        raise httpx.ReadError("private response body failed", request=self.request)
        yield b""  # pragma: no cover - keeps this an async iterator

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_get_complete_conversation_replaces_partial_embedded_threads():
    requested_urls = []

    def handler(request):
        requested_urls.append(str(request.url))
        assert request.headers["Authorization"] == "Bearer test-token"
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(
                200,
                json=_conversation_payload(123, embedded_threads=[{"id": 999}]),
            )
        return httpx.Response(
            200,
            json=_thread_page([{"id": 1}, {"id": 2}], number=1, total_pages=1),
        )

    client = await _client_with_transport(handler)
    try:
        result = await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert result["id"] == 123
    assert result["requested_conversation_id"] == 123
    assert result["_embedded"]["threads"] == [{"id": 1}, {"id": 2}]
    assert requested_urls == [
        "https://api.helpscout.net/v2/conversations/123?embed=threads",
        "https://api.helpscout.net/v2/conversations/123/threads?page=1",
    ]


@pytest.mark.asyncio
async def test_get_complete_conversation_fetches_every_one_based_thread_page_once():
    requested_urls = []

    def handler(request):
        requested_urls.append(str(request.url))
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(200, json=_conversation_payload(123))
        requested_page = int(request.url.params["page"])
        return httpx.Response(
            200,
            json=_thread_page(
                [{"id": requested_page}],
                number=requested_page,
                total_pages=3,
                size=1,
                total_elements=3,
            ),
        )

    client = await _client_with_transport(handler)
    try:
        result = await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert result["_embedded"]["threads"] == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert requested_urls == [
        "https://api.helpscout.net/v2/conversations/123?embed=threads",
        "https://api.helpscout.net/v2/conversations/123/threads?page=1",
        "https://api.helpscout.net/v2/conversations/123/threads?page=2",
        "https://api.helpscout.net/v2/conversations/123/threads?page=3",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("page_size", "total_pages"),
    [(1, 100), (7, 15), (25, 4), (100, 1)],
)
async def test_get_complete_conversation_allows_variable_pages_up_to_100_threads(
    page_size,
    total_pages,
):
    requested_pages = []

    def handler(request):
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(200, json=_conversation_payload(123))
        page_number = int(request.url.params["page"])
        requested_pages.append(page_number)
        start = (page_number - 1) * page_size
        stop = min(start + page_size, 100)
        return httpx.Response(
            200,
            json=_thread_page(
                [{"id": thread_id} for thread_id in range(start + 1, stop + 1)],
                number=page_number,
                total_pages=total_pages,
                size=page_size,
                total_elements=100,
            ),
        )

    client = await _client_with_transport(handler)
    try:
        result = await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert len(result["_embedded"]["threads"]) == 100
    assert requested_pages == list(range(1, total_pages + 1))


@pytest.mark.asyncio
async def test_get_complete_conversation_rejects_more_than_1000_reported_threads():
    request_count = 0

    def handler(request):
        nonlocal request_count
        request_count += 1
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(200, json=_conversation_payload(123))
        return httpx.Response(
            200,
            json=_thread_page(
                [{"id": thread_id} for thread_id in range(1, 1001)],
                number=1,
                total_pages=2,
                size=1000,
                total_elements=1001,
            ),
        )

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutProtocolError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.code == "invalid_pagination"
    assert request_count == 2


@pytest.mark.asyncio
async def test_get_complete_conversation_rejects_more_than_1000_accumulated_threads():
    def handler(request):
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(200, json=_conversation_payload(123))
        page_number = int(request.url.params["page"])
        start = (page_number - 1) * 1000
        stop = 1000 if page_number == 1 else 1001
        return httpx.Response(
            200,
            json=_thread_page(
                [{"id": thread_id} for thread_id in range(start + 1, stop + 1)],
                number=page_number,
                total_pages=2,
            ),
        )

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutProtocolError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.code == "too_many_threads"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "second_value"),
    [("size", 25), ("totalPages", 3), ("totalElements", 99)],
)
async def test_get_complete_conversation_rejects_inconsistent_page_metadata(
    field,
    second_value,
):
    def handler(request):
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(200, json=_conversation_payload(123))
        page_number = int(request.url.params["page"])
        metadata = {"size": 50, "totalPages": 2, "totalElements": 100}
        if page_number == 2:
            metadata[field] = second_value
        start = (page_number - 1) * 50
        return httpx.Response(
            200,
            json=_thread_page(
                [{"id": thread_id} for thread_id in range(start + 1, start + 51)],
                number=page_number,
                total_pages=metadata["totalPages"],
                size=metadata["size"],
                total_elements=metadata["totalElements"],
            ),
        )

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutProtocolError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.code == "invalid_pagination"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("threads", "number", "total_pages", "total_elements"),
    [
        ([{"id": 1}], 0, 1, 1),
        ([{"id": 1}], 0, 0, 0),
        ([], 0, 0, 1),
    ],
)
async def test_get_complete_conversation_only_accepts_zero_page_as_empty_sentinel(
    threads,
    number,
    total_pages,
    total_elements,
):
    def handler(request):
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(200, json=_conversation_payload(123))
        return httpx.Response(
            200,
            json=_thread_page(
                threads,
                number=number,
                total_pages=total_pages,
                size=25,
                total_elements=total_elements,
            ),
        )

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutProtocolError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.code == "invalid_pagination"


@pytest.mark.asyncio
async def test_get_complete_conversation_requires_later_one_based_page_number():
    def handler(request):
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(200, json=_conversation_payload(123))
        requested_page = int(request.url.params["page"])
        reported_page = 4 if requested_page == 2 else requested_page
        return httpx.Response(
            200,
            json=_thread_page(
                [{"id": requested_page}],
                number=reported_page,
                total_pages=2,
                size=1,
                total_elements=2,
            ),
        )

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutProtocolError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.code == "invalid_pagination"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "page_metadata",
    [
        {},
        {"number": 0, "totalPages": 0, "totalElements": 0},
    ],
)
async def test_get_complete_conversation_treats_empty_or_zero_page_metadata_as_one_page(
    page_metadata,
):
    requested_urls = []

    def handler(request):
        requested_urls.append(str(request.url))
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(200, json=_conversation_payload(123))
        return httpx.Response(
            200,
            json={"_embedded": {"threads": []}, "page": page_metadata},
        )

    client = await _client_with_transport(handler)
    try:
        result = await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert result["_embedded"]["threads"] == []
    assert requested_urls[-1].endswith("/conversations/123/threads?page=1")
    assert len(requested_urls) == 2


@pytest.mark.asyncio
async def test_get_complete_conversation_follows_one_official_merge_redirect():
    requested_urls = []

    def handler(request):
        requested_urls.append(str(request.url))
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(
                301,
                headers={"Location": "https://api.helpscout.net/v2/conversations/456"},
            )
        if request.url.path == "/v2/conversations/456":
            return httpx.Response(200, json=_conversation_payload(456))
        return httpx.Response(
            200,
            json=_thread_page([{"id": 7}], number=1, total_pages=1),
        )

    client = await _client_with_transport(handler)
    try:
        result = await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert result["requested_conversation_id"] == 123
    assert result["id"] == 456
    assert requested_urls == [
        "https://api.helpscout.net/v2/conversations/123?embed=threads",
        "https://api.helpscout.net/v2/conversations/456?embed=threads",
        "https://api.helpscout.net/v2/conversations/456/threads?page=1",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location",
    [
        None,
        "javascript:alert(1)",
        "https://evil.example/v2/conversations/456",
        "https://api.helpscout.net.evil.example/v2/conversations/456",
        "https://api.helpscout.net@evil.example/v2/conversations/456",
        "https://api.helpscout.net/v2/conversations/123",
    ],
)
async def test_get_complete_conversation_rejects_invalid_merge_redirects(location):
    request_count = 0

    def handler(_request):
        nonlocal request_count
        request_count += 1
        headers = {"Location": location} if location is not None else {}
        return httpx.Response(301, headers=headers)

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutRedirectError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.status == 301
    assert raised.value.code == "invalid_merge_redirect"
    assert raised.value.retryable is False
    assert request_count == 1


@pytest.mark.asyncio
async def test_get_complete_conversation_rejects_a_second_merge_redirect():
    def handler(request):
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(
                301,
                headers={"Location": "https://api.helpscout.net/v2/conversations/456"},
            )
        return httpx.Response(
            301,
            headers={"Location": "https://api.helpscout.net/v2/conversations/789"},
        )

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutRedirectError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.code == "merge_redirect_limit"
    assert raised.value.details == {
        "requested_conversation_id": 123,
        "resolved_conversation_id": 456,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 429, 500, 503])
async def test_get_complete_conversation_preserves_upstream_http_status(status):
    def handler(_request):
        return httpx.Response(status, text="secret upstream response")

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutUpstreamError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.status == status
    assert raised.value.code == "upstream_http_error"
    assert raised.value.retryable is (status == 429 or status >= 500)
    assert "secret upstream response" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_get_complete_conversation_types_resource_authorization_failures(status):
    def handler(_request):
        return httpx.Response(status, text="secret authorization response")

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutAuthError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert not isinstance(raised.value, HelpScoutUpstreamError)
    assert raised.value.status == status
    assert raised.value.code == "upstream_auth_error"
    assert raised.value.retryable is False
    assert raised.value.details == {
        "endpoint": "/conversations/123?embed=threads",
    }
    assert "secret authorization response" not in str(raised.value)
    assert "secret authorization response" not in repr(raised.value.details)


@pytest.mark.asyncio
async def test_get_complete_conversation_preserves_thread_page_http_failure():
    def handler(request):
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(200, json=_conversation_payload(123))
        return httpx.Response(502, text="private thread error")

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutUpstreamError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.status == 502
    assert raised.value.retryable is True
    assert "private thread error" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("total_pages", [-1, 101, "2", True])
async def test_get_complete_conversation_rejects_invalid_thread_page_bounds(total_pages):
    def handler(request):
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(200, json=_conversation_payload(123))
        return httpx.Response(
            200,
            json=_thread_page([], number=1, total_pages=total_pages),
        )

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutProtocolError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.code == "invalid_pagination"


@pytest.mark.asyncio
async def test_get_complete_conversation_rejects_repeated_thread_page():
    def handler(request):
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(200, json=_conversation_payload(123))
        requested_page = int(request.url.params["page"])
        number = 1 if requested_page in (1, 2) else requested_page
        return httpx.Response(
            200,
            json=_thread_page([{"id": requested_page}], number=number, total_pages=2),
        )

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutProtocolError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.code == "repeated_page"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pages",
    [
        [[{"id": 1}, {"id": 1}]],
        [[{"id": 1}], [{"id": 1}]],
    ],
)
async def test_get_complete_conversation_rejects_duplicate_thread_ids(pages):
    def handler(request):
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(200, json=_conversation_payload(123))
        page_number = int(request.url.params["page"])
        return httpx.Response(
            200,
            json=_thread_page(
                pages[page_number - 1],
                number=page_number,
                total_pages=len(pages),
                size=max(len(page) for page in pages),
                total_elements=sum(len(page) for page in pages),
            ),
        )

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutProtocolError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.code == "duplicate_thread"


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id", [None, "9" * 128])
async def test_get_complete_conversation_rejects_thread_without_usable_id(thread_id):
    def handler(request):
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(200, json=_conversation_payload(123))
        thread = {"type": "customer"}
        if thread_id is not None:
            thread["id"] = thread_id
        return httpx.Response(
            200,
            json=_thread_page(
                [thread],
                number=1,
                total_pages=1,
                size=25,
                total_elements=1,
            ),
        )

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutProtocolError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.code == "invalid_thread_id"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"{"),
        httpx.Response(200, json=[{"id": 123}]),
    ],
)
async def test_get_complete_conversation_rejects_invalid_conversation_json_shape(response):
    client = await _client_with_transport(lambda _request: response)
    try:
        with pytest.raises(HelpScoutProtocolError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.code == "invalid_json"


@pytest.mark.asyncio
async def test_new_reads_use_streaming_transport_with_existing_auth(monkeypatch):
    authorization = []

    def handler(request):
        authorization.append(request.headers.get("Authorization"))
        return httpx.Response(200, json={"data": "ZmlsZQ=="})

    client = await _client_with_transport(handler)

    async def forbidden_buffered_request(*_args, **_kwargs):
        raise AssertionError("new reads must not call buffered _request")

    monkeypatch.setattr(client, "_request", forbidden_buffered_request)
    try:
        data, claimed_mime_type = await client.get_attachment_data(123, 456, max_bytes=4)
    finally:
        await client.close()

    assert data == b"file"
    assert claimed_mime_type is None
    assert authorization == ["Bearer test-token"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (httpx.ReadTimeout("slow upstream"), "request_timeout"),
        (httpx.ConnectError("network unavailable"), "network_error"),
    ],
)
async def test_new_reads_wrap_transport_failures(failure, expected_code):
    def handler(request):
        failure.request = request
        raise failure

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutTransportError) as raised:
            await client.get_attachment_data(1, 2, max_bytes=100)
    finally:
        await client.close()

    assert raised.value.status is None
    assert raised.value.code == expected_code
    assert raised.value.retryable is True
    assert raised.value.__cause__ is failure
    assert raised.value.details == {
        "endpoint": "/conversations/1/attachments/2/data",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 500])
async def test_new_reads_wrap_token_endpoint_http_failures(status):
    def handler(request):
        assert request.url.path == "/v2/oauth2/token"
        return httpx.Response(status, text="secret oauth response")

    client = await _client_with_transport(handler, authenticated=False)
    try:
        with pytest.raises(HelpScoutAuthError) as raised:
            await client.get_attachment_data(1, 2, max_bytes=100)
    finally:
        await client.close()

    assert raised.value.status == status
    assert raised.value.code == "oauth_token_error"
    assert raised.value.retryable is (status == 429 or status >= 500)
    assert raised.value.details == {"operation": "oauth_token"}
    assert isinstance(raised.value.__cause__, httpx.HTTPStatusError)
    assert "secret oauth response" not in str(raised.value)
    assert "secret oauth response" not in repr(raised.value.details)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token_response",
    [
        httpx.Response(200, content=b"{", headers={"Content-Type": "application/json"}),
        httpx.Response(200, json={"expires_in": 7200}),
    ],
)
async def test_new_reads_wrap_invalid_token_responses(token_response):
    client = await _client_with_transport(
        lambda _request: token_response,
        authenticated=False,
    )
    try:
        with pytest.raises(HelpScoutAuthError) as raised:
            await client.get_attachment_data(1, 2, max_bytes=100)
    finally:
        await client.close()

    assert raised.value.status == 200
    assert raised.value.code == "invalid_oauth_token_response"
    assert raised.value.retryable is False
    assert raised.value.details == {"operation": "oauth_token"}
    assert raised.value.__cause__ is not None


@pytest.mark.asyncio
async def test_attachment_content_length_over_cap_rejects_without_reading_body():
    stream = _TrackingAsyncStream([b"must not be consumed"])

    def handler(_request):
        return httpx.Response(
            200,
            headers={"Content-Length": "9999999"},
            stream=stream,
        )

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutResponseTooLargeError) as raised:
            await client.get_attachment_data(1, 2, max_bytes=1)
    finally:
        await client.close()

    assert raised.value.code == "response_too_large"
    assert stream.yielded == 0
    assert stream.closed is True


@pytest.mark.asyncio
async def test_attachment_stream_cap_stops_before_consuming_remainder():
    secret_remainder = b"secret remainder must stay unread"
    stream = _TrackingAsyncStream(
        [b'{"data":"', b"A" * 5000, secret_remainder]
    )

    client = await _client_with_transport(
        lambda _request: httpx.Response(200, stream=stream)
    )
    try:
        with pytest.raises(HelpScoutResponseTooLargeError) as raised:
            await client.get_attachment_data(1, 2, max_bytes=1)
    finally:
        await client.close()

    assert raised.value.code == "response_too_large"
    assert stream.yielded == 2
    assert stream.closed is True
    assert secret_remainder.decode() not in str(raised.value)
    assert secret_remainder.decode() not in repr(raised.value.details)


@pytest.mark.asyncio
async def test_conversation_content_length_over_general_cap_is_not_read():
    stream = _TrackingAsyncStream([b'{"id":123}'])

    client = await _client_with_transport(
        lambda _request: httpx.Response(
            200,
            headers={"Content-Length": "50000000"},
            stream=stream,
        )
    )
    try:
        with pytest.raises(HelpScoutResponseTooLargeError) as raised:
            await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert raised.value.code == "response_too_large"
    assert stream.yielded == 0
    assert stream.closed is True


@pytest.mark.asyncio
async def test_merge_redirect_body_is_never_read():
    redirect_stream = _TrackingAsyncStream([b"merge response body must remain unread"])

    def handler(request):
        if request.url.path == "/v2/conversations/123":
            return httpx.Response(
                301,
                headers={"Location": "https://api.helpscout.net/v2/conversations/456"},
                stream=redirect_stream,
            )
        if request.url.path == "/v2/conversations/456":
            return httpx.Response(200, json=_conversation_payload(456))
        return httpx.Response(
            200,
            json=_thread_page(
                [{"id": 7}],
                number=1,
                total_pages=1,
                size=25,
                total_elements=1,
            ),
        )

    client = await _client_with_transport(handler)
    try:
        result = await client.get_complete_conversation(123)
    finally:
        await client.close()

    assert result["id"] == 456
    assert redirect_stream.yielded == 0
    assert redirect_stream.closed is True


@pytest.mark.asyncio
async def test_get_attachment_data_uses_official_endpoint_and_decodes_base64():
    requested_urls = []

    def handler(request):
        requested_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={"data": "ZmlsZQ=="},
            headers={"Content-Type": "application/hal+json"},
        )

    client = await _client_with_transport(handler)
    try:
        data, mime_type = await client.get_attachment_data(123, 456, max_bytes=4)
    finally:
        await client.close()

    assert data == b"file"
    assert mime_type is None
    assert requested_urls == [
        "https://api.helpscout.net/v2/conversations/123/attachments/456/data"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "headers", "expected_mime"),
    [
        ({"data": "eA==", "mimeType": "image/png"}, {}, "image/png"),
        ({"data": "eA=="}, {"Content-Type": "image/jpeg; charset=binary"}, "image/jpeg"),
        ({"data": "eA=="}, {}, None),
        (
            {"data": "eA==", "mimeType": "application/hal+json"},
            {"Content-Type": "application/hal+json"},
            None,
        ),
    ],
)
async def test_get_attachment_data_normalizes_only_claimed_file_mime_metadata(
    payload,
    headers,
    expected_mime,
):
    client = await _client_with_transport(
        lambda _request: httpx.Response(200, json=payload, headers=headers)
    )
    try:
        _data, mime_type = await client.get_attachment_data(1, 2, max_bytes=1)
    finally:
        await client.close()

    assert mime_type == expected_mime


@pytest.mark.asyncio
async def test_get_attachment_data_allows_exact_limit_and_rejects_over_limit():
    payload = {"data": base64.b64encode(b"four").decode("ascii")}
    client = await _client_with_transport(
        lambda _request: httpx.Response(200, json=payload)
    )
    try:
        data, _mime_type = await client.get_attachment_data(1, 2, max_bytes=4)
        with pytest.raises(HelpScoutAttachmentTooLargeError) as raised:
            await client.get_attachment_data(1, 2, max_bytes=3)
    finally:
        await client.close()

    assert data == b"four"
    assert raised.value.code == "attachment_too_large"
    assert raised.value.status is None
    assert "Zm91cg==" not in str(raised.value)


@pytest.mark.asyncio
async def test_get_attachment_data_prechecks_encoded_size_without_exposing_data():
    encoded = base64.b64encode(b"not small").decode("ascii")
    client = await _client_with_transport(
        lambda _request: httpx.Response(200, json={"data": encoded})
    )
    try:
        with pytest.raises(HelpScoutAttachmentTooLargeError) as raised:
            await client.get_attachment_data(1, 2, max_bytes=1)
    finally:
        await client.close()

    assert raised.value.details["max_bytes"] == 1
    assert encoded not in str(raised.value)
    assert encoded not in repr(raised.value.details)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        ({}, "invalid_attachment_data"),
        ({"data": 123}, "invalid_attachment_data"),
        ({"data": "not base64!!"}, "invalid_attachment_data"),
    ],
)
async def test_get_attachment_data_rejects_missing_wrong_type_or_malformed_data(
    payload,
    expected_code,
):
    client = await _client_with_transport(
        lambda _request: httpx.Response(200, json=payload)
    )
    try:
        with pytest.raises(HelpScoutProtocolError) as raised:
            await client.get_attachment_data(1, 2, max_bytes=100)
    finally:
        await client.close()

    assert raised.value.code == expected_code


@pytest.mark.asyncio
@pytest.mark.parametrize("max_bytes", [0, -1, True])
async def test_get_attachment_data_requires_a_positive_byte_limit(max_bytes):
    request_count = 0

    def handler(_request):
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json={"data": "eA=="})

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutProtocolError) as raised:
            await client.get_attachment_data(1, 2, max_bytes=max_bytes)
    finally:
        await client.close()

    assert raised.value.code == "invalid_max_bytes"
    assert request_count == 0


@pytest.mark.asyncio
async def test_get_attachment_data_preserves_upstream_failure_without_body():
    client = await _client_with_transport(
        lambda _request: httpx.Response(429, text="credential-like secret")
    )
    try:
        with pytest.raises(HelpScoutUpstreamError) as raised:
            await client.get_attachment_data(1, 2, max_bytes=100)
    finally:
        await client.close()

    assert raised.value.status == 429
    assert raised.value.retryable is True
    assert "credential-like secret" not in str(raised.value)


def test_parse_webhook_payload(journal_products):
    payload = {
        "id": 601,
        "subject": "Help with Field Journal",
        "primaryCustomer": {
            "first": "John",
            "last": "Doe"
        },
        "tags": [{"tag": "field-journal"}],
        "_embedded": {
            "threads": [
                {
                    "type": "customer",
                    "body": "<p>I need help with my header.</p>",
                    "createdBy": {"type": "customer"}
                }
            ]
        }
    }

    result = parse_webhook_payload(payload)

    assert result["id"] == 601
    assert result["subject"] == "Help with Field Journal"
    assert result["customer_name"] == "John Doe"
    assert result["theme_tag"] == "field-journal"
    assert "header" in result["message"]


def test_parse_webhook_with_internal_note():
    payload = {
        "id": 601,
        "subject": "Refund request",
        "primaryCustomer": {"first": "Jane"},
        "tags": [],
        "_embedded": {
            "threads": [
                {
                    "type": "note",
                    "body": "<p>Please process refund</p>"
                },
                {
                    "type": "customer",
                    "body": "<p>I want a refund</p>"
                }
            ]
        }
    }

    result = parse_webhook_payload(payload)

    assert result["internal_note"] == "Please process refund"
    assert "refund" in result["message"]


def test_parse_webhook_uses_latest_internal_note_when_note_is_not_first_thread():
    payload = {
        "id": 601,
        "subject": "Follow-up",
        "primaryCustomer": {"first": "Jane"},
        "tags": [],
        "_embedded": {
            "threads": [
                {
                    "type": "customer",
                    "body": "<p>I still need help</p>",
                    "createdBy": {"type": "customer"},
                },
                {
                    "type": "note",
                    "body": "<p>Older note</p>",
                    "createdAt": "2026-03-28T09:00:00Z",
                },
                {
                    "type": "message",
                    "body": "<p>Agent reply</p>",
                    "createdBy": {"type": "user"},
                },
                {
                    "type": "note",
                    "body": "<p>Use the new troubleshooting steps</p>",
                    "createdAt": "2026-03-29T10:00:00Z",
                },
            ]
        }
    }

    result = parse_webhook_payload(payload)

    assert result["internal_note"] == "Use the new troubleshooting steps"
    assert result["message"] == "I still need help"


def test_parse_api_conversation_uses_latest_internal_note():
    data = {
        "id": 601,
        "subject": "Follow-up",
        "primaryCustomer": {"first": "Jane"},
        "tags": [],
        "_embedded": {
            "threads": [
                {
                    "type": "note",
                    "body": "<p>Old instructions</p>",
                    "createdAt": "2026-03-28T09:00:00Z",
                },
                {
                    "type": "customer",
                    "body": "<p>Still broken</p>",
                    "createdBy": {"type": "customer"},
                },
                {
                    "type": "note",
                    "body": "<p>Reply with the LT-specific fix</p>",
                    "createdAt": "2026-03-29T10:00:00Z",
                },
            ]
        }
    }

    result = parse_api_conversation(data)

    assert result["internal_note"] == "Reply with the LT-specific fix"


def test_parse_webhook_ignores_supa_system_notes_when_selecting_internal_note():
    payload = {
        "id": 601,
        "subject": "Follow-up",
        "primaryCustomer": {"first": "Jane"},
        "tags": [],
        "_embedded": {
            "threads": [
                {
                    "type": "note",
                    "body": "<p>Please verify the plugin versions first.</p>",
                    "createdAt": "2026-03-28T09:00:00Z",
                },
                {
                    "type": "note",
                    "body": "<p>Supa decision note: Mode: auto_send</p>",
                    "createdAt": "2026-03-29T10:00:00Z",
                    "source": {"type": "api"},
                },
                {
                    "type": "customer",
                    "body": "<p>I still need help</p>",
                    "createdBy": {"type": "customer"},
                },
            ]
        }
    }

    result = parse_webhook_payload(payload)

    assert result["internal_note"] == "Please verify the plugin versions first."


def test_parse_api_conversation_ignores_supa_system_notes_when_selecting_internal_note():
    data = {
        "id": 601,
        "subject": "Follow-up",
        "primaryCustomer": {"first": "Jane"},
        "tags": [],
        "_embedded": {
            "threads": [
                {
                    "type": "note",
                    "body": "<p>Reply with the LT-specific fix</p>",
                    "createdAt": "2026-03-28T09:00:00Z",
                },
                {
                    "type": "note",
                    "body": "<p>Supa auto-improvement note: detected change</p>",
                    "createdAt": "2026-03-29T10:00:00Z",
                    "source": {"type": "api"},
                },
                {
                    "type": "customer",
                    "body": "<p>Still broken</p>",
                    "createdBy": {"type": "customer"},
                },
            ]
        }
    }

    result = parse_api_conversation(data)

    assert result["internal_note"] == "Reply with the LT-specific fix"


def test_parse_webhook_api_source():
    payload = {
        "id": 999,
        "subject": "Test",
        "primaryCustomer": {"first": "Test"},
        "tags": [],
        "_embedded": {
            "threads": [
                {
                    "type": "customer",
                    "body": "Hello",
                    "source": {"type": "api"}
                }
            ]
        }
    }

    result = parse_webhook_payload(payload)
    assert result["source_type"] == "api"


def test_parse_webhook_uses_latest_customer_thread_for_message_and_source():
    payload = {
        "id": 1001,
        "subject": "Follow-up",
        "primaryCustomer": {"first": "Jane"},
        "tags": [],
        "_embedded": {
            "threads": [
                {
                    "type": "customer",
                    "body": "<p>Older customer message</p>",
                    "createdAt": "2026-04-01T09:00:00Z",
                    "createdBy": {"type": "customer"},
                    "source": {"type": "email"},
                },
                {
                    "type": "message",
                    "body": "<p>Agent reply</p>",
                    "createdAt": "2026-04-01T10:00:00Z",
                    "createdBy": {"type": "user"},
                },
                {
                    "type": "customer",
                    "body": "<p>Latest customer reply</p>",
                    "createdAt": "2026-04-01T11:00:00Z",
                    "createdBy": {"type": "customer"},
                    "source": {"type": "web"},
                },
            ]
        },
    }

    result = parse_webhook_payload(payload)

    assert result["message"] == "Latest customer reply"
    assert result["source_type"] == "web"


def test_parse_api_conversation_uses_latest_customer_thread_for_message():
    data = {
        "id": 1002,
        "subject": "Follow-up",
        "primaryCustomer": {"first": "Jane"},
        "tags": [],
        "_embedded": {
            "threads": [
                {
                    "type": "customer",
                    "body": "<p>Older customer message</p>",
                    "createdAt": "2026-04-01T09:00:00Z",
                    "createdBy": {"type": "customer"},
                },
                {
                    "type": "message",
                    "body": "<p>Agent reply</p>",
                    "createdAt": "2026-04-01T10:00:00Z",
                    "createdBy": {"type": "user"},
                },
                {
                    "type": "customer",
                    "body": "<p>Latest customer reply</p>",
                    "createdAt": "2026-04-01T11:00:00Z",
                    "createdBy": {"type": "customer"},
                },
            ]
        },
    }

    result = parse_api_conversation(data)

    assert result["message"] == "Latest customer reply"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("header_name", "header_value", "expected_resource_id"),
    [
        ("Resource-ID", "123", 123),
        ("resource-id", "0007", 7),
        ("RESOURCE-ID", str((1 << 63) - 1), (1 << 63) - 1),
    ],
)
async def test_create_mutation_result_captures_case_insensitive_resource_id(
    header_name,
    header_value,
    expected_resource_id,
):
    client = await _client_with_transport(
        lambda _request: httpx.Response(
            201,
            headers={header_name: header_value},
        )
    )
    try:
        result = await client.add_note_result(123, "Safe note")
    finally:
        await client.close()

    assert result == MutationResult(
        ok=True,
        status_code=201,
        resource_id=expected_resource_id,
    )


@pytest.mark.asyncio
async def test_add_note_can_preserve_pending_conversation_status():
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(201, headers={"Resource-ID": "123"})

    client = await _client_with_transport(handler)
    try:
        result = await client.add_note_result(123, "Safe note", status="pending")
    finally:
        await client.close()

    assert result.ok is True
    assert captured == [{"text": "Safe note", "status": "pending"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "expected_detail"),
    [
        ({}, "resource_id_missing"),
        ({"Resource-ID": "not-numeric"}, "resource_id_invalid"),
        ({"Resource-ID": "0"}, "resource_id_invalid"),
        ({"Resource-ID": str(1 << 63)}, "resource_id_invalid"),
    ],
)
async def test_create_mutation_success_never_invents_resource_id(
    headers,
    expected_detail,
):
    client = await _client_with_transport(
        lambda _request: httpx.Response(201, headers=headers)
    )
    try:
        result = await client.add_draft_result(123, "<p>Reply</p>")
    finally:
        await client.close()

    assert result == MutationResult(
        ok=True,
        status_code=201,
        detail=expected_detail,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("update_draft_result", (123, 456, "<p>Updated</p>")),
        ("delete_draft_result", (123, 456)),
        ("set_conversation_status_result", (123, "pending")),
        ("close_conversation_result", (123,)),
        ("update_subject_result", (123, "New subject")),
        ("add_tags_result", (123, ["one"], ["existing"])),
        ("remove_tag_result", (123, "old", ["old", "keep"])),
    ],
)
async def test_update_mutation_results_accept_only_204(method_name, args):
    client = await _client_with_transport(
        lambda _request: httpx.Response(204)
    )
    try:
        result = await getattr(client, method_name)(*args)
    finally:
        await client.close()

    assert result == MutationResult(ok=True, status_code=204)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_detail"),
    [
        (301, "upstream_redirect"),
        (400, "upstream_http_error"),
        (401, "upstream_auth_error"),
        (404, "upstream_not_found"),
        (412, "conversation_locked"),
        (429, "upstream_rate_limited"),
        (500, "upstream_server_error"),
    ],
)
async def test_mutation_http_failures_are_definitive_and_body_safe(
    status,
    expected_detail,
):
    request_count = 0

    def handler(_request):
        nonlocal request_count
        request_count += 1
        return httpx.Response(status, text="private upstream body")

    client = await _client_with_transport(handler)
    try:
        result = await client.add_note_result(123, "Safe note")
    finally:
        await client.close()

    assert result == MutationResult(
        ok=False,
        status_code=status,
        detail=expected_detail,
    )
    assert result.outcome_unknown is False
    assert "private upstream body" not in repr(result)
    assert request_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_type", "expected_detail"),
    [
        (httpx.ReadTimeout, "request_timeout"),
        (httpx.ConnectError, "network_error"),
    ],
)
async def test_post_submission_transport_failure_is_unknown_without_retry(
    failure_type,
    expected_detail,
):
    request_count = 0

    def handler(request):
        nonlocal request_count
        request_count += 1
        raise failure_type("mutation transport failed", request=request)

    client = await _client_with_transport(handler)
    try:
        result = await client.add_note_result(123, "Safe note")
    finally:
        await client.close()

    assert result == MutationResult(
        ok=False,
        status_code=None,
        outcome_unknown=True,
        detail=expected_detail,
    )
    assert request_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "headers", "expected"),
    [
        (
            201,
            {"Resource-ID": "321"},
            MutationResult(ok=True, status_code=201, resource_id=321),
        ),
        (
            412,
            {},
            MutationResult(
                ok=False,
                status_code=412,
                detail="conversation_locked",
            ),
        ),
    ],
)
async def test_mutation_classifies_headers_without_reading_response_body(
    status,
    headers,
    expected,
):
    streams = []

    def handler(request):
        stream = _ExplodingMutationBody(request)
        streams.append(stream)
        return httpx.Response(status, headers=headers, stream=stream)

    client = await _client_with_transport(handler)
    try:
        result = await client.add_note_result(123, "Safe note")
    finally:
        await client.close()

    assert result == expected
    assert streams[0].read_attempted is False
    assert streams[0].closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("oauth_outcome", "expected_status", "expected_detail"),
    [
        (401, 401, "oauth_token_error"),
        (500, 500, "oauth_token_error"),
        (httpx.ReadTimeout, None, "oauth_request_timeout"),
        (httpx.ConnectError, None, "oauth_network_error"),
    ],
)
async def test_pre_submission_oauth_failure_is_definitive_and_typed(
    oauth_outcome,
    expected_status,
    expected_detail,
):
    requested_paths = []

    def handler(request):
        requested_paths.append(request.url.path)
        assert request.url.path == "/v2/oauth2/token"
        if isinstance(oauth_outcome, int):
            return httpx.Response(oauth_outcome, text="private oauth body")
        raise oauth_outcome("oauth transport failed", request=request)

    client = await _client_with_transport(handler, authenticated=False)
    try:
        result = await client.add_note_result(123, "Safe note")
    finally:
        await client.close()

    assert result == MutationResult(
        ok=False,
        status_code=expected_status,
        outcome_unknown=False,
        detail=expected_detail,
    )
    assert "private oauth body" not in repr(result)
    assert requested_paths == ["/v2/oauth2/token"]


@pytest.mark.asyncio
async def test_oauth_expiry_overflow_is_definitive_invalid_token_result():
    requested_paths = []

    def handler(request):
        requested_paths.append(request.url.path)
        return httpx.Response(
            200,
            json={"access_token": "test-token", "expires_in": 10**100},
        )

    client = await _client_with_transport(handler, authenticated=False)
    try:
        result = await client.add_note_result(123, "Safe note")
    finally:
        await client.close()

    assert result == MutationResult(
        ok=False,
        status_code=200,
        outcome_unknown=False,
        detail="invalid_oauth_token_response",
    )
    assert requested_paths == ["/v2/oauth2/token"]


@pytest.mark.asyncio
async def test_result_paths_preserve_reply_payload_spacing_and_customer(monkeypatch):
    captured = []

    def handler(request):
        captured.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(201, headers={"Resource-ID": "321"})

    client = await _client_with_transport(handler)

    async def fake_ensure(_conversation_id, _status, **_kwargs):
        return {"verified": True, "conversation": {"status": "pending"}}

    monkeypatch.setattr(client, "ensure_conversation_status", fake_ensure)
    body = "<p>First.</p><p>Second.</p>"
    try:
        draft = await client.add_draft_result(123, body, customer_id=77)
        sent = await client.send_reply_result(123, body, customer_id=77)
    finally:
        await client.close()

    assert draft.resource_id == 321
    assert sent.resource_id == 321
    assert captured == [
        (
            "POST",
            "/v2/conversations/123/reply",
            {
                "text": "<p>First.</p><p>&nbsp;</p><p>Second.</p>",
                "draft": True,
                "customer": {"id": 77},
            },
        ),
        (
            "POST",
            "/v2/conversations/123/reply",
            {
                "text": "<p>First.</p><p>&nbsp;</p><p>Second.</p>",
                "status": "pending",
                "customer": {"id": 77},
            },
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verification_result", "expected_detail"),
    [
        ({"verified": False, "conversation": None}, "status_unverified"),
        (RuntimeError("status write failed"), "status_update_failed"),
    ],
)
async def test_successful_send_preserves_resource_id_with_safe_status_warning(
    monkeypatch,
    verification_result,
    expected_detail,
):
    client = await _client_with_transport(
        lambda _request: httpx.Response(
            201,
            headers={"Resource-ID": "654"},
        )
    )

    async def fake_ensure(_conversation_id, _status, **_kwargs):
        if isinstance(verification_result, Exception):
            raise verification_result
        return verification_result

    monkeypatch.setattr(client, "ensure_conversation_status", fake_ensure)
    try:
        result = await client.send_reply_result(123, "<p>Hello</p>")
    finally:
        await client.close()

    assert result == MutationResult(
        ok=True,
        status_code=201,
        resource_id=654,
        detail=expected_detail,
    )


@pytest.mark.asyncio
async def test_send_result_can_defer_status_verification_after_resource_id(monkeypatch):
    client = await _client_with_transport(
        lambda _request: httpx.Response(
            201,
            headers={"Resource-ID": "654"},
        )
    )

    async def forbidden_status_work(*_args, **_kwargs):
        raise AssertionError("status bookkeeping must be deferred")

    monkeypatch.setattr(client, "ensure_conversation_status", forbidden_status_work)
    try:
        result = await client.send_reply_result(
            123,
            "<p>Hello</p>",
            verify_status=False,
        )
    finally:
        await client.close()

    assert result == MutationResult(
        ok=True,
        status_code=201,
        resource_id=654,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("legacy_name", "result_name", "args"),
    [
        ("add_note", "add_note_result", (123, "Note")),
        ("add_draft", "add_draft_result", (123, "Draft")),
        ("send_reply", "send_reply_result", (123, "Reply")),
        ("update_draft", "update_draft_result", (123, 456, "Draft")),
        ("delete_draft", "delete_draft_result", (123, 456)),
        ("set_conversation_status", "set_conversation_status_result", (123, "pending")),
        ("close_conversation", "close_conversation_result", (123,)),
        ("update_subject", "update_subject_result", (123, "Subject")),
        ("add_tags", "add_tags_result", (123, ["tag"])),
        ("remove_tag", "remove_tag_result", (123, "tag", ["tag"])),
    ],
)
async def test_every_legacy_mutation_wrapper_returns_literal_bool(
    monkeypatch,
    legacy_name,
    result_name,
    args,
):
    client = HelpScoutClient(client_id="id", client_secret="secret")

    async def fake_result(*_args, **_kwargs):
        return MutationResult(ok=False, status_code=500)

    monkeypatch.setattr(client, result_name, fake_result)
    try:
        result = await getattr(client, legacy_name)(*args)
    finally:
        await client.close()

    assert result is False
    assert type(result) is bool


@pytest.mark.asyncio
async def test_send_reply_includes_customer_when_provided(monkeypatch):
    captured = {}

    async def fake_request(method, endpoint, **kwargs):
        captured["method"] = method
        captured["endpoint"] = endpoint
        captured["json"] = kwargs["json"]
        return SimpleNamespace(status_code=201)

    ensure_calls = []

    async def fake_ensure(conversation_id, status, **_kwargs):
        ensure_calls.append((conversation_id, status))
        return {"verified": True, "conversation": {"status": status}}

    client = HelpScoutClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(client, "ensure_conversation_status", fake_ensure)

    ok = await client.send_reply(123, "<p>Hello</p>", customer_id=77)

    assert ok is True
    assert captured["method"] == "POST"
    assert captured["endpoint"] == "/conversations/123/reply"
    assert captured["json"] == {
        "text": "<p>Hello</p>",
        "status": "pending",
        "customer": {"id": 77},
    }
    # After a successful send the conversation status is explicitly set + verified,
    # because HelpScout's reply `status` field alone does not reliably move it.
    assert ensure_calls == [(123, "pending")]

    await client.close()


@pytest.mark.asyncio
async def test_send_reply_skips_status_update_when_send_fails(monkeypatch):
    async def fake_request(method, endpoint, **kwargs):
        return SimpleNamespace(status_code=500)

    ensure_calls = []

    async def fake_ensure(conversation_id, status, **_kwargs):
        ensure_calls.append((conversation_id, status))
        return {"verified": True, "conversation": {"status": status}}

    client = HelpScoutClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(client, "ensure_conversation_status", fake_ensure)

    ok = await client.send_reply(123, "<p>Hello</p>", customer_id=77)

    assert ok is False
    assert ensure_calls == []  # never touch status when the reply itself failed

    await client.close()


@pytest.mark.asyncio
async def test_send_reply_succeeds_even_if_status_update_unverified(monkeypatch):
    async def fake_request(method, endpoint, **kwargs):
        return SimpleNamespace(status_code=201)

    async def fake_ensure(conversation_id, status, **_kwargs):
        # Status patch could not be verified — the customer still got the reply,
        # so the send must not be reported as a failure.
        return {"verified": False, "conversation": None}

    client = HelpScoutClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(client, "ensure_conversation_status", fake_ensure)

    ok = await client.send_reply(123, "<p>Hello</p>", customer_id=77)

    assert ok is True

    await client.close()


@pytest.mark.asyncio
async def test_set_conversation_status_patches_status(monkeypatch):
    captured = {}

    async def fake_request(method, endpoint, **kwargs):
        captured["method"] = method
        captured["endpoint"] = endpoint
        captured["json"] = kwargs["json"]
        return SimpleNamespace(status_code=204)

    client = HelpScoutClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_request", fake_request)

    ok = await client.set_conversation_status(123, "pending")

    assert ok is True
    assert captured["method"] == "PATCH"
    assert captured["endpoint"] == "/conversations/123"
    assert captured["json"] == {"op": "replace", "path": "/status", "value": "pending"}

    await client.close()


def test_get_draft_thread_ids_returns_all_drafts_in_created_order():
    threads = [
        {"id": 998, "type": "message", "state": "draft", "createdAt": "2026-04-09T10:00:00Z"},
        {"id": 999, "type": "message", "state": "draft", "createdAt": "2026-04-09T11:00:00Z"},
        {"id": 777, "type": "customer", "createdAt": "2026-04-09T12:00:00Z"},
    ]

    assert get_draft_thread_ids(threads) == [998, 999]


@pytest.mark.asyncio
async def test_ensure_conversation_status_never_retries_status_mutation(monkeypatch):
    client = HelpScoutClient(client_id="id", client_secret="secret")
    status_calls = []

    async def fake_set_status(conversation_id, status):
        status_calls.append((conversation_id, status))
        return True

    async def fake_get_conversation(conversation_id):
        return {"id": conversation_id, "status": "active"}

    monkeypatch.setattr(client, "set_conversation_status", fake_set_status)
    monkeypatch.setattr(client, "get_conversation", fake_get_conversation)

    result = await client.ensure_conversation_status(123, "pending")

    assert result["verified"] is False
    assert result["conversation"]["status"] == "active"
    assert status_calls == [(123, "pending")]

    await client.close()


@pytest.mark.asyncio
async def test_get_conversations_with_drafts_uses_latest_draft_thread(monkeypatch, journal_products):
    payload = {
        "_embedded": {
            "conversations": [
                {
                    "id": 604,
                    "subject": "Draft mismatch",
                    "tags": [{"tag": "trail-atlas"}],
                    "primaryCustomer": {
                        "first": "Jane",
                        "last": "Doe",
                        "email": "jane@example.com",
                    },
                    "_embedded": {
                        "threads": [
                            {
                                "id": 101,
                                "type": "message",
                                "state": "draft",
                                "body": "<p>Older draft</p>",
                                "createdAt": "2026-04-01T09:00:00Z",
                            },
                            {
                                "id": 202,
                                "type": "message",
                                "state": "draft",
                                "body": "<p>Newest draft</p>",
                                "createdAt": "2026-04-02T10:00:00Z",
                            },
                            {
                                "id": 303,
                                "type": "customer",
                                "body": "<p>Latest customer message</p>",
                                "createdBy": {"type": "customer"},
                            },
                        ]
                    },
                }
            ]
        }
    }

    async def fake_request(method, endpoint, **kwargs):
        return SimpleNamespace(status_code=200, json=lambda: payload)

    client = HelpScoutClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_request", fake_request)

    drafts = await client.get_conversations_with_drafts(include_details=True)

    assert drafts == [
        {
            "id": 604,
            "subject": "Draft mismatch",
            "theme": "trail-atlas",
            "url": "https://secure.helpscout.net/conversation/604",
            "customer_name": "Jane Doe",
            "customer_email": "jane@example.com",
            "customer_message": "Latest customer message",
            "draft_content": "<p>Newest draft</p>",
            "draft_thread_id": 202,
            "tags": ["trail-atlas"],
        }
    ]

    await client.close()


@pytest.mark.asyncio
async def test_get_conversations_with_drafts_uses_latest_customer_message(monkeypatch):
    payload = {
        "_embedded": {
            "conversations": [
                {
                    "id": 605,
                    "subject": "Draft customer context",
                    "tags": [{"tag": "field-journal"}],
                    "primaryCustomer": {
                        "first": "Jane",
                        "last": "Doe",
                        "email": "jane@example.com",
                    },
                    "_embedded": {
                        "threads": [
                            {
                                "id": 401,
                                "type": "customer",
                                "body": "<p>Older customer message</p>",
                                "createdAt": "2026-04-01T09:00:00Z",
                                "createdBy": {"type": "customer"},
                            },
                            {
                                "id": 402,
                                "type": "message",
                                "state": "draft",
                                "body": "<p>Draft reply</p>",
                                "createdAt": "2026-04-01T10:00:00Z",
                            },
                            {
                                "id": 403,
                                "type": "customer",
                                "body": "<p>Latest customer message</p>",
                                "createdAt": "2026-04-01T11:00:00Z",
                                "createdBy": {"type": "customer"},
                            },
                        ]
                    },
                }
            ]
        }
    }

    async def fake_request(method, endpoint, **kwargs):
        return SimpleNamespace(status_code=200, json=lambda: payload)

    client = HelpScoutClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_request", fake_request)

    drafts = await client.get_conversations_with_drafts(include_details=True)

    assert drafts[0]["customer_message"] == "Latest customer message"

    await client.close()


@pytest.mark.asyncio
async def test_get_follow_ups_uses_latest_actionable_thread(monkeypatch):
    payload = {
        "_embedded": {
            "conversations": [
                {
                    "id": 606,
                    "subject": "Real follow-up",
                    "tags": [],
                    "_embedded": {
                        "threads": [
                            {
                                "type": "customer",
                                "body": "<p>Older customer message</p>",
                                "createdAt": "2026-04-01T09:00:00Z",
                                "createdBy": {"type": "customer"},
                            },
                            {
                                "type": "message",
                                "body": "<p>Agent reply</p>",
                                "createdAt": "2026-04-01T10:00:00Z",
                                "createdBy": {"type": "user"},
                            },
                            {
                                "type": "customer",
                                "body": "<p>Latest customer reply</p>",
                                "createdAt": "2026-04-01T11:00:00Z",
                                "createdBy": {"type": "customer"},
                            },
                        ]
                    },
                }
            ]
        }
    }

    async def fake_request(method, endpoint, **kwargs):
        return SimpleNamespace(status_code=200, json=lambda: payload)

    client = HelpScoutClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_request", fake_request)

    follow_ups = await client.get_follow_ups()

    assert [item["id"] for item in follow_ups] == [606]

    await client.close()


@pytest.mark.asyncio
async def test_get_follow_ups_skips_when_latest_actionable_thread_is_agent(monkeypatch):
    payload = {
        "_embedded": {
            "conversations": [
                {
                    "id": 607,
                    "subject": "Not a follow-up",
                    "tags": [],
                    "_embedded": {
                        "threads": [
                            {
                                "type": "customer",
                                "body": "<p>Older customer message</p>",
                                "createdAt": "2026-04-01T09:00:00Z",
                                "createdBy": {"type": "customer"},
                            },
                            {
                                "type": "message",
                                "body": "<p>Latest agent reply</p>",
                                "createdAt": "2026-04-01T11:00:00Z",
                                "createdBy": {"type": "user"},
                            },
                        ]
                    },
                }
            ]
        }
    }

    async def fake_request(method, endpoint, **kwargs):
        return SimpleNamespace(status_code=200, json=lambda: payload)

    client = HelpScoutClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_request", fake_request)

    follow_ups = await client.get_follow_ups()

    assert follow_ups == []

    await client.close()


@pytest.mark.asyncio
async def test_get_recent_conversations_retries_once_on_timeout(monkeypatch):
    payload = {
        "_embedded": {
            "conversations": [
                {
                    "id": 608,
                    "subject": "Recovered after timeout",
                }
            ]
        }
    }
    attempts = 0

    async def fake_request(method, endpoint, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("timed out")
        return SimpleNamespace(status_code=200, json=lambda: payload)

    client = HelpScoutClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_request", fake_request)

    conversations = await client.get_recent_conversations()

    assert [conversation["id"] for conversation in conversations] == [608]
    assert attempts == 2

    await client.close()


PLACEHOLDER_DRAFT_HTML = "<p>Manual review required before sending a reply.</p>"


def test_draft_is_superseded_when_a_reply_was_sent_after_it():
    threads = [
        {
            "id": 202,
            "type": "message",
            "state": "draft",
            "body": PLACEHOLDER_DRAFT_HTML,
            "createdAt": "2026-07-30T04:13:35Z",
        },
        {
            "id": 303,
            "type": "message",
            "state": "published",
            "createdBy": {"type": "user"},
            "body": "<p>The reply that actually went out</p>",
            "createdAt": "2026-07-31T11:55:46Z",
        },
    ]

    assert is_draft_superseded_by_reply(threads, threads[0]) is True


def test_system_user_message_is_an_agent_actionable_reply_and_supersedes_draft():
    draft = {
        "id": 201,
        "type": "message",
        "state": "draft",
        "draft": True,
        "createdAt": "2026-07-30T04:13:35Z",
        "createdBy": {"type": "user"},
    }
    system_reply = {
        "id": 202,
        "type": "message",
        "state": "published",
        "createdAt": "2026-07-31T11:55:46Z",
        "createdBy": {"type": "system_user"},
    }

    assert _is_agent_message_thread(system_reply) is True
    assert is_draft_superseded_by_reply([draft, system_reply], draft) is True


def test_same_timestamp_higher_reply_id_supersedes_draft():
    draft = {
        "id": 201,
        "type": "message",
        "state": "draft",
        "createdAt": "2026-07-30T04:13:35Z",
    }
    reply = {
        "id": 202,
        "type": "message",
        "state": "published",
        "createdAt": "2026-07-30T04:13:35Z",
        "createdBy": {"type": "user"},
    }

    assert is_draft_superseded_by_reply([reply, draft], draft) is True


def test_same_timestamp_lower_reply_id_does_not_supersede_draft():
    draft = {
        "id": 202,
        "type": "message",
        "state": "draft",
        "createdAt": "2026-07-30T04:13:35Z",
    }
    reply = {
        "id": 201,
        "type": "message",
        "state": "published",
        "createdAt": "2026-07-30T04:13:35Z",
        "createdBy": {"type": "user"},
    }

    assert is_draft_superseded_by_reply([draft, reply], draft) is False


@pytest.mark.parametrize(
    "draft_id,reply_id",
    [(None, 202), (201, None), (True, 202), (201, False), ("invalid", 202), (201, "invalid")],
)
def test_same_timestamp_invalid_ids_conservatively_keep_draft(draft_id, reply_id):
    draft = {
        "id": draft_id,
        "type": "message",
        "state": "draft",
        "createdAt": "2026-07-30T04:13:35Z",
    }
    reply = {
        "id": reply_id,
        "type": "message",
        "state": "published",
        "createdAt": "2026-07-30T04:13:35Z",
        "createdBy": {"type": "user"},
    }

    assert is_draft_superseded_by_reply([draft, reply], draft) is False


def test_customer_predicate_handles_null_or_non_mapping_created_by():
    from services.helpscout import _is_customer_thread

    assert _is_customer_thread({"type": "message", "createdBy": None}) is False
    assert _is_customer_thread({"type": "message", "createdBy": "customer"}) is False
    assert _is_customer_thread({"type": "customer", "createdBy": None}) is True


def test_manual_review_placeholder_survives_when_no_reply_followed():
    """The placeholder body alone must never disqualify a draft.

    Supa posts the same text when it cannot generate a reply at all, and those
    tickets are exactly the ones that still need a human.
    """
    threads = [
        {
            "id": 202,
            "type": "message",
            "state": "draft",
            "body": PLACEHOLDER_DRAFT_HTML,
            "createdAt": "2026-07-25T10:33:39Z",
        },
        {
            "id": 101,
            "type": "customer",
            "createdBy": {"type": "customer"},
            "body": "<p>Original question</p>",
            "createdAt": "2026-07-25T10:21:24Z",
        },
    ]

    assert is_draft_superseded_by_reply(threads, threads[0]) is False


def test_draft_created_after_the_last_reply_is_still_pending():
    threads = [
        {
            "id": 303,
            "type": "message",
            "state": "published",
            "createdBy": {"type": "user"},
            "body": "<p>Earlier reply</p>",
            "createdAt": "2026-07-29T19:22:18Z",
        },
        {
            "id": 404,
            "type": "message",
            "state": "draft",
            "body": "<p>Follow-up draft awaiting review</p>",
            "createdAt": "2026-07-30T04:13:35Z",
        },
    ]

    assert is_draft_superseded_by_reply(threads, threads[1]) is False


def test_draft_without_timestamp_is_kept_visible():
    threads = [
        {"id": 202, "type": "message", "state": "draft", "body": "<p>Draft</p>"},
        {
            "id": 303,
            "type": "message",
            "state": "published",
            "createdBy": {"type": "user"},
            "body": "<p>Reply</p>",
            "createdAt": "2026-07-31T11:55:46Z",
        },
    ]

    assert is_draft_superseded_by_reply(threads, threads[0]) is False


def test_another_draft_thread_does_not_count_as_a_sent_reply():
    """Draft threads are also type=message/createdBy=user — they must not supersede."""
    threads = [
        {
            "id": 202,
            "type": "message",
            "state": "draft",
            "createdBy": {"type": "user"},
            "body": "<p>Older draft</p>",
            "createdAt": "2026-07-30T04:13:35Z",
        },
        {
            "id": 303,
            "type": "message",
            "state": "draft",
            "createdBy": {"type": "user"},
            "body": "<p>Newer draft</p>",
            "createdAt": "2026-07-31T11:55:46Z",
        },
    ]

    assert is_draft_superseded_by_reply(threads, threads[0]) is False


@pytest.mark.asyncio
async def test_get_conversations_with_drafts_skips_superseded_leftover(monkeypatch):
    payload = {
        "_embedded": {
            "conversations": [
                {
                    "id": 617,
                    "subject": "Already replied to",
                    "tags": [{"tag": "journal-base"}],
                    "primaryCustomer": {"first": "Mica", "email": "m@example.com"},
                    "_embedded": {
                        "threads": [
                            {
                                "id": 202,
                                "type": "message",
                                "state": "draft",
                                "body": PLACEHOLDER_DRAFT_HTML,
                                "createdAt": "2026-07-30T04:13:35Z",
                            },
                            {
                                "id": 303,
                                "type": "message",
                                "state": "published",
                                "createdBy": {"type": "user"},
                                "body": "<p>Sent reply</p>",
                                "createdAt": "2026-07-31T11:55:46Z",
                            },
                        ]
                    },
                },
                {
                    "id": 616,
                    "subject": "Still needs a human",
                    "tags": [{"tag": "trail-atlas"}],
                    "primaryCustomer": {"first": "Mica", "email": "ray@example.com"},
                    "_embedded": {
                        "threads": [
                            {
                                "id": 404,
                                "type": "message",
                                "state": "draft",
                                "body": PLACEHOLDER_DRAFT_HTML,
                                "createdAt": "2026-07-25T10:33:39Z",
                            },
                            {
                                "id": 505,
                                "type": "customer",
                                "createdBy": {"type": "customer"},
                                "body": "<p>Question</p>",
                                "createdAt": "2026-07-25T10:21:24Z",
                            },
                        ]
                    },
                },
            ]
        }
    }

    async def fake_request(method, endpoint, **kwargs):
        return SimpleNamespace(status_code=200, json=lambda: payload)

    client = HelpScoutClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_request", fake_request)

    drafts = await client.get_conversations_with_drafts()

    assert [d["id"] for d in drafts] == [616]

    await client.close()


# --- outgoing paragraph spacing ----------------------------------------

from services.helpscout import ensure_email_paragraph_spacing


def test_bare_paragraph_html_gets_spacers_between_blocks():
    """Keep one spacer between each adjacent pair of short blocks."""
    html = "<p>Hello, Reader.</p>\n<p>Here are the steps.</p>\n<ol><li>One</li></ol>\n<p>Thank you,<br />Operator</p>"

    spaced = ensure_email_paragraph_spacing(html)

    assert "</p><p>&nbsp;</p>" in spaced
    assert spaced.count("<p>&nbsp;</p>") == 3  # p->p, p->ol, ol->p
    # Content untouched
    assert "Hello, Reader." in spaced and "<ol><li>One</li></ol>" in spaced


def test_pipeline_br_style_bodies_are_left_alone():
    html = "Hello, Reader.<br><br>The appendix has three entries.<br><br>Thank you,<br>Operator"

    assert ensure_email_paragraph_spacing(html) == html


def test_already_spaced_html_is_not_double_spaced():
    html = "<p>Hey!</p><p>&nbsp;</p><p>Second paragraph.</p>"

    assert ensure_email_paragraph_spacing(html) == html


def test_spacing_is_idempotent():
    html = "<p>One</p><p>Two</p><p>Three</p>"

    once = ensure_email_paragraph_spacing(html)
    twice = ensure_email_paragraph_spacing(once)

    assert once == twice
    assert once.count("<p>&nbsp;</p>") == 2


def test_single_paragraph_placeholder_is_unchanged():
    html = "<p>Manual review required before sending a reply.</p>"

    assert ensure_email_paragraph_spacing(html) == html


@pytest.mark.asyncio
async def test_outgoing_send_and_draft_paths_apply_spacing(monkeypatch):
    sent_payloads = []

    async def fake_request(method, endpoint, **kwargs):
        sent_payloads.append((endpoint, kwargs.get("json")))
        return SimpleNamespace(status_code=201, json=lambda: {})

    client = HelpScoutClient(client_id="id", client_secret="secret")
    monkeypatch.setattr(client, "_request", fake_request)
    # send_reply's status verification would make extra calls; stub it out
    async def fake_ensure(*a, **k):
        return {"verified": True}
    monkeypatch.setattr(client, "ensure_conversation_status", fake_ensure)

    html = "<p>First.</p><p>Second.</p>"
    await client.add_draft(1, html)
    await client.send_reply(1, html)

    for endpoint, payload in sent_payloads:
        assert "<p>&nbsp;</p>" in payload["text"], f"no spacer in payload to {endpoint}"

    await client.close()


# --- customer attachments as evidence ----------------------------------

from services.helpscout import describe_attachments, extract_customer_attachments


def _thread_with_attachments(attachments, created="2026-08-03T21:28:42Z"):
    return {
        "type": "customer",
        "createdBy": {"type": "customer"},
        "body": "<p>See screenshots</p>",
        "createdAt": created,
        "_embedded": {"attachments": attachments},
    }


def test_extract_attachments_flags_blank_transport_artifacts():
    """The 185-byte Outlook placeholders seen across an entire Field Journal thread."""
    threads = [_thread_with_attachments([
        {"filename": "wp-admin.png", "mimeType": "image/png", "size": 8192},
        {"filename": "image001.png", "mimeType": "image/png", "size": 185},
        {"filename": "notes.pdf", "mimeType": "application/pdf", "size": 1063},
    ])]

    result = extract_customer_attachments(threads)

    assert [a["filename"] for a in result] == ["wp-admin.png", "image001.png", "notes.pdf"]
    assert result[0]["is_image"] and not result[0]["is_blank"]
    assert result[1]["is_image"] and result[1]["is_blank"]
    assert not result[2]["is_image"]


def test_extract_attachments_uses_only_the_latest_customer_message():
    threads = [
        _thread_with_attachments([{"filename": "old.png", "mimeType": "image/png", "size": 5000}],
                                 created="2026-08-01T10:00:00Z"),
        _thread_with_attachments([{"filename": "new.png", "mimeType": "image/png", "size": 6000}],
                                 created="2026-08-03T10:00:00Z"),
    ]

    assert [a["filename"] for a in extract_customer_attachments(threads)] == ["new.png"]


def test_describe_attachments_demands_analysis_for_real_screenshots():
    summary = describe_attachments([
        {"filename": "wp-admin.png", "mime_type": "image/png", "size": 8192,
         "is_image": True, "is_blank": False},
    ])

    assert "wp-admin.png" in summary
    assert "primary evidence" in summary


def test_describe_attachments_flags_all_blank_as_never_arrived():
    summary = describe_attachments([
        {"filename": "image001.png", "mime_type": "image/png", "size": 185,
         "is_image": True, "is_blank": True},
    ])

    assert "BLANK" in summary
    assert "did not come through" in summary


def test_parsers_expose_attachments():
    payload = {
        "id": 1, "subject": "s", "primaryCustomer": {"first": "M"}, "tags": [],
        "_embedded": {"threads": [_thread_with_attachments(
            [{"filename": "shot.png", "mimeType": "image/png", "size": 4000}])]},
    }

    assert parse_webhook_payload(payload)["attachments"][0]["filename"] == "shot.png"
    assert parse_api_conversation(payload)["attachments"][0]["filename"] == "shot.png"
class _FakeResponse:
    def __init__(self, status_code=201, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = ""

    def json(self):
        return {}


async def test_create_conversation_posts_a_reply_thread_from_operator():
    from unittest.mock import AsyncMock
    from services.helpscout import HelpScoutClient

    hs = HelpScoutClient("id", "secret")
    hs._request = AsyncMock(return_value=_FakeResponse(headers={"Resource-Id": "4242"}))

    conversation_id = await hs.create_conversation(
        mailbox_id=609,
        subject="Checking a new index",
        customer={"email": "mica@woodland.example", "firstName": "Mica"},
        text="Hello Mica, the sample index is ready to review.",
        tags=["reader-question", "field-journal"],
        user_id=612,
        draft=True,
    )

    assert conversation_id == 4242
    method, endpoint = hs._request.await_args.args
    assert method == "POST"
    assert endpoint == "/conversations"
    body = hs._request.await_args.kwargs["json"]
    assert body["mailboxId"] == 609
    assert body["type"] == "email"
    assert body["status"] == "pending"
    assert body["tags"] == ["reader-question", "field-journal"]
    assert body["customer"]["email"] == "mica@woodland.example"
    thread = body["threads"][0]
    assert thread["type"] == "reply"
    assert thread["user"] == 612
    assert thread["draft"] is True
    assert thread["text"] == "Hello Mica, the sample index is ready to review."


async def test_create_conversation_returns_none_on_failure():
    from unittest.mock import AsyncMock
    from services.helpscout import HelpScoutClient

    hs = HelpScoutClient("id", "secret")
    hs._request = AsyncMock(return_value=_FakeResponse(status_code=400))
    assert await hs.create_conversation(mailbox_id=1, subject="s", customer={"email": "mica@woodland.example"},
                                        text="t", tags=[], user_id=2) is None


async def test_list_conversations_is_a_read_only_get():
    from unittest.mock import AsyncMock
    from services.helpscout import HelpScoutClient

    class Resp:
        status_code = 200

        def json(self):
            return {
                "_embedded": {
                    "conversations": [{"id": value} for value in range(1, 26)]
                },
                "page": {"totalPages": 3},
            }

    hs = HelpScoutClient("id", "secret")
    hs._read_bounded = AsyncMock(return_value=Resp())
    result = await hs.list_conversations(query="modifiedAt:[2026-02-18T00:00:00Z TO *]", page=2)
    assert result["conversations"] == [{"id": value} for value in range(1, 26)]
    assert result["pages"] == 3
    endpoint = hs._read_bounded.await_args.args[0]
    assert "page=2" in endpoint
    assert "pageSize" not in endpoint
    assert hs._read_bounded.await_args.kwargs == {"max_bytes": 8 * 1024 * 1024}


async def test_list_conversations_passes_tag_as_a_separate_encoded_filter():
    """A tag must never be interpolated into Help Scout's free-text query syntax."""
    from unittest.mock import AsyncMock
    from services.helpscout import HelpScoutClient

    class Resp:
        status_code = 200

        def json(self):
            return {"_embedded": {"conversations": []}, "page": {"totalPages": 0}}

    hs = HelpScoutClient("id", "secret")
    hs._read_bounded = AsyncMock(return_value=Resp())

    await hs.list_conversations(
        query='subject:"Field Journal"',
        tag="needs review & follow-up",
        page=1,
    )

    endpoint = hs._read_bounded.await_args.args[0]
    assert "query=subject%3A%22Field+Journal%22" in endpoint
    assert "tag=needs+review+%26+follow-up" in endpoint
    assert "needs review" not in endpoint
    assert "pageSize" not in endpoint


@pytest.mark.parametrize(
    "status,error_type,retryable",
    [
        (401, HelpScoutAuthError, False),
        (403, HelpScoutAuthError, False),
        (400, HelpScoutUpstreamError, False),
        (429, HelpScoutUpstreamError, True),
        (500, HelpScoutUpstreamError, True),
    ],
)
async def test_list_conversations_preserves_typed_upstream_failures(
    status, error_type, retryable
):
    async def handler(request):
        return httpx.Response(status, content=b'{"secret":"must-not-be-read"}')

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(error_type) as raised:
            await client.list_conversations()
    finally:
        await client.close()

    assert raised.value.status == status
    assert raised.value.retryable is retryable
    assert "must-not-be-read" not in str(raised.value)


@pytest.mark.parametrize(
    "content",
    [
        b"not-json",
        b"[]",
        b"{}",
        b'{"_embedded":{},"page":{"totalPages":1}}',
        b'{"_embedded":{"conversations":{}},"page":{"totalPages":1}}',
        b'{"_embedded":{"conversations":[1]},"page":{"totalPages":1}}',
        b'{"_embedded":{"conversations":[]}}',
        b'{"_embedded":{"conversations":[]},"page":{"totalPages":true}}',
        b'{"_embedded":{"conversations":[]},"page":{"totalPages":-1}}',
        b'{"_embedded":{"conversations":[]},"page":{"totalPages":"1"}}',
        b'{"_embedded":{"conversations":[{"id":1}]},"page":{"totalPages":0}}',
    ],
)
async def test_list_conversations_rejects_malformed_success_shapes(content):
    async def handler(request):
        return httpx.Response(200, content=content)

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutProtocolError):
            await client.list_conversations()
    finally:
        await client.close()


@pytest.mark.parametrize("number", [0, 1])
async def test_list_conversations_accepts_explicit_empty_zero_page_sentinel(number):
    async def handler(request):
        return httpx.Response(
            200,
            json={
                "_embedded": {"conversations": []},
                "page": {"number": number, "totalPages": 0},
            },
        )

    client = await _client_with_transport(handler)
    try:
        result = await client.list_conversations()
    finally:
        await client.close()

    assert result == {"conversations": [], "pages": 0}


@pytest.mark.parametrize(
    "count,total_pages",
    [(24, 2), (26, 2), (26, 1)],
)
async def test_list_conversations_enforces_fixed_twenty_five_item_pages(
    count, total_pages
):
    async def handler(request):
        return httpx.Response(
            200,
            json={
                "_embedded": {
                    "conversations": [
                        {"id": value} for value in range(1, count + 1)
                    ]
                },
                "page": {"number": 1, "totalPages": total_pages},
            },
        )

    client = await _client_with_transport(handler)
    try:
        with pytest.raises(HelpScoutProtocolError) as raised:
            await client.list_conversations(page_size=100)
    finally:
        await client.close()

    assert raised.value.code == "invalid_pagination"


def test_add_tags_without_existing_tags_reads_and_keeps_the_current_ones():
    """add_tags(conversation, tags) must not wipe the conversation's other tags (2026-09-30: supa-ignore and the
    campaign tags were dropped from a pilot conversation for a minute by exactly this call)."""
    import asyncio
    from services.helpscout import HelpScoutClient, MutationResult

    client = HelpScoutClient("id", "secret")
    sent = {}

    async def fake_get(conversation_id):
        return {"id": conversation_id, "tags": [{"tag": "supa-ignore"}, {"tag": "reader-question"}]}

    async def fake_mutation(method, path, success_status=None, capture_resource_id=False, json=None):
        sent.update(method=method, path=path, tags=sorted(json["tags"]))
        return MutationResult(ok=True, status_code=204)

    client.get_conversation = fake_get
    client._mutation_result = fake_mutation
    assert asyncio.run(client.add_tags_result(123, ["reader-question-operator"])).ok
    assert sent["tags"] == ["reader-question", "reader-question-operator", "supa-ignore"]
    # An explicit empty list still means "the conversation has no tags".
    asyncio.run(client.add_tags_result(123, ["one"], []))
    assert sent["tags"] == ["one"]


# --- Drafts never read as sent replies (2026-10-02) -------------------------

from services.helpscout import (  # noqa: E402
    agent_replies_from_threads,
    extract_conversation_history,
    get_supa_owned_draft_thread_ids,
)

_LEFTOVER = "<p>Manual review required before sending a reply.</p>"


def _t(thread_id, kind, body, created, *, state="published", source="email", by="customer"):
    thread = {"id": thread_id, "type": kind, "state": state, "body": body, "createdAt": created,
              "createdBy": {"type": by}, "source": {"type": source}}
    if state == "draft":
        thread["draft"] = True
    return thread


def test_conversation_history_leaves_out_drafts_and_leftovers():
    threads = [  # Help Scout lists newest first
        _t(4, "message", _LEFTOVER, "2026-10-02T10:51:00Z", state="draft", source="api", by="user"),
        _t(3, "message", "<p>Fixed it, refund issued.</p>", "2026-10-02T10:40:00Z", state="draft", source="web", by="user"),
        _t(2, "message", "<p>We are on it.</p>", "2026-10-02T10:30:00Z", source="api", by="user"),
        _t(1, "customer", "<p>Help</p>", "2026-10-02T10:00:00Z"),
    ]
    assert extract_conversation_history(threads) == [
        {"role": "customer", "message": "Help"},
        {"role": "support", "message": "We are on it."},
    ]


def test_agent_replies_are_published_messages_only():
    threads = [
        _t(3, "message", "<p>Unsent idea</p>", "2026-10-02T10:40:00Z", state="draft", source="web", by="user"),
        _t(2, "message", "<p>Sent answer</p>", "2026-10-02T10:30:00Z", source="api", by="user"),
        _t(1, "customer", "<p>Help</p>", "2026-10-02T10:00:00Z"),
    ]
    assert [r["body"] for r in agent_replies_from_threads(threads)] == ["Sent answer"]


def test_history_preserves_the_fix_after_a_long_explanation():
    body = "<p>" + ("Earlier investigation details. " * 35) + "</p>"
    body += "<pre>wp-content/themes/index-book/assets/js/admin/pix_builder.js: remove the stale serialize call.</pre>"
    history = extract_conversation_history([
        _t(1, "message", body, "2026-10-02T10:30:00Z", source="api", by="user"),
    ])

    assert "pix_builder.js: remove the stale serialize call" in history[0]["message"]


@pytest.mark.parametrize("oldest_first", [False, True])
def test_history_total_budget_keeps_the_newest_published_messages(oldest_first):
    threads = [
        _t(2, "customer", "Latest question", "2026-10-02T10:40:00Z"),
        _t(1, "message", "Old details " * 100, "2026-10-02T10:30:00Z", source="api", by="user"),
    ]
    history = extract_conversation_history(list(reversed(threads)) if oldest_first else threads,
                                           max_total_chars=100)

    assert history[-1] == {"role": "customer", "message": "Latest question"}
    assert sum(len(m["message"]) for m in history) <= 100


def test_history_budget_orders_scrambled_threads_before_selecting_recent_messages():
    threads = [
        _t(2, "customer", "Middle", "2026-10-02T10:40:00Z"),
        _t(1, "message", "Old details " * 100, "2026-10-02T10:30:00Z", by="user"),
        _t(3, "customer", "Latest", "2026-10-02T10:50:00Z"),
    ]
    history = extract_conversation_history(threads, max_total_chars=20)

    assert [m["message"] for m in history] == ["Middle", "Latest"]


def test_only_supa_owned_live_drafts_are_neutralized():
    threads = [
        _t(5, "message", "<p>Operator typing</p>", "2026-10-02T10:50:00Z", state="draft", source="web", by="user"),
        _t(4, "message", _LEFTOVER, "2026-10-02T10:45:00Z", state="draft", source="api", by="user"),
        _t(3, "message", "<p>Supa draft</p>", "2026-10-02T10:40:00Z", state="draft", source="api", by="user"),
        _t(2, "message", "<p>Sent</p>", "2026-10-02T10:30:00Z", source="api", by="user"),
    ]
    assert get_supa_owned_draft_thread_ids(threads) == [3]


@pytest.mark.asyncio
async def test_complete_conversation_reads_more_than_one_hundred_real_threads():
    pages=[]
    def handler(request):
        if request.url.path=="/v2/conversations/123":return httpx.Response(200,json=_conversation_payload(123))
        number=int(request.url.params["page"]);pages.append(number)
        start=(number-1)*100;stop=min(number*100,101)
        return httpx.Response(200,json=_thread_page([{"id":i} for i in range(start+1,stop+1)],number=number,total_pages=2,size=100,total_elements=101))
    client=await _client_with_transport(handler)
    try:result=await client.get_complete_conversation(123)
    finally:await client.close()
    assert len(result["_embedded"]["threads"])==101 and pages==[1,2]


@pytest.mark.parametrize('value,expected', [('37',37),('SECRET',None),('-1',None),('613',None)])
def test_read_rate_limit_retains_only_bounded_numeric_retry_after(value, expected):
    from services.helpscout import _raise_for_read_status
    with pytest.raises(HelpScoutUpstreamError) as raised:
        _raise_for_read_status(httpx.Response(429, headers={'Retry-After':value}), 'conversations')
    assert raised.value.details.get('retry_after_seconds') == expected
    assert value != 'SECRET' or 'SECRET' not in str(raised.value.details)


# --- images pasted into the message body --------------------------------

from services.helpscout import extract_inline_images, is_helpscout_inline_image_url  # noqa: E402

_CDN = "https://assets.woodland.example/inline/603/abc/def"


def test_pasted_screenshot_counts_as_primary_evidence(inline_assets):
    """A pasted illustration is part of the current message."""
    threads = [{
        "type": "customer", "createdBy": {"type": "customer"}, "createdAt": "2026-11-08T15:27:01Z",
        "body": f'<div>Here is the screenshot you asked for:</div><img alt="Journal-index-screenshot.png" '
                f'width="446" height="137" src="{_CDN}/Journal-index-screenshot.png"/>',
        "_embedded": {"attachments": []},
    }]
    [image] = extract_customer_attachments(threads)
    assert image["inline"] and image["is_image"] and not image["is_blank"]
    assert image["url"].endswith("Journal-index-screenshot.png")
    assert "primary evidence" in describe_attachments([image])


def test_signature_logo_and_icons_do_not_block(inline_assets):
    body = (
        "<div>The index has an extra page.</div><p>Thank you,</p>"
        f'<div class="helpscout-signature"><img alt="reader-signature.png" src="{_CDN}/reader-signature.png" /></div>'
        f'<img alt="linkedin" width="16" height="16" src="{_CDN}/li.png" />'
        '<img src="https://evil.example.com/inline/x.png" />'
    )
    images = extract_inline_images({"body": body})
    assert [(i["filename"], i["is_image"]) for i in images] == [("reader-signature.png", False)]
    assert "signature logo" in describe_attachments(images)


def test_inline_image_url_must_be_help_scout_cdn(inline_assets):
    assert is_helpscout_inline_image_url(f"{_CDN}/a.png")
    assert not is_helpscout_inline_image_url("http://assets.woodland.example/inline/a.png")
    assert not is_helpscout_inline_image_url("https://assets.woodland.example/other/a.png")
    assert not is_helpscout_inline_image_url("https://assets.woodland.example.evil.com/inline/a.png")


@pytest.mark.asyncio
async def test_helpscout_webhook_methods_are_typed_bounded_and_use_documented_shapes(
    monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from services.helpscout import HelpScoutClient, HelpScoutWebhook, MutationResult

    callback_url = "https://mailbox.woodland.example/events"
    client = HelpScoutClient("client", "credential")
    response = SimpleNamespace(
        status_code=200,
        json=lambda: {
            "_embedded": {
                "webhooks": [
                    {
                        "id": 12,
                        "url": callback_url,
                        "state": "enabled",
                        "events": ["convo.created"],
                        "notification": False,
                        "payloadVersion": "V3",
                        "label": "Supa",
                        "mailboxIds": [7],
                    }
                ]
            },
            "page": {"number": 1, "totalPages": 1, "totalElements": 1},
        },
    )
    monkeypatch.setattr(client, "_read_bounded", AsyncMock(return_value=response))
    mutations = []

    async def fake_mutation(method, endpoint, **kwargs):
        mutations.append((method, endpoint, kwargs))
        return MutationResult(ok=True, status_code=kwargs["success_status"])

    monkeypatch.setattr(client, "_mutation_result", fake_mutation)

    hooks = await client.list_webhooks()
    create = await client.create_webhook(
        url=callback_url,
        events=("convo.created",),
        secret="z" * 40,
        notification=False,
        payload_version="V3",
        label="Supa",
        mailbox_ids=(7,),
    )
    update = await client.update_webhook(
        12,
        url=callback_url,
        events=("convo.created",),
        secret="z" * 40,
        notification=False,
        payload_version="V3",
        label="Supa",
        mailbox_ids=(7,),
    )
    await client.close()

    assert hooks == [HelpScoutWebhook(
        id=12,
        url=callback_url,
        state="enabled",
        events=("convo.created",),
        notification=False,
        payload_version="V3",
        label="Supa",
        mailbox_ids=(7,),
    )]
    assert create.ok and update.ok
    assert [(method, endpoint) for method, endpoint, _ in mutations] == [
        ("POST", "/webhooks"),
        ("PUT", "/webhooks/12"),
    ]
    assert mutations[0][2]["success_status"] == 201
    assert mutations[1][2]["success_status"] == 204
    assert mutations[0][2]["json"]["payloadVersion"] == "V3"
    assert mutations[1][2]["json"]["mailboxIds"] == [7]


def test_product_detection_prefers_modern_variant_before_the_base(journal_products):
    from services.themes import detect_theme

    assert detect_theme(tags=["trail-atlas", "field-journal"]) == "field-journal"
    assert detect_theme(tags=["trail-atlas"]) == "trail-atlas"
