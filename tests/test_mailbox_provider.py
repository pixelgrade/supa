import pytest

from services import mailbox_provider
from services.helpscout import (
    HelpScoutClient,
    HelpScoutRedirectError,
    _parse_merge_location,
)


def configure(monkeypatch):
    monkeypatch.setenv("SUPA_MAILBOX_ORIGIN", "https://inbox.example.test")
    monkeypatch.setenv("SUPA_MAILBOX_CLIENT_ID", "native-id")
    monkeypatch.setenv("SUPA_MAILBOX_CLIENT_SECRET", "native-secret")


@pytest.mark.asyncio
async def test_new_provider_uses_dedicated_credentials_and_keeps_legacy_citations(
    monkeypatch,
):
    configure(monkeypatch)
    client = HelpScoutClient("old-id", "old-secret")
    try:
        assert client.BASE_URL == "https://inbox.example.test/v2"
        assert client.TOKEN_URL == "https://inbox.example.test/v2/oauth2/token"
        assert (client.client_id, client.client_secret) == (
            "native-id",
            "native-secret",
        )
        assert (
            mailbox_provider.conversation_url(21, thread_id=41)
            == "https://inbox.example.test/conversation/21#thread-41"
        )
        for origin in ("https://inbox.example.test", mailbox_provider.VENDOR_WEB):
            assert mailbox_provider.matches_citation(
                origin + "/conversation/21#thread-41", 21, 41
            )
        assert not mailbox_provider.matches_citation(
            "https://bad.example/conversation/21#thread-41", 21, 41
        )
        assert (
            _parse_merge_location(
                "https://inbox.example.test/v2/conversations/99", 1, client.BASE_URL
            )
            == 99
        )
        with pytest.raises(HelpScoutRedirectError):
            _parse_merge_location(
                "https://bad.example/v2/conversations/99", 1, client.BASE_URL
            )
    finally:
        await client.close()


def test_native_provider_never_reuses_vendor_secret(monkeypatch):
    configure(monkeypatch)
    monkeypatch.delenv("SUPA_MAILBOX_CLIENT_SECRET")
    with pytest.raises(ValueError, match="credentials"):
        HelpScoutClient("old-id", "old-secret")


@pytest.mark.parametrize(
    "value",
    [
        "http://inbox.example.test",
        "https://" + "reader:fixture" + "@inbox.example.test",
        "https://inbox.example.test?token=123",
        "https://inbox.example.test/foo",
        "https://inbox.example.test/#fragment",
    ],
)
def test_invalid_native_origin_rejected(monkeypatch, value):
    monkeypatch.setenv("SUPA_MAILBOX_ORIGIN", value)
    with pytest.raises(ValueError, match="origin"):
        mailbox_provider.web_origin()


@pytest.mark.asyncio
async def test_inline_images_fetch_authenticated_local_copy(monkeypatch):
    import hashlib
    import httpx

    configure(monkeypatch)
    monkeypatch.setattr("services.helpscout.HELPSCOUT_INLINE_IMAGE_HOSTS", {"assets.woodland.example"})
    client = HelpScoutClient("vendor-id", "vendor-secret")
    await client._http.aclose()
    source = "https://assets.woodland.example/inline/123/photo.png"
    key = hashlib.sha256(source.encode()).hexdigest()
    calls = []

    def fixture(request):
        calls.append(str(request.url))
        if request.url.path.endswith("oauth2/token"):
            assert (
                b"native-secret" in request.content
                and b"vendor-secret" not in request.content
            )
            return httpx.Response(
                200, json={"access_token": "session", "expires_in": 7200}
            )
        assert request.url.path == "/v2/inline/" + key
        assert request.headers["Authorization"] == "Bearer session"
        return httpx.Response(200, content=b"pixels")

    client._http = httpx.AsyncClient(transport=httpx.MockTransport(fixture))
    try:
        assert await client.get_inline_image_data(source, max_bytes=10) == b"pixels"
        assert calls == [client.TOKEN_URL, client.BASE_URL + "/inline/" + key]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_native_uncertain_receipt_preserves_reconcile_before_retry(monkeypatch):
    import httpx

    configure(monkeypatch)
    client = HelpScoutClient("", "")
    await client._http.aclose()

    def fixture(request):
        if request.url.path.endswith("oauth2/token"):
            return httpx.Response(
                200, json={"access_token": "session", "expires_in": 7200}
            )
        return httpx.Response(504, headers={"X-Inbox-Delivery-State": "uncertain"})

    client._http = httpx.AsyncClient(transport=httpx.MockTransport(fixture))
    try:
        result = await client.send_reply_result(17, "<p>reply</p>")
        assert not result.ok and result.outcome_unknown
    finally:
        await client.close()
