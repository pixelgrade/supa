import httpx
import pytest

from services.helpscout import HelpScoutClient


@pytest.mark.asyncio
async def test_create_outbound_uses_documented_reply_draft(monkeypatch):
    client = HelpScoutClient("id", "secret")
    seen = []
    async def request(method, path, **kwargs):
        seen.append((method, path, kwargs))
        return httpx.Response(201, headers={"Resource-ID": "789"})
    monkeypatch.setattr(client, "_request", request)
    result = await client.create_conversation_result(
        mailbox_id=3, customer_id=4, subject="Private review", text="<p>Hello</p>")
    assert result.ok and result.resource_id == 789
    payload = seen[0][2]["json"]
    assert seen[0][:2] == ("POST", "/conversations")
    assert payload["tags"] == ["supa-ignore"]
    assert payload["threads"] == [{"type": "reply", "customer": {"id": 4},
                                    "text": "<p>Hello</p>", "draft": True}]


@pytest.mark.asyncio
async def test_delete_is_whole_conversation_and_timeout_is_unknown(monkeypatch):
    client = HelpScoutClient("id", "secret")
    seen = []
    async def request(method, path, **kwargs):
        seen.append((method, path))
        if len(seen) == 1:
            return httpx.Response(204)
        raise httpx.ReadTimeout("private response")
    monkeypatch.setattr(client, "_request", request)
    assert (await client.delete_conversation_result(789)).ok
    unknown = await client.delete_conversation_result(789)
    assert unknown.outcome_unknown and not unknown.ok
    assert seen == [("DELETE", "/conversations/789")] * 2
