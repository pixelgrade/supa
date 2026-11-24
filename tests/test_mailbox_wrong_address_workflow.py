"""The mailbox client runs the one wrong-address manual workflow, once per conversation."""

import asyncio
from unittest.mock import AsyncMock

import httpx

from services.mailbox_provider import VENDOR_API
from services.helpscout import HelpScoutClient

NATIVE = {"SUPA_MAILBOX_ORIGIN": "https://mailbox.woodland.example",
          "SUPA_MAILBOX_CLIENT_ID": "fixture", "SUPA_MAILBOX_CLIENT_SECRET": "fixture"}


def _client(monkeypatch, payload=None, status=200):
    monkeypatch.setattr("services.installation.OPERATIONS_ORIGIN", NATIVE["SUPA_MAILBOX_ORIGIN"])
    client = HelpScoutClient(None, None, environ=NATIVE)
    read = AsyncMock(return_value=httpx.Response(status, json=payload if payload is not None else {}))
    monkeypatch.setattr(client, "_read_bounded", read)
    return client, read


def test_wrong_address_workflow_runs_the_one_matching_manual_rule_once(monkeypatch):
    async def run():
        rules = {"_embedded": {"workflows": [
            {"id": 1, "name": "Email sent to the wrong address", "type": "automatic", "status": "active"},
            {"id": 42, "name": "Email sent to the wrong address", "type": "manual", "status": "active"},
            {"id": 3, "name": "Spammy email reach", "type": "manual", "status": "active"}]}}
        client, read = _client(monkeypatch, rules)
        sent = []

        async def mutation(method, endpoint, **kwargs):
            sent.append((method, endpoint, kwargs))
            from services.helpscout import MutationResult
            return MutationResult(ok=True, status_code=200)

        monkeypatch.setattr(client, "_mutation_result", mutation)
        try:
            assert (await client.run_wrong_address_workflow_result(77)).ok
            read.assert_awaited_once_with("/workflows", max_bytes=262144)
            assert sent == [("POST", "/conversations/77/workflows/42",
                             {"success_status": 200, "headers": {"Idempotency-Key": "supa-wrong-address-77"}})]
            # No rule, or more than one, sends nothing.
            for broken in ({"_embedded": {"workflows": []}}, {"_embedded": {"workflows": rules["_embedded"]["workflows"][1:2] * 2}}):
                sent.clear()
                read.return_value = httpx.Response(200, json=broken)
                result = await client.run_wrong_address_workflow_result(77)
                assert not result.ok and result.detail == "wrong_address_workflow_not_found" and sent == []
            read.return_value = httpx.Response(500, json={})
            assert (await client.run_wrong_address_workflow_result(77)).detail == "workflow_list_unavailable"
            client.BASE_URL = VENDOR_API
            assert (await client.run_wrong_address_workflow_result(77)).detail == "native_mailbox_required"
        finally:
            await client.close()
    asyncio.run(run())


def test_the_idempotency_key_reaches_the_mailbox(monkeypatch):
    """The real request carries the key, so a retried call cannot send a second email."""
    async def run():
        rules = {"_embedded": {"workflows": [
            {"id": 42, "name": "Email sent to the wrong address", "type": "manual", "status": "active"}]}}
        client, _ = _client(monkeypatch, rules)
        seen = []

        async def request(method, endpoint, **kwargs):
            seen.append((method, endpoint, dict(kwargs.get("headers") or {})))
            return httpx.Response(200, json={"activityThreadId": 1, "replyThreadId": 2})

        monkeypatch.setattr(client, "_request", request)
        try:
            assert (await client.run_wrong_address_workflow_result(77)).ok
            assert (await client.run_wrong_address_workflow_result(77)).ok
            assert [call[:2] for call in seen] == [("POST", "/conversations/77/workflows/42")] * 2
            assert all(call[2]["Idempotency-Key"] == "supa-wrong-address-77" for call in seen)
        finally:
            await client.close()
    asyncio.run(run())
