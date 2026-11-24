"""Past tickets come from the owned mailbox's ranked answers when it is selected."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from services.mailbox_provider import VENDOR_API
from services.helpscout import HelpScoutClient
from services.past_tickets import find_similar_tickets, format_past_tickets_context

NATIVE = {"SUPA_MAILBOX_ORIGIN": "https://mailbox.woodland.example",
          "SUPA_MAILBOX_CLIENT_ID": "fixture", "SUPA_MAILBOX_CLIENT_SECRET": "fixture"}


class _Ranked:
    def __init__(self, answer):
        self.answer, self.calls, self.searches = answer, [], 0

    async def precedents(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    async def search_conversations(self, **kwargs):
        self.searches += 1
        return []


async def test_ranked_answers_replace_the_keyword_search():
    client = _Ranked([{"id": 5, "subject": "Almanac: parallax", "replies": [{"body": "Update Journal Tools."}], "shared_tags": ["almanac"]},
                      {"id": 6, "subject": "Almanac: hero", "replies": [{"body": "Second"}], "shared_tags": ["almanac"]},
                      {"id": 7, "subject": "Third", "replies": [{"body": "Third"}], "shared_tags": ["almanac"]}])
    found = await find_similar_tickets(client, "almanac", "Parallax stopped", "The hero no longer moves", exclude_id=44, limit=2)
    assert [t["id"] for t in found] == [5, 6] and client.searches == 0
    assert client.calls == [{"conversation_id": 44, "text": "Parallax stopped\nThe hero no longer moves", "tag": "almanac", "limit": 2}]
    assert "Update Journal Tools." in format_past_tickets_context(found)


async def test_the_theme_filters_ranked_answers_and_an_empty_result_falls_back():
    client = _Ranked([{"id": 5, "subject": "Trail Atlas: parallax", "replies": [{"body": "x"}], "shared_tags": []},
                      {"id": 6, "subject": "Almanac: hero", "replies": [{"body": "y"}], "shared_tags": ["almanac"]}])
    found = await find_similar_tickets(client, "Almanac", "Parallax stopped", "The hero no longer moves", exclude_id=44, limit=2)
    assert [t["id"] for t in found] == [6] and client.searches == 0
    other = _Ranked([{"id": 5, "subject": "Trail Atlas: parallax", "replies": [{"body": "x"}], "shared_tags": []}])
    assert await find_similar_tickets(other, "almanac", "Parallax stopped", "The hero", exclude_id=44) == [] and other.searches == 1
    theme_less = _Ranked([{"id": 5, "subject": "Any", "replies": [{"body": "x"}], "shared_tags": []}])
    assert [t["id"] for t in await find_similar_tickets(theme_less, "", "Parallax stopped", "The hero", exclude_id=44)] == [5]


@pytest.mark.parametrize("answer", [None, [], RuntimeError("down")])
async def test_keyword_search_remains_the_fallback(answer):
    client = _Ranked(answer)
    assert await find_similar_tickets(client, "almanac", "Parallax stopped", "The hero no longer moves", exclude_id=44) == []
    assert client.searches == 1


def _client(monkeypatch, payload=None, status=200):
    monkeypatch.setattr("services.installation.OPERATIONS_ORIGIN", NATIVE["SUPA_MAILBOX_ORIGIN"])
    client = HelpScoutClient(None, None, environ=NATIVE)
    read = AsyncMock(return_value=httpx.Response(status, json=payload if payload is not None else {}))
    monkeypatch.setattr(client, "_read_bounded", read)
    return client, read


def test_precedents_request_and_shape(monkeypatch):
    async def run():
        client, read = _client(monkeypatch, {"terms": ["parallax"], "precedents": [
            {"id": 10, "subject": "Parallax", "answer": "Please update.", "answered": "2025-03-01T12:00:00Z", "rating": "great", "shared_tags": ["Almanac"]},
            {"id": "x", "subject": "Bad id", "answer": "ignored"}, {"id": 11, "subject": "No answer", "answer": " "}]})
        try:
            assert await client.precedents(conversation_id=44, tag="almanac", limit=30) == [
                {"id": 10, "subject": "Parallax", "replies": [{"body": "Please update.", "created_at": "2025-03-01T12:00:00Z"}], "rating": "great", "shared_tags": ["almanac"]}]
            read.assert_awaited_once_with("/precedents?limit=10&conversation=44&tag=almanac", max_bytes=262144)
            read.reset_mock()
            await client.precedents(text="sticky header & logo", limit=2)
            read.assert_awaited_once_with("/precedents?limit=2&q=sticky+header+%26+logo", max_bytes=262144)
            read.reset_mock()
            # An expanded query travels with the conversation it belongs to.
            await client.precedents(conversation_id=44, text="expanded words", limit=5)
            read.assert_awaited_once_with("/precedents?limit=5&conversation=44&q=expanded+words", max_bytes=262144)
            client.BASE_URL = VENDOR_API
            read.reset_mock()
            assert await client.precedents(conversation_id=44) is None
            read.assert_not_awaited()
        finally:
            await client.close()
    asyncio.run(run())


def test_precedents_failure_returns_none(monkeypatch):
    async def run():
        client, _ = _client(monkeypatch, {"message": "no"}, status=503)
        try:
            assert await client.precedents(conversation_id=44) is None
        finally:
            await client.close()
    asyncio.run(run())
