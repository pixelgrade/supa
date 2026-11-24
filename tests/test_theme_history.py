from unittest.mock import AsyncMock

import pytest

from services import theme_history, themes
from agents.support_agent import theme_label


@pytest.fixture
def books(monkeypatch):
    aliases = {"field journal": "field-journal", "trail atlas": "trail-atlas"}
    for module in (theme_history, themes):
        monkeypatch.setattr(module, "THEMES", aliases)
        monkeypatch.setattr(module, "VALID_SLUGS", set(aliases.values()))
    client = AsyncMock()
    client.search_conversations.return_value = [
        {"id": 31, "number": 101, "createdAt": "2023-03-01", "tags": [{"tag": "field-journal"}]},
        {"id": 29, "number": 99, "createdAt": "2023-02-01", "tags": [{"tag": "trail-atlas"}]},
    ]
    return client


async def test_history_excludes_current_conversation_and_marks_inference(books):
    hint = await theme_history.infer_theme_from_history(
        helpscout=books, customer_email="reader@woodland.example", exclude_id=31, customer_data=None)
    assert (hint.slug, hint.source) == ("trail-atlas", "past tickets")
    assert "hypothesis" in hint.context_block() and "#99" in hint.context_block()
    assert "unconfirmed" in theme_label(hint.slug, hint.source)
    books.search_conversations.assert_awaited_once_with(
        'email:"reader@woodland.example"', status="all", limit=10)


async def test_latest_licence_wins_and_search_failure_is_harmless(books):
    books.search_conversations.side_effect = TimeoutError()
    data = {"licenses": {"22": {"licenses": [{"product_title": "Field Journal"}]},
                         "12": {"licenses": [{"product_title": "Trail Atlas"}]}}}
    async def hint(records):
        return await theme_history.infer_theme_from_history(
            helpscout=books, customer_email="reader@woodland.example", exclude_id=31, customer_data=records)
    assert (await hint(data)).slug == "field-journal"
    assert await hint(None) is None
