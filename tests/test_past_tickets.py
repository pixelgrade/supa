from unittest.mock import AsyncMock

import httpx

from services.past_tickets import find_similar_tickets


async def test_find_similar_tickets_returns_empty_list_when_search_times_out():
    helpscout_client = type("HelpScoutStub", (), {})()
    helpscout_client.search_conversations = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

    results = await find_similar_tickets(
        helpscout_client=helpscout_client,
        theme_tag="",
        subject="Journal entry number",
        message="The page has no number",
        exclude_id=72,
        limit=2,
    )

    assert results == []
