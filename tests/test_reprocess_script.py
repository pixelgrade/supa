import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from db.database import init_db
from scripts import reprocess_tickets
from tests.lease_recording import RecordingLeaseFactory


@pytest.mark.asyncio
async def test_non_support_close_and_tags_share_a_guarded_lease(monkeypatch):
    init_db(os.environ["SUPA_DB_PATH"])
    client = SimpleNamespace(
        search_conversations=AsyncMock(
            return_value=[{"id": 321, "subject": "Newsletter"}]
        ),
        get_conversation=AsyncMock(
            return_value={
                "id": 321,
                "subject": "Newsletter",
                "tags": [{"tag": "old"}],
                "_embedded": {
                    "threads": [{"type": "customer", "body": "subscribe"}]
                },
            }
        ),
        close_conversation=AsyncMock(return_value=True),
        add_tags=AsyncMock(return_value=True),
        close=AsyncMock(),
    )
    monkeypatch.setattr(
        "scripts.reprocess_tickets.HelpScoutClient", lambda **_kwargs: client
    )
    monkeypatch.setattr(
        "scripts.reprocess_tickets.classify_ticket",
        AsyncMock(return_value="marketing"),
    )
    leases = RecordingLeaseFactory()
    monkeypatch.setattr("scripts.reprocess_tickets.conversation_lease", leases)

    await reprocess_tickets.find_and_reprocess_tickets([99])

    assert client.get_conversation.await_count == 2
    leases.assert_guarded_count(321, 2)
    client.close.assert_awaited_once()
