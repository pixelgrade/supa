from datetime import datetime, timezone

import pytest

from db.database import init_db, fetch_scheduled_pending_sends_by_conversation
from services.processor import process_auto_send
from services.autosend_scheduler import AutoSendConfig
from tests.test_processor_confidence_loop import FakeHelpScout


async def test_customer_change_before_queue_cannot_rebind_old_answer(tmp_path):
    path = str(tmp_path / "audit.db")
    init_db(path)
    hs = FakeHelpScout()
    hs.conversation_payloads[987] = {"id": 987, "_embedded": {"threads": [
        {"id": 2, "type": "customer", "createdAt": "2026-11-17T09:00:00Z"}]}}
    with pytest.raises(RuntimeError, match="customer message changed"):
        await process_auto_send(helpscout=hs, conversation_id=987, customer_id=33,
            response_html="<p>Answer for message1.</p>", tags_to_add=[], existing_tags=[], confidence_score=8,
            confidence_provider="claude", db_path=path, expected_customer_thread_id=1,
            config=AutoSendConfig(300, 1200, 30, True), now=datetime.now(timezone.utc))
    assert hs.sent_replies == []
    assert hs.drafts == []
    assert fetch_scheduled_pending_sends_by_conversation(db_path=path, conversation_id="987") == []
