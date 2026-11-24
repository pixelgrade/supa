from unittest.mock import AsyncMock, Mock

import pytest

from services.processor import TicketProcessor


@pytest.mark.parametrize("trigger", ["new_conversation", "customer_reply", "note_trigger"])
async def test_ignored_conversation_is_not_classified_or_changed(monkeypatch, trigger):
    processor = TicketProcessor.__new__(TicketProcessor)
    processor.helpscout = AsyncMock()
    read_log = Mock(side_effect=AssertionError("Ignored ticket reached the processing audit lookup"))
    classify = AsyncMock(return_value="spam")
    monkeypatch.setattr("services.processor.get_ticket_log_by_helpscout_id", read_log)
    monkeypatch.setattr("services.processor.classify_ticket", classify)
    monkeypatch.setattr("services.processor.log_ticket", Mock())
    payload = {"id": 900, "subject": "Journal entry", "tags": [{"tag": "supa-ignore"}],
               "_embedded": {"threads": [{"type": "customer", "body": "Please review my site"}]}}
    result = await processor.process_ticket(payload, trigger=trigger)
    assert result["status"] == "skipped"
    assert result["response_posted"] is False
    read_log.assert_not_called()
    classify.assert_not_awaited()
    assert processor.helpscout.mock_calls == []
