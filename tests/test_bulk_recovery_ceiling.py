from unittest.mock import AsyncMock, Mock

from api import dashboard


async def test_bulk_recovery_cannot_publish_another_sample(monkeypatch):
    conv = {"id": 900, "status": "active", "tags": [], "_embedded": {"threads": [
        {"type": "customer", "body": "Question", "id": 1, "createdAt": "2026-09-13T13:00:00Z"}]}}
    processor = Mock()
    processor.helpscout.get_conversation = AsyncMock(return_value=conv)
    processor.process_ticket = AsyncMock(return_value={"delivery_mode": "draft", "response_posted": True, "confidence_score": 8})
    monkeypatch.setattr(dashboard, "get_processor", lambda: processor)
    monkeypatch.setattr(dashboard, "find_casualties", lambda *a: [{"helpscout_id": "900"}])
    test_ticket = AsyncMock(side_effect=AssertionError("Bulk writes must use canonical processor"))
    monkeypatch.setattr(dashboard, "test_ticket", test_ticket)
    result = await dashboard.reprocess_stuck(dashboard.ReprocessRequest(send=True), _=True)
    processor.process_ticket.assert_awaited_once_with(conv, trigger="manual_run", delivery_ceiling="draft")
    assert result["results"][0]["action"] == "drafted"
