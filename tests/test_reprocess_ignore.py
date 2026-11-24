from unittest.mock import AsyncMock, Mock

from api import dashboard


async def test_bulk_recovery_does_not_test_or_send_ignored_conversations(monkeypatch):
    processor = Mock()
    processor.helpscout.get_conversation = AsyncMock(return_value={
        "id": 900, "status": "active", "tags": [{"tag": "supa-ignore"}],
        "_embedded": {"threads": [{"type": "customer", "body": "My site", "id": 1,
                                   "createdAt": "2026-09-13T13:00:00Z"}]}})
    monkeypatch.setattr(dashboard, "get_processor", lambda: processor)
    monkeypatch.setattr(dashboard, "find_casualties", lambda *a: [{"helpscout_id": "900"}])
    test_ticket = AsyncMock(return_value=Mock(response_html=""))
    post_cached = AsyncMock()
    monkeypatch.setattr(dashboard, "test_ticket", test_ticket)
    monkeypatch.setattr(dashboard, "post_cached", post_cached)
    result = await dashboard.reprocess_stuck(dashboard.ReprocessRequest(send=True), _=True)
    test_ticket.assert_not_awaited()
    post_cached.assert_not_awaited()
    assert "supa-ignore" in str(result)
