from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


@patch("api.dashboard.get_live_runs_registry")
def test_live_runs_endpoint_lists_active_runs(mock_get_registry, monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)

    registry = Mock()
    registry.list_active_runs.return_value = [
        {
            "run_id": "run-123",
            "ticket_id": 601,
            "source": "webhook",
            "stage": "investigate",
            "status": "running",
        }
    ]
    mock_get_registry.return_value = registry

    response = client.get("/dashboard/live-runs")

    assert response.status_code == 200
    body = response.json()
    assert body["runs"][0]["run_id"] == "run-123"
    assert body["runs"][0]["source"] == "webhook"


@patch("api.dashboard.get_live_runs_registry")
def test_live_run_events_endpoint_returns_incremental_events(mock_get_registry, monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)

    registry = Mock()
    registry.get_events.return_value = [
        {"type": "step", "message": "Searching knowledge base"},
        {"type": "run_completed", "message": "Run completed"},
    ]
    registry.get_run.return_value = None
    mock_get_registry.return_value = registry

    response = client.get("/dashboard/live-runs/run-123/events?after=2")

    assert response.status_code == 200
    assert response.json()["events"][0]["message"] == "Searching knowledge base"
    registry.get_events.assert_called_once_with("run-123", after_index=2)


@patch("api.dashboard.asyncio.create_task")
@patch("api.dashboard.run_ticket_job", new_callable=AsyncMock)
@patch("api.dashboard.get_live_runs_registry")
def test_run_ticket_returns_run_id(mock_get_registry, mock_run_ticket_job, mock_create_task, monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    mock_create_task.side_effect = lambda coro: coro.close()

    registry = Mock()
    registry.start_run.return_value = "run-456"
    mock_get_registry.return_value = registry

    response = client.post("/dashboard/run-ticket", json={"conversation_id": 601})

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run-456"
    assert body["status"] == "queued"

    registry.start_run.assert_called_once_with(ticket_id=601, source="run")
    mock_run_ticket_job.assert_called_once_with(601, "run-456")
    mock_create_task.assert_called_once()
