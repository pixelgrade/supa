from services.live_runs import LiveRunsRegistry


def test_registry_tracks_active_run_and_replays_events():
    registry = LiveRunsRegistry(retention_seconds=30)

    run_id = registry.start_run(ticket_id=601, source="test")
    registry.emit(run_id, stage="search", message="Searching knowledge base")

    active_runs = registry.list_active_runs()
    assert len(active_runs) == 1
    assert active_runs[0]["run_id"] == run_id
    assert active_runs[0]["stage"] == "search"

    events = registry.get_events(run_id, after_index=-1)
    assert [event["type"] for event in events] == ["run_started", "step"]
    assert events[1]["message"] == "Searching knowledge base"


def test_completed_run_leaves_active_list_but_keeps_terminal_event_temporarily():
    registry = LiveRunsRegistry(retention_seconds=30)

    run_id = registry.start_run(ticket_id=601, source="webhook")
    registry.complete(run_id, message="Run completed", result={"status": "ok"})

    assert registry.list_active_runs() == []

    events = registry.get_events(run_id, after_index=-1)
    assert events[-1]["type"] == "run_completed"
    assert events[-1]["result"] == {"status": "ok"}

    registry.cleanup_expired(now=events[-1]["timestamp_epoch"] + 31)
    assert registry.get_run(run_id) is None
