from services.live_runs import LiveRunsRegistry
from services.ticket_runs import run_ticket_job


class _FailingHelpScout:
    async def get_conversation(self, conversation_id: int):
        raise RuntimeError(f"fetch failed for {conversation_id}")


class _FailingProcessor:
    def __init__(self):
        self.helpscout = _FailingHelpScout()


async def test_run_ticket_job_marks_live_run_failed_when_fetch_raises():
    registry = LiveRunsRegistry(retention_seconds=60)
    run_id = registry.start_run(ticket_id=41, source="run")

    result = await run_ticket_job(
        41,
        run_id,
        processor=_FailingProcessor(),
        registry=registry,
    )

    assert result["status"] == "error"
    assert "fetch failed for 41" in result["error"]

    run = registry.get_run(run_id)
    assert run is not None
    assert run["status"] == "failed"
    assert run["stage"] == "failed"

    events = registry.get_events(run_id, after_index=-1)
    assert [event["type"] for event in events] == ["run_started", "step", "run_failed"]
    assert "fetch failed for 41" in events[-1]["detail"]
