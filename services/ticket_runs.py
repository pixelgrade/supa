from __future__ import annotations

from typing import Any

from services.live_runs import ActivityReporter, LiveRunsRegistry, get_live_runs_registry
from services.processor import get_processor


def _result_message(result: dict[str, Any]) -> str:
    status = result.get("status", "processed")
    delivery_mode = result.get("delivery_mode")
    classification = result.get("classification")
    parts = [status]
    if classification:
        parts.append(classification)
    if delivery_mode:
        parts.append(delivery_mode)
    return " · ".join(parts)


async def run_payload_job(
    payload: dict[str, Any],
    run_id: str,
    *,
    trigger: str,
    source: str,
    processor=None,
    registry: LiveRunsRegistry | None = None,
) -> dict[str, Any]:
    active_processor = processor or get_processor()
    active_registry = registry or get_live_runs_registry()
    activity = ActivityReporter(active_registry, run_id)
    try:
        result = await active_processor.process_ticket(payload, trigger=trigger, activity=activity)
    except TypeError as exc:
        if "activity" not in str(exc):
            raise
        result = await active_processor.process_ticket(payload, trigger=trigger)

    if result.get("status") == "error":
        activity.fail("Run failed", result.get("error") or "Unknown error")
    else:
        activity.complete("Run finished", result={"summary": _result_message(result), **result})
    return result


async def run_ticket_job(
    conversation_id: int,
    run_id: str,
    *,
    trigger: str = "manual_run",
    source: str = "run",
    processor=None,
    registry: LiveRunsRegistry | None = None,
) -> dict[str, Any]:
    active_processor = processor or get_processor()
    active_registry = registry or get_live_runs_registry()
    activity = ActivityReporter(active_registry, run_id)

    activity.step("fetch", "Fetching conversation from HelpScout")
    try:
        payload = await active_processor.helpscout.get_conversation(conversation_id)
    except Exception as exc:
        message = str(exc) or f"Failed to fetch conversation {conversation_id}"
        activity.fail("Run failed", message)
        return {"status": "error", "error": message}

    return await run_payload_job(
        payload,
        run_id,
        trigger=trigger,
        source=source,
        processor=active_processor,
        registry=active_registry,
    )
