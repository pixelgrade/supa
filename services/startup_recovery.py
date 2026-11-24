from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
import asyncio
import logging

from db.database import get_ticket_log_by_helpscout_id
from services.helpscout import get_latest_note_thread, is_supa_ignored
from services.live_runs import ActivityReporter, get_live_runs_registry
from services.runtime_paths import get_supa_db_path


logger = logging.getLogger(__name__)
DEFAULT_STARTUP_NOTE_RECOVERY_MINUTES = 15
DEFAULT_STARTUP_CONVERSATION_RECOVERY_MINUTES = 30
DEFAULT_PERIODIC_RECOVERY_INTERVAL_SECONDS = 300
DEFAULT_DB_PATH = get_supa_db_path()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _parse_db_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


async def _run_recovery_payload_job(
    payload: dict[str, Any],
    run_id: str,
    *,
    trigger: str,
    processor,
    registry,
) -> dict[str, Any]:
    activity = ActivityReporter(registry, run_id)
    try:
        result = await processor.process_ticket(payload, trigger=trigger, activity=activity)
    except TypeError as exc:
        if "activity" not in str(exc):
            raise
        result = await processor.process_ticket(payload, trigger=trigger)

    if result.get("status") == "error":
        activity.fail("Run failed", result.get("error") or "Unknown error")
    else:
        activity.complete("Run finished", result=result)
    return result


def _get_latest_customer_thread(threads: list[dict[str, Any]]) -> dict[str, Any] | None:
    customer_threads = [
        thread
        for thread in threads
        if thread.get("type") == "customer" or (thread.get("createdBy", {}) or {}).get("type") == "customer"
    ]
    if not customer_threads:
        return None

    threads_with_timestamps = [thread for thread in customer_threads if thread.get("createdAt")]
    if threads_with_timestamps:
        return max(threads_with_timestamps, key=lambda thread: thread.get("createdAt", ""))

    return customer_threads[0]


def _is_recent_human_note(note_thread: dict[str, Any] | None, *, now: datetime, cutoff: datetime) -> bool:
    if not note_thread:
        return False

    source_type = (note_thread.get("source", {}) or {}).get("type")
    if source_type == "api":
        return False

    created_by_type = (note_thread.get("createdBy", {}) or {}).get("type")
    if created_by_type and created_by_type != "user":
        return False

    note_created_at = _parse_timestamp(note_thread.get("createdAt"))
    if note_created_at is None:
        return False

    return cutoff <= note_created_at <= now


def _is_recent_customer_thread(customer_thread: dict[str, Any] | None, *, now: datetime, cutoff: datetime) -> bool:
    if not customer_thread:
        return False

    customer_created_at = _parse_timestamp(customer_thread.get("createdAt"))
    if customer_created_at is None:
        return False

    return cutoff <= customer_created_at <= now


def _was_processed_after_timestamp(helpscout_id: int, processed_threshold: datetime, *, db_path: str) -> bool:
    row = get_ticket_log_by_helpscout_id(db_path, str(helpscout_id))
    if not row:
        return False

    status = (row.get("status") or "").lower()
    if status not in {"processed", "warning"}:
        return False

    processed_at = _parse_db_timestamp(row.get("created_at"))
    if processed_at is None:
        return False

    return processed_at >= processed_threshold


async def run_startup_note_recovery(
    *,
    processor=None,
    registry=None,
    now: datetime | None = None,
    window_minutes: int = DEFAULT_STARTUP_NOTE_RECOVERY_MINUTES,
    limit: int = 100,
    db_path: str | None = None,
) -> list[int]:
    if processor is None:
        from services.processor import get_processor

        active_processor = get_processor()
    else:
        active_processor = processor
    active_registry = registry or get_live_runs_registry()
    current_time = now or _utc_now()
    cutoff = current_time - timedelta(minutes=window_minutes)
    active_db_path = db_path or DEFAULT_DB_PATH

    conversations = await active_processor.helpscout.get_recent_conversations(limit=limit)
    recovered_ids: list[int] = []

    for conversation in conversations:
        if is_supa_ignored(conversation):
            continue
        conversation_id = conversation.get("id")
        if not conversation_id:
            continue

        threads = conversation.get("_embedded", {}).get("threads", [])
        note_thread = get_latest_note_thread(threads)
        if not _is_recent_human_note(note_thread, now=current_time, cutoff=cutoff):
            continue

        note_created_at = _parse_timestamp(note_thread.get("createdAt"))
        if note_created_at is None:
            continue

        if _was_processed_after_timestamp(conversation_id, note_created_at, db_path=active_db_path):
            continue

        run_id = active_registry.start_run(ticket_id=conversation_id, source="startup_recovery")
        active_registry.emit(
            run_id,
            stage="recovery",
            message="Recovering missed note-trigger after startup",
            detail=f"note_created_at={note_thread.get('createdAt', '')}",
        )
        await _run_recovery_payload_job(
            conversation,
            run_id,
            trigger="note_trigger",
            processor=active_processor,
            registry=active_registry,
        )
        recovered_ids.append(conversation_id)

    if recovered_ids:
        logger.info("Startup note recovery processed tickets: %s", recovered_ids)
    else:
        logger.info("Startup note recovery found no missed note-trigger tickets")

    return recovered_ids


async def run_startup_conversation_recovery(
    *,
    processor=None,
    registry=None,
    now: datetime | None = None,
    window_minutes: int = DEFAULT_STARTUP_CONVERSATION_RECOVERY_MINUTES,
    limit: int = 100,
    db_path: str | None = None,
) -> list[int]:
    if processor is None:
        from services.processor import get_processor

        active_processor = get_processor()
    else:
        active_processor = processor
    active_registry = registry or get_live_runs_registry()
    current_time = now or _utc_now()
    cutoff = current_time - timedelta(minutes=window_minutes)
    active_db_path = db_path or DEFAULT_DB_PATH

    conversations = await active_processor.helpscout.get_recent_conversations(limit=limit)
    recovered_ids: list[int] = []

    for conversation in conversations:
        if is_supa_ignored(conversation):
            continue
        conversation_id = conversation.get("id")
        if not conversation_id:
            continue

        threads = conversation.get("_embedded", {}).get("threads", [])
        customer_thread = _get_latest_customer_thread(threads)
        if not _is_recent_customer_thread(customer_thread, now=current_time, cutoff=cutoff):
            continue

        customer_created_at = _parse_timestamp(customer_thread.get("createdAt"))
        if customer_created_at is None:
            continue

        if _was_processed_after_timestamp(conversation_id, customer_created_at, db_path=active_db_path):
            continue

        existing_row = get_ticket_log_by_helpscout_id(active_db_path, str(conversation_id))
        trigger = "customer_reply" if existing_row else "new_conversation"
        run_id = active_registry.start_run(ticket_id=conversation_id, source="startup_recovery")
        active_registry.emit(
            run_id,
            stage="recovery",
            message="Recovering missed customer conversation after startup",
            detail=f"customer_created_at={customer_thread.get('createdAt', '')} trigger={trigger}",
        )
        await _run_recovery_payload_job(
            conversation,
            run_id,
            trigger=trigger,
            processor=active_processor,
            registry=active_registry,
        )
        recovered_ids.append(conversation_id)

    if recovered_ids:
        logger.info("Startup conversation recovery processed tickets: %s", recovered_ids)
    else:
        logger.info("Startup conversation recovery found no missed customer conversations")

    return recovered_ids


async def run_startup_recovery(*, processor=None, registry=None) -> dict[str, list[int]]:
    note_ids = await run_startup_note_recovery(processor=processor, registry=registry)
    conversation_ids = await run_startup_conversation_recovery(processor=processor, registry=registry)
    return {
        "note_recovery": note_ids,
        "conversation_recovery": conversation_ids,
    }


async def run_periodic_recovery(
    *,
    processor=None,
    registry=None,
    now: datetime | None = None,
    note_window_minutes: int = DEFAULT_STARTUP_NOTE_RECOVERY_MINUTES,
    conversation_window_minutes: int = DEFAULT_STARTUP_CONVERSATION_RECOVERY_MINUTES,
    interval_seconds: int = DEFAULT_PERIODIC_RECOVERY_INTERVAL_SECONDS,
    iterations: int | None = None,
    sleep_func=asyncio.sleep,
) -> None:
    cycle = 0

    while iterations is None or cycle < iterations:
        try:
            await run_startup_note_recovery(
                processor=processor,
                registry=registry,
                now=now,
                window_minutes=note_window_minutes,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Periodic note recovery pass failed")

        try:
            await run_startup_conversation_recovery(
                processor=processor,
                registry=registry,
                now=now,
                window_minutes=conversation_window_minutes,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Periodic conversation recovery pass failed")

        cycle += 1
        if iterations is not None and cycle >= iterations:
            break

        await sleep_func(interval_seconds)


def schedule_startup_recovery(*, processor=None, registry=None) -> asyncio.Task:
    return asyncio.create_task(run_startup_recovery(processor=processor, registry=registry))


def schedule_periodic_recovery(*, processor=None, registry=None) -> asyncio.Task:
    return asyncio.create_task(
        run_periodic_recovery(
            processor=processor,
            registry=registry,
            interval_seconds=DEFAULT_PERIODIC_RECOVERY_INTERVAL_SECONDS,
        )
    )
