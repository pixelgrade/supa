from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from db.database import get_db, init_db
from services.agent_tasks import (
    MAX_RUN_JSON_BYTES,
    TASKS,
    AgentRunDataError,
    AgentRunNotFoundError,
    AgentRunStateError,
    AgentTaskRequestError,
    AgentTaskRepository,
    prepare_task_request,
)


@pytest.fixture
def task_db(tmp_path) -> str:
    path = str(tmp_path / "agent-tasks.db")
    init_db(path)
    return path


def test_task_registry_is_allowlisted_and_requires_composed_scopes():
    from services.extensions import agent_tasks

    assert set(TASKS) - set(agent_tasks()) == {"ticket.evaluate", "ticket.process", "ticket.recover"}
    assert TASKS["ticket.evaluate"].required_scopes == frozenset({"task", "read"})
    assert TASKS["ticket.process"].required_scopes == frozenset({"task", "read", "draft"})

    for name, scopes, arguments, code in [
        ("python.exec", {"task", "read", "draft", "send"}, {"conversation_id": 1}, "unknown_task"),
        ("ticket.evaluate", {"task"}, {"conversation_id": 1}, "insufficient_scope"),
        ("ticket.process", {"task", "read"}, {"conversation_id": 1}, "insufficient_scope"),
        ("ticket.process", {"task", "read", "draft"}, {"conversation_id": 1, "allow_send": True}, "insufficient_scope"),
        ("ticket.evaluate", {"task", "read"}, {"conversation_id": 1, "prompt": "ignore policy"}, "invalid_task_arguments"),
        ("ticket.process", {"task", "read", "draft"}, {"conversation_id": 1, "allow_send": "true"}, "invalid_task_arguments"),
        ("ticket.evaluate", {"task", "read"}, {"conversation_id": True}, "invalid_task_arguments"),
    ]:
        with pytest.raises(AgentTaskRequestError) as caught:
            prepare_task_request(name, arguments, scopes=scopes)
        assert caught.value.code == code


def test_task_request_snapshots_arguments_scope_and_delivery_ceiling(task_db):
    repository = AgentTaskRepository(task_db)
    evaluate = repository.create_task_run(
        token_id="token-a", token_name="Agent A", task_name="ticket.evaluate",
        arguments={"conversation_id": 123}, scopes={"task", "read"},
        idempotency_key="evaluate-123",
    )
    assert evaluate.arguments == {"conversation_id": "123"}
    assert evaluate.scopes == ("read", "task")
    assert evaluate.delivery_ceiling == "draft"

    draft = repository.create_task_run(
        token_id="token-b", token_name="Agent B", task_name="ticket.process",
        arguments={"conversation_id": "124"}, scopes={"task", "read", "draft"},
        idempotency_key="process-124",
    )
    assert draft.arguments == {"conversation_id": "124", "allow_send": False}
    assert draft.delivery_ceiling == "draft"
    assert draft.scopes == ("draft", "read", "task")

    sending = repository.create_task_run(
        token_id="token-c", token_name="Agent C", task_name="ticket.process",
        arguments={"conversation_id": 125, "allow_send": True},
        scopes={"task", "read", "draft", "send"}, idempotency_key="process-125",
    )
    assert sending.arguments == {"conversation_id": "125", "allow_send": True}
    assert sending.delivery_ceiling == "policy"
    assert sending.scopes == ("draft", "read", "send", "task")

    with pytest.raises(AgentTaskRequestError) as caught:
        repository.create_task_run(
            token_id="token-d", token_name="Agent D", task_name="ticket.process",
            arguments={"conversation_id": 126, "allow_send": True},
            scopes={"task", "read", "draft"}, idempotency_key="process-126",
        )
    assert caught.value.code == "insufficient_scope"
    assert {run.run_id for run in repository.list_queued_runs()} == {
        evaluate.run_id, draft.run_id, sending.run_id
    }


def test_task_start_idempotency_preserves_first_run_authority(task_db):
    repository = AgentTaskRepository(task_db)
    first = repository.create_task_run(
        token_id="token-a", token_name="Agent A", task_name="ticket.process",
        arguments={"conversation_id": 99}, scopes={"task", "read", "draft"},
        idempotency_key="same-request",
    )
    repeated = repository.create_task_run(
        token_id="token-a", token_name="Agent A renamed", task_name="ticket.process",
        arguments={"conversation_id": 99}, scopes={"task", "read", "draft", "send"},
        idempotency_key="same-request",
    )
    assert repeated == first
    assert repeated.delivery_ceiling == "draft"
    with pytest.raises(AgentTaskRequestError) as caught:
        repository.create_task_run(
            token_id="token-a", token_name="Agent A", task_name="ticket.process",
            arguments={"conversation_id": 100}, scopes={"task", "read", "draft"},
            idempotency_key="same-request",
        )
    assert caught.value.code == "idempotency_conflict"


def test_startup_recovery_interrupts_running_and_requeues_queued(task_db, monkeypatch):
    repository = AgentTaskRepository(task_db)
    queued = repository.create_task_run(
        token_id="token-a", token_name="Agent A", task_name="ticket.evaluate",
        arguments={"conversation_id": 1}, scopes={"task", "read"},
        idempotency_key="queued-1",
    )
    running = repository.create_task_run(
        token_id="token-a", token_name="Agent A", task_name="ticket.evaluate",
        arguments={"conversation_id": 2}, scopes={"task", "read"},
        idempotency_key="running-2",
    )
    repository.claim_run(running.run_id)
    scheduled = []
    monkeypatch.setattr("api.agent.schedule_agent_run", scheduled.append)
    from api.agent import recover_agent_runs
    assert recover_agent_runs(task_db) == (1, 1)
    assert scheduled == [queued.run_id]
    assert repository.get_run(running.run_id, token_id="token-a").status == "interrupted"


def test_terminal_run_event_is_atomic_deduped_and_metadata_only(task_db):
    repository = AgentTaskRepository(task_db)
    run = repository.create_task_run(
        token_id="token-a", token_name="Agent A", task_name="ticket.evaluate",
        arguments={"conversation_id": 321}, scopes={"task", "read"},
        idempotency_key="eval-events-321",
    )
    repository.claim_run(run.run_id)
    repository.complete_run(run.run_id, result={"reply_html": "<p>Private customer answer</p>"})
    replay = repository.create_task_run(
        token_id="token-a", token_name="Agent A", task_name="ticket.evaluate",
        arguments={"conversation_id": 321}, scopes={"task", "read"},
        idempotency_key="eval-events-321",
    )
    assert replay.run_id == run.run_id
    db = get_db(task_db)
    try:
        rows = db.execute("SELECT event_type, run_id, conversation_id, summary_json FROM agent_events WHERE run_id = ?", (run.run_id,)).fetchall()
    finally:
        db.close()
    assert len(rows) == 1
    assert rows[0]["event_type"] == "supa.run.completed"
    assert rows[0]["conversation_id"] == "321"
    assert rows[0]["summary_json"] == '{"status":"completed"}'


def test_terminal_event_failure_rolls_back_run_transition(task_db, monkeypatch):
    repository = AgentTaskRepository(task_db)
    run = repository.create_task_run(
        token_id="token-a", token_name="Agent A", task_name="ticket.evaluate",
        arguments={"conversation_id": 321}, scopes={"task", "read"},
        idempotency_key="eval-rollback-321",
    )
    repository.claim_run(run.run_id)
    def fail_event(*_args, **_kwargs):
        raise RuntimeError("event insert failed")
    monkeypatch.setattr("services.agent_tasks.append_supa_event_in_transaction", fail_event)
    with pytest.raises(RuntimeError):
        repository.complete_run(run.run_id, result={"reply_html": "private"})
    assert repository.get_run(run.run_id, token_id="token-a").status == "running"


@pytest.mark.asyncio
async def test_shutdown_cancels_and_awaits_active_agent_runs():
    from api import agent as agent_api

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def worker():
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(worker())
    agent_api._active_agent_run_tasks.add(task)
    await started.wait()
    await agent_api.close_agent_run_tasks()
    assert task.done()
    assert cancelled.is_set()
    agent_api._active_agent_run_tasks.discard(task)


@pytest.mark.asyncio
async def test_evaluate_task_persists_complete_dry_run_result(task_db):
    repository = AgentTaskRepository(task_db)
    run = repository.create_task_run(
        token_id="token-a", token_name="Agent A", task_name="ticket.evaluate",
        arguments={"conversation_id": 601}, scopes={"task", "read"},
        idempotency_key="eval-601",
    )
    with patch("api.dashboard.evaluate_ticket", new_callable=AsyncMock) as evaluate:
        evaluate.return_value = SimpleNamespace(model_dump=lambda **_kwargs: {
            "ticket": {"id": 601}, "classification": "support",
            "response_html": "<p>Draft</p>", "posted": False,
        })
        from services.agent_tasks import run_evaluate_task
        await run_evaluate_task(repository, run.run_id)
    evaluate.assert_awaited_once_with(601)
    saved = AgentTaskRepository(task_db).get_run(run.run_id, token_id="token-a")
    assert saved.status == "completed"
    assert saved.result["response_html"] == "<p>Draft</p>"
    assert saved.result["posted"] is False
    assert [event.stage for event in repository.list_run_events(run.run_id, token_id="token-a")] == [
        "fetch", "evaluate", "completed"
    ]


@pytest.mark.asyncio
async def test_evaluate_task_failure_is_durable_and_sanitized(task_db):
    repository = AgentTaskRepository(task_db)
    run = repository.create_task_run(
        token_id="token-a", token_name="Agent A", task_name="ticket.evaluate",
        arguments={"conversation_id": 601}, scopes={"task", "read"},
        idempotency_key="eval-fail-601",
    )
    with patch("api.dashboard.evaluate_ticket", new_callable=AsyncMock) as evaluate:
        evaluate.side_effect = RuntimeError("Bearer secret-value provider failed")
        from services.agent_tasks import run_evaluate_task
        await run_evaluate_task(repository, run.run_id)
    saved = repository.get_run(run.run_id, token_id="token-a")
    assert saved.status == "failed"
    assert saved.error == {"code": "evaluation_failed", "message": "Ticket evaluation failed."}


@pytest.mark.asyncio
async def test_process_task_refuses_manual_conversation_before_processor(task_db):
    repository = AgentTaskRepository(task_db)
    run = repository.create_task_run(
        token_id="token-a", token_name="Agent A", task_name="ticket.process",
        arguments={"conversation_id": 601}, scopes={"task", "read", "draft"},
        idempotency_key="process-manual-601",
    )
    upstream = SimpleNamespace(id=601, tags=[{"tag": "supa-ignore"}])
    processor = SimpleNamespace(
        helpscout=SimpleNamespace(get_conversation=AsyncMock(return_value=vars(upstream))),
        process_ticket=AsyncMock(),
    )
    from services.agent_tasks import run_process_task
    await run_process_task(repository, run.run_id, processor=processor)
    processor.process_ticket.assert_not_awaited()
    saved = repository.get_run(run.run_id, token_id="token-a")
    assert saved.status == "completed"
    assert saved.result == {"status": "manual_conversation", "reason": "supa_ignore", "retryable": False}


@pytest.mark.asyncio
async def test_process_task_passes_snapshotted_draft_ceiling(task_db):
    repository = AgentTaskRepository(task_db)
    run = repository.create_task_run(
        token_id="token-a", token_name="Agent A", task_name="ticket.process",
        arguments={"conversation_id": 601}, scopes={"task", "read", "draft"},
        idempotency_key="process-draft-601",
    )
    payload = {"id": 601, "tags": []}
    processor = SimpleNamespace(
        helpscout=SimpleNamespace(get_conversation=AsyncMock(return_value=payload)),
        process_ticket=AsyncMock(return_value={"status": "processed", "delivery_mode": "draft"}),
    )
    from services.agent_tasks import run_process_task
    with patch("services.agent_tasks.manual_route", return_value=(None, None)):
        await run_process_task(repository, run.run_id, processor=processor)
    processor.process_ticket.assert_awaited_once_with(
        payload, trigger="manual_run", delivery_ceiling="draft"
    )
    assert repository.get_run(run.run_id, token_id="token-a").result["delivery_mode"] == "draft"


def test_create_run_uses_uuid_and_survives_repository_reconstruction(task_db):
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    repository = AgentTaskRepository(task_db, clock=lambda: now)

    created = repository.create_run(
        token_id="token-owner-1",
        token_name="Operator Laptop",
        task_name="ticket.evaluate",
        conversation_id=601,
        arguments={"conversation_id": 601},
        scopes={"task", "read"},
        delivery_ceiling="draft",
        idempotency_key="task-evaluate-601",
    )

    parsed_id = UUID(created.run_id)
    assert parsed_id.version == 4
    assert created.status == "queued"
    assert created.arguments == {"conversation_id": "601"}
    assert created.scopes == ("read", "task")
    assert created.created_at == now
    assert created.started_at is None
    assert created.completed_at is None

    reconstructed = AgentTaskRepository(task_db, clock=lambda: now)
    loaded = reconstructed.get_run(created.run_id, token_id="token-owner-1")
    assert loaded == created
    with pytest.raises(AgentRunNotFoundError):
        reconstructed.get_run(created.run_id, token_id="another-token")


def test_guarded_lifecycle_and_ordered_append_only_progress(task_db):
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    repository = AgentTaskRepository(task_db, clock=lambda: now)
    queued = repository.create_run(
        token_id="token-owner-1",
        token_name="Operator Laptop",
        task_name="ticket.evaluate",
        conversation_id=601,
    )

    running = repository.claim_run(queued.run_id)
    assert running is not None
    assert running.status == "running"
    assert running.started_at == now
    assert repository.claim_run(queued.run_id) is None

    first = repository.append_progress(
        queued.run_id,
        stage="fetch",
        progress={"message": "Fetched conversation", "conversation_id": 601},
    )
    second = repository.append_progress(
        queued.run_id,
        stage="evaluate",
        progress={"message": "Evaluating reply", "attempt": 1},
    )
    assert (first.sequence, second.sequence) == (1, 2)
    assert [event.stage for event in repository.list_run_events(
        queued.run_id, token_id="token-owner-1"
    )] == ["fetch", "evaluate"]

    completed = repository.complete_run(
        queued.run_id,
        result={
            "reply_html": "<p>Hello <strong>Operator</strong></p>",
            "status": "completed",
            "authorization": "Bearer private-credential",
            "nested": {
                "password": "private-password",
                "note": "safe note",
                "accidental_token": f"prefix {'A' * 43} suffix",
                "warning": 'authorization="top-secret" password: hidden',
            },
        },
    )
    assert completed.status == "completed"
    assert completed.result == {
        "nested": {
            "accidental_token": "prefix [REDACTED] suffix",
            "note": "safe note",
            "warning": "authorization=[REDACTED] password: [REDACTED]",
        },
        "reply_html": "<p>Hello <strong>Operator</strong></p>",
        "status": "completed",
    }
    assert completed.error is None
    assert completed.completed_at == now
    with pytest.raises(AgentRunStateError):
        repository.fail_run(queued.run_id, error={"code": "late_failure"})
    with pytest.raises(AgentRunStateError):
        repository.append_progress(
            queued.run_id,
            stage="late",
            progress={"message": "must not append"},
        )

    db = get_db(task_db)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "UPDATE agent_run_events SET stage = 'changed' WHERE run_id = ?",
                (queued.run_id,),
            )
        db.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "DELETE FROM agent_run_events WHERE run_id = ?", (queued.run_id,)
            )
    finally:
        db.close()


def test_failed_and_interrupted_runs_store_sanitized_errors(task_db):
    repository = AgentTaskRepository(task_db)
    failed = repository.create_run(
        token_id="token-owner-1",
        token_name="Operator Laptop",
        task_name="ticket.evaluate",
        conversation_id=1,
    )
    interrupted = repository.create_run(
        token_id="token-owner-1",
        token_name="Operator Laptop",
        task_name="ticket.process",
        conversation_id=2,
    )
    assert repository.claim_run(failed.run_id) is not None
    assert repository.claim_run(interrupted.run_id) is not None

    failed_record = repository.fail_run(
        failed.run_id,
        error={
            "code": "provider_failed",
            "message": "Provider failed safely",
            "headers": {"Authorization": "Bearer top-secret", "Retry-After": "2"},
        },
    )
    interrupted_record = repository.interrupt_run(
        interrupted.run_id,
        error={"code": "run_interrupted", "message": "Worker stopped"},
    )

    assert failed_record.status == "failed"
    assert failed_record.error == {
        "code": "provider_failed",
        "headers": {"Retry-After": "2"},
        "message": "Provider failed safely",
    }
    assert interrupted_record.status == "interrupted"
    assert interrupted_record.error == {
        "code": "run_interrupted",
        "message": "Worker stopped",
    }


def test_list_queued_and_atomic_startup_interruption(task_db):
    current = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)

    def clock() -> datetime:
        return current

    repository = AgentTaskRepository(task_db, clock=clock)
    first = repository.create_run(
        token_id="token-a",
        token_name="Agent A",
        task_name="ticket.evaluate",
        conversation_id=1,
    )
    current += timedelta(seconds=1)
    second = repository.create_run(
        token_id="token-b",
        token_name="Agent B",
        task_name="ticket.process",
        conversation_id=2,
    )
    current += timedelta(seconds=1)
    third = repository.create_run(
        token_id="token-c",
        token_name="Agent C",
        task_name="ticket.evaluate",
        conversation_id=3,
    )
    assert repository.claim_run(second.run_id) is not None
    assert repository.claim_run(third.run_id) is not None
    repository.complete_run(third.run_id, result={"status": "done"})

    reconstructed = AgentTaskRepository(task_db, clock=clock)
    assert [record.run_id for record in reconstructed.list_queued_runs()] == [
        first.run_id
    ]
    assert reconstructed.interrupt_stale_running() == 1
    assert reconstructed.interrupt_stale_running() == 0
    recovered = reconstructed.get_run(second.run_id, token_id="token-b")
    assert recovered.status == "interrupted"
    assert recovered.error == {
        "code": "run_interrupted",
        "message": "Run interrupted by service restart",
    }
    assert reconstructed.get_run(third.run_id, token_id="token-c").status == "completed"


def test_validation_failure_does_not_partially_mutate_run(task_db):
    repository = AgentTaskRepository(task_db)
    created = repository.create_run(
        token_id="token-owner-1",
        token_name="Operator Laptop",
        task_name="ticket.evaluate",
        conversation_id=601,
    )
    assert repository.claim_run(created.run_id) is not None

    with pytest.raises(ValueError):
        repository.complete_run(
            created.run_id,
            result={"reply_html": "x" * (MAX_RUN_JSON_BYTES + 1)},
        )
    assert repository.get_run(
        created.run_id, token_id="token-owner-1"
    ).status == "running"


def test_malformed_stored_row_is_nonretryable_and_batch_recovery_rolls_back(task_db):
    repository = AgentTaskRepository(task_db)
    first = repository.create_run(
        token_id="token-owner-1",
        token_name="Operator Laptop",
        task_name="ticket.evaluate",
        conversation_id=1,
    )
    second = repository.create_run(
        token_id="token-owner-1",
        token_name="Operator Laptop",
        task_name="ticket.evaluate",
        conversation_id=2,
    )
    assert repository.claim_run(first.run_id) is not None
    assert repository.claim_run(second.run_id) is not None
    db = get_db(task_db)
    try:
        db.execute(
            "UPDATE agent_runs SET arguments_json = '{' WHERE run_id = ?",
            (second.run_id,),
        )
        db.commit()
    finally:
        db.close()

    with pytest.raises(AgentRunDataError) as caught:
        AgentTaskRepository(task_db).interrupt_stale_running()
    assert caught.value.retryable is False
    assert repository.get_run(first.run_id, token_id="token-owner-1").status == "running"


def test_recovery_admission_is_atomic_across_keys_and_tokens(task_db):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from services.agent_tasks import AgentRunError
    barrier = Barrier(2)
    args = {'conversation_id':123, 'expected_thread_id':'1', 'expected_draft_version':'a'*64,
            'expected_conversation_version':'b'*64}
    def admit(index):
        repository = AgentTaskRepository(task_db)
        barrier.wait()
        try:
            return repository.create_task_run(token_id=f'agent-{index}',token_name='Agent',task_name='ticket.recover',
                arguments=args,scopes={'task','read','draft'},idempotency_key=f'recovery-{index}')
        except AgentRunError as exc:
            return exc
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(admit,range(2)))
    runs = [value for value in outcomes if not isinstance(value,AgentRunError)]
    errors = [value for value in outcomes if isinstance(value,AgentRunError)]
    assert len(runs) == 1
    assert [error.code for error in errors] == ['recovery_inflight']
    assert errors[0].retryable is True
    repository = AgentTaskRepository(task_db)
    assert len(repository.list_queued_runs()) == 1
    first = runs[0]
    replay = repository.create_task_run(token_id=first.token_id,token_name='Agent',task_name='ticket.recover',
        arguments=args,scopes={'task','read','draft'},idempotency_key=first.idempotency_key)
    assert replay.run_id == first.run_id
    repository.claim_run(first.run_id)
    with pytest.raises(AgentRunError) as caught:
        repository.create_task_run(token_id='new-agent',token_name='Agent',task_name='ticket.recover',
            arguments=args,scopes={'task','read','draft'},idempotency_key='new-key')
    assert caught.value.code == 'recovery_inflight'
    repository.interrupt_run(first.run_id,error={'code':'run_interrupted','message':'shutdown'})
    retry = repository.create_task_run(token_id='new-agent',token_name='Agent',task_name='ticket.recover',
        arguments=args,scopes={'task','read','draft'},idempotency_key='new-key')
    assert retry.status == 'queued'
    assert retry.run_id != first.run_id
