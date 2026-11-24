import asyncio
from unittest.mock import AsyncMock

import pytest

import main


def test_autosend_worker_task_name_registered():
    assert "autosend_worker_task" in main.RECOVERY_TASK_NAMES


def test_agent_operation_startup_resolves_one_database_path(monkeypatch, tmp_path):
    selected = tmp_path / "selected.db"
    calls = []
    monkeypatch.setattr(main, "get_supa_db_path", lambda: str(selected), raising=False)
    monkeypatch.setattr(main, "init_db", lambda path: calls.append(("init", path)))
    monkeypatch.setattr(
        main,
        "recover_stale_operations",
        lambda *, db_path: calls.append(("recover", db_path)) or "summary",
    )

    result = main.recover_agent_operations_on_startup()

    assert result == "summary"
    assert calls == [("init", str(selected)), ("recover", str(selected))]


@pytest.mark.asyncio
async def test_startup_event_stores_recovery_task_references(monkeypatch):
    startup_task = asyncio.create_task(asyncio.sleep(3600))
    periodic_task = asyncio.create_task(asyncio.sleep(3600))
    autosend_task = asyncio.create_task(asyncio.sleep(3600))

    monkeypatch.setattr(main, "schedule_startup_recovery", lambda: startup_task)
    monkeypatch.setattr(main, "schedule_periodic_recovery", lambda: periodic_task)
    monkeypatch.setattr(main, "schedule_autosend_worker", lambda *, helpscout: autosend_task)
    monkeypatch.setattr(main, "recover_agent_operations_on_startup", lambda: None)

    await main.startup_recovery()

    assert main.app.state.startup_recovery_task is startup_task
    assert main.app.state.periodic_recovery_task is periodic_task
    assert main.app.state.autosend_worker_task is autosend_task

    await main.shutdown_recovery_tasks()


@pytest.mark.asyncio
async def test_agent_operation_recovery_precedes_mutation_background_work(monkeypatch):
    calls = []
    tasks = []

    def task(name):
        calls.append(name)
        created = asyncio.create_task(asyncio.sleep(3600))
        tasks.append(created)
        return created

    monkeypatch.setattr(
        main,
        "recover_agent_operations_on_startup",
        lambda: calls.append("agent-operation-recovery"),
        raising=False,
    )
    monkeypatch.setattr(main, "schedule_startup_recovery", lambda: task("ticket-recovery"))
    monkeypatch.setattr(main, "schedule_periodic_recovery", lambda: task("periodic-recovery"))
    monkeypatch.setattr(
        main,
        "schedule_autosend_worker",
        lambda *, helpscout: task("autosend"),
    )

    await main.startup_recovery()

    assert calls[0] == "agent-operation-recovery"
    assert calls.index("agent-operation-recovery") < calls.index("autosend")
    await main.shutdown_recovery_tasks()


@pytest.mark.asyncio
async def test_shutdown_event_cancels_recovery_tasks():
    startup_task = asyncio.create_task(asyncio.sleep(3600))
    periodic_task = asyncio.create_task(asyncio.sleep(3600))
    autosend_task = asyncio.create_task(asyncio.sleep(3600))
    main.app.state.startup_recovery_task = startup_task
    main.app.state.periodic_recovery_task = periodic_task
    main.app.state.autosend_worker_task = autosend_task

    await main.shutdown_recovery_tasks()

    assert startup_task.cancelled()
    assert periodic_task.cancelled()
    assert autosend_task.cancelled()
    assert main.app.state.startup_recovery_task is None
    assert main.app.state.periodic_recovery_task is None
    assert main.app.state.autosend_worker_task is None


@pytest.mark.asyncio
async def test_shutdown_closes_agent_gateway_after_recovery_cleanup(monkeypatch):
    close_gateway = AsyncMock()
    monkeypatch.setattr(
        main,
        "close_conversation_gateway",
        close_gateway,
        raising=False,
    )
    for task_name in main.RECOVERY_TASK_NAMES:
        setattr(main.app.state, task_name, None)

    await main.shutdown_recovery_tasks()

    close_gateway.assert_awaited_once_with()
