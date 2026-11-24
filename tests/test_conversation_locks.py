import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from db.database import init_db
import services.conversation_locks as conversation_locks_module
from services.conversation_locks import (
    ConversationBusyError,
    ConversationLockLostError,
    ConversationLockStorageError,
    acquire_conversation_lock,
    conversation_lease,
    current_owner,
    release_conversation_lock,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class FakeMonotonic:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class BlockingSleep:
    """A cancellable fake sleep that never advances without the test."""

    def __init__(self) -> None:
        self.calls: list[float] = []
        self.waiting = asyncio.Event()

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.waiting.set()
        await asyncio.Event().wait()


class MonotonicSleep:
    def __init__(self, monotonic: FakeMonotonic) -> None:
        self.monotonic = monotonic
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.monotonic.advance(seconds)
        await asyncio.sleep(0)


class UnlockingSleep(MonotonicSleep):
    def __init__(self, monotonic: FakeMonotonic) -> None:
        super().__init__(monotonic)
        self.blocker: sqlite3.Connection | None = None

    async def __call__(self, seconds: float) -> None:
        await super().__call__(seconds)
        if self.blocker is not None:
            self.blocker.rollback()
            self.blocker.close()
            self.blocker = None


class ExpiringUnlockingSleep(UnlockingSleep):
    def __init__(self, clock: FakeClock, monotonic: FakeMonotonic) -> None:
        super().__init__(monotonic)
        self.clock = clock

    async def __call__(self, seconds: float) -> None:
        self.clock.advance(seconds)
        await super().__call__(seconds)


class AdvancingClocksSleep(MonotonicSleep):
    def __init__(self, clock: FakeClock, monotonic: FakeMonotonic) -> None:
        super().__init__(monotonic)
        self.clock = clock

    async def __call__(self, seconds: float) -> None:
        self.clock.advance(seconds)
        await super().__call__(seconds)


class GuardRetrySleep:
    """Advance only the foreground guard while leaving the heartbeat parked."""

    def __init__(
        self,
        clock: FakeClock,
        monotonic: FakeMonotonic,
        *,
        block_foreground: bool = False,
    ) -> None:
        self.clock = clock
        self.monotonic = monotonic
        self.block_foreground = block_foreground
        self.foreground_task: asyncio.Task | None = None
        self.calls: list[float] = []
        self.waiting = asyncio.Event()

    async def __call__(self, seconds: float) -> None:
        if asyncio.current_task() is not self.foreground_task:
            await asyncio.Event().wait()
            return
        self.calls.append(seconds)
        self.waiting.set()
        if self.block_foreground:
            await asyncio.Event().wait()
        self.clock.advance(seconds)
        self.monotonic.advance(seconds)
        await asyncio.sleep(0)


class ControlledSleep:
    """Let a test release exactly one automatic-heartbeat interval."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[float] = []
        self.waiting = asyncio.Event()
        self._permits: asyncio.Queue[None] = asyncio.Queue()

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.waiting.set()
        await self._permits.get()
        self.clock.advance(seconds)

    def release(self) -> None:
        self.waiting.clear()
        self._permits.put_nowait(None)


@pytest.fixture
def lock_db(tmp_path):
    path = tmp_path / "locks.db"
    init_db(str(path))
    return path


def _lock_row(path, conversation_id):
    with sqlite3.connect(path) as connection:
        return connection.execute(
            """
            SELECT conversation_id, owner, acquired_at, expires_at
            FROM agent_conversation_locks
            WHERE conversation_id = ?
            """,
            (str(conversation_id),),
        ).fetchone()


async def _yield_until(predicate, *, attempts=20):
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached after asynchronous yields")


async def test_first_acquisition_is_visible_and_context_release_is_owner_scoped(lock_db):
    clock = FakeClock()
    sleeper = BlockingSleep()

    async def mutation(value):
        return f"mutated-{value}"

    async with conversation_lease(
        lock_db,
        123,
        owner="req-a",
        ttl_seconds=30,
        clock=clock,
        sleep=sleeper,
    ) as lease:
        assert current_owner(lock_db, 123, clock=clock) == "req-a"
        assert lease.owner == "req-a"
        await lease.verify_owned()
        assert await lease.run_guarded(mutation, "safely") == "mutated-safely"
        assert (
            release_conversation_lock(
                lock_db, 123, owner="not-the-owner", clock=clock
            )
            is False
        )
        assert current_owner(lock_db, 123, clock=clock) == "req-a"

    assert current_owner(lock_db, 123, clock=clock) is None


async def test_contention_has_bounded_retry_and_typed_retryable_error(lock_db):
    clock = FakeClock()
    monotonic = FakeMonotonic()
    assert acquire_conversation_lock(
        lock_db, 123, owner="req-a", ttl_seconds=30, clock=clock
    )
    sleeper = MonotonicSleep(monotonic)

    with pytest.raises(ConversationBusyError) as caught:
        async with conversation_lease(
            lock_db,
            123,
            owner="req-b",
            ttl_seconds=30,
            acquire_timeout_seconds=1,
            poll_interval_seconds=0.25,
            clock=clock,
            monotonic_clock=monotonic,
            sleep=sleeper,
        ):
            pytest.fail("contended lease must not enter")

    assert caught.value.code == "conversation_busy"
    assert caught.value.retryable is True
    assert caught.value.status == 503
    assert caught.value.exit_code == 6
    assert sum(sleeper.calls) == pytest.approx(1)
    assert current_owner(lock_db, 123, clock=clock) == "req-a"


async def test_contention_deadline_uses_monotonic_time_when_wall_clock_is_frozen(lock_db):
    clock = FakeClock()
    monotonic = FakeMonotonic()
    sleeper = MonotonicSleep(monotonic)
    assert acquire_conversation_lock(
        lock_db, 123, owner="req-a", ttl_seconds=30, clock=clock
    )

    with pytest.raises(ConversationBusyError):
        async with conversation_lease(
            lock_db,
            123,
            owner="req-b",
            acquire_timeout_seconds=1,
            poll_interval_seconds=0.25,
            clock=clock,
            monotonic_clock=monotonic,
            sleep=sleeper,
        ):
            pytest.fail("a frozen wall clock must not make contention unbounded")

    assert monotonic() == pytest.approx(101)
    assert clock() == datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def test_same_owner_renews_expiry_without_replacing_acquisition_time(lock_db):
    clock = FakeClock()
    assert acquire_conversation_lock(
        lock_db, 123, owner="req-a", ttl_seconds=9, clock=clock
    )
    first = _lock_row(lock_db, 123)

    clock.advance(2)
    assert acquire_conversation_lock(
        lock_db, 123, owner="req-a", ttl_seconds=9, clock=clock
    )
    renewed = _lock_row(lock_db, 123)

    assert renewed[1] == "req-a"
    assert renewed[2] == first[2]
    assert renewed[3] > first[3]


def test_same_owner_renewal_never_shortens_expiry_after_clock_moves_backward(lock_db):
    clock = FakeClock()
    assert acquire_conversation_lock(
        lock_db, 123, owner="req-a", ttl_seconds=30, clock=clock
    )
    first = _lock_row(lock_db, 123)

    clock.value -= timedelta(minutes=1)
    assert acquire_conversation_lock(
        lock_db, 123, owner="req-a", ttl_seconds=30, clock=clock
    )
    renewed = _lock_row(lock_db, 123)

    assert renewed[2] == first[2]
    assert renewed[3] == first[3]


async def test_explicit_renew_keeps_cached_expiry_synced_after_clock_moves_backward(
    lock_db,
):
    clock = FakeClock()

    async with conversation_lease(
        lock_db,
        123,
        owner="req-a",
        ttl_seconds=30,
        clock=clock,
        sleep=BlockingSleep(),
    ) as lease:
        first_expiry = _lock_row(lock_db, 123)[3]
        clock.value -= timedelta(minutes=1)

        await lease.renew()

        assert _lock_row(lock_db, 123)[3] == first_expiry
        assert lease.expires_at == datetime.fromisoformat(first_expiry)


@pytest.mark.parametrize(
    ("acquired_at", "expires_at"),
    [
        (
            "2026-09-24T11:59:00.000000+00:00",
            "2026-09-24T12:01:00.000000+00:00",
        ),
        ("2026-09-24T11:59:00", "2026-09-24T12:01:00"),
        ("2026-09-24T11:59:00Z", "2026-09-24T12:01:00Z"),
        ("2026-09-24T13:59:00+02:00", "2026-09-24T14:01:00+02:00"),
        ("2026-09-24T07:59:00-04:00", "2026-09-24T08:01:00-04:00"),
    ],
)
def test_valid_stored_timestamp_variants_are_compared_and_rewritten_canonically(
    lock_db, acquired_at, expires_at
):
    clock = FakeClock()
    with sqlite3.connect(lock_db) as connection:
        connection.execute(
            """
            INSERT INTO agent_conversation_locks (
                conversation_id, owner, acquired_at, expires_at
            ) VALUES (?, ?, ?, ?)
            """,
            ("123", "req-a", acquired_at, expires_at),
        )

    assert acquire_conversation_lock(
        lock_db, 123, owner="req-a", ttl_seconds=9, clock=clock
    )
    row = _lock_row(lock_db, 123)

    assert row[2] == "2026-09-24T11:59:00.000000+00:00"
    assert row[3] == "2026-09-24T12:01:00.000000+00:00"


@pytest.mark.parametrize(
    ("acquired_at", "expires_at"),
    [
        ("not-a-timestamp", "2026-09-24T12:01:00+00:00"),
        ("2026-09-24T11:59:00+00:00", "not-a-timestamp"),
        (123, "2026-09-24T12:01:00+00:00"),
        ("0001-01-01T00:00:00+14:00", "2026-09-24T12:01:00+00:00"),
        ("2026-09-24T11:59:00+00:00", "9999-12-31T23:59:59-14:00"),
    ],
)
def test_malformed_stored_timestamp_raises_stable_storage_error_without_mutation(
    lock_db, acquired_at, expires_at
):
    clock = FakeClock()
    with sqlite3.connect(lock_db) as connection:
        connection.execute(
            """
            INSERT INTO agent_conversation_locks (
                conversation_id, owner, acquired_at, expires_at
            ) VALUES (?, ?, ?, ?)
            """,
            ("123", "req-a", acquired_at, expires_at),
        )

    before = _lock_row(lock_db, 123)
    with pytest.raises(ConversationLockStorageError) as caught:
        acquire_conversation_lock(
            lock_db, 123, owner="req-a", ttl_seconds=9, clock=clock
        )

    assert caught.value.code == "conversation_lock_storage_error"
    assert caught.value.retryable is False
    assert caught.value.status == 500
    assert _lock_row(lock_db, 123) == before


def test_missing_lock_schema_raises_stable_storage_error(tmp_path):
    path = tmp_path / "missing-schema.db"
    sqlite3.connect(path).close()

    with pytest.raises(ConversationLockStorageError) as caught:
        acquire_conversation_lock(path, 123, owner="req-a")

    assert caught.value.code == "conversation_lock_storage_error"
    assert caught.value.retryable is False
    assert caught.value.status == 500

    for operation in (
        lambda: current_owner(path, 123),
        lambda: release_conversation_lock(path, 123, owner="req-a"),
    ):
        with pytest.raises(ConversationLockStorageError) as other:
            operation()
        assert other.value.code == "conversation_lock_storage_error"


def test_database_open_failure_raises_stable_storage_error(tmp_path):
    with pytest.raises(ConversationLockStorageError) as caught:
        acquire_conversation_lock(tmp_path, 123, owner="req-a")

    assert caught.value.code == "conversation_lock_storage_error"
    assert caught.value.retryable is False


def test_release_compares_offset_expiry_as_datetime_not_text(lock_db):
    clock = FakeClock()
    with sqlite3.connect(lock_db) as connection:
        connection.execute(
            """
            INSERT INTO agent_conversation_locks (
                conversation_id, owner, acquired_at, expires_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                "123",
                "req-a",
                "2026-09-24T12:00:00+02:00",
                "2026-09-24T13:00:00+02:00",
            ),
        )

    assert release_conversation_lock(
        lock_db, 123, owner="req-a", clock=clock
    ) is False
    assert _lock_row(lock_db, 123) is not None


async def test_explicit_renew_compares_offsets_and_rewrites_both_times(lock_db):
    clock = FakeClock()

    async with conversation_lease(
        lock_db,
        123,
        owner="req-a",
        ttl_seconds=9,
        clock=clock,
        sleep=BlockingSleep(),
    ) as lease:
        with sqlite3.connect(lock_db) as connection:
            connection.execute(
                """
                UPDATE agent_conversation_locks
                SET acquired_at = ?, expires_at = ?
                WHERE conversation_id = ?
                """,
                (
                    "2026-09-24T07:59:00-04:00",
                    "2026-09-24T08:01:00-04:00",
                    "123",
                ),
            )

        await lease.renew()
        row = _lock_row(lock_db, 123)
        assert row[2] == "2026-09-24T11:59:00.000000+00:00"
        assert row[3] == "2026-09-24T12:01:00.000000+00:00"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("ttl_seconds", float("nan")),
        ("ttl_seconds", float("inf")),
        ("ttl_seconds", float("-inf")),
        ("ttl_seconds", 0),
        ("ttl_seconds", 1e-9),
        ("ttl_seconds", 86_400.1),
        ("ttl_seconds", 1e300),
        pytest.param("ttl_seconds", 10**10_000, id="ttl-huge-int"),
        ("acquire_timeout_seconds", float("nan")),
        ("acquire_timeout_seconds", float("inf")),
        ("acquire_timeout_seconds", float("-inf")),
        ("acquire_timeout_seconds", -1),
        ("acquire_timeout_seconds", 300.1),
        ("poll_interval_seconds", float("nan")),
        ("poll_interval_seconds", float("inf")),
        ("poll_interval_seconds", float("-inf")),
        ("poll_interval_seconds", 0),
        ("poll_interval_seconds", 0.0001),
        ("poll_interval_seconds", 30.1),
        ("heartbeat_interval_seconds", float("nan")),
        ("heartbeat_interval_seconds", float("inf")),
        ("heartbeat_interval_seconds", float("-inf")),
        ("heartbeat_interval_seconds", 0),
        ("heartbeat_interval_seconds", 0.0001),
        ("heartbeat_interval_seconds", 86_400.1),
    ],
)
def test_lease_rejects_non_finite_and_out_of_range_timing(tmp_path, option, value):
    db_path = tmp_path / "must-not-exist.db"
    with pytest.raises(ValueError):
        conversation_lease(db_path, 123, owner="req-a", **{option: value})
    assert not db_path.exists()


@pytest.mark.parametrize(
    "ttl_seconds",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        0,
        1e-9,
        86_400.1,
        1e300,
        pytest.param(10**10_000, id="huge-int"),
    ],
)
def test_direct_acquire_rejects_invalid_ttl_before_database_access(
    tmp_path, ttl_seconds
):
    db_path = tmp_path / "must-not-exist.db"
    with pytest.raises(ValueError):
        acquire_conversation_lock(
            db_path, 123, owner="req-a", ttl_seconds=ttl_seconds
        )
    assert not db_path.exists()


async def test_automatic_heartbeat_renews_at_no_more_than_one_third_ttl(lock_db):
    clock = FakeClock()
    sleeper = ControlledSleep(clock)

    async with conversation_lease(
        lock_db,
        123,
        owner="req-a",
        ttl_seconds=9,
        poll_interval_seconds=0.5,
        clock=clock,
        sleep=sleeper,
    ):
        await sleeper.waiting.wait()
        before = _lock_row(lock_db, 123)[3]
        assert sleeper.calls[0] <= 3

        sleeper.release()
        await _yield_until(lambda: _lock_row(lock_db, 123)[3] > before)

        assert current_owner(lock_db, 123, clock=clock) == "req-a"


def test_expired_lock_can_be_taken_over_and_old_owner_cannot_release(lock_db):
    clock = FakeClock()
    assert acquire_conversation_lock(
        lock_db, 123, owner="req-a", ttl_seconds=1, clock=clock
    )

    clock.advance(2)
    assert current_owner(lock_db, 123, clock=clock) is None
    assert acquire_conversation_lock(
        lock_db, 123, owner="req-b", ttl_seconds=10, clock=clock
    )

    assert current_owner(lock_db, 123, clock=clock) == "req-b"
    assert (
        release_conversation_lock(lock_db, 123, owner="req-a", clock=clock)
        is False
    )
    assert current_owner(lock_db, 123, clock=clock) == "req-b"


async def test_context_exception_releases_lease_without_masking_exception(lock_db):
    clock = FakeClock()

    with pytest.raises(ValueError, match="upstream failed"):
        async with conversation_lease(
            lock_db,
            123,
            owner="req-a",
            clock=clock,
            sleep=BlockingSleep(),
        ):
            raise ValueError("upstream failed")

    assert current_owner(lock_db, 123, clock=clock) is None


async def test_cancellation_propagates_and_releases_owned_lock(lock_db):
    clock = FakeClock()
    entered = asyncio.Event()
    never = asyncio.Event()

    async def worker():
        async with conversation_lease(
            lock_db,
            123,
            owner="req-a",
            clock=clock,
            sleep=BlockingSleep(),
        ):
            entered.set()
            await never.wait()

    task = asyncio.create_task(worker())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert current_owner(lock_db, 123, clock=clock) is None


async def test_exit_preserves_storage_error_instead_of_reporting_lock_lost(lock_db):
    clock = FakeClock()

    with pytest.raises(ConversationLockStorageError) as caught:
        async with conversation_lease(
            lock_db,
            123,
            owner="req-a",
            clock=clock,
            sleep=BlockingSleep(),
        ):
            with sqlite3.connect(lock_db) as connection:
                connection.execute(
                    """
                    UPDATE agent_conversation_locks
                    SET acquired_at = ?
                    WHERE conversation_id = ?
                    """,
                    ("malformed", "123"),
                )

    assert caught.value.code == "conversation_lock_storage_error"
    assert caught.value.retryable is False
    assert _lock_row(lock_db, 123)[2] == "malformed"


async def test_release_retries_transient_sqlite_busy_then_cleans_up(lock_db):
    clock = FakeClock()
    monotonic = FakeMonotonic()
    sleeper = UnlockingSleep(monotonic)

    async with conversation_lease(
        lock_db,
        123,
        owner="req-a",
        acquire_timeout_seconds=0.5,
        poll_interval_seconds=0.1,
        clock=clock,
        monotonic_clock=monotonic,
        sleep=sleeper,
    ):
        blocker = sqlite3.connect(lock_db, timeout=0)
        blocker.execute("PRAGMA busy_timeout=0")
        blocker.execute("BEGIN IMMEDIATE")
        sleeper.blocker = blocker

    assert sleeper.calls == [pytest.approx(0.1)]
    assert current_owner(lock_db, 123, clock=clock) is None


async def test_release_timeout_reports_retryable_busy_not_false_ownership_loss(lock_db):
    clock = FakeClock()
    monotonic = FakeMonotonic()
    sleeper = MonotonicSleep(monotonic)
    blocker = None

    try:
        with pytest.raises(ConversationBusyError) as caught:
            async with conversation_lease(
                lock_db,
                123,
                owner="req-a",
                acquire_timeout_seconds=0.5,
                poll_interval_seconds=0.1,
                clock=clock,
                monotonic_clock=monotonic,
                sleep=sleeper,
            ):
                blocker = sqlite3.connect(lock_db, timeout=0)
                blocker.execute("PRAGMA busy_timeout=0")
                blocker.execute("BEGIN IMMEDIATE")
    finally:
        if blocker is not None:
            blocker.rollback()
            blocker.close()

    assert caught.value.code == "conversation_busy"
    assert caught.value.retryable is True
    assert sum(sleeper.calls) == pytest.approx(0.5)
    assert current_owner(lock_db, 123, clock=clock) == "req-a"
    assert release_conversation_lock(
        lock_db, 123, owner="req-a", clock=clock
    ) is True


async def test_release_never_succeeds_after_lease_expires_during_busy_retry(lock_db):
    clock = FakeClock()
    monotonic = FakeMonotonic()
    sleeper = ExpiringUnlockingSleep(clock, monotonic)

    with pytest.raises(ConversationLockLostError):
        async with conversation_lease(
            lock_db,
            123,
            owner="req-a",
            ttl_seconds=0.1,
            acquire_timeout_seconds=1,
            poll_interval_seconds=0.2,
            clock=clock,
            monotonic_clock=monotonic,
            sleep=sleeper,
        ):
            blocker = sqlite3.connect(lock_db, timeout=0)
            blocker.execute("PRAGMA busy_timeout=0")
            blocker.execute("BEGIN IMMEDIATE")
            sleeper.blocker = blocker

    assert sleeper.calls == pytest.approx([0.1])
    assert current_owner(lock_db, 123, clock=clock) is None
    assert _lock_row(lock_db, 123)[1] == "req-a"


async def test_release_prefers_lock_lost_when_timeout_and_ttl_expire_together(lock_db):
    clock = FakeClock()
    monotonic = FakeMonotonic()
    sleeper = AdvancingClocksSleep(clock, monotonic)
    blocker = None

    try:
        with pytest.raises(ConversationLockLostError):
            async with conversation_lease(
                lock_db,
                123,
                owner="req-a",
                ttl_seconds=0.1,
                acquire_timeout_seconds=0.1,
                poll_interval_seconds=0.2,
                clock=clock,
                monotonic_clock=monotonic,
                sleep=sleeper,
            ):
                blocker = sqlite3.connect(lock_db, timeout=0)
                blocker.execute("PRAGMA busy_timeout=0")
                blocker.execute("BEGIN IMMEDIATE")
    finally:
        if blocker is not None:
            blocker.rollback()
            blocker.close()

    assert sleeper.calls == pytest.approx([0.1])
    assert current_owner(lock_db, 123, clock=clock) is None


async def test_guarded_mutation_rejects_sync_callback_before_invocation(lock_db):
    clock = FakeClock()
    mutation_called = False

    def blocking_mutation():
        nonlocal mutation_called
        mutation_called = True

    async with conversation_lease(
        lock_db,
        123,
        owner="req-a",
        clock=clock,
        sleep=BlockingSleep(),
    ) as lease:
        with pytest.raises(TypeError, match="async"):
            await lease.run_guarded(blocking_mutation)

    assert mutation_called is False


async def test_guarded_mutation_accepts_callable_object_with_async_call(lock_db):
    clock = FakeClock()

    class AsyncMutation:
        async def __call__(self, value):
            return f"mutated-{value}"

    async with conversation_lease(
        lock_db,
        123,
        owner="req-a",
        clock=clock,
        sleep=BlockingSleep(),
    ) as lease:
        assert await lease.run_guarded(AsyncMutation(), "safely") == "mutated-safely"


async def test_guarded_mutation_retries_transient_busy_before_invocation(
    lock_db, monkeypatch
):
    clock = FakeClock()
    monotonic = FakeMonotonic()
    sleeper = GuardRetrySleep(clock, monotonic)
    sleeper.foreground_task = asyncio.current_task()
    ownership_checks: list[str] = []
    statuses = iter(("busy", "busy", "owned"))

    def checked_ownership(*args, **kwargs):
        status = next(statuses)
        ownership_checks.append(status)
        return status

    mutation_checks: list[int] = []

    async def mutation():
        mutation_checks.append(len(ownership_checks))
        return "written"

    async with conversation_lease(
        lock_db,
        123,
        owner="req-a",
        acquire_timeout_seconds=0.5,
        poll_interval_seconds=0.1,
        clock=clock,
        monotonic_clock=monotonic,
        sleep=sleeper,
    ) as lease:
        monkeypatch.setattr(
            conversation_locks_module, "_owns_unexpired_lock", checked_ownership
        )
        assert await lease.run_guarded(mutation) == "written"
        monkeypatch.setattr(
            conversation_locks_module,
            "_owns_unexpired_lock",
            lambda *args, **kwargs: "owned",
        )

    assert ownership_checks == ["busy", "busy", "owned"]
    assert sleeper.calls == pytest.approx([0.1, 0.1])
    assert mutation_checks == [3]


async def test_guarded_mutation_persistent_busy_is_bounded_and_never_invoked(
    lock_db, monkeypatch
):
    clock = FakeClock()
    monotonic = FakeMonotonic()
    sleeper = GuardRetrySleep(clock, monotonic)
    sleeper.foreground_task = asyncio.current_task()
    mutation_called = False

    async def mutation():
        nonlocal mutation_called
        mutation_called = True

    async with conversation_lease(
        lock_db,
        123,
        owner="req-a",
        acquire_timeout_seconds=0.25,
        poll_interval_seconds=0.1,
        clock=clock,
        monotonic_clock=monotonic,
        sleep=sleeper,
    ) as lease:
        monkeypatch.setattr(
            conversation_locks_module,
            "_owns_unexpired_lock",
            lambda *args, **kwargs: "busy",
        )
        with pytest.raises(ConversationBusyError):
            await lease.run_guarded(mutation)
        monkeypatch.setattr(
            conversation_locks_module,
            "_owns_unexpired_lock",
            lambda *args, **kwargs: "owned",
        )

    assert mutation_called is False
    assert sum(sleeper.calls) == pytest.approx(0.25)


async def test_guarded_mutation_busy_then_actual_loss_never_invokes(
    lock_db, monkeypatch
):
    clock = FakeClock()
    monotonic = FakeMonotonic()
    sleeper = GuardRetrySleep(clock, monotonic)
    sleeper.foreground_task = asyncio.current_task()
    statuses = iter(("busy", "lost"))
    mutation_called = False

    async def mutation():
        nonlocal mutation_called
        mutation_called = True

    monkeypatch.setattr(
        conversation_locks_module,
        "_owns_unexpired_lock",
        lambda *args, **kwargs: next(statuses),
    )
    with pytest.raises(ConversationLockLostError):
        async with conversation_lease(
            lock_db,
            123,
            owner="req-a",
            acquire_timeout_seconds=0.5,
            poll_interval_seconds=0.1,
            clock=clock,
            monotonic_clock=monotonic,
            sleep=sleeper,
        ) as lease:
            await lease.run_guarded(mutation)

    assert mutation_called is False
    assert sleeper.calls == pytest.approx([0.1])


async def test_guarded_mutation_busy_until_confirmed_expiry_is_lost(
    lock_db, monkeypatch
):
    clock = FakeClock()
    monotonic = FakeMonotonic()
    sleeper = GuardRetrySleep(clock, monotonic)
    sleeper.foreground_task = asyncio.current_task()
    mutation_called = False

    async def mutation():
        nonlocal mutation_called
        mutation_called = True

    monkeypatch.setattr(
        conversation_locks_module,
        "_owns_unexpired_lock",
        lambda *args, **kwargs: "busy",
    )
    with pytest.raises(ConversationLockLostError):
        async with conversation_lease(
            lock_db,
            123,
            owner="req-a",
            ttl_seconds=0.2,
            acquire_timeout_seconds=1,
            poll_interval_seconds=0.1,
            clock=clock,
            monotonic_clock=monotonic,
            sleep=sleeper,
        ) as lease:
            await lease.run_guarded(mutation)

    assert mutation_called is False
    assert sum(sleeper.calls) == pytest.approx(0.2)


async def test_guarded_mutation_cancellation_during_busy_retry_propagates(
    lock_db, monkeypatch
):
    clock = FakeClock()
    monotonic = FakeMonotonic()
    sleeper = GuardRetrySleep(clock, monotonic, block_foreground=True)
    statuses = iter(("busy", "owned"))
    mutation_called = False

    async def mutation():
        nonlocal mutation_called
        mutation_called = True

    monkeypatch.setattr(
        conversation_locks_module,
        "_owns_unexpired_lock",
        lambda *args, **kwargs: next(statuses),
    )

    async def worker():
        sleeper.foreground_task = asyncio.current_task()
        async with conversation_lease(
            lock_db,
            123,
            owner="req-a",
            acquire_timeout_seconds=0.5,
            poll_interval_seconds=0.1,
            clock=clock,
            monotonic_clock=monotonic,
            sleep=sleeper,
        ) as lease:
            await lease.run_guarded(mutation)

    task = asyncio.create_task(worker())
    await sleeper.waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert mutation_called is False
    assert current_owner(lock_db, 123, clock=clock) is None


async def test_lost_owner_blocks_guarded_mutation_and_is_checked_on_exit(lock_db):
    clock = FakeClock()
    mutation_called = False

    async def mutation():
        nonlocal mutation_called
        mutation_called = True

    with pytest.raises(ConversationLockLostError) as exit_error:
        async with conversation_lease(
            lock_db,
            123,
            owner="req-a",
            ttl_seconds=30,
            clock=clock,
            sleep=BlockingSleep(),
        ) as lease:
            with sqlite3.connect(lock_db) as connection:
                connection.execute(
                    """
                    UPDATE agent_conversation_locks
                    SET owner = ?
                    WHERE conversation_id = ?
                    """,
                    ("intruder", "123"),
                )

            with pytest.raises(ConversationLockLostError) as mutation_error:
                await lease.run_guarded(mutation)

            assert mutation_error.value.code == "conversation_lock_lost"
            assert mutation_error.value.retryable is False
            assert mutation_error.value.status == 409
            assert mutation_error.value.exit_code == 5
            assert mutation_called is False

    assert exit_error.value.code == "conversation_lock_lost"
    assert _lock_row(lock_db, 123)[1] == "intruder"


async def test_heartbeat_detects_lost_owner_and_stops_future_writes(lock_db):
    clock = FakeClock()
    sleeper = ControlledSleep(clock)

    with pytest.raises(ConversationLockLostError):
        async with conversation_lease(
            lock_db,
            123,
            owner="req-a",
            ttl_seconds=9,
            clock=clock,
            sleep=sleeper,
        ) as lease:
            await sleeper.waiting.wait()
            with sqlite3.connect(lock_db) as connection:
                connection.execute(
                    """
                    UPDATE agent_conversation_locks
                    SET owner = ?
                    WHERE conversation_id = ?
                    """,
                    ("intruder", "123"),
                )

            sleeper.release()
            await _yield_until(lambda: lease.lost)

            with pytest.raises(ConversationLockLostError):
                await lease.verify_owned()

    assert _lock_row(lock_db, 123)[1] == "intruder"


async def test_heartbeat_retries_sqlite_busy_without_declaring_owner_lost(lock_db):
    clock = FakeClock()
    sleeper = ControlledSleep(clock)

    async with conversation_lease(
        lock_db,
        123,
        owner="req-a",
        ttl_seconds=9,
        poll_interval_seconds=0.5,
        clock=clock,
        sleep=sleeper,
    ) as lease:
        await sleeper.waiting.wait()
        before = _lock_row(lock_db, 123)[3]
        blocker = sqlite3.connect(lock_db, timeout=0)
        blocker.execute("PRAGMA busy_timeout=0")
        blocker.execute("BEGIN IMMEDIATE")

        try:
            sleeper.release()
            await _yield_until(lambda: len(sleeper.calls) >= 2)
            assert lease.lost is False

            assert sleeper.calls[1] <= 0.5
            sleeper.release()
            await _yield_until(lambda: len(sleeper.calls) >= 3)
            assert lease.lost is False
        finally:
            blocker.rollback()
            blocker.close()

        sleeper.release()
        await _yield_until(lambda: _lock_row(lock_db, 123)[3] > before)
        assert lease.lost is False


async def test_persistent_heartbeat_busy_marks_lost_at_known_expiry(lock_db):
    clock = FakeClock()
    sleeper = ControlledSleep(clock)
    blocker = None

    with pytest.raises(ConversationLockLostError):
        async with conversation_lease(
            lock_db,
            123,
            owner="req-a",
            ttl_seconds=0.3,
            poll_interval_seconds=0.1,
            clock=clock,
            sleep=sleeper,
        ) as lease:
            await sleeper.waiting.wait()
            blocker = sqlite3.connect(lock_db, timeout=0)
            blocker.execute("PRAGMA busy_timeout=0")
            blocker.execute("BEGIN IMMEDIATE")
            try:
                for _ in range(20):
                    previous_calls = len(sleeper.calls)
                    sleeper.release()
                    await _yield_until(
                        lambda: lease.lost or len(sleeper.calls) > previous_calls
                    )
                    if lease.lost:
                        break
                assert lease.lost is True
            finally:
                blocker.rollback()
                blocker.close()
                blocker = None

    assert current_owner(lock_db, 123, clock=clock) is None


async def test_database_write_lock_is_bounded_across_independent_connections(lock_db):
    clock = FakeClock()
    monotonic = FakeMonotonic()
    sleeper = MonotonicSleep(monotonic)
    blocker = sqlite3.connect(lock_db, timeout=0)
    blocker.execute("PRAGMA busy_timeout=0")
    blocker.execute("BEGIN IMMEDIATE")

    try:
        with pytest.raises(ConversationBusyError):
            async with conversation_lease(
                lock_db,
                123,
                owner="req-a",
                ttl_seconds=30,
                acquire_timeout_seconds=0.5,
                poll_interval_seconds=0.1,
                clock=clock,
                monotonic_clock=monotonic,
                sleep=sleeper,
            ):
                pytest.fail("a second SQLite connection must not bypass the writer")
    finally:
        blocker.rollback()
        blocker.close()

    assert sum(sleeper.calls) == pytest.approx(0.5)
    assert current_owner(lock_db, 123, clock=clock) is None
