"""Cross-process renewable leases for Help Scout conversation mutations.

The lease table is deliberately small, but access to it is transactional: every
acquisition attempt takes a SQLite write reservation before comparing and
swapping the owner.  The async context manager adds bounded polling and a
heartbeat while keeping the individual database operations synchronous and
short.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import os
import re
import sqlite3
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, TypeVar

from services.runtime_paths import get_supa_db_path


Clock = Callable[[], datetime]
MonotonicClock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]
PathValue = str | os.PathLike[str]
ResultT = TypeVar("ResultT")
RenewalStatus = Literal["renewed", "lost", "busy"]
RenewalResult = tuple[RenewalStatus, datetime | None]
ReleaseStatus = Literal["released", "not_owner", "busy"]
OwnershipStatus = Literal["owned", "lost", "busy"]

DEFAULT_TTL_SECONDS = 30.0
DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 5.0
DEFAULT_POLL_INTERVAL_SECONDS = 0.05
# Public timing inputs are bounded to prevent sub-millisecond busy loops,
# datetime precision loss, and operational waits that outlive an API request.
MIN_TTL_SECONDS = 0.1
MAX_TTL_SECONDS = 86_400.0
MAX_ACQUIRE_TIMEOUT_SECONDS = 300.0
MIN_POLL_INTERVAL_SECONDS = 0.001
MAX_POLL_INTERVAL_SECONDS = 30.0
MIN_HEARTBEAT_INTERVAL_SECONDS = 0.001
MAX_HEARTBEAT_INTERVAL_SECONDS = 86_400.0
_STORED_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?$"
)


class ConversationLockError(Exception):
    """Base class for stable lease failures exposed by the Agent API."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool,
        status: int = 409,
        exit_code: int = 6,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status = status
        self.exit_code = exit_code
        self.details = dict(details or {})


class ConversationBusyError(ConversationLockError):
    """Another writer retained the conversation lease until the deadline."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(
            "The conversation is busy with another mutation.",
            code="conversation_busy",
            retryable=True,
            status=503,
            exit_code=6,
            details={"conversation_id": conversation_id},
        )


class ConversationLockLostError(ConversationLockError):
    """The caller no longer owns a lease required for a safe mutation."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(
            "The conversation lock was lost before the mutation completed.",
            code="conversation_lock_lost",
            retryable=False,
            status=409,
            exit_code=5,
            details={"conversation_id": conversation_id},
        )


class ConversationLockStorageError(ConversationLockError):
    """The lock store is unavailable, invalid, or has incompatible data."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(
            "The conversation lock store could not be read or updated safely.",
            code="conversation_lock_storage_error",
            retryable=False,
            status=500,
            exit_code=6,
            details={"conversation_id": conversation_id},
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_conversation_id(conversation_id: int | str) -> str:
    normalized = str(conversation_id).strip()
    if not normalized:
        raise ValueError("conversation_id must not be empty")
    return normalized


def _normalize_owner(owner: str) -> str:
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError("owner must not be empty")
    return owner


def _normalize_now(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("clock must return a datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return _normalize_now(value).isoformat(timespec="microseconds")


def _parse_stored_timestamp(value: Any, conversation_id: str) -> datetime:
    if not isinstance(value, str) or not _STORED_TIMESTAMP_RE.fullmatch(value):
        raise ConversationLockStorageError(conversation_id)
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        return _normalize_now(datetime.fromisoformat(normalized))
    except (TypeError, ValueError, OverflowError) as error:
        raise ConversationLockStorageError(conversation_id) from error


def _stored_lock_times(row: sqlite3.Row, conversation_id: str) -> tuple[datetime, datetime]:
    acquired_at = _parse_stored_timestamp(row["acquired_at"], conversation_id)
    expires_at = _parse_stored_timestamp(row["expires_at"], conversation_id)
    if acquired_at >= expires_at:
        raise ConversationLockStorageError(conversation_id)
    return acquired_at, expires_at


def _stored_lock_owner(row: sqlite3.Row, conversation_id: str) -> str:
    owner = row["owner"]
    if not isinstance(owner, str) or not owner.strip():
        raise ConversationLockStorageError(conversation_id)
    return owner


def _validate_timing(
    name: str,
    value: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized < minimum or normalized > maximum:
        raise ValueError(
            f"{name} must be between {minimum:g} and {maximum:g} seconds"
        )
    return normalized


def _expiry_after(now: datetime, ttl_seconds: float) -> datetime:
    try:
        expires_at = now + timedelta(seconds=ttl_seconds)
    except OverflowError as error:
        raise ValueError("ttl_seconds is outside the supported datetime range") from error
    if expires_at <= now:
        raise ValueError("ttl_seconds is below datetime precision")
    return expires_at


def _resolved_path(path: PathValue | None) -> str:
    if path is None:
        return get_supa_db_path()
    return os.fspath(Path(path))


def _connect(path: PathValue | None) -> sqlite3.Connection:
    # Lease waits are implemented explicitly with the injected async sleeper.
    # SQLite must therefore return BUSY immediately rather than hiding an
    # additional 30-second blocking wait inside a synchronous call.
    connection = sqlite3.connect(_resolved_path(path), timeout=0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=0")
    return connection


def _connect_for_lock(
    path: PathValue | None,
    conversation_id: str,
) -> sqlite3.Connection:
    try:
        return _connect(path)
    except sqlite3.Error as error:
        raise ConversationLockStorageError(conversation_id) from error


def _is_sqlite_busy(error: sqlite3.OperationalError) -> bool:
    busy_codes = {
        getattr(sqlite3, "SQLITE_BUSY", 5),
        getattr(sqlite3, "SQLITE_LOCKED", 6),
    }
    error_code = getattr(error, "sqlite_errorcode", None)
    return error_code in busy_codes or "locked" in str(error).lower()


def _rollback(connection: sqlite3.Connection) -> None:
    try:
        connection.rollback()
    except sqlite3.Error:
        pass


def _acquire_or_renew_lock(
    db_path: PathValue | None,
    conversation_id: int | str,
    *,
    owner: str,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    clock: Clock = _utc_now,
) -> datetime | None:
    """Try once to acquire or renew and return the actual stored expiry.

    Each call uses a fresh SQLite connection.  ``None`` means either another
    unexpired owner won the comparison or SQLite currently has another writer;
    the async lease wrapper performs bounded polling for both cases.
    """

    ttl = _validate_timing(
        "ttl_seconds",
        ttl_seconds,
        minimum=MIN_TTL_SECONDS,
        maximum=MAX_TTL_SECONDS,
    )
    lock_id = _normalize_conversation_id(conversation_id)
    lock_owner = _normalize_owner(owner)
    now = _normalize_now(clock())
    now_text = _timestamp(now)
    proposed_expires_at = _expiry_after(now, ttl)
    actual_expires_at = proposed_expires_at
    expires_text = _timestamp(proposed_expires_at)
    connection = _connect_for_lock(db_path, lock_id)

    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT owner, acquired_at, expires_at
            FROM agent_conversation_locks
            WHERE conversation_id = ?
            """,
            (lock_id,),
        ).fetchone()

        if row is None:
            connection.execute(
                """
                INSERT INTO agent_conversation_locks (
                    conversation_id, owner, acquired_at, expires_at
                ) VALUES (?, ?, ?, ?)
                """,
                (lock_id, lock_owner, now_text, expires_text),
            )
        else:
            stored_owner = _stored_lock_owner(row, lock_id)
            acquired_at, stored_expires_at = _stored_lock_times(row, lock_id)

        if row is not None and stored_owner == lock_owner and stored_expires_at > now:
            actual_expires_at = max(stored_expires_at, proposed_expires_at)
            expires_text = _timestamp(actual_expires_at)
            updated = connection.execute(
                """
                UPDATE agent_conversation_locks
                SET acquired_at = ?, expires_at = ?
                WHERE conversation_id = ?
                  AND owner = ?
                  AND acquired_at = ?
                  AND expires_at = ?
                """,
                (
                    _timestamp(acquired_at),
                    expires_text,
                    lock_id,
                    lock_owner,
                    row["acquired_at"],
                    row["expires_at"],
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                return None
        elif row is not None and stored_expires_at <= now:
            replaced = connection.execute(
                """
                UPDATE agent_conversation_locks
                SET owner = ?, acquired_at = ?, expires_at = ?
                WHERE conversation_id = ?
                  AND owner = ?
                  AND acquired_at = ?
                  AND expires_at = ?
                """,
                (
                    lock_owner,
                    now_text,
                    expires_text,
                    lock_id,
                    stored_owner,
                    row["acquired_at"],
                    row["expires_at"],
                ),
            )
            if replaced.rowcount != 1:
                connection.rollback()
                return None
        elif row is not None:
            connection.rollback()
            return None

        connection.commit()
        return actual_expires_at
    except sqlite3.OperationalError as error:
        _rollback(connection)
        if _is_sqlite_busy(error):
            return None
        raise ConversationLockStorageError(lock_id) from error
    except sqlite3.Error as error:
        _rollback(connection)
        raise ConversationLockStorageError(lock_id) from error
    except BaseException:
        _rollback(connection)
        raise
    finally:
        connection.close()


def acquire_conversation_lock(
    db_path: PathValue | None,
    conversation_id: int | str,
    *,
    owner: str,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    clock: Clock = _utc_now,
) -> bool:
    """Try once to acquire or renew a conversation lock."""

    return (
        _acquire_or_renew_lock(
            db_path,
            conversation_id,
            owner=owner,
            ttl_seconds=ttl_seconds,
            clock=clock,
        )
        is not None
    )


def _renew_owned_lock(
    db_path: PathValue | None,
    conversation_id: str,
    *,
    owner: str,
    ttl_seconds: float,
    clock: Clock,
) -> RenewalResult:
    now = _normalize_now(clock())
    proposed_expires_at = _expiry_after(now, ttl_seconds)
    connection = _connect_for_lock(db_path, conversation_id)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT owner, acquired_at, expires_at
            FROM agent_conversation_locks
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if row is None:
            connection.rollback()
            return "lost", None

        stored_owner = _stored_lock_owner(row, conversation_id)
        acquired_at, stored_expires_at = _stored_lock_times(row, conversation_id)
        if stored_owner != owner or stored_expires_at <= now:
            connection.rollback()
            return "lost", None

        actual_expires_at = max(stored_expires_at, proposed_expires_at)
        expires_text = _timestamp(actual_expires_at)

        updated = connection.execute(
            """
            UPDATE agent_conversation_locks
            SET acquired_at = ?, expires_at = ?
            WHERE conversation_id = ?
              AND owner = ?
              AND acquired_at = ?
              AND expires_at = ?
            """,
            (
                _timestamp(acquired_at),
                expires_text,
                conversation_id,
                owner,
                row["acquired_at"],
                row["expires_at"],
            ),
        )
        if updated.rowcount != 1:
            connection.rollback()
            return "lost", None
        connection.commit()
        return "renewed", actual_expires_at
    except sqlite3.OperationalError as error:
        _rollback(connection)
        if _is_sqlite_busy(error):
            return "busy", None
        raise ConversationLockStorageError(conversation_id) from error
    except sqlite3.Error as error:
        _rollback(connection)
        raise ConversationLockStorageError(conversation_id) from error
    except BaseException:
        _rollback(connection)
        raise
    finally:
        connection.close()


def _owns_unexpired_lock(
    db_path: PathValue | None,
    conversation_id: str,
    *,
    owner: str,
    clock: Clock,
) -> OwnershipStatus:
    now = _normalize_now(clock())
    connection = _connect_for_lock(db_path, conversation_id)
    try:
        row = connection.execute(
            """
            SELECT owner, acquired_at, expires_at
            FROM agent_conversation_locks
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
    except sqlite3.OperationalError as error:
        if _is_sqlite_busy(error):
            return "busy"
        raise ConversationLockStorageError(conversation_id) from error
    except sqlite3.Error as error:
        raise ConversationLockStorageError(conversation_id) from error
    finally:
        connection.close()
    if row is None:
        return "lost"
    stored_owner = _stored_lock_owner(row, conversation_id)
    _, expires_at = _stored_lock_times(row, conversation_id)
    if stored_owner != owner or expires_at <= now:
        return "lost"
    return "owned"


def _release_owned_lock(
    db_path: PathValue | None,
    conversation_id: str,
    *,
    owner: str,
    clock: Clock,
) -> ReleaseStatus:
    now = _normalize_now(clock())
    connection = _connect_for_lock(db_path, conversation_id)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT owner, acquired_at, expires_at
            FROM agent_conversation_locks
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if row is None:
            connection.rollback()
            return "not_owner"

        stored_owner = _stored_lock_owner(row, conversation_id)
        _, expires_at = _stored_lock_times(row, conversation_id)
        if stored_owner != owner or expires_at <= now:
            connection.rollback()
            return "not_owner"

        deleted = connection.execute(
            """
            DELETE FROM agent_conversation_locks
            WHERE conversation_id = ?
              AND owner = ?
              AND acquired_at = ?
              AND expires_at = ?
            """,
            (
                conversation_id,
                owner,
                row["acquired_at"],
                row["expires_at"],
            ),
        )
        connection.commit()
        return "released" if deleted.rowcount == 1 else "not_owner"
    except sqlite3.OperationalError as error:
        _rollback(connection)
        if _is_sqlite_busy(error):
            return "busy"
        raise ConversationLockStorageError(conversation_id) from error
    except sqlite3.Error as error:
        _rollback(connection)
        raise ConversationLockStorageError(conversation_id) from error
    except BaseException:
        _rollback(connection)
        raise
    finally:
        connection.close()


def release_conversation_lock(
    db_path: PathValue | None,
    conversation_id: int | str,
    *,
    owner: str,
    clock: Clock = _utc_now,
) -> bool:
    """Release an owned lock, distinguishing database contention from mismatch."""

    lock_id = _normalize_conversation_id(conversation_id)
    lock_owner = _normalize_owner(owner)
    release_status = _release_owned_lock(
        db_path,
        lock_id,
        owner=lock_owner,
        clock=clock,
    )
    if release_status == "busy":
        raise ConversationBusyError(lock_id)
    return release_status == "released"


def current_owner(
    db_path: PathValue | None,
    conversation_id: int | str,
    *,
    clock: Clock = _utc_now,
) -> str | None:
    """Return the current unexpired owner, or ``None`` when unlocked/expired."""

    lock_id = _normalize_conversation_id(conversation_id)
    now = _normalize_now(clock())
    connection = _connect_for_lock(db_path, lock_id)
    try:
        row = connection.execute(
            """
            SELECT owner, acquired_at, expires_at
            FROM agent_conversation_locks
            WHERE conversation_id = ?
            """,
            (lock_id,),
        ).fetchone()
    except sqlite3.OperationalError as error:
        if _is_sqlite_busy(error):
            raise ConversationBusyError(lock_id) from error
        raise ConversationLockStorageError(lock_id) from error
    except sqlite3.Error as error:
        raise ConversationLockStorageError(lock_id) from error
    finally:
        connection.close()
    if row is None:
        return None
    stored_owner = _stored_lock_owner(row, lock_id)
    _, expires_at = _stored_lock_times(row, lock_id)
    if expires_at <= now:
        return None
    return stored_owner


class ConversationLease:
    """Async lease context and mutation guard for one conversation."""

    def __init__(
        self,
        db_path: PathValue | None,
        conversation_id: int | str,
        *,
        owner: str,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        acquire_timeout_seconds: float = DEFAULT_ACQUIRE_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        heartbeat_interval_seconds: float | None = None,
        clock: Clock = _utc_now,
        monotonic_clock: MonotonicClock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        ttl = _validate_timing(
            "ttl_seconds",
            ttl_seconds,
            minimum=MIN_TTL_SECONDS,
            maximum=MAX_TTL_SECONDS,
        )
        acquire_timeout = _validate_timing(
            "acquire_timeout_seconds",
            acquire_timeout_seconds,
            minimum=0,
            maximum=MAX_ACQUIRE_TIMEOUT_SECONDS,
        )
        poll_interval = _validate_timing(
            "poll_interval_seconds",
            poll_interval_seconds,
            minimum=MIN_POLL_INTERVAL_SECONDS,
            maximum=MAX_POLL_INTERVAL_SECONDS,
        )
        heartbeat_interval = (
            None
            if heartbeat_interval_seconds is None
            else _validate_timing(
                "heartbeat_interval_seconds",
                heartbeat_interval_seconds,
                minimum=MIN_HEARTBEAT_INTERVAL_SECONDS,
                maximum=MAX_HEARTBEAT_INTERVAL_SECONDS,
            )
        )

        maximum_heartbeat_interval = ttl / 3
        requested_heartbeat_interval = (
            maximum_heartbeat_interval
            if heartbeat_interval is None
            else heartbeat_interval
        )

        self.db_path = db_path
        self.conversation_id = _normalize_conversation_id(conversation_id)
        self.owner = _normalize_owner(owner)
        self.ttl_seconds = ttl
        self.acquire_timeout_seconds = acquire_timeout
        self.poll_interval_seconds = poll_interval
        self.heartbeat_interval_seconds = min(
            float(requested_heartbeat_interval), maximum_heartbeat_interval
        )
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._sleep = sleep
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._acquired = False
        self._entered = False
        self._lost_error: ConversationLockLostError | None = None
        self._heartbeat_error: ConversationLockError | None = None
        self._expires_at: datetime | None = None

    @property
    def lost(self) -> bool:
        return self._lost_error is not None

    @property
    def expires_at(self) -> datetime | None:
        """Return the last expiry confirmed by an acquire or renewal transaction."""

        return self._expires_at

    def _mark_lost(self) -> ConversationLockLostError:
        if self._lost_error is None:
            self._lost_error = ConversationLockLostError(self.conversation_id)
        return self._lost_error

    async def _acquire_bounded(self) -> None:
        deadline = self._monotonic_clock() + self.acquire_timeout_seconds

        while True:
            attempt_now = _normalize_now(self._clock())
            acquired_expires_at = _acquire_or_renew_lock(
                self.db_path,
                self.conversation_id,
                owner=self.owner,
                ttl_seconds=self.ttl_seconds,
                clock=lambda: attempt_now,
            )
            if acquired_expires_at is not None:
                self._acquired = True
                self._expires_at = acquired_expires_at
                return

            remaining = deadline - self._monotonic_clock()
            if remaining <= 0:
                raise ConversationBusyError(self.conversation_id)
            await self._sleep(min(self.poll_interval_seconds, remaining))

    async def _heartbeat_loop(self) -> None:
        delay = self.heartbeat_interval_seconds
        busy_retry_interval = min(
            self.poll_interval_seconds,
            self.heartbeat_interval_seconds / 2,
        )
        try:
            while True:
                await self._sleep(delay)
                renewal_now = _normalize_now(self._clock())
                if self._expires_at is None or renewal_now >= self._expires_at:
                    self._mark_lost()
                    return
                renewal_status, renewed_expires_at = _renew_owned_lock(
                    self.db_path,
                    self.conversation_id,
                    owner=self.owner,
                    ttl_seconds=self.ttl_seconds,
                    clock=lambda: renewal_now,
                )
                if renewal_status == "lost":
                    self._mark_lost()
                    return
                if renewal_status == "renewed":
                    if renewed_expires_at is None:
                        self._heartbeat_error = ConversationLockStorageError(
                            self.conversation_id
                        )
                        return
                    self._expires_at = renewed_expires_at
                delay = (
                    busy_retry_interval
                    if renewal_status == "busy"
                    else self.heartbeat_interval_seconds
                )
        except asyncio.CancelledError:
            raise
        except ConversationLockError as error:
            self._heartbeat_error = error
        except Exception as error:
            self._heartbeat_error = ConversationLockStorageError(
                self.conversation_id
            )

    async def _stop_heartbeat(self) -> None:
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except ConversationLockError as error:
            self._heartbeat_error = error
        except Exception:
            self._heartbeat_error = ConversationLockStorageError(
                self.conversation_id
            )

    async def _release_bounded(self) -> ReleaseStatus:
        deadline = self._monotonic_clock() + self.acquire_timeout_seconds
        while True:
            release_status = _release_owned_lock(
                self.db_path,
                self.conversation_id,
                owner=self.owner,
                clock=self._clock,
            )
            if release_status != "busy":
                return release_status

            if self._expires_at is None:
                return "not_owner"
            lease_remaining = (
                self._expires_at - _normalize_now(self._clock())
            ).total_seconds()
            if lease_remaining <= 0:
                return "not_owner"
            remaining = deadline - self._monotonic_clock()
            if remaining <= 0:
                raise ConversationBusyError(self.conversation_id)
            await self._sleep(
                min(self.poll_interval_seconds, remaining, lease_remaining)
            )

    async def verify_owned(self) -> None:
        """Raise a stable error unless this lease is still current and owned."""

        deadline = self._monotonic_clock() + self.acquire_timeout_seconds
        while True:
            if self._lost_error is not None:
                raise self._lost_error
            if self._heartbeat_error is not None:
                raise self._heartbeat_error
            if not self._acquired or self._expires_at is None:
                raise self._mark_lost()
            now = _normalize_now(self._clock())
            if now >= self._expires_at:
                raise self._mark_lost()
            ownership_status = _owns_unexpired_lock(
                self.db_path,
                self.conversation_id,
                owner=self.owner,
                clock=lambda: now,
            )
            if ownership_status == "owned":
                return
            if ownership_status == "lost":
                raise self._mark_lost()

            remaining = deadline - self._monotonic_clock()
            if remaining <= 0:
                raise ConversationBusyError(self.conversation_id)
            lease_remaining = (self._expires_at - now).total_seconds()
            await self._sleep(
                min(self.poll_interval_seconds, remaining, lease_remaining)
            )

    async def renew(self) -> None:
        """Renew immediately, primarily for explicit long-operation checkpoints."""

        if self._lost_error is not None or not self._acquired:
            raise self._mark_lost()
        if self._heartbeat_error is not None:
            raise self._heartbeat_error
        renewal_now = _normalize_now(self._clock())
        renewal_status, renewed_expires_at = _renew_owned_lock(
            self.db_path,
            self.conversation_id,
            owner=self.owner,
            ttl_seconds=self.ttl_seconds,
            clock=lambda: renewal_now,
        )
        if renewal_status == "lost":
            raise self._mark_lost()
        if renewal_status == "busy":
            raise ConversationBusyError(self.conversation_id)
        if renewed_expires_at is None:
            raise ConversationLockStorageError(self.conversation_id)
        self._expires_at = renewed_expires_at

    async def run_guarded(
        self,
        mutation: Callable[..., Awaitable[ResultT]],
        *args: Any,
        **kwargs: Any,
    ) -> ResultT:
        """Verify ownership immediately before invoking one async mutation."""

        async_call = getattr(mutation, "__call__", None)
        if not (
            inspect.iscoroutinefunction(mutation)
            or inspect.iscoroutinefunction(async_call)
        ):
            raise TypeError("guarded mutation callback must be async")
        await self.verify_owned()
        return await mutation(*args, **kwargs)

    async def __aenter__(self) -> ConversationLease:
        if self._entered:
            raise RuntimeError("a conversation lease context cannot be re-entered")
        self._entered = True
        try:
            await self._acquire_bounded()
        except BaseException:
            self._entered = False
            raise
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
        await self._stop_heartbeat()
        verification_error: Exception | None = None
        cleanup_error: Exception | None = None

        try:
            await self.verify_owned()
        except Exception as error:
            verification_error = error

        try:
            release_status = await self._release_bounded()
            if verification_error is None and release_status == "not_owner":
                verification_error = self._mark_lost()
        except Exception as error:
            cleanup_error = error
        finally:
            self._acquired = False
            self._entered = False

        # Cleanup must never replace the exception raised by the protected
        # upstream operation.  Normal exit, however, reports lost ownership.
        if exc_type is not None:
            return False
        if verification_error is not None:
            raise verification_error
        if cleanup_error is not None:
            raise cleanup_error
        return False


def conversation_lease(
    db_path: PathValue | None,
    conversation_id: int | str,
    *,
    owner: str,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    acquire_timeout_seconds: float = DEFAULT_ACQUIRE_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    heartbeat_interval_seconds: float | None = None,
    clock: Clock = _utc_now,
    monotonic_clock: MonotonicClock = time.monotonic,
    sleep: Sleep = asyncio.sleep,
) -> ConversationLease:
    """Build a renewable async lease for one conversation mutation sequence."""

    return ConversationLease(
        db_path,
        conversation_id,
        owner=owner,
        ttl_seconds=ttl_seconds,
        acquire_timeout_seconds=acquire_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        clock=clock,
        monotonic_clock=monotonic_clock,
        sleep=sleep,
    )
