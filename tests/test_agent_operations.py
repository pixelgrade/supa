from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from db.database import init_db
from services.agent_auth import AgentIdentity
from services.agent_operations import (
    DEFAULT_STALE_AFTER_SECONDS,
    AgentOperationConflictError,
    AgentOperationDataError,
    AgentOperationNotFoundError,
    AgentOperationStateError,
    AgentOperationStorageError,
    begin_operation,
    canonical_request_json,
    finish_operation,
    get_operation,
    hash_request,
    mark_upstream_started,
    recover_stale_operations,
    reserve_operation,
    resolve_operation_locally,
)


UTC = timezone.utc
IDENTITY = AgentIdentity(
    token_id="token-1",
    name="Laptop agent",
    scopes=frozenset({"read", "draft", "send"}),
)
OTHER_IDENTITY = AgentIdentity(
    token_id="token-2",
    name="Other agent",
    scopes=frozenset({"read", "draft"}),
)


class FakeClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def advance(self, **kwargs: float) -> None:
        self.current += timedelta(**kwargs)


@pytest.fixture
def owner_controlled_directory(tmp_path):
    """Use an entirely owner-controlled path for the local maintenance CLI."""
    import pwd
    import tempfile

    directory = tmp_path.resolve()
    if all(path.stat().st_mode & 0o022 == 0 for path in (directory, *directory.parents)):
        yield directory
        return
    # A sticky system /tmp is intentionally rejected by the real CLI.
    home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    with tempfile.TemporaryDirectory(prefix=".supa-ledger-test-", dir=home) as directory:
        yield Path(directory)


@pytest.fixture
def operations_db(owner_controlled_directory):
    path = owner_controlled_directory / "operations.db"
    init_db(str(path))
    return path


def _reserve(
    operations_db,
    *,
    key: str = "idem-1",
    identity: AgentIdentity = IDENTITY,
    request: dict | None = None,
    metadata: dict | None = None,
    operation_name: str = "draft.save",
    clock=None,
):
    return reserve_operation(
        db_path=operations_db,
        idempotency_key=key,
        identity=identity,
        scope="draft",
        operation_name=operation_name,
        target_id="602",
        request_id=f"request-{key}",
        request=request or {"body": "Hello", "body_format": "text"},
        request_metadata=metadata
        or {"body_sha256": "a" * 64, "expected_version": None},
        clock=clock,
    )


def _all_audit_text(path) -> str:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT * FROM agent_audit ORDER BY id"
        ).fetchall()
    return "\n".join(str(value) for row in rows for value in row)


def test_request_hash_uses_sorted_compact_canonical_json():
    left = {"z": [3, {"b": 2, "a": 1}], "a": "value"}
    right = {"a": "value", "z": [3, {"a": 1, "b": 2}]}

    assert canonical_request_json(left) == (
        '{"a":"value","z":[3,{"a":1,"b":2}]}'
    )
    assert hash_request(left) == hash_request(right)
    assert len(hash_request(left)) == 64


@pytest.mark.parametrize(
    "invalid_request",
    [
        {1: "numeric key", "1": "string key"},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": {"unsupported"}},
        {"value": "\ud800"},
        {"value": 10**5_000},
    ],
)
def test_canonical_request_rejects_non_json_or_ambiguous_values(invalid_request):
    with pytest.raises(ValueError, match="request must be valid JSON"):
        canonical_request_json(invalid_request)


def test_canonical_request_rejects_cycles_excess_depth_and_excess_items():
    recursive = []
    recursive.append(recursive)
    too_deep: object = "leaf"
    for _ in range(40):
        too_deep = [too_deep]

    for invalid_request in (
        {"recursive": recursive},
        {"deep": too_deep},
        {"many": list(range(10_001))},
    ):
        with pytest.raises(ValueError, match="request must be valid JSON"):
            canonical_request_json(invalid_request)


def test_reserve_persists_one_operation_and_append_only_audit(operations_db):
    secret_body = "private customer message"
    raw_token = "Bearer raw-secret-token"
    reservation = _reserve(
        operations_db,
        request={"body": secret_body, "body_format": "text"},
        metadata={
            "api_key": "fixture-b",
            "body": secret_body,
            "customer_message": secret_body,
            "note": secret_body,
            "authorization": raw_token,
            "payload": secret_body,
            "status": secret_body,
            "body_sha256": "b" * 64,
            "expected_thread_id": "12",
        },
    )

    assert reservation.replayed is False
    assert reservation.operation.state == "reserved"
    assert reservation.operation.request_hash == hash_request(
        {"body": secret_body, "body_format": "text"}
    )
    assert reservation.operation.request_metadata["body_sha256"] == "b" * 64
    assert secret_body not in json.dumps(reservation.operation.request_metadata)
    assert raw_token not in json.dumps(reservation.operation.request_metadata)
    assert "fixture-b" not in json.dumps(
        reservation.operation.request_metadata
    )

    with sqlite3.connect(operations_db) as connection:
        operation_count = connection.execute(
            "SELECT COUNT(*) FROM agent_operations"
        ).fetchone()[0]
        audit = connection.execute(
            "SELECT token_id, token_name, scope, result, request_hash "
            "FROM agent_audit"
        ).fetchone()
    assert operation_count == 1
    assert audit == (
        IDENTITY.token_id,
        IDENTITY.name,
        "draft",
        "reserved",
        reservation.operation.request_hash,
    )
    assert secret_body not in _all_audit_text(operations_db)
    assert raw_token not in _all_audit_text(operations_db)


def test_same_key_same_request_replays_without_new_audit(operations_db):
    first = _reserve(operations_db)
    second = _reserve(operations_db, request={"body_format": "text", "body": "Hello"})

    assert first.replayed is False
    assert second.replayed is True
    assert second.operation == first.operation
    with sqlite3.connect(operations_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM agent_audit").fetchone()[0] == 1


def test_same_key_different_request_conflicts_without_mutation(operations_db):
    first = _reserve(operations_db)

    with pytest.raises(AgentOperationConflictError) as raised:
        _reserve(operations_db, request={"body": "Different", "body_format": "text"})

    assert raised.value.code == "idempotency_conflict"
    assert raised.value.retryable is False
    assert raised.value.status == 409
    assert get_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
    ) == first.operation


def test_same_key_cannot_cross_token_ownership(operations_db):
    _reserve(operations_db)

    with pytest.raises(AgentOperationConflictError):
        _reserve(operations_db, identity=OTHER_IDENTITY)

    with pytest.raises(AgentOperationNotFoundError) as raised:
        get_operation(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=OTHER_IDENTITY,
        )
    assert raised.value.code == "operation_not_found"
    assert raised.value.status == 404

    with pytest.raises(AgentOperationNotFoundError):
        begin_operation(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=OTHER_IDENTITY,
            scope="draft",
            request_id="foreign-begin",
        )


def test_true_concurrent_reservation_has_one_winner_and_one_replay(operations_db):
    def reserve_once():
        return _reserve(operations_db)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reserve_once(), range(2)))

    assert sorted(result.replayed for result in results) == [False, True]
    with sqlite3.connect(operations_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM agent_operations"
        ).fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM agent_audit").fetchone()[0] == 1


def test_exact_legal_transition_records_upstream_start_before_completion(operations_db):
    clock = FakeClock(datetime(2026, 9, 25, 12, 0, tzinfo=UTC))
    _reserve(operations_db, clock=clock)
    clock.advance(seconds=1)

    active = begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="request-begin",
        clock=clock,
    )
    assert active.state == "in_progress"
    assert active.upstream_started_at is None

    clock.advance(seconds=1)
    started = mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="request-submit",
        clock=clock,
    )
    assert started.upstream_started_at == clock.current

    clock.advance(seconds=1)
    complete = finish_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="request-finish",
        outcome="succeeded",
        upstream_thread_id="987",
        response={"thread_id": "987", "status": "saved"},
        clock=clock,
    )

    assert complete.state == "succeeded"
    assert complete.upstream_thread_id == "987"
    assert complete.completed_at == clock.current
    assert complete.upstream_started_at < complete.completed_at
    with sqlite3.connect(operations_db) as connection:
        results = [
            row[0]
            for row in connection.execute(
                "SELECT result FROM agent_audit ORDER BY id"
            ).fetchall()
        ]
    assert results == ["reserved", "in_progress", "upstream_started", "succeeded"]


def test_upstream_marker_atomically_persists_strict_attempt_metadata(operations_db):
    clock = FakeClock(datetime(2026, 9, 25, 12, 0, tzinfo=UTC))
    _reserve(
        operations_db,
        metadata={"body_format": "html", "expected_version": "b" * 64},
        clock=clock,
    )
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="begin",
        clock=clock,
    )
    clock.advance(seconds=1)
    attempt_metadata = {
        "normalized_body_hash": "A" * 64,
        "action": "updated",
        "conversation_id": 602,
        "expected_thread_id": 701,
    }

    started = mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="submit",
        attempt_metadata=attempt_metadata,
        clock=clock,
    )

    assert started.request_metadata == {
        "action": "updated",
        "attempted_at": "2026-09-25T12:00:01.000000+00:00",
        "body_format": "html",
        "conversation_id": "602",
        "expected_thread_id": "701",
        "expected_version": "b" * 64,
        "normalized_body_hash": "a" * 64,
    }
    with sqlite3.connect(operations_db) as connection:
        audit_metadata = json.loads(
            connection.execute(
                "SELECT metadata_json FROM agent_audit "
                "WHERE result = 'upstream_started'"
            ).fetchone()[0]
        )
    assert audit_metadata == {
        "action": "updated",
        "attempted_at": "2026-09-25T12:00:01.000000+00:00",
        "conversation_id": "602",
        "expected_thread_id": "701",
        "normalized_body_hash": "a" * 64,
    }


@pytest.mark.parametrize("operation_name", ["draft.send", "reply.send"])
def test_upstream_marker_accepts_distinct_strict_send_attempt_schema(
    operations_db, operation_name
):
    _reserve(operations_db, operation_name=operation_name)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="begin",
    )

    started = mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="submit",
        attempt_metadata={
            "normalized_body_hash": "d" * 64,
            "action": "sent",
            "conversation_id": 602,
            "expected_predecessor_thread_id": 500,
        },
    )

    assert started.request_metadata["action"] == "sent"
    assert started.request_metadata["conversation_id"] == "602"
    assert started.request_metadata["expected_predecessor_thread_id"] == "500"
    assert started.request_metadata["normalized_body_hash"] == "d" * 64
    assert started.request_metadata["attempted_at"]


def test_send_attempt_facts_survive_immediate_startup_unknown_recovery(
    operations_db,
):
    clock = FakeClock(datetime(2026, 9, 25, 12, 0, tzinfo=UTC))
    _reserve(
        operations_db,
        operation_name="reply.send",
        metadata={"conversation_id": 602},
        clock=clock,
    )
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="begin",
        clock=clock,
    )
    mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="submit",
        attempt_metadata={
            "normalized_body_hash": "d" * 64,
            "action": "sent",
            "conversation_id": 602,
            "expected_predecessor_thread_id": 500,
        },
        clock=clock,
    )

    summary = recover_stale_operations(db_path=operations_db, clock=clock)
    recovered = get_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
    )

    assert summary.outcome_unknown == 1
    assert recovered.state == "outcome_unknown"
    assert recovered.request_metadata == {
        "action": "sent",
        "attempted_at": "2026-09-25T12:00:00.000000+00:00",
        "conversation_id": "602",
        "expected_predecessor_thread_id": "500",
        "normalized_body_hash": "d" * 64,
    }


@pytest.mark.parametrize(
    "attempt_metadata",
    [
        {
            "normalized_body_hash": "d" * 64,
            "action": "sent",
            "conversation_id": 602,
        },
        {
            "normalized_body_hash": "d" * 64,
            "action": "created",
            "conversation_id": 602,
            "expected_predecessor_thread_id": 500,
        },
        {
            "normalized_body_hash": "d" * 64,
            "action": "sent",
            "conversation_id": 602,
            "expected_predecessor_thread_id": None,
        },
    ],
)
def test_invalid_send_attempt_schema_does_not_weaken_draft_schema(
    operations_db, attempt_metadata
):
    _reserve(operations_db)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="begin",
    )

    with pytest.raises(ValueError):
        mark_upstream_started(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
            scope="send",
            request_id="submit",
            attempt_metadata=attempt_metadata,
        )

    record = get_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
    )
    assert record.state == "in_progress"
    assert record.upstream_started_at is None


@pytest.mark.parametrize(
    ("operation_name", "attempt_metadata"),
    [
        (
            "draft.save",
            {
                "normalized_body_hash": "d" * 64,
                "action": "sent",
                "conversation_id": 602,
                "expected_predecessor_thread_id": 500,
            },
        ),
        (
            "reply.send",
            {
                "normalized_body_hash": "d" * 64,
                "action": "created",
                "conversation_id": 602,
                "expected_thread_id": None,
            },
        ),
    ],
)
def test_attempt_schema_is_bound_to_durable_operation_name(
    operations_db, operation_name, attempt_metadata
):
    _reserve(operations_db, operation_name=operation_name)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="misleading-caller-scope",
        request_id="begin",
    )

    with pytest.raises(ValueError, match="does not match the operation"):
        mark_upstream_started(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
            scope="misleading-caller-scope",
            request_id="submit",
            attempt_metadata=attempt_metadata,
        )

    record = get_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
    )
    assert record.state == "in_progress"
    assert record.upstream_started_at is None


@pytest.mark.parametrize(
    "attempt_metadata",
    [
        {
            "action": "created",
            "conversation_id": 602,
            "expected_thread_id": None,
        },
        {
            "normalized_body_hash": "not-a-hash",
            "action": "created",
            "conversation_id": 602,
            "expected_thread_id": None,
        },
        {
            "normalized_body_hash": "a" * 64,
            "action": "updated",
            "conversation_id": 602,
            "expected_thread_id": None,
        },
        {
            "normalized_body_hash": "a" * 64,
            "action": "created",
            "conversation_id": 602,
            "expected_thread_id": None,
            "payload": "secret body",
        },
        {
            "normalized_body_hash": "a" * 64,
            "action": "sent",
            "conversation_id": 602,
            "expected_thread_id": None,
        },
    ],
)
def test_invalid_attempt_metadata_rolls_back_marker_and_audit(
    operations_db, attempt_metadata
):
    _reserve(operations_db)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="begin",
    )
    with sqlite3.connect(operations_db) as connection:
        original_metadata = connection.execute(
            "SELECT request_metadata_json FROM agent_operations"
        ).fetchone()[0]
        original_audit_count = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]

    with pytest.raises(ValueError):
        mark_upstream_started(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
            scope="draft",
            request_id="submit",
            attempt_metadata=attempt_metadata,
        )

    with sqlite3.connect(operations_db) as connection:
        row = connection.execute(
            "SELECT state, upstream_started_at, request_metadata_json "
            "FROM agent_operations"
        ).fetchone()
        audit_count = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]
    assert row == ("in_progress", None, original_metadata)
    assert audit_count == original_audit_count


def test_attempt_marker_rolls_back_if_bounded_storage_would_drop_attempt_facts(
    operations_db,
):
    _reserve(operations_db)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="begin",
    )
    long_reason = "x " * 2020
    nested_details = {"reason": long_reason}
    for _ in range(3):
        nested_details = {"reason": long_reason, "details": nested_details}
    near_limit_metadata = json.dumps(
        {"details": nested_details},
        separators=(",", ":"),
    )
    assert len(near_limit_metadata.encode("utf-8")) < 16 * 1024
    with sqlite3.connect(operations_db) as connection:
        connection.execute(
            "UPDATE agent_operations SET request_metadata_json = ? "
            "WHERE idempotency_key = 'idem-1'",
                (near_limit_metadata,),
        )
        original_audit_count = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]
        connection.commit()

    with pytest.raises(ValueError, match="attempt metadata could not be persisted"):
        mark_upstream_started(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
            scope="draft",
            request_id="submit",
            attempt_metadata={
                "normalized_body_hash": "a" * 64,
                "action": "created",
                "conversation_id": 602,
                "expected_thread_id": None,
            },
        )

    with sqlite3.connect(operations_db) as connection:
        row = connection.execute(
            "SELECT state, upstream_started_at, request_metadata_json "
            "FROM agent_operations WHERE idempotency_key = 'idem-1'"
        ).fetchone()
        audit_count = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]
    assert row == ("in_progress", None, near_limit_metadata)
    assert audit_count == original_audit_count


def test_upstream_start_is_one_shot_to_prevent_duplicate_submission(operations_db):
    _reserve(operations_db)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="begin",
    )
    mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="submit",
    )

    with pytest.raises(AgentOperationStateError):
        mark_upstream_started(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
            scope="draft",
            request_id="submit-again",
        )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("reserved", "succeeded"),
        ("reserved", "failed"),
        ("reserved", "outcome_unknown"),
        ("in_progress", "reserved"),
        ("succeeded", "failed"),
        ("failed", "in_progress"),
        ("sent_with_warning", "succeeded"),
    ],
)
def test_forbidden_state_changes_and_terminal_immutability(
    operations_db, source, target
):
    _reserve(operations_db)
    if source != "reserved":
        with sqlite3.connect(operations_db) as connection:
            connection.execute(
                "UPDATE agent_operations SET state = ? WHERE idempotency_key = ?",
                (source, "idem-1"),
            )
            connection.commit()

    if target == "in_progress":
        action = lambda: begin_operation(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
            scope="draft",
            request_id="invalid",
        )
    else:
        action = lambda: finish_operation(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
            scope="draft",
            request_id="invalid",
            outcome=target,
        )

    with pytest.raises(AgentOperationStateError) as raised:
        action()
    assert raised.value.code == "operation_state_conflict"


def test_unknown_replay_never_re_reserves_or_retries(operations_db):
    _reserve(operations_db)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="begin",
    )
    mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="submit",
    )
    finish_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="unknown",
        outcome="outcome_unknown",
        error={"code": "response_lost"},
    )

    replay = _reserve(operations_db)

    assert replay.replayed is True
    assert replay.operation.state == "outcome_unknown"
    with sqlite3.connect(operations_db) as connection:
        results = [
            row[0]
            for row in connection.execute(
                "SELECT result FROM agent_audit ORDER BY id"
            )
        ]
    assert results[-1] == "outcome_unknown"


def test_response_error_and_public_read_are_sanitized_and_bounded(operations_db):
    _reserve(operations_db)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="begin",
    )
    mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="submit",
    )
    private_body = "PRIVATE-BODY-SHOULD-NOT-PERSIST"
    raw_token = "RAW-BEARER-SHOULD-NOT-PERSIST"
    operation = finish_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="finish",
        outcome="failed",
        error={
            "code": "upstream_error",
            "body": private_body,
            "authorization": raw_token,
            "apiKey": "fixture-b",
            "payload": private_body,
            "note": private_body,
            "status": private_body,
            "detail": f"upstream echoed Bearer {raw_token} " + ("x" * 100_000),
        },
    )

    public = operation.to_public_dict()
    serialized = json.dumps(public, sort_keys=True)
    assert "token-1" not in serialized
    assert "Laptop agent" not in serialized
    assert private_body not in serialized
    assert raw_token not in serialized
    assert "fixture-b" not in serialized
    with sqlite3.connect(operations_db) as connection:
        stored = "\n".join(
            str(value)
            for row in connection.execute(
                "SELECT request_metadata_json, response_json, error_json "
                "FROM agent_operations"
            )
            for value in row
            if value is not None
        )
    assert private_body not in stored
    assert raw_token not in stored
    assert "fixture-b" not in stored
    assert len(stored.encode("utf-8")) < 40_000


def test_public_read_defensively_sanitizes_preexisting_unsanitized_json(operations_db):
    _reserve(operations_db)
    with sqlite3.connect(operations_db) as connection:
        connection.execute(
            "UPDATE agent_operations SET response_json = ?, error_json = ? "
            "WHERE idempotency_key = ?",
            (
                json.dumps({"body_html": "LEGACY-PRIVATE-BODY", "status": "ok"}),
                json.dumps(
                    {
                        "authorization": "Bearer LEGACY-SECRET",
                        "code": "old",
                        "detail": "upstream echoed Bearer LEGACY-SECRET",
                    }
                ),
                "idem-1",
            ),
        )
        connection.commit()

    public = get_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
    ).to_public_dict()

    serialized = json.dumps(public)
    assert "LEGACY-PRIVATE-BODY" not in serialized
    assert "LEGACY-SECRET" not in serialized
    assert public["response"]["status"] == "ok"


def test_persisted_fields_use_explicit_hash_id_version_semantics(operations_db):
    token_like = "A" * 43
    reservation = _reserve(
        operations_db,
        metadata={
            "access_id": "123",
            "body_hash": "b" * 64,
            "message_id": "456",
            "body_sha256": "c" * 64,
            "normalized_body_hash": "d" * 64,
            "expected_thread_id": "789",
            "expected_version": "e" * 64,
            "thread_id": "not-a-numeric-id",
            "code": token_like,
            "reason": f"Observed leaked token {token_like}",
        },
    )

    metadata = reservation.operation.request_metadata
    assert metadata["body_sha256"] == "c" * 64
    assert metadata["normalized_body_hash"] == "d" * 64
    assert metadata["expected_thread_id"] == "789"
    assert metadata["expected_version"] == "e" * 64
    assert "access_id" not in metadata
    assert "body_hash" not in metadata
    assert "message_id" not in metadata
    assert "thread_id" not in metadata
    assert token_like not in json.dumps(metadata)


def test_concatenated_supa_token_is_redacted_but_semantic_hashes_survive(
    operations_db,
):
    token_like = "A" * 43
    exposed = f"prefix_{token_like}_suffix"

    _reserve(
        operations_db,
        metadata={
            "body_sha256": "b" * 64,
            "version": "c" * 64,
            "reason": f"Observed {exposed}",
        },
    )

    public = get_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
    ).to_public_dict()
    metadata = public["request_metadata"]
    serialized = json.dumps(public)
    assert metadata["body_sha256"] == "b" * 64
    assert metadata["version"] == "c" * 64
    assert token_like not in serialized
    assert exposed not in serialized


def test_public_read_revalidates_semantic_fields_and_redacts_supa_tokens(operations_db):
    token_like = "Z" * 43
    _reserve(operations_db)
    with sqlite3.connect(operations_db) as connection:
        connection.execute(
            "UPDATE agent_operations SET response_json = ? "
            "WHERE idempotency_key = ?",
            (
                json.dumps(
                    {
                        "access_id": "12",
                        "body_hash": "b" * 64,
                        "message_id": "13",
                        "thread_id": "14",
                        "version": "f" * 64,
                        "code": token_like,
                        "reason": f"Leaked {token_like}",
                    }
                ),
                "idem-1",
            ),
        )
        connection.commit()

    response = get_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
    ).response

    assert response["thread_id"] == "14"
    assert response["version"] == "f" * 64
    assert "access_id" not in response
    assert "body_hash" not in response
    assert "message_id" not in response
    assert token_like not in json.dumps(response)


def test_public_read_rejects_token_like_top_level_stored_strings(operations_db):
    token_like = "Q" * 43
    _reserve(operations_db)
    with sqlite3.connect(operations_db) as connection:
        connection.execute(
            "UPDATE agent_operations SET target_id = ? "
            "WHERE idempotency_key = 'idem-1'",
            (token_like,),
        )
        connection.commit()

    with pytest.raises(AgentOperationDataError):
        get_operation(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
        )


def test_startup_recovery_fails_only_pre_attempt_reservation_and_marks_possible_writes_unknown(
    operations_db,
):
    old = datetime(2026, 9, 25, 10, 0, tzinfo=UTC)
    clock = FakeClock(old)
    _reserve(operations_db, key="reserved", clock=clock)
    _reserve(operations_db, key="active", clock=clock)
    begin_operation(
        db_path=operations_db,
        idempotency_key="active",
        identity=IDENTITY,
        scope="draft",
        request_id="active-begin",
        clock=clock,
    )
    _reserve(operations_db, key="started-reserved", clock=clock)
    with sqlite3.connect(operations_db) as connection:
        connection.execute(
            "UPDATE agent_operations SET upstream_started_at = ? "
            "WHERE idempotency_key = ?",
            (old.isoformat(), "started-reserved"),
        )
        connection.commit()
    clock.advance(seconds=DEFAULT_STALE_AFTER_SECONDS + 1)
    _reserve(operations_db, key="fresh", clock=clock)
    _reserve(operations_db, key="fresh-started", clock=clock)
    begin_operation(
        db_path=operations_db,
        idempotency_key="fresh-started",
        identity=IDENTITY,
        scope="send",
        request_id="fresh-started-begin",
        clock=clock,
    )
    mark_upstream_started(
        db_path=operations_db,
        idempotency_key="fresh-started",
        identity=IDENTITY,
        scope="send",
        request_id="fresh-started-submit",
        attempt_metadata={
            "normalized_body_hash": "c" * 64,
            "action": "created",
            "conversation_id": 602,
            "expected_thread_id": None,
        },
        clock=clock,
    )

    summary = recover_stale_operations(db_path=operations_db, clock=clock)

    assert summary.failed_before_attempt == 1
    assert summary.outcome_unknown == 3
    assert summary.examined == 4
    assert get_operation(
        db_path=operations_db,
        idempotency_key="reserved",
        identity=IDENTITY,
    ).error["code"] == "interrupted_before_attempt"
    assert get_operation(
        db_path=operations_db,
        idempotency_key="active",
        identity=IDENTITY,
    ).state == "outcome_unknown"
    assert get_operation(
        db_path=operations_db,
        idempotency_key="fresh-started",
        identity=IDENTITY,
    ).request_metadata["normalized_body_hash"] == "c" * 64
    assert get_operation(
        db_path=operations_db,
        idempotency_key="started-reserved",
        identity=IDENTITY,
    ).state == "outcome_unknown"
    assert get_operation(
        db_path=operations_db,
        idempotency_key="fresh",
        identity=IDENTITY,
    ).state == "reserved"
    assert get_operation(
        db_path=operations_db,
        idempotency_key="fresh-started",
        identity=IDENTITY,
    ).state == "outcome_unknown"
    with sqlite3.connect(operations_db) as connection:
        recovery_metadata = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT metadata_json FROM agent_audit "
                "WHERE result IN ('failed', 'outcome_unknown')"
            )
        ]
    assert recovery_metadata
    assert all(row["source"] == "startup_recovery" for row in recovery_metadata)


def test_startup_recovery_validates_every_row_before_any_mutation(operations_db):
    old = datetime(2026, 9, 25, 10, 0, tzinfo=UTC)
    clock = FakeClock(old)
    _reserve(operations_db, key="valid-stale", clock=clock)
    _reserve(operations_db, key="corrupt-marker", clock=clock)
    with sqlite3.connect(operations_db) as connection:
        connection.execute(
            "UPDATE agent_operations SET upstream_started_at = 'not-a-timestamp' "
            "WHERE idempotency_key = 'corrupt-marker'"
        )
        original_audit_count = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]
        connection.commit()
    clock.advance(seconds=DEFAULT_STALE_AFTER_SECONDS + 1)

    with pytest.raises(AgentOperationDataError):
        recover_stale_operations(db_path=operations_db, clock=clock)

    with sqlite3.connect(operations_db) as connection:
        states = dict(
            connection.execute(
                "SELECT idempotency_key, state FROM agent_operations"
            ).fetchall()
        )
        audit_count = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]
    assert states["valid-stale"] == "reserved"
    assert states["corrupt-marker"] == "reserved"
    assert audit_count == original_audit_count


def test_local_resolution_only_accepts_unknown_and_audits_reason_and_thread(
    operations_db,
):
    _reserve(operations_db)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="begin",
    )
    mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="submit",
    )
    finish_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="unknown",
        outcome="outcome_unknown",
    )

    resolved = resolve_operation_locally(
        db_path=operations_db,
        idempotency_key="idem-1",
        outcome="succeeded",
        reason="Observed exactly one matching Help Scout reply",
        upstream_thread_id="777",
    )

    assert resolved.state == "succeeded"
    assert resolved.upstream_thread_id == "777"
    assert resolved.response["resolution"] == "manual"
    with sqlite3.connect(operations_db) as connection:
        row = connection.execute(
            "SELECT scope, result, metadata_json FROM agent_audit ORDER BY id DESC"
        ).fetchone()
    assert row[0:2] == ("maintenance", "resolved_succeeded")
    assert json.loads(row[2]) == {
        "reason": "Observed exactly one matching Help Scout reply",
        "thread_id": "777",
    }


def test_successful_local_resolution_requires_an_observed_thread_id(operations_db):
    _reserve(operations_db)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="begin",
    )
    mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="submit",
    )
    finish_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="unknown",
        outcome="outcome_unknown",
    )

    with pytest.raises(AgentOperationStateError):
        resolve_operation_locally(
            db_path=operations_db,
            idempotency_key="idem-1",
            outcome="succeeded",
            reason="Observed a success but did not record which reply",
        )

    assert get_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
    ).state == "outcome_unknown"


def test_successful_local_resolution_audits_existing_effective_thread_id(operations_db):
    _reserve(operations_db)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="begin",
    )
    mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="submit",
    )
    finish_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="unknown",
        outcome="outcome_unknown",
        upstream_thread_id="778",
    )

    resolved = resolve_operation_locally(
        db_path=operations_db,
        idempotency_key="idem-1",
        outcome="succeeded",
        reason="Confirmed the recorded reply",
    )

    assert resolved.upstream_thread_id == "778"
    with sqlite3.connect(operations_db) as connection:
        metadata = json.loads(
            connection.execute(
                "SELECT metadata_json FROM agent_audit ORDER BY id DESC"
            ).fetchone()[0]
        )
    assert metadata["thread_id"] == "778"


def test_local_resolution_rejects_fresh_active_and_terminal_operations(operations_db):
    _reserve(operations_db)
    with pytest.raises(AgentOperationStateError):
        resolve_operation_locally(
            db_path=operations_db,
            idempotency_key="idem-1",
            outcome="failed",
            reason="Not attempted",
        )
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="begin",
    )
    finish_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="draft",
        request_id="failed",
        outcome="failed",
    )
    with pytest.raises(AgentOperationStateError):
        resolve_operation_locally(
            db_path=operations_db,
            idempotency_key="idem-1",
            outcome="succeeded",
            reason="Changed mind",
        )


def test_local_resolution_accepts_stale_in_progress_after_conservative_recovery(
    operations_db,
):
    clock = FakeClock(datetime(2026, 9, 25, 10, 0, tzinfo=UTC))
    _reserve(operations_db, clock=clock)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="begin",
        clock=clock,
    )
    clock.advance(seconds=DEFAULT_STALE_AFTER_SECONDS + 1)

    resolved = resolve_operation_locally(
        db_path=operations_db,
        idempotency_key="idem-1",
        outcome="failed",
        reason="Help Scout showed no matching reply",
        clock=clock,
    )

    assert resolved.state == "failed"
    assert resolved.error["code"] == "manually_resolved_failed"
    with sqlite3.connect(operations_db) as connection:
        audits = [
            (row[0], json.loads(row[1]))
            for row in connection.execute(
                "SELECT result, metadata_json FROM agent_audit ORDER BY id"
            )
        ]
    assert [row[0] for row in audits[-2:]] == ["outcome_unknown", "resolved_failed"]
    assert audits[-2][1]["source"] == "local_resolution_recovery"


def test_management_script_show_and_resolve_use_configured_local_database(
    operations_db, monkeypatch, capsys
):
    from scripts import manage_agent_operations

    monkeypatch.setenv("SUPA_DB_PATH", str(operations_db))
    _reserve(operations_db)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="begin",
    )
    mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="submit",
    )
    finish_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="unknown",
        outcome="outcome_unknown",
    )

    assert manage_agent_operations.main(["show", "idem-1"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["idempotency_key"] == "idem-1"
    assert shown["state"] == "outcome_unknown"
    assert "token_id" not in shown

    assert manage_agent_operations.main(
        [
            "resolve",
            "idem-1",
            "--as",
            "failed",
            "--reason",
            "No upstream write was observed",
        ]
    ) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["state"] == "failed"


def test_management_script_does_not_create_a_mistyped_database(
    tmp_path, monkeypatch
):
    from scripts import manage_agent_operations

    missing = tmp_path / "mistyped.db"
    monkeypatch.setenv("SUPA_DB_PATH", str(missing))

    with pytest.raises(SystemExit) as raised:
        manage_agent_operations.main(["show", "no-such-key"])

    assert raised.value.code == 1
    assert not missing.exists()


def test_management_script_imports_from_repo_when_invoked_through_symlink(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "manage_agent_operations.py"
    symlink = tmp_path / "manage-agent-operations"
    symlink.symlink_to(script)

    completed = subprocess.run(
        [sys.executable, str(symlink), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "show" in completed.stdout
    assert "resolve" in completed.stdout


def test_management_script_rejects_symlinked_or_writable_database_paths(owner_controlled_directory):
    tmp_path = owner_controlled_directory
    from scripts import manage_agent_operations

    secure_dir = tmp_path / "secure"
    secure_dir.mkdir(mode=0o700)
    secure_db = secure_dir / "operations.db"
    init_db(str(secure_db))

    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(secure_dir, target_is_directory=True)
    with pytest.raises(ValueError):
        manage_agent_operations._existing_local_db(linked_dir / "operations.db")

    os.chmod(secure_db, 0o666)
    with pytest.raises(ValueError):
        manage_agent_operations._existing_local_db(secure_db)
    os.chmod(secure_db, 0o600)

    os.chmod(secure_dir, 0o777)
    with pytest.raises(ValueError):
        manage_agent_operations._existing_local_db(secure_db)


def test_management_script_detects_database_replacement_during_use(
    operations_db, tmp_path, monkeypatch
):
    from scripts import manage_agent_operations

    replacement = operations_db.with_name("replacement.db")
    init_db(str(replacement))
    operation = _reserve(operations_db).operation
    monkeypatch.setenv("SUPA_DB_PATH", str(operations_db))

    def replace_during_read(**kwargs):
        os.replace(replacement, operations_db)
        return operation

    monkeypatch.setattr(
        manage_agent_operations,
        "get_operation_locally",
        replace_during_read,
    )

    with pytest.raises(SystemExit) as raised:
        manage_agent_operations.main(["show", "idem-1"])

    assert raised.value.code == 1


def test_state_transition_rolls_back_atomically_when_audit_insert_fails(operations_db):
    _reserve(operations_db)
    with sqlite3.connect(operations_db) as connection:
        connection.executescript(
            """
            CREATE TRIGGER fail_in_progress_audit
            BEFORE INSERT ON agent_audit
            WHEN NEW.result = 'in_progress'
            BEGIN
                SELECT RAISE(ABORT, 'forced audit failure');
            END;
            """
        )

    with pytest.raises(AgentOperationStorageError):
        begin_operation(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
            scope="draft",
            request_id="begin",
        )

    with sqlite3.connect(operations_db) as connection:
        state = connection.execute(
            "SELECT state FROM agent_operations WHERE idempotency_key = 'idem-1'"
        ).fetchone()[0]
        results = [
            row[0]
            for row in connection.execute(
                "SELECT result FROM agent_audit ORDER BY id"
            )
        ]
    assert state == "reserved"
    assert results == ["reserved"]


def test_finish_validates_existing_stored_json_before_overwriting(operations_db):
    _reserve(operations_db)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="begin",
    )
    mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="submit",
    )
    with sqlite3.connect(operations_db) as connection:
        connection.execute(
            "UPDATE agent_operations SET response_json = 'not-json' "
            "WHERE idempotency_key = 'idem-1'"
        )
        original_audits = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]
        connection.commit()

    with pytest.raises(AgentOperationDataError):
        finish_operation(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
            scope="send",
            request_id="finish",
            outcome="succeeded",
            upstream_thread_id="999",
            response={"thread_id": "999", "status": "sent"},
        )

    with sqlite3.connect(operations_db) as connection:
        row = connection.execute(
            "SELECT state, response_json FROM agent_operations "
            "WHERE idempotency_key = 'idem-1'"
        ).fetchone()
        audit_count = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]
    assert row == ("in_progress", "not-json")
    assert audit_count == original_audits


def test_local_resolution_validates_existing_json_before_overwriting(operations_db):
    _reserve(operations_db)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="begin",
    )
    mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="submit",
    )
    finish_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="unknown",
        outcome="outcome_unknown",
    )
    with sqlite3.connect(operations_db) as connection:
        connection.execute(
            "UPDATE agent_operations SET response_json = 'not-json' "
            "WHERE idempotency_key = 'idem-1'"
        )
        original_audits = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]
        connection.commit()

    with pytest.raises(AgentOperationDataError):
        resolve_operation_locally(
            db_path=operations_db,
            idempotency_key="idem-1",
            outcome="failed",
            reason="No matching upstream reply",
        )

    with sqlite3.connect(operations_db) as connection:
        row = connection.execute(
            "SELECT state, response_json FROM agent_operations "
            "WHERE idempotency_key = 'idem-1'"
        ).fetchone()
        audit_count = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]
    assert row == ("outcome_unknown", "not-json")
    assert audit_count == original_audits


def test_embedded_supa_token_is_rejected_before_audit_persistence(operations_db):
    token_like = "T" * 43

    with pytest.raises(ValueError):
        reserve_operation(
            db_path=operations_db,
            idempotency_key="idem-token-test",
            identity=IDENTITY,
            scope="draft",
            operation_name="draft.save",
            target_id="602",
            request_id=f"prefix_{token_like}_suffix",
            request={"body": "Hello", "body_format": "text"},
            request_metadata={"body_sha256": "a" * 64},
        )

    with sqlite3.connect(operations_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM agent_operations"
        ).fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM agent_audit").fetchone()[0] == 0


def test_local_resolution_rejects_concatenated_token_without_state_or_audit_change(
    operations_db,
):
    _reserve(operations_db)
    begin_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="begin",
    )
    mark_upstream_started(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="submit",
    )
    finish_operation(
        db_path=operations_db,
        idempotency_key="idem-1",
        identity=IDENTITY,
        scope="send",
        request_id="unknown",
        outcome="outcome_unknown",
    )
    with sqlite3.connect(operations_db) as connection:
        original_audit_count = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]

    token_like = "U" * 43
    with pytest.raises(ValueError):
        resolve_operation_locally(
            db_path=operations_db,
            idempotency_key="idem-1",
            outcome="failed",
            reason=f"prefix_{token_like}_suffix",
        )

    with sqlite3.connect(operations_db) as connection:
        state = connection.execute(
            "SELECT state FROM agent_operations WHERE idempotency_key = 'idem-1'"
        ).fetchone()[0]
        audit_count = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]
    assert state == "outcome_unknown"
    assert audit_count == original_audit_count
    assert token_like not in _all_audit_text(operations_db)


def test_idempotency_key_surrogate_is_plain_value_error_without_persistence(
    operations_db,
):
    with pytest.raises(ValueError) as raised:
        reserve_operation(
            db_path=operations_db,
            idempotency_key="idem-\ud800",
            identity=IDENTITY,
            scope="draft",
            operation_name="draft.save",
            target_id="602",
            request_id="request-surrogate-key",
            request={"body": "Hello", "body_format": "text"},
        )

    assert type(raised.value) is ValueError
    with sqlite3.connect(operations_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM agent_operations"
        ).fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM agent_audit").fetchone()[0] == 0


def test_request_id_surrogate_rolls_back_transition_and_audit(operations_db):
    _reserve(operations_db)
    with sqlite3.connect(operations_db) as connection:
        original_audit_count = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]

    with pytest.raises(ValueError) as raised:
        begin_operation(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
            scope="draft",
            request_id="request-\ud800",
        )

    assert type(raised.value) is ValueError
    with sqlite3.connect(operations_db) as connection:
        state = connection.execute(
            "SELECT state FROM agent_operations WHERE idempotency_key = 'idem-1'"
        ).fetchone()[0]
        audit_count = connection.execute(
            "SELECT COUNT(*) FROM agent_audit"
        ).fetchone()[0]
    assert state == "reserved"
    assert audit_count == original_audit_count


def test_malformed_stored_json_is_a_stable_non_retryable_data_error(operations_db):
    _reserve(operations_db)
    with sqlite3.connect(operations_db) as connection:
        connection.execute(
            "UPDATE agent_operations SET response_json = 'not-json' "
            "WHERE idempotency_key = 'idem-1'"
        )
        connection.commit()

    with pytest.raises(AgentOperationDataError) as raised:
        get_operation(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
        )

    assert raised.value.code == "operation_data_error"
    assert raised.value.retryable is False


def test_stored_json_with_oversized_integer_is_a_stable_data_error(operations_db):
    _reserve(operations_db)
    oversized_integer_json = '{"version":' + ("9" * 5_000) + "}"
    with sqlite3.connect(operations_db) as connection:
        connection.execute(
            "UPDATE agent_operations SET response_json = ? "
            "WHERE idempotency_key = 'idem-1'",
            (oversized_integer_json,),
        )
        connection.commit()

    with pytest.raises(AgentOperationDataError) as raised:
        get_operation(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
        )

    assert raised.value.code == "operation_data_error"
    assert raised.value.retryable is False


def test_stored_json_with_invalid_unicode_is_a_stable_data_error(operations_db):
    _reserve(operations_db)
    with sqlite3.connect(operations_db) as connection:
        connection.execute(
            "UPDATE agent_operations SET response_json = ? "
            "WHERE idempotency_key = 'idem-1'",
            ('{"code":"\\ud800"}',),
        )
        connection.commit()

    with pytest.raises(AgentOperationDataError):
        get_operation(
            db_path=operations_db,
            idempotency_key="idem-1",
            identity=IDENTITY,
        )


def test_partial_operation_schema_is_a_stable_non_retryable_data_error(tmp_path):
    path = tmp_path / "partial-schema.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE agent_operations (idempotency_key TEXT PRIMARY KEY)"
        )
        connection.execute(
            "INSERT INTO agent_operations (idempotency_key) VALUES ('idem-1')"
        )

    with pytest.raises(AgentOperationDataError) as raised:
        get_operation(
            db_path=path,
            idempotency_key="idem-1",
            identity=IDENTITY,
        )

    assert raised.value.code == "operation_data_error"
    assert raised.value.retryable is False


def test_sqlite_schema_failure_remains_a_retryable_storage_error(tmp_path):
    path = tmp_path / "wrong-schema.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")

    with pytest.raises(AgentOperationStorageError) as raised:
        get_operation(
            db_path=path,
            idempotency_key="idem-1",
            identity=IDENTITY,
        )

    assert raised.value.code == "operation_storage_error"
    assert raised.value.retryable is True
