from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sqlite3
import threading

import pytest

import services.conversation_gateway as gateway_module
from db.database import init_db
from services.agent_auth import AgentIdentity
from services.agent_events import AgentEventStorageError
from services.agent_operations import AgentOperationNotFoundError, get_operation
from services.conversation_gateway import (
    ConversationGateway,
    ConversationGatewayError,
    draft_version,
    normalize_draft_body,
)
from services.helpscout import (
    HelpScoutAuthError,
    HelpScoutProtocolError,
    HelpScoutRedirectError,
    HelpScoutTransportError,
    HelpScoutUpstreamError,
    MutationResult,
)
from services.conversation_locks import (
    ConversationBusyError,
    ConversationLockLostError,
    ConversationLockStorageError,
)


IDENTITY = AgentIdentity(
    token_id="writer-1",
    name="Draft writer",
    scopes=frozenset({"read", "draft"}),
)


def _draft(thread_id: int, body: str, created_at: str) -> dict:
    return {
        "id": thread_id,
        "type": "message",
        "state": "draft",
        "draft": True,
        "body": body,
        "createdAt": created_at,
        "createdBy": {"type": "user", "id": 9},
        "source": {"type": "api"},
    }


def _customer(thread_id: int, body: str, created_at: str) -> dict:
    return {
        "id": thread_id,
        "type": "customer",
        "state": "published",
        "body": body,
        "createdAt": created_at,
        "createdBy": {"type": "customer", "id": 44},
        "source": {"type": "email"},
    }


def _published_reply(thread_id: int, body: str, created_at: str) -> dict:
    return {
        "id": thread_id,
        "type": "message",
        "state": "published",
        "draft": False,
        "body": body,
        "createdAt": created_at,
        "createdBy": {"type": "user", "id": 9},
        "source": {"type": "api"},
    }


def _conversation(
    conversation_id: int = 321,
    *,
    requested_id: int | None = None,
    drafts: list[dict] | None = None,
    threads: list[dict] | None = None,
) -> dict:
    return {
        "id": conversation_id,
        "requested_conversation_id": requested_id or conversation_id,
        "subject": "Support question",
        "status": "active",
        "tags": [{"tag": "field-journal"}, {"tag": "supa-manual-review"}],
        "primaryCustomer": {
            "id": 44,
            "type": "customer",
            "first": "Ada",
            "email": "ada@example.test",
        },
        "_embedded": {"threads": list(threads or []) + list(drafts or [])},
    }


class FakeDraftClient:
    def __init__(self, conversation: dict):
        self.conversation = copy.deepcopy(conversation)
        self.calls: list[tuple] = []
        self.create_result = MutationResult(True, 201, resource_id=901)
        self.update_result = MutationResult(True, 204)
        self.send_result = MutationResult(True, 201, resource_id=990)
        self.delete_result = MutationResult(True, 204)
        self.tags_result = MutationResult(True, 204)
        self.apply_create = True
        self.apply_update = True
        self.apply_send = True
        self.created_thread_id: int | None = None
        self.sent_thread_id: int | None = None
        self.created_at = "2026-09-25T12:00:00Z"
        self.updated_at = "2026-09-25T12:00:00Z"
        self.sent_at = "2099-01-01T00:00:00Z"
        self.status_verified = True
        self.additional_created_threads: list[dict] = []
        self.additional_sent_threads: list[dict] = []
        self.sent_body_override: str | None = None
        self.sent_source_type = "api"
        self.send_verify_status_flags: list[bool] = []
        self.raise_on_create: BaseException | None = None
        self.raise_on_send: BaseException | None = None
        self.get_error_at: int | None = None
        self.get_exception: Exception = RuntimeError(
            "read failed with private body"
        )
        self._get_count = 0
        self.before_get = None
        self.before_submit = None
        # How Help Scout stores a body; the real API re-serializes HTML (2026-10-02).
        self.store = lambda text: text

    async def get_complete_conversation(self, conversation_id: int) -> dict:
        self._get_count += 1
        self.calls.append(("get", conversation_id))
        if self.get_error_at == self._get_count:
            raise self.get_exception
        if self.before_get:
            self.before_get(self._get_count)
        return copy.deepcopy(self.conversation)

    async def add_draft_result(
        self, conversation_id: int, text: str, customer_id: int | None = None
    ) -> MutationResult:
        self.calls.append(("create", conversation_id, text, customer_id))
        if self.before_submit:
            self.before_submit()
        if self.raise_on_create is not None:
            raise self.raise_on_create
        if self.apply_create:
            resource_id = (
                self.created_thread_id or self.create_result.resource_id or 901
            )
            self.conversation["_embedded"]["threads"].append(
                _draft(resource_id, self.store(text), self.created_at)
            )
            self.conversation["_embedded"]["threads"].extend(
                copy.deepcopy(self.additional_created_threads)
            )
        return self.create_result

    async def update_draft_result(
        self, conversation_id: int, thread_id: int, text: str
    ) -> MutationResult:
        self.calls.append(("update", conversation_id, thread_id, text))
        if self.before_submit:
            self.before_submit()
        if self.apply_update:
            for thread in self.conversation["_embedded"]["threads"]:
                if thread["id"] == thread_id:
                    thread["body"] = self.store(text)
                    thread["updatedAt"] = self.updated_at
        return self.update_result

    async def send_reply_result(
        self,
        conversation_id: int,
        text: str,
        status: str = "pending",
        customer_id: int | None = None,
        verify_status: bool = True,
    ) -> MutationResult:
        self.send_verify_status_flags.append(verify_status)
        self.calls.append(("send", conversation_id, text, status, customer_id))
        if self.before_submit:
            self.before_submit()
        if self.raise_on_send is not None:
            raise self.raise_on_send
        if self.apply_send:
            thread_id = self.sent_thread_id or self.send_result.resource_id or 990
            reply = _published_reply(
                thread_id,
                self.sent_body_override or self.store(text),
                self.sent_at,
            )
            reply["source"] = {"type": self.sent_source_type}
            self.conversation["_embedded"]["threads"].append(reply)
            self.conversation["_embedded"]["threads"].extend(
                copy.deepcopy(self.additional_sent_threads)
            )
        return self.send_result

    async def delete_draft_result(
        self, conversation_id: int, thread_id: int
    ) -> MutationResult:
        self.calls.append(("delete", conversation_id, thread_id))
        if self.delete_result.ok:
            for thread in self.conversation["_embedded"]["threads"]:
                if thread.get("id") == thread_id:
                    thread["body"] = "<p>Manual review required before sending a reply.</p>"
        return self.delete_result

    async def ensure_conversation_status(
        self, conversation_id: int, status: str
    ) -> dict:
        self.calls.append(("ensure_status", conversation_id, status))
        if self.status_verified:
            self.conversation["status"] = status
        return {
            "verified": self.status_verified,
            "conversation": copy.deepcopy(self.conversation),
        }

    async def add_tags_result(
        self,
        conversation_id: int,
        tags: list[str],
        existing_tags: list[str] | None = None,
    ) -> MutationResult:
        self.calls.append(
            ("tags", conversation_id, tuple(tags), tuple(existing_tags or []))
        )
        if self.tags_result.ok:
            self.conversation["tags"] = [
                {"tag": tag} for tag in dict.fromkeys((existing_tags or []) + tags)
            ]
        return self.tags_result


@pytest.fixture
def writes_db(tmp_path):
    path = tmp_path / "writes.db"
    init_db(str(path))
    return path


def _gateway(client: FakeDraftClient, writes_db) -> ConversationGateway:
    return ConversationGateway(client, db_path=writes_db)


async def _save(
    gateway: ConversationGateway,
    *,
    key: str = "draft-save-1",
    conversation_id: int = 321,
    body: str = "Hello\n\nWorld",
    body_format: str = "text",
    expected_version: str | None = None,
):
    return await gateway.save_draft(
        conversation_id,
        body=body,
        body_format=body_format,
        expected_version=expected_version,
        idempotency_key=key,
        identity=IDENTITY,
        request_id=f"request-{key}",
    )


class FakeSendBookkeeper:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.error: Exception | None = None

    async def __call__(self, **kwargs) -> None:
        self.calls.append(copy.deepcopy(kwargs))
        if self.error is not None:
            raise self.error


async def _send_draft(
    gateway: ConversationGateway,
    *,
    expected_version: str,
    expected_latest_thread_id: int = 500,
    key: str = "draft-send-1",
    conversation_id: int = 321,
):
    return await gateway.send_draft(
        conversation_id,
        expected_version=expected_version,
        expected_latest_thread_id=expected_latest_thread_id,
        idempotency_key=key,
        identity=IDENTITY,
        request_id=f"request-{key}",
    )


async def _send_reply(
    gateway: ConversationGateway,
    *,
    body: str = "Hello\n\nWorld",
    body_format: str = "text",
    expected_latest_thread_id: int = 500,
    key: str = "reply-send-1",
    conversation_id: int = 321,
):
    return await gateway.send_reply(
        conversation_id,
        body=body,
        body_format=body_format,
        expected_latest_thread_id=expected_latest_thread_id,
        idempotency_key=key,
        identity=IDENTITY,
        request_id=f"request-{key}",
    )


def _audit_text(path) -> str:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT result, metadata_json FROM agent_audit ORDER BY id"
        ).fetchall()
    return "\n".join(str(value) for row in rows for value in row)


def _stored_supa_events(path) -> list[dict]:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT event_type, conversation_id, thread_id, source, summary_json
            FROM agent_events
            WHERE source = 'supa'
            ORDER BY id
            """
        ).fetchall()
    return [
        {
            "type": row["event_type"],
            "conversation_id": row["conversation_id"],
            "thread_id": row["thread_id"],
            "source": row["source"],
            "summary": json.loads(row["summary_json"]),
        }
        for row in rows
    ]


@pytest.mark.asyncio
async def test_create_saves_exact_normalized_html_and_durable_safe_result(writes_db):
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)
    raw_body = 'Hello <private> sk-secret-value\n\n<a href="javascript:bad">World</a>'
    expected_html = normalize_draft_body(raw_body, "text")

    def assert_marker_precedes_submission():
        operation = get_operation(
            db_path=writes_db,
            idempotency_key="draft-save-1",
            identity=IDENTITY,
        )
        assert operation.state == "in_progress"
        assert operation.upstream_started_at is not None
        assert operation.request_metadata["action"] == "created"
        assert operation.request_metadata["conversation_id"] == "321"
        assert operation.request_metadata["expected_thread_id"] is None
        assert operation.request_metadata["normalized_body_hash"] == hashlib.sha256(
            expected_html.encode()
        ).hexdigest()

    client.before_submit = assert_marker_precedes_submission
    result = await _save(gateway, body=raw_body)

    assert result == {
        "action": "created",
        "conversation_id": 321,
        "thread_id": 901,
        "body_sha256": hashlib.sha256(expected_html.encode()).hexdigest(),
        "version": draft_version(client.conversation["_embedded"]["threads"][-1]),
    }
    assert [call[0] for call in client.calls] == ["get", "create", "get"]
    assert client.calls[1][2] == expected_html
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "succeeded"
    assert operation.response == {
        **result,
        "conversation_id": "321",
        "thread_id": "901",
    }
    persisted = (
        _audit_text(writes_db)
        + repr(operation.request_metadata)
        + repr(operation.response)
    )
    assert raw_body not in persisted
    assert "sk-secret-value" not in persisted


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["created", "updated"])
async def test_verified_draft_save_emits_one_compact_replay_safe_event(
    writes_db,
    action,
):
    raw_body = "Private draft body sk-secret-value"
    draft = _draft(701, "<p>Old draft</p>", "2026-09-25T11:00:00Z")
    client = FakeDraftClient(
        _conversation(drafts=[draft] if action == "updated" else [])
    )
    gateway = _gateway(client, writes_db)
    expected_version = draft_version(draft) if action == "updated" else None

    first = await _save(
        gateway,
        key=f"draft-event-{action}",
        body=raw_body,
        expected_version=expected_version,
    )
    replay = await _save(
        gateway,
        key=f"draft-event-{action}",
        body=raw_body,
        expected_version=expected_version,
    )

    assert replay == first
    assert _stored_supa_events(writes_db) == [
        {
            "type": "supa.draft.saved",
            "conversation_id": "321",
            "thread_id": str(first["thread_id"]),
            "source": "supa",
            "summary": {
                "operation_id": f"draft-event-{action}",
                "operation_name": "draft.save",
                "request_id": f"request-draft-event-{action}",
                "result": action,
                "status": "saved",
                "token_id": "writer-1",
                "token_name": "Draft writer",
            },
        }
    ]
    persisted = json.dumps(_stored_supa_events(writes_db), sort_keys=True)
    assert raw_body not in persisted
    assert "sk-secret-value" not in persisted


@pytest.mark.asyncio
async def test_terminal_operation_replay_heals_a_failed_event_append(
    writes_db,
    monkeypatch,
):
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)
    original_append = gateway_module.append_supa_event
    attempts = 0

    def flaky_append(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise AgentEventStorageError("event store unavailable")
        return original_append(**kwargs)

    monkeypatch.setattr(gateway_module, "append_supa_event", flaky_append)

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(gateway, key="draft-event-recovery")
    assert raised.value.code == "event_storage_error"
    assert get_operation(
        db_path=writes_db,
        idempotency_key="draft-event-recovery",
        identity=IDENTITY,
    ).state == "succeeded"
    assert _stored_supa_events(writes_db) == []
    calls = list(client.calls)

    replay = await _save(gateway, key="draft-event-recovery")

    assert replay["action"] == "created"
    assert client.calls == calls
    assert attempts == 2
    assert len(_stored_supa_events(writes_db)) == 1


@pytest.mark.asyncio
async def test_operation_stays_reserved_through_preflight_then_marks_before_io(writes_db):
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)
    observed_states: list[tuple[int, str, bool]] = []

    def record_state(read_number: int):
        operation = get_operation(
            db_path=writes_db,
            idempotency_key="draft-save-1",
            identity=IDENTITY,
        )
        observed_states.append(
            (read_number, operation.state, operation.upstream_started_at is not None)
        )

    client.before_get = record_state
    await _save(gateway)

    assert observed_states == [
        (1, "reserved", False),
        (2, "in_progress", True),
    ]


@pytest.mark.asyncio
async def test_first_upstream_read_occurs_only_after_requested_lock_is_held(
    writes_db, monkeypatch
):
    active_locks: set[int] = set()
    entered_locks: list[int] = []

    class RecordingLease:
        def __init__(self, conversation_id):
            self.conversation_id = int(conversation_id)

        async def __aenter__(self):
            active_locks.add(self.conversation_id)
            entered_locks.append(self.conversation_id)
            return self

        async def __aexit__(self, *args):
            active_locks.remove(self.conversation_id)
            return False

        async def verify_owned(self):
            assert self.conversation_id in active_locks

        async def run_guarded(self, callback):
            await self.verify_owned()
            return await callback()

    monkeypatch.setattr(
        gateway_module,
        "conversation_lease",
        lambda _path, conversation_id, **kwargs: RecordingLease(conversation_id),
    )
    client = FakeDraftClient(_conversation())

    def assert_locked(_read_number):
        assert active_locks == {321}

    client.before_get = assert_locked
    await _save(_gateway(client, writes_db))

    assert entered_locks == [321]


@pytest.mark.asyncio
async def test_version_conflict_does_not_run_body_sanitizer(writes_db, monkeypatch):
    live = _draft(701, "<p>new</p>", "2026-09-21T10:00:00Z")
    client = FakeDraftClient(_conversation(drafts=[live]))
    sanitizer_calls: list[tuple[object, object]] = []

    def reject_sanitizer(body, body_format):
        sanitizer_calls.append((body, body_format))
        raise AssertionError("sanitizer must not run before version check")

    monkeypatch.setattr(
        gateway_module,
        "normalize_draft_body",
        reject_sanitizer,
    )

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(_gateway(client, writes_db), expected_version="a" * 64)

    assert raised.value.code == "draft_conflict"
    assert sanitizer_calls == []


@pytest.mark.asyncio
async def test_alias_reacquires_requested_and_canonical_locks_in_order(
    writes_db, monkeypatch
):
    active_locks: set[int] = set()
    events: list[tuple[str, int]] = []

    class RecordingLease:
        def __init__(self, conversation_id):
            self.conversation_id = int(conversation_id)

        async def __aenter__(self):
            active_locks.add(self.conversation_id)
            events.append(("enter", self.conversation_id))
            return self

        async def __aexit__(self, *args):
            events.append(("exit", self.conversation_id))
            active_locks.remove(self.conversation_id)
            return False

        async def verify_owned(self):
            assert self.conversation_id in active_locks

        async def run_guarded(self, callback):
            await self.verify_owned()
            return await callback()

    monkeypatch.setattr(
        gateway_module,
        "conversation_lease",
        lambda _path, conversation_id, **kwargs: RecordingLease(conversation_id),
    )
    client = FakeDraftClient(_conversation(321, requested_id=123))

    def assert_expected_lock_set(read_number):
        if read_number == 1:
            assert active_locks == {123}
        else:
            assert active_locks == {123, 321}

    client.before_get = assert_expected_lock_set
    result = await _save(
        _gateway(client, writes_db),
        key="alias-lock-order",
        conversation_id=123,
    )

    assert result["conversation_id"] == 321
    assert events == [
        ("enter", 123),
        ("exit", 123),
        ("enter", 123),
        ("enter", 321),
        ("exit", 321),
        ("exit", 123),
    ]


@pytest.mark.asyncio
async def test_interruption_during_initial_preflight_observes_reserved_state(writes_db):
    class BlockingReadClient(FakeDraftClient):
        def __init__(self, conversation):
            super().__init__(conversation)
            self.read_started = asyncio.Event()

        async def get_complete_conversation(self, conversation_id: int) -> dict:
            self.calls.append(("get", conversation_id))
            self.read_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    client = BlockingReadClient(_conversation())
    gateway = _gateway(client, writes_db)
    task = asyncio.create_task(_save(gateway))
    await client.read_started.wait()

    during_preflight = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert during_preflight.state == "reserved"
    assert during_preflight.upstream_started_at is None

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    after_cancel = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert after_cancel.state == "failed"
    assert after_cancel.error["code"] == "cancelled_before_attempt"


@pytest.mark.asyncio
async def test_same_key_same_request_replays_without_any_upstream_io(writes_db):
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)
    first = await _save(gateway)
    calls_after_first = list(client.calls)

    second = await _save(gateway)

    assert second == first
    assert client.calls == calls_after_first


@pytest.mark.asyncio
async def test_replayed_reserved_or_in_progress_operation_never_writes(writes_db):
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)
    original_begin = gateway._operation_begin

    async def pause_after_reservation(**kwargs):
        entered.set()
        await release.wait()
        return await original_begin(**kwargs)

    entered = asyncio.Event()
    release = asyncio.Event()
    gateway._operation_begin = pause_after_reservation
    first = asyncio.create_task(_save(gateway))
    await entered.wait()

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(gateway)
    assert raised.value.code == "operation_in_progress"
    assert not [call for call in client.calls if call[0] in {"create", "update"}]

    release.set()
    await first
    assert len([call for call in client.calls if call[0] == "create"]) == 1


@pytest.mark.asyncio
async def test_updates_exact_newest_live_draft_and_rejects_stale_version(writes_db):
    older = _draft(700, "<p>old</p>", "2026-09-20T10:00:00Z")
    newest = _draft(701, "<p>new</p>", "2026-09-21T10:00:00Z")
    client = FakeDraftClient(_conversation(drafts=[newest, older]))
    gateway = _gateway(client, writes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(gateway, key="stale", expected_version=draft_version(older))
    assert raised.value.code == "draft_conflict"
    assert not [call for call in client.calls if call[0] == "update"]
    rejected = get_operation(
        db_path=writes_db, idempotency_key="stale", identity=IDENTITY
    )
    assert rejected.state == "failed"
    assert rejected.upstream_started_at is None

    current_version = draft_version(newest)
    result = await _save(gateway, key="current", expected_version=current_version)
    update = [call for call in client.calls if call[0] == "update"]
    assert update == [("update", 321, 701, "<p>Hello</p><p>&nbsp;</p><p>World</p>")]
    assert result["action"] == "updated"
    assert result["thread_id"] == 701
    attempt = get_operation(
        db_path=writes_db,
        idempotency_key="current",
        identity=IDENTITY,
    ).request_metadata
    assert attempt["action"] == "updated"
    assert attempt["conversation_id"] == "321"
    assert attempt["expected_thread_id"] == "701"
    assert attempt["normalized_body_hash"] == result["body_sha256"]


@pytest.mark.asyncio
async def test_create_only_conflict_is_terminal_before_upstream_submission(writes_db):
    live = _draft(701, "<p>new</p>", "2026-09-21T10:00:00Z")
    client = FakeDraftClient(_conversation(drafts=[live]))

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(_gateway(client, writes_db))

    assert raised.value.code == "draft_conflict"
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.upstream_started_at is None


@pytest.mark.asyncio
async def test_distinct_key_concurrent_creates_are_serialized(writes_db):
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)

    results = await asyncio.gather(
        _save(gateway, key="create-a"),
        _save(gateway, key="create-b"),
        return_exceptions=True,
    )

    assert len([result for result in results if isinstance(result, dict)]) == 1
    conflicts = [
        result
        for result in results
        if isinstance(result, ConversationGatewayError)
        and result.code == "draft_conflict"
    ]
    assert len(conflicts) == 1
    assert len([call for call in client.calls if call[0] == "create"]) == 1


@pytest.mark.asyncio
async def test_alias_and_canonical_ids_serialize_on_resolved_conversation(writes_db):
    client = FakeDraftClient(_conversation(321, requested_id=123))
    gateway = _gateway(client, writes_db)

    results = await asyncio.gather(
        _save(gateway, key="alias", conversation_id=123),
        _save(gateway, key="canonical", conversation_id=321),
        return_exceptions=True,
    )

    assert len([result for result in results if isinstance(result, dict)]) == 1
    assert len([call for call in client.calls if call[0] == "create"]) == 1
    create_call = next(call for call in client.calls if call[0] == "create")
    assert create_call[1] == 321


@pytest.mark.asyncio
async def test_unknown_create_reconciles_only_exact_unambiguous_new_draft(writes_db):
    client = FakeDraftClient(_conversation())
    client.create_result = MutationResult(
        False, None, outcome_unknown=True, detail="network_error"
    )
    client.created_at = "2099-01-01T00:00:00Z"
    gateway = _gateway(client, writes_db)

    result = await _save(gateway)

    assert result["action"] == "created"
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "succeeded"


@pytest.mark.asyncio
async def test_unknown_update_reconciles_exact_selected_thread(writes_db):
    live = _draft(701, "<p>old</p>", "2026-09-21T10:00:00Z")
    client = FakeDraftClient(_conversation(drafts=[live]))
    client.update_result = MutationResult(
        False, None, outcome_unknown=True, detail="network_error"
    )
    client.updated_at = "2099-01-01T00:00:00Z"
    gateway = _gateway(client, writes_db)

    result = await _save(
        gateway,
        expected_version=draft_version(live),
    )

    assert result["action"] == "updated"
    assert result["thread_id"] == 701
    assert get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    ).state == "succeeded"


@pytest.mark.asyncio
async def test_success_without_resource_id_uses_one_unambiguous_new_draft(writes_db):
    client = FakeDraftClient(_conversation())
    client.create_result = MutationResult(True, 201, resource_id=None)
    client.created_at = "2099-01-01T00:00:00Z"

    result = await _save(_gateway(client, writes_db))

    assert result["thread_id"] == 901
    assert result["action"] == "created"


@pytest.mark.asyncio
async def test_success_without_resource_id_does_not_claim_a_pre_attempt_candidate(
    writes_db,
):
    client = FakeDraftClient(_conversation())
    client.create_result = MutationResult(True, 201, resource_id=None)
    client.created_at = "2000-01-01T00:00:00Z"

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(_gateway(client, writes_db))

    assert raised.value.code == "outcome_unknown"
    assert get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    ).state == "outcome_unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_count", [0, 2])
async def test_success_without_resource_id_stays_unknown_when_identity_is_ambiguous(
    writes_db, candidate_count
):
    intended = normalize_draft_body("Hello\n\nWorld", "text")
    client = FakeDraftClient(_conversation())
    client.create_result = MutationResult(
        True, 201, resource_id=None, detail="resource_id_missing"
    )
    if candidate_count == 0:
        client.apply_create = False
    else:
        client.additional_created_threads = [
            _draft(902, intended, "2026-09-25T12:00:01Z")
        ]

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(_gateway(client, writes_db))

    assert raised.value.code == "outcome_unknown"
    assert get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    ).state == "outcome_unknown"


@pytest.mark.asyncio
async def test_unknown_create_with_multiple_exact_candidates_stays_unknown(writes_db):
    intended = normalize_draft_body("Hello\n\nWorld", "text")
    client = FakeDraftClient(_conversation())
    client.create_result = MutationResult(
        False, None, outcome_unknown=True, detail="network_error"
    )
    client.additional_created_threads = [
        _draft(902, intended, "2026-09-25T12:00:01Z")
    ]
    gateway = _gateway(client, writes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(gateway)

    assert raised.value.code == "outcome_unknown"
    assert get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    ).state == "outcome_unknown"


@pytest.mark.asyncio
async def test_unreconciled_unknown_is_durable_and_never_replayed(writes_db):
    client = FakeDraftClient(_conversation())
    client.apply_create = False
    client.create_result = MutationResult(
        False, None, outcome_unknown=True, detail="network_error"
    )
    gateway = _gateway(client, writes_db)

    with pytest.raises(ConversationGatewayError) as first:
        await _save(gateway)
    assert first.value.code == "outcome_unknown"
    calls = list(client.calls)

    with pytest.raises(ConversationGatewayError) as replay:
        await _save(gateway)
    assert replay.value.code == "outcome_unknown"
    assert client.calls == calls
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "outcome_unknown"
    assert operation.request_metadata["normalized_body_hash"] == hashlib.sha256(
        normalize_draft_body("Hello\n\nWorld", "text").encode()
    ).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["created", "updated"])
async def test_uncertain_write_does_not_reconcile_a_matching_thread_from_before_attempt(
    writes_db, action
):
    if action == "created":
        client = FakeDraftClient(_conversation())
        expected_version = None
        client.create_result = MutationResult(False, None, outcome_unknown=True)
        client.created_at = "2000-01-01T00:00:00Z"
    else:
        live = _draft(701, "<p>old</p>", "2000-01-01T00:00:00Z")
        client = FakeDraftClient(_conversation(drafts=[live]))
        expected_version = draft_version(live)
        client.update_result = MutationResult(False, None, outcome_unknown=True)
        client.updated_at = "2000-01-01T00:00:00Z"

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(
            _gateway(client, writes_db),
            expected_version=expected_version,
        )

    assert raised.value.code == "outcome_unknown"
    assert get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    ).state == "outcome_unknown"


@pytest.mark.asyncio
async def test_malformed_post_write_state_is_unknown_not_external_conflict(writes_db):
    class MalformedPostWriteClient(FakeDraftClient):
        async def get_complete_conversation(self, conversation_id: int) -> dict:
            result = await super().get_complete_conversation(conversation_id)
            if self._get_count == 2:
                return {"id": 321, "_embedded": {"threads": "private malformed"}}
            return result

    client = MalformedPostWriteClient(_conversation())

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(_gateway(client, writes_db))

    assert raised.value.code == "outcome_unknown"
    assert get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    ).state == "outcome_unknown"


@pytest.mark.asyncio
async def test_resource_id_and_exact_final_body_are_required_for_success(writes_db):
    client = FakeDraftClient(_conversation())
    client.create_result = MutationResult(True, 201, resource_id=902)
    client.created_thread_id = 901
    gateway = _gateway(client, writes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(gateway)

    assert raised.value.code == "external_write_conflict"
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert len([call for call in client.calls if call[0] == "create"]) == 1


@pytest.mark.asyncio
async def test_update_success_must_verify_exact_selected_thread_and_body(writes_db):
    live = _draft(701, "<p>old</p>", "2026-09-21T10:00:00Z")
    client = FakeDraftClient(_conversation(drafts=[live]))
    client.apply_update = False

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(
            _gateway(client, writes_db),
            expected_version=draft_version(live),
        )

    assert raised.value.code == "external_write_conflict"
    assert get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    ).state == "failed"


@pytest.mark.asyncio
async def test_definitive_upstream_failure_is_terminal_and_safe(writes_db):
    client = FakeDraftClient(_conversation())
    client.apply_create = False
    client.create_result = MutationResult(
        False, 500, outcome_unknown=False, detail="upstream_server_error"
    )
    gateway = _gateway(client, writes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(gateway)

    assert raised.value.code == "upstream_server_error"
    assert raised.value.retryable is True
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.error == {
        "code": "upstream_server_error",
        "retryable": True,
        "status": 500,
    }


@pytest.mark.asyncio
async def test_pre_submission_read_failure_is_terminal_not_unknown(writes_db):
    client = FakeDraftClient(_conversation())
    client.get_error_at = 1
    gateway = _gateway(client, writes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(gateway)

    assert raised.value.code == "upstream_read_failed"
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.upstream_started_at is None
    assert "private body" not in _audit_text(writes_db)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_code", "expected_status", "expected_retryable"),
    [
        (
            HelpScoutAuthError(
                "private auth detail",
                status=401,
                code="upstream_auth_error",
            ),
            "upstream_auth_error",
            502,
            False,
        ),
        (
            HelpScoutRedirectError(
                "private redirect location",
                status=301,
                code="invalid_merge_redirect",
            ),
            "invalid_merge_redirect",
            502,
            False,
        ),
        (
            HelpScoutProtocolError(
                "private response body",
                code="invalid_json",
            ),
            "invalid_json",
            502,
            False,
        ),
        (
            HelpScoutUpstreamError(
                "private not found body",
                status=404,
                code="upstream_http_error",
            ),
            "upstream_not_found",
            404,
            False,
        ),
        (
            HelpScoutUpstreamError(
                "private lock body",
                status=412,
                code="upstream_http_error",
            ),
            "conversation_locked",
            412,
            False,
        ),
        (
            HelpScoutUpstreamError(
                "private rate body",
                status=429,
                code="upstream_http_error",
                retryable=True,
            ),
            "upstream_rate_limited",
            429,
            True,
        ),
        (
            HelpScoutUpstreamError(
                "private server body",
                status=503,
                code="upstream_http_error",
                retryable=True,
            ),
            "upstream_server_error",
            503,
            True,
        ),
        (
            HelpScoutTransportError(
                "private timeout detail",
                code="request_timeout",
                retryable=True,
            ),
            "request_timeout",
            504,
            True,
        ),
        (
            HelpScoutTransportError(
                "private network detail",
                code="network_error",
                retryable=True,
            ),
            "network_error",
            503,
            True,
        ),
    ],
)
async def test_typed_preflight_read_errors_map_safely(
    writes_db,
    error,
    expected_code,
    expected_status,
    expected_retryable,
):
    client = FakeDraftClient(_conversation())
    client.get_error_at = 1
    client.get_exception = error

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(_gateway(client, writes_db))

    assert raised.value.code == expected_code
    assert raised.value.status == expected_status
    assert raised.value.retryable is expected_retryable
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.error == {
        "code": expected_code,
        "retryable": expected_retryable,
        "status": expected_status,
    }
    persisted = _audit_text(writes_db) + repr(operation.error)
    assert "private" not in persisted


@pytest.mark.asyncio
async def test_typed_error_on_alias_canonical_refetch_uses_safe_mapping(writes_db):
    client = FakeDraftClient(_conversation(321, requested_id=123))
    client.get_error_at = 2
    client.get_exception = HelpScoutTransportError(
        "private locked-refetch network detail",
        code="network_error",
        retryable=True,
    )

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(_gateway(client, writes_db), conversation_id=123)

    assert (raised.value.code, raised.value.status, raised.value.retryable) == (
        "network_error",
        503,
        True,
    )
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.upstream_started_at is None


@pytest.mark.asyncio
async def test_invalid_initial_resolution_is_non_retryable_and_terminal(writes_db):
    client = FakeDraftClient(_conversation(0))

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(_gateway(client, writes_db))

    assert raised.value.code == "invalid_conversation"
    assert raised.value.status == 502
    assert raised.value.retryable is False
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.upstream_started_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "threads",
    [
        [_draft(701, "\ud800", "2026-09-21T10:00:00Z")],
        ["not-a-thread-object"],
    ],
)
async def test_malformed_conversation_threads_fail_safely_before_submission(
    writes_db, threads
):
    client = FakeDraftClient(_conversation(drafts=threads))

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(_gateway(client, writes_db))

    assert raised.value.code == "invalid_conversation"
    assert raised.value.status == 502
    assert raised.value.retryable is False
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.upstream_started_at is None
    assert operation.error == {
        "code": "invalid_conversation",
        "retryable": False,
        "status": 502,
    }
    assert not [call for call in client.calls if call[0] in {"create", "update"}]


@pytest.mark.asyncio
async def test_lease_contention_is_terminal_and_replay_never_writes(
    writes_db, monkeypatch
):
    class BusyLease:
        async def __aenter__(self):
            raise ConversationBusyError("321")

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        gateway_module,
        "conversation_lease",
        lambda *args, **kwargs: BusyLease(),
    )
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)

    with pytest.raises(ConversationBusyError):
        await _save(gateway)
    calls = list(client.calls)
    with pytest.raises(ConversationGatewayError) as replay:
        await _save(gateway)

    assert replay.value.code == "conversation_busy"
    assert client.calls == calls
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.upstream_started_at is None


@pytest.mark.asyncio
async def test_lease_storage_failure_is_terminal_without_upstream_io(
    writes_db, monkeypatch
):
    class BrokenLease:
        async def __aenter__(self):
            raise ConversationLockStorageError("321")

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        gateway_module,
        "conversation_lease",
        lambda *args, **kwargs: BrokenLease(),
    )
    client = FakeDraftClient(_conversation())

    with pytest.raises(ConversationLockStorageError):
        await _save(_gateway(client, writes_db))

    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.error["code"] == "conversation_lock_storage_error"
    assert operation.upstream_started_at is None
    assert not [call for call in client.calls if call[0] in {"create", "update"}]


@pytest.mark.asyncio
async def test_lost_lease_guard_after_marker_never_invokes_mutation(
    writes_db, monkeypatch
):
    class LostGuardLease:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def verify_owned(self):
            return None

        async def run_guarded(self, callback):
            raise ConversationLockLostError("321")

    monkeypatch.setattr(
        gateway_module,
        "conversation_lease",
        lambda *args, **kwargs: LostGuardLease(),
    )
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)

    with pytest.raises(ConversationLockLostError):
        await _save(gateway)

    assert not [call for call in client.calls if call[0] in {"create", "update"}]
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.upstream_started_at is not None
    assert operation.error["code"] == "conversation_lock_lost"


@pytest.mark.asyncio
async def test_exit_lock_loss_after_durable_verified_success_returns_success(
    writes_db, monkeypatch
):
    class LostOnExitLease:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            raise ConversationLockLostError("321")

        async def verify_owned(self):
            return None

        async def run_guarded(self, callback):
            await self.verify_owned()
            return await callback()

    monkeypatch.setattr(
        gateway_module,
        "conversation_lease",
        lambda *args, **kwargs: LostOnExitLease(),
    )
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)

    first = await _save(gateway)
    calls = list(client.calls)
    replay = await _save(gateway)

    assert first == replay
    assert first["action"] == "created"
    assert client.calls == calls
    assert get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    ).state == "succeeded"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("upstream_status", "detail", "public_status"),
    [
        (301, "upstream_redirect", 502),
        (401, "upstream_auth_error", 502),
        (412, "conversation_locked", 412),
        (429, "upstream_rate_limited", 429),
    ],
)
async def test_mutation_statuses_are_mapped_without_redirects(
    writes_db, upstream_status, detail, public_status
):
    client = FakeDraftClient(_conversation())
    client.apply_create = False
    client.create_result = MutationResult(False, upstream_status, detail=detail)

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(_gateway(client, writes_db))

    assert raised.value.code == detail
    assert raised.value.status == public_status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("detail", "public_status"),
    [
        ("oauth_request_timeout", 504),
        ("oauth_network_error", 503),
    ],
)
async def test_pre_submission_oauth_transport_failures_remain_retryable(
    writes_db, detail, public_status
):
    client = FakeDraftClient(_conversation())
    client.apply_create = False
    client.create_result = MutationResult(False, None, detail=detail)

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(_gateway(client, writes_db))

    assert raised.value.code == detail
    assert raised.value.status == public_status
    assert raised.value.retryable is True
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.error == {
        "code": detail,
        "retryable": True,
        "status": public_status,
    }


@pytest.mark.asyncio
async def test_parser_failure_after_reservation_is_terminal_and_safe(
    writes_db, monkeypatch
):
    def fail_parser(_markup):
        raise RecursionError("private pathological parser detail")

    monkeypatch.setattr(gateway_module, "_draft_soup", fail_parser)
    client = FakeDraftClient(_conversation())

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(
            _gateway(client, writes_db),
            body="<p>Hello</p>",
            body_format="html",
        )

    assert raised.value.code == "invalid_body"
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.upstream_started_at is None
    assert "private pathological parser detail" not in _audit_text(writes_db)


@pytest.mark.asyncio
async def test_marker_failure_is_terminal_before_callback_and_replay_is_safe(
    writes_db, monkeypatch
):
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)

    async def fail_marker(**kwargs):
        raise RuntimeError("marker storage unavailable")

    monkeypatch.setattr(gateway, "_operation_mark_upstream", fail_marker)
    with pytest.raises(RuntimeError, match="marker storage unavailable"):
        await _save(gateway)
    calls = list(client.calls)

    with pytest.raises(ConversationGatewayError) as replay:
        await _save(gateway)
    assert replay.value.code == "operation_marker_failed"
    assert client.calls == calls
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.upstream_started_at is None


@pytest.mark.asyncio
async def test_cancellation_during_begin_terminalizes_without_upstream_io(writes_db):
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)
    original_begin = gateway._operation_begin
    entered = asyncio.Event()
    calls = 0

    async def interruptible_begin(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await asyncio.Event().wait()
        return await original_begin(**kwargs)

    gateway._operation_begin = interruptible_begin
    task = asyncio.create_task(_save(gateway))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert not [call for call in client.calls if call[0] in {"create", "update"}]
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.error["code"] == "cancelled_before_attempt"


@pytest.mark.asyncio
async def test_begin_exception_is_terminal_when_ledger_recovers(writes_db):
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)
    original_begin = gateway._operation_begin
    calls = 0

    async def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient begin failure")
        return await original_begin(**kwargs)

    gateway._operation_begin = fail_once
    with pytest.raises(RuntimeError, match="transient begin failure"):
        await _save(gateway)

    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.error["code"] == "operation_begin_failed"
    assert not [call for call in client.calls if call[0] in {"create", "update"}]


@pytest.mark.asyncio
async def test_cancellation_during_marker_terminalizes_as_not_submitted(writes_db):
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)
    entered = asyncio.Event()

    async def interruptible_marker(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    gateway._operation_mark_upstream = interruptible_marker
    task = asyncio.create_task(_save(gateway))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert not [call for call in client.calls if call[0] in {"create", "update"}]
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.upstream_started_at is None


@pytest.mark.asyncio
async def test_cancellation_during_final_ledger_write_is_durably_unknown(writes_db):
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)
    original_finish = gateway._operation_finish
    entered = asyncio.Event()
    calls = 0

    async def interruptible_finish(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await asyncio.Event().wait()
        return await original_finish(**kwargs)

    gateway._operation_finish = interruptible_finish
    task = asyncio.create_task(_save(gateway))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "outcome_unknown"
    assert operation.upstream_started_at is not None


@pytest.mark.asyncio
async def test_cancellation_after_submission_marker_is_unknown_and_propagates(writes_db):
    client = FakeDraftClient(_conversation())
    client.raise_on_create = asyncio.CancelledError()
    gateway = _gateway(client, writes_db)

    with pytest.raises(asyncio.CancelledError):
        await _save(gateway)

    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "outcome_unknown"
    assert operation.upstream_started_at is not None


@pytest.mark.asyncio
async def test_deterministic_invalid_body_is_terminal_and_replays_without_write(writes_db):
    client = FakeDraftClient(_conversation())
    gateway = _gateway(client, writes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(gateway, body="   ")
    assert raised.value.code == "invalid_body"
    calls_after_first = list(client.calls)

    with pytest.raises(ConversationGatewayError) as replay:
        await _save(gateway, body="   ")
    assert replay.value.code == "invalid_body"

    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-save-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert client.calls == calls_after_first
    assert [call[0] for call in client.calls] == ["get"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "body_format", "expected_code"),
    [
        (123, "text", "invalid_body"),
        (["valid", "json"], "html", "invalid_body"),
        ("Hello", [], "invalid_body_format"),
        ("Hello", {"format": "html"}, "invalid_body_format"),
    ],
)
async def test_wrong_input_types_are_rejected_before_reservation(
    writes_db, body, body_format, expected_code
):
    client = FakeDraftClient(_conversation())

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(
            _gateway(client, writes_db),
            body=body,
            body_format=body_format,
    )

    assert raised.value.code == expected_code
    with pytest.raises(AgentOperationNotFoundError):
        get_operation(
            db_path=writes_db,
            idempotency_key="draft-save-1",
            identity=IDENTITY,
        )
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "body_format", "expected_code"),
    [
        ("\ud800", "text", "invalid_body"),
        ({"not-json"}, "text", "invalid_body"),
        ("Hello", {"not-json"}, "invalid_body_format"),
    ],
)
async def test_noncanonical_json_inputs_raise_stable_errors_before_reservation(
    writes_db, body, body_format, expected_code
):
    client = FakeDraftClient(_conversation())

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(
            _gateway(client, writes_db),
            body=body,
            body_format=body_format,
        )

    assert raised.value.code == expected_code
    with pytest.raises(AgentOperationNotFoundError):
        get_operation(
            db_path=writes_db,
            idempotency_key="draft-save-1",
            identity=IDENTITY,
        )
    assert client.calls == []


@pytest.mark.asyncio
async def test_draft_body_has_explicit_pre_reservation_byte_bound(writes_db):
    client = FakeDraftClient(_conversation())

    with pytest.raises(ConversationGatewayError) as raised:
        await _save(
            _gateway(client, writes_db),
            body="x" * (1024 * 1024 + 1),
        )

    assert raised.value.code == "body_too_large"
    assert raised.value.status == 413
    with pytest.raises(AgentOperationNotFoundError):
        get_operation(
            db_path=writes_db,
            idempotency_key="draft-save-1",
            identity=IDENTITY,
        )
    assert client.calls == []


@pytest.mark.asyncio
async def test_send_draft_verifies_preconditions_marks_then_sends_before_cleanup(
    writes_db,
):
    customer = _customer(500, "<p>Can you help?</p>", "2026-09-25T10:00:00Z")
    draft = _draft(
        701,
        '<p>Hello <script>private-secret</script><strong>World</strong></p>',
        "2026-09-25T11:00:00Z",
    )
    expected_body = normalize_draft_body(draft["body"], "html")
    client = FakeDraftClient(_conversation(drafts=[draft], threads=[customer]))
    bookkeeper = FakeSendBookkeeper()
    gateway = ConversationGateway(
        client,
        db_path=writes_db,
        post_send_bookkeeper=bookkeeper,
    )

    def assert_durable_marker():
        operation = get_operation(
            db_path=writes_db,
            idempotency_key="draft-send-1",
            identity=IDENTITY,
        )
        assert operation.state == "in_progress"
        assert operation.upstream_started_at is not None
        assert operation.request_metadata["action"] == "sent"
        assert operation.request_metadata["conversation_id"] == "321"
        assert operation.request_metadata["expected_predecessor_thread_id"] == "500"
        assert operation.request_metadata["normalized_body_hash"] == hashlib.sha256(
            expected_body.encode()
        ).hexdigest()

    client.before_submit = assert_durable_marker
    result = await _send_draft(
        gateway,
        expected_version=draft_version(draft),
    )

    assert result == {
        "action": "sent",
        "body_sha256": hashlib.sha256(expected_body.encode()).hexdigest(),
        "conversation_id": 321,
        "status": "sent",
        "thread_id": 990,
        "warnings": [],
    }
    assert [call[0] for call in client.calls] == [
        "get",
        "send",
        "get",
        "delete",
        "ensure_status",
        "tags",
    ]
    assert client.calls[1] == ("send", 321, expected_body, "pending", 44)
    assert client.send_verify_status_flags == [False]
    assert client.calls.index(("send", 321, expected_body, "pending", 44)) < next(
        index for index, call in enumerate(client.calls) if call[0] == "delete"
    )
    assert client.calls[-1] == (
        "tags",
        321,
        ("supa-processed", "supa-auto-sent"),
        ("field-journal",),
    )
    assert len(bookkeeper.calls) == 1
    assert bookkeeper.calls[0]["thread_id"] == 990
    assert bookkeeper.calls[0]["reply_html"] == expected_body
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-send-1",
        identity=IDENTITY,
    )
    assert operation.state == "succeeded"
    assert operation.upstream_thread_id == "990"
    persisted = _audit_text(writes_db) + repr(operation.to_public_dict())
    assert "private-secret" not in persisted


@pytest.mark.asyncio
async def test_send_reply_normalizes_text_and_propagates_customer_id(writes_db):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    bookkeeper = FakeSendBookkeeper()
    gateway = ConversationGateway(
        client,
        db_path=writes_db,
        post_send_bookkeeper=bookkeeper,
    )
    expected_body = normalize_draft_body('Hello <b> & "there"', "text")

    result = await _send_reply(
        gateway,
        body='Hello <b> & "there"',
    )

    assert result["status"] == "sent"
    assert client.calls[1] == ("send", 321, expected_body, "pending", 44)
    assert result["body_sha256"] == hashlib.sha256(expected_body.encode()).hexdigest()
    assert bookkeeper.calls[0]["customer_message"] == "Question"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["reply", "draft"])
async def test_definitive_send_emits_one_reply_event_and_replay_dedupes(
    writes_db,
    mode,
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    draft = _draft(701, "<p>Private draft body</p>", "2026-09-25T11:00:00Z")
    client = FakeDraftClient(
        _conversation(
            threads=[customer],
            drafts=[draft] if mode == "draft" else [],
        )
    )
    if mode == "draft":
        client.delete_result = MutationResult(
            False,
            500,
            detail="upstream_server_error",
        )
    gateway = _gateway(client, writes_db)
    key = f"{mode}-event"

    if mode == "draft":
        first = await _send_draft(
            gateway,
            expected_version=draft_version(draft),
            key=key,
        )
        replay = await _send_draft(
            gateway,
            expected_version=draft_version(draft),
            key=key,
        )
        operation_name = "draft.send"
    else:
        first = await _send_reply(
            gateway,
            body="Private reply sk-secret-value",
            key=key,
        )
        replay = await _send_reply(
            gateway,
            body="Private reply sk-secret-value",
            key=key,
        )
        operation_name = "reply.send"

    assert replay == first
    assert _stored_supa_events(writes_db) == [
        {
            "type": "supa.reply.sent",
            "conversation_id": "321",
            "thread_id": "990",
            "source": "supa",
            "summary": {
                "operation_id": key,
                "operation_name": operation_name,
                "request_id": f"request-{key}",
                "result": "sent",
                "status": first["status"],
                "token_id": "writer-1",
                "token_name": "Draft writer",
            },
        }
    ]
    persisted = json.dumps(_stored_supa_events(writes_db), sort_keys=True)
    assert "Private draft body" not in persisted
    assert "Private reply" not in persisted
    assert "sk-secret-value" not in persisted


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failed", "unknown"])
async def test_failed_or_unknown_send_emits_no_success_event(writes_db, outcome):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    client.apply_send = False
    client.send_result = (
        MutationResult(False, 412, detail="conversation_locked")
        if outcome == "failed"
        else MutationResult(
            False,
            None,
            outcome_unknown=True,
            detail="network_error",
        )
    )

    with pytest.raises(ConversationGatewayError):
        await _send_reply(_gateway(client, writes_db), key=f"send-{outcome}")

    assert _stored_supa_events(writes_db) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "body_format", "expected_code"),
    [
        ("\ud800", "text", "invalid_body"),
        ("Hello", "\ud800", "invalid_body_format"),
        (123, "text", "invalid_body"),
        ("Hello", [], "invalid_body_format"),
    ],
)
async def test_send_reply_rejects_noncanonical_inputs_before_reservation(
    writes_db, body, body_format, expected_code
):
    client = FakeDraftClient(_conversation())

    with pytest.raises(ConversationGatewayError) as raised:
        await _send_reply(
            _gateway(client, writes_db),
            body=body,
            body_format=body_format,
            key="invalid-send-input",
        )

    assert raised.value.code == expected_code
    with pytest.raises(AgentOperationNotFoundError):
        get_operation(
            db_path=writes_db,
            idempotency_key="invalid-send-input",
            identity=IDENTITY,
        )
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["draft", "reply"])
@pytest.mark.parametrize("given_thread_id", [700, 701])
async def test_send_rejects_draft_as_expected_thread_and_replays_details(
    writes_db, method, given_thread_id
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    older = _draft(700, "<p>Older</p>", "2026-09-25T10:30:00Z")
    draft = _draft(701, "<p>Draft</p>", "2026-09-25T11:00:00Z")
    client = FakeDraftClient(_conversation(drafts=[older, draft], threads=[customer]))
    gateway = _gateway(client, writes_db)
    details = {
        "latest_actionable_thread_id": 500,
        "given_thread_id": given_thread_id,
        "draft_thread_id": given_thread_id,
    }

    for _ in range(2):
        with pytest.raises(ConversationGatewayError) as raised:
            if method == "draft":
                await _send_draft(
                    gateway,
                    expected_version=draft_version(draft),
                    expected_latest_thread_id=given_thread_id,
                )
            else:
                await _send_reply(gateway, expected_latest_thread_id=given_thread_id)

        assert raised.value.code == "wrong_expected_thread"
        assert raised.value.status == 409
        assert raised.value.retryable is False
        assert raised.value.details == details
        assert "not the draft's thread" in str(raised.value)

    assert not [call for call in client.calls if call[0] != "get"]
    operation = get_operation(
        db_path=writes_db,
        idempotency_key=("draft-send-1" if method == "draft" else "reply-send-1"),
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.upstream_started_at is None
    assert operation.error["code"] == "wrong_expected_thread"
    for key, value in details.items():
        assert operation.error[key] == str(value)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["draft", "reply"])
async def test_send_rejects_changed_latest_actionable_thread_before_submission(
    writes_db, method
):
    customer = _customer(501, "<p>New reply</p>", "2026-09-25T12:00:00Z")
    draft = _draft(701, "<p>Draft</p>", "2026-09-25T11:00:00Z")
    client = FakeDraftClient(_conversation(drafts=[draft], threads=[customer]))
    gateway = ConversationGateway(client, db_path=writes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        if method == "draft":
            await _send_draft(
                gateway,
                expected_version=draft_version(draft),
                expected_latest_thread_id=500,
            )
        else:
            await _send_reply(gateway, expected_latest_thread_id=500)

    assert raised.value.code == "conversation_changed"
    assert raised.value.status == 409
    assert raised.value.retryable is False
    assert raised.value.details == {
        "thread_id": 501,
        "latest_actionable_thread_id": 501,
        "given_thread_id": 500,
    }
    assert str(raised.value) == "The latest actionable thread is now 501, not 500."
    assert not [call for call in client.calls if call[0] == "send"]
    operation = get_operation(
        db_path=writes_db,
        idempotency_key=("draft-send-1" if method == "draft" else "reply-send-1"),
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.upstream_started_at is None
    assert operation.error["latest_actionable_thread_id"] == "501"
    assert operation.error["given_thread_id"] == "500"
    with pytest.raises(ConversationGatewayError) as replayed:
        if method == "draft":
            await _send_draft(gateway, expected_version=draft_version(draft))
        else:
            await _send_reply(gateway)
    assert replayed.value.details == raised.value.details
    assert str(replayed.value) == str(raised.value)


@pytest.mark.asyncio
async def test_send_draft_rejects_stale_draft_version_before_submission(writes_db):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    draft = _draft(701, "<p>Current</p>", "2026-09-25T11:00:00Z")
    client = FakeDraftClient(_conversation(drafts=[draft], threads=[customer]))

    with pytest.raises(ConversationGatewayError) as raised:
        await _send_draft(
            ConversationGateway(client, db_path=writes_db),
            expected_version="a" * 64,
        )

    assert raised.value.code == "draft_conflict"
    assert not [call for call in client.calls if call[0] == "send"]


@pytest.mark.asyncio
async def test_send_draft_terminalizes_malformed_version_with_stable_code(writes_db):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    draft = _draft(701, "<p>Current</p>", "2026-09-25T11:00:00Z")
    client = FakeDraftClient(_conversation(drafts=[draft], threads=[customer]))

    with pytest.raises(ConversationGatewayError) as raised:
        await _send_draft(
            ConversationGateway(client, db_path=writes_db),
            expected_version="not-a-version",
            key="invalid-send-version",
        )

    assert raised.value.code == "invalid_draft_version"
    assert raised.value.status == 400
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="invalid-send-version",
        identity=IDENTITY,
    )
    assert operation.state == "failed"
    assert operation.upstream_started_at is None
    assert not [call for call in client.calls if call[0] == "send"]


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown_response", [False, True])
async def test_send_without_resource_id_reconciles_one_exact_new_reply(
    writes_db, unknown_response
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    client.sent_thread_id = 991
    client.send_result = MutationResult(
        ok=not unknown_response,
        status_code=None if unknown_response else 201,
        resource_id=None,
        outcome_unknown=unknown_response,
        detail="network_error" if unknown_response else "resource_id_missing",
    )

    result = await _send_reply(_gateway(client, writes_db))

    assert result["status"] == "sent"
    assert result["thread_id"] == 991
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="reply-send-1",
        identity=IDENTITY,
    )
    assert operation.state == "succeeded"
    assert operation.upstream_thread_id == "991"


@pytest.mark.asyncio
async def test_send_reconciliation_counts_all_matching_post_attempt_replies(
    writes_db,
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    expected_body = normalize_draft_body("Hello\n\nWorld", "text")
    client.sent_thread_id = 991
    client.send_result = MutationResult(
        False,
        None,
        outcome_unknown=True,
        detail="network_error",
    )
    client.additional_sent_threads = [
        _published_reply(992, expected_body, "2099-01-01T00:00:01Z")
    ]

    with pytest.raises(ConversationGatewayError) as raised:
        await _send_reply(_gateway(client, writes_db))

    assert raised.value.code == "outcome_unknown"
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="reply-send-1",
        identity=IDENTITY,
    )
    assert operation.state == "outcome_unknown"
    assert not [call for call in client.calls if call[0] == "delete"]


@pytest.mark.asyncio
async def test_unknown_send_with_no_exact_candidate_stays_unknown_and_never_replays(
    writes_db,
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    client.apply_send = False
    client.send_result = MutationResult(
        False,
        None,
        outcome_unknown=True,
        detail="request_timeout",
    )
    gateway = _gateway(client, writes_db)

    with pytest.raises(ConversationGatewayError) as first:
        await _send_reply(gateway)
    assert first.value.code == "outcome_unknown"
    calls = list(client.calls)

    with pytest.raises(ConversationGatewayError) as replay:
        await _send_reply(gateway)
    assert replay.value.code == "outcome_unknown"
    assert client.calls == calls
    assert not [call for call in client.calls if call[0] in {"delete", "tags"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_resource_id", [True, 0, -1, 2**63])
async def test_invalid_resource_id_is_not_authoritative(
    writes_db, invalid_resource_id
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    client.apply_send = False
    client.send_result = MutationResult(
        True,
        201,
        resource_id=invalid_resource_id,
        detail="resource_id_invalid",
    )

    with pytest.raises(ConversationGatewayError) as raised:
        await _send_reply(_gateway(client, writes_db))

    assert raised.value.code == "outcome_unknown"
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="reply-send-1",
        identity=IDENTITY,
    )
    assert operation.state == "outcome_unknown"


@pytest.mark.asyncio
async def test_definitive_resource_id_survives_verification_read_failure_as_warning(
    writes_db,
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    client.get_error_at = 2
    gateway = _gateway(client, writes_db)

    result = await _send_reply(gateway)

    assert result["thread_id"] == 990
    assert result["status"] == "sent_with_warning"
    assert "verification_failed" in result["warnings"]
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="reply-send-1",
        identity=IDENTITY,
    )
    assert operation.state == "sent_with_warning"
    assert operation.upstream_thread_id == "990"


@pytest.mark.asyncio
@pytest.mark.parametrize("post_state", ["lagging", "mismatched"])
async def test_definitive_resource_id_keeps_delivery_but_warns_on_unverified_post_state(
    writes_db, post_state
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    if post_state == "lagging":
        client.apply_send = False
    else:
        client.sent_body_override = "<p>Different body</p>"

    result = await _send_reply(
        _gateway(client, writes_db),
        key=f"definitive-{post_state}",
    )

    assert result["thread_id"] == 990
    assert result["status"] == "sent_with_warning"
    assert result["warnings"] == [
        "verification_unconfirmed"
        if post_state == "lagging"
        else "verification_mismatch"
    ]
    operation = get_operation(
        db_path=writes_db,
        idempotency_key=f"definitive-{post_state}",
        identity=IDENTITY,
    )
    assert operation.state == "sent_with_warning"


@pytest.mark.asyncio
async def test_cancellation_after_definitive_resource_id_is_durable_sent_warning(
    writes_db,
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    client.get_error_at = 2
    client.get_exception = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _send_reply(_gateway(client, writes_db))

    operation = get_operation(
        db_path=writes_db,
        idempotency_key="reply-send-1",
        identity=IDENTITY,
    )
    assert operation.state == "sent_with_warning"
    assert operation.upstream_thread_id == "990"
    assert operation.response["warnings"] == ["verification_cancelled"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cancellation_point", "expected_warning"),
    [
        ("verification", "verification_cancelled"),
        ("bookkeeping", "bookkeeping_cancelled"),
    ],
)
async def test_event_store_failure_cannot_mask_definitive_send_cancellation(
    writes_db,
    monkeypatch,
    cancellation_point,
    expected_warning,
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    bookkeeper = None
    if cancellation_point == "verification":
        client.get_error_at = 2
        client.get_exception = asyncio.CancelledError()
        gateway = _gateway(client, writes_db)
    else:
        bookkeeper = FakeSendBookkeeper()
        bookkeeper.error = asyncio.CancelledError()
        gateway = ConversationGateway(
            client,
            db_path=writes_db,
            post_send_bookkeeper=bookkeeper,
        )
    key = f"{cancellation_point[:6]}-cancel-evt"
    original_append = gateway_module.append_supa_event
    event_attempts = 0

    def flaky_append(**kwargs):
        nonlocal event_attempts
        event_attempts += 1
        if event_attempts == 1:
            raise AgentEventStorageError("event store unavailable")
        return original_append(**kwargs)

    monkeypatch.setattr(gateway_module, "append_supa_event", flaky_append)

    with pytest.raises(asyncio.CancelledError):
        await _send_reply(gateway, key=key)

    operation = get_operation(
        db_path=writes_db,
        idempotency_key=key,
        identity=IDENTITY,
    )
    assert operation.state == "sent_with_warning"
    assert operation.response["warnings"] == [expected_warning]
    assert _stored_supa_events(writes_db) == []
    upstream_calls = list(client.calls)
    bookkeeper_calls = list(bookkeeper.calls) if bookkeeper is not None else None

    replay = await _send_reply(gateway, key=key)

    assert replay["action"] == "sent"
    assert replay["conversation_id"] == 321
    assert replay["thread_id"] == 990
    assert replay["status"] == "sent_with_warning"
    assert replay["warnings"] == [expected_warning]
    assert client.calls == upstream_calls
    if bookkeeper is not None:
        assert bookkeeper.calls == bookkeeper_calls
    assert event_attempts == 2
    assert len(_stored_supa_events(writes_db)) == 1


@pytest.mark.asyncio
async def test_definitive_send_failure_is_terminal_and_does_not_clean_drafts(
    writes_db,
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    draft = _draft(701, "<p>Draft</p>", "2026-09-25T11:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer], drafts=[draft]))
    client.apply_send = False
    client.send_result = MutationResult(False, 412, detail="conversation_locked")

    with pytest.raises(ConversationGatewayError) as raised:
        await _send_draft(
            _gateway(client, writes_db),
            expected_version=draft_version(draft),
        )

    assert raised.value.code == "conversation_locked"
    assert raised.value.status == 412
    assert not [call for call in client.calls if call[0] in {"delete", "tags"}]
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-send-1",
        identity=IDENTITY,
    )
    assert operation.state == "failed"


@pytest.mark.asyncio
async def test_post_send_cleanup_failure_is_warning_and_replay_never_resends(
    writes_db,
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    draft = _draft(701, "<p>Draft</p>", "2026-09-25T11:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer], drafts=[draft]))
    client.delete_result = MutationResult(False, 500, detail="upstream_server_error")
    gateway = _gateway(client, writes_db)

    first = await _send_draft(gateway, expected_version=draft_version(draft))
    calls = list(client.calls)
    replay = await _send_draft(gateway, expected_version=draft_version(draft))

    assert first == replay
    assert first["status"] == "sent_with_warning"
    assert first["warnings"] == ["draft_cleanup_failed"]
    assert client.calls == calls
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="draft-send-1",
        identity=IDENTITY,
    )
    assert operation.state == "sent_with_warning"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("candidate_case", "sent_at", "body_override", "source_type"),
    [
        ("pre_attempt", "2000-01-01T00:00:00Z", None, "api"),
        ("wrong_body", "2099-01-01T00:00:00Z", "<p>Different</p>", "api"),
        ("non_api", "2099-01-01T00:00:00Z", None, "web"),
    ],
)
async def test_unknown_send_rejects_each_inexact_candidate_dimension(
    writes_db, candidate_case, sent_at, body_override, source_type
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    client.send_result = MutationResult(
        False,
        None,
        outcome_unknown=True,
        detail="network_error",
    )
    client.sent_at = sent_at
    client.sent_body_override = body_override
    client.sent_source_type = source_type

    with pytest.raises(ConversationGatewayError) as raised:
        await _send_reply(_gateway(client, writes_db), key=f"inexact-{candidate_case}")

    assert raised.value.code == "outcome_unknown"
    operation = get_operation(
        db_path=writes_db,
        idempotency_key=f"inexact-{candidate_case}",
        identity=IDENTITY,
    )
    assert operation.state == "outcome_unknown"


def test_send_reconciliation_rejects_wrong_locked_predecessor():
    body = "<p>Exact</p>"
    before = _conversation(
        threads=[_customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")]
    )
    after = copy.deepcopy(before)
    after["_embedded"]["threads"].append(
        _published_reply(991, body, "2099-01-01T00:00:00Z")
    )

    assert ConversationGateway._sent_reply_candidates(
        before=before,
        after=after,
        normalized_body=body,
        attempted_at=gateway_module.datetime.fromisoformat(
            "2026-09-25T11:00:00+00:00"
        ),
        expected_predecessor_thread_id=499,
    ) == []


def test_send_reconciliation_respects_helpscout_whole_second_timestamp_precision():
    body = "<p>Exact</p>"
    before = _conversation(
        threads=[_customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")]
    )
    after = copy.deepcopy(before)
    after["_embedded"]["threads"].append(
        _published_reply(991, body, "2026-09-25T12:00:00Z")
    )
    attempted_at = gateway_module.datetime.fromisoformat(
        "2026-09-25T12:00:00.601+00:00"
    )

    assert ConversationGateway._sent_reply_candidates(
        before=before,
        after=after,
        normalized_body=body,
        attempted_at=attempted_at,
        expected_predecessor_thread_id=500,
    ) == [991]
    assert ConversationGateway._definitive_send_verification_warning(
        after=after,
        conversation_id=321,
        thread_id=991,
        normalized_body=body,
        attempted_at=attempted_at,
    ) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_warning"),
    [
        ("pending", "pending_status_unverified"),
        ("tags", "tag_update_failed"),
        ("learning", "learning_audit_failed"),
    ],
)
async def test_post_send_bookkeeping_failures_are_durable_warnings(
    writes_db, failure, expected_warning
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    bookkeeper = FakeSendBookkeeper()
    if failure == "pending":
        client.status_verified = False
    elif failure == "tags":
        client.tags_result = MutationResult(False, 500, detail="upstream_server_error")
    else:
        bookkeeper.error = RuntimeError("private learning failure")
    gateway = ConversationGateway(
        client,
        db_path=writes_db,
        post_send_bookkeeper=bookkeeper,
    )

    result = await _send_reply(gateway, key=f"bookkeeping-{failure}")

    assert result["status"] == "sent_with_warning"
    assert expected_warning in result["warnings"]
    operation = get_operation(
        db_path=writes_db,
        idempotency_key=f"bookkeeping-{failure}",
        identity=IDENTITY,
    )
    assert operation.state == "sent_with_warning"
    assert "private learning failure" not in _audit_text(writes_db)


@pytest.mark.asyncio
async def test_tag_replace_preserves_fresh_post_send_external_tags(writes_db):
    class FreshTagClient(FakeDraftClient):
        async def ensure_conversation_status(self, conversation_id: int, status: str):
            result = await super().ensure_conversation_status(conversation_id, status)
            result["conversation"]["tags"].append({"tag": "external-new-tag"})
            return result

    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FreshTagClient(_conversation(threads=[customer]))

    result = await _send_reply(
        _gateway(client, writes_db),
        key="fresh-tag-preservation",
    )

    assert result["status"] == "sent"
    assert client.calls[-1] == (
        "tags",
        321,
        ("supa-processed", "supa-auto-sent"),
        ("field-journal", "external-new-tag"),
    )


@pytest.mark.asyncio
async def test_definitive_success_replays_without_any_second_upstream_call(writes_db):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    gateway = _gateway(client, writes_db)

    first = await _send_reply(gateway, key="normal-replay")
    calls = list(client.calls)
    replay = await _send_reply(gateway, key="normal-replay")

    assert replay == first
    assert replay["status"] == "sent"
    assert client.calls == calls
    assert len([call for call in client.calls if call[0] == "send"]) == 1


@pytest.mark.asyncio
async def test_cancellation_during_post_send_bookkeeping_is_durable_and_propagates(
    writes_db,
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    bookkeeper = FakeSendBookkeeper()
    bookkeeper.error = asyncio.CancelledError()
    gateway = ConversationGateway(
        client,
        db_path=writes_db,
        post_send_bookkeeper=bookkeeper,
    )

    with pytest.raises(asyncio.CancelledError):
        await _send_reply(gateway, key="bookkeeping-cancelled")

    operation = get_operation(
        db_path=writes_db,
        idempotency_key="bookkeeping-cancelled",
        identity=IDENTITY,
    )
    assert operation.state == "sent_with_warning"
    assert operation.upstream_thread_id == "990"
    assert operation.response["warnings"] == ["bookkeeping_cancelled"]


@pytest.mark.asyncio
async def test_sync_learning_bookkeeper_is_offloaded_from_lease_event_loop(writes_db):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    caller_thread = threading.get_ident()
    observed_threads: list[int] = []

    def sync_bookkeeper(**_kwargs):
        observed_threads.append(threading.get_ident())
        return True

    gateway = ConversationGateway(
        client,
        db_path=writes_db,
        post_send_bookkeeper=sync_bookkeeper,
    )

    result = await _send_reply(gateway, key="sync-bookkeeper")

    assert result["status"] == "sent"
    assert observed_threads and observed_threads != [caller_thread]


@pytest.mark.asyncio
async def test_lock_loss_before_post_send_bookkeeping_stops_mutations_but_keeps_send(
    writes_db, monkeypatch
):
    class LostBeforeBookkeepingLease:
        def __init__(self):
            self.guarded_calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def verify_owned(self):
            return None

        async def run_guarded(self, callback):
            self.guarded_calls += 1
            if self.guarded_calls == 2:
                raise ConversationLockLostError("321")
            return await callback()

    lease = LostBeforeBookkeepingLease()
    monkeypatch.setattr(
        gateway_module,
        "conversation_lease",
        lambda *args, **kwargs: lease,
    )
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))

    result = await _send_reply(
        _gateway(client, writes_db),
        key="lost-before-bookkeeping",
    )

    assert result["status"] == "sent_with_warning"
    assert result["thread_id"] == 990
    assert result["warnings"] == ["conversation_lock_lost"]
    assert [call[0] for call in client.calls] == ["get", "send", "get"]
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="lost-before-bookkeeping",
        identity=IDENTITY,
    )
    assert operation.state == "sent_with_warning"
    assert operation.upstream_thread_id == "990"


@pytest.mark.asyncio
async def test_cancelling_same_key_replay_during_reserve_cannot_corrupt_active_send(
    writes_db,
):
    class BlockingSendClient(FakeDraftClient):
        def __init__(self, conversation):
            super().__init__(conversation)
            self.send_entered = asyncio.Event()
            self.send_release = asyncio.Event()

        async def send_reply_result(self, *args, **kwargs):
            self.send_entered.set()
            await self.send_release.wait()
            return await super().send_reply_result(*args, **kwargs)

    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = BlockingSendClient(_conversation(threads=[customer]))
    gateway = _gateway(client, writes_db)
    original = asyncio.create_task(_send_reply(gateway, key="reserve-cancel-race"))
    await client.send_entered.wait()

    real_reserve = gateway._operation_reserve
    replay_reserved = asyncio.Event()

    async def pause_after_replay_reservation(**kwargs):
        reservation = await real_reserve(**kwargs)
        assert reservation.replayed is True
        replay_reserved.set()
        await asyncio.Event().wait()

    gateway._operation_reserve = pause_after_replay_reservation
    duplicate = asyncio.create_task(
        _send_reply(gateway, key="reserve-cancel-race")
    )
    await replay_reserved.wait()
    duplicate.cancel()
    with pytest.raises(asyncio.CancelledError):
        await duplicate

    client.send_release.set()
    result = await original

    assert result["status"] == "sent"
    assert len([call for call in client.calls if call[0] == "send"]) == 1
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="reserve-cancel-race",
        identity=IDENTITY,
    )
    assert operation.state == "succeeded"


@pytest.mark.asyncio
async def test_send_persistence_and_structured_logs_never_contain_body_or_error_text(
    writes_db, caplog
):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    client.raise_on_send = RuntimeError("private-exception-text sk-secret-value")
    gateway = _gateway(client, writes_db)
    secret_body = "Customer answer sk-secret-value"

    with pytest.raises(ConversationGatewayError) as raised:
        await _send_reply(gateway, body=secret_body, key="secret-unknown")

    assert raised.value.code == "outcome_unknown"
    operation = get_operation(
        db_path=writes_db,
        idempotency_key="secret-unknown",
        identity=IDENTITY,
    )
    persisted = _audit_text(writes_db) + repr(operation.to_public_dict())
    assert secret_body not in persisted
    assert "sk-secret-value" not in persisted
    logs = caplog.text
    assert "private-exception-text" not in logs
    assert "sk-secret-value" not in logs


def _helpscout_reserialize(html: str) -> str:
    """Simulate the stored HTML returned by the mailbox."""
    return html.replace("\xa0", "&nbsp;").replace("<br/>", "<br>")


SPACED_HTML = "<p>Hello Morgan,</p><p>&nbsp;</p><p>The binder's ready for a proof.</p><p>&nbsp;</p><p>Thank you,<br>Rowan</p>"


@pytest.mark.asyncio
async def test_html_draft_create_verifies_after_help_scout_reserializes(writes_db):
    client = FakeDraftClient(_conversation())
    client.store = _helpscout_reserialize
    result = await _save(_gateway(client, writes_db), body=SPACED_HTML, body_format="html")

    assert result["action"] == "created"
    assert "&nbsp;" in client.conversation["_embedded"]["threads"][-1]["body"]
    operation = get_operation(db_path=writes_db, idempotency_key="draft-save-1", identity=IDENTITY)
    assert operation.state == "succeeded"


@pytest.mark.asyncio
async def test_html_draft_update_verifies_after_help_scout_reserializes(writes_db):
    draft = _draft(701, SPACED_HTML.replace("same", "old"), "2026-09-25T11:00:00Z")
    client = FakeDraftClient(_conversation(drafts=[draft]))
    client.store = _helpscout_reserialize
    result = await _save(
        _gateway(client, writes_db),
        body=SPACED_HTML,
        body_format="html",
        expected_version=draft_version(draft),
    )

    assert result["action"] == "updated"
    operation = get_operation(db_path=writes_db, idempotency_key="draft-save-1", identity=IDENTITY)
    assert operation.state == "succeeded"


@pytest.mark.asyncio
async def test_html_reply_send_has_no_warning_after_help_scout_reserializes(writes_db):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    client.store = _helpscout_reserialize
    result = await _send_reply(_gateway(client, writes_db), body=SPACED_HTML, body_format="html")

    assert result["status"] == "sent"
    assert result.get("warnings", []) == []


@pytest.mark.asyncio
async def test_html_reply_send_without_resource_id_still_reconciles_after_reserialize(writes_db):
    customer = _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")
    client = FakeDraftClient(_conversation(threads=[customer]))
    client.store = _helpscout_reserialize
    client.sent_thread_id = 991
    client.send_result = MutationResult(ok=True, status_code=201, resource_id=None, detail="resource_id_missing")
    result = await _send_reply(_gateway(client, writes_db), body=SPACED_HTML, body_format="html")

    assert result["status"] == "sent"
    assert result["thread_id"] == 991


def test_normalized_body_survives_help_scout_reserialization():
    once = normalize_draft_body(SPACED_HTML, "html")
    assert normalize_draft_body(_helpscout_reserialize(once), "html") == once

@pytest.mark.asyncio
async def test_plain_text_draft_update_verifies_after_entity_and_break_reserialization(writes_db):
    body = "Hello, Morgan!\n\nThe binder's ready. Please check the last sheet.\n\nThank you,\nRowan"
    draft = _draft(701, "<p>Old draft</p>", "2026-09-25T11:00:00Z")
    client = FakeDraftClient(_conversation(drafts=[draft]))
    # Real Help Scout parses the submitted HTML before returning its serialization.
    client.store = lambda value: _helpscout_reserialize(normalize_draft_body(value, "html"))
    result = await _save(_gateway(client, writes_db), body=body, body_format="text",
                         expected_version=draft_version(draft))
    assert result["action"] == "updated"
    assert get_operation(db_path=writes_db, idempotency_key="draft-save-1", identity=IDENTITY).state == "succeeded"

def test_body_reserialization_does_not_accept_different_text_or_link_target():
    from services.conversation_gateway import _same_body
    expected = normalize_draft_body("<p>It's ready: <a href='https://example.com/one'>review</a></p>", "html")
    assert not _same_body(expected.replace("ready", "changed"), expected)
    assert not _same_body(expected.replace("example.com/one", "example.com/two"), expected)
