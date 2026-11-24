"""Editing an internal note through the Agent API: API notes only, versioned, replayable."""

from __future__ import annotations

import io
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from api import agent as agent_api
from db.database import init_db
from main import app
from services.agent_auth import AgentIdentity, write_token_records
from services.agent_operations import AgentOperationConflictError, get_operation
from services.conversation_gateway import (
    ConversationGateway,
    ConversationGatewayError,
    conversation_version,
    normalize_conversation,
    normalize_draft_body,
    note_version,
)
from services.helpscout import MutationResult
from supa_cli.main import main as cli_main
from tests.test_agent_notes import (
    NOTE_HEADERS,
    NOTE_IDENTITY,
    NOTE_TOKEN,
    READ_HEADERS,
    READ_TOKEN,
    WRITER_HEADERS,
    WRITER_TOKEN,
    FakeNoteClient,
    _note_thread,
    _record,
)
from tests.test_helpscout import _client_with_transport
from tests.test_agent_writes import (
    _conversation,
    _customer,
    _draft,
    _stored_supa_events,
)


DECISION_NOTE = "<p>Supa responder · DRAFT — NOT SENT</p>"
SENT_NOTE = "Supa responder · SENT by Operator"


def _web_note(thread_id: int, body: str) -> dict:
    return {**_note_thread(thread_id, body, "2026-09-30T09:00:00Z"), "source": {"type": "web"}}


def _client(status: str = "pending", *, extra: list[dict] | None = None) -> FakeNoteClient:
    conversation = _conversation(
        threads=[
            _customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z"),
            _note_thread(880, DECISION_NOTE, "2026-09-30T10:00:00Z"),
            *(extra or []),
        ]
    )
    conversation["status"] = status
    return FakeNoteClient(conversation)


def _thread(client: FakeNoteClient, thread_id: int) -> dict:
    return next(
        thread
        for thread in client.conversation["_embedded"]["threads"]
        if thread["id"] == thread_id
    )


def _current_version(client: FakeNoteClient, thread_id: int = 880) -> str:
    return note_version(_thread(client, thread_id))


@pytest.fixture
def notes_db(tmp_path):
    path = tmp_path / "note-update.db"
    init_db(str(path))
    return path


async def _update_note(
    gateway: ConversationGateway,
    *,
    expected_version: str,
    thread_id: int = 880,
    body: str = SENT_NOTE,
    body_format: str = "text",
    key: str = "note-update-1",
    conversation_id: int = 321,
):
    return await gateway.update_note(
        conversation_id,
        thread_id=thread_id,
        body=body,
        body_format=body_format,
        expected_version=expected_version,
        idempotency_key=key,
        identity=NOTE_IDENTITY,
        request_id=f"request-{key}",
    )


def _mutations(client: FakeNoteClient) -> list:
    return [call for call in client.calls if call[0] != "get"]


# --- Normalized conversation ------------------------------------------------


def test_note_threads_expose_a_version_other_threads_do_not():
    client = _client(extra=[_draft(901, "<p>Draft</p>", "2026-09-30T11:00:00Z")])

    conversation = normalize_conversation(client.conversation)

    by_id = {thread["id"]: thread for thread in conversation["threads"]}
    assert by_id[880]["version"] == _current_version(client)
    assert len(by_id[880]["version"]) == 64
    assert "version" not in by_id[500]
    assert "version" not in by_id[901]
    # The public response model accepts the additive field.
    agent_api.ConversationResponse.model_validate(conversation)


def test_note_version_follows_the_body_and_leaves_conversation_version_alone():
    client = _client()
    threads = client.conversation["_embedded"]["threads"]
    conversation_before = conversation_version(threads)
    original = _current_version(client)

    _thread(client, 880)["body"] = "<p>Edited</p>"

    assert _current_version(client) != original
    _thread(client, 880)["body"] = DECISION_NOTE
    assert _current_version(client) == original
    # The note version is additive: the conversation version hashes what it always did.
    assert conversation_version(threads) == conversation_before
    assert normalize_conversation(client.conversation)["version"] == conversation_before


# --- Gateway ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "closed", "active"])
async def test_update_replaces_an_api_note_and_keeps_the_status(notes_db, status):
    client = _client(status)
    gateway = ConversationGateway(client, db_path=notes_db)
    old_version = _current_version(client)

    result = await _update_note(gateway, expected_version=old_version)

    normalized = normalize_draft_body(SENT_NOTE, "text")
    assert _mutations(client) == [("note_update", 321, 880, normalized)]
    assert _thread(client, 880)["body"] == normalized
    assert client.conversation["status"] == status
    assert result["action"] == "updated"
    assert result["status"] == "updated"
    assert result["warnings"] == []
    assert result["conversation_id"] == 321
    assert result["thread_id"] == 880
    assert result["conversation_status"] == status
    assert result["version"] == _current_version(client)
    assert result["version"] != old_version


@pytest.mark.asyncio
@pytest.mark.parametrize("source_type", ["web", "email", None])
async def test_a_note_a_person_wrote_is_never_edited(notes_db, source_type):
    note = _web_note(881, "<p>Operator's own note</p>")
    note["source"] = {"type": source_type} if source_type else {}
    client = _client(extra=[note])
    gateway = ConversationGateway(client, db_path=notes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _update_note(
            gateway, thread_id=881, expected_version=_current_version(client, 881)
        )

    assert raised.value.code == "note_not_editable"
    assert raised.value.status == 409
    assert raised.value.retryable is False
    assert _mutations(client) == []
    record = get_operation(
        idempotency_key="note-update-1", identity=NOTE_IDENTITY, db_path=notes_db
    )
    assert record.state == "failed"
    assert record.error["code"] == "note_not_editable"


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id", [500, 901, 902, 601])
async def test_a_thread_that_is_not_a_published_note_is_not_found(notes_db, thread_id):
    draft_note = {**_note_thread(902, "<p>Draft note</p>", "2026-09-30T11:00:00Z")}
    draft_note.update(state="draft", draft=True)
    client = _client(
        extra=[_draft(901, "<p>Draft</p>", "2026-09-30T11:00:00Z"), draft_note]
    )
    gateway = ConversationGateway(client, db_path=notes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _update_note(gateway, thread_id=thread_id, expected_version="a" * 64)

    assert raised.value.code == "note_not_found"
    assert raised.value.status == 404
    assert raised.value.retryable is False
    assert _mutations(client) == []


@pytest.mark.asyncio
async def test_stale_expected_version_is_a_conflict_with_the_current_version(notes_db):
    client = _client()
    gateway = ConversationGateway(client, db_path=notes_db)
    current = _current_version(client)

    for _ in range(2):  # the replay returns the same refusal
        with pytest.raises(ConversationGatewayError) as raised:
            await _update_note(gateway, expected_version="b" * 64)
        assert raised.value.code == "note_conflict"
        assert raised.value.status == 409
        assert raised.value.retryable is False
        assert raised.value.details == {"current_version": current, "thread_id": 880}

    assert _mutations(client) == []
    assert _thread(client, 880)["body"] == DECISION_NOTE


@pytest.mark.asyncio
async def test_same_key_replays_without_a_second_edit_and_records_operation(notes_db):
    client = _client()
    gateway = ConversationGateway(client, db_path=notes_db)
    version = _current_version(client)

    first = await _update_note(gateway, expected_version=version, body="Sent sk-secret-value")
    replay = await _update_note(gateway, expected_version=version, body="Sent sk-secret-value")

    assert replay == first
    assert len(_mutations(client)) == 1
    record = get_operation(
        idempotency_key="note-update-1", identity=NOTE_IDENTITY, db_path=notes_db
    )
    assert record.operation_name == "note.update"
    assert record.state == "succeeded"
    assert record.upstream_thread_id == "880"
    assert record.response == {**first, "conversation_id": "321", "thread_id": "880"}
    assert record.request_metadata["expected_thread_id"] == "880"
    events = _stored_supa_events(notes_db)
    assert events == [
        {
            "type": "supa.note.updated",
            "conversation_id": "321",
            "thread_id": "880",
            "source": "supa",
            "summary": {
                "operation_id": "note-update-1",
                "operation_name": "note.update",
                "request_id": "request-note-update-1",
                "result": "updated",
                "status": "pending",
                "token_id": "editor-1",
                "token_name": "Index editor",
            },
        }
    ]
    persisted = json.dumps(events) + json.dumps(record.to_public_dict())
    assert "sk-secret-value" not in persisted


@pytest.mark.asyncio
async def test_same_key_different_body_is_a_conflict(notes_db):
    client = _client()
    gateway = ConversationGateway(client, db_path=notes_db)
    version = _current_version(client)
    await _update_note(gateway, expected_version=version, body="First")

    with pytest.raises(AgentOperationConflictError):
        await _update_note(gateway, expected_version=version, body="Second")

    assert len(_mutations(client)) == 1


@pytest.mark.asyncio
async def test_rejected_edit_is_terminal_and_replay_never_rewrites(notes_db):
    client = _client()
    client.note_update_result = MutationResult(False, 403, detail="upstream_rejected")
    client.apply_note_update = False
    gateway = ConversationGateway(client, db_path=notes_db)
    version = _current_version(client)

    for _ in range(2):
        with pytest.raises(ConversationGatewayError) as raised:
            await _update_note(gateway, expected_version=version)
        assert raised.value.status == 502

    assert len(_mutations(client)) == 1
    assert _stored_supa_events(notes_db) == []


@pytest.mark.asyncio
async def test_submission_exception_without_the_new_body_is_outcome_unknown(notes_db):
    client = _client()
    client.raise_on_note_update = RuntimeError("timeout with private body")
    gateway = ConversationGateway(client, db_path=notes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _update_note(gateway, expected_version=_current_version(client))

    assert raised.value.code == "outcome_unknown"
    record = get_operation(
        idempotency_key="note-update-1", identity=NOTE_IDENTITY, db_path=notes_db
    )
    assert record.state == "outcome_unknown"


@pytest.mark.asyncio
async def test_uncertain_submission_is_confirmed_when_the_readback_shows_the_body(
    notes_db,
):
    client = _client()
    client.note_update_result = MutationResult(
        False, None, detail="request_timeout", outcome_unknown=True
    )
    gateway = ConversationGateway(client, db_path=notes_db)

    result = await _update_note(gateway, expected_version=_current_version(client))

    assert result["status"] == "updated"
    assert result["version"] == _current_version(client)


@pytest.mark.asyncio
async def test_changed_status_after_edit_is_reported_as_warning(notes_db):
    client = _client("pending")
    original = client.update_note_result

    async def reopening_edit(conversation_id, thread_id, text):
        result = await original(conversation_id, thread_id, text)
        client.conversation["status"] = "active"
        return result

    client.update_note_result = reopening_edit
    gateway = ConversationGateway(client, db_path=notes_db)

    result = await _update_note(gateway, expected_version=_current_version(client))

    assert result["status"] == "updated_with_warning"
    assert result["warnings"] == ["status_changed"]


@pytest.mark.asyncio
async def test_confirmed_edit_whose_body_did_not_land_is_a_warning(notes_db):
    client = _client()
    client.apply_note_update = False
    gateway = ConversationGateway(client, db_path=notes_db)

    result = await _update_note(gateway, expected_version=_current_version(client))

    assert result["status"] == "updated_with_warning"
    assert result["warnings"] == ["verification_mismatch"]


@pytest.mark.asyncio
async def test_html_edit_has_no_warning_after_help_scout_reserializes(notes_db):
    client = _client()
    client.store = lambda html: html.replace("\xa0", "&nbsp;").replace("<br/>", "<br>")
    gateway = ConversationGateway(client, db_path=notes_db)

    result = await _update_note(
        gateway,
        expected_version=_current_version(client),
        body="<p>SENT</p><p>&nbsp;</p><p>by Operator<br>today</p>",
        body_format="html",
    )

    assert result["warnings"] == []


@pytest.mark.asyncio
async def test_invalid_thread_id_is_rejected_before_reservation(notes_db):
    client = _client()
    gateway = ConversationGateway(client, db_path=notes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _update_note(gateway, thread_id=0, expected_version="a" * 64)

    assert raised.value.code == "invalid_thread_id"
    assert client.calls == []


# --- Help Scout client -----------------------------------------------------


@pytest.mark.asyncio
async def test_update_note_result_patches_the_thread_text_as_given():
    captured = []

    def handler(request):
        captured.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(204)

    client = await _client_with_transport(handler)
    try:
        result = await client.update_note_result(321, 880, "<p>One</p><p>Two</p>")
    finally:
        await client.close()

    assert result == MutationResult(ok=True, status_code=204)
    assert captured == [
        (
            "PATCH",
            "/v2/conversations/321/threads/880",
            {"op": "replace", "path": "/text", "value": "<p>One</p><p>Two</p>"},
        )
    ]


# --- API -------------------------------------------------------------------


@pytest.fixture
def note_api(tmp_path, monkeypatch):
    # This read includes scheduled-send observation; initialize its real isolated
    # schema instead of depending on a different test having started the app.
    from db.database import init_db
    monkeypatch.setenv("SUPA_DB_PATH", str(tmp_path / "note-api.db"))
    init_db(str(tmp_path / "note-api.db"))
    registry_path = tmp_path / "private-agent-tokens.json"
    write_token_records(
        registry_path,
        [
            _record(NOTE_TOKEN, "token-note-1", "index-editor", ("read", "note")),
            _record(
                WRITER_TOKEN,
                "token-writer-1",
                "writer",
                ("read", "draft", "send", "events", "task"),
            ),
            _record(READ_TOKEN, "token-read-1", "reader", ("read",)),
        ],
    )
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(registry_path))
    monkeypatch.delenv("HELPSCOUT_WEBHOOK_SECRET", raising=False)
    gateway = AsyncMock()
    dependency = agent_api.get_conversation_gateway
    app.dependency_overrides[dependency] = lambda: gateway
    try:
        yield TestClient(app), gateway
    finally:
        app.dependency_overrides.pop(dependency, None)


UPDATE_PATH = "/agent/v1/conversations/321/notes/880"
UPDATE_PAYLOAD = {
    "body": SENT_NOTE,
    "body_format": "text",
    "expected_version": "a" * 64,
    "idempotency_key": "note-edit-api-1",
}
UPDATE_RESULT = {
    "action": "updated",
    "body_sha256": "c" * 64,
    "conversation_id": 321,
    "conversation_status": "pending",
    "status": "updated",
    "thread_id": 880,
    "version": "d" * 64,
    "warnings": [],
}


@pytest.mark.parametrize("headers", [WRITER_HEADERS, READ_HEADERS])
def test_update_route_requires_note_scope(note_api, headers):
    client, gateway = note_api

    response = client.put(UPDATE_PATH, headers=headers, json=UPDATE_PAYLOAD)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "insufficient_scope"
    gateway.update_note.assert_not_awaited()


def test_update_route_forwards_to_gateway_with_read_note_token(note_api):
    client, gateway = note_api
    gateway.update_note.return_value = UPDATE_RESULT

    response = client.put(UPDATE_PATH, headers=NOTE_HEADERS, json=UPDATE_PAYLOAD)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"] == UPDATE_RESULT
    gateway.update_note.assert_awaited_once_with(
        321,
        thread_id=880,
        body=SENT_NOTE,
        body_format="text",
        expected_version="a" * 64,
        idempotency_key="note-edit-api-1",
        identity=AgentIdentity(
            token_id="token-note-1",
            name="index-editor",
            scopes=frozenset({"read", "note"}),
        ),
        request_id=body["request_id"],
    )


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (UPDATE_PATH, {**UPDATE_PAYLOAD, "body_format": "markdown"}),
        (UPDATE_PATH, {**UPDATE_PAYLOAD, "body": ""}),
        (UPDATE_PATH, {**UPDATE_PAYLOAD, "expected_version": "not-a-version"}),
        (UPDATE_PATH, {**UPDATE_PAYLOAD, "expected_version": None}),
        (UPDATE_PATH, {k: v for k, v in UPDATE_PAYLOAD.items() if k != "expected_version"}),
        (UPDATE_PATH, {**UPDATE_PAYLOAD, "status": "active"}),
        ("/agent/v1/conversations/321/notes/0", UPDATE_PAYLOAD),
    ],
)
def test_update_route_rejects_invalid_requests_before_gateway(note_api, path, payload):
    client, gateway = note_api

    response = client.put(path, headers=NOTE_HEADERS, json=payload)

    assert response.status_code == 422
    gateway.update_note.assert_not_awaited()


@pytest.mark.parametrize(
    ("code", "status", "details"),
    [
        ("note_conflict", 409, {"current_version": "e" * 64, "thread_id": 880}),
        ("note_not_editable", 409, {}),
        ("note_not_found", 404, {}),
    ],
)
def test_update_route_maps_note_errors(note_api, code, status, details):
    client, gateway = note_api
    gateway.update_note.side_effect = ConversationGatewayError(
        "private text", code=code, status=status, retryable=False, details=details
    )

    response = client.put(UPDATE_PATH, headers=NOTE_HEADERS, json=UPDATE_PAYLOAD)

    assert response.status_code == status
    error = response.json()["error"]
    assert error["code"] == code
    assert error["retryable"] is False
    assert error["details"] == details
    assert "private text" not in response.text


def test_update_route_maps_idempotency_conflict(note_api):
    client, gateway = note_api
    gateway.update_note.side_effect = AgentOperationConflictError("note-edit-api-1")

    response = client.put(UPDATE_PATH, headers=NOTE_HEADERS, json=UPDATE_PAYLOAD)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "idempotency_conflict"


def test_conversation_get_shows_the_note_version_only_on_notes(note_api):
    client, gateway = note_api
    fake = _client()
    gateway.get.return_value = normalize_conversation(fake.conversation)

    response = client.get("/agent/v1/conversations/321", headers=NOTE_HEADERS)

    assert response.status_code == 200, response.text
    threads = {thread["id"]: thread for thread in response.json()["result"]["threads"]}
    assert threads[880]["version"] == _current_version(fake)
    assert "version" not in threads[500]


def test_capabilities_lists_note_update(note_api):
    client, _gateway = note_api

    response = client.get("/agent/v1/capabilities", headers=NOTE_HEADERS)

    assert "note.update" in response.json()["result"]["commands"]["available"]


# --- CLI -------------------------------------------------------------------


def _run_cli(argv, handler):
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli_main(
        argv,
        transport=httpx.MockTransport(handler),
        environ={"SUPA_URL": "https://supa.example.test", "SUPA_TOKEN": "A" * 43},
        stdin=io.StringIO(),
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def test_cli_note_update_puts_one_edit_request(tmp_path):
    body_file = tmp_path / "note.txt"
    body_file.write_text(SENT_NOTE, encoding="utf-8")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"api_version": "v1", "request_id": "req_edit", "result": UPDATE_RESULT},
        )

    code, stdout, stderr = _run_cli(
        [
            "note",
            "update",
            "321",
            "--thread",
            "880",
            "--body-file",
            str(body_file),
            "--expected-version",
            "a" * 64,
            "--idempotency-key",
            "note-edit-cli-1",
        ],
        handler,
    )

    assert code == 0, stderr
    assert len(seen) == 1
    assert seen[0].method == "PUT"
    assert seen[0].url.path == "/agent/v1/conversations/321/notes/880"
    assert json.loads(seen[0].content) == {
        "body": SENT_NOTE,
        "body_format": "text",
        "expected_version": "a" * 64,
        "idempotency_key": "note-edit-cli-1",
    }
    output = json.loads(stdout)
    assert output["result"]["version"] == "d" * 64
    assert output["result"]["idempotency_key"] == "note-edit-cli-1"


@pytest.mark.parametrize(
    "argv",
    [
        ["note", "update", "321", "--body-file", "x", "--expected-version", "a" * 64],
        ["note", "update", "321", "--thread", "880", "--body-file", "x"],
        ["note", "update", "321", "--thread", "0", "--body-file", "x",
         "--expected-version", "a" * 64],
        ["note", "update", "321", "--thread", "880", "--thread", "881",
         "--body-file", "x", "--expected-version", "a" * 64],
        ["note", "update", "321", "--thread", "880", "--body-file", "x",
         "--expected-version", "short"],
    ],
)
def test_cli_note_update_rejects_bad_arguments_without_a_request(argv):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request expected")

    code, _stdout, _stderr = _run_cli(argv, handler)

    assert code == 2


def test_cli_note_update_maps_conflict_to_exit_5(tmp_path):
    body_file = tmp_path / "note.html"
    body_file.write_text("<p>SENT</p>", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["body_format"] == "html"
        return httpx.Response(
            409,
            json={
                "api_version": "v1",
                "request_id": "req_conflict",
                "error": {
                    "code": "note_conflict",
                    "message": "The note changed after it was read",
                    "retryable": False,
                    "details": {"current_version": "e" * 64, "thread_id": 880},
                },
            },
        )

    code, _stdout, stderr = _run_cli(
        [
            "note",
            "update",
            "321",
            "--thread",
            "880",
            "--body-file",
            str(body_file),
            "--body-format",
            "html",
            "--expected-version",
            "a" * 64,
            "--idempotency-key",
            "note-edit-cli-2",
        ],
        handler,
    )

    assert code == 5
    assert "note_conflict" in stderr
