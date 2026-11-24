"""Internal notes through the Agent API: scope, status preservation, replay."""

from __future__ import annotations

import copy
import hashlib
import io
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

import scripts.manage_agent_tokens as manage_agent_tokens
from api import agent as agent_api
from db.database import init_db
from main import app
from services.agent_auth import (
    ALLOWED_SCOPES,
    AgentIdentity,
    hash_token,
    write_token_records,
)
from services.agent_events import normalize_webhook_event
from services.agent_operations import AgentOperationConflictError, get_operation
from services.conversation_gateway import (
    ConversationGateway,
    ConversationGatewayError,
    normalize_draft_body,
)
from services.helpscout import MutationResult
from supa_cli.main import main as cli_main
from tests.test_agent_writes import (
    FakeDraftClient,
    _conversation,
    _customer,
    _stored_supa_events,
)


NOTE_IDENTITY = AgentIdentity(
    token_id="editor-1",
    name="Index editor",
    scopes=frozenset({"read", "note"}),
)


def _note_thread(thread_id: int, body: str, created_at: str) -> dict:
    return {
        "id": thread_id,
        "type": "note",
        "state": "published",
        "body": body,
        "createdAt": created_at,
        "createdBy": {"type": "user", "id": 9},
        "source": {"type": "api"},
    }


class FakeNoteClient(FakeDraftClient):
    def __init__(self, conversation: dict):
        super().__init__(conversation)
        self.note_result = MutationResult(True, 201, resource_id=880)
        self.apply_note = True
        self.note_at = "2099-01-01T00:00:00Z"
        self.raise_on_note: BaseException | None = None
        self.reopen_without_status = True
        self.note_update_result = MutationResult(True, 204)
        self.apply_note_update = True
        self.raise_on_note_update: BaseException | None = None
        self.note_updated_at = "2099-01-01T00:00:05Z"

    async def update_note_result(
        self,
        conversation_id: int,
        thread_id: int,
        text: str,
    ) -> MutationResult:
        self.calls.append(("note_update", conversation_id, thread_id, text))
        if self.raise_on_note_update is not None:
            raise self.raise_on_note_update
        if self.apply_note_update:
            for thread in self.conversation["_embedded"]["threads"]:
                if thread["id"] == thread_id:
                    thread["body"] = self.store(text)
                    thread["updatedAt"] = self.note_updated_at
        return self.note_update_result

    async def add_note_result(
        self,
        conversation_id: int,
        text: str,
        status: str | None = None,
    ) -> MutationResult:
        self.calls.append(("note", conversation_id, text, status))
        if self.raise_on_note is not None:
            raise self.raise_on_note
        if self.apply_note:
            self.conversation["_embedded"]["threads"].append(
                _note_thread(
                    self.note_result.resource_id or 880,
                    self.store(text),
                    self.note_at,
                )
            )
            # Real Help Scout behaviour: a note without a status reopens the
            # conversation as Active.
            if status:
                self.conversation["status"] = status
            elif self.reopen_without_status:
                self.conversation["status"] = "active"
        return self.note_result


@pytest.fixture
def notes_db(tmp_path):
    path = tmp_path / "notes.db"
    init_db(str(path))
    return path


def _client(status: str = "pending") -> FakeNoteClient:
    conversation = _conversation(
        threads=[_customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")]
    )
    conversation["status"] = status
    return FakeNoteClient(conversation)


async def _add_note(
    gateway: ConversationGateway,
    *,
    body: str = 'Check the appendix entry.',
    body_format: str = "text",
    key: str = "note-add-1",
    conversation_id: int = 321,
):
    return await gateway.add_note(
        conversation_id,
        body=body,
        body_format=body_format,
        idempotency_key=key,
        identity=NOTE_IDENTITY,
        request_id=f"request-{key}",
    )


# --- Gateway ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "closed", "active"])
async def test_note_passes_current_status_so_help_scout_keeps_it(notes_db, status):
    client = _client(status)
    gateway = ConversationGateway(client, db_path=notes_db)

    result = await _add_note(gateway)

    note_calls = [call for call in client.calls if call[0] == "note"]
    assert note_calls == [
        ("note", 321, normalize_draft_body('Check the appendix entry.', "text"), status)
    ]
    assert client.conversation["status"] == status
    assert result["conversation_status"] == status
    assert result["status"] == "added"
    assert result["warnings"] == []
    # A note never drafts, sends, deletes, or retags.
    assert {call[0] for call in client.calls} == {"get", "note"}


@pytest.mark.asyncio
async def test_note_without_a_known_status_is_refused_before_writing(notes_db):
    client = _client()
    client.conversation["status"] = None
    gateway = ConversationGateway(client, db_path=notes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _add_note(gateway)

    assert raised.value.code == "invalid_conversation"
    assert not [call for call in client.calls if call[0] == "note"]
    record = get_operation(
        idempotency_key="note-add-1", identity=NOTE_IDENTITY, db_path=notes_db
    )
    assert record.state == "failed"


@pytest.mark.asyncio
async def test_note_body_is_sanitized_like_drafts(notes_db):
    client = _client()
    gateway = ConversationGateway(client, db_path=notes_db)
    raw = '<p onclick="x()">Next entry<script>drop()</script> <b>chapter seven</b></p>'

    result = await _add_note(gateway, body=raw, body_format="html")

    sent = [call for call in client.calls if call[0] == "note"][0][2]
    assert sent == normalize_draft_body(raw, "html")
    assert "script" not in sent and "onclick" not in sent
    assert result["body_sha256"] == hashlib.sha256(sent.encode()).hexdigest()


@pytest.mark.asyncio
async def test_empty_note_body_is_rejected_without_writing(notes_db):
    client = _client()
    gateway = ConversationGateway(client, db_path=notes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _add_note(gateway, body="<script>x</script>", body_format="html")

    assert raised.value.code == "invalid_body"
    assert not [call for call in client.calls if call[0] == "note"]


@pytest.mark.asyncio
async def test_same_key_replays_without_a_second_note_and_records_operation(
    notes_db,
):
    client = _client()
    gateway = ConversationGateway(client, db_path=notes_db)

    first = await _add_note(gateway, body="Private advice sk-secret-value")
    replay = await _add_note(gateway, body="Private advice sk-secret-value")

    assert replay == first
    assert len([call for call in client.calls if call[0] == "note"]) == 1
    record = get_operation(
        idempotency_key="note-add-1", identity=NOTE_IDENTITY, db_path=notes_db
    )
    assert record.operation_name == "note.add"
    assert record.state == "succeeded"
    assert record.upstream_thread_id == "880"
    assert record.response == {
        **first,
        "conversation_id": "321",
        "thread_id": "880",
    }
    assert record.request_metadata["conversation_status"] == "pending"
    events = _stored_supa_events(notes_db)
    assert events == [
        {
            "type": "supa.note.added",
            "conversation_id": "321",
            "thread_id": "880",
            "source": "supa",
            "summary": {
                "operation_id": "note-add-1",
                "operation_name": "note.add",
                "request_id": "request-note-add-1",
                "result": "added",
                "status": "pending",
                "token_id": "editor-1",
                "token_name": "Index editor",
            },
        }
    ]
    persisted = json.dumps(events) + json.dumps(record.to_public_dict())
    assert "Private advice" not in persisted
    assert "sk-secret-value" not in persisted


@pytest.mark.asyncio
async def test_same_key_different_body_is_a_conflict(notes_db):
    client = _client()
    gateway = ConversationGateway(client, db_path=notes_db)
    await _add_note(gateway, body="First")

    with pytest.raises(AgentOperationConflictError):
        await _add_note(gateway, body="Second")

    assert len([call for call in client.calls if call[0] == "note"]) == 1


@pytest.mark.asyncio
async def test_rejected_note_is_terminal_and_replay_never_rewrites(notes_db):
    client = _client()
    client.note_result = MutationResult(False, 403, detail="upstream_rejected")
    client.apply_note = False
    gateway = ConversationGateway(client, db_path=notes_db)

    for _ in range(2):
        with pytest.raises(ConversationGatewayError) as raised:
            await _add_note(gateway)
        assert raised.value.status == 502

    assert len([call for call in client.calls if call[0] == "note"]) == 1
    assert _stored_supa_events(notes_db) == []


@pytest.mark.asyncio
async def test_submission_exception_without_new_note_is_outcome_unknown(notes_db):
    client = _client()
    client.raise_on_note = RuntimeError("timeout with private body")
    gateway = ConversationGateway(client, db_path=notes_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _add_note(gateway)

    assert raised.value.code == "outcome_unknown"
    record = get_operation(
        idempotency_key="note-add-1", identity=NOTE_IDENTITY, db_path=notes_db
    )
    assert record.state == "outcome_unknown"


@pytest.mark.asyncio
async def test_success_without_resource_id_reconciles_one_exact_new_note(notes_db):
    client = _client()
    client.note_result = MutationResult(True, 201, resource_id=None)
    gateway = ConversationGateway(client, db_path=notes_db)

    result = await _add_note(gateway)

    assert result["thread_id"] == 880
    assert result["status"] == "added"


@pytest.mark.asyncio
async def test_changed_status_after_note_is_reported_as_warning(notes_db):
    client = _client("pending")
    original = client.add_note_result

    async def reopening_note(conversation_id, text, status=None):
        result = await original(conversation_id, text, status)
        client.conversation["status"] = "active"
        return result

    client.add_note_result = reopening_note
    gateway = ConversationGateway(client, db_path=notes_db)

    result = await _add_note(gateway)

    assert result["status"] == "added_with_warning"
    assert result["warnings"] == ["status_changed"]


# --- Scopes ----------------------------------------------------------------


def test_note_scope_is_known_and_separate_from_writes():
    assert "note" in ALLOWED_SCOPES
    assert manage_agent_tokens._parse_scopes("read,note") == ["note", "read"]


NOTE_TOKEN = "synthetic-note-token"
NOTE_HEADERS = {"Authorization": f"Bearer {NOTE_TOKEN}"}
WRITER_TOKEN = "synthetic-writer-token"
WRITER_HEADERS = {"Authorization": f"Bearer {WRITER_TOKEN}"}
READ_TOKEN = "synthetic-read-only-token"
READ_HEADERS = {"Authorization": f"Bearer {READ_TOKEN}"}


def _record(token: str, token_id: str, name: str, scopes: tuple[str, ...]) -> dict:
    return {
        "token_id": token_id,
        "name": name,
        "hash": hash_token(token),
        "scopes": list(scopes),
        "created_at": "2026-09-30T12:00:00+00:00",
    }


@pytest.fixture
def note_api(tmp_path, monkeypatch):
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


NOTE_PAYLOAD = {
    "body": 'Check the appendix entry.',
    "body_format": "text",
    "idempotency_key": "note-api-1",
}
NOTE_RESULT = {
    "action": "added",
    "body_sha256": "c" * 64,
    "conversation_id": 321,
    "conversation_status": "pending",
    "status": "added",
    "thread_id": 880,
    "warnings": [],
}


@pytest.mark.parametrize("headers", [WRITER_HEADERS, READ_HEADERS])
def test_note_route_requires_note_scope(note_api, headers):
    client, gateway = note_api

    response = client.post(
        "/agent/v1/conversations/321/notes", headers=headers, json=NOTE_PAYLOAD
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "insufficient_scope"
    gateway.add_note.assert_not_awaited()


def test_note_token_cannot_draft_or_send(note_api):
    client, gateway = note_api

    draft = client.put(
        "/agent/v1/conversations/321/draft",
        headers=NOTE_HEADERS,
        json={**NOTE_PAYLOAD, "expected_version": None},
    )
    reply = client.post(
        "/agent/v1/conversations/321/replies",
        headers=NOTE_HEADERS,
        json={**NOTE_PAYLOAD, "expected_latest_thread_id": 40},
    )

    assert draft.status_code == reply.status_code == 403
    gateway.save_draft.assert_not_awaited()
    gateway.send_reply.assert_not_awaited()


def test_note_route_forwards_to_gateway_with_read_note_token(note_api):
    client, gateway = note_api
    gateway.add_note.return_value = NOTE_RESULT

    response = client.post(
        "/agent/v1/conversations/321/notes", headers=NOTE_HEADERS, json=NOTE_PAYLOAD
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result"] == NOTE_RESULT
    gateway.add_note.assert_awaited_once_with(
        321,
        body='Check the appendix entry.',
        body_format="text",
        idempotency_key="note-api-1",
        identity=AgentIdentity(
            token_id="token-note-1",
            name="index-editor",
            scopes=frozenset({"read", "note"}),
        ),
        request_id=body["request_id"],
    )


@pytest.mark.parametrize(
    "payload",
    [
        {**NOTE_PAYLOAD, "body_format": "markdown"},
        {**NOTE_PAYLOAD, "body": ""},
        {**NOTE_PAYLOAD, "idempotency_key": "A" * 43},
        {**NOTE_PAYLOAD, "status": "active"},
        {key: value for key, value in NOTE_PAYLOAD.items() if key != "idempotency_key"},
    ],
)
def test_note_route_rejects_invalid_requests_before_gateway(note_api, payload):
    client, gateway = note_api

    response = client.post(
        "/agent/v1/conversations/321/notes", headers=NOTE_HEADERS, json=payload
    )

    assert response.status_code == 422
    gateway.add_note.assert_not_awaited()


def test_note_route_maps_idempotency_conflict(note_api):
    client, gateway = note_api
    gateway.add_note.side_effect = AgentOperationConflictError("note-api-1")

    response = client.post(
        "/agent/v1/conversations/321/notes", headers=NOTE_HEADERS, json=NOTE_PAYLOAD
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "idempotency_conflict"


def test_capabilities_lists_note_add(note_api):
    client, _gateway = note_api

    response = client.get("/agent/v1/capabilities", headers=NOTE_HEADERS)

    result = response.json()["result"]
    assert "note.add" in result["commands"]["available"]
    assert result["scopes"] == ["note", "read"]


# --- CLI -------------------------------------------------------------------


def test_cli_note_add_posts_one_note_request(tmp_path):
    body_file = tmp_path / "note.txt"
    body_file.write_text('Check the appendix entry.', encoding="utf-8")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"api_version": "v1", "request_id": "req_note", "result": NOTE_RESULT},
        )

    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli_main(
        [
            "note",
            "add",
            "321",
            "--body-file",
            str(body_file),
            "--idempotency-key",
            "note-cli-1",
        ],
        transport=httpx.MockTransport(handler),
        environ={"SUPA_URL": "https://supa.example.test", "SUPA_TOKEN": "A" * 43},
        stdin=io.StringIO(),
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0, stderr.getvalue()
    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/agent/v1/conversations/321/notes"
    assert json.loads(seen[0].content) == {
        "body": 'Check the appendix entry.',
        "body_format": "text",
        "idempotency_key": "note-cli-1",
    }
    output = json.loads(stdout.getvalue())
    assert output["result"]["thread_id"] == 880
    assert output["result"]["idempotency_key"] == "note-cli-1"


def test_cli_note_add_maps_scope_error_to_nonzero_exit(tmp_path):
    body_file = tmp_path / "note.html"
    body_file.write_text("<p>Advice</p>", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["body_format"] == "html"
        return httpx.Response(
            403,
            json={
                "api_version": "v1",
                "request_id": "req_denied",
                "error": {
                    "code": "insufficient_scope",
                    "message": "Insufficient scope",
                    "retryable": False,
                    "details": {},
                },
            },
        )

    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli_main(
        [
            "note",
            "add",
            "321",
            "--body-file",
            str(body_file),
            "--body-format",
            "html",
            "--idempotency-key",
            "note-cli-2",
        ],
        transport=httpx.MockTransport(handler),
        environ={"SUPA_URL": "https://supa.example.test", "SUPA_TOKEN": "A" * 43},
        stdin=io.StringIO(),
        stdout=stdout,
        stderr=stderr,
    )

    assert code != 0
    assert "insufficient_scope" in stderr.getvalue()


# --- Webhook + events ------------------------------------------------------


def _note_webhook(source_type: str) -> dict:
    return {
        "id": 321,
        "number": 1442,
        "status": "pending",
        "subject": "Renewal",
        "primaryCustomer": {"first": "Jane"},
        "createdAt": "2026-09-30T09:00:00Z",
        "updatedAt": "2026-09-30T10:00:00Z",
        "_embedded": {
            "threads": [
                {
                    "id": 880,
                    "type": "note",
                    "state": "published",
                    "body": "<p>Private advice body</p>",
                    "createdAt": "2026-09-30T10:00:00Z",
                    "createdBy": {"type": "user", "id": 9},
                    "source": {"type": source_type},
                },
            ]
        },
    }


@patch("api.webhooks.process_in_background", new_callable=AsyncMock)
def test_api_sourced_note_webhook_never_triggers_supa(mock_process, tmp_path, monkeypatch):
    db_path = str(tmp_path / "webhook.db")
    init_db(db_path)
    monkeypatch.setattr("api.webhooks.DEFAULT_DB_PATH", db_path)
    monkeypatch.delenv("HELPSCOUT_WEBHOOK_SECRET", raising=False)

    response = TestClient(app).post(
        "/webhook/helpscout",
        json=_note_webhook("api"),
        headers={"X-HelpScout-Event": "convo.note.created"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert "api" in response.json()["reason"].lower()
    mock_process.assert_not_called()


@pytest.mark.parametrize("source_type", ["api", "web", "email"])
def test_note_created_event_summary_carries_source_type(source_type):
    event = normalize_webhook_event("convo.note.created", _note_webhook(source_type))

    assert event.type == "helpscout.note.created"
    assert event.summary["source_type"] == source_type
    assert "Private advice body" not in json.dumps(event.summary)


def test_non_note_event_summary_has_no_source_type():
    payload = copy.deepcopy(_note_webhook("email"))
    thread = payload["_embedded"]["threads"][0]
    thread.update({"type": "customer", "createdBy": {"type": "customer", "id": 4}})

    event = normalize_webhook_event("convo.customer.reply.created", payload)

    assert "source_type" not in event.summary


@pytest.mark.asyncio
async def test_html_note_has_no_warning_after_help_scout_reserializes(notes_db):
    client = _client()
    client.store = lambda html: html.replace("\xa0", "&nbsp;").replace("<br/>", "<br>")
    gateway = ConversationGateway(client, db_path=notes_db)

    result = await _add_note(
        gateway,
        body='<p>Review</p><p>Appendix ready.</p><p>Chapter<br>seven</p>',
        body_format="html",
    )

    assert result.get("warnings", []) == []


# --- Conversation state: read and changed as data, in place of tag requests to a person ---

STATE = {"handling": "agents", "campaign": "journal-notes", "stage": "feedback", "waiting_on": None, "reminders": {"state": "off"}}


def test_state_is_read_with_the_read_scope(note_api):
    client, gateway = note_api
    gateway.client.ledger.return_value = {"conversation": 321, "state": STATE, "events": []}

    response = client.get("/agent/v1/conversations/321/state", headers=READ_HEADERS)

    assert response.status_code == 200
    assert response.json()["result"]["state"]["stage"] == "feedback"
    gateway.client.ledger.assert_awaited_once_with(321)
    gateway.client.ledger.return_value = None
    unavailable = client.get("/agent/v1/conversations/321/state", headers=READ_HEADERS)
    assert (unavailable.status_code, unavailable.json()["error"]["code"], unavailable.json()["error"]["retryable"]) == (503, "state_unavailable", True)
    gateway.client.ledger.return_value = {}
    missing = client.get("/agent/v1/conversations/321/state", headers=READ_HEADERS)
    assert (missing.status_code, missing.json()["error"]["code"], missing.json()["error"]["retryable"]) == (404, "not_found", False)


def test_state_change_needs_the_note_scope_and_names_the_agent(note_api):
    client, gateway = note_api
    gateway.client.change_state.return_value = (200, STATE)
    payload = {"changes": {"stage": "feedback", "waiting_on": "nobody"}, "reason": "The reader will check the appendix."}

    for headers in (WRITER_HEADERS, READ_HEADERS):
        assert client.post("/agent/v1/conversations/321/state", headers=headers, json=payload).status_code == 403
    gateway.client.change_state.assert_not_awaited()

    response = client.post("/agent/v1/conversations/321/state", headers=NOTE_HEADERS, json=payload)

    assert response.status_code == 200
    assert response.json()["result"] == {"conversation_id": 321, "changes": payload["changes"], "state": STATE}
    gateway.client.change_state.assert_awaited_once_with(
        321, {"stage": "feedback", "waiting_on": "nobody"}, actor="index-editor", reason="The reader will check the appendix.")


@pytest.mark.parametrize("changes", [
    {}, {"handling": "agents"}, {"waiting_on": "unknown-person"}, {"reminders": "maybe"},
    {"tags": ["x"]}, {"paid": "yes"}, {"unrecognized_field": "extra"},
])
def test_state_change_refuses_what_an_agent_may_not_set(note_api, changes):
    client, gateway = note_api

    response = client.post("/agent/v1/conversations/321/state", headers=NOTE_HEADERS, json={"changes": changes})

    assert response.status_code in (400, 422)
    gateway.client.change_state.assert_not_awaited()


@pytest.mark.parametrize("upstream,status,code,retryable", [
    ((409, "Only the mailbox owner hands a conversation back to the agents"), 409, "state_refused", False),
    ((400, "This conversation belongs to no campaign"), 400, "invalid_state", False),
    ((404, "Not found"), 404, "not_found", False),
    ((None, "The mailbox could not be reached."), 503, "state_unavailable", True),
    ((500, "x"), 503, "state_unavailable", True),
    ((401, "x"), 502, "state_unavailable", False),
    ((422, "x"), 502, "state_unavailable", False),
])
def test_state_change_reports_the_mailbox_s_refusal(note_api, upstream, status, code, retryable):
    client, gateway = note_api
    gateway.client.change_state.return_value = upstream

    response = client.post("/agent/v1/conversations/321/state", headers=NOTE_HEADERS, json={"changes": {"reminders": "off"}})

    assert (response.status_code, response.json()["error"]["code"], response.json()["error"]["retryable"]) == (status, code, retryable)
    assert upstream[1] in response.json()["error"]["message"]


@pytest.mark.parametrize("changes", [{"waiting_on": "customer"}, {"reminders": "on"}, {"stage": "feedback", "waiting_on": "customer"}])
def test_a_change_that_can_lead_to_a_customer_email_needs_the_send_scope(note_api, changes):
    client, gateway = note_api

    response = client.post("/agent/v1/conversations/321/state", headers=NOTE_HEADERS, json={"changes": changes})

    assert (response.status_code, response.json()["error"]["code"]) == (403, "insufficient_scope")
    gateway.client.change_state.assert_not_awaited()


def test_a_fault_while_changing_state_is_an_answer_not_a_crash(note_api):
    client, gateway = note_api
    gateway.client.change_state.side_effect = KeyError("access_token")

    response = client.post("/agent/v1/conversations/321/state", headers=NOTE_HEADERS, json={"changes": {"reminders": "off"}})

    assert (response.status_code, response.json()["error"]["code"], response.json()["error"]["retryable"]) == (503, "state_unavailable", True)


@pytest.mark.asyncio
async def test_a_note_written_through_the_api_names_the_agent_that_wrote_it(notes_db):
    client = _client()
    gateway = ConversationGateway(client, db_path=notes_db)

    await _add_note(gateway)

    (_kind, _conversation, text, _status), = [call for call in client.calls if call[0] == "note"]
    assert text.event == {"actor": NOTE_IDENTITY.name}
    assert text == normalize_draft_body('Check the appendix entry.', "text")


@pytest.mark.parametrize("headers", [WRITER_HEADERS, READ_HEADERS])
def test_asking_the_owner_needs_the_note_scope(note_api, headers):
    client, gateway = note_api

    asked = client.post("/agent/v1/conversations/321/requests", headers=headers,
                        json={"text": "Confirm the price.", "idempotency_key": "ask-1"})
    closed = client.post("/agent/v1/conversations/321/requests/7/done", headers=headers, json={})

    assert asked.status_code == closed.status_code == 403
    assert asked.json()["error"]["code"] == "insufficient_scope"
    assert gateway.mock_calls == []
