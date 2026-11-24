"""Per-token conversation allowlists: registry, admin script, API, gateway, events.

A token restricted to conversations {A, B} must never read, write, lease events
for, or start tasks on any other conversation -- including through a merged id
that resolves elsewhere -- and must be refused before any upstream call, lock,
or operation reservation.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock

import scripts.manage_agent_tokens as manage_agent_tokens
from api import agent as agent_api
from db.database import init_db
from main import app
from services.agent_auth import (
    AgentAuthError,
    AgentIdentity,
    AgentTokenRegistry,
    AgentTokenRegistryError,
    hash_token,
    load_token_records,
    write_token_records,
)
from services.agent_events import append_supa_event, lease_next_event
from services.agent_operations import AgentOperationNotFoundError, get_operation
from services.conversation_gateway import ConversationGateway, ConversationGatewayError
from supa_cli.main import main as cli_main
from tests.test_agent_api import _normalized_conversation
from tests.test_agent_notes import FakeNoteClient
from tests.test_agent_writes import _conversation, _customer, _draft


ALLOWED = 321
ALSO_ALLOWED = 322
OTHER = 999

CASE_TOKEN = "synthetic-case-token"
CASE_HEADERS = {"Authorization": f"Bearer {CASE_TOKEN}"}
FULL_TOKEN = "synthetic-full-token"
FULL_HEADERS = {"Authorization": f"Bearer {FULL_TOKEN}"}
ALL_SCOPES = ("read", "draft", "send", "note", "events", "task")

CASE_IDENTITY = AgentIdentity(
    token_id="token-case-1",
    name="case-004",
    scopes=frozenset(ALL_SCOPES),
    conversations=frozenset({ALLOWED, ALSO_ALLOWED}),
)


def _record(
    token: str,
    token_id: str,
    name: str,
    scopes: tuple[str, ...] = ALL_SCOPES,
    **extra,
) -> dict:
    record = {
        "token_id": token_id,
        "name": name,
        "hash": hash_token(token),
        "scopes": list(scopes),
        "created_at": "2026-09-30T12:00:00+00:00",
    }
    record.update(extra)
    return record


def _write_raw(path, records) -> None:
    path.write_text(json.dumps({"tokens": records}), encoding="utf-8")
    path.chmod(0o600)


# --- Registry ----------------------------------------------------------------


def test_registry_loads_allowlist_into_identity(tmp_path):
    path = tmp_path / "tokens.json"
    _write_raw(
        path,
        [_record(CASE_TOKEN, "token-case-1", "case-004", conversations=[322, 321])],
    )

    identity = AgentTokenRegistry(path).authenticate(CASE_TOKEN)

    assert identity.conversations == frozenset({321, 322})
    assert identity.is_conversation_restricted
    assert identity.allows_conversation(321)
    assert identity.allows_conversation("322")
    assert not identity.allows_conversation(999)
    assert not identity.allows_conversation(True)
    assert not identity.allows_conversation(None)
    assert not identity.allows_conversation("321abc")


@pytest.mark.parametrize("extra", [{}, {"conversations": None}])
def test_absent_or_null_allowlist_is_unrestricted(tmp_path, extra):
    path = tmp_path / "tokens.json"
    _write_raw(path, [_record(FULL_TOKEN, "token-full-1", "full", **extra)])

    identity = AgentTokenRegistry(path).authenticate(FULL_TOKEN)

    assert identity.conversations is None
    assert not identity.is_conversation_restricted
    assert identity.allows_conversation(OTHER)
    # A null field is normalized away on the next write.
    records = load_token_records(path)
    write_token_records(path, records)
    assert "conversations" not in json.loads(path.read_text())["tokens"][0]


@pytest.mark.parametrize(
    "conversations",
    [
        [],
        [0],
        [-5],
        ["321"],
        [True],
        [321, 321],
        [1.5],
        "321",
        {"321": True},
        [1 << 63],
        list(range(1, 102)),
    ],
)
def test_invalid_allowlist_fails_closed(tmp_path, conversations):
    path = tmp_path / "tokens.json"
    _write_raw(
        path,
        [_record(CASE_TOKEN, "token-case-1", "case", conversations=conversations)],
    )

    with pytest.raises(AgentAuthError):
        AgentTokenRegistry(path).authenticate(CASE_TOKEN)
    with pytest.raises(AgentTokenRegistryError):
        load_token_records(path)
    with pytest.raises(AgentTokenRegistryError):
        write_token_records(
            tmp_path / "other.json",
            [_record(CASE_TOKEN, "token-case-1", "case", conversations=conversations)],
        )


def test_write_normalizes_allowlist_sorted(tmp_path):
    path = tmp_path / "tokens.json"
    write_token_records(
        path, [_record(CASE_TOKEN, "token-case-1", "case", conversations=[9, 3, 7])]
    )

    assert json.loads(path.read_text())["tokens"][0]["conversations"] == [3, 7, 9]


# --- Admin script -------------------------------------------------------------


def test_script_create_set_and_list_conversations(tmp_path, monkeypatch, capsys):
    path = tmp_path / "agent-tokens.json"
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(path))

    assert manage_agent_tokens.main(
        [
            "create",
            "--name",
            "case-004",
            "--scopes",
            "read,draft,send",
            "--conversations",
            "602, 601",
        ]
    ) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["conversations"] == [601, 602]
    stored = json.loads(path.read_text())["tokens"][0]
    assert stored["conversations"] == [601, 602]
    original_hash = stored["hash"]

    assert manage_agent_tokens.main(
        ["set-conversations", "--name", "case-004", "--conversations", "602"]
    ) == 0
    assert json.loads(capsys.readouterr().out) == {
        "name": "case-004",
        "conversations": [602],
    }
    stored = json.loads(path.read_text())["tokens"][0]
    assert stored["conversations"] == [602]
    # Changing the list never rotates the secret.
    assert stored["hash"] == original_hash
    identity = AgentTokenRegistry(path).authenticate(created["token"])
    assert identity.conversations == frozenset({602})

    assert manage_agent_tokens.main(["list"]) == 0
    listed = json.loads(capsys.readouterr().out)["tokens"]
    assert listed[0]["conversations"] == [602]
    assert created["token"] not in json.dumps(listed)

    assert manage_agent_tokens.main(
        ["set-conversations", "--name", "case-004", "--unrestricted"]
    ) == 0
    capsys.readouterr()
    assert "conversations" not in json.loads(path.read_text())["tokens"][0]
    assert AgentTokenRegistry(path).authenticate(created["token"]).conversations is None


def test_script_create_without_conversations_is_unrestricted(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "agent-tokens.json"
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(path))

    assert manage_agent_tokens.main(["create", "--name", "laptop", "--scopes", "read"]) == 0
    assert json.loads(capsys.readouterr().out)["conversations"] is None
    assert "conversations" not in json.loads(path.read_text())["tokens"][0]
    assert manage_agent_tokens.main(["list"]) == 0
    assert json.loads(capsys.readouterr().out)["tokens"][0]["conversations"] is None


@pytest.mark.parametrize("value", ["", " , ", "abc", "0", "-3", "12,12", "12,,13", "1e3"])
def test_script_rejects_invalid_conversation_lists(tmp_path, monkeypatch, value):
    path = tmp_path / "agent-tokens.json"
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(path))

    with pytest.raises(SystemExit) as raised:
        manage_agent_tokens.main(
            ["create", "--name", "case", "--scopes", "read", "--conversations", value]
        )
    assert raised.value.code == 1
    assert not path.exists()


def test_script_set_conversations_requires_existing_name_and_one_target(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "agent-tokens.json"
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(path))
    assert manage_agent_tokens.main(["create", "--name", "case", "--scopes", "read"]) == 0
    capsys.readouterr()
    original = path.read_bytes()

    with pytest.raises(SystemExit) as missing:
        manage_agent_tokens.main(
            ["set-conversations", "--name", "nobody", "--conversations", "5"]
        )
    assert missing.value.code == 1
    with pytest.raises(SystemExit) as neither:
        manage_agent_tokens.main(["set-conversations", "--name", "case"])
    assert neither.value.code == 2
    with pytest.raises(SystemExit) as both:
        manage_agent_tokens.main(
            ["set-conversations", "--name", "case", "--conversations", "5", "--unrestricted"]
        )
    assert both.value.code == 2
    assert path.read_bytes() == original


# --- API ---------------------------------------------------------------------


@pytest.fixture
def allowlist_api(tmp_path, monkeypatch):
    registry_path = tmp_path / "private-agent-tokens.json"
    write_token_records(
        registry_path,
        [
            _record(
                CASE_TOKEN,
                "token-case-1",
                "case-004",
                conversations=[ALLOWED, ALSO_ALLOWED],
            ),
            _record(FULL_TOKEN, "token-full-1", "full"),
        ],
    )
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(registry_path))
    monkeypatch.delenv("HELPSCOUT_WEBHOOK_SECRET", raising=False)
    db_path = str(tmp_path / "allowlist.db")
    init_db(db_path)
    monkeypatch.setattr(agent_api, "get_supa_db_path", lambda: db_path)
    scheduled: list[str] = []
    monkeypatch.setattr(agent_api, "schedule_agent_run", scheduled.append)
    gateway = AsyncMock()
    dependency = agent_api.get_conversation_gateway
    app.dependency_overrides[dependency] = lambda: gateway
    try:
        yield TestClient(app), gateway, db_path, scheduled
    finally:
        app.dependency_overrides.pop(dependency, None)


WRITE_BODY = {"body": "Hello", "body_format": "text"}

# (method, path template, json payload, gateway method)
CONVERSATION_ENDPOINTS = [
    ("get", "/agent/v1/conversations/{id}", None, "get"),
    ("get", "/agent/v1/conversations/{id}/draft", None, "get"),
    (
        "put",
        "/agent/v1/conversations/{id}/draft",
        {**WRITE_BODY, "expected_version": None, "idempotency_key": "k-draft"},
        "save_draft",
    ),
    (
        "post",
        "/agent/v1/conversations/{id}/draft/send",
        {
            "expected_version": "a" * 64,
            "expected_latest_thread_id": 40,
            "idempotency_key": "k-send",
        },
        "send_draft",
    ),
    (
        "post",
        "/agent/v1/conversations/{id}/replies",
        {**WRITE_BODY, "expected_latest_thread_id": 40, "idempotency_key": "k-reply"},
        "send_reply",
    ),
    (
        "post",
        "/agent/v1/conversations/{id}/notes",
        {**WRITE_BODY, "idempotency_key": "k-note"},
        "add_note",
    ),
    (
        "put",
        "/agent/v1/conversations/{id}/notes/880",
        {**WRITE_BODY, "expected_version": "a" * 64, "idempotency_key": "k-note-edit"},
        "update_note",
    ),
    ("get", "/agent/v1/conversations/{id}/attachments/701", None, "attachment"),
]


def _call(client, method, path, payload, headers):
    kwargs = {"headers": headers}
    if payload is not None:
        kwargs["json"] = payload
    return getattr(client, method)(path, **kwargs)


def _assert_not_allowed(response) -> None:
    assert response.status_code == 403
    error = response.json()["error"]
    assert error["code"] == "conversation_not_allowed"
    assert error["retryable"] is False
    assert str(OTHER) not in error["message"]


@pytest.mark.parametrize(("method", "template", "payload", "gateway_method"), CONVERSATION_ENDPOINTS)
def test_restricted_token_is_refused_other_conversation_before_gateway(
    allowlist_api, method, template, payload, gateway_method
):
    client, gateway, _db, _scheduled = allowlist_api

    response = _call(client, method, template.format(id=OTHER), payload, CASE_HEADERS)

    _assert_not_allowed(response)
    # Refused before any upstream call, lock, or operation reservation.
    assert gateway.mock_calls == []


@pytest.mark.parametrize(("method", "template", "payload", "gateway_method"), CONVERSATION_ENDPOINTS)
def test_unrestricted_token_still_reaches_any_conversation(
    allowlist_api, method, template, payload, gateway_method
):
    client, gateway, _db, _scheduled = allowlist_api
    gateway.get.return_value = _conversation_result(OTHER, OTHER)

    _call(client, method, template.format(id=OTHER), payload, FULL_HEADERS)

    getattr(gateway, gateway_method).assert_awaited_once()
    assert getattr(gateway, gateway_method).await_args.args[0] == OTHER


def _conversation_result(conversation_id: int, requested_id: int) -> dict:
    result = _normalized_conversation()
    result.update(
        id=conversation_id,
        resolved_id=conversation_id,
        requested_id=requested_id,
        was_merged=conversation_id != requested_id,
        help_scout_url=f"https://secure.helpscout.net/conversation/{conversation_id}",
    )
    return result


def test_restricted_token_reads_allowed_conversation_and_draft(allowlist_api):
    client, gateway, _db, _scheduled = allowlist_api
    gateway.get.return_value = _conversation_result(ALLOWED, ALLOWED)

    conversation = client.get(f"/agent/v1/conversations/{ALLOWED}", headers=CASE_HEADERS)
    draft = client.get(f"/agent/v1/conversations/{ALLOWED}/draft", headers=CASE_HEADERS)

    assert conversation.status_code == 200
    assert conversation.json()["result"]["id"] == ALLOWED
    assert draft.status_code == 200


@pytest.mark.parametrize("path", ["/agent/v1/conversations/{id}", "/agent/v1/conversations/{id}/draft"])
def test_merged_id_resolving_outside_the_list_is_refused_on_read(allowlist_api, path):
    client, gateway, _db, _scheduled = allowlist_api
    # The allowed id was merged into a conversation the token does not hold.
    gateway.get.return_value = _conversation_result(OTHER, ALLOWED)

    response = client.get(path.format(id=ALLOWED), headers=CASE_HEADERS)

    _assert_not_allowed(response)
    assert "Synthetic conversation" not in response.text


def test_restricted_writes_pass_the_restricted_identity_to_the_gateway(allowlist_api):
    client, gateway, _db, _scheduled = allowlist_api
    gateway.add_note.return_value = {
        "action": "added",
        "body_sha256": "c" * 64,
        "conversation_id": ALLOWED,
        "conversation_status": "pending",
        "status": "added",
        "thread_id": 880,
        "warnings": [],
    }

    response = client.post(
        f"/agent/v1/conversations/{ALLOWED}/notes",
        headers=CASE_HEADERS,
        json={**WRITE_BODY, "idempotency_key": "k-note"},
    )

    assert response.status_code == 200
    identity = gateway.add_note.await_args.kwargs["identity"]
    assert identity.conversations == frozenset({ALLOWED, ALSO_ALLOWED})


def test_restricted_attachment_passes_identity_for_resolved_check(allowlist_api):
    client, gateway, _db, _scheduled = allowlist_api

    client.get(f"/agent/v1/conversations/{ALLOWED}/attachments/701", headers=CASE_HEADERS)

    gateway.attachment.assert_awaited_once()
    assert gateway.attachment.await_args.args == (ALLOWED, 701)
    assert gateway.attachment.await_args.kwargs["identity"].conversations == frozenset(
        {ALLOWED, ALSO_ALLOWED}
    )


def test_list_and_search_are_filtered_to_allowed_conversations(allowlist_api):
    client, gateway, _db, _scheduled = allowlist_api
    gateway.list.return_value = {
        "conversations": [
            _conversation_result(ALLOWED, ALLOWED),
            _conversation_result(OTHER, OTHER),
            _conversation_result(ALSO_ALLOWED, ALSO_ALLOWED),
        ],
        "next_cursor": "cursor-2",
    }

    restricted = client.get(
        "/agent/v1/conversations", params={"query": "Field Journal"}, headers=CASE_HEADERS
    )
    full = client.get("/agent/v1/conversations", params={"query": "Field Journal"}, headers=FULL_HEADERS)

    assert restricted.status_code == 200
    assert [c["id"] for c in restricted.json()["result"]["conversations"]] == [
        ALLOWED,
        ALSO_ALLOWED,
    ]
    assert restricted.json()["result"]["next_cursor"] == "cursor-2"
    assert str(OTHER) not in json.dumps(restricted.json()["result"])
    assert [c["id"] for c in full.json()["result"]["conversations"]] == [
        ALLOWED,
        OTHER,
        ALSO_ALLOWED,
    ]


def test_capabilities_show_the_conversation_restriction(allowlist_api):
    client, _gateway, _db, _scheduled = allowlist_api

    restricted = client.get("/agent/v1/capabilities", headers=CASE_HEADERS).json()["result"]
    full = client.get("/agent/v1/capabilities", headers=FULL_HEADERS).json()["result"]

    assert restricted["conversations"] == [ALLOWED, ALSO_ALLOWED]
    assert full["conversations"] is None


@pytest.mark.parametrize("task_name", ["ticket.evaluate", "ticket.process"])
def test_task_start_is_restricted_to_allowed_conversations(allowlist_api, task_name):
    client, _gateway, db_path, scheduled = allowlist_api

    refused = client.post(
        "/agent/v1/tasks",
        headers=CASE_HEADERS,
        json={
            "task_name": task_name,
            "arguments": {"conversation_id": OTHER},
            "idempotency_key": "task-other",
        },
    )
    refused_as_text = client.post(
        "/agent/v1/tasks",
        headers=CASE_HEADERS,
        json={
            "task_name": task_name,
            "arguments": {"conversation_id": str(OTHER)},
            "idempotency_key": "task-other-text",
        },
    )
    allowed = client.post(
        "/agent/v1/tasks",
        headers=CASE_HEADERS,
        json={
            "task_name": task_name,
            "arguments": {"conversation_id": ALLOWED},
            "idempotency_key": "task-allowed",
        },
    )
    unrestricted = client.post(
        "/agent/v1/tasks",
        headers=FULL_HEADERS,
        json={
            "task_name": task_name,
            "arguments": {"conversation_id": OTHER},
            "idempotency_key": "task-full",
        },
    )

    _assert_not_allowed(refused)
    _assert_not_allowed(refused_as_text)
    assert allowed.status_code == 200
    assert unrestricted.status_code == 200
    assert len(scheduled) == 2


# --- Events ------------------------------------------------------------------


NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def _append(db_path, conversation_id, key):
    return append_supa_event(
        db_path,
        type="supa.reply.sent",
        conversation_id=conversation_id,
        dedupe_key=key,
        summary={"status": "sent"},
        clock=lambda: NOW,
    )


def test_restricted_event_lease_sees_only_allowed_conversations(tmp_path):
    db_path = tmp_path / "events.db"
    init_db(str(db_path))
    _append(db_path, OTHER, "e1")
    _append(db_path, None, "e2")
    allowed = _append(db_path, ALLOWED, "e3")
    _append(db_path, OTHER, "e4")

    leased = lease_next_event(
        db_path,
        token_id="token-case-1",
        consumer_name="case",
        start="oldest",
        clock=lambda: NOW + timedelta(seconds=1),
        conversations=frozenset({ALLOWED}),
    )
    unrestricted = lease_next_event(
        db_path,
        token_id="token-full-1",
        consumer_name="router",
        start="oldest",
        clock=lambda: NOW + timedelta(seconds=1),
    )

    assert leased["event_id"] == allowed["event_id"]
    assert leased["conversation_id"] == str(ALLOWED)
    assert unrestricted["conversation_id"] == str(OTHER)


def test_restricted_event_lease_drops_a_lease_no_longer_allowed(tmp_path):
    db_path = tmp_path / "events.db"
    init_db(str(db_path))
    _append(db_path, OTHER, "e1")
    allowed = _append(db_path, ALLOWED, "e2")
    clock = lambda: NOW + timedelta(seconds=1)  # noqa: E731

    # Leased while the token was unrestricted, then the list was narrowed.
    first = lease_next_event(
        db_path, token_id="token-case-1", consumer_name="case", start="oldest", clock=clock
    )
    assert first["conversation_id"] == str(OTHER)
    narrowed = lease_next_event(
        db_path,
        token_id="token-case-1",
        consumer_name="case",
        clock=clock,
        conversations=frozenset({ALLOWED}),
    )

    assert narrowed["event_id"] == allowed["event_id"]


def test_events_next_endpoint_filters_for_restricted_tokens(allowlist_api, monkeypatch):
    client, _gateway, db_path, _scheduled = allowlist_api
    monkeypatch.setenv("HELPSCOUT_WEBHOOK_SECRET", "synthetic-webhook-secret")
    monkeypatch.setattr(agent_api, "_event_clock", lambda: NOW + timedelta(seconds=1))
    _append(db_path, OTHER, "e1")
    allowed = _append(db_path, ALLOWED, "e2")

    response = client.post(
        "/agent/v1/events/next",
        headers=CASE_HEADERS,
        json={"consumer": "case", "start": "oldest", "wait": 0},
    )

    assert response.status_code == 200
    assert response.json()["result"]["event"]["event_id"] == allowed["event_id"]


# --- Gateway (resolved/merged id) -------------------------------------------


@pytest.fixture
def gateway_db(tmp_path):
    path = tmp_path / "gateway.db"
    init_db(str(path))
    return path


def _merged_client() -> FakeNoteClient:
    """Requesting ALLOWED returns a conversation whose real id is OTHER."""
    conversation = _conversation(
        OTHER,
        requested_id=ALLOWED,
        threads=[_customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")],
    )
    conversation["status"] = "pending"
    return FakeNoteClient(conversation)


WRITE_KINDS = ["save_draft", "send_reply", "send_draft", "add_note", "update_note"]


async def _write(gateway, kind, conversation_id, identity, key):
    common = dict(idempotency_key=key, identity=identity, request_id=f"req-{key}")
    if kind == "save_draft":
        return await gateway.save_draft(
            conversation_id, body="Hi", body_format="text", expected_version=None, **common
        )
    if kind == "send_reply":
        return await gateway.send_reply(
            conversation_id,
            body="Hi",
            body_format="text",
            expected_latest_thread_id=500,
            **common,
        )
    if kind == "send_draft":
        return await gateway.send_draft(
            conversation_id,
            expected_version="a" * 64,
            expected_latest_thread_id=500,
            **common,
        )
    if kind == "update_note":
        return await gateway.update_note(
            conversation_id,
            thread_id=880,
            body="Hi",
            body_format="text",
            expected_version="a" * 64,
            **common,
        )
    return await gateway.add_note(conversation_id, body="Hi", body_format="text", **common)


def _mutations(client) -> list:
    return [call for call in client.calls if call[0] != "get"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", WRITE_KINDS)
async def test_gateway_refuses_merged_id_resolving_outside_the_list(gateway_db, kind):
    client = _merged_client()
    gateway = ConversationGateway(client, db_path=gateway_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _write(gateway, kind, ALLOWED, CASE_IDENTITY, f"merged-{kind}")

    assert raised.value.code == "conversation_not_allowed"
    assert raised.value.status == 403
    assert _mutations(client) == []
    record = get_operation(
        idempotency_key=f"merged-{kind}", identity=CASE_IDENTITY, db_path=gateway_db
    )
    assert record.state == "failed"
    assert record.error["code"] == "conversation_not_allowed"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", WRITE_KINDS)
async def test_gateway_refuses_requested_id_before_reservation(gateway_db, kind):
    client = _merged_client()
    gateway = ConversationGateway(client, db_path=gateway_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await _write(gateway, kind, OTHER, CASE_IDENTITY, f"direct-{kind}")

    assert raised.value.code == "conversation_not_allowed"
    assert client.calls == []
    with pytest.raises(AgentOperationNotFoundError):
        get_operation(
            idempotency_key=f"direct-{kind}", identity=CASE_IDENTITY, db_path=gateway_db
        )


@pytest.mark.asyncio
async def test_gateway_allows_restricted_write_on_allowed_conversation(gateway_db):
    conversation = _conversation(
        ALLOWED, threads=[_customer(500, "<p>Question</p>", "2026-09-25T10:00:00Z")]
    )
    conversation["status"] = "pending"
    client = FakeNoteClient(conversation)
    gateway = ConversationGateway(client, db_path=gateway_db)

    result = await _write(gateway, "add_note", ALLOWED, CASE_IDENTITY, "ok-note")

    assert result["conversation_id"] == ALLOWED
    assert [call[0] for call in _mutations(client)] == ["note"]


@pytest.mark.asyncio
async def test_gateway_attachment_refuses_merged_id_before_download(gateway_db):
    conversation = _conversation(OTHER, requested_id=ALLOWED)
    conversation["_embedded"]["threads"] = [
        {
            **_customer(500, "<p>See</p>", "2026-09-25T10:00:00Z"),
            "_embedded": {"attachments": [{"id": 701, "filename": "a.png"}]},
        }
    ]
    client = FakeNoteClient(conversation)
    client.get_attachment_data = AsyncMock(return_value=(b"x", "image/png"))
    gateway = ConversationGateway(client, db_path=gateway_db)

    with pytest.raises(ConversationGatewayError) as raised:
        await gateway.attachment(ALLOWED, 701, identity=CASE_IDENTITY)
    assert raised.value.code == "conversation_not_allowed"
    client.get_attachment_data.assert_not_awaited()

    with pytest.raises(ConversationGatewayError):
        await gateway.attachment(OTHER, 701, identity=CASE_IDENTITY)
    assert client.calls == [("get", ALLOWED)]

    unrestricted = await gateway.attachment(ALLOWED, 701)
    assert unrestricted.conversation_id == OTHER


# --- CLI -----------------------------------------------------------------------


def _cli(handler, argv):
    import io

    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli_main(
        argv,
        environ={"SUPA_URL": "https://supa.example.test", "SUPA_TOKEN": "A" * 43},
        transport=httpx.MockTransport(handler),
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def test_cli_maps_conversation_not_allowed_to_permission_exit(tmp_path):
    body = tmp_path / "reply.txt"
    body.write_text("Hi", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "api_version": "v1",
                "request_id": "req_x",
                "error": {
                    "code": "conversation_not_allowed",
                    "message": "Token is not allowed to access this conversation",
                    "retryable": False,
                    "details": {},
                },
            },
        )

    code, stdout, stderr = _cli(handler, ["conversation", "get", str(OTHER)])

    assert code == 3
    assert json.loads(stderr)["error"]["code"] == "conversation_not_allowed"


def test_cli_capabilities_shows_conversation_restriction():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "api_version": "v1",
                "request_id": "req_c",
                "result": {"scopes": ["read"], "conversations": [ALLOWED]},
            },
        )

    code, stdout, _stderr = _cli(handler, ["capabilities"])

    assert code == 0
    assert json.loads(stdout)["result"]["conversations"] == [ALLOWED]


# --- Conversation state (the mailbox follows a merge to the surviving conversation) ---


@pytest.mark.parametrize(("method", "payload"), [("get", None), ("post", {"changes": {"waiting_on": "customer"}})])
def test_state_routes_refuse_another_conversation_before_any_upstream_call(allowlist_api, method, payload):
    client, gateway, _db, _scheduled = allowlist_api

    response = _call(client, method, f"/agent/v1/conversations/{OTHER}/state", payload, CASE_HEADERS)

    _assert_not_allowed(response)
    assert gateway.mock_calls == []


@pytest.mark.parametrize(("method", "payload"), [("get", None), ("post", {"changes": {"waiting_on": "customer"}})])
def test_state_routes_refuse_a_merged_id_that_resolves_elsewhere(allowlist_api, method, payload):
    client, gateway, _db, _scheduled = allowlist_api
    gateway.client.ledger.return_value = {"conversation": OTHER, "state": {"waiting_on": "customer"}, "events": []}

    response = _call(client, method, f"/agent/v1/conversations/{ALLOWED}/state", payload, CASE_HEADERS)

    _assert_not_allowed(response)
    gateway.client.change_state.assert_not_awaited()  # Refused before the mailbox changed anything.
    assert "feedback" not in response.text


def test_state_routes_serve_the_case_s_own_conversation(allowlist_api):
    client, gateway, _db, _scheduled = allowlist_api
    gateway.client.ledger.return_value = {"conversation": ALLOWED, "state": {"waiting_on": "customer"}, "events": []}
    gateway.client.change_state.return_value = (200, {"waiting_on": "customer"})

    assert client.get(f"/agent/v1/conversations/{ALLOWED}/state", headers=CASE_HEADERS).json()["result"]["conversation"] == ALLOWED
    changed = client.post(f"/agent/v1/conversations/{ALLOWED}/state", headers=CASE_HEADERS, json={"changes": {"waiting_on": "customer"}, "reason": "Later."})
    assert changed.status_code == 200
    gateway.client.change_state.assert_awaited_once_with(ALLOWED, {"waiting_on": "customer"}, actor="case-004", reason="Later.")
    # An unrestricted token needs no extra read before a change.
    gateway.client.ledger.reset_mock()
    client.post(f"/agent/v1/conversations/{OTHER}/state", headers=FULL_HEADERS, json={"changes": {"waiting_on": "customer"}})
    gateway.client.ledger.assert_not_awaited()


# --- Requests to the owner ---

ASK = {"text": "Confirm whether old captions are kept.", "idempotency_key": "captions-1"}
OWN_LEDGER = {"conversation": ALLOWED, "state": {}, "events": [
    {"id": 41, "kind": "request", "actor": "case-004", "status": "open"},
    {"id": 42, "kind": "request", "actor": "case-099", "status": "open"},
    {"id": 43, "kind": "decision", "actor": "supa", "status": ""}]}


@pytest.mark.parametrize("path", ["/requests", "/requests/41/done"])
def test_request_routes_refuse_another_conversation_a_merged_id_and_an_unreachable_mailbox(allowlist_api, path):
    client, gateway, _db, _scheduled = allowlist_api
    payload = ASK if path == "/requests" else {}

    _assert_not_allowed(_call(client, "post", f"/agent/v1/conversations/{OTHER}{path}", payload, CASE_HEADERS))
    assert gateway.mock_calls == []
    gateway.client.ledger.return_value = {**OWN_LEDGER, "conversation": OTHER}
    _assert_not_allowed(_call(client, "post", f"/agent/v1/conversations/{ALLOWED}{path}", payload, CASE_HEADERS))
    # A per-case token is not waved through when the mailbox cannot say which conversation this is.
    gateway.client.ledger.return_value = None
    assert _call(client, "post", f"/agent/v1/conversations/{ALLOWED}{path}", payload, CASE_HEADERS).status_code in (403, 503)
    gateway.client.add_request.assert_not_awaited()
    gateway.client.close_request.assert_not_awaited()


def test_a_case_agent_asks_the_owner_and_closes_only_its_own_request(allowlist_api):
    from services.helpscout import MutationResult

    client, gateway, _db, _scheduled = allowlist_api
    gateway.client.ledger.return_value = OWN_LEDGER
    gateway.client.add_request.return_value = (201, {"id": 41, "status": "open", "replayed": False})
    gateway.client.close_request.return_value = MutationResult(ok=True, status_code=200)

    asked = client.post(f"/agent/v1/conversations/{ALLOWED}/requests", headers=CASE_HEADERS, json=ASK)

    assert asked.status_code == 200
    assert asked.json()["result"] == {"conversation_id": ALLOWED, "request_id": 41, "status": "open", "replayed": False}
    gateway.client.add_request.assert_awaited_once_with(ALLOWED, "case-004", ASK["text"], "captions-1")
    done = client.post(f"/agent/v1/conversations/{ALLOWED}/requests/41/done", headers=CASE_HEADERS, json={})
    assert done.json()["result"] == {"conversation_id": ALLOWED, "request_id": 41, "status": "done"}
    gateway.client.close_request.assert_awaited_once_with(ALLOWED, 41, "case-004")
    # Another agent's request, something that is no request, an id that does not exist: not closed.
    other = client.post(f"/agent/v1/conversations/{ALLOWED}/requests/42/done", headers=CASE_HEADERS, json={})
    assert (other.status_code, other.json()["error"]["code"]) == (403, "request_not_yours")
    for missing in (43, 999):
        assert client.post(f"/agent/v1/conversations/{ALLOWED}/requests/{missing}/done", headers=CASE_HEADERS, json={}).status_code == 404
    assert gateway.client.close_request.await_count == 1
    # The owner's own token closes any request.
    assert client.post(f"/agent/v1/conversations/{ALLOWED}/requests/42/done", headers=FULL_HEADERS, json={}).status_code == 200
    # A refusal by the mailbox is reported, never as "done".
    gateway.client.close_request.return_value = MutationResult(ok=False, status_code=400)
    refused = client.post(f"/agent/v1/conversations/{ALLOWED}/requests/41/done", headers=FULL_HEADERS, json={})
    assert (refused.status_code, refused.json()["error"]["code"]) == (400, "invalid_request")


def test_a_request_already_closed_is_reported_as_closed_and_a_reused_key_as_a_conflict(allowlist_api):
    client, gateway, _db, _scheduled = allowlist_api
    gateway.client.add_request.return_value = (201, {"id": 41, "status": "done", "replayed": True})

    again = client.post(f"/agent/v1/conversations/{OTHER}/requests", headers=FULL_HEADERS, json=ASK)

    assert again.json()["result"] == {"conversation_id": OTHER, "request_id": 41, "status": "done", "replayed": True}
    gateway.client.add_request.return_value = (409, {"message": "Operation key already used for another record"})
    conflict = client.post(f"/agent/v1/conversations/{OTHER}/requests", headers=FULL_HEADERS, json=ASK)
    error = conflict.json()["error"]
    assert (conflict.status_code, error["code"], error["retryable"]) == (409, "request_conflict", False)
    assert "new key" in error["message"] and "Operation key" not in error["message"]  # The mailbox's own words stay inside.


@pytest.mark.parametrize(("answer", "status", "code", "retryable"), [
    ((400, {}), 400, "invalid_request", False),
    ((404, {}), 404, "not_found", False),
    ((None, {}), 503, "request_unavailable", True),
    ((500, {}), 503, "request_unavailable", True),
    ((201, {}), 503, "request_unavailable", True),
    ((201, {"id": "x"}), 503, "request_unavailable", True),
    ("raises", 503, "request_unavailable", True),
])
def test_a_request_the_mailbox_did_not_record_is_reported_as_such(allowlist_api, answer, status, code, retryable):
    client, gateway, _db, _scheduled = allowlist_api
    if answer == "raises":
        gateway.client.add_request.side_effect = KeyError("access_token")
    else:
        gateway.client.add_request.return_value = answer

    response = client.post(f"/agent/v1/conversations/{OTHER}/requests", headers=FULL_HEADERS, json=ASK)

    error = response.json()["error"]
    assert (response.status_code, error["code"], error["retryable"]) == (status, code, retryable)


@pytest.mark.parametrize("payload", [{}, {"text": "ok"}, {"text": "x", "idempotency_key": "k"}, {"text": "a" * 2001, "idempotency_key": "k"},
                                     {"text": "Confirm.", "idempotency_key": "k", "actor": "operator"}])
def test_a_request_needs_a_text_and_a_key_and_names_nobody_else(allowlist_api, payload):
    client, gateway, _db, _scheduled = allowlist_api

    response = client.post(f"/agent/v1/conversations/{ALLOWED}/requests", headers=FULL_HEADERS, json=payload)

    assert response.status_code in (400, 422)
    gateway.client.add_request.assert_not_awaited()
