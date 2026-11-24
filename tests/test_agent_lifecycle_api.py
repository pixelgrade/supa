"""HTTP lifecycle contract tests; never contact Help Scout."""
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from api import agent as agent_api
from main import app
from services.agent_auth import hash_token, write_token_records
from services.conversation_gateway import ConversationGatewayError
import services.conversation_lifecycle as lifecycle


@pytest.fixture
def client(tmp_path, monkeypatch):
    registry = tmp_path / "fixture-tokens.json"
    records = []
    for name, scopes in [("reader", ["read"]), ("creator", ["create"]),
                         ("operator", ["create", "send", "delete"])]:
        records.append({"token_id": "fixture-" + name, "name": name,
                        "hash": hash_token(name), "scopes": scopes,
                        "created_at": "2026-09-24T12:00:00+00:00"})
    write_token_records(registry, records)
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(registry))
    gateway = AsyncMock()
    app.dependency_overrides[agent_api.get_conversation_gateway] = lambda: gateway
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(agent_api.get_conversation_gateway, None)


def headers(name):
    return {"Authorization": "Bearer " + name}


def payload(**changes):
    return {"mailbox_id": 1, "customer_id": 8, "subject": "Synthetic invitation",
            "body": "Hello", "idempotency_key": "fixture-create", **changes}


def test_create_api_defaults_to_draft(client, monkeypatch):
    create = AsyncMock(return_value={"action": "created", "conversation_id": 321,
                                    "status": "created", "body_sha256": "a" * 64})
    monkeypatch.setattr(lifecycle, "create_conversation", create)
    result = client.post("/agent/v1/conversations", headers=headers("creator"), json=payload())
    assert result.status_code == 200
    assert result.json()["result"]["conversation_id"] == 321
    assert create.call_args.kwargs["draft"] is True
    assert create.call_args.kwargs["body_format"] == "text"


def test_create_api_scope_denied(client, monkeypatch):
    create = AsyncMock()
    monkeypatch.setattr(lifecycle, "create_conversation", create)
    result = client.post("/agent/v1/conversations", headers=headers("reader"), json=payload())
    assert result.status_code == 403
    create.assert_not_awaited()


@pytest.mark.parametrize("changes", [{"mailbox_id": 0}, {"customer_id": -1},
                                      {"subject": ""}, {"body_format": "markdown"},
                                      {"extra": "not allowed"}, {"idempotency_key": ""}])
def test_create_api_invalid_input(client, monkeypatch, changes):
    create = AsyncMock()
    monkeypatch.setattr(lifecycle, "create_conversation", create)
    result = client.post("/agent/v1/conversations", headers=headers("creator"), json=payload(**changes))
    assert result.status_code == 422
    create.assert_not_awaited()


def test_discard_api_success(client, monkeypatch):
    discard = AsyncMock(return_value={"action": "discarded", "conversation_id": 321,
                                     "status": "discarded"})
    monkeypatch.setattr(lifecycle, "discard_conversation", discard)
    result = client.post("/agent/v1/conversations/321/discard", headers=headers("operator"),
                         json={"expected_version": "a" * 64, "idempotency_key": "fixture-discard"})
    assert result.status_code == 200
    assert result.json()["result"]["status"] == "discarded"
    assert discard.call_args.args[1] == 321


def test_discard_api_scope_denied(client, monkeypatch):
    discard = AsyncMock()
    monkeypatch.setattr(lifecycle, "discard_conversation", discard)
    result = client.post("/agent/v1/conversations/321/discard", headers=headers("creator"),
                         json={"expected_version": "a" * 64, "idempotency_key": "fixture-discard"})
    assert result.status_code == 403
    discard.assert_not_awaited()


def test_discard_api_stale_conflict(client, monkeypatch):
    discard = AsyncMock(side_effect=ConversationGatewayError(
        "Synthetic stale snapshot", code="conversation_changed", status=409))
    monkeypatch.setattr(lifecycle, "discard_conversation", discard)
    result = client.post("/agent/v1/conversations/321/discard", headers=headers("operator"),
                         json={"expected_version": "a" * 64, "idempotency_key": "fixture-discard"})
    assert result.status_code == 409


@pytest.mark.parametrize("version", ["", "abc", "A" * 64, "z" * 64])
def test_discard_api_requires_snapshot_version(client, monkeypatch, version):
    discard = AsyncMock()
    monkeypatch.setattr(lifecycle, "discard_conversation", discard)
    result = client.post("/agent/v1/conversations/321/discard", headers=headers("operator"),
                         json={"expected_version": version, "idempotency_key": "fixture-discard"})
    assert result.status_code == 422
    discard.assert_not_awaited()
