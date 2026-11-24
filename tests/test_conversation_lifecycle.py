"""Synthetic-only regression coverage for outbound creation and safe discard."""
import asyncio
import copy
from unittest.mock import AsyncMock

import pytest

from db.database import init_db
from services.agent_auth import AgentIdentity
from services.conversation_gateway import ConversationGateway, ConversationGatewayError, normalize_conversation
from services.conversation_lifecycle import create_conversation, discard_conversation
from services.helpscout import HelpScoutError, MutationResult


def identity(scopes=("create", "delete", "send"), conversations=None):
    return AgentIdentity("synthetic-token", "fixture-agent", frozenset(scopes), conversations)


def raw(*, threads=None, status="draft", cid=321):
    return {"id": cid, "subject": "Synthetic invitation", "status": status,
            "mailboxId": 1, "primaryCustomer": {"id": 8, "type": "customer"},
            "_embedded": {"threads": [] if threads is None else threads}}


def thread(**changes):
    return {"id": 90, "type": "message", "state": "draft", "draft": True,
            "body": "<p>Hello</p>", "createdBy": {"id": 9, "type": "user"}, **changes}


@pytest.fixture
def gateway(tmp_path):
    db_path = str(tmp_path / "lifecycle.db")
    init_db(db_path)
    client = AsyncMock()
    client.create_conversation_result.return_value = MutationResult(True, 201, resource_id=321)
    client.delete_conversation_result.return_value = MutationResult(True, 204)
    return ConversationGateway(client, db_path=db_path)


async def create(gateway, **changes):
    args = dict(mailbox_id=1, customer_id=8, subject="Synthetic invitation", body="Hello",
                body_format="text", draft=True, idempotency_key="synthetic-create",
                identity=identity(), request_id="req_synthetic")
    args.update(changes)
    return await create_conversation(gateway, **args)


async def discard(gateway, source=None, **changes):
    args = dict(expected_version=normalize_conversation(source or raw())["version"],
                idempotency_key="synthetic-discard", identity=identity(), request_id="req_synthetic")
    args.update(changes)
    return await discard_conversation(gateway, 321, **args)


def test_create_verified_and_replayed(gateway):
    gateway.client.get_complete_conversation.return_value = raw(threads=[thread()])
    first = asyncio.run(create(gateway))
    assert first["action"] == "created"
    assert asyncio.run(create(gateway)) == first
    gateway.client.create_conversation_result.assert_awaited_once()
    assert gateway.client.create_conversation_result.call_args.kwargs["draft"] is True


def test_create_verification_allows_routine_activity_entries(gateway):
    gateway.client.get_complete_conversation.return_value = raw(
        threads=[thread(), {"id": 91, "type": "lineitem", "state": None}])
    assert asyncio.run(create(gateway))["status"] == "created"


@pytest.mark.parametrize("who,draft,code", [
    (identity(("read",)), True, "insufficient_scope"),
    (identity(("create",)), False, "insufficient_scope"),
    (identity(conversations=frozenset({321})), True, "conversation_not_allowed"),
])
def test_create_authorization_before_ledger_or_upstream(gateway, who, draft, code):
    gateway._operation_reserve = AsyncMock()
    with pytest.raises(ConversationGatewayError) as error:
        asyncio.run(create(gateway, identity=who, draft=draft))
    assert error.value.code == code
    gateway._operation_reserve.assert_not_awaited()
    gateway.client.create_conversation_result.assert_not_awaited()


def test_send_requires_explicit_authority_and_published_readback(gateway):
    gateway.client.get_complete_conversation.return_value = raw(
        status="pending", threads=[thread(state="published", draft=False)])
    assert asyncio.run(create(gateway, draft=False))["status"] == "sent"
    assert gateway.client.create_conversation_result.call_args.kwargs["draft"] is False


@pytest.mark.parametrize("field,value", [("mailboxId", 99),
    ("primaryCustomer", {"id": 999, "type": "customer"})])
def test_create_verifies_recipient_and_mailbox(gateway, field, value):
    source = raw(threads=[thread()])
    source[field] = value
    gateway.client.get_complete_conversation.return_value = source
    with pytest.raises(ConversationGatewayError) as error:
        asyncio.run(create(gateway))
    assert error.value.code == "outcome_unknown"


@pytest.mark.parametrize("changes", [{"type": "customer"},
    {"createdBy": {"type": "customer"}}, {"state": "published", "draft": True},
    {"state": "draft", "draft": False}])
def test_create_verifies_agent_author_and_state(gateway, changes):
    gateway.client.get_complete_conversation.return_value = raw(threads=[thread(**changes)])
    with pytest.raises(ConversationGatewayError) as error:
        asyncio.run(create(gateway))
    assert error.value.code == "outcome_unknown"


@pytest.mark.parametrize("kind", ["transport", "5xx", "missing_id", "readback", "mismatch"])
def test_create_uncertainty_never_resubmits(gateway, kind):
    gateway.client.get_complete_conversation.return_value = raw(threads=[thread()])
    if kind == "transport":
        gateway.client.create_conversation_result.side_effect = ConnectionError("synthetic timeout")
    elif kind == "5xx":
        gateway.client.create_conversation_result.return_value = MutationResult(False, 503)
    elif kind == "missing_id":
        gateway.client.create_conversation_result.return_value = MutationResult(True, 201)
    elif kind == "readback":
        gateway.client.get_complete_conversation.side_effect = HelpScoutError("synthetic", status=503)
    else:
        gateway.client.get_complete_conversation.return_value = raw(threads=[thread(body="wrong")])
    for _ in range(2):
        with pytest.raises(ConversationGatewayError):
            asyncio.run(create(gateway))
    gateway.client.create_conversation_result.assert_awaited_once()


@pytest.mark.parametrize("threads", [[], [thread()], [thread(type="reply")]])
def test_discard_only_unsent_and_replay(gateway, threads):
    source = raw(threads=threads)
    gateway.client.get_complete_conversation.side_effect = [source, HelpScoutError(
        "gone", status=404, details={"endpoint": "/conversations/321?embed=threads"})]
    first = asyncio.run(discard(gateway, source))
    assert first["action"] == "discarded"
    assert asyncio.run(discard(gateway, source)) == first
    gateway.client.delete_conversation_result.assert_awaited_once_with(321)


def test_discard_allows_routine_activity_entries(gateway):
    # Help Scout adds a lineitem (activity entry) to an API-created draft conversation;
    # A refused discard must leave the original draft intact.
    activity = {"id": 91, "type": "lineitem", "state": None,
                "createdBy": {"id": 9, "type": "user"}, "source": {"type": "api"}}
    source = raw(status="pending", threads=[activity, thread()])
    gateway.client.get_complete_conversation.side_effect = [source, HelpScoutError(
        "gone", status=404, details={"endpoint": "/conversations/321?embed=threads"})]
    assert asyncio.run(discard(gateway, source))["status"] == "discarded"
    gateway.client.delete_conversation_result.assert_awaited_once_with(321)


@pytest.mark.parametrize("source,code", [
    (raw(status="pending", threads=[{"id": 91, "type": "lineitem", "state": None}]),
     "conversation_not_discardable"),
    (raw(threads=[{"id": 91, "type": "lineitem", "state": None},
                  thread(state="published", draft=False)]), "conversation_not_discardable"),
    (raw(cid=999), "conversation_resolution_changed"),
    (raw(status="pending"), "conversation_not_discardable"),
    (raw(threads=[thread(type="note")]), "conversation_not_discardable"),
    (raw(threads=[thread(createdBy={"type": "customer"})]), "conversation_not_discardable"),
    (raw(threads=[thread(state="published", draft=False)]), "conversation_not_discardable"),
    (raw(threads=[thread(state="published", draft=True)]), "conversation_not_discardable"),
    (raw(threads=[thread(state="draft", draft=False)]), "conversation_not_discardable"),
    (raw(threads=[thread(), thread(id=91, state="published", draft=False)]), "conversation_not_discardable"),
    ({"id": 321, "status": "draft"}, "invalid_conversation"),
])
def test_discard_refuses_unsafe_resources(gateway, source, code):
    gateway.client.get_complete_conversation.return_value = source
    with pytest.raises(ConversationGatewayError) as error:
        asyncio.run(discard(gateway, source))
    assert error.value.code == code
    gateway.client.delete_conversation_result.assert_not_awaited()


def test_discard_requires_current_version(gateway):
    source = raw(threads=[thread()])
    changed = copy.deepcopy(source)
    changed["_embedded"]["threads"][0]["body"] = "changed"
    gateway.client.get_complete_conversation.return_value = changed
    with pytest.raises(ConversationGatewayError) as error:
        asyncio.run(discard(gateway, source))
    assert error.value.code == "conversation_changed"
    gateway.client.delete_conversation_result.assert_not_awaited()


def test_initial_404_is_not_success(gateway):
    gateway.client.get_complete_conversation.side_effect = HelpScoutError("gone", status=404)
    with pytest.raises(HelpScoutError):
        asyncio.run(discard(gateway))
    gateway.client.delete_conversation_result.assert_not_awaited()


def test_missing_thread_page_is_not_delete_proof(gateway):
    gateway.client.get_complete_conversation.side_effect = [raw(), HelpScoutError(
        "missing thread page", status=404, details={"endpoint": "/conversations/321/threads?page=1"})]
    with pytest.raises(ConversationGatewayError) as error:
        asyncio.run(discard(gateway))
    assert error.value.code == "outcome_unknown"


def test_unexpected_success_status_is_unknown(gateway):
    gateway.client.get_complete_conversation.return_value = raw()
    gateway.client.delete_conversation_result.return_value = MutationResult(True, 200)
    with pytest.raises(ConversationGatewayError) as error:
        asyncio.run(discard(gateway))
    assert error.value.code == "outcome_unknown"


@pytest.mark.parametrize("kind", ["transport", "5xx", "still_exists", "readback"])
def test_discard_uncertainty_never_repeats_delete(gateway, kind):
    gateway.client.get_complete_conversation.side_effect = [
        raw(), HelpScoutError("synthetic", status=503) if kind == "readback" else raw()]
    if kind == "transport":
        gateway.client.delete_conversation_result.side_effect = ConnectionError("synthetic timeout")
    elif kind == "5xx":
        gateway.client.delete_conversation_result.return_value = MutationResult(False, 503)
    for _ in range(2):
        with pytest.raises(ConversationGatewayError):
            asyncio.run(discard(gateway))
    gateway.client.delete_conversation_result.assert_awaited_once()


@pytest.mark.parametrize("who", [identity(("read",)), identity(conversations=frozenset({999}))])
def test_discard_authorization_before_upstream(gateway, who):
    with pytest.raises(ConversationGatewayError):
        asyncio.run(discard(gateway, identity=who))
    gateway.client.get_complete_conversation.assert_not_awaited()
    gateway.client.delete_conversation_result.assert_not_awaited()
