"""An agent may copy an address supplied by this customer on this conversation."""

from __future__ import annotations

import io
import json
import random
from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from api import agent as agent_api
from db.database import init_db
from main import app
from services import scheduled_sends
from services.agent_auth import AgentIdentity, write_token_records
from services.agent_operations import get_operation
from services.conversation_gateway import (
    ConversationGateway,
    ConversationGatewayError,
    cc_allowed_addresses,
    draft_version,
    normalize_conversation,
)
from supa_cli.main import main as cli_main
from tests.test_agent_conversation_allowlist import (
    CASE_HEADERS,
    CASE_TOKEN,
    OTHER,
    _assert_not_allowed,
    _record,
)
from tests.test_agent_writes import FakeDraftClient, _conversation, _customer, _draft
from tests.test_scheduled_sends import BOOKED_AT, _at

CONVERSATION = 321
CUSTOMER_THREAD = 500
GIVEN = "a-proofs@bindery.test"
CUSTOMER_BODY = (
    "<p>Please copy the bindery when the proofs are ready. "
    f'<a href="mailto:{GIVEN}">{GIVEN.upper()}</a> handles our proofs.</p>'
)
IDENTITY = AgentIdentity(
    token_id="case-cc-1",
    name="case-cc",
    scopes=frozenset({"read", "draft", "send", "note"}),
    conversations=frozenset({CONVERSATION}),
)


class CcClient(FakeDraftClient):
    """The write-test fake, plus Help Scout's reply `cc` array on the sent thread."""

    def __init__(self, conversation: dict):
        super().__init__(conversation)
        self.sent_cc: list = []

    async def send_reply_result(
        self,
        conversation_id,
        text,
        status="pending",
        customer_id=None,
        verify_status=True,
        cc=None,
    ):
        self.sent_cc.append(cc)
        before = len(self.conversation["_embedded"]["threads"])
        result = await super().send_reply_result(
            conversation_id, text, status, customer_id, verify_status
        )
        if cc and len(self.conversation["_embedded"]["threads"]) > before:
            self.conversation["_embedded"]["threads"][before]["cc"] = list(cc)
        return result


def _client(*, body: str = CUSTOMER_BODY, drafts: list[dict] | None = None) -> CcClient:
    return CcClient(
        _conversation(
            CONVERSATION,
            threads=[_customer(CUSTOMER_THREAD, body, "2026-09-25T10:00:00Z")],
            drafts=drafts,
        )
    )


def _gateway(client, db_path, *, block=None) -> ConversationGateway:
    return ConversationGateway(
        client, db_path=db_path, cc_block_check=block or (lambda address: None)
    )


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "cc.db")
    init_db(path)
    return path


async def _reply(gateway, *, cc, key="reply-cc-1"):
    return await gateway.send_reply(
        CONVERSATION,
        body="Buna ziua,\n\nVa trimit detaliile.",
        body_format="text",
        expected_latest_thread_id=CUSTOMER_THREAD,
        idempotency_key=key,
        identity=IDENTITY,
        request_id=f"request-{key}",
        cc=cc,
    )


# --- the allow rule -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_cc_given_in_a_customer_thread_goes_out_and_is_recorded(db_path):
    client = _client()

    result = await _reply(_gateway(client, db_path), cc=[" a-proofs@bindery.test "])

    assert result["status"] == "sent"
    assert result["cc"] == [GIVEN]
    assert client.sent_cc == [[GIVEN]]
    operation = get_operation(db_path=db_path, idempotency_key="reply-cc-1", identity=IDENTITY)
    assert operation.request_metadata["cc"] == [GIVEN]
    assert operation.response["cc"] == [GIVEN]
    # The replay answers from the ledger with the same CC and never sends again.
    replay = await _reply(_gateway(client, db_path), cc=[GIVEN])
    assert replay == result
    assert len(client.sent_cc) == 1


@pytest.mark.asyncio
async def test_customer_own_address_is_allowed(db_path):
    client = _client(body="<p>Thanks!</p>")

    result = await _reply(_gateway(client, db_path), cc=["Ada@Example.test"])

    assert result["cc"] == ["ada@example.test"]


@pytest.mark.asyncio
async def test_without_cc_the_send_is_unchanged(db_path):
    client = _client()

    result = await _reply(_gateway(client, db_path), cc=None)

    assert "cc" not in result
    assert client.sent_cc == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "someone@else.test",
        "proofs@bindery.test",  # a substring of the given address is not the address
        "editor@woodland.test",
    ],
)
async def test_an_address_the_customer_never_gave_is_refused_before_sending(db_path, address):
    client = _client()

    with pytest.raises(ConversationGatewayError) as raised:
        await _reply(_gateway(client, db_path), cc=[GIVEN, address])

    assert raised.value.code == "cc_not_allowed"
    assert raised.value.status == 403
    assert raised.value.details == {"refused_cc": [address]}
    assert client.sent_cc == []
    operation = get_operation(db_path=db_path, idempotency_key="reply-cc-1", identity=IDENTITY)
    assert operation.state == "failed"
    assert operation.error["code"] == "cc_not_allowed"


@pytest.mark.asyncio
async def test_an_address_only_in_our_own_reply_or_a_note_is_refused(db_path):
    client = _client(body="<p>Hello</p>")
    threads = client.conversation["_embedded"]["threads"]
    threads.insert(0, {
        "id": 400, "type": "message", "state": "published", "body": f"<p>{GIVEN}</p>",
        "createdAt": "2026-09-25T09:00:00Z", "createdBy": {"type": "user", "id": 9},
    })
    threads.append({
        "id": 450, "type": "note", "state": "published", "body": f"<p>{GIVEN}</p>",
        "createdAt": "2026-09-25T09:30:00Z", "createdBy": {"type": "user", "id": 9},
    })

    with pytest.raises(ConversationGatewayError) as raised:
        await _reply(_gateway(client, db_path), cc=[GIVEN])

    assert raised.value.code == "cc_not_allowed"
    assert client.sent_cc == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cc",
    [
        ["not-an-address"],
        ["missing.multiple.labels@invalid"],
        [GIVEN, "ada@example.test", "x@y.test"],  # at most two
        "a-proofs@bindery.test",  # not a list
        [42],
    ],
)
async def test_malformed_or_too_many_cc_is_refused_before_any_work(db_path, cc):
    client = _client()

    with pytest.raises(ConversationGatewayError) as raised:
        await _reply(_gateway(client, db_path), cc=cc)

    assert raised.value.code == "invalid_cc"
    assert raised.value.status == 422
    assert client.calls == []


@pytest.mark.asyncio
async def test_same_key_with_a_different_cc_is_an_idempotency_conflict(db_path):
    client = _client(body=f"<p>{GIVEN} or ada.office@example.test</p>")
    gateway = _gateway(client, db_path)
    await _reply(gateway, cc=[GIVEN])

    with pytest.raises(Exception) as raised:
        await _reply(gateway, cc=["ada.office@example.test"])

    assert getattr(raised.value, "code", None) == "idempotency_conflict"
    with pytest.raises(Exception) as without:
        await _reply(gateway, cc=None)
    assert getattr(without.value, "code", None) == "idempotency_conflict"
    assert len(client.sent_cc) == 1


@pytest.mark.asyncio
async def test_same_cc_in_another_order_or_case_replays_the_same_request(db_path):
    client = _client(body=f"<p>{GIVEN} or ada.office@example.test</p>")
    gateway = _gateway(client, db_path)
    first = await _reply(gateway, cc=["ada.office@example.test", GIVEN])

    replay = await _reply(gateway, cc=[GIVEN.upper(), "Ada.Office@example.test"])

    assert replay == first
    assert first["cc"] == [GIVEN, "ada.office@example.test"]
    assert len(client.sent_cc) == 1


@pytest.mark.asyncio
async def test_a_ui_draft_prefilled_with_the_customer_thread_cc_and_to_can_be_sent(db_path):
    client = _client(body="<p>Please include them.</p>")
    customer = client.conversation["_embedded"]["threads"][0]
    customer["cc"] = ["Colleague@Bindery.test"]
    customer["to"] = ["hello@woodland.test", "partner@example.test"]
    draft = _draft(701, "<p>The proofs are ready.</p>", "2026-09-25T11:00:00Z")
    draft["cc"] = ["colleague@bindery.test", "partner@example.test"]
    client.conversation["_embedded"]["threads"].append(draft)

    result = await _gateway(client, db_path).send_draft(
        CONVERSATION,
        expected_version=draft_version(draft),
        expected_latest_thread_id=CUSTOMER_THREAD,
        idempotency_key="draft-cc-ui",
        identity=IDENTITY,
        request_id="request-draft-cc-ui",
    )

    assert result["cc"] == ["colleague@bindery.test", "partner@example.test"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "address"),
    [
        # Gmail-style reply header, in Romanian and English.
        ("<p>Mulțumesc pentru ajutor.</p><p>Pe 4 oct. 2026, Rowan &lt;editor@woodland.test&gt; a scris:</p>"
         "<p>scrieti si la third@party.test</p>", "third@party.test"),
        ("<p>Yes please.</p><p>On Sat, Oct 4, 2026 at 10:00 Rowan wrote:</p>"
         "<p>&gt; third@party.test</p>", "third@party.test"),
        # Outlook "From: ... Sent:" block of a forwarded chain.
        ("<p>See below.</p><p>From: Other Person &lt;other@vendor.test&gt; Sent: Friday</p>"
         "<p>To: team</p>", "other@vendor.test"),
        # Quoted history in a blockquote.
        ("<p>Ok.</p><blockquote><p>Write to quoted@vendor.test</p></blockquote>", "quoted@vendor.test"),
    ],
)
async def test_addresses_in_quoted_history_are_not_given_by_the_customer(db_path, body, address):
    client = _client(body=body)

    with pytest.raises(ConversationGatewayError) as raised:
        await _reply(_gateway(client, db_path), cc=[address])

    assert raised.value.code == "cc_not_allowed"
    assert client.sent_cc == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    ["editor@woodland.test", "hello@woodland.test", "support@mail.woodland.test",
     "agent.nine@example.test"],
)
async def test_our_own_mailboxes_and_team_are_never_allowed(db_path, address, monkeypatch):
    monkeypatch.setattr("services.conversation_gateway._OWN_CC_DOMAINS", ("woodland.test",))
    client = _client(body=f"<p>Please also copy {address}</p>")
    customer = client.conversation["_embedded"]["threads"][0]
    customer["cc"] = [address]
    customer["to"] = [address]
    client.conversation["_embedded"]["threads"].insert(0, {
        "id": 400, "type": "message", "state": "published", "body": "<p>Hi</p>",
        "createdAt": "2026-09-25T09:00:00Z",
        "createdBy": {"type": "user", "id": 9, "email": "agent.nine@example.test"},
    })

    with pytest.raises(ConversationGatewayError) as raised:
        await _reply(_gateway(client, db_path), cc=[address])

    assert raised.value.code == "cc_not_allowed"


@pytest.mark.asyncio
async def test_extra_own_domains_come_from_the_environment(db_path, monkeypatch):
    monkeypatch.setenv("SUPA_CC_OWN_DOMAINS", "bindery.test")
    client = _client(body="<p>office@bindery.test</p>")

    with pytest.raises(ConversationGatewayError) as raised:
        await _reply(_gateway(client, db_path), cc=["office@bindery.test"])

    assert raised.value.code == "cc_not_allowed"


@pytest.mark.parametrize(
    ("body", "allowed", "refused"),
    [
        ("<p>Write to o'brien@x.test please</p>", "o'brien@x.test", "brien@x.test"),
        ("<p>Write to 'quoted@x.test' please</p>", "quoted@x.test", "'quoted@x.test"),
        ("<p>Mail: reader@bad_domain.example</p>", None, "reader@bad.example"),
        ("<p>Mail: ann@studio.example.</p>", "ann@studio.example", "ann@studio.test"),
    ],
)
def test_text_matching_uses_the_same_address_shape_as_the_validator(body, allowed, refused):
    addresses = cc_allowed_addresses(normalize_conversation(_client(body=body).conversation))

    assert addresses - {"ada@example.test"} == ({allowed} if allowed else set())
    assert refused not in addresses


# --- do-not-contact -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unreadable_do_not_contact_list_fails_closed(db_path):
    client = _client()

    def broken(address):
        raise OSError("ledger unreadable")

    with pytest.raises(ConversationGatewayError) as raised:
        await _reply(_gateway(client, db_path, block=broken), cc=[GIVEN])

    assert raised.value.code == "cc_check_failed"
    assert raised.value.retryable is True
    assert client.sent_cc == []


# --- draft send ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_draft_send_carries_the_draft_cc_and_the_requested_cc(db_path):
    draft = _draft(701, "<p>The proofs are ready.</p>", "2026-09-25T11:00:00Z")
    draft["cc"] = ["Ada@example.test"]
    client = _client(drafts=[draft])
    conversation = normalize_conversation(client.conversation)
    assert conversation["draft"]["cc"] == ["Ada@example.test"]
    assert draft_version(draft) != draft_version({**draft, "cc": []})

    result = await _gateway(client, db_path).send_draft(
        CONVERSATION,
        expected_version=draft_version(draft),
        expected_latest_thread_id=CUSTOMER_THREAD,
        idempotency_key="draft-cc-1",
        identity=IDENTITY,
        request_id="request-draft-cc-1",
        cc=[GIVEN],
    )

    assert client.sent_cc == [[GIVEN, "ada@example.test"]]  # sorted
    assert result["cc"] == [GIVEN, "ada@example.test"]


@pytest.mark.asyncio
async def test_draft_send_refuses_a_stored_cc_the_customer_never_gave(db_path):
    draft = _draft(701, "<p>The proofs are ready.</p>", "2026-09-25T11:00:00Z")
    draft["cc"] = ["stranger@example.test"]
    client = _client(drafts=[draft])

    with pytest.raises(ConversationGatewayError) as raised:
        await _gateway(client, db_path).send_draft(
            CONVERSATION,
            expected_version=draft_version(draft),
            expected_latest_thread_id=CUSTOMER_THREAD,
            idempotency_key="draft-cc-2",
            identity=IDENTITY,
            request_id="request-draft-cc-2",
        )

    assert raised.value.code == "cc_not_allowed"
    assert client.sent_cc == []


def test_conversation_shows_cc_on_our_threads_only_when_present():
    client = _client()
    threads = client.conversation["_embedded"]["threads"]
    threads.append({
        "id": 990, "type": "message", "state": "published", "body": "<p>Sent</p>",
        "createdAt": "2026-09-25T12:00:00Z", "createdBy": {"type": "user", "id": 9},
        "cc": [GIVEN], "bcc": ["hidden@example.test"],
    })

    normalized = normalize_conversation(client.conversation)

    by_id = {thread["id"]: thread for thread in normalized["threads"]}
    assert by_id[990]["cc"] == [GIVEN]
    assert "bcc" not in by_id[990]
    assert "cc" not in by_id[CUSTOMER_THREAD]


# --- scheduled sends ----------------------------------------------------------------


async def _book(client, db_path, *, cc, key="sched-cc-1", **kwargs):
    kwargs.setdefault("body", "Buna ziua,\n\nVa trimit detaliile.")
    kwargs.setdefault("body_format", "text")
    return await scheduled_sends.schedule_send(
        gateway=_gateway(client, db_path),
        db_path=db_path,
        identity=IDENTITY,
        conversation_id=CONVERSATION,
        expected_latest_thread_id=CUSTOMER_THREAD,
        idempotency_key=key,
        clock=_at(BOOKED_AT),
        rng=random.Random(7),
        cc=cc,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_a_booked_reply_keeps_its_cc_until_it_is_sent(db_path):
    client = _client()

    booked = await _book(client, db_path, cc=[GIVEN])
    assert booked["cc"] == [GIVEN]

    outcome = await scheduled_sends.process_row(
        booked["schedule_id"],
        gateway=_gateway(client, db_path),
        db_path=db_path,
        clock=_at(BOOKED_AT + timedelta(days=3)),
        identity_lookup=lambda _token_id: IDENTITY,
    )

    assert outcome == "sent"
    assert client.sent_cc == [[GIVEN]]


@pytest.mark.asyncio
async def test_booking_the_saved_draft_keeps_the_draft_cc(db_path):
    draft = _draft(701, "<p>The proofs are ready.</p>", "2099-01-01T00:00:00Z")
    draft["cc"] = [GIVEN]
    client = _client(drafts=[draft])

    booked = await _book(
        client, db_path, cc=None, body=None, body_format=None,
        expected_version=draft_version(draft),
    )

    assert booked["cc"] == [GIVEN]


@pytest.mark.asyncio
async def test_booking_refuses_a_cc_the_customer_never_gave_and_keeps_nothing(db_path):
    client = _client()

    with pytest.raises(ConversationGatewayError) as raised:
        await _book(client, db_path, cc=["stranger@example.test"])

    assert raised.value.code == "cc_not_allowed"
    assert scheduled_sends.latest_for_conversation(db_path, CONVERSATION) is None


@pytest.mark.asyncio
async def test_booking_with_the_same_key_and_another_cc_conflicts(db_path):
    client = _client(body=f"<p>{GIVEN} or ada.office@example.test</p>")
    await _book(client, db_path, cc=[GIVEN])

    with pytest.raises(ConversationGatewayError) as raised:
        await _book(client, db_path, cc=["ada.office@example.test"])

    assert raised.value.code == "idempotency_conflict"


# --- API ----------------------------------------------------------------------------


@pytest.fixture
def cc_api(tmp_path, monkeypatch):
    registry_path = tmp_path / "private-agent-tokens.json"
    write_token_records(
        registry_path,
        [_record(CASE_TOKEN, "token-case-1", "case-004", conversations=[CONVERSATION])],
    )
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(registry_path))
    db = str(tmp_path / "api.db")
    init_db(db)
    monkeypatch.setattr(agent_api, "get_supa_db_path", lambda: db)
    client = _client()
    gateway = _gateway(client, db)
    dependency = agent_api.get_conversation_gateway
    app.dependency_overrides[dependency] = lambda: gateway
    try:
        yield TestClient(app), client
    finally:
        app.dependency_overrides.pop(dependency, None)


REPLY = {
    "body": "Buna ziua",
    "body_format": "text",
    "expected_latest_thread_id": CUSTOMER_THREAD,
    "idempotency_key": "api-cc-1",
}


def test_api_reply_with_a_given_cc_is_sent(cc_api):
    http, client = cc_api

    response = http.post(
        f"/agent/v1/conversations/{CONVERSATION}/replies",
        headers=CASE_HEADERS,
        json={**REPLY, "cc": [GIVEN]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["result"]["cc"] == [GIVEN]
    assert client.sent_cc == [[GIVEN]]
    timeline = http.get(f"/agent/v1/conversations/{CONVERSATION}", headers=CASE_HEADERS)
    sent = [t for t in timeline.json()["result"]["threads"] if t["id"] == 990]
    assert sent[0]["cc"] == [GIVEN]


def test_api_refuses_a_cc_the_customer_never_gave(cc_api):
    http, client = cc_api

    response = http.post(
        f"/agent/v1/conversations/{CONVERSATION}/replies",
        headers=CASE_HEADERS,
        json={**REPLY, "cc": ["stranger@example.test"]},
    )

    assert response.status_code == 403
    error = response.json()["error"]
    assert error["code"] == "cc_not_allowed"
    assert error["details"] == {"refused_cc": ["stranger@example.test"]}
    assert client.sent_cc == []


def test_api_case_insensitive_duplicates_count_once(cc_api):
    http, client = cc_api

    response = http.post(
        f"/agent/v1/conversations/{CONVERSATION}/replies",
        headers=CASE_HEADERS,
        json={**REPLY, "cc": [GIVEN.upper(), GIVEN, "ada@example.test"]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["result"]["cc"] == [GIVEN, "ada@example.test"]


def test_api_three_distinct_addresses_are_invalid_cc_not_a_schema_error(cc_api):
    http, client = cc_api

    response = http.post(
        f"/agent/v1/conversations/{CONVERSATION}/replies",
        headers=CASE_HEADERS,
        json={**REPLY, "cc": ["a@x.test", "b@x.test", "c@x.test"]},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_cc"
    assert client.calls == []


@pytest.mark.parametrize(
    "path",
    [
        "/agent/v1/conversations/{id}/replies",
        "/agent/v1/conversations/{id}/draft/send",
        "/agent/v1/conversations/{id}/draft/schedule",
    ],
)
def test_api_case_token_is_still_refused_another_conversation(cc_api, path):
    http, client = cc_api
    payload = {**REPLY, "cc": [GIVEN]}
    if path.endswith("draft/send"):
        payload = {
            "expected_version": "a" * 64,
            "expected_latest_thread_id": CUSTOMER_THREAD,
            "idempotency_key": "api-cc-2",
            "cc": [GIVEN],
        }

    response = http.post(path.format(id=OTHER), headers=CASE_HEADERS, json=payload)

    _assert_not_allowed(response)
    assert client.calls == []


# --- CLI ----------------------------------------------------------------------------


def _cli(argv, handler, stdin_text=""):
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli_main(
        argv,
        transport=httpx.MockTransport(handler),
        environ={"SUPA_URL": "https://supa.example.test", "SUPA_TOKEN": "A" * 43},
        stdin=io.StringIO(stdin_text),
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def _envelope(result):
    return httpx.Response(200, json={"api_version": "v1", "request_id": "req_t", "result": result})


@pytest.mark.parametrize(
    ("argv", "path"),
    [
        (["reply", "send", "321", "--body-file", "-", "--body-format", "text",
          "--expected-thread", "500"], "/agent/v1/conversations/321/replies"),
        (["draft", "send", "321", "--expected-version", "a" * 64,
          "--expected-thread", "500"], "/agent/v1/conversations/321/draft/send"),
        (["draft", "schedule", "321", "--body-file", "-", "--body-format", "text",
          "--expected-thread", "500"], "/agent/v1/conversations/321/draft/schedule"),
    ],
)
def test_cli_cc_is_repeatable_and_sent_in_the_payload(argv, path):
    seen = []

    def handler(request):
        seen.append((request.url.path, json.loads(request.content)))
        return _envelope({"status": "sent"})

    code, _, stderr = _cli(
        argv + ["--cc", GIVEN, "--cc", "ada@example.test", "--idempotency-key", "cc-k1"],
        handler, stdin_text="Buna ziua",
    )

    assert code == 0, stderr
    assert seen[0][0] == path
    assert seen[0][1]["cc"] == [GIVEN, "ada@example.test"]


def test_cli_without_cc_sends_no_cc_field():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _envelope({"status": "sent"})

    code, _, _ = _cli(
        ["reply", "send", "321", "--body-file", "-", "--body-format", "text",
         "--expected-thread", "500", "--idempotency-key", "cc-k2"],
        handler, stdin_text="Buna ziua",
    )

    assert code == 0
    assert "cc" not in seen[0]


@pytest.mark.parametrize(
    "cc_args",
    [
        ["--cc", "not-an-address"],
        ["--cc", "a@b.test", "--cc", "c@d.test", "--cc", "e@f.test"],
    ],
)
def test_cli_rejects_bad_cc_without_a_request(cc_args):
    def handler(request):  # pragma: no cover - must not be reached
        raise AssertionError("no request expected")

    code, _, stderr = _cli(
        ["reply", "send", "321", "--body-file", "-", "--body-format", "text",
         "--expected-thread", "500", *cc_args],
        handler, stdin_text="x",
    )

    assert code == 2
    assert json.loads(stderr)["error"]["code"] == "usage_error"


def test_cli_maps_cc_not_allowed_to_the_auth_exit_code():
    def handler(request):
        return httpx.Response(403, json={
            "api_version": "v1", "request_id": "req_t",
            "error": {"code": "cc_not_allowed", "message": "m", "retryable": False,
                      "details": {"refused_cc": ["x@y.test"]}},
        })

    code, _, _ = _cli(
        ["reply", "send", "321", "--body-file", "-", "--body-format", "text",
         "--expected-thread", "500", "--cc", "x@y.test", "--idempotency-key", "cc-k3"],
        handler, stdin_text="x",
    )

    assert code == 3


def test_cli_timeline_shows_cc_on_sent_threads():
    def handler(request):
        return _envelope({
            "id": 321, "status": "pending", "subject": "S",
            "threads": [
                {"id": 990, "type": "message", "state": "published", "draft": False,
                 "created_at": "2026-10-05T10:00:00Z", "author": {"name": "Rowan"},
                 "body_text": "Detaliile", "cc": [GIVEN]},
            ],
        })

    code, stdout, _ = _cli(["conversation", "timeline", "321"], handler)

    assert code == 0
    assert f"cc: {GIVEN}" in stdout
