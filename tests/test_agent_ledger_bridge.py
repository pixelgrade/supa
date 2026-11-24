"""Supa hands its decisions and state changes to the owned mailbox as data, not as notes and tags."""

from types import SimpleNamespace

import httpx
import pytest

from services import mailbox_provider
from services.helpscout import HelpScoutClient, MutationResult
from services.support_pipeline import decision_record

NOTE = ("Supa decision note:\nMode: draft (auto-send suppressed: the customer is not waiting)\n"
        "Confidence: 9/10 (auto-send needs 8/10 and no blockers)\nWhy: short\n"
        "Scheduled send: 2026-11-24 10:00 UTC\nAuto-send will cancel if the customer or an agent changes the conversation first.")


def result(**changes):
    base = dict(
        delivery_mode="draft", confidence_score=9, confidence_verdict="accept", decision_summary="Answered from the docs.",
        evidence_summary="Index Book docs, 'Adding a store'.", risk_flags=["pricing"], hard_blockers=[], manual_review=False,
        investigator_provider="claude", evaluator_provider="codex",
        investigator_result=SimpleNamespace(sources=["docs/border", "repo:journal-base"]),
        evaluator_result=SimpleNamespace(summary="Every claim traces to the docs. " * 40, missing_information=["Checkout styling"],
                                         investigation_actions=["Check the cart page"], rewrite_brief="Soften one sentence."),
        trace=[{"score": 7}, {"score": 9}, "junk"],
    )
    return SimpleNamespace(**{**base, **changes})


def test_the_record_carries_the_decision_unabridged_and_the_final_mode():
    record = decision_record(result(), NOTE)
    data = record["data"]
    assert (record["kind"], record["actor"], record["title"]) == ("decision", "supa", "Draft left for review · 9/10")
    # The note's Mode line is written last (a late hold rewrites it), so it wins over the pipeline's own field.
    assert (data["mode"], data["mode_note"]) == ("draft", "auto-send suppressed: the customer is not waiting")
    assert (data["confidence"], data["target"], data["scores"], data["scheduled"]) == (9, 8, [7, 9], "2026-11-24 10:00 UTC")
    assert len(data["why"]) > 600  # Not cut to fit a note.
    assert (data["missing"], data["next"]) == ("Checkout styling", "Check the cart page; Soften one sentence.")
    assert (data["sources"], data["providers"], data["risk_flags"]) == (
        ["docs/border", "repo:journal-base"], {"investigator": "claude", "evaluator": "codex"}, ["pricing"])
    assert data["text"] == NOTE and "blockers" not in data and data["manual_review"] is False
    held = decision_record(result(delivery_mode="manual_review", hard_blockers=["refund_request"], manual_review=True),
                           "Supa decision note:\nConfidence: 4/10")
    assert (held["title"], held["data"]["mode"], held["data"]["blockers"], held["data"]["manual_review"]) == (
        "Held for manual review · 9/10 · blocked", "manual_review", ["refund_request"], True)
    assert decision_record(result(delivery_mode="auto_send"), "Supa decision note:\nMode: auto_send")["title"] == "Sent automatically · 9/10"
    booked = decision_record(result(delivery_mode="auto_send"), "Supa decision note:\nMode: auto_send\nScheduled send: 2026-11-24 10:00 UTC")
    assert (booked["title"], booked["data"]["mode"]) == ("Send booked · 9/10", "auto_send")  # Booked is not sent.


def client(base, monkeypatch, outcome=None, response=None):
    made = HelpScoutClient.__new__(HelpScoutClient)
    made.BASE_URL = base
    calls = []

    async def mutation(method, endpoint, **kwargs):
        calls.append((method, endpoint, kwargs.get("json")))
        return outcome

    async def note(conversation_id, text, status=None):
        calls.append(("note", conversation_id, text))
        return True

    async def request(method, endpoint, **kwargs):
        calls.append((method, endpoint, kwargs.get("json")))
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(made, "_mutation_result", mutation, raising=False)
    monkeypatch.setattr(made, "add_note", note, raising=False)
    monkeypatch.setattr(made, "_request", request, raising=False)
    return made, calls


@pytest.mark.asyncio
async def test_a_decision_is_a_record_in_the_owned_mailbox_and_a_note_everywhere_else(monkeypatch):
    record = decision_record(result(), NOTE)
    owned, calls = client("https://mailbox.woodland.example/v2", monkeypatch, MutationResult(ok=True, status_code=201))
    assert await owned.add_decision(7, NOTE, record) is True
    (method, endpoint, body), = calls
    assert (method, endpoint, body["kind"], body["data"]["confidence"]) == ("POST", "/conversations/7/events", "decision", 9)
    first = body["operation"]
    assert first.startswith("supa-decision-") and 8 <= len(first) <= 200
    await owned.add_decision(7, NOTE, record)
    assert calls[1][2]["operation"] != first  # Another decision with the same wording is another record.

    # An uncertain answer is asked again under the same key, so the mailbox stores the record once.
    answers = [MutationResult(ok=False, status_code=None, outcome_unknown=True), MutationResult(ok=True, status_code=201)]
    unsure, calls = client("https://mailbox.woodland.example/v2", monkeypatch)

    async def flaky(method, endpoint, **kwargs):
        calls.append((method, endpoint, kwargs.get("json")))
        return answers.pop(0)

    monkeypatch.setattr(unsure, "_mutation_result", flaky, raising=False)
    assert await unsure.add_decision(7, NOTE, record) is True
    assert [c[0] for c in calls] == ["POST", "POST"] and calls[0][2]["operation"] == calls[1][2]["operation"]
    lost, calls = client("https://mailbox.woodland.example/v2", monkeypatch, MutationResult(ok=False, status_code=None, outcome_unknown=True))
    assert await lost.add_decision(7, NOTE, record) is True
    assert [c[0] for c in calls] == ["POST", "POST", "note"]  # Still unknown: the note is written, the decision kept.

    # A record that cannot be encoded or sent never takes the note with it.
    for broken in ({**record, "data": {**record["data"], "why": "half \ud83d"}}, {**record, "data": {"scores": [float("nan")], "x": object()}}):
        odd, calls = client("https://mailbox.woodland.example/v2", monkeypatch, MutationResult(ok=True, status_code=201))
        assert await odd.add_decision(7, NOTE, broken) is True
        assert calls == [("note", 7, NOTE)]
    raising, calls = client("https://mailbox.woodland.example/v2", monkeypatch)

    async def explode(method, endpoint, **kwargs):
        raise RuntimeError("transport")

    monkeypatch.setattr(raising, "_mutation_result", explode, raising=False)
    assert await raising.add_decision(7, NOTE, record) is True and calls == [("note", 7, NOTE)]

    refused, calls = client("https://mailbox.woodland.example/v2", monkeypatch, MutationResult(ok=False, status_code=404))
    assert await refused.add_decision(7, NOTE, record) is True
    assert [c[0] for c in calls] == ["POST", "note"] and calls[1] == ("note", 7, NOTE)  # The decision is never lost.

    vendor, calls = client(mailbox_provider.VENDOR_API, monkeypatch, MutationResult(ok=True, status_code=201))
    await vendor.add_decision(7, NOTE, record)
    no_record, more = client("https://mailbox.woodland.example/v2", monkeypatch, MutationResult(ok=True, status_code=201))
    await no_record.add_decision(7, NOTE, None)
    assert calls == [("note", 7, NOTE)] and more == [("note", 7, NOTE)]


@pytest.mark.asyncio
async def test_a_state_change_is_one_request_and_a_refusal_is_reported_as_one(monkeypatch):
    state = {"stage": "feedback", "reminders": {"state": "off"}}
    owned, calls = client("https://mailbox.woodland.example/v2", monkeypatch, response=httpx.Response(200, json=state))
    assert await owned.change_state(7, {"stage": "feedback"}, "fixture:desk-7", "Later.") == (200, state)
    assert calls == [("PATCH", "/conversations/7/state", {"changes": {"stage": "feedback"}, "actor": "fixture:desk-7", "reason": "Later."})]
    refused, _ = client("https://mailbox.woodland.example/v2", monkeypatch,
                        response=httpx.Response(409, json={"message": "Only the mailbox owner hands a conversation back to the agents"}))
    assert await refused.change_state(7, {"handling": "agents"}, "supa") == (409, "Only the mailbox owner hands a conversation back to the agents")
    broken, _ = client("https://mailbox.woodland.example/v2", monkeypatch, response=httpx.ConnectError("down"))
    assert await broken.change_state(7, {"stage": "feedback"}, "supa") == (None, "The mailbox could not be reached.")
    for fault in (KeyError("access_token"), ValueError("token")):  # A malformed token answer is an answer too.
        bad_token, _ = client("https://mailbox.woodland.example/v2", monkeypatch, response=fault)
        assert (await bad_token.change_state(7, {"stage": "feedback"}, "supa"))[0] is None
    plain, _ = client("https://mailbox.woodland.example/v2", monkeypatch, response=httpx.Response(502, text="<html>bad gateway</html>"))
    assert await plain.change_state(7, {"stage": "feedback"}, "supa") == (502, "The mailbox refused the change.")
    vendor, calls = client(mailbox_provider.VENDOR_API, monkeypatch)
    status, text = await vendor.change_state(7, {"stage": "feedback"}, "supa")
    assert status is None and "owned mailbox" in text and calls == []
    assert await vendor.ledger(7) is None


@pytest.mark.asyncio
async def test_the_ledger_is_read_from_the_owned_mailbox_only(monkeypatch):
    asked = []

    def reader(status, body):
        async def read(endpoint, *, max_bytes):
            asked.append(endpoint)
            return httpx.Response(status, json=body)
        return read

    owned, _ = client("https://mailbox.woodland.example/v2", monkeypatch)
    monkeypatch.setattr(owned, "_read_bounded", reader(200, {"conversation": 7, "state": {"stage": "feedback"}, "events": []}), raising=False)
    assert (await owned.ledger(7, limit=999))["state"]["stage"] == "feedback"
    assert asked == ["/conversations/7/ledger?limit=200"]
    monkeypatch.setattr(owned, "_read_bounded", reader(404, {"message": "Not found"}), raising=False)
    assert await owned.ledger(7) == {}  # No such conversation: said plainly, not "try again".
    monkeypatch.setattr(owned, "_read_bounded", reader(500, {}), raising=False)
    assert await owned.ledger(7) is None
