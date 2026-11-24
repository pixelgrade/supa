"""Supa's own writers state their records as data; a note still reads as the same text."""

import json

import httpx
import pytest

from services import mailbox_provider
from services.helpscout import HelpScoutClient, MutationResult, RecordedNote, recorded

OWNED = "https://mail.woodland.test/v2"



def client(base, monkeypatch, answers=None):
    monkeypatch.setenv("SUPA_MAILBOX_BASE_URL", OWNED)
    monkeypatch.setattr("services.installation.OPERATIONS_ORIGIN", "https://mail.woodland.test")
    made = HelpScoutClient.__new__(HelpScoutClient)
    made.BASE_URL = base
    calls = []
    answers = list(answers or [])

    async def mutation(method, endpoint, **kwargs):
        calls.append((method, endpoint, kwargs.get("json")))
        return answers.pop(0) if answers else MutationResult(ok=True, status_code=201, resource_id=77)

    monkeypatch.setattr(made, "_mutation_result", mutation, raising=False)
    return made, calls


def test_a_recorded_note_is_still_the_same_text():
    note = recorded("<p>Hello</p>", {"kind": "note", "actor": "supa", "title": "t"})
    assert note == "<p>Hello</p>" and isinstance(note, str) and note.event["kind"] == "note"
    assert json.dumps({"text": note}) == '{"text": "<p>Hello</p>"}'
    assert not hasattr("plain", "event") and RecordedNote("x").event is None


@pytest.mark.asyncio
async def test_the_owned_mailbox_gets_the_record_and_the_vendor_only_the_text(monkeypatch):
    note = recorded("<p>Reviewed.</p>", {"kind": "note", "actor": "supa", "title": "Review complete", "data": {"score": 9}})
    owned, calls = client(OWNED, monkeypatch)
    result = await owned.add_note_result(7, note, "pending")
    assert result.ok and calls == [("POST", "/conversations/7/notes", {"text": str(note), "status": "pending", "event": note.event})]
    assert type(calls[0][2]["text"]) is str
    await owned.add_note_result(7, "<p>Plain.</p>")
    assert calls[1][2] == {"text": "<p>Plain.</p>", "event": {"actor": "supa"}}  # Whatever Supa writes is Supa's.
    await owned.add_note_result(7, recorded("<p>By an agent.</p>", {"actor": "claude:case-7"}))
    assert calls[2][2]["event"] == {"actor": "claude:case-7"}
    vendor, calls = client(mailbox_provider.VENDOR_API, monkeypatch)
    await vendor.add_note_result(7, note, "pending")
    assert calls == [("POST", "/conversations/7/notes", {"text": str(note), "status": "pending"})]


@pytest.mark.asyncio
async def test_a_record_the_mailbox_refuses_never_costs_the_note(monkeypatch):
    note = recorded("<p>Reviewed.</p>", {"kind": "note", "actor": "supa", "title": "Review complete", "data": {"score": 9}})
    refused, calls = client(OWNED, monkeypatch, [MutationResult(ok=False, status_code=400), MutationResult(ok=True, status_code=201)])
    assert (await refused.add_note_result(7, note)).ok
    assert [("event" in c[2]) for c in calls] == [True, False] and calls[1][2] == {"text": str(note)}
    # Anything but a definite refusal is not asked twice: a second note could be a second copy.
    for answer in (MutationResult(ok=False, status_code=None, outcome_unknown=True), MutationResult(ok=False, status_code=503),
                   MutationResult(ok=False, status_code=409)):
        once, calls = client(OWNED, monkeypatch, [answer])
        assert not (await once.add_note_result(7, note)).ok and len(calls) == 1
    odd, calls = client(OWNED, monkeypatch)
    await odd.add_note_result(7, recorded("<p>x</p>", {"kind": "note", "actor": "supa", "title": "t", "data": {"n": float("nan")}}))
    assert calls[0][2]["event"] == {"actor": "supa"}  # An unencodable record is left out before anything is sent.


@pytest.mark.asyncio
async def test_an_edit_restates_a_full_record_and_nothing_else(monkeypatch):
    owned, calls = client(OWNED, monkeypatch)
    note = recorded("<p>Reviewed.</p>", {"kind": "note", "actor": "supa", "title": "Review complete", "data": {"score": 9}})
    await owned.update_note_result(7, 70, note)
    await owned.update_note_result(7, 70, recorded("<p>Edited by an agent.</p>", {"actor": "claude:case-7"}))
    await owned.update_note_result(7, 70, "<p>Plain.</p>")
    assert calls[0] == ("PATCH", "/conversations/7/threads/70", {"op": "replace", "path": "/text", "value": str(note), "event": note.event})
    assert [c[2] for c in calls[1:]] == [{"op": "replace", "path": "/text", "value": "<p>Edited by an agent.</p>"},
                                         {"op": "replace", "path": "/text", "value": "<p>Plain.</p>"}]
    vendor, calls = client(mailbox_provider.VENDOR_API, monkeypatch)
    await vendor.update_note_result(7, 70, note)
    assert "event" not in calls[0][2]


@pytest.mark.asyncio
async def test_a_request_to_the_owner_is_one_open_record(monkeypatch):
    sent = []
    answer = {"status": 201, "body": {"id": 77, "status": "open", "replayed": False}}

    def owned():
        made, _ = client(OWNED, monkeypatch)

        async def request(method, endpoint, **kwargs):
            sent.append((method, endpoint, kwargs.get("json")))
            if isinstance(answer["body"], Exception):
                raise answer["body"]
            return httpx.Response(answer["status"], json=answer["body"]) if answer["body"] is not None else httpx.Response(answer["status"], text="<html>")

        monkeypatch.setattr(made, "_request", request, raising=False)
        return made

    mailbox = owned()
    status, body = await mailbox.add_request(7, "claude:case-7", "  Confirm whether old captions\n are kept. ", "captions-1")
    (method, endpoint, payload), = sent
    assert (status, body["id"], method, endpoint) == (201, 77, "POST", "/conversations/7/events")
    assert (payload["kind"], payload["actor"], payload["status"], payload["title"], payload["data"]) == (
        "request", "claude:case-7", "open", "Confirm whether old captions are kept.", {"action": "Confirm whether old captions are kept."})
    await mailbox.add_request(7, "claude:case-7", "Different words, same key.", "captions-1")
    await mailbox.add_request(7, "claude:case-7", "Confirm.", "captions-2")
    await mailbox.add_request(8, "claude:case-7", "Confirm.", "captions-1")
    # A colon in a name or a key must not make two askers share a request.
    await mailbox.add_request(7, "claude", "Confirm.", "case-7:captions-1")
    # Two agents using the same key on one conversation ask two different things.
    await mailbox.add_request(7, "codex:case-7", "Confirm.", "captions-1")
    keys = [c[2]["operation"] for c in sent]
    assert keys[0] == keys[1] and len(set(keys)) == 5 and all(8 <= len(k) <= 200 for k in keys)
    # What the mailbox says comes back as it is: closed stays closed, a conflict is a conflict.
    answer.update(status=201, body={"id": 77, "status": "done", "replayed": True})
    assert await mailbox.add_request(7, "claude:case-7", "Confirm.", "captions-1") == (201, {"id": 77, "status": "done", "replayed": True})
    answer.update(status=409, body={"message": "Operation key already used for another record"})
    assert (await mailbox.add_request(7, "claude:case-7", "Other.", "captions-1"))[0] == 409
    answer.update(status=502, body=None)
    assert await mailbox.add_request(7, "claude:case-7", "Confirm.", "k") == (502, {})
    answer.update(body=KeyError("access_token"))
    assert await mailbox.add_request(7, "claude:case-7", "Confirm.", "k") == (None, {})

    closing, calls = client(OWNED, monkeypatch)
    await closing.close_request(7, 77, "claude:case-7")
    assert calls[-1] == ("PATCH", "/conversations/7/events/77", {"status": "done", "actor": "claude:case-7"})
    vendor, calls = client(mailbox_provider.VENDOR_API, monkeypatch)
    assert (await vendor.add_request(7, "a", "Confirm.", "k"))[0] == 409
    assert (await vendor.close_request(7, 77, "a")).detail == "native_mailbox_required" and calls == []


@pytest.mark.asyncio
async def test_non_text_is_still_refused_as_before_and_an_edit_keeps_its_fallback(monkeypatch):
    owned, calls = client(OWNED, monkeypatch)
    await owned.add_note_result(7, None)
    assert calls[0][2]["text"] is None  # Not turned into the word "None".
    note = recorded("<p>Reviewed.</p>", {"kind": "note", "actor": "supa", "title": "Review complete", "data": {"score": 9}})
    refused, calls = client(OWNED, monkeypatch, [MutationResult(ok=False, status_code=400), MutationResult(ok=True, status_code=204)])
    assert (await refused.update_note_result(7, 70, note)).ok
    assert [("event" in c[2]) for c in calls] == [True, False]  # A refused record never costs the edit.
    unsure, calls = client(OWNED, monkeypatch, [MutationResult(ok=False, status_code=None, outcome_unknown=True)])
    assert not (await unsure.update_note_result(7, 70, note)).ok and len(calls) == 1
