"""Processor classification follows the current single-agent contract in AGENTS.md.

The retired structured triage agent's confidence/shadow modes are not processor
inputs. The current classifier returns support/admin/marketing/spam/wrong_address;
admin, marketing and spam close in both support delivery modes, and wrong_address has
its own gated reply (see the last section). Provider uncertainty defaults to support.
"""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from agents import support_agent
from db.database import get_ticket_log_by_helpscout_id
from services.processor import TicketProcessor
from tests.lease_recording import RecordingLeaseFactory
from tests.test_processor_confidence_loop import (
    FakeHelpScout, FakeVectorStore, _support_result, _ticket_payload,
)


@pytest.fixture(autouse=True)
def _clean_support_state(monkeypatch):
    support_agent.reset_state()
    monkeypatch.setenv("SUPA_AUTOSEND_DELAY_ENABLED", "false")
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "live")
    yield
    support_agent.reset_state()


def _payload():
    payload = _ticket_payload()
    # This incoming customer message is the same obligation returned by the fresh
    # fake. An older human note is not part of this classification scenario.
    payload["_embedded"]["threads"] = [payload["_embedded"]["threads"][0]]
    payload["_embedded"]["threads"][0].update(id=1, createdAt="2026-04-03T09:00:00Z")
    return payload


def _wire_common(monkeypatch, tmp_path):
    helpscout = FakeHelpScout()
    helpscout.conversation_payloads[987] = deepcopy(_payload())
    vector = FakeVectorStore(str(tmp_path / "vectors.db"))
    leases = RecordingLeaseFactory()
    monkeypatch.setattr("services.processor.DB_DIR", tmp_path)
    monkeypatch.setattr("services.processor.HelpScoutClient", lambda **kwargs: helpscout)
    monkeypatch.setattr("services.processor.VectorStore", lambda path: vector)
    monkeypatch.setattr("services.processor.conversation_lease", leases)
    return helpscout, vector, leases


def _wire_pipeline(monkeypatch, vector):
    vector.search_results = [
        {"title": "Doc", "url": "https://woodland.example/docs/example", "content": "Evidence", "score": 0.9}
    ]
    monkeypatch.setattr("services.processor.generate_subject", AsyncMock(return_value="Subject"))
    monkeypatch.setattr("services.processor.fetch_customer_data", AsyncMock(return_value=None))
    monkeypatch.setattr("services.processor.related_customer_context", AsyncMock(return_value=""))
    monkeypatch.setattr("services.processor.find_similar_tickets", AsyncMock(return_value=[]))
    monkeypatch.setattr("services.processor.format_customer_context", lambda value: "")
    monkeypatch.setattr("services.processor.format_past_tickets_context", lambda value: "")
    monkeypatch.setattr("services.processor.get_repo_evidence_context", lambda **kwargs: "")
    monkeypatch.setattr("services.processor.get_recent_semantic_failures", lambda **kwargs: [])
    calls = []

    def pipeline(**kwargs):
        calls.append(kwargs)
        return _support_result(delivery_mode="draft", confidence_score=7)

    monkeypatch.setattr("services.processor.run_support_pipeline", pipeline)
    return calls


@pytest.mark.parametrize("category", ["admin", "marketing", "spam"])
@pytest.mark.parametrize("support_mode", ["live", "draft_only"])
async def test_non_support_categories_close_without_drafting_in_either_support_mode(
    monkeypatch, tmp_path, category, support_mode,
):
    monkeypatch.setenv("SUPA_SUPPORT_MODE", support_mode)
    helpscout, _, leases = _wire_common(monkeypatch, tmp_path)
    classify = AsyncMock(return_value=category)
    monkeypatch.setattr("services.processor.classify_ticket", classify)
    monkeypatch.setattr(
        "services.processor.run_support_pipeline",
        lambda **kwargs: pytest.fail("Non-support classification must not enter drafting"),
    )

    result = await TicketProcessor(auto_index=False).process_ticket(_payload())

    assert result["classification"] == category
    assert result["status"] == "processed"
    assert result["response_posted"] is False
    assert result["delivery_mode"] is None
    assert helpscout.closed == [987]
    assert helpscout.tag_updates == [(987, ["supa-processed", category], ["field-journal"])]
    assert helpscout.drafts == helpscout.sent_replies == helpscout.notes == []
    row = get_ticket_log_by_helpscout_id(str(tmp_path / "supa.db"), "987")
    assert row["classification"] == category and row["status"] == "processed"
    assert not row["response_posted"]
    leases.assert_mutation_guarded(987, "close_conversation")
    leases.assert_mutation_guarded(987, "add_tags")
    classify.assert_awaited_once_with(subject="Original subject", message="My index is missing its page numbers.")


@pytest.mark.parametrize("failure", [False, RuntimeError("close failed")])
async def test_close_failure_is_not_logged_as_success(monkeypatch, tmp_path, failure):
    helpscout, _, _ = _wire_common(monkeypatch, tmp_path)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="marketing"))
    close = AsyncMock(side_effect=failure) if isinstance(failure, Exception) else AsyncMock(return_value=failure)
    monkeypatch.setattr(helpscout, "close_conversation", close)

    result = await TicketProcessor(auto_index=False).process_ticket(_payload())

    assert result["status"] == "error"
    assert result["response_posted"] is False
    assert helpscout.tag_updates == []
    assert helpscout.drafts == helpscout.sent_replies == []
    row = get_ticket_log_by_helpscout_id(str(tmp_path / "supa.db"), "987")
    assert row["status"] == "error" and row["error_message"]


@pytest.mark.parametrize("failure", [False, RuntimeError("tags failed")])
async def test_close_tag_failure_is_not_logged_as_success(monkeypatch, tmp_path, failure):
    helpscout, _, _ = _wire_common(monkeypatch, tmp_path)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="spam"))
    tags = AsyncMock(side_effect=failure) if isinstance(failure, Exception) else AsyncMock(return_value=failure)
    monkeypatch.setattr(helpscout, "add_tags", tags)

    result = await TicketProcessor(auto_index=False).process_ticket(_payload())

    assert result["status"] == "error"
    assert result["response_posted"] is False
    assert helpscout.closed == [987]
    assert helpscout.drafts == helpscout.sent_replies == []
    row = get_ticket_log_by_helpscout_id(str(tmp_path / "supa.db"), "987")
    assert row["status"] == "error" and row["error_message"]


async def test_fresh_manual_tag_stops_non_support_close_under_the_lease(monkeypatch, tmp_path):
    helpscout, _, leases = _wire_common(monkeypatch, tmp_path)
    helpscout.conversation_payloads[987]["tags"].append({"tag": "supa-ignore"})
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="spam"))

    result = await TicketProcessor(auto_index=False).process_ticket(_payload())

    assert result["status"] == "skipped"
    assert helpscout.closed == helpscout.tag_updates == []
    assert helpscout.drafts == helpscout.sent_replies == []
    leases.assert_guarded_count(987, 0)


async def test_non_support_tags_preserve_the_fresh_tag_list(monkeypatch, tmp_path):
    helpscout, _, _ = _wire_common(monkeypatch, tmp_path)
    helpscout.conversation_payloads[987]["tags"].append({"tag": "customer-vip"})
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="admin"))

    await TicketProcessor(auto_index=False).process_ticket(_payload())

    assert helpscout.tag_updates[-1][2] == ["field-journal", "customer-vip"]


@pytest.mark.parametrize("support_mode", ["live", "draft_only"])
async def test_support_classification_continues_to_a_draft(monkeypatch, tmp_path, support_mode):
    monkeypatch.setenv("SUPA_SUPPORT_MODE", support_mode)
    helpscout, vector, leases = _wire_common(monkeypatch, tmp_path)
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="support"))
    calls = _wire_pipeline(monkeypatch, vector)

    result = await TicketProcessor(auto_index=False).process_ticket(_payload())

    assert result["status"] == "processed"
    assert result["classification"] == "support"
    assert result["delivery_mode"] == "draft"
    assert result["response_posted"] is True
    assert len(calls) == 1
    assert len(helpscout.drafts) == 1
    assert helpscout.closed == helpscout.sent_replies == []
    leases.assert_mutation_guarded(987, "add_draft")


async def test_provider_uncertainty_defaults_to_support_instead_of_closing(monkeypatch, tmp_path):
    helpscout, vector, _ = _wire_common(monkeypatch, tmp_path)
    calls = _wire_pipeline(monkeypatch, vector)
    monkeypatch.setattr("agents.classifier.should_skip_claude", lambda: False)
    monkeypatch.setattr(
        "agents.classifier.invoke_claude",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("Claude unavailable")),
    )
    monkeypatch.setattr(
        "agents.classifier.invoke_codex",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("Codex unavailable")),
    )

    result = await TicketProcessor(auto_index=False).process_ticket(_payload())

    assert result["classification"] == "support"
    assert result["delivery_mode"] == "draft"
    assert len(calls) == len(helpscout.drafts) == 1
    assert helpscout.closed == helpscout.sent_replies == []


# --- A message meant for another business -------------------------------------------------

class _Sent:
    def __init__(self, ok=True, detail=None):
        self.ok, self.detail = ok, detail


def _wire_wrong_address(monkeypatch, tmp_path, *, reading=(True, 0.95, "A guest asks for a table."),
                        flagged=True, native=True, outcome=None):
    helpscout, vector, leases = _wire_common(monkeypatch, tmp_path)
    if flagged:
        helpscout.conversation_payloads[987]["tags"].append({"tag": "possible-wrong-address"})
    helpscout.conversation_payloads[987]["status"] = "active"
    monkeypatch.setattr("services.processor.classify_ticket", AsyncMock(return_value="wrong_address"))
    confirm = AsyncMock(return_value=reading)
    monkeypatch.setattr("services.processor.confirm_wrong_address", confirm)
    if native:
        monkeypatch.setenv("SUPA_MAILBOX_ORIGIN", "https://mailbox.woodland.example")
    else:
        monkeypatch.delenv("SUPA_MAILBOX_ORIGIN", raising=False)
    runs = []

    async def run(conversation_id):
        runs.append(conversation_id)
        return outcome or _Sent()

    helpscout.run_wrong_address_workflow_result = run
    helpscout.confirm = confirm
    return helpscout, vector, leases, runs


async def test_agreed_wrong_address_message_gets_the_mailbox_reply(monkeypatch, tmp_path):
    helpscout, _, leases, runs = _wire_wrong_address(monkeypatch, tmp_path)
    monkeypatch.setattr("services.processor.run_support_pipeline",
                        lambda **kwargs: pytest.fail("A wrong-address message is not drafted"))

    result = await TicketProcessor(auto_index=False).process_ticket(_payload())

    assert runs == [987]
    assert (result["classification"], result["status"]) == ("wrong_address", "processed")
    assert result["response_posted"] is True and result["delivery_mode"] == "wrong_address_reply"
    assert "0.95" in result["decision_note"]
    # The workflow closes and files the conversation itself; Supa only adds its tags,
    # reading the list again because the workflow changed the conversation.
    assert helpscout.closed == [] and helpscout.sent_replies == helpscout.drafts == []
    assert helpscout.tag_updates == [(987, ["supa-processed", "wrong-address"], [])]
    row = get_ticket_log_by_helpscout_id(str(tmp_path / "supa.db"), "987")
    assert row["classification"] == "wrong_address" and row["response_posted"]
    assert row["delivery_mode"] == "wrong_address_reply" and "sent" in row["decision_note"]
    leases.assert_mutation_guarded(987, "run")
    leases.assert_mutation_guarded(987, "add_tags")


@pytest.mark.parametrize("case", ["low_confidence", "not_flagged", "vendor_mailbox", "draft_only", "closed",
                                  "check_unavailable", "draft_ceiling"])
async def test_wrong_address_reply_is_held_unless_every_gate_holds(monkeypatch, tmp_path, case):
    reading = (True, 0.80 if case == "low_confidence" else 0.95, "Probably a guest.")
    if case == "check_unavailable":
        reading = (None, 0.0, "Check unavailable.")
    helpscout, _, _, runs = _wire_wrong_address(
        monkeypatch, tmp_path, reading=reading,
        flagged=case != "not_flagged", native=case != "vendor_mailbox")
    monkeypatch.setattr("services.processor.run_support_pipeline",
                        lambda **kwargs: pytest.fail("A held wrong-address message is not drafted"))
    if case == "draft_only":
        monkeypatch.setenv("SUPA_SUPPORT_MODE", "draft_only")
    if case == "closed":
        helpscout.conversation_payloads[987]["status"] = "closed"
    kwargs = {"trigger": "manual_run", "delivery_ceiling": "draft"} if case == "draft_ceiling" else {}

    result = await TicketProcessor(auto_index=False).process_ticket(_payload(), **kwargs)

    assert runs == []
    assert helpscout.closed == [] and helpscout.sent_replies == helpscout.drafts == []
    assert result["classification"] == "wrong_address" and result["response_posted"] is False
    assert result["decision_note"].startswith("Wrong-address reply held:")
    added = helpscout.tag_updates[-1][1]
    # Supa's own hold tags only: it must never write the tag its intake gate reads.
    assert "supa-wrong-address-held" in added and "supa-triage-review" in added
    assert "possible-wrong-address" not in added and "wrong-address" not in added
    row = get_ticket_log_by_helpscout_id(str(tmp_path / "supa.db"), "987")
    assert row["classification"] == "wrong_address" and not row["response_posted"]
    assert row["delivery_mode"] == "wrong_address_held" and row["decision_note"].startswith("Wrong-address reply held:")


async def test_a_hold_cannot_open_the_gate_for_a_later_pass(monkeypatch, tmp_path):
    helpscout, _, _, runs = _wire_wrong_address(monkeypatch, tmp_path, flagged=False)
    processor = TicketProcessor(auto_index=False)
    await processor.process_ticket(_payload())
    # The mailbox now shows what Supa wrote; a second pass with a confident reading still holds.
    helpscout.conversation_payloads[987]["tags"] += [{"tag": t} for t in helpscout.tag_updates[-1][1]]
    result = await processor.process_ticket(_payload())
    assert runs == [] and result["response_posted"] is False


@pytest.mark.parametrize("case", ["second_customer_message", "draft_present", "published_reply", "other_thread_classified"])
async def test_only_a_lone_first_message_can_be_a_wrong_address_message(monkeypatch, tmp_path, case):
    helpscout, vector, _, runs = _wire_wrong_address(monkeypatch, tmp_path)
    calls = _wire_pipeline(monkeypatch, vector)
    threads = helpscout.conversation_payloads[987]["_embedded"]["threads"]
    payload = _payload()
    if case == "second_customer_message":
        later = {"id": 5, "type": "customer", "createdAt": "2026-04-04T09:00:00Z", "createdBy": {"type": "customer"},
                 "body": "<p>Guests cannot reserve a table for Friday.</p>", "source": {"type": "email"}}
        threads.append(later)
        payload["_embedded"]["threads"].append(dict(later))
    elif case == "draft_present":
        threads.append({"id": 6, "type": "message", "state": "draft", "createdBy": {"type": "user"}, "createdAt": "2026-04-03T09:30:00Z"})
    elif case == "published_reply":
        threads.append({"id": 7, "type": "message", "state": "published", "createdBy": {"type": "user"}, "createdAt": "2026-04-03T09:30:00Z"})
    else:
        payload["_embedded"]["threads"][0]["id"] = 99

    result = await TicketProcessor(auto_index=False).process_ticket(payload)

    assert runs == []
    helpscout.confirm.assert_not_awaited()
    assert result["classification"] == "support" and len(calls) == 1


async def test_wrong_address_guess_the_second_reading_rejects_is_drafted_as_support(monkeypatch, tmp_path):
    helpscout, vector, _, runs = _wire_wrong_address(monkeypatch, tmp_path, reading=(False, 0.9, "A theme customer."))
    calls = _wire_pipeline(monkeypatch, vector)

    result = await TicketProcessor(auto_index=False).process_ticket(_payload())

    assert runs == [] and len(calls) == 1
    assert result["classification"] == "support" and result["delivery_mode"] == "draft"


@pytest.mark.parametrize("outcome", [_Sent(ok=False, detail="wrong_address_workflow_not_found"), RuntimeError("network")])
async def test_wrong_address_workflow_failure_is_not_logged_as_sent(monkeypatch, tmp_path, outcome):
    helpscout, _, _, runs = _wire_wrong_address(monkeypatch, tmp_path)
    if isinstance(outcome, Exception):
        helpscout.run_wrong_address_workflow_result = AsyncMock(side_effect=outcome)
    else:
        helpscout.run_wrong_address_workflow_result = AsyncMock(return_value=outcome)

    result = await TicketProcessor(auto_index=False).process_ticket(_payload())

    assert result["status"] == "error" and result["response_posted"] is False
    assert helpscout.tag_updates == []
    row = get_ticket_log_by_helpscout_id(str(tmp_path / "supa.db"), "987")
    assert row["status"] == "error" and row["error_message"]


async def test_tag_failure_after_the_reply_is_a_warning_about_a_sent_reply(monkeypatch, tmp_path):
    helpscout, _, _, runs = _wire_wrong_address(monkeypatch, tmp_path)
    helpscout.fail_on_tags = True

    result = await TicketProcessor(auto_index=False).process_ticket(_payload())

    assert runs == [987]
    assert result["status"] == "warning" and result["response_posted"] is True
    row = get_ticket_log_by_helpscout_id(str(tmp_path / "supa.db"), "987")
    assert row["response_posted"] and row["delivery_mode"] == "wrong_address_reply"


@pytest.mark.parametrize("status", ["active", "closed"])
async def test_an_answered_wrong_address_conversation_is_never_answered_again(monkeypatch, tmp_path, status):
    helpscout, _, _, runs = _wire_wrong_address(monkeypatch, tmp_path)
    conversation = helpscout.conversation_payloads[987]
    conversation["tags"].append({"tag": "wrong-address"})
    conversation["status"] = status
    conversation["_embedded"]["threads"] += [
        {"id": 7, "type": "message", "state": "published", "createdBy": {"type": "user"}, "createdAt": "2026-04-03T09:30:00Z"},
        {"id": 8, "type": "customer", "createdAt": "2026-04-04T09:00:00Z", "createdBy": {"type": "customer"}, "body": "<p>Oh, thanks!</p>"}]

    result = await TicketProcessor(auto_index=False).process_ticket(_payload())

    assert runs == [] and helpscout.tag_updates == [] and helpscout.sent_replies == helpscout.drafts == []
    helpscout.confirm.assert_not_awaited()
    # A thank-you reopened it: close it again. A redelivery on a closed one changes nothing.
    assert helpscout.closed == ([987] if status == "active" else [])
    assert result["classification"] == "wrong_address" and result["response_posted"] is False
    assert get_ticket_log_by_helpscout_id(str(tmp_path / "supa.db"), "987") is None  # The earlier audit row is left alone.


async def test_manual_handling_tag_stops_the_wrong_address_reply(monkeypatch, tmp_path):
    helpscout, _, leases, runs = _wire_wrong_address(monkeypatch, tmp_path)
    helpscout.conversation_payloads[987]["tags"].append({"tag": "supa-ignore"})

    result = await TicketProcessor(auto_index=False).process_ticket(_payload())

    assert result["status"] == "skipped" and runs == [] and helpscout.tag_updates == []


@pytest.mark.parametrize("value,sent", [("0.99", False), ("0.9", True), ("nonsense", True), ("0.1", True), ("7", True)])
async def test_confidence_threshold_setting(monkeypatch, tmp_path, value, sent):
    helpscout, _, _, runs = _wire_wrong_address(monkeypatch, tmp_path)
    monkeypatch.setenv("SUPA_WRONG_ADDRESS_MIN_CONFIDENCE", value)
    await TicketProcessor(auto_index=False).process_ticket(_payload())
    # A reading of 0.95 clears the default 0.90; an unusable setting falls back to that default.
    assert bool(runs) is sent
