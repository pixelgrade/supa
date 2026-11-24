from unittest.mock import AsyncMock

import pytest

from api.webhooks import handle_agent_reply
from db.database import (
    get_semantic_failures_by_helpscout_id,
    init_db,
    log_ticket,
)
from services.semantic_failure_learning import build_auto_improvement_note, learn_from_sent_reply
from agents.semantic_review import SemanticReviewResult


class FakeHelpScout:
    def __init__(self):
        self.notes = []

    async def add_note(self, conversation_id, text):
        self.notes.append((conversation_id, text))
        return True


class StatusChangingHelpScout(FakeHelpScout):
    def __init__(self):
        super().__init__()
        self.status = "pending"
        self.status_updates = []

    async def get_conversation(self, conversation_id):
        return {"id": conversation_id, "status": self.status}

    async def add_note(self, conversation_id, text):
        self.notes.append((conversation_id, text))
        self.status = "active"
        return True

    async def ensure_conversation_status(self, conversation_id, status):
        self.status_updates.append((conversation_id, status))
        self.status = status
        return {"verified": True, "conversation": {"id": conversation_id, "status": status}}


def test_build_auto_improvement_note_mentions_no_action_needed():
    note = build_auto_improvement_note(
        SemanticReviewResult(
            same_meaning=False,
            change_type="built_in_vs_custom",
            summary="The final reply corrected the implementation path.",
            wrong_claim="It is a built-in feature.",
            correct_claim="It requires custom work.",
            provider="claude",
        )
    )

    assert "Supa auto-improvement note" in note
    assert "built_in_vs_custom" in note
    assert "No action needed here" in note


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["workflow_source", "wrong_address_tag", "filed_tag", "site_notice_tag"])
async def test_workflow_replies_are_not_learned_from(monkeypatch, case):
    """The wrong-address reply is fixed wording; indexing it would teach it as an answer."""
    reply = {"type": "message", "body": "<p>Your message did not reach the venue.</p>", "state": "published",
             "createdBy": {"type": "user"}, "createdAt": "2026-11-24T10:00:00Z"}
    tags = []
    if case == "workflow_source":
        reply["source"] = {"type": "manual-workflow", "via": "user"}
    elif case == "wrong_address_tag":
        tags = [{"tag": "wrong-address"}]
    elif case == "site_notice_tag":
        tags = [{"tag": "site-mail-notice"}]
    else:
        tags = [{"tag": "non-support"}]
    payload = {"id": 1, "subject": "Table for four", "primaryCustomer": {"first": "Reader"}, "tags": tags,
               "_embedded": {"threads": [reply, {"type": "customer", "body": "<p>Book for four?</p>",
                                                 "createdBy": {"type": "customer"}, "createdAt": "2026-11-24T09:00:00Z"}]}}
    monkeypatch.setattr("api.webhooks.get_processor", lambda: pytest.fail("A workflow reply must not be indexed"))
    monkeypatch.setattr("api.webhooks.learn_from_sent_reply", AsyncMock(side_effect=AssertionError("not learned")))

    response = await handle_agent_reply(payload)

    assert response.status == "skipped" and "Workflow" in response.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
async def test_semantic_learning_links_only_corrections_to_the_source(tmp_path, monkeypatch, changed):
    db_path = str(tmp_path / "audit.db")
    init_db(db_path)
    log_ticket(db_path=db_path, helpscout_id="52", subject="Ordering an index",
        customer_name="Reader", classification="support", status="processed", response_posted=True,
        draft_response_html="<p>The order is alphabetical.</p>", confidence_score=9)
    monkeypatch.setattr("services.semantic_failure_learning.review_sent_reply", lambda **kwargs:
        SemanticReviewResult(same_meaning=not changed, change_type="capability" if changed else "none",
            summary="Checked the ordering rule.", wrong_claim="The order is alphabetical.",
            correct_claim="Entries follow page order.", provider="fixture"))
    client = FakeHelpScout()
    outcome = await learn_from_sent_reply(db_path=db_path, helpscout_client=client, conversation_id=52,
        subject="Ordering an index", customer_message="How are entries ordered?",
        final_reply_html="<p>Entries follow page order.</p>", theme_tag="field-journal")
    assert outcome.recorded is changed and outcome.note_added is changed
    rows = get_semantic_failures_by_helpscout_id(db_path, "52")
    assert len(rows) == int(changed)
    if changed:
        assert rows[0]["correct_claim"] == "Entries follow page order."
        assert client.notes[0][0] == 52
