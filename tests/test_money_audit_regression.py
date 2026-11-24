import json

import pytest

from services.support_pipeline import SupportPipelineContext, _money_decision


@pytest.mark.parametrize(("role", "earlier", "question", "reply", "held"), [
    ("customer", "I would like a refund.", "Is there an update?", "I will check.", True),
    ("customer", "I have a journal.", "Where is my index?", "I can offer you a discount.", True),
    ("support", "You can upgrade the index format.", "Where is my index?", "Open the index.", False),
    ("customer", "The index is empty.", "How do I upgrade the software?", "Use the update screen.", False),
])
def test_commercial_holds_follow_the_request_and_the_author(role, earlier, question, reply, held):
    context = SupportPipelineContext(customer_name="Reader", subject="Journal",
        message=question, conversation_history=json.dumps([{"role": role, "message": earlier}]))
    assert _money_decision(context=context, reply_html=reply) is held
