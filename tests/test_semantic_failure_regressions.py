import pytest

from services.support_pipeline import _has_semantic_failure_match


@pytest.mark.parametrize(("reply", "repeated"), [
    ("The journal can sort every index entry by name.", True),
    ("The journal can not sort every index entry by name.", False),
    ("You can add another entry to the journal index.", False),
    ("The journal can sort every index entry by name. It cannot number pages.", True),
])
def test_precedent_requires_a_repeated_claim_with_the_same_polarity(reply, repeated):
    failures = [{"conversation_id": 42, "wrong_claim": "The journal can sort every index entry by name.",
                 "correct_claim": "Entries appear in page order."}]
    assert _has_semantic_failure_match(reply, failures) is repeated
