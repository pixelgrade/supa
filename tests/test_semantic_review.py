from agents.semantic_review import (
    SemanticReviewResult,
    fallback_semantic_review_result,
    parse_semantic_review_result,
    review_sent_reply,
)


def test_parse_semantic_review_result_normalizes_same_meaning():
    result = parse_semantic_review_result(
        """
        {
          "same_meaning": true,
          "change_type": "supported_vs_unsupported",
          "summary": "Only tone changed.",
          "wrong_claim": "Wrong",
          "correct_claim": "Correct"
        }
        """
    )

    assert result.same_meaning is True
    assert result.change_type == "none"
    assert result.summary == "Only tone changed."
    assert result.wrong_claim == ""
    assert result.correct_claim == ""


def test_parse_semantic_review_result_preserves_meaning_change():
    result = parse_semantic_review_result(
        """
        {
          "same_meaning": false,
          "change_type": "built_in_vs_custom",
          "summary": "The human reply removed a made-up built-in feature claim.",
          "wrong_claim": "The feature exists as a built-in setting.",
          "correct_claim": "The feature requires custom code."
        }
        """,
        provider="claude",
    )

    assert result == SemanticReviewResult(
        same_meaning=False,
        change_type="built_in_vs_custom",
        summary="The human reply removed a made-up built-in feature claim.",
        wrong_claim="The feature exists as a built-in setting.",
        correct_claim="The feature requires custom code.",
        provider="claude",
    )


def test_fallback_semantic_review_result_is_safe():
    result = fallback_semantic_review_result("claude", "Evaluator failed")

    assert result.same_meaning is True
    assert result.change_type == "unknown"
    assert result.summary == "Evaluator failed"
    assert result.provider == "claude"


def test_semantic_model_returns_a_structured_comparison(monkeypatch):
    def fake_invoke(prompt, **kwargs):
        assert "Supa draft reply" in prompt
        assert "Final human reply" in prompt
        return """
        {
          "same_meaning": false,
          "change_type": "supported_vs_unsupported",
          "summary": "The final reply changed the product capability claim.",
          "wrong_claim": "The journal sorts index entries alphabetically.",
          "correct_claim": "The journal sorts entries by their position in the book."
        }
        """

    monkeypatch.setattr("agents.semantic_review.invoke_claude", fake_invoke)

    result = review_sent_reply(
        subject="Ordering a journal index",
        customer_message="Can I alphabetize index entries?",
        supa_draft_html="<p>The journal sorts index entries alphabetically.</p>",
        final_reply_html="<p>The journal sorts entries by their position in the book.</p>",
        theme_tag="field-journal",
        confidence_summary="Accepted at 9/10",
        evidence_summary="Past answer says unsupported.",
    )

    assert result.same_meaning is False
    assert result.change_type == "supported_vs_unsupported"
    assert result.wrong_claim == "The journal sorts index entries alphabetically."
