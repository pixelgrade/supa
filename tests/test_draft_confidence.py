import pytest

from agents.draft_confidence import (
    CONFIDENCE_OUTPUT_SCHEMA,
    CONFIDENCE_SYSTEM_PROMPT,
    evaluate_draft,
    parse_confidence_result,
)


def test_codex_confidence_schema_requires_every_declared_field():
    # OpenAI rejects a strict response schema when a property is omitted from required.
    assert set(CONFIDENCE_OUTPUT_SCHEMA["required"]) == set(CONFIDENCE_OUTPUT_SCHEMA["properties"])


def test_parse_confidence_json_response():
    raw = (
        '{"score":7,"verdict":"retry","summary":"Too generic",'
        '"investigation_actions":["expand_kb_search"],'
        '"rewrite_brief":"Address the actual symptom first."}'
    )

    result = parse_confidence_result(raw)

    assert result.score == 7
    assert result.verdict == "retry"
    assert result.summary == "Too generic"
    assert result.investigation_actions == ["expand_kb_search"]
    assert result.rewrite_brief == "Address the actual symptom first."


def test_parse_confidence_json_response_includes_explicit_hard_blockers():
    raw = (
        '{"score":10,"verdict":"accept","summary":"Accurate",'
        '"missing_information":[],"investigation_actions":[],"rewrite_brief":"",'
        '"risk_flags":[],"hard_blockers":["missing_critical_evidence","malformed_provider_output"]}'
    )

    result = parse_confidence_result(raw)

    assert result.score == 10
    assert result.verdict == "accept"
    assert result.hard_blockers == ["missing_critical_evidence", "malformed_provider_output"]


def test_parse_confidence_rejects_non_list_hard_blockers():
    raw = (
        '{"score":10,"verdict":"accept","summary":"Accurate",'
        '"missing_information":[],"investigation_actions":[],"rewrite_brief":"",'
        '"risk_flags":[],"hard_blockers":"malformed_provider_output"}'
    )

    with pytest.raises(ValueError) as exc:
        parse_confidence_result(raw)
    assert "hard_blockers" in str(exc.value)


def test_evaluate_draft_fails_closed_for_malformed_hard_blockers(monkeypatch):
    def fake_claude(**_kwargs):
        return (
            '{"score":10,"verdict":"accept","summary":"Accurate",'
            '"missing_information":[],"investigation_actions":[],"rewrite_brief":"",'
            '"risk_flags":[],"hard_blockers":"malformed_provider_output"}'
        )

    monkeypatch.setattr("agents.draft_confidence.invoke_claude", fake_claude)

    result = evaluate_draft(
        customer_name="Mica",
        subject="Map issue",
        message="Street is not recognized",
        draft_html="<p>Draft</p>",
        provider="claude",
    )

    assert result.verdict == "manual_review"
    assert "evaluator_failure" in result.risk_flags
    assert "malformed_provider_output" in result.hard_blockers


def test_evaluate_draft_uses_openclaw_provider(monkeypatch):
    def fake_openclaw(prompt, timeout=180):
        assert "Draft reply HTML:" in prompt
        return '{"score":9,"verdict":"accept","summary":"good","investigation_actions":[],"rewrite_brief":""}'

    monkeypatch.setattr("agents.draft_confidence.invoke_openclaw", fake_openclaw)

    result = evaluate_draft(
        customer_name="Mica",
        subject="Map issue",
        message="Street is not recognized",
        draft_html="<p>Draft</p>",
        provider="openclaw",
    )

    assert result.score == 9
    assert result.verdict == "accept"
    assert result.provider == "openclaw"


def test_evaluate_draft_uses_codex_provider(monkeypatch):
    captured = {}

    def fake_codex(*, prompt, output_schema, timeout=180, working_dir=None, retries=1, text_only=False):
        captured["prompt"] = prompt
        captured["output_schema"] = output_schema
        captured["timeout"] = timeout
        captured["working_dir"] = working_dir
        captured["text_only"] = text_only
        return (
            '{"score":8,"verdict":"accept","summary":"good",'
            '"missing_information":[],"investigation_actions":[],"rewrite_brief":"",'
            '"risk_flags":[],"hard_blockers":[]}'
        )

    monkeypatch.setattr("agents.draft_confidence.invoke_codex", fake_codex)

    result = evaluate_draft(
        customer_name="Mica",
        subject="Map issue",
        message="Street is not recognized",
        draft_html="<p>Draft</p>",
        provider="codex",
    )

    assert "Draft reply HTML:" in captured["prompt"]
    assert captured["output_schema"]["type"] == "object"
    # Evaluator runs Codex in a scoped scratch dir (no repo/project context load).
    assert captured["working_dir"] and "supa-codex-eval" in captured["working_dir"]
    assert captured["text_only"] is True
    assert result.score == 8
    assert result.verdict == "accept"
    assert result.provider == "codex"
