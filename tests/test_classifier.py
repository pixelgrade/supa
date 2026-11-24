import pytest

import agents.classifier as classifier


@pytest.mark.asyncio
async def test_classify_ticket_uses_claude_when_available(monkeypatch):
    monkeypatch.setattr("agents.classifier.invoke_claude", lambda **_kwargs: "marketing")
    monkeypatch.setattr(
        "agents.classifier.invoke_codex",
        lambda **_kwargs: pytest.fail("codex fallback should not run"),
    )

    result = await classifier.classify_ticket("Guest post idea", "We want to collaborate")

    assert result == "marketing"


@pytest.mark.asyncio
async def test_classify_ticket_falls_back_to_codex_when_claude_fails(monkeypatch):
    monkeypatch.setattr(
        "agents.classifier.invoke_claude",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("claude unavailable")),
    )

    captured = {}

    def fake_codex(*, prompt, output_schema, timeout=60):
        captured["prompt"] = prompt
        captured["output_schema"] = output_schema
        captured["timeout"] = timeout
        return '{"classification":"admin"}'

    monkeypatch.setattr("agents.classifier.invoke_codex", fake_codex)

    result = await classifier.classify_ticket("Delivery Status Notification", "This is an automated bounce.")

    assert result == "admin"
    assert captured["output_schema"]["properties"]["classification"]["enum"] == [
        "support",
        "admin",
        "marketing",
        "spam",
        "wrong_address",
    ]
    assert classifier.CLASSIFIER_PROMPT in captured["prompt"]


@pytest.mark.asyncio
async def test_classify_ticket_uses_codex_during_claude_outage(monkeypatch):
    monkeypatch.setattr("agents.classifier.should_skip_claude", lambda: True)
    monkeypatch.setattr("agents.classifier.invoke_claude", lambda **_: pytest.fail("Claude should be skipped"))
    monkeypatch.setattr("agents.classifier.invoke_codex", lambda **_: '{"classification":"support"}')
    assert await classifier.classify_ticket("Help", "Site issue") == "support"


@pytest.mark.asyncio
async def test_generate_subject_uses_codex_during_claude_outage(monkeypatch):
    monkeypatch.setattr("agents.classifier.should_skip_claude", lambda: True)
    monkeypatch.setattr("agents.classifier.invoke_claude", lambda **_: pytest.fail("Claude should be skipped"))
    monkeypatch.setattr("agents.classifier.invoke_codex", lambda **_: '{"subject":"Field Journal: Header font size"}')
    assert await classifier.generate_subject("Help", "Header text is too large") == "Field Journal: Header font size"


@pytest.mark.asyncio
async def test_classify_ticket_defaults_to_support_when_both_providers_fail(monkeypatch):
    monkeypatch.setattr(
        "agents.classifier.invoke_claude",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("claude unavailable")),
    )
    monkeypatch.setattr(
        "agents.classifier.invoke_codex",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("codex unavailable")),
    )

    result = await classifier.classify_ticket("Need help", "The site is broken")

    assert result == "support"


@pytest.mark.asyncio
async def test_generate_subject_uses_claude_when_available(monkeypatch):
    monkeypatch.setattr("agents.classifier.invoke_claude", lambda **_kwargs: "Field Journal: Header font size")
    monkeypatch.setattr(
        "agents.classifier.invoke_codex",
        lambda **_kwargs: pytest.fail("codex fallback should not run"),
    )

    result = await classifier.generate_subject("Help", "How do I resize the header text?", "field-journal")

    assert result == "Field Journal: Header font size"


@pytest.mark.asyncio
async def test_generate_subject_falls_back_to_codex_when_claude_fails(monkeypatch):
    monkeypatch.setattr(
        "agents.classifier.invoke_claude",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("claude unavailable")),
    )

    captured = {}

    def fake_codex(*, prompt, output_schema, timeout=60):
        captured["prompt"] = prompt
        captured["output_schema"] = output_schema
        captured["timeout"] = timeout
        return '{"subject":"Field Journal: Header font size"}'

    monkeypatch.setattr("agents.classifier.invoke_codex", fake_codex)

    result = await classifier.generate_subject("Help", "How do I resize the header text?", "field-journal")

    assert result == "Field Journal: Header font size"
    assert captured["output_schema"]["properties"]["subject"]["type"] == "string"
    assert classifier.SUBJECT_GENERATOR_PROMPT in captured["prompt"]


@pytest.mark.asyncio
async def test_generate_subject_returns_original_when_both_providers_fail(monkeypatch):
    monkeypatch.setattr(
        "agents.classifier.invoke_claude",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("claude unavailable")),
    )
    monkeypatch.setattr(
        "agents.classifier.invoke_codex",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("codex unavailable")),
    )

    result = await classifier.generate_subject("Help", "How do I resize the header text?", "field-journal")

    assert result == "Help"


@pytest.mark.asyncio
async def test_classifier_accepts_the_wrong_address_class(monkeypatch):
    monkeypatch.setattr("agents.classifier.invoke_claude", lambda **_kwargs: " Wrong-Address\n")
    assert await classifier.classify_ticket("Table for four", "Could we book for Friday at eight?") == "wrong_address"
    assert "wrong_address" in classifier.CLASSIFIER_PROMPT
    with pytest.raises(ValueError):
        classifier._normalize_classification("reservation")


@pytest.mark.asyncio
async def test_wrong_address_check_returns_the_second_reading(monkeypatch):
    seen = {}

    def claude(**kwargs):
        seen.update(kwargs)
        return 'Here: {"wrong_address": true, "confidence": 0.96, "reason": "A guest asks a restaurant for a table."}'

    monkeypatch.setattr("agents.classifier.invoke_claude", claude)
    assert await classifier.confirm_wrong_address("Table", "Book for four") == (
        True, 0.96, "A guest asks a restaurant for a table.")
    assert seen["system_prompt"] == classifier.WRONG_ADDRESS_CHECK_PROMPT
    assert "Book for four" in seen["prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["not json", '{"wrong_address": "yes", "confidence": 0.9}',
                                    '{"wrong_address": true, "confidence": 7}', '{"wrong_address": true}',
                                    '{"wrong_address": true, "confidence": true}',
                                    '{"wrong_address": true, "confidence": "0.99"}'])
async def test_unreadable_wrong_address_check_is_unavailable_not_a_no(monkeypatch, output):
    monkeypatch.setattr("agents.classifier.invoke_claude", lambda **_: output)
    monkeypatch.setattr("agents.classifier.invoke_codex", lambda **_: output)
    assert (await classifier.confirm_wrong_address("Table", "Book for four"))[:2] == (None, 0.0)


@pytest.mark.asyncio
async def test_wrong_address_check_uses_codex_when_claude_is_down(monkeypatch):
    monkeypatch.setattr("agents.classifier.should_skip_claude", lambda: True)
    monkeypatch.setattr("agents.classifier.invoke_claude", lambda **_: pytest.fail("Claude should be skipped"))
    monkeypatch.setattr("agents.classifier.invoke_codex",
                        lambda **_: '{"wrong_address": false, "confidence": 0.8, "reason": "Theme customer."}')
    assert await classifier.confirm_wrong_address("Help", "My Field Journal site") == (False, 0.8, "Theme customer.")
