import pytest

import agents.triage as triage
from agents.triage import (
    TriageConfig,
    decide_action,
    load_triage_config,
    triage_ticket,
)


# --- decision rules -------------------------------------------------------

def test_decide_action_support_continues_pipeline():
    action, tags = decide_action("support", 0.4, autoclose_min=0.90)
    assert action == "process_support"
    assert tags == []


def test_decide_action_high_confidence_nonsupport_closes():
    action, tags = decide_action("marketing", 0.95, autoclose_min=0.90)
    assert action == "close"
    assert tags == ["marketing"]


def test_decide_action_low_confidence_nonsupport_reviews():
    action, tags = decide_action("spam", 0.6, autoclose_min=0.90)
    assert action == "review"
    assert tags == ["supa-triage-review"]


def test_decide_action_threshold_is_inclusive():
    action, _ = decide_action("admin", 0.90, autoclose_min=0.90)
    assert action == "close"


# --- config ---------------------------------------------------------------

def test_load_triage_config_defaults_to_shadow(monkeypatch):
    monkeypatch.delenv("SUPA_TRIAGE_MODE", raising=False)
    monkeypatch.delenv("SUPA_TRIAGE_AUTOCLOSE_MIN_CONFIDENCE", raising=False)
    monkeypatch.delenv("SUPA_TRIAGE_PROVIDER", raising=False)

    cfg = load_triage_config()

    assert cfg.mode == "shadow"
    assert cfg.autoclose_min_confidence == 0.90
    assert cfg.provider == "claude"


def test_load_triage_config_reads_overrides(monkeypatch):
    monkeypatch.setenv("SUPA_TRIAGE_MODE", "live")
    monkeypatch.setenv("SUPA_TRIAGE_AUTOCLOSE_MIN_CONFIDENCE", "0.75")
    monkeypatch.setenv("SUPA_TRIAGE_PROVIDER", "codex")

    cfg = load_triage_config()

    assert cfg.mode == "live"
    assert cfg.autoclose_min_confidence == 0.75
    assert cfg.provider == "codex"


def test_load_triage_config_rejects_unknown_mode_and_bad_threshold(monkeypatch):
    monkeypatch.setenv("SUPA_TRIAGE_MODE", "wild")
    monkeypatch.setenv("SUPA_TRIAGE_AUTOCLOSE_MIN_CONFIDENCE", "not-a-number")

    cfg = load_triage_config()

    assert cfg.mode == "shadow"
    assert cfg.autoclose_min_confidence == 0.90


# --- provider execution ---------------------------------------------------

@pytest.mark.asyncio
async def test_triage_ticket_uses_claude_when_available(monkeypatch):
    monkeypatch.setattr(
        "agents.triage.invoke_claude",
        lambda **_kwargs: '{"classification":"marketing","confidence":0.97,"reason":"Cold guest-post pitch."}',
    )
    monkeypatch.setattr(
        "agents.triage.invoke_codex",
        lambda **_kwargs: pytest.fail("codex fallback should not run"),
    )

    result = await triage_ticket(
        subject="Guest post idea",
        message="We want to collaborate",
        config=TriageConfig(mode="live", autoclose_min_confidence=0.90, provider="claude"),
    )

    assert result.classification == "marketing"
    assert result.confidence == 0.97
    assert result.recommended_action == "close"
    assert result.tags == ["marketing"]
    assert "guest-post" in result.reason.lower()


@pytest.mark.asyncio
async def test_triage_ticket_parses_claude_output_with_code_fences(monkeypatch):
    fenced = '```json\n{"classification":"support","confidence":0.8,"reason":"Theme bug."}\n```'
    monkeypatch.setattr("agents.triage.invoke_claude", lambda **_kwargs: fenced)

    result = await triage_ticket(subject="Almanac broken", message="Header gone")

    assert result.classification == "support"
    assert result.recommended_action == "process_support"


@pytest.mark.asyncio
async def test_triage_ticket_falls_back_to_codex_when_claude_fails(monkeypatch):
    monkeypatch.setattr(
        "agents.triage.invoke_claude",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("claude unavailable")),
    )

    captured = {}

    def fake_codex(*, prompt, output_schema, timeout=60):
        captured["output_schema"] = output_schema
        return '{"classification":"admin","confidence":0.99,"reason":"Bounce notification."}'

    monkeypatch.setattr("agents.triage.invoke_codex", fake_codex)

    result = await triage_ticket(
        subject="Delivery Status Notification",
        message="This is an automated bounce.",
        config=TriageConfig(mode="live", autoclose_min_confidence=0.90, provider="claude"),
    )

    assert result.classification == "admin"
    assert result.recommended_action == "close"
    assert captured["output_schema"]["properties"]["confidence"]["type"] == "number"


@pytest.mark.asyncio
async def test_triage_ticket_defaults_to_support_when_all_providers_fail(monkeypatch):
    monkeypatch.setattr(
        "agents.triage.invoke_claude",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("claude unavailable")),
    )
    monkeypatch.setattr(
        "agents.triage.invoke_codex",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("codex unavailable")),
    )

    result = await triage_ticket(subject="Need help", message="Site broken")

    assert result.classification == "support"
    assert result.confidence == 0.0
    assert result.recommended_action == "process_support"
    assert result.tags == []


@pytest.mark.asyncio
async def test_triage_ticket_defaults_to_support_on_malformed_output(monkeypatch):
    # Claude returns junk (no JSON); codex also fails -> safe support default.
    monkeypatch.setattr("agents.triage.invoke_claude", lambda **_kwargs: "marketing, probably")
    monkeypatch.setattr(
        "agents.triage.invoke_codex",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("codex unavailable")),
    )

    result = await triage_ticket(subject="???", message="???")

    assert result.classification == "support"
    assert result.recommended_action == "process_support"
