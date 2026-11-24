import pytest

from agents import support_agent
from agents.support_agent import SupportInvestigationRequest, draft_reply, parse_investigator_result


@pytest.fixture(autouse=True)
def _claude_available(monkeypatch):
    monkeypatch.setattr("agents.support_agent.should_skip_claude", lambda: False)


def test_parse_investigator_result_accepts_valid_structured_json():
    raw = (
        '{"reply_html":"<p>Hello</p>","confidence_score":9,'
        '"decision_summary":"Evidence is aligned.","evidence_summary":"Docs + repo checks",'
        '"investigation_steps":["Checked docs","Checked repo"],'
        '"risk_flags":[],"provider_name":"codex","requires_manual_review":false,'
        '"sources":["https://woodland.example/docs/example"]}'
    )

    result = parse_investigator_result(raw, provider="codex")

    assert result.reply_html == "<p>Hello</p>"
    assert result.confidence_score == 9
    assert result.decision_summary == "Evidence is aligned."
    assert result.evidence_summary == "Docs + repo checks"
    assert result.investigation_steps == ["Checked docs", "Checked repo"]
    assert result.risk_flags == []
    assert result.provider_name == "codex"
    assert result.requires_manual_review is False
    assert result.sources == ["https://woodland.example/docs/example"]


@pytest.mark.parametrize(
    "raw",
    [
        (
            '{"reply_html":"<p>Hello</p>","confidence_score":11,'
            '"decision_summary":"Evidence is aligned.","evidence_summary":"Docs + repo checks",'
            '"investigation_steps":["Checked docs"],"risk_flags":[],"provider_name":"codex",'
            '"requires_manual_review":false,"sources":[]}'
        ),
        (
            '{"reply_html":"<p>Hello</p>","confidence_score":8,'
            '"decision_summary":"Evidence is aligned.","evidence_summary":"Docs + repo checks",'
            '"investigation_steps":["Checked docs"],"risk_flags":[],"provider_name":"",'
            '"requires_manual_review":false,"sources":[]}'
        ),
        (
            '{"reply_html":"<p>Hello</p>","confidence_score":8,'
            '"decision_summary":"Evidence is aligned.","evidence_summary":"Docs + repo checks",'
            '"investigation_steps":["Checked docs"],"risk_flags":[],"requires_manual_review":false,'
            '"sources":[]}'
        ),
        (
            '{"reply_html":"<p>Hello</p>","confidence_score":8,'
            '"decision_summary":"Evidence is aligned.","evidence_summary":"Docs + repo checks",'
            '"investigation_steps":["Checked docs"],"risk_flags":[],"provider_name":"codex",'
            '"requires_manual_review":false,"sources":[],"extra":"unexpected"}'
        ),
    ],
)
def test_parse_investigator_result_rejects_strict_shape_violations(raw):
    with pytest.raises(ValueError):
        parse_investigator_result(raw, provider="codex")


def test_draft_reply_falls_back_for_malformed_provider_output(monkeypatch):
    monkeypatch.setattr("agents.support_agent.invoke_codex", lambda **_: "not-json")

    result = draft_reply(
        SupportInvestigationRequest(
            customer_name="Reader",
            subject="Journal display",
            message="Address lookup is not working.",
        ),
        models=("codex",),
    )

    assert result.requires_manual_review is True
    assert result.confidence_score == 0
    assert "failed" in result.decision_summary.lower()
    assert "investigator_failure" in result.risk_flags
    assert result.provider_name == "codex"


def test_draft_reply_falls_back_for_shape_violations(monkeypatch):
    monkeypatch.setattr(
        "agents.support_agent.invoke_codex",
        lambda **_: (
            '{"reply_html":"<p>Hello</p>","confidence_score":12,'
            '"decision_summary":"Evidence is aligned.","evidence_summary":"Docs + repo checks",'
            '"investigation_steps":[],"risk_flags":[],"provider_name":"codex",'
            '"requires_manual_review":false,"sources":[]}'
        ),
    )

    result = draft_reply(
        SupportInvestigationRequest(
            customer_name="Reader",
            subject="Journal display",
            message="Address lookup is not working.",
        ),
        models=("codex",),
    )

    assert result.requires_manual_review is True
    assert result.confidence_score == 0
    assert "failed" in result.decision_summary.lower()
    assert "investigator_failure" in result.risk_flags
    assert result.provider_name == "codex"


def test_draft_reply_falls_back_when_provider_raises(monkeypatch):
    def fake_codex(**_):
        raise RuntimeError("exit 1")

    monkeypatch.setattr("agents.support_agent.invoke_codex", fake_codex)

    result = draft_reply(
        SupportInvestigationRequest(
            customer_name="Reader",
            subject="Fatal error",
            message="Site crashes.",
        ),
        models=("codex",),
    )

    assert result.requires_manual_review is True
    assert result.confidence_score == 0
    assert "exit 1" in result.decision_summary
    assert result.provider_name == "codex"


_VALID = (
    '{"reply_html":"<p>%s</p>","confidence_score":8,'
    '"decision_summary":"summary","evidence_summary":"evidence",'
    '"investigation_steps":[],"risk_flags":[],"provider_name":"spoofed",'
    '"requires_manual_review":false,"sources":[]}'
)

_REQUEST = SupportInvestigationRequest(
    customer_name="Reader",
    subject="Menus",
    message="Menu not showing.",
    theme_tag="field-journal",
    knowledge_context="KB evidence",
    internal_note="Check the header block first.",
)


class _Models:
    """Fake Claude + Codex CLIs that record exactly what each drafting model received."""

    def __init__(self, monkeypatch, *, failing=()):
        self.calls = []
        self.failing = set(failing)

        def fake_claude(*, prompt, model, system_prompt, timeout, ignore_failure_cooldown=False):
            self.calls.append({
                "model": model, "system": system_prompt, "prompt": prompt,
                "timeout": timeout, "ignore_cooldown": ignore_failure_cooldown,
            })
            if model in self.failing:
                raise RuntimeError(f"{model} unavailable")
            return _VALID % model

        def fake_codex(*, prompt, output_schema, timeout, **_):
            self.calls.append({
                "model": "codex", "system": "", "prompt": prompt,
                "timeout": timeout, "schema": output_schema,
            })
            if "codex" in self.failing:
                raise RuntimeError("codex unavailable")
            return _VALID % "codex"

        monkeypatch.setattr("agents.support_agent.invoke_claude", fake_claude)
        monkeypatch.setattr("agents.support_agent.invoke_codex", fake_codex)

    @property
    def order(self):
        return [call["model"] for call in self.calls]

    def full_input(self, call):
        return f"{call['system']}\n\n{call['prompt']}"


def _rule_lines(text):
    return [line[2:].strip() for line in text.splitlines() if line.startswith("- ")]


def test_step_models_live_inside_one_agent():
    assert support_agent.CLASSIFY_MODEL == "haiku"
    assert support_agent.DRAFT_MODELS == ("opus", "sonnet", "codex")
    assert support_agent.LEARN_MODEL == "sonnet"


def test_opus_drafts_first_and_stops_the_chain(monkeypatch):
    models = _Models(monkeypatch)

    result = draft_reply(_REQUEST, timeout=77)

    assert models.order == ["opus"]
    assert models.calls[0]["timeout"] == 77
    assert result.model == "opus"
    assert result.provider_name == "claude"  # never the model's self-reported name
    assert result.reply_html == "<p>opus</p>"


def test_fallback_order_is_opus_then_sonnet_then_codex_with_the_same_instructions(monkeypatch):
    models = _Models(monkeypatch, failing={"opus", "sonnet"})

    result = draft_reply(_REQUEST, timeout=30)

    assert models.order == ["opus", "sonnet", "codex"]
    assert result.model == "codex"
    assert result.provider_name == "codex"
    assert "investigator_failure" not in result.risk_flags
    opus, sonnet, codex = models.calls
    # One instruction set and one ticket prompt: the fallbacks are the same agent.
    assert opus["system"] == sonnet["system"] == support_agent.SUPPORT_AGENT_INSTRUCTIONS
    assert opus["prompt"] == sonnet["prompt"]
    assert codex["prompt"] == f"{support_agent.SUPPORT_AGENT_INSTRUCTIONS}\n\n{opus['prompt']}"
    assert codex["schema"] is support_agent.INVESTIGATOR_OUTPUT_SCHEMA


def test_every_rule_in_the_single_rules_source_reaches_every_fallback(monkeypatch):
    rules = _rule_lines(support_agent.load_rules())
    assert rules, "The selected pack must exercise at least one drafting rule"
    models = _Models(monkeypatch, failing={"opus", "sonnet", "codex"})

    draft_reply(_REQUEST)

    assert models.order == ["opus", "sonnet", "codex"]
    for call in models.calls:
        seen = models.full_input(call)
        missing = [rule for rule in rules if rule not in seen]
        assert missing == [], f"{call['model']} did not get: {missing}"


def test_a_rule_written_once_applies_to_every_drafting_model(monkeypatch, tmp_path):
    rules_file = tmp_path / "rules.md"
    rules_file.write_text(
        support_agent.RULES_PATH.read_text(encoding="utf-8")
        + "\n- Never promise a release date.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("agents.support_agent.RULES_PATH", rules_file)
    models = _Models(monkeypatch, failing={"opus", "sonnet", "codex"})

    draft_reply(_REQUEST)

    assert len(models.calls) == 3
    for call in models.calls:
        assert "- Never promise a release date." in models.full_input(call)


def test_sonnet_fallback_gets_its_try_during_the_cooldown_opus_started(monkeypatch):
    models = _Models(monkeypatch, failing={"opus"})

    result = draft_reply(_REQUEST)

    assert models.order == ["opus", "sonnet"]
    assert [c["ignore_cooldown"] for c in models.calls] == [False, True]
    assert result.model == "sonnet"


def test_claude_outage_or_pause_drafts_with_codex_only(monkeypatch):
    monkeypatch.setattr("agents.support_agent.should_skip_claude", lambda: True)
    models = _Models(monkeypatch)

    result = draft_reply(_REQUEST)

    assert models.order == ["codex"]
    assert result.model == "codex"


def test_every_model_failing_returns_the_first_failure_for_manual_review(monkeypatch):
    models = _Models(monkeypatch, failing={"opus", "sonnet", "codex"})

    result = draft_reply(_REQUEST)

    assert models.order == ["opus", "sonnet", "codex"]
    assert result.requires_manual_review is True
    assert result.confidence_score == 0
    assert "investigator_failure" in result.risk_flags
    assert result.model == "opus"
    assert "opus unavailable" in result.decision_summary


def test_drafting_instructions_state_the_output_contract():
    system_prompt = support_agent.SUPPORT_AGENT_INSTRUCTIONS
    assert "JSON" in system_prompt
    assert "{{output_schema}}" not in system_prompt
    for name in support_agent.INVESTIGATOR_REQUIRED_FIELDS:
        assert name in system_prompt


def test_drafting_default_timeout_comes_from_agent_config(monkeypatch):
    monkeypatch.setenv("SUPA_SUPPORT_AGENT_TIMEOUT", "123")
    models = _Models(monkeypatch)

    draft_reply(_REQUEST)

    assert models.calls[0]["timeout"] == 123


# --- configuration and delivery gates ---------------------------------------

_MODE_ENV = ("SUPA_SUPPORT_MODE", "SUPA_SKILL_AGENT_MODE")


def _clear_mode(monkeypatch):
    for name in _MODE_ENV + (
        "SUPA_SUPPORT_AGENT_TIMEOUT", "SUPA_SKILL_AGENT_TIMEOUT",
        "SUPA_SUPPORT_AGENT_CONCURRENCY", "SUPA_SKILL_AGENT_CONCURRENCY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_config_defaults_to_draft_only(monkeypatch):
    _clear_mode(monkeypatch)
    cfg = support_agent.load_config()
    assert cfg.mode == "draft_only"
    assert cfg.timeout == 600
    assert cfg.concurrency == 2


def test_config_honours_the_legacy_skill_agent_names(monkeypatch):
    _clear_mode(monkeypatch)
    monkeypatch.setenv("SUPA_SKILL_AGENT_MODE", "live")
    monkeypatch.setenv("SUPA_SKILL_AGENT_TIMEOUT", "720")
    monkeypatch.setenv("SUPA_SKILL_AGENT_CONCURRENCY", "3")
    cfg = support_agent.load_config()
    assert (cfg.mode, cfg.timeout, cfg.concurrency) == ("live", 720, 3)


def test_new_mode_name_wins_and_garbage_means_draft_only(monkeypatch):
    _clear_mode(monkeypatch)
    monkeypatch.setenv("SUPA_SKILL_AGENT_MODE", "live")
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "draft_only")
    assert support_agent.load_config().mode == "draft_only"
    monkeypatch.setenv("SUPA_SUPPORT_MODE", "yes-send-everything")
    assert support_agent.load_config().mode == "draft_only"


@pytest.mark.parametrize(
    "raw,expected", [("", 8), ("8", 8), ("9", 9), ("10", 10), ("5", 8), ("7", 8), ("11", 10), ("x", 8)]
)
def test_autosend_threshold_defaults_to_eight_and_never_drops_below_it(monkeypatch, raw, expected):
    monkeypatch.setenv("SUPA_SUPPORT_AUTOSEND_MIN_SCORE", raw)
    assert support_agent.autosend_threshold() == expected


@pytest.mark.parametrize(
    "ceiling,mode,waiting,expected",
    [
        ("policy", "live", True, None),
        ("policy", "draft_only", True, "Mode: draft (auto-send suppressed by draft_only mode)"),
        ("draft", "live", True, "Mode: draft (auto-send suppressed by per-run draft ceiling)"),
        ("policy", "live", False, "Mode: draft (auto-send suppressed: the customer is not waiting)"),
        ("policy", "draft_only", False, "Mode: draft (auto-send suppressed by draft_only mode)"),
    ],
)
def test_delivery_hold_line(ceiling, mode, waiting, expected):
    assert support_agent.delivery_hold_line(
        delivery_ceiling=ceiling, mode=mode, customer_is_waiting=waiting
    ) == expected


def test_customer_waiting_uses_the_newest_non_note_thread():
    customer = {"type": "customer", "createdAt": "2026-09-01T10:00:00Z"}
    reply = {"type": "message", "createdAt": "2026-09-01T11:00:00Z", "createdBy": {"type": "user"}}
    note = {"type": "note", "createdAt": "2026-09-01T12:00:00Z"}
    draft = {"type": "message", "state": "draft", "createdAt": "2026-09-01T13:00:00Z"}
    assert support_agent.customer_waiting([customer, note, draft]) is True
    assert support_agent.customer_waiting([customer, reply, note]) is False
    assert support_agent.customer_waiting([]) is False


def test_one_run_per_conversation():
    support_agent.reset_state()
    with support_agent.claim_conversation(42) as first:
        assert first is True
        with support_agent.claim_conversation(42) as duplicate:
            assert duplicate is False
        with support_agent.claim_conversation(43) as other:
            assert other is True
    with support_agent.claim_conversation(42) as again:
        assert again is True


def test_send_confidence_judge_is_a_separate_role_with_one_model_setting(monkeypatch):
    from agents import draft_confidence

    captured = {}

    def fake_claude(*, prompt, model, system_prompt, timeout):
        captured.update(model=model, system=system_prompt)
        return (
            '{"score":9,"verdict":"accept","summary":"ok","missing_information":[],'
            '"investigation_actions":[],"rewrite_brief":"","risk_flags":[],"hard_blockers":[]}'
        )

    monkeypatch.setattr("agents.draft_confidence.invoke_claude", fake_claude)
    monkeypatch.delenv("SUPA_SUPPORT_JUDGE_MODEL", raising=False)
    draft_confidence.evaluate_draft(customer_name="T", subject="s", message="m", draft_html="<p>d</p>")
    assert captured["model"] == "sonnet"
    assert captured["system"] == draft_confidence.CONFIDENCE_SYSTEM_PROMPT
    assert captured["system"] != support_agent.SUPPORT_AGENT_INSTRUCTIONS

    monkeypatch.setenv("SUPA_SUPPORT_JUDGE_MODEL", "opus")
    draft_confidence.evaluate_draft(customer_name="T", subject="s", message="m", draft_html="<p>d</p>")
    assert captured["model"] == "opus"


def test_pack_voice_follows_its_rules_in_the_drafting_prompt(monkeypatch, tmp_path):
    from services import company_pack, extensions

    (tmp_path / "voice.md").write_text("Name the chapter before describing its index.")
    monkeypatch.setattr(company_pack, "PACK_DIR", tmp_path)
    monkeypatch.setattr(extensions, "_voice_sources", [])
    monkeypatch.setattr(extensions, "load", lambda: None)
    text = support_agent.build_draft_prompt(_REQUEST)
    assert text.index(support_agent.load_rules()) < text.index("Name the chapter before describing its index.")
