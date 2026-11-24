import json
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

import agents.claude_cli as cc
import agents.provider_alerts as pa
from agents.claude_cli import (
    build_claude_env,
    clear_claude_warning,
    get_claude_warning,
    invoke_claude,
    set_claude_warning,
)


def test_build_claude_env_prefers_subscription_auth():
    env = build_claude_env(
        "/root/.nvm/versions/node/v22.22.0/bin/claude",
        {
            "PATH": "/usr/bin:/bin",
            "HOME": "/root",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        },
    )

    assert env["PATH"].startswith("/root/.nvm/versions/node/v22.22.0/bin:")
    assert env["HOME"] == "/root"
    assert "ANTHROPIC_API_KEY" not in env


def test_build_claude_env_keeps_api_key_when_explicitly_requested():
    env = build_claude_env(
        "/root/.nvm/versions/node/v22.22.0/bin/claude",
        {
            "PATH": "/usr/bin:/bin",
            "HOME": "/root",
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "CLAUDE_AUTH_MODE": "api_key",
        },
    )

    assert env["ANTHROPIC_API_KEY"] == "sk-ant-test"


def test_credit_balance_warning_is_normalized():
    clear_claude_warning()
    set_claude_warning("Credit balance is too low")
    warning = get_claude_warning()

    assert warning is not None
    assert warning["code"] == "claude_credit_exhausted"
    assert "codex fallback" in warning["message"].lower()
    assert warning["detail"] == "Credit balance is too low."

    clear_claude_warning()


def test_claude_failure_circuit_expires_and_success_resets(monkeypatch):
    clear_claude_warning()
    clock = [1000.0]
    monkeypatch.setattr(cc.time, "monotonic", lambda: clock[0])
    set_claude_warning("You've hit your weekly limit")
    assert cc.should_skip_claude() is True
    clock[0] += cc.CLAUDE_FAILURE_COOLDOWN_SECONDS + 1
    assert cc.should_skip_claude() is False
    set_claude_warning("Not logged in")
    assert cc.should_skip_claude() is True
    clear_claude_warning()
    assert cc.should_skip_claude() is False


def test_operator_pause_skips_claude_until_deadline(monkeypatch):
    clear_claude_warning()
    now = datetime.now(timezone.utc)

    monkeypatch.setenv(
        "SUPA_CLAUDE_DISABLED_UNTIL", (now + timedelta(hours=1)).isoformat()
    )
    assert cc.should_skip_claude() is True

    monkeypatch.setenv(
        "SUPA_CLAUDE_DISABLED_UNTIL", (now - timedelta(hours=1)).isoformat()
    )
    assert cc.should_skip_claude() is False

    monkeypatch.setenv("SUPA_CLAUDE_DISABLED_UNTIL", "invalid")
    assert cc.should_skip_claude() is False


def test_operator_pause_prevents_direct_claude_invocation(monkeypatch):
    clear_claude_warning()
    monkeypatch.setenv(
        "SUPA_CLAUDE_DISABLED_UNTIL",
        (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    )
    monkeypatch.setattr(
        cc.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Claude was launched")
    )

    with pytest.raises(RuntimeError, match="paused"):
        invoke_claude("Test")


def test_claude_timeout_opens_failure_circuit(monkeypatch):
    clear_claude_warning()
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    with pytest.raises(subprocess.TimeoutExpired):
        invoke_claude("Test", timeout=2)
    assert cc.should_skip_claude() is True
    clear_claude_warning()


def test_empty_claude_result_opens_failure_circuit(monkeypatch):
    clear_claude_warning()
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps({"is_error": False, "result": "  "}), stderr=""
        )
    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="empty response"):
        invoke_claude("Test")
    assert cc.should_skip_claude() is True
    clear_claude_warning()


def test_malformed_claude_result_opens_failure_circuit(monkeypatch):
    clear_claude_warning()
    monkeypatch.setattr(
        cc.subprocess, "run",
        lambda cmd, **_: subprocess.CompletedProcess(cmd, 0, stdout="not JSON", stderr=""),
    )
    with pytest.raises(RuntimeError, match="invalid response"):
        invoke_claude("Test")
    assert cc.should_skip_claude() is True
    clear_claude_warning()


def test_extract_claude_error_prefers_structured_result():
    stdout = json.dumps(
        {"is_error": True, "result": "You've hit your session limit"}
    )

    assert (
        cc._extract_claude_error(stdout, "generic stderr")
        == "You've hit your session limit"
    )


def test_extract_claude_error_reads_error_message():
    stdout = json.dumps({"error": {"message": "OAuth token is invalid"}})

    assert cc._extract_claude_error(stdout, "") == "OAuth token is invalid"


def test_extract_claude_error_reads_nested_error_message():
    stdout = json.dumps(
        {"error": {"error": {"message": "OAuth token is invalid"}}}
    )

    assert cc._extract_claude_error(stdout, "") == "OAuth token is invalid"


def test_extract_claude_error_falls_back_to_stderr():
    stdout = json.dumps({"unexpected": "value"})

    assert cc._extract_claude_error(stdout, "specific stderr") == "specific stderr"


@pytest.mark.parametrize(
    "stdout",
    [
        "not-json-" + ("x" * 600),
        json.dumps({"unexpected": "x" * 600}),
    ],
)
def test_extract_claude_error_bounds_unstructured_stdout(stdout):
    detail = cc._extract_claude_error(stdout, "")

    assert detail == stdout.strip()[:500]
    assert len(detail) == 500


def test_session_limit_warning_is_normalized_with_full_reset_detail():
    clear_claude_warning()
    detail = "You've hit your session limit · resets Sep 9 at 6:30am"

    set_claude_warning(detail)
    warning = get_claude_warning()

    assert warning is not None
    assert warning["code"] == "claude_rate_limited"
    assert "rate limited" in warning["message"].lower()
    assert "codex fallback" in warning["message"].lower()
    assert warning["detail"] == detail

    clear_claude_warning()


def test_invoke_claude_normalizes_nonzero_structured_error(monkeypatch):
    detail = "You've hit your session limit · resets Sep 9 at 6:30am"

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout=json.dumps({"is_error": True, "result": detail}),
            stderr="generic stderr",
        )

    warnings = []
    alerts = []
    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    monkeypatch.setattr(cc, "set_claude_warning", warnings.append)
    monkeypatch.setattr(
        cc,
        "maybe_alert_provider_login",
        lambda provider, message: alerts.append((provider, message)),
    )

    with pytest.raises(RuntimeError, match=detail):
        invoke_claude("Classify this ticket.", timeout=10)

    assert warnings == [detail]
    assert alerts == [("claude", detail)]


def test_invoke_claude_sanitizes_long_structured_error_but_classifies_raw_detail(
    monkeypatch,
):
    token = "sk-ant-oat01-do-not-expose-this-token"
    detail = (
        f"Provider failed with {token}. "
        + ("x" * (pa.PROVIDER_DETAIL_LIMIT + 100))
        + " You've hit your session limit"
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout=json.dumps({"is_error": True, "result": detail}),
            stderr="generic stderr",
        )

    dispatched = []
    pa.reset_state()
    clear_claude_warning()
    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    monkeypatch.setattr(
        pa, "_dispatch_alert", lambda subject, body: dispatched.append((subject, body))
    )

    with pytest.raises(RuntimeError) as exc_info:
        invoke_claude("Classify this ticket.", timeout=10)

    runtime_detail = str(exc_info.value).split(": ", 1)[1]
    warning = get_claude_warning()
    assert warning is not None
    assert warning["code"] == "claude_rate_limited"
    assert token not in warning["detail"]
    assert len(warning["detail"]) <= pa.PROVIDER_DETAIL_LIMIT
    assert warning["detail"].endswith(pa.PROVIDER_DETAIL_TRUNCATION_MARKER)
    assert token not in runtime_detail
    assert len(runtime_detail) <= pa.PROVIDER_DETAIL_LIMIT
    assert runtime_detail.endswith(pa.PROVIDER_DETAIL_TRUNCATION_MARKER)
    assert len(dispatched) == 1
    alert_detail = dispatched[0][1].split("Provider detail:\n", 1)[1]
    assert token not in alert_detail
    assert len(alert_detail) <= pa.PROVIDER_DETAIL_LIMIT
    assert alert_detail.endswith(pa.PROVIDER_DETAIL_TRUNCATION_MARKER)

    clear_claude_warning()


def test_invoke_claude_sanitizes_and_bounds_long_stderr(monkeypatch):
    token = "Bearer eyJhbGciOi.do-not-expose.signature"
    stderr = f"transport failed with {token} " + (
        "x" * (pa.PROVIDER_DETAIL_LIMIT + 100)
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=stderr)

    clear_claude_warning()
    monkeypatch.setattr(cc.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        invoke_claude("Classify this ticket.", timeout=10)

    runtime_detail = str(exc_info.value).split(": ", 1)[1]
    warning = get_claude_warning()
    assert warning is not None
    assert token not in warning["detail"]
    assert len(warning["detail"]) <= pa.PROVIDER_DETAIL_LIMIT
    assert warning["detail"].endswith(pa.PROVIDER_DETAIL_TRUNCATION_MARKER)
    assert token not in runtime_detail
    assert len(runtime_detail) <= pa.PROVIDER_DETAIL_LIMIT
    assert runtime_detail.endswith(pa.PROVIDER_DETAIL_TRUNCATION_MARKER)

    clear_claude_warning()


def test_invoke_claude_result_passes_the_card_dir_and_returns_the_cost_estimate(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({
            "result": '{"verdict": "yes"}', "total_cost_usd": 0.0312, "usage": {"input_tokens": 1200}}), stderr="")

    clear_claude_warning()
    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    result = cc.invoke_claude_result("judge", system_prompt="s", add_dirs=["/tmp/card/7"], allowed_tools=["Read"])
    assert result == {"text": '{"verdict": "yes"}', "total_cost_usd": 0.0312, "usage": {"input_tokens": 1200}, "image_input_count": 0}
    cmd = seen["cmd"]
    assert cmd[cmd.index("--add-dir") + 1] == "/tmp/card/7" and cmd[cmd.index("--allowedTools") + 1] == "Read"
    assert invoke_claude("judge") == '{"verdict": "yes"}'
    assert "--add-dir" not in seen["cmd"]


def test_isolated_calls_load_no_mcp_servers_and_no_settings(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["cwd"] = cmd, kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"result": "ok", "total_cost_usd": 0}), stderr="")

    clear_claude_warning()
    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    cc.invoke_claude_result("x", tools=["Read"], allowed_tools=["Read(//tmp/card/7/**)"],
                            disallowed_tools=["Bash", "Write"], cwd="/tmp/card/7", isolated=True)
    cmd = seen["cmd"]
    assert "--strict-mcp-config" in cmd and "--mcp-config" not in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--tools") + 1] == "Read" and cmd[cmd.index("--allowedTools") + 1] == "Read(//tmp/card/7/**)"
    assert cmd[cmd.index("--disallowedTools") + 1] == "Bash,Write" and seen["cwd"] == "/tmp/card/7"
    cc.invoke_claude_result("x", tools=[], isolated=True)
    assert seen["cmd"][seen["cmd"].index("--tools") + 1] == ""
    cc.invoke_claude_result("x")
    assert "--strict-mcp-config" not in seen["cmd"] and "--setting-sources" not in seen["cmd"]


@pytest.mark.parametrize("deadline",["2099-10-07T03:05:00Z","2000-10-07T03:05:00Z","invalid","2099-10-07T03:05:00",""])
def test_temporary_subscription_token_expires_safely_and_never_leaks_override_env(deadline):
    base={"HOME":"/tmp/test", "CLAUDE_CODE_OAUTH_TOKEN":"synthetic-primary", "SUPA_CLAUDE_TEMP_OAUTH_TOKEN":"synthetic-temporary",
          "SUPA_CLAUDE_TEMP_UNTIL":deadline}
    env=build_claude_env("claude",base)
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ("synthetic-temporary" if deadline=="2099-10-07T03:05:00Z" else "synthetic-primary")
    assert "SUPA_CLAUDE_TEMP_OAUTH_TOKEN" not in env and "SUPA_CLAUDE_TEMP_UNTIL" not in env
    env=build_claude_env("claude",{**base,"CLAUDE_AUTH_MODE":"api_key"})
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "synthetic-primary"
