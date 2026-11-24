import inspect
import json
import os
import shutil
import subprocess
import types
from pathlib import Path

import pytest

import agents.codex_cli as cc
from agents.codex_cli import DEFAULT_WORKING_DIRECTORY, invoke_codex


@pytest.fixture(autouse=True)
def isolate_codex_resolver_cache(monkeypatch):
    """Keep resolver discovery out of invocation tests and state out of later tests."""
    monkeypatch.setenv("CODEX_BIN", "codex")

    def clear_cache():
        resolver = getattr(cc, "resolve_codex_bin", None)
        if resolver is not None and hasattr(resolver, "cache_clear"):
            resolver.cache_clear()

    clear_cache()
    yield
    clear_cache()


def _configure_codex_candidates(monkeypatch, tmp_path, versions):
    home = tmp_path / "home"
    candidates = {
        "standalone": home / ".local" / "bin" / "codex",
        "nvm_old": home / ".nvm" / "versions" / "node" / "v20.0.0" / "bin" / "codex",
        "nvm": home / ".nvm" / "versions" / "node" / "v24.0.0" / "bin" / "codex",
        "path": tmp_path / "path" / "codex",
        "homebrew": Path("/opt/homebrew/bin/codex"),
    }

    for name in ("standalone", "nvm_old", "nvm"):
        candidate = candidates[name]
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.touch()

    version_by_path = {
        str(candidates[name]): version for name, version in versions.items()
    }
    monkeypatch.delenv("CODEX_BIN", raising=False)
    monkeypatch.setattr(cc.Path, "home", lambda: home)
    monkeypatch.setattr(
        shutil,
        "which",
        lambda command: str(candidates["path"]) if "path" in versions else None,
    )
    monkeypatch.setattr(
        cc.os,
        "access",
        lambda path, mode: str(path) in version_by_path,
    )
    return candidates, version_by_path


def test_resolve_codex_bin_returns_non_empty_override_unchanged(monkeypatch):
    monkeypatch.setenv("CODEX_BIN", "  /custom/bin/codex  ")
    monkeypatch.setattr(
        cc.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("an explicit override must not be probed"),
    )

    assert cc.resolve_codex_bin() == "  /custom/bin/codex  "


def test_resolve_codex_bin_selects_newest_version_across_all_sources(
    monkeypatch, tmp_path
):
    candidates, version_by_path = _configure_codex_candidates(
        monkeypatch,
        tmp_path,
        {
            "standalone": "0.151.0",
            "nvm_old": "0.140.0",
            "nvm": "0.153.4",
            "path": "0.152.1",
            "homebrew": "0.145.0",
        },
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            cmd, 0, stdout=f"codex-cli {version_by_path[cmd[0]]}\n", stderr=""
        )

    monkeypatch.setattr(cc.subprocess, "run", fake_run)

    assert cc.resolve_codex_bin() == str(candidates["nvm"])
    assert cc.resolve_codex_bin() == str(candidates["nvm"])
    assert {call[0][0] for call in calls} == set(version_by_path)
    assert len(calls) == len(version_by_path)
    assert all(call[0][1:] == ["--version"] for call in calls)
    assert all(call[1]["timeout"] == 5 for call in calls)


@pytest.mark.parametrize(
    ("versions", "winner"),
    [
        (
            {
                "standalone": "0.153.4",
                "nvm": "0.153.4",
                "path": "0.153.4",
                "homebrew": "0.153.4",
            },
            "standalone",
        ),
        (
            {"nvm": "0.153.4", "path": "0.153.4", "homebrew": "0.153.4"},
            "nvm",
        ),
        ({"path": "0.153.4", "homebrew": "0.153.4"}, "path"),
        ({"homebrew": "0.153.4"}, "homebrew"),
    ],
)
def test_resolve_codex_bin_uses_deterministic_source_priority_for_version_ties(
    monkeypatch, tmp_path, versions, winner
):
    candidates, version_by_path = _configure_codex_candidates(
        monkeypatch, tmp_path, versions
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout=f"codex-cli {version_by_path[cmd[0]]}", stderr=""
        )

    monkeypatch.setattr(cc.subprocess, "run", fake_run)

    assert cc.resolve_codex_bin() == str(candidates[winner])


def test_resolve_codex_bin_falls_back_to_codex_when_no_candidate_is_usable(
    monkeypatch, tmp_path
):
    candidates, _ = _configure_codex_candidates(
        monkeypatch,
        tmp_path,
        {
            "standalone": "invalid",
            "nvm": "nonzero",
            "path": "timeout",
            "homebrew": "oserror",
        },
    )

    def fake_run(cmd, **kwargs):
        if cmd[0] == str(candidates["standalone"]):
            return subprocess.CompletedProcess(cmd, 0, stdout="Codex unknown", stderr="")
        if cmd[0] == str(candidates["nvm"]):
            return subprocess.CompletedProcess(
                cmd, 1, stdout="codex-cli 99.0.0", stderr="broken"
            )
        if cmd[0] == str(candidates["path"]):
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        raise OSError("cannot execute")

    monkeypatch.setattr(cc.subprocess, "run", fake_run)

    assert cc.resolve_codex_bin() == "codex"


def test_resolve_codex_bin_deduplicates_real_paths_and_retains_priority_path(
    monkeypatch, tmp_path
):
    home = tmp_path / "home"
    nvm = home / ".nvm" / "versions" / "node" / "v24.0.0" / "bin" / "codex"
    nvm.parent.mkdir(parents=True)
    nvm.touch()
    standalone = home / ".local" / "bin" / "codex"
    standalone.parent.mkdir(parents=True)
    standalone.symlink_to(nvm)

    monkeypatch.delenv("CODEX_BIN", raising=False)
    monkeypatch.setattr(cc.Path, "home", lambda: home)
    monkeypatch.setattr(shutil, "which", lambda command: str(nvm))
    monkeypatch.setattr(
        cc.os,
        "access",
        lambda path, mode: str(path) in {str(standalone), str(nvm)},
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="codex-cli 0.153.4", stderr="")

    monkeypatch.setattr(cc.subprocess, "run", fake_run)

    assert cc.resolve_codex_bin() == str(standalone)
    assert calls == [[str(standalone), "--version"]]


def test_invoke_codex_constructs_non_interactive_command(monkeypatch):
    captured = {}

    def fake_run(cmd, input, capture_output, text, timeout, env):
        captured["cmd"] = cmd
        captured["input"] = input
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["timeout"] = timeout
        captured["env"] = env

        schema_path = Path(cmd[cmd.index("--output-schema") + 1])
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        captured["schema"] = json.loads(schema_path.read_text(encoding="utf-8"))

        output_path.write_text('{"reply_html":"<p>ok</p>"}', encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ignored", stderr="")

    monkeypatch.setenv("CODEX_BIN", "/usr/local/bin/codex")
    monkeypatch.setattr("agents.codex_cli.subprocess.run", fake_run)

    result = invoke_codex(
        prompt="Investigate this support ticket.",
        output_schema={
            "type": "object",
            "required": ["reply_html"],
            "additionalProperties": False,
            "properties": {"reply_html": {"type": "string"}},
        },
        timeout=45,
    )

    assert result == '{"reply_html":"<p>ok</p>"}'
    assert captured["cmd"][0:2] == ["/usr/local/bin/codex", "exec"]
    assert "--output-schema" in captured["cmd"]
    assert "--output-last-message" in captured["cmd"]
    assert "-C" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("-C") + 1] == DEFAULT_WORKING_DIRECTORY
    assert "--sandbox" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--sandbox") + 1] == "read-only"
    assert "--skip-git-repo-check" in captured["cmd"]
    assert "--color" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--color") + 1] == "never"
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["timeout"] == 45
    assert captured["input"] == "Investigate this support ticket."
    assert captured["schema"]["required"] == ["reply_html"]


def test_invoke_codex_uses_resolved_binary(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(_cmd_value(cmd, "--output-last-message")).write_text('{"score":9}')
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cc, "resolve_codex_bin", lambda: "/resolved/bin/codex")
    monkeypatch.setattr(cc.subprocess, "run", fake_run)

    invoke_codex(prompt="hi", output_schema=SCHEMA, timeout=10)

    assert captured["cmd"][0] == "/resolved/bin/codex"


def test_invoke_codex_allows_scoped_working_dir():
    # Scoping the evaluator's Codex call to an empty dir avoids loading the
    # project's AGENTS.md context (~37k tokens) on every confidence check.
    params = inspect.signature(invoke_codex).parameters
    assert "working_dir" in params
    assert params["working_dir"].default is None  # defaults to the repo


def test_scoped_eval_working_dir_is_empty_and_outside_repo():
    from agents.codex_cli import scoped_eval_working_dir

    path = scoped_eval_working_dir()
    assert os.path.isdir(path)
    assert DEFAULT_WORKING_DIRECTORY not in path
    # No AGENTS.md/CLAUDE.md to slurp in this scratch dir.
    assert not os.path.exists(os.path.join(path, "AGENTS.md"))


def test_invoke_codex_raises_on_non_zero_exit(monkeypatch):
    def fake_run(cmd, input, capture_output, text, timeout, env):
        return subprocess.CompletedProcess(cmd, 23, stdout="", stderr="boom")

    monkeypatch.setattr("agents.codex_cli.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="exit 23"):
        invoke_codex(
            prompt="Investigate this support ticket.",
            output_schema={"type": "object", "properties": {}},
        )


def test_invoke_codex_raises_when_output_last_message_file_missing(monkeypatch):
    def fake_run(cmd, input, capture_output, text, timeout, env):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("agents.codex_cli.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="output-last-message"):
        invoke_codex(
            prompt="Investigate this support ticket.",
            output_schema={"type": "object", "properties": {}},
        )


SCHEMA = {"type": "object", "required": ["score"], "properties": {"score": {"type": "integer"}}}


def _cmd_value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_invoke_codex_retries_once_on_transient_failure(monkeypatch):
    state = {"n": 0}

    def fake_run(cmd, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="stream closed")
        Path(_cmd_value(cmd, "--output-last-message")).write_text('{"score":9}')
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("agents.codex_cli.subprocess.run", fake_run)
    out = invoke_codex(prompt="hi", output_schema=SCHEMA, timeout=10)
    assert out == '{"score":9}'
    assert state["n"] == 2  # failed once, retried, succeeded


def test_invoke_codex_exhausts_retries_then_raises(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=1, stdout="", stderr="persistent boom")

    monkeypatch.setattr("agents.codex_cli.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="persistent boom"):
        invoke_codex(prompt="hi", output_schema=SCHEMA, timeout=10)
    assert len(calls) == 2  # original + one retry


def test_invoke_codex_login_failure_skips_retry_and_alerts(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(
            returncode=1, stdout="", stderr="Not logged in · Please run /login"
        )

    alerts = []
    monkeypatch.setattr("agents.codex_cli.subprocess.run", fake_run)
    monkeypatch.setattr(cc, "maybe_alert_provider_login", lambda p, m: alerts.append((p, m)))
    with pytest.raises(RuntimeError):
        invoke_codex(prompt="hi", output_schema=SCHEMA, timeout=10)
    assert len(calls) == 1  # login failure must NOT be retried
    assert alerts and alerts[0][0] == "codex"


def test_invoke_codex_upgrade_failure_skips_retry_and_alerts(monkeypatch):
    calls = []
    detail = "This model requires a newer version of Codex"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=1, stdout="", stderr=detail)

    alerts = []
    monkeypatch.setattr("agents.codex_cli.subprocess.run", fake_run)
    monkeypatch.setattr(
        cc, "maybe_alert_provider_login", lambda p, m: alerts.append((p, m))
    )

    assert cc.classify_provider_failure(detail) == "needs_upgrade"
    with pytest.raises(RuntimeError, match="requires a newer version of Codex"):
        invoke_codex(prompt="hi", output_schema=SCHEMA, timeout=10)

    assert len(calls) == 1
    assert alerts == [("codex", detail)]


def test_invoke_codex_working_dir_override(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(_cmd_value(cmd, "--output-last-message")).write_text('{"score":1}')
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("agents.codex_cli.subprocess.run", fake_run)
    invoke_codex(prompt="hi", output_schema=SCHEMA, timeout=10, working_dir="/tmp/scoped")
    assert _cmd_value(captured["cmd"], "-C") == "/tmp/scoped"


def test_text_only_call_disables_configured_mcp_and_builtin_action_tools(monkeypatch, tmp_path):
    config = tmp_path / ".codex"
    config.mkdir()
    (config / "config.toml").write_text('[mcp_servers.mail]\ncommand="mail-server"\n'
                                         '[mcp_servers.browser-tool]\ncommand="browser"\n')
    monkeypatch.setattr(cc.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    captured = {}
    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(_cmd_value(cmd, "--output-last-message")).write_text('{"score":9}')
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    invoke_codex(prompt="Only score the supplied text", output_schema=SCHEMA, text_only=True)
    overrides = [captured["cmd"][i + 1] for i, value in enumerate(captured["cmd"]) if value == "-c"]
    assert 'mcp_servers.mail.enabled=false' in overrides
    assert 'mcp_servers.browser-tool.enabled=false' in overrides
    for feature in ["apps", "plugins", "shell_tool", "browser_use", "computer_use", "multi_agent"]:
        assert f"features.{feature}=false" in overrides
    assert 'web_search="disabled"' in overrides


def test_text_only_call_refuses_unreadable_config_instead_of_exposing_tools(monkeypatch, tmp_path):
    config = tmp_path / ".codex"
    config.mkdir()
    (config / "config.toml").write_text('not = valid = TOML')
    monkeypatch.setattr(cc.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(cc.subprocess, "run", lambda *a, **kw: pytest.fail("must refuse before launching Codex"))
    with pytest.raises(RuntimeError, match="Cannot isolate Codex text call"):
        invoke_codex(prompt="hi", output_schema=SCHEMA, text_only=True)


def test_text_only_call_refuses_server_name_that_cli_cannot_override_safely(monkeypatch, tmp_path):
    config = tmp_path / ".codex"
    config.mkdir()
    (config / "config.toml").write_text('[mcp_servers."browser.with.dot"]\ncommand="browser"\n')
    monkeypatch.setattr(cc.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(cc.subprocess, "run", lambda *a, **kw: pytest.fail("must refuse before launching Codex"))
    with pytest.raises(RuntimeError, match="Cannot isolate Codex text call"):
        invoke_codex(prompt="hi", output_schema=SCHEMA, text_only=True)
