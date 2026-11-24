import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

from services import installation

from agents.provider_alerts import classify_provider_failure, maybe_alert_provider_login


DEFAULT_WORKING_DIRECTORY = installation.CODEX_DIRECTORY
_CODEX_VERSION_RE = re.compile(r"\bcodex-cli\s+(\d+)\.(\d+)\.(\d+)\b")


def _codex_candidates():
    home = Path.home()
    candidates = [home / ".local" / "bin" / "codex"]
    candidates.extend(sorted(home.glob(".nvm/versions/node/*/bin/codex"), key=str))

    path_candidate = shutil.which("codex")
    if path_candidate:
        candidates.append(Path(path_candidate))
    candidates.append(Path("/opt/homebrew/bin/codex"))

    seen = set()
    for candidate in candidates:
        executable = str(candidate)
        if not os.access(executable, os.X_OK):
            continue
        try:
            real_path = str(candidate.resolve())
        except OSError:
            continue
        if real_path in seen:
            continue
        seen.add(real_path)
        yield executable


def _codex_version(candidate: str):
    try:
        result = subprocess.run(
            [candidate, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None
    match = _CODEX_VERSION_RE.search(f"{result.stdout}\n{result.stderr}")
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


@lru_cache(maxsize=1)
def _resolve_codex_bin_cached() -> str:
    best_path = "codex"
    best_version = None
    for candidate in _codex_candidates():
        version = _codex_version(candidate)
        if version is not None and (best_version is None or version > best_version):
            best_path = candidate
            best_version = version
    return best_path


def resolve_codex_bin() -> str:
    """Return an explicit override or the newest usable installed Codex CLI."""
    override = os.environ.get("CODEX_BIN")
    if override:
        return override
    return _resolve_codex_bin_cached()


resolve_codex_bin.cache_clear = _resolve_codex_bin_cached.cache_clear


def scoped_eval_working_dir() -> str:
    """A stable empty scratch dir for Codex calls that do NOT need repo access
    (e.g. the confidence evaluator). Running Codex here keeps it from auto-loading
    the project's AGENTS.md/CLAUDE.md context (~37k tokens) on every call."""
    path = os.path.join(tempfile.gettempdir(), "supa-codex-eval")
    os.makedirs(path, exist_ok=True)
    return path


def _text_only_overrides():
    """Disable action tools for a bounded text classification/writing/judging call.

    A filesystem sandbox alone does not restrict MCP/browser/email tools. Explicit
    per-server overrides are needed: an empty table would merge with user config.
    """
    config_file = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml"
    try:
        config = tomllib.loads(config_file.read_text()) if config_file.exists() else {}
        names = set(config.get("mcp_servers", {}))
        for profile in config.get("profiles", {}).values():
            names.update(profile.get("mcp_servers", {}))
    except (OSError, ValueError, TypeError, AttributeError):
        raise RuntimeError("Cannot isolate Codex text call") from None
    # Codex -c splits keys on dots without TOML quoted-key parsing. Quoting a
    # name creates a different server without transport and prevents bootstrap.
    # Refuse ambiguous names instead of logging/serializing complete transport
    # config (which may contain credentials) into command-line arguments.
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        raise RuntimeError("Cannot isolate Codex text call")
    overrides = [f'mcp_servers.{name}.enabled=false' for name in sorted(names)]
    overrides += [f"features.{feature}=false" for feature in (
        "apps", "plugins", "shell_tool", "browser_use", "computer_use", "multi_agent",
    )]
    overrides.append('web_search="disabled"')
    return [arg for value in overrides for arg in ("-c", value)]


def invoke_codex(
    *,
    prompt: str,
    output_schema: Mapping[str, Any],
    timeout: int = 600,
    working_dir: Optional[str] = None,
    retries: int = 1,
    text_only: bool = False,
) -> str:
    """Invoke `codex exec` with a strict output schema and deterministic output capture.

    Retries once on a transient non-zero exit (the kind of crash seen on Codex
    0.136.0), but does NOT retry login/quota failures — those need a human, so we
    fail fast and raise an operator alert instead.

    Args:
        working_dir: Directory Codex runs in (``-C``). Defaults to the repo so the
            investigator can inspect product repos; pass ``scoped_eval_working_dir()``
            for evaluator-style calls that don't need repo/project context.
        retries: Extra attempts after the first on transient failures (default 1).
        text_only: Disable configured MCP and built-in action tools. Use only with
            a scratch working directory, not for support repository investigations.
    """
    codex_bin = resolve_codex_bin()
    workdir = working_dir or DEFAULT_WORKING_DIRECTORY
    tool_overrides = _text_only_overrides() if text_only else []

    last_error = "unknown codex error"
    last_returncode = 1
    attempts = 0

    for _ in range(retries + 1):
        attempts += 1
        with tempfile.TemporaryDirectory(prefix="supa-codex-") as temp_dir:
            schema_path = Path(temp_dir) / "output-schema.json"
            output_path = Path(temp_dir) / "last-message.txt"

            schema_path.write_text(json.dumps(dict(output_schema)), encoding="utf-8")

            cmd = [
                codex_bin,
                "exec",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "-C",
                workdir,
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--color",
                "never",
            ]
            cmd.extend(tool_overrides)

            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy(),
            )

            if result.returncode == 0:
                if output_path.exists():
                    return output_path.read_text(encoding="utf-8").strip()
                last_error = "Codex CLI did not produce an output-last-message file"
                last_returncode = 0
            else:
                last_error = (result.stderr or result.stdout or "").strip() or "no error output"
                last_returncode = result.returncode

        # A login/quota failure won't be fixed by retrying — stop and alert.
        if classify_provider_failure(last_error) is not None:
            break

    maybe_alert_provider_login("codex", last_error)
    raise RuntimeError(
        f"Codex CLI failed (exit {last_returncode}) after {attempts} attempt(s): {last_error}"
    )
