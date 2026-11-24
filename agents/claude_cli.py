import base64
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from agents.provider_alerts import classify_provider_failure
from agents.provider_alerts import maybe_alert_provider_login, sanitize_provider_detail


class ClaudeUnavailable(RuntimeError):
    """A provider execution failure; public code deliberately contains no provider output."""
    def __init__(self, message, code="unavailable"):
        super().__init__(message)
        self.code = code if code in {"needs_login", "rate_limited", "needs_upgrade", "cooldown", "transport", "invalid_response", "unavailable"} else "unavailable"


_last_claude_warning: Optional[Dict[str, str]] = None
_claude_unavailable_until = 0.0
CLAUDE_FAILURE_COOLDOWN_SECONDS = 300


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _parse_claude_response(stdout: str):
    """Single JSON for text calls, final result JSON from the image-input stream."""
    try:
        return json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        events = [json.loads(line) for line in (stdout or "").splitlines() if line.strip()]
        return next((event for event in reversed(events)
                     if isinstance(event, dict) and event.get("type") == "result"), None)


def _extract_claude_error(stdout: str, stderr: str) -> str:
    """Extract an actionable Claude CLI failure without hiding structured output."""
    raw_stdout = (stdout or "").strip()
    try:
        response = _parse_claude_response(raw_stdout)
    except (json.JSONDecodeError, TypeError):
        response = None

    if isinstance(response, dict):
        result = response.get("result")
        if isinstance(result, str) and result.strip():
            return result.strip()

        error = response.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()

            nested_error = error.get("error")
            if isinstance(nested_error, dict):
                message = nested_error.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()

    detail = (stderr or "").strip()
    if detail:
        return detail
    return raw_stdout[:500]


def _build_warning(message: str) -> Dict[str, str]:
    """Normalize Claude CLI failures for the dashboard."""
    normalized = (message or "").strip()
    lowered = normalized.lower()
    safe_detail = sanitize_provider_detail(normalized)

    if "credit balance is too low" in lowered:
        return {
            "code": "claude_credit_exhausted",
            "message": "Claude is unavailable. Supa is using Codex fallback where available.",
            "detail": "Credit balance is too low.",
            "occurred_at": _utc_now_iso(),
        }

    if classify_provider_failure(normalized) == "rate_limited":
        return {
            "code": "claude_rate_limited",
            "message": "Claude is rate limited. Supa is using Codex fallback where available.",
            "detail": safe_detail,
            "occurred_at": _utc_now_iso(),
        }

    if "command not found" in lowered or "no such file or directory" in lowered:
        return {
            "code": "claude_binary_unavailable",
            "message": "Claude CLI is unavailable. Supa is using Codex fallback where available.",
            "detail": safe_detail,
            "occurred_at": _utc_now_iso(),
        }

    return {
        "code": "claude_unavailable",
        "message": "Claude is unavailable. Supa is using Codex fallback where available.",
        "detail": safe_detail or "Unknown Claude CLI error.",
        "occurred_at": _utc_now_iso(),
    }


def set_claude_warning(message: str) -> None:
    """Store the last Claude operational issue for dashboard visibility."""
    global _last_claude_warning, _claude_unavailable_until
    _last_claude_warning = _build_warning(message)
    _claude_unavailable_until = time.monotonic() + CLAUDE_FAILURE_COOLDOWN_SECONDS


def clear_claude_warning() -> None:
    """Clear the last Claude warning after a successful invocation."""
    global _last_claude_warning, _claude_unavailable_until
    _last_claude_warning = None
    _claude_unavailable_until = 0.0


def claude_paused_by_operator() -> bool:
    """True while SUPA_CLAUDE_DISABLED_UNTIL is in the future."""
    disabled_until = (os.getenv("SUPA_CLAUDE_DISABLED_UNTIL") or "").strip()
    if disabled_until:
        try:
            deadline = datetime.fromisoformat(disabled_until.replace("Z", "+00:00"))
            if deadline.tzinfo and datetime.now(timezone.utc) < deadline:
                return True
        except ValueError:
            pass
    return False


def should_skip_claude() -> bool:
    """Use Codex during an operator pause or a short outage cooldown."""
    if claude_paused_by_operator():
        return True
    return time.monotonic() < _claude_unavailable_until


def get_claude_warning() -> Optional[Dict[str, str]]:
    """Return the last Claude operational warning, if any."""
    return _last_claude_warning.copy() if _last_claude_warning else None


def build_claude_env(claude_bin: str, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build a subprocess environment for Claude CLI.

    By default Supa uses Claude subscription authentication through OAuth.
    Set `CLAUDE_AUTH_MODE=api_key` to force direct Anthropic API-key auth.
    """
    env = (base_env or os.environ).copy()

    # Ensure node is in PATH (claude CLI needs it via #!/usr/bin/env node).
    node_dir = os.path.dirname(claude_bin)
    current_path = env.get("PATH", "")
    if node_dir and node_dir not in current_path.split(":"):
        env["PATH"] = f"{node_dir}:{current_path}" if current_path else node_dir

    # Claude CLI needs HOME for subscription/OAuth auth files.
    if not env.get("HOME"):
        env["HOME"] = "/root"

    auth_mode = (env.get("CLAUDE_AUTH_MODE") or "subscription").strip().lower()
    temporary = env.pop("SUPA_CLAUDE_TEMP_OAUTH_TOKEN", "")
    temporary_until = env.pop("SUPA_CLAUDE_TEMP_UNTIL", "")
    if auth_mode not in {"api", "api_key"}:
        env.pop("ANTHROPIC_API_KEY", None)
        try:
            deadline = datetime.fromisoformat(temporary_until.replace("Z", "+00:00"))
            if temporary.strip() and deadline.tzinfo and datetime.now(timezone.utc) < deadline:
                env["CLAUDE_CODE_OAUTH_TOKEN"] = temporary.strip()
        except (ValueError, TypeError):
            pass

    return env


def _image_content(path: Path) -> Dict:
    """Bounded image bytes for a Claude user message; never put them in argv or logs."""
    with Path(path).open("rb") as source:
        data = source.read(5 * 1024 * 1024 + 1)
    if len(data) > 5 * 1024 * 1024:
        raise ValueError("Image input exceeds Claude's 5 MB image limit")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        mime = "image/gif"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError("Unsupported image input; expected PNG, JPEG, GIF or WebP")
    return {"type": "image", "source": {
        "type": "base64", "media_type": mime, "data": base64.b64encode(data).decode("ascii"),
    }}


def _run_claude(prompt: str, model: str, system_prompt: Optional[str], timeout: int,
                add_dirs=(), allowed_tools=None, tools=None, disallowed_tools=None, cwd=None, isolated=False,
                ignore_failure_cooldown: bool = False, effort: Optional[str] = None, image_paths=None) -> Dict:
    """Run the Claude CLI once and return its parsed JSON response (result, cost, usage).

    `ignore_failure_cooldown` lets a same-run model fallback (Opus failed, try Sonnet) run
    during the outage cooldown the first failure started; an operator pause still applies."""
    paused = claude_paused_by_operator() if ignore_failure_cooldown else should_skip_claude()
    if paused:
        raise ClaudeUnavailable("Claude is paused; use Codex fallback", "cooldown")

    claude_bin = os.environ.get("CLAUDE_BIN", "claude")

    cmd = [
        claude_bin, "-p",
        "--output-format", "json",
        "--model", model,
        "--permission-mode", "dontAsk",
        "--no-session-persistence",
    ]

    if effort:
        cmd.extend(["--effort", effort])
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    for directory in add_dirs or ():
        cmd.extend(["--add-dir", str(directory)])
    if isolated:  # no MCP servers, no user/project/local settings (hooks, permissions, CLAUDE.md dirs)
        cmd.extend(["--strict-mcp-config", "--setting-sources", ""])
    if tools is not None:
        cmd.extend(["--tools", ",".join(tools)])
    if allowed_tools:
        cmd.extend(["--allowedTools", ",".join(allowed_tools)])
    if disallowed_tools:
        cmd.extend(["--disallowedTools", ",".join(disallowed_tools)])

    env = build_claude_env(claude_bin)
    image_blocks = [_image_content(Path(path)) for path in (image_paths or ())]
    user_input = prompt
    if image_blocks:
        # Claude CLI requires streamed output whenever its input is streamed. The final
        # result event retains the same JSON result/cost/usage contract as text calls.
        cmd[cmd.index("--output-format") + 1] = "stream-json"
        cmd.extend(["--input-format", "stream-json", "--verbose"])
        user_input = json.dumps({"type": "user", "message": {
            "role": "user", "content": [{"type": "text", "text": prompt}, *image_blocks],
        }}) + "\n"

    try:
        result = subprocess.run(
            cmd,
            input=user_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            **({"cwd": str(cwd)} if cwd else {}),
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        set_claude_warning(f"Claude CLI unavailable: {exc}")
        raise

    if result.returncode != 0:
        detail = _extract_claude_error(result.stdout, result.stderr)
        safe_detail = sanitize_provider_detail(detail)
        set_claude_warning(detail)
        maybe_alert_provider_login("claude", detail)
        raise ClaudeUnavailable(
            f"Claude CLI failed (exit {result.returncode}): {safe_detail}", classify_provider_failure(detail) or "unavailable"
        )

    try:
        response = _parse_claude_response(result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        detail = "Claude CLI returned an invalid response"
        set_claude_warning(detail)
        raise ClaudeUnavailable(detail, "invalid_response") from exc
    if not isinstance(response, dict):
        detail = "Claude CLI returned an invalid response"
        set_claude_warning(detail)
        raise ClaudeUnavailable(detail, "invalid_response")

    if response.get("is_error"):
        detail = response.get("result", "")
        safe_detail = sanitize_provider_detail(detail)
        set_claude_warning(detail)
        maybe_alert_provider_login("claude", detail)
        raise ClaudeUnavailable(f"Claude returned error: {safe_detail}", classify_provider_failure(detail) or "unavailable")

    answer = response.get("result", "")
    if not isinstance(answer, str) or not answer.strip():
        detail = "Claude CLI returned an empty response"
        set_claude_warning(detail)
        raise ClaudeUnavailable(detail, "invalid_response")

    clear_claude_warning()
    # The count is transport evidence produced here, never a field supplied by the model.
    response["image_input_count"] = len(image_blocks)
    return response


def invoke_claude(
    prompt: str,
    model: str = "sonnet",
    system_prompt: str = None,
    timeout: int = 600,
    ignore_failure_cooldown: bool = False,
) -> str:
    """Invoke Claude CLI and return the text response.

    Args:
        prompt: The user prompt (passed via stdin).
        model: Claude model name (e.g. 'haiku', 'sonnet').
        system_prompt: Optional system prompt.
        timeout: Timeout in seconds (default 10 min).

    Returns:
        The text response from Claude.

    Raises:
        RuntimeError: If the CLI returns an error or times out.
    """
    return _run_claude(
        prompt, model, system_prompt, timeout, ignore_failure_cooldown=ignore_failure_cooldown
    )["result"]


def invoke_claude_result(
    prompt: str,
    model: str = "sonnet",
    system_prompt: str = None,
    timeout: int = 600,
    add_dirs=(),
    allowed_tools=None,
    tools=None,
    disallowed_tools=None,
    cwd=None,
    isolated=False,
    effort=None,
    image_paths=None,
) -> Dict:
    """Like invoke_claude, but also returns what the CLI reports about cost.

    `add_dirs` are passed as --add-dir, `tools` as --tools (the built-in tools available at all; [] means none),
    `allowed_tools` / `disallowed_tools` as --allowedTools / --disallowedTools permission rules (e.g.
    "Read(//abs/dir/**)": `//` marks an absolute path), `cwd` is the working directory, and `isolated` adds
    --strict-mcp-config (with no --mcp-config: no MCP servers) and --setting-sources "" (no user, project or
    local settings), and `effort` as --effort (low, medium, high, xhigh, max). `image_paths` adds
    bounded PNG/JPEG/GIF/WebP image content over stdin. Returns text, total_cost_usd, usage and
    the code-produced image_input_count. Calls run on the
    subscription, so total_cost_usd is an API-equivalent estimate, never money spent.
    """
    response = _run_claude(prompt, model, system_prompt, timeout, add_dirs, allowed_tools, tools, disallowed_tools, cwd, isolated,
                           effort=effort, image_paths=image_paths)
    cost = response.get("total_cost_usd")
    try:
        cost = float(cost) if cost is not None else 0.0
    except (TypeError, ValueError):
        cost = 0.0
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return {"text": response["result"], "total_cost_usd": cost, "usage": usage,
            "image_input_count": response["image_input_count"]}
