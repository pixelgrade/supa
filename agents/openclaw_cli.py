import json
import os
import subprocess


def _extract_text(data):
    if isinstance(data, str):
        return data

    candidates = [
        data.get("result"),
        data.get("message"),
        data.get("output"),
        data.get("text"),
        data.get("content"),
    ]

    reply = data.get("reply")
    if isinstance(reply, str):
        candidates.append(reply)
    elif isinstance(reply, dict):
        candidates.extend([reply.get("text"), reply.get("content"), reply.get("message")])

    nested = data.get("data")
    if isinstance(nested, dict):
        candidates.extend([nested.get("result"), nested.get("text"), nested.get("content")])

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate

    raise RuntimeError("OpenClaw returned JSON without a text result")


def invoke_openclaw(prompt: str, timeout: int = 180) -> str:
    """Invoke the local OpenClaw agent and return its text response."""
    openclaw_bin = os.environ.get("OPENCLAW_BIN", "openclaw")
    cmd = [
        openclaw_bin,
        "agent",
        "--local",
        "--json",
        "--thinking",
        "medium",
        "--timeout",
        str(timeout),
        "--message",
        prompt,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"OpenClaw failed (exit {result.returncode}): {stderr}")

    stdout = (result.stdout or "").strip()
    if not stdout:
        raise RuntimeError("OpenClaw returned empty output")

    data = json.loads(stdout)
    return _extract_text(data)
