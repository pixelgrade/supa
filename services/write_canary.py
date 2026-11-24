"""Check create, update and discard of an internal draft using a scoped token and configured internal mailbox and customer. Report verification failures through the operator's alert channels."""

from __future__ import annotations

from services import extensions, mailbox_provider

import os
import stat
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from services import installation

from services.company_pack import wording

MAILBOX_ID = installation.CANARY_MAILBOX
CUSTOMER_ID = installation.CANARY_CUSTOMER
DEFAULT_URL = "http://127.0.0.1:8100"
DEFAULT_TOKEN_FILE = "~/.config/supa-write-canary/token"
SUBJECT = "Supa write canary (internal draft, discarded automatically)"

CREATE_BODY = (
    "<p>Supa write canary: an unsent internal draft, discarded within a minute.</p>"
    "<p>&nbsp;</p>"
    "<p>Line one<br>line two</p>"
    "<p>&nbsp;</p>"
    "<p>Nobody receives this.</p>"
)
UPDATE_BODY = (
    "<p>Supa write canary: draft replaced.</p>"
    "<p>&nbsp;</p>"
    "<p>First line<br>second line</p>"
)

_TOKEN_MAX_BYTES = 16 * 1024


@dataclass
class CanaryResult:
    ok: bool
    conversation_id: int | None = None
    discarded: bool = False
    problems: list[str] = field(default_factory=list)
    keys: dict[str, str] = field(default_factory=dict)


def _result(envelope: Any) -> dict[str, Any]:
    result = envelope.get("result") if isinstance(envelope, dict) else None
    if not isinstance(result, dict):
        raise ValueError("the server returned no result object")
    return result


def _warnings(result: Any) -> list[str]:
    """Every warning a write result carries (``warnings`` or a ``*_with_warning`` status)."""
    if not isinstance(result, dict):
        return []
    found: list[str] = []
    raw = result.get("warnings")
    if raw:
        found.extend(str(item) for item in (raw if isinstance(raw, list) else [raw]))
    status = result.get("status")
    if isinstance(status, str) and "warning" in status:
        found.append(status)
    return found


def _error_text(error: BaseException) -> str:
    code = getattr(error, "code", None)
    name = type(error).__name__
    return f"{code or name}: {error}"


def _check_operation(client: Any, step: str, key: str, problems: list[str]) -> None:
    operation = _result(client.get_json(f"/agent/v1/operations/{key}"))
    state = operation.get("state")
    if state != "succeeded":
        error = operation.get("error") if isinstance(operation.get("error"), dict) else {}
        problems.append(f"{step}: operation {key} is {state} ({error.get('code')})")
    elif operation.get("error"):
        problems.append(f"{step}: operation {key} succeeded with an error record")
    found = _warnings(operation.get("response"))
    if found:
        problems.append(f"{step}: operation {key} carries warnings {found}")


def _check_created(conversation: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    threads = [
        thread
        for thread in conversation.get("threads") or []
        if isinstance(thread, dict) and thread.get("type") != "lineitem"
    ]
    if len(threads) != 1:
        problems.append(f"read: expected one thread, found {len(threads)}")
    elif threads[0].get("draft") is not True or threads[0].get("state") != "draft":
        problems.append("read: the canary thread is not an unsent draft")
    if (conversation.get("customer") or {}).get("id") != CUSTOMER_ID:
        problems.append("read: the conversation is not addressed to the internal customer")
    if (conversation.get("mailbox") or {}).get("id") != MAILBOX_ID:
        problems.append("read: the conversation is not in the expected mailbox")
    if "supa-ignore" not in (conversation.get("tags") or []):
        problems.append("read: the conversation lacks the supa-ignore tag")
    draft = conversation.get("draft")
    if not isinstance(draft, dict) or not isinstance(draft.get("version"), str):
        problems.append("read: no live draft version")
    return problems


def _discard(client: Any, conversation_id: int, key: str, problems: list[str]) -> bool:
    """Discard the canary conversation and prove it is gone (404 on readback)."""
    path = f"/agent/v1/conversations/{conversation_id}"
    try:
        version = _result(client.get_json(path)).get("version")
        discarded = _result(
            client.post_json(
                f"{path}/discard",
                json={"expected_version": version, "idempotency_key": key},
            )
        )
        if discarded.get("status") != "discarded":
            problems.append(f"discard: status {discarded.get('status')!r}")
        found = _warnings(discarded)
        if found:
            problems.append(f"discard: warnings {found}")
        _check_operation(client, "discard", key, problems)
    except Exception as error:  # noqa: BLE001 - reported, then the readback decides
        problems.append(f"discard: {_error_text(error)}")
    try:
        client.get_json(path)
    except Exception as error:  # noqa: BLE001
        if getattr(error, "exit_code", None) == 4:
            return True
        problems.append(f"discard readback: {_error_text(error)}")
        return False
    problems.append("discard readback: the conversation still exists")
    return False


def run_canary(client: Any, *, key_prefix: str) -> CanaryResult:
    """Create, update and discard one internal draft conversation; collect every problem."""
    keys = {step: f"{key_prefix}.{step}" for step in ("create", "save", "discard")}
    result = CanaryResult(ok=False, keys=keys)
    problems = result.problems
    if MAILBOX_ID <= 0 or CUSTOMER_ID <= 0:
        problems.append("Configure an internal canary mailbox and customer before running this check.")
        return result
    step = "create"
    try:
        created = _result(
            client.post_json(
                "/agent/v1/conversations",
                json={
                    "subject": SUBJECT,
                    "mailbox_id": MAILBOX_ID,
                    "customer_id": CUSTOMER_ID,
                    "body": CREATE_BODY,
                    "body_format": "html",
                    "draft": True,
                    "idempotency_key": keys["create"],
                },
            )
        )
        conversation_id = created.get("conversation_id")
        if type(conversation_id) is int and conversation_id > 0:
            result.conversation_id = conversation_id
        else:
            raise ValueError("the create result has no conversation id")
        if created.get("action") != "created" or created.get("status") != "created":
            problems.append(
                f"create: action {created.get('action')!r} status {created.get('status')!r}"
            )
        found = _warnings(created)
        if found:
            problems.append(f"create: warnings {found}")
        _check_operation(client, "create", keys["create"], problems)

        step = "read"
        conversation = _result(client.get_json(f"/agent/v1/conversations/{conversation_id}"))
        read_problems = _check_created(conversation)
        problems.extend(read_problems)
        if read_problems:
            raise ValueError("the created conversation is not the expected draft")

        step = "save"
        saved = _result(
            client.put_json(
                f"/agent/v1/conversations/{conversation_id}/draft",
                json={
                    "body": UPDATE_BODY,
                    "body_format": "html",
                    "expected_version": conversation["draft"]["version"],
                    "idempotency_key": keys["save"],
                },
            )
        )
        if saved.get("action") != "updated":
            problems.append(f"save: action {saved.get('action')!r}")
        found = _warnings(saved)
        if found:
            problems.append(f"save: warnings {found}")
        _check_operation(client, "save", keys["save"], problems)
    except Exception as error:  # noqa: BLE001 - every failure becomes an alert
        problems.append(f"{step}: {_error_text(error)}")
    finally:
        if result.conversation_id is not None:
            result.discarded = _discard(client, result.conversation_id, keys["discard"], problems)
    result.ok = not problems and result.discarded
    return result


def read_token_file(path: Path) -> str:
    """Read the canary token from a private regular file (never from args or logs)."""
    from supa_cli.config import normalize_token

    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{path} is not a regular file")
    if info.st_mode & 0o077:
        raise ValueError(f"{path} must be readable by its owner only (chmod 600)")
    if info.st_size > _TOKEN_MAX_BYTES:
        raise ValueError(f"{path} is too large")
    return normalize_token(path.read_text(encoding="utf-8").strip())


def load_client_config(environ: Mapping[str, str]):
    from supa_cli.config import CLIConfig, normalize_url

    token_file = Path(
        environ.get("SUPA_WRITE_CANARY_TOKEN_FILE") or DEFAULT_TOKEN_FILE
    ).expanduser()
    url = environ.get("SUPA_WRITE_CANARY_URL") or DEFAULT_URL
    return CLIConfig(url=normalize_url(url), token=read_token_file(token_file))


def _alert_body(result: CanaryResult) -> str:
    lines = [
        "The daily Supa write canary failed. Supa's write or verification path may be",
        "misreporting drafts, sends or notes (like the 2026-10-02 HTML re-serialization bug).",
        "Agents must not learn to ignore write warnings: fix the cause.",
        "",
        "Problems:",
        *[f"- {problem}" for problem in result.problems],
        "",
    ]
    if result.conversation_id is not None:
        lines.append(
            f"Canary conversation: {result.conversation_id} "
            f"({mailbox_provider.conversation_url(result.conversation_id)})"
        )
        lines.append(
            "Discarded: yes"
            if result.discarded
            else f"Discarded: NO - an internal unsent draft to {installation.CANARY_EMAIL} is left; "
            "delete it in Help Scout or rerun the canary."
        )
    else:
        lines.append(
            "No conversation id came back. If the create reached Help Scout, an unsent draft "
            f"titled {SUBJECT!r} to {installation.CANARY_EMAIL} may exist."
        )
    lines.append(f"Idempotency keys (write-canary identity): {', '.join(result.keys.values())}")
    lines.append(
        wording("canary_rerun")
    )
    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    client_factory: Callable[[Any], Any] | None = None,
    alert: Callable[[str, str], Any] | None = None,
    now: Callable[[], datetime] | None = None,
    stdout: Any = None,
) -> int:
    environ = os.environ if environ is None else environ
    extensions.load(environ)
    stdout = sys.stdout if stdout is None else stdout
    moment = (now or (lambda: datetime.now(timezone.utc)))()
    if alert is None:
        from agents.provider_alerts import send_operator_alert

        def alert(subject: str, body: str) -> Any:
            return send_operator_alert(subject, body, log_label="SUPA WRITE CANARY ALERT")

    key_prefix = f"write-canary.{moment.strftime('%Y%m%dT%H%M%SZ')}.{uuid.uuid4().hex[:8]}"
    try:
        config = load_client_config(environ)
        if client_factory is None:
            from supa_cli.client import SupaClient

            client_factory = SupaClient
        with client_factory(config) as client:
            result = run_canary(client, key_prefix=key_prefix)
    except Exception as error:  # noqa: BLE001 - setup failures alert too
        result = CanaryResult(
            ok=False, problems=[f"setup: {_error_text(error)}"],
            keys={"prefix": key_prefix},
        )

    stamp = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    if result.ok:
        stdout.write(
            f"[{stamp}] write canary ok: conversation {result.conversation_id} created, "
            f"draft replaced, discarded (404 on readback)\n"
        )
        stdout.flush()
        return 0
    stdout.write(
        f"[{stamp}] write canary FAILED: conversation {result.conversation_id} "
        f"discarded={'yes' if result.discarded else 'no'}; " + " | ".join(result.problems) + "\n"
    )
    stdout.flush()
    try:
        delivered = alert("[Supa] write canary failed", _alert_body(result))
        if delivered is False:
            stdout.write(f"[{stamp}] write canary alert reached the log only (no SMTP/webhook)\n")
    except Exception as error:  # noqa: BLE001 - the log line above still records it
        stdout.write(f"[{stamp}] write canary alert delivery failed: {_error_text(error)}\n")
    return 1


__all__ = ["CanaryResult", "main", "read_token_file", "run_canary"]
