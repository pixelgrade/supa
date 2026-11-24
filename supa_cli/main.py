"""Command-line entry point for the standalone Supa Agent API client."""

from __future__ import annotations

import argparse
import contextlib
import getpass
import os
import re
import stat
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO
from uuid import UUID, uuid4

import httpx

from . import __version__, extensions
from .client import API_VERSION, ClientError, SupaClient
from .config import ConfigError, TOKEN_MAX_BYTES, load_config, write_config
from .jsonutil import StrictJSONError, dumps as strict_json_dumps


MAX_RESOURCE_ID = (1 << 63) - 1
MAX_WRITE_BODY_BYTES = 1024 * 1024
MAX_IDEMPOTENCY_KEY_LENGTH = 200
_VERSION_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_LIKE_RE = re.compile(r"[A-Za-z0-9_-]{43,}")
_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9._~-]+$")
_EVENT_CONSUMER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_EVENT_ACK_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43,512}$")
EVENT_WATCH_WAIT_SECONDS = 55
EVENT_WATCH_INITIAL_BACKOFF_SECONDS = 1.0
EVENT_WATCH_MAX_BACKOFF_SECONDS = 30.0
_watch_sleep = time.sleep


class UsageError(Exception):
    pass


class ParserExit(Exception):
    def __init__(self, status: int) -> None:
        super().__init__()
        self.status = status


class CLIArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        if re.match(r"argument (?:command|\w+_command): invalid choice:", message):
            message += (
                "; if this command is new, your supa client may be older than the server; "
                "run `supa capabilities`"
            )
        raise UsageError(f"Invalid command usage: {message}")

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if message:
            self._print_message(message, sys.stderr)
        raise ParserExit(status)


class SingleValueAction(argparse.Action):
    """Reject duplicated scalar filters rather than silently choosing one."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        seen_attribute = f"_supa_seen_{self.dest}"
        if getattr(namespace, seen_attribute, False):
            raise argparse.ArgumentError(self, f"{option_string} may be provided only once")
        setattr(namespace, seen_attribute, True)
        setattr(namespace, self.dest, values)


def _positive_resource_id(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= parsed <= MAX_RESOURCE_ID:
        raise argparse.ArgumentTypeError("must be a positive 64-bit integer")
    return parsed


def _run_id(value: str) -> str:
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a UUID4 run ID") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise argparse.ArgumentTypeError("must be a UUID4 run ID")
    return value


def _limit(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return parsed


def _event_wait(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 0 <= parsed <= 55:
        raise argparse.ArgumentTypeError("must be between 0 and 55")
    return parsed


def _event_start(value: str) -> str | int:
    if value in {"now", "oldest"}:
        return value
    return _positive_resource_id(value)


def _event_consumer(value: str) -> str:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise argparse.ArgumentTypeError("must contain valid Unicode") from exc
    if _EVENT_CONSUMER_RE.fullmatch(value) is None or _TOKEN_LIKE_RE.search(value):
        raise argparse.ArgumentTypeError("must be a safe bounded identifier")
    return value


def _event_ack_token(value: str) -> str:
    if _EVENT_ACK_TOKEN_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a valid acknowledgement token")
    return value


def _version(value: str) -> str:
    if _VERSION_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 value")
    return value


def _idempotency_key(value: str) -> str:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise argparse.ArgumentTypeError("must contain valid Unicode") from exc
    if (
        not value
        or value != value.strip()
        or len(value) > MAX_IDEMPOTENCY_KEY_LENGTH
        or _TOKEN_LIKE_RE.search(value)
        or _IDEMPOTENCY_KEY_RE.fullmatch(value) is None
    ):
        raise argparse.ArgumentTypeError("must be a safe bounded identifier")
    return value


def _after_customer(value: str) -> tuple[int, int]:
    parts = value.split("-")
    if len(parts) != 2 or not all(part.isascii() and part.isdecimal() for part in parts):
        raise argparse.ArgumentTypeError("must be MIN-MAX minutes, e.g. 30-60")
    low, high = int(parts[0]), int(parts[1])
    if not 0 <= low <= high <= 1440:
        raise argparse.ArgumentTypeError("must be MIN-MAX minutes with MIN <= MAX <= 1440")
    return low, high


def _send_at(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an ISO 8601 time") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("must include a timezone, e.g. 2026-10-01T09:30:00Z")
    return value


_CC_ADDRESS_RE = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)
MAX_CC_ADDRESSES = 2


def _cc_address(value: str) -> str:
    address = value.strip()
    if _CC_ADDRESS_RE.fullmatch(address) is None:
        raise argparse.ArgumentTypeError("must be one plain email address")
    return address


def _add_cc_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cc",
        type=_cc_address,
        action="append",
        metavar="EMAIL",
        help=(
            "copy this address (repeatable, at most 2); only an address the customer "
            "gave in this conversation (or their own), e.g. 'write me at my personal mail'"
        ),
    )


def _cc_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {"cc": list(args.cc)} if getattr(args, "cc", None) else {}


def _add_idempotency_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--idempotency-key",
        type=_idempotency_key,
        action=SingleValueAction,
    )


def _add_body_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--body-file", required=True, action=SingleValueAction)
    parser.add_argument(
        "--body-format",
        required=True,
        choices=("html", "text"),
        action=SingleValueAction,
    )


def _extract_format(argv: Sequence[str]) -> tuple[list[str], str]:
    cleaned: list[str] = []
    selected: str | None = None
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--format":
            if selected is not None or index + 1 >= len(argv):
                raise UsageError("Invalid command usage.")
            selected = argv[index + 1]
            index += 2
            continue
        if argument.startswith("--format="):
            if selected is not None:
                raise UsageError("Invalid command usage.")
            selected = argument.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    if selected is None:
        return cleaned, "json"
    if selected not in {"json", "text"}:
        raise UsageError("Invalid command usage.")
    return cleaned, selected


def _parser(environ=None) -> CLIArgumentParser:
    from .installation import client_defaults, waiting_on_choices

    parser = CLIArgumentParser(prog="supa", description="Supa Agent API client")
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="output format (default: json)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("health", help="check the authenticated service")
    commands.add_parser("capabilities", help="show scopes and server capabilities")

    desk = commands.add_parser("desk", help="show held reviews and today’s handled activity")
    desk.add_argument("--page", type=_positive_resource_id, default=1)

    mailbox = commands.add_parser("mailbox", help="read mailbox operations")
    mailbox_commands = mailbox.add_subparsers(dest="mailbox_command", required=True)
    insights = mailbox_commands.add_parser("insights", help="read the Insights report")
    insights.add_argument("--months", choices=(3, 6, 12, 24), type=int, default=12)
    mailbox_commands.add_parser("operations", help="read native delivery issue counts")
    modules = extensions.client_modules(environ)
    for module in modules:
        register = getattr(module, "register_client_reads", None)
        if register:
            register(commands)

    ticket = commands.add_parser("ticket", help="start named ticket tasks")
    ticket_commands = ticket.add_subparsers(dest="ticket_command", required=True)
    ticket_evaluate = ticket_commands.add_parser("evaluate", help="start read-only evaluation")
    ticket_evaluate.add_argument("conversation_id", type=_positive_resource_id)
    _add_idempotency_option(ticket_evaluate)
    ticket_process = ticket_commands.add_parser("process", help="start policy-guarded processing")
    ticket_process.add_argument("conversation_id", type=_positive_resource_id)
    ticket_process.add_argument("--allow-send", action="store_true")
    _add_idempotency_option(ticket_process)
    ticket_recover = ticket_commands.add_parser("recover", help="recover an unchanged machine draft without sending")
    ticket_recover.add_argument("conversation_id", type=_positive_resource_id)
    ticket_recover.add_argument("--expected-thread-id", required=True)
    ticket_recover.add_argument("--expected-draft-version", required=True)
    ticket_recover.add_argument("--expected-conversation-version", required=True)
    _add_idempotency_option(ticket_recover)

    for module in modules:
        register = getattr(module, "register_client_tasks", None)
        if register:
            register(commands)

    run = commands.add_parser("run", help="inspect durable task runs")
    run_commands = run.add_subparsers(dest="run_command", required=True)
    run_get = run_commands.add_parser("get", help="get run status and result")
    run_get.add_argument("run_id", type=_run_id)
    run_watch = run_commands.add_parser("watch", help="watch progress to a terminal result")
    run_watch.add_argument("run_id", type=_run_id)

    configure = commands.add_parser("configure", help="write private client configuration")
    configure.add_argument("--url", required=True, action=SingleValueAction)
    configure.add_argument("--token-stdin", action="store_true", required=True)

    conversation = commands.add_parser("conversation", help="read, create or discard unsent conversations")
    conversation_commands = conversation.add_subparsers(
        dest="conversation_command", required=True
    )
    conversation_get = conversation_commands.add_parser("get", help="get one conversation")
    conversation_get.add_argument("conversation_id", type=_positive_resource_id)
    brief = conversation_commands.add_parser("brief", help="read a bounded plain-text conversation")
    brief.add_argument("conversation_id", type=_positive_resource_id)
    brief.add_argument("--max-chars", type=int, default=24000)
    brief.add_argument("--precedents", type=int, choices=range(11), default=0)
    brief.add_argument("--quotes", action="store_true", help="keep quoted mail")
    precedents = conversation_commands.add_parser("precedents", help="find similar questions with sent answers")
    source = precedents.add_mutually_exclusive_group(required=True)
    source.add_argument("conversation_id", nargs="?", type=_positive_resource_id)
    source.add_argument("--question", help="search by question instead of conversation ID")
    precedents.add_argument("--tag", help="favor earlier answers sharing this tag")
    conversation_timeline = conversation_commands.add_parser(
        "timeline",
        help="one line per thread, labelled SENT, DRAFT (NOT SENT), NOTE or CUSTOMER",
    )
    conversation_timeline.add_argument("conversation_id", type=_positive_resource_id)
    conversation_state = conversation_commands.add_parser(
        "state",
        help="show a conversation's state and agent records, or change the state (replaces tag requests)",
    )
    conversation_state.add_argument("conversation_id", type=_positive_resource_id)
    stage_aliases = {
        stage.replace("_", "-"): stage
        for module in modules for stage in getattr(module, "STAGES", ())
    }
    conversation_state.set_defaults(stage_aliases=stage_aliases)
    conversation_state.add_argument(
        "--stage", action=SingleValueAction, choices=tuple(stage_aliases) or None,
    )
    conversation_state.add_argument("--waiting-on", action=SingleValueAction, choices=waiting_on_choices(environ, standalone=True))
    conversation_state.add_argument("--reminders", action=SingleValueAction, choices=("on", "off"))
    conversation_state.add_argument("--take-over", action="store_true", help="support conversations only: stop Supa on this conversation; a person handles it")
    conversation_state.add_argument("--has-site", action=SingleValueAction, choices=("yes", "no"))
    conversation_state.add_argument("--paid", action=SingleValueAction, choices=("yes", "no"))
    conversation_state.add_argument("--reason", action=SingleValueAction, default="", help="one line the owner and other agents will read")
    conversation_list = conversation_commands.add_parser("list", help="list conversations")
    _add_list_filters(conversation_list)
    conversation_search = conversation_commands.add_parser(
        "search", help="search conversations"
    )
    conversation_search.add_argument("query")
    _add_list_filters(conversation_search)

    conversation_create = conversation_commands.add_parser(
        "create", help="create a new conversation (draft unless --send is explicit)"
    )
    conversation_create.add_argument("--subject", required=True, action=SingleValueAction)
    conversation_create.add_argument(
        "--mailbox", required=True, type=_positive_resource_id, action=SingleValueAction
    )
    conversation_create.add_argument(
        "--customer", required=True, type=_positive_resource_id, action=SingleValueAction
    )
    conversation_create.add_argument("--body-file", required=True, action=SingleValueAction)
    conversation_create.add_argument(
        "--body-format", default="text", choices=("html", "text"), action=SingleValueAction
    )
    conversation_create.add_argument("--send", action="store_true")
    _add_idempotency_option(conversation_create)
    conversation_discard = conversation_commands.add_parser(
        "discard", help="discard an unsent draft conversation (no restore in Supa)"
    )
    conversation_discard.add_argument("conversation_id", type=_positive_resource_id)
    conversation_discard.add_argument(
        "--expected-version", required=True, type=_version, action=SingleValueAction
    )
    _add_idempotency_option(conversation_discard)

    draft = commands.add_parser("draft", help="read or write conversation drafts")
    draft_commands = draft.add_subparsers(dest="draft_command", required=True)
    draft_get = draft_commands.add_parser("get", help="get the newest live draft")
    draft_get.add_argument("conversation_id", type=_positive_resource_id)
    draft_save = draft_commands.add_parser("save", help="create or update a draft")
    draft_save.add_argument("conversation_id", type=_positive_resource_id)
    _add_body_options(draft_save)
    draft_save.add_argument(
        "--expected-version",
        type=_version,
        action=SingleValueAction,
    )
    _add_idempotency_option(draft_save)
    draft_send = draft_commands.add_parser("send", help="publish a verified draft")
    draft_send.add_argument("conversation_id", type=_positive_resource_id)
    draft_send.add_argument(
        "--expected-version",
        required=True,
        type=_version,
        action=SingleValueAction,
    )
    draft_send.add_argument(
        "--expected-thread",
        required=True,
        type=_positive_resource_id,
        action=SingleValueAction,
        help=(
            "the latest actionable thread id (the customer's latest message, "
            "latest_actionable_thread_id in draft get/conversation get), not the draft's thread id"
        ),
    )
    _add_cc_option(draft_send)
    _add_idempotency_option(draft_send)

    draft_schedule = draft_commands.add_parser(
        "schedule",
        help="book a reply to go out later (default 30-60 min after the customer's message)",
    )
    draft_schedule.add_argument("conversation_id", type=_positive_resource_id)
    draft_schedule.add_argument("--body-file", action=SingleValueAction)
    draft_schedule.add_argument(
        "--body-format", choices=("html", "text"), action=SingleValueAction
    )
    draft_schedule.add_argument(
        "--expected-version", type=_version, action=SingleValueAction
    )
    draft_schedule.add_argument(
        "--expected-thread",
        required=True,
        type=_positive_resource_id,
        action=SingleValueAction,
        help=(
            "the latest actionable thread id (the customer's latest message, "
            "latest_actionable_thread_id in draft get/conversation get), not the draft's thread id"
        ),
    )
    schedule_timing = draft_schedule.add_mutually_exclusive_group()
    schedule_timing.add_argument("--send-at", type=_send_at, action=SingleValueAction)
    schedule_timing.add_argument(
        "--after-customer", type=_after_customer, action=SingleValueAction
    )
    _add_cc_option(draft_schedule)
    _add_idempotency_option(draft_schedule)
    draft_schedule_get = draft_commands.add_parser(
        "schedule-get", help="show the conversation's booked reply"
    )
    draft_schedule_get.add_argument("conversation_id", type=_positive_resource_id)
    draft_schedule_cancel = draft_commands.add_parser(
        "schedule-cancel", help="cancel the conversation's queued reply"
    )
    draft_schedule_cancel.add_argument("conversation_id", type=_positive_resource_id)

    reply = commands.add_parser("reply", help="send conversation replies")
    reply_commands = reply.add_subparsers(dest="reply_command", required=True)
    reply_send = reply_commands.add_parser("send", help="send a new reply")
    reply_send.add_argument("conversation_id", type=_positive_resource_id)
    _add_body_options(reply_send)
    reply_send.add_argument(
        "--expected-thread",
        required=True,
        type=_positive_resource_id,
        action=SingleValueAction,
        help=(
            "the latest actionable thread id (the customer's latest message, "
            "latest_actionable_thread_id in draft get/conversation get), not the draft's thread id"
        ),
    )
    _add_cc_option(reply_send)
    _add_idempotency_option(reply_send)

    owner_request = commands.add_parser(
        "request", help="ask the mailbox owner something on a conversation, or close such a request"
    )
    request_commands = owner_request.add_subparsers(dest="request_command", required=True)
    request_add = request_commands.add_parser(
        "add", help=client_defaults().get("request_add_help", 'record an open request the owner sees under "Needs you"')
    )
    request_add.add_argument("conversation_id", type=_positive_resource_id)
    request_add.add_argument("--text", required=True, action=SingleValueAction,
                             help="what you need from the owner, in a sentence or two")
    request_add.add_argument("--idempotency-key", required=True, type=_idempotency_key, action=SingleValueAction,
                             help="names this request; sending it again with the same key is safe")
    request_done = request_commands.add_parser("done", help="close a request that was answered or is no longer needed")
    request_done.add_argument("conversation_id", type=_positive_resource_id)
    request_done.add_argument("request_id", type=_positive_resource_id)

    note = commands.add_parser("note", help="add or edit internal notes")
    note_commands = note.add_subparsers(dest="note_command", required=True)
    note_add = note_commands.add_parser(
        "add", help="add an internal note without changing the status"
    )
    note_add.add_argument("conversation_id", type=_positive_resource_id)
    note_add.add_argument("--body-file", required=True, action=SingleValueAction)
    note_add.add_argument(
        "--body-format",
        default="text",
        choices=("html", "text"),
        action=SingleValueAction,
    )
    _add_idempotency_option(note_add)
    note_update = note_commands.add_parser(
        "update",
        help="replace the text of an API-written internal note at its version",
    )
    note_update.add_argument("conversation_id", type=_positive_resource_id)
    note_update.add_argument(
        "--thread",
        required=True,
        type=_positive_resource_id,
        action=SingleValueAction,
    )
    note_update.add_argument("--body-file", required=True, action=SingleValueAction)
    note_update.add_argument(
        "--body-format",
        default="text",
        choices=("html", "text"),
        action=SingleValueAction,
    )
    note_update.add_argument(
        "--expected-version",
        required=True,
        type=_version,
        action=SingleValueAction,
    )
    _add_idempotency_option(note_update)

    operation = commands.add_parser("operation", help="inspect write operations")
    operation_commands = operation.add_subparsers(
        dest="operation_command", required=True
    )
    operation_get = operation_commands.add_parser("get", help="get operation status")
    operation_get.add_argument("idempotency_key", type=_idempotency_key)

    events = commands.add_parser("events", help="consume durable Supa events")
    event_commands = events.add_subparsers(dest="event_command", required=True)
    event_next = event_commands.add_parser("next", help="read at most one event")
    event_next.add_argument(
        "--consumer", required=True, type=_event_consumer, action=SingleValueAction
    )
    event_next.add_argument(
        "--start", type=_event_start, default="now", action=SingleValueAction
    )
    event_next.add_argument(
        "--wait", type=_event_wait, default=0, action=SingleValueAction
    )
    event_next.add_argument("--manual-ack", action="store_true")

    event_watch = event_commands.add_parser("watch", help="continuously read events")
    event_watch.add_argument(
        "--consumer", required=True, type=_event_consumer, action=SingleValueAction
    )
    event_watch.add_argument("--manual-ack", action="store_true")

    event_ack = event_commands.add_parser("ack", help="acknowledge one event")
    event_ack.add_argument(
        "--consumer", required=True, type=_event_consumer, action=SingleValueAction
    )
    event_ack.add_argument("event_id", type=_positive_resource_id)
    event_ack.add_argument(
        "--ack-token", required=True, type=_event_ack_token, action=SingleValueAction
    )

    event_reset = event_commands.add_parser("reset", help="reset a durable event cursor")
    event_reset.add_argument(
        "--consumer", required=True, type=_event_consumer, action=SingleValueAction
    )
    event_reset.add_argument(
        "--to", required=True, type=_event_start, action=SingleValueAction
    )

    attachment = commands.add_parser("attachment", help="download attachments")
    attachment_commands = attachment.add_subparsers(
        dest="attachment_command", required=True
    )
    attachment_download = attachment_commands.add_parser(
        "download", help="download an attachment to a new file"
    )
    attachment_download.add_argument("conversation_id", type=_positive_resource_id)
    attachment_download.add_argument("attachment_id", type=_positive_resource_id)
    attachment_download.add_argument("--output", required=True, action=SingleValueAction)
    return parser


def _add_list_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--status", default="all", action=SingleValueAction)
    parser.add_argument("--tag", action=SingleValueAction)
    parser.add_argument("--limit", type=_limit, default=25, action=SingleValueAction)
    parser.add_argument("--cursor", action=SingleValueAction)


def _implemented_commands(environ=None) -> list[str]:
    """Derive server command names from the CLI parser's runnable leaves."""
    aliases = {
        "conversation.timeline": "conversation.get",
        "draft.schedule-cancel": "draft.schedule.cancel",
        "draft.schedule-get": "draft.schedule.get",
        "events.watch": "events.next",
    }
    names: set[str] = set()

    def collect(parser: argparse.ArgumentParser, path: tuple[str, ...]) -> None:
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, child in action.choices.items():
                    collect(child, (*path, name))
                return
        command = ".".join(path)
        if command != "configure":
            names.add(aliases.get(command, command))

    collect(_parser(environ), ())
    return sorted(names)


def _local_envelope(
    *,
    result: Any | None = None,
    code: str | None = None,
    message: str = "",
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {"api_version": API_VERSION, "request_id": None}
    if code is None:
        envelope["result"] = result
    else:
        envelope["error"] = {
            "code": code,
            "message": message,
            "retryable": retryable,
            "details": dict(details or {}),
        }
    return envelope


def _redact(value: Any, token: str | None) -> Any:
    if token and isinstance(value, str):
        return value.replace(token, "[REDACTED]")
    if isinstance(value, dict):
        return {
            _redact(key, token) if isinstance(key, str) else key: _redact(item, token)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, token) for item in value]
    return value


def _thread_label(thread: dict[str, Any]) -> str:
    if thread.get("draft") is True or thread.get("state") == "draft":
        return "DRAFT (NOT SENT)"
    if thread.get("is_internal_note") is True or thread.get("type") == "note":
        return "NOTE"
    if thread.get("type") == "customer":
        return "CUSTOMER"
    if thread.get("type") in ("message", "reply"):
        return "SENT"
    return str(thread.get("type") or "?").upper()


def _write_timeline(stream: TextIO, envelope: Any) -> None:
    result = envelope.get("result") if isinstance(envelope, dict) else None
    if not isinstance(result, dict) or not isinstance(result.get("threads"), list):
        raise ClientError(
            "The server returned an invalid conversation.",
            code="invalid_response",
            exit_code=6,
        )
    stream.write(
        f"Conversation {result.get('id')} [{result.get('status')}]: "
        f"{result.get('subject')}\n"
    )
    threads = [
        thread
        for thread in result["threads"]
        if isinstance(thread, dict) and thread.get("type") != "lineitem"
    ]
    sent_times = [
        str(thread.get("created_at") or "")
        for thread in threads
        if _thread_label(thread) == "SENT"
    ]
    drafts = []
    for thread in threads:
        label = _thread_label(thread)
        if label.startswith("DRAFT"):
            created = str(thread.get("created_at") or "")
            # Help Scout cannot delete a draft thread; Supa neutralizes it after a
            # send, so a draft older than a sent reply is a leftover, not pending.
            # The server's superseded_by_reply decides; older servers lack it, so
            # fall back to comparing timestamps.
            superseded = thread.get("superseded_by_reply")
            if not isinstance(superseded, bool):
                superseded = bool(created) and any(sent > created for sent in sent_times)
            if superseded:
                label = "DRAFT LEFTOVER"
            else:
                drafts.append(str(thread.get("id")))
        author = (thread.get("author") or {}).get("name") or "?"
        body = " ".join(str(thread.get("body_text") or "").split())
        if len(body) > 140:
            body = body[:139] + "\u2026"
        cc = thread.get("cc")
        cc_text = (
            f"  [cc: {', '.join(str(item) for item in cc)}]"
            if isinstance(cc, list) and cc
            else ""
        )
        stream.write(
            f"{thread.get('created_at')}  {label:<16}  {author}  "
            f"(thread {thread.get('id')}){cc_text}  {body}\n"
        )
    if drafts:
        stream.write(f"Unsent draft: yes (thread {', '.join(drafts)})\n")
    else:
        stream.write("Unsent draft: none\n")


def _write_json(stream: TextIO, value: Any, *, token: str | None = None) -> None:
    try:
        serialized = strict_json_dumps(_redact(value, token))
    except StrictJSONError as exc:
        raise ClientError(
            "The server returned an invalid response.",
            code="invalid_response",
            exit_code=6,
        ) from exc
    record = serialized + "\n"
    if stream.write(record) != len(record):
        raise OSError("Output stream accepted only part of the JSON record")
    stream.flush()


def _desk_report(client: SupaClient, args: argparse.Namespace) -> dict:
    """Preserve resource errors, and identify a server that predates these commands."""
    command = args.command
    if command == "mailbox":
        command = "mailbox.insights"
    elif command == "conversation":
        command = f"conversation.{args.conversation_command}"
    try:
        if command == "desk":
            return client.get_json("/agent/v1/desk", params=[("p", str(args.page))])
        if command == "mailbox.insights":
            return client.get_json("/agent/v1/mailbox/insights", params=[("months", str(args.months))])
        if command == "conversation.brief":
            if not 1000 <= args.max_chars <= 200000:
                raise UsageError("--max-chars must be between 1000 and 200000.")
            return client.get_json(f"/agent/v1/conversations/{args.conversation_id}/brief",
                params=[("max_chars", str(args.max_chars)), ("precedents", str(args.precedents)),
                        ("quotes", str(int(args.quotes)))])
        if args.question is not None:
            if not args.question.strip():
                raise UsageError("Give a nonempty question.")
            return client.post_json("/agent/v1/precedents", json={"q": args.question, "tag": args.tag or ""})
        elif args.tag is not None:
            raise UsageError("--tag goes with --question.")
        else:
            return client.get_json(f"/agent/v1/conversations/{args.conversation_id}/precedents")
    except ClientError as exc:
        if exc.code not in {"not_found", "method_not_allowed"}:
            raise
        caps = client.get_json("/agent/v1/capabilities").get("result", {})
        commands = caps.get("commands") if isinstance(caps, dict) else None
        if not isinstance(commands, dict) or command in commands.get("available", []):
            raise
        message = f"The server does not offer {command}. Run `supa capabilities` to check server support."
        envelope = dict(exc.envelope or {})
        envelope["error"] = {**envelope.get("error", {}), "message": message}
        raise ClientError(message, code=exc.code, exit_code=exc.exit_code, envelope=envelope) from exc


def _write_desk(stream: TextIO, result: dict) -> None:
    def line(value):
        # Customer and ledger text may contain line breaks or terminal controls.
        return " ".join("".join(ch if ch.isprintable() else " " for ch in str(value or "")).split())

    try:
        if result["unavailable_reason"]:
            stream.write("Review desk unavailable: " + line(result["unavailable_reason"]) + "\n")
            return
        lines = []
        for key, heading in (("held", "Held for you"), ("handled", "Handled without you")):
            count = result["counts"][key]
            more = "+" if result[key + "_more"] else ""
            day = " · " + line(result["day"]) + " UTC" if key == "handled" else ""
            lines.append(f"{heading} ({int(count)}{more}{day})")
            for item in result[key]:
                label = f"#{int(item['number'])} {line(item['subject'])[:120]}"
                if key == "held":
                    waiting = line(item.get("waiting"))
                    lines.extend((label + (f" · waiting {waiting}" if waiting else ""), "  " + line(item["reason"])[:280]))
                else:
                    clock = line(item["at"])[11:16]
                    lines.append(f"{label} · {clock} UTC · {line(item['actor'])} · {line(item['action'])[:160]}")
        if result["held_more"] or result["handled_more"]:
            lines.append(f"More: supa desk --page {int(result['page']) + 1}")
    except (TypeError, KeyError, ValueError) as exc:
        raise ClientError("The server returned an invalid desk.", code="invalid_response", exit_code=6) from exc
    stream.write("\n".join(lines) + "\n")


def _write_success(
    stdout: TextIO,
    envelope: dict[str, Any],
    *,
    output_format: str,
    token: str | None,
) -> None:
    if output_format == "json":
        _write_json(stdout, envelope, token=token)
        return
    result = _redact(envelope.get("result"), token)
    if isinstance(result, str):
        try:
            rendered = strict_json_dumps(result)[1:-1]
        except StrictJSONError as exc:
            raise ClientError(
                "The server returned an invalid response.",
                code="invalid_response",
                exit_code=6,
            ) from exc
        stdout.write(rendered + "\n")
    else:
        try:
            rendered = strict_json_dumps(result, indent=2, sort_keys=True)
        except StrictJSONError as exc:
            raise ClientError(
                "The server returned an invalid response.",
                code="invalid_response",
                exit_code=6,
            ) from exc
        stdout.write(rendered + "\n")
    stdout.flush()


def _query_params(args: argparse.Namespace, *, query: str | None = None) -> list[tuple[str, str]]:
    params = [("status", args.status), ("limit", str(args.limit))]
    if query is not None:
        params.insert(0, ("query", query))
    if args.tag is not None:
        params.append(("tag", args.tag))
    if args.cursor is not None:
        params.append(("cursor", args.cursor))
    return params


def _read_body_file(path_value: str, stdin: TextIO) -> str:
    """Read one bounded body exactly once and decode it as strict UTF-8."""
    try:
        if path_value == "-":
            binary_stream = getattr(stdin, "buffer", None)
            if binary_stream is not None:
                raw: bytes | str = binary_stream.read(MAX_WRITE_BODY_BYTES + 1)
            else:
                raw = stdin.read(MAX_WRITE_BODY_BYTES + 1)
        else:
            with Path(path_value).open("rb") as body_file:
                raw = body_file.read(MAX_WRITE_BODY_BYTES + 1)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigError(
            "The body file could not be read as UTF-8.",
            code="invalid_body_file",
            exit_code=2,
        ) from exc

    try:
        if isinstance(raw, str):
            encoded = raw.encode("utf-8", errors="strict")
            body = raw
        elif isinstance(raw, bytes):
            encoded = raw
            body = raw.decode("utf-8", errors="strict")
        else:
            raise TypeError("body reader returned an unsupported value")
    except (TypeError, UnicodeError) as exc:
        raise ConfigError(
            "The body file could not be read as UTF-8.",
            code="invalid_body_file",
            exit_code=2,
        ) from exc
    if len(encoded) > MAX_WRITE_BODY_BYTES:
        raise ConfigError(
            "The body file exceeds the client limit.",
            code="invalid_body_file",
            exit_code=2,
        )
    return body


def _write_idempotency_key(args: argparse.Namespace) -> str:
    configured = getattr(args, "idempotency_key", None)
    return configured if configured is not None else str(uuid4())


def _with_idempotency_result(
    envelope: dict[str, Any], idempotency_key: str
) -> dict[str, Any]:
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise ClientError(
            "The server returned an invalid write response.",
            code="invalid_response",
            exit_code=6,
        )
    existing = result.get("idempotency_key")
    if existing is not None and existing != idempotency_key:
        raise ClientError(
            "The server returned an invalid write response.",
            code="invalid_response",
            exit_code=6,
        )
    return {
        **envelope,
        "result": {**result, "idempotency_key": idempotency_key},
    }


def _with_idempotency_error(
    error: ClientError, idempotency_key: str
) -> ClientError:
    command = f"supa operation get {idempotency_key}"
    if error.envelope is None:
        if error.code == "transport_error":
            message = (
                "The write outcome is uncertain. "
                f"Do not retry automatically; inspect it with `{command}`."
            )
        else:
            message = (
                "The write response could not be validated. "
                f"Do not retry automatically; inspect it with `{command}`."
            )
        return ClientError(
            message,
            code=error.code,
            exit_code=error.exit_code,
            envelope=_local_envelope(
                code=error.code,
                message=message,
                retryable=False,
                details={
                    "idempotency_key": idempotency_key,
                    "operation_get_command": command,
                },
            ),
        )
    envelope = error.envelope
    if not isinstance(envelope, dict):
        return error
    error_record = envelope.get("error")
    if not isinstance(error_record, dict):
        return error
    details = error_record.get("details")
    if not isinstance(details, dict):
        return error
    extra_details: dict[str, Any] = {"idempotency_key": idempotency_key}
    if error.code in {
        "operation_in_progress",
        "outcome_unknown",
        "upstream_outcome_unknown",
    }:
        extra_details["operation_get_command"] = command
    return ClientError(
        str(error),
        code=error.code,
        exit_code=error.exit_code,
        envelope={
            **envelope,
            "error": {
                **error_record,
                "details": {**details, **extra_details},
            },
        },
    )


def _execute_write(
    client: SupaClient,
    *,
    method: str,
    path: str,
    payload: dict[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    try:
        if method == "PUT":
            envelope = client.put_json(path, json=payload)
        elif method == "POST":
            envelope = client.post_json(path, json=payload)
        else:  # pragma: no cover - private callers are fixed
            raise AssertionError("unsupported write method")
        return _with_idempotency_result(envelope, idempotency_key)
    except ClientError as error:
        mapped = _with_idempotency_error(error, idempotency_key)
        if mapped is error:
            raise
        raise mapped from error


def _close_quietly(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


def _open_exclusive_output(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    descriptor_info: os.stat_result | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        descriptor_info = os.fstat(descriptor)
        os.fchmod(descriptor, 0o600)
        return descriptor, descriptor_info
    except OSError as exc:
        if descriptor is not None:
            _close_quietly(descriptor)
        if descriptor_info is not None:
            _remove_partial(path, descriptor_info)
        elif descriptor is not None:
            try:
                path.unlink()
            except OSError:
                pass
        raise ConfigError(
            "The output path must be a new, writable regular file.",
            code="unsafe_output",
            exit_code=2,
        ) from exc


def _remove_partial(path: Path, descriptor_info: os.stat_result) -> None:
    try:
        path_info = path.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(path_info.st_mode)
        and path_info.st_dev == descriptor_info.st_dev
        and path_info.st_ino == descriptor_info.st_ino
    ):
        try:
            path.unlink()
        except OSError:
            pass


def _fsync_parent_directory(path: Path) -> None:
    parent = path.parent if path.parent != Path("") else Path(".")
    descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    except BaseException:
        _close_quietly(descriptor)
        raise
    else:
        os.close(descriptor)


def _read_token(stdin: TextIO, stderr: TextIO) -> str:
    if stdin.isatty():
        return getpass.getpass("Supa token: ", stream=stderr)
    token = stdin.read(TOKEN_MAX_BYTES + 1)
    if len(token) > TOKEN_MAX_BYTES:
        raise ConfigError(
            "A valid Supa agent token is required.",
            code="authentication_required",
            exit_code=3,
        )
    if token.endswith("\r\n"):
        return token[:-2]
    if token.endswith("\n"):
        return token[:-1]
    return token


def _invalid_event_response() -> ClientError:
    return ClientError(
        "The server returned an invalid event response.",
        code="invalid_response",
        exit_code=6,
    )


def _validate_event_envelope(
    envelope: dict[str, Any],
) -> dict[str, Any] | None:
    result = envelope.get("result")
    if not isinstance(result, dict) or set(result) != {"event"}:
        raise _invalid_event_response()
    event = result["event"]
    if event is None:
        return None
    required = {
        "event_id",
        "type",
        "conversation_id",
        "thread_id",
        "run_id",
        "source",
        "actor_type",
        "occurred_at",
        "received_at",
        "summary",
        "ack_token",
        "lease_expires_at",
    }
    if not isinstance(event, dict) or set(event) != required:
        raise _invalid_event_response()
    event_id = event["event_id"]
    if (
        not isinstance(event_id, int)
        or isinstance(event_id, bool)
        or not 1 <= event_id <= MAX_RESOURCE_ID
    ):
        raise _invalid_event_response()
    if (
        not isinstance(event["type"], str)
        or re.fullmatch(r"[a-z][a-z0-9_.-]{0,199}", event["type"]) is None
        or not isinstance(event["source"], str)
        or event["source"] not in {"helpscout", "supa"}
        or (
            event["actor_type"] is not None
            and not isinstance(event["actor_type"], str)
        )
        or event["actor_type"]
        not in {None, "customer", "user", "system_user", "team", "system"}
        or not isinstance(event["summary"], dict)
        or not isinstance(event["ack_token"], str)
        or _EVENT_ACK_TOKEN_RE.fullmatch(event["ack_token"]) is None
    ):
        raise _invalid_event_response()
    for key in ("conversation_id", "thread_id"):
        value = event[key]
        if value is not None and (
            not isinstance(value, str)
            or re.fullmatch(r"[1-9][0-9]{0,18}", value) is None
        ):
            raise _invalid_event_response()
    run_id = event["run_id"]
    if run_id is not None and (
        not isinstance(run_id, str) or _EVENT_CONSUMER_RE.fullmatch(run_id) is None
    ):
        raise _invalid_event_response()
    for key in ("occurred_at", "received_at", "lease_expires_at"):
        value = event[key]
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 64
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise _invalid_event_response()
    return event


def _validate_event_ack_envelope(
    envelope: dict[str, Any], *, event_id: int
) -> None:
    result = envelope.get("result")
    acknowledged_event_id = result.get("event_id") if isinstance(result, dict) else None
    if (
        not isinstance(result, dict)
        or set(result) != {"acknowledged", "event_id", "replayed"}
        or result["acknowledged"] is not True
        or not isinstance(acknowledged_event_id, int)
        or isinstance(acknowledged_event_id, bool)
        or acknowledged_event_id != event_id
        or not isinstance(result["replayed"], bool)
    ):
        raise _invalid_event_response()


def _validate_event_reset_envelope(
    envelope: dict[str, Any], *, consumer: str
) -> None:
    result = envelope.get("result")
    acknowledged = result.get("acknowledged_event_id") if isinstance(result, dict) else None
    if (
        not isinstance(result, dict)
        or set(result)
        != {"acknowledged_event_id", "consumer_name", "start_mode"}
        or not isinstance(acknowledged, int)
        or isinstance(acknowledged, bool)
        or not 0 <= acknowledged <= MAX_RESOURCE_ID
        or result["consumer_name"] != consumer
        or not isinstance(result["start_mode"], str)
        or result["start_mode"] not in {"now", "oldest", "event_id"}
    ):
        raise _invalid_event_response()


def _with_cursor_expired_guidance(
    error: ClientError, *, consumer: str
) -> ClientError:
    if error.code != "cursor_expired" or not isinstance(error.envelope, dict):
        return error
    error_record = error.envelope.get("error")
    if not isinstance(error_record, dict):
        return error
    details = error_record.get("details")
    if not isinstance(details, dict):
        return error
    command = f"supa events reset --consumer {consumer} --to now"
    message = f"{error_record.get('message', str(error))} Reset explicitly with `{command}`."
    return ClientError(
        message,
        code=error.code,
        exit_code=error.exit_code,
        envelope={
            **error.envelope,
            "error": {
                **error_record,
                "message": message,
                "details": {**details, "event_reset_command": command},
            },
        },
    )


def _request_next_event(
    client: SupaClient,
    *,
    consumer: str,
    start: str | int,
    wait_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        envelope = client.post_read_json(
            "/agent/v1/events/next",
            json={"consumer": consumer, "start": start, "wait": wait_seconds},
            wait_seconds=wait_seconds,
        )
    except ClientError as error:
        mapped = _with_cursor_expired_guidance(error, consumer=consumer)
        if mapped is error:
            raise
        raise mapped from error
    return envelope, _validate_event_envelope(envelope)


def _ack_event(
    client: SupaClient,
    *,
    consumer: str,
    event: dict[str, Any],
) -> None:
    event_id = event["event_id"]
    envelope = client.post_json(
        f"/agent/v1/events/{event_id}/ack",
        json={"consumer": consumer, "ack_token": event["ack_token"]},
    )
    _validate_event_ack_envelope(envelope, event_id=event_id)


def _write_event_envelope(
    stdout: TextIO,
    envelope: dict[str, Any],
    *,
    token: str,
) -> None:
    try:
        _write_json(stdout, envelope, token=token)
    except OSError as error:
        message = "The event could not be written; it was not acknowledged."
        raise ClientError(
            message,
            code="output_error",
            exit_code=6,
            envelope=_local_envelope(
                code="output_error",
                message=message,
                retryable=False,
            ),
        ) from error


def _is_retryable_event_read(error: ClientError) -> bool:
    if error.code == "transport_error":
        return True
    if not isinstance(error.envelope, dict):
        return False
    record = error.envelope.get("error")
    return isinstance(record, dict) and record.get("retryable") is True


def _watch_events(
    client: SupaClient,
    *,
    consumer: str,
    manual_ack: bool,
    stdout: TextIO,
    token: str,
) -> None:
    backoff = EVENT_WATCH_INITIAL_BACKOFF_SECONDS
    try:
        while True:
            try:
                envelope, event = _request_next_event(
                    client,
                    consumer=consumer,
                    start="now",
                    wait_seconds=EVENT_WATCH_WAIT_SECONDS,
                )
            except ClientError as error:
                if not _is_retryable_event_read(error):
                    raise
                _watch_sleep(backoff)
                backoff = min(backoff * 2.0, EVENT_WATCH_MAX_BACKOFF_SECONDS)
                continue
            backoff = EVENT_WATCH_INITIAL_BACKOFF_SECONDS
            if event is None:
                continue
            _write_event_envelope(stdout, envelope, token=token)
            if not manual_ack:
                _ack_event(client, consumer=consumer, event=event)
    except KeyboardInterrupt:
        return


def _watch_run(client: SupaClient, *, run_id: str, stdout: TextIO, token: str) -> None:
    after = 0
    backoff = EVENT_WATCH_INITIAL_BACKOFF_SECONDS
    try:
        while True:
            try:
                envelope = client.get_long_poll_json(
                    f"/agent/v1/runs/{run_id}/events",
                    params=[("after", str(after)), ("wait", str(EVENT_WATCH_WAIT_SECONDS))],
                    wait_seconds=EVENT_WATCH_WAIT_SECONDS,
                )
            except ClientError as error:
                if not _is_retryable_event_read(error):
                    raise
                _watch_sleep(backoff)
                backoff = min(backoff * 2, EVENT_WATCH_MAX_BACKOFF_SECONDS)
                continue
            backoff = EVENT_WATCH_INITIAL_BACKOFF_SECONDS
            result = envelope.get("result")
            run = result.get("run") if isinstance(result, dict) else None
            events = result.get("events") if isinstance(result, dict) else None
            if (
                not isinstance(run, dict)
                or run.get("run_id") != run_id
                or run.get("status") not in {"queued", "running", "completed", "failed", "interrupted"}
                or not isinstance(events, list)
            ):
                raise ClientError("Invalid run response.", code="invalid_response", exit_code=6)
            for event in events:
                sequence = event.get("sequence") if isinstance(event, dict) else None
                if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence != after + 1:
                    raise ClientError("Invalid run event sequence.", code="invalid_response", exit_code=6)
                _write_event_envelope(stdout, {**envelope, "result": event}, token=token)
                after = sequence
            if run["status"] in {"completed", "failed", "interrupted"}:
                _write_event_envelope(stdout, {**envelope, "result": run}, token=token)
                if run["status"] != "completed":
                    raise ClientError(
                        "Task run failed or was interrupted.",
                        code="run_failed" if run["status"] == "failed" else "run_interrupted",
                        exit_code=7,
                    )
                return
    except KeyboardInterrupt:
        return


def _execute(
    args: argparse.Namespace,
    *,
    output_format: str,
    transport: httpx.BaseTransport | None,
    environ: Mapping[str, str],
    stdin: TextIO,
    stdout: TextIO,
) -> str | None:
    if args.command == "configure":
        token = _read_token(stdin, sys.stderr)
        normalized_url = write_config(environ, url=args.url, token=token)
        _write_success(
            stdout,
            _local_envelope(result={"configured": True, "url": normalized_url}),
            output_format=output_format,
            token=token.strip() or None,
        )
        return None

    config = load_config(environ)
    write_idempotency_key: str | None = None
    with SupaClient(config, transport=transport) as client:
        if args.command == "health":
            envelope = client.get_json("/agent/v1/health")
        elif args.command == "capabilities":
            envelope = client.get_json("/agent/v1/capabilities")
            result = envelope.get("result")
            if not isinstance(result, dict):
                raise ClientError(
                    "The server returned invalid capabilities.",
                    code="invalid_response",
                    exit_code=6,
                )
            envelope = dict(envelope)
            commands = result.get("commands", {})
            available = commands.get("available", []) if isinstance(commands, dict) else []
            if not isinstance(available, list) or any(not isinstance(name, str) for name in available):
                raise ClientError(
                    "The server returned invalid capabilities.",
                    code="invalid_response",
                    exit_code=6,
                )
            client_missing = sorted(set(available) - set(_implemented_commands(environ)))
            envelope["result"] = {
                **result,
                "cli_version": __version__,
                "client_missing": client_missing,
            }
            if client_missing:
                from .installation import client_defaults
                print(
                    client_defaults().get(
                        "client_update_warning",
                        "Warning: this supa client lacks server commands; reinstall with "
                        "`uv tool install --force --reinstall --no-cache .` from a Supa checkout.",
                    ),
                    file=sys.stderr,
                )
        elif (args.command == "desk" or
              (args.command == "mailbox" and args.mailbox_command == "insights") or
              (args.command == "conversation" and args.conversation_command in ("brief", "precedents"))):
            envelope = _desk_report(client, args)
            if args.command == "desk" and output_format == "text":
                _write_desk(stdout, _redact(envelope.get("result"), config.token))
                return config.token
        elif args.command == "mailbox" and args.mailbox_command == "operations":
            envelope = client.get_json("/agent/v1/mailbox/operations")
        elif hasattr(args, "extension_read"):
            envelope = args.extension_read(client, args)
        elif args.command == "ticket" or hasattr(args, "extension_task"):
            idempotency_key = _write_idempotency_key(args)
            write_idempotency_key = idempotency_key
            arguments: dict[str, Any] = {"conversation_id": args.conversation_id}
            task_command = args.ticket_command if args.command == "ticket" else args.extension_task
            if task_command == "process":
                arguments["allow_send"] = bool(args.allow_send)
            if task_command == "recover":
                for field in ("expected_thread_id", "expected_draft_version", "expected_conversation_version"):
                    arguments[field] = getattr(args, field)
            envelope = client.post_json(
                "/agent/v1/tasks",
                json={
                    "task_name": f"{args.command}.{task_command}",
                    "arguments": arguments,
                    "idempotency_key": idempotency_key,
                },
            )
        elif args.command == "run" and args.run_command == "get":
            envelope = client.get_json(f"/agent/v1/runs/{args.run_id}")
        elif args.command == "run" and args.run_command == "watch":
            _watch_run(client, run_id=args.run_id, stdout=stdout, token=config.token)
            return config.token
        elif args.command == "conversation" and args.conversation_command == "get":
            envelope = client.get_json(
                f"/agent/v1/conversations/{args.conversation_id}"
            )
        elif args.command == "conversation" and args.conversation_command == "timeline":
            envelope = client.get_json(
                f"/agent/v1/conversations/{args.conversation_id}"
            )
            _write_timeline(stdout, envelope)
            return config.token
        elif args.command == "conversation" and args.conversation_command == "state":
            changes = {
                "stage": args.stage_aliases.get(args.stage, args.stage),
                "waiting_on": args.waiting_on,
                "reminders": args.reminders,
                "handling": "person" if args.take_over else None,
                "has_site": {"yes": True, "no": False}.get(args.has_site),
                "paid": {"yes": True, "no": False}.get(args.paid),
            }
            changes = {key: value for key, value in changes.items() if value is not None}
            path = f"/agent/v1/conversations/{args.conversation_id}/state"
            if changes:
                envelope = client.post_json(path, json={"changes": changes, "reason": args.reason})
            elif args.reason:
                raise UsageError("--reason goes with a change (--stage, --waiting-on, --reminders, …).")
            else:
                envelope = client.get_json(path)
        elif args.command == "conversation" and args.conversation_command == "list":
            envelope = client.get_json(
                "/agent/v1/conversations", params=_query_params(args)
            )
        elif args.command == "conversation" and args.conversation_command == "search":
            envelope = client.get_json(
                "/agent/v1/conversations",
                params=_query_params(args, query=args.query),
            )
        elif args.command == "conversation" and args.conversation_command == "create":
            body = _read_body_file(args.body_file, stdin)
            idempotency_key = _write_idempotency_key(args)
            write_idempotency_key = idempotency_key
            envelope = _execute_write(
                client,
                method="POST",
                path="/agent/v1/conversations",
                payload={
                    "subject": args.subject,
                    "mailbox_id": args.mailbox,
                    "customer_id": args.customer,
                    "body": body,
                    "body_format": args.body_format,
                    "draft": not args.send,
                    "idempotency_key": idempotency_key,
                },
                idempotency_key=idempotency_key,
            )
        elif args.command == "conversation" and args.conversation_command == "discard":
            idempotency_key = _write_idempotency_key(args)
            write_idempotency_key = idempotency_key
            envelope = _execute_write(
                client,
                method="POST",
                path=f"/agent/v1/conversations/{args.conversation_id}/discard",
                payload={
                    "expected_version": args.expected_version,
                    "idempotency_key": idempotency_key,
                },
                idempotency_key=idempotency_key,
            )
        elif args.command == "draft" and args.draft_command == "get":
            envelope = client.get_json(
                f"/agent/v1/conversations/{args.conversation_id}/draft"
            )
        elif args.command == "draft" and args.draft_command == "save":
            body = _read_body_file(args.body_file, stdin)
            idempotency_key = _write_idempotency_key(args)
            write_idempotency_key = idempotency_key
            envelope = _execute_write(
                client,
                method="PUT",
                path=f"/agent/v1/conversations/{args.conversation_id}/draft",
                payload={
                    "body": body,
                    "body_format": args.body_format,
                    "expected_version": args.expected_version,
                    "idempotency_key": idempotency_key,
                },
                idempotency_key=idempotency_key,
            )
        elif args.command == "draft" and args.draft_command == "send":
            idempotency_key = _write_idempotency_key(args)
            write_idempotency_key = idempotency_key
            envelope = _execute_write(
                client,
                method="POST",
                path=(
                    f"/agent/v1/conversations/{args.conversation_id}/draft/send"
                ),
                payload={
                    "expected_version": args.expected_version,
                    "expected_latest_thread_id": args.expected_thread,
                    **_cc_payload(args),
                    "idempotency_key": idempotency_key,
                },
                idempotency_key=idempotency_key,
            )
        elif args.command == "draft" and args.draft_command == "schedule":
            payload: dict[str, Any] = {
                "expected_latest_thread_id": args.expected_thread,
            }
            if args.body_file is not None:
                payload["body"] = _read_body_file(args.body_file, stdin)
                payload["body_format"] = args.body_format
            if args.expected_version is not None:
                payload["expected_version"] = args.expected_version
            if args.send_at is not None:
                payload["send_at"] = args.send_at
            if args.after_customer is not None:
                payload["after_customer_min_minutes"] = args.after_customer[0]
                payload["after_customer_max_minutes"] = args.after_customer[1]
            payload.update(_cc_payload(args))
            idempotency_key = _write_idempotency_key(args)
            write_idempotency_key = idempotency_key
            payload["idempotency_key"] = idempotency_key
            envelope = _execute_write(
                client,
                method="POST",
                path=f"/agent/v1/conversations/{args.conversation_id}/draft/schedule",
                payload=payload,
                idempotency_key=idempotency_key,
            )
        elif args.command == "draft" and args.draft_command == "schedule-get":
            envelope = client.get_json(
                f"/agent/v1/conversations/{args.conversation_id}/draft/schedule"
            )
        elif args.command == "draft" and args.draft_command == "schedule-cancel":
            envelope = client.post_json(
                f"/agent/v1/conversations/{args.conversation_id}/draft/schedule/cancel",
                json={},
            )
        elif args.command == "reply" and args.reply_command == "send":
            body = _read_body_file(args.body_file, stdin)
            idempotency_key = _write_idempotency_key(args)
            write_idempotency_key = idempotency_key
            envelope = _execute_write(
                client,
                method="POST",
                path=f"/agent/v1/conversations/{args.conversation_id}/replies",
                payload={
                    "body": body,
                    "body_format": args.body_format,
                    "expected_latest_thread_id": args.expected_thread,
                    **_cc_payload(args),
                    "idempotency_key": idempotency_key,
                },
                idempotency_key=idempotency_key,
            )
        elif args.command == "request" and args.request_command == "add":
            # No operation record backs a request: after an uncertain answer, send it again with
            # the same key; the mailbox stores it once and reports its present status.
            envelope = client.post_json(
                f"/agent/v1/conversations/{args.conversation_id}/requests",
                json={"text": args.text, "idempotency_key": args.idempotency_key},
            )
        elif args.command == "request" and args.request_command == "done":
            envelope = client.post_json(
                f"/agent/v1/conversations/{args.conversation_id}/requests/{args.request_id}/done", json={}
            )
        elif args.command == "note" and args.note_command == "add":
            body = _read_body_file(args.body_file, stdin)
            idempotency_key = _write_idempotency_key(args)
            write_idempotency_key = idempotency_key
            envelope = _execute_write(
                client,
                method="POST",
                path=f"/agent/v1/conversations/{args.conversation_id}/notes",
                payload={
                    "body": body,
                    "body_format": args.body_format,
                    "idempotency_key": idempotency_key,
                },
                idempotency_key=idempotency_key,
            )
        elif args.command == "note" and args.note_command == "update":
            body = _read_body_file(args.body_file, stdin)
            idempotency_key = _write_idempotency_key(args)
            write_idempotency_key = idempotency_key
            envelope = _execute_write(
                client,
                method="PUT",
                path=(
                    f"/agent/v1/conversations/{args.conversation_id}"
                    f"/notes/{args.thread}"
                ),
                payload={
                    "body": body,
                    "body_format": args.body_format,
                    "expected_version": args.expected_version,
                    "idempotency_key": idempotency_key,
                },
                idempotency_key=idempotency_key,
            )
        elif args.command == "operation" and args.operation_command == "get":
            envelope = client.get_json(
                f"/agent/v1/operations/{args.idempotency_key}"
            )
        elif args.command == "events" and args.event_command == "next":
            envelope, event = _request_next_event(
                client,
                consumer=args.consumer,
                start=args.start,
                wait_seconds=args.wait,
            )
            _write_event_envelope(stdout, envelope, token=config.token)
            if event is not None and not args.manual_ack:
                _ack_event(client, consumer=args.consumer, event=event)
            return config.token
        elif args.command == "events" and args.event_command == "watch":
            _watch_events(
                client,
                consumer=args.consumer,
                manual_ack=args.manual_ack,
                stdout=stdout,
                token=config.token,
            )
            return config.token
        elif args.command == "events" and args.event_command == "ack":
            envelope = client.post_json(
                f"/agent/v1/events/{args.event_id}/ack",
                json={"consumer": args.consumer, "ack_token": args.ack_token},
            )
            _validate_event_ack_envelope(envelope, event_id=args.event_id)
        elif args.command == "events" and args.event_command == "reset":
            envelope = client.post_json(
                "/agent/v1/events/reset",
                json={"consumer": args.consumer, "to": args.to},
            )
            _validate_event_reset_envelope(envelope, consumer=args.consumer)
        elif args.command == "attachment" and args.attachment_command == "download":
            output = Path(args.output)
            descriptor, descriptor_info = _open_exclusive_output(output)
            try:
                metadata = client.download_to(
                    (
                        f"/agent/v1/conversations/{args.conversation_id}"
                        f"/attachments/{args.attachment_id}"
                    ),
                    descriptor=descriptor,
                )
                os.fsync(descriptor)
                _fsync_parent_directory(output)
            except OSError as exc:
                _close_quietly(descriptor)
                _remove_partial(output, descriptor_info)
                raise ConfigError(
                    "The attachment could not be written to the output path.",
                    code="unsafe_output",
                    exit_code=2,
                ) from exc
            except BaseException:
                _close_quietly(descriptor)
                _remove_partial(output, descriptor_info)
                raise
            else:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    _remove_partial(output, descriptor_info)
                    raise ConfigError(
                        "The attachment could not be written to the output path.",
                        code="unsafe_output",
                        exit_code=2,
                    ) from exc
            envelope = {
                "api_version": metadata["api_version"],
                "request_id": metadata["request_id"],
                "result": {
                    "attachment_id": args.attachment_id,
                    "bytes": metadata["bytes"],
                    "content_type": metadata["content_type"],
                    "conversation_id": args.conversation_id,
                    "output": str(output),
                },
            }
        else:  # pragma: no cover - argparse owns command coverage
            raise UsageError("Invalid command usage.")
        try:
            _write_success(
                stdout,
                envelope,
                output_format=output_format,
                token=config.token,
            )
        except OSError as exc:
            if args.command == "events" and args.event_command == "ack":
                message = (
                    "The acknowledgement completed, but its result could not be written. "
                    "Repeating the same acknowledgement is safe."
                )
                raise ClientError(
                    message,
                    code="output_error",
                    exit_code=6,
                    envelope=_local_envelope(
                        code="output_error",
                        message=message,
                        retryable=False,
                        details={"event_id": args.event_id},
                    ),
                ) from exc
            if args.command == "events" and args.event_command == "reset":
                message = (
                    "The reset completed, but its result could not be written. "
                    "Do not repeat it automatically because doing so may move the cursor again."
                )
                raise ClientError(
                    message,
                    code="output_error",
                    exit_code=6,
                    envelope=_local_envelope(
                        code="output_error",
                        message=message,
                        retryable=False,
                        details={"consumer": args.consumer},
                    ),
                ) from exc
            if write_idempotency_key is None:
                raise
            if args.command == "ticket":
                message = (
                    "The task may have started, but its run ID could not be written. "
                    "Repeat the same ticket command with the same idempotency key to retrieve it."
                )
                raise ClientError(
                    message,
                    code="output_error",
                    exit_code=6,
                    envelope=_local_envelope(
                        code="output_error", message=message, retryable=False,
                        details={"idempotency_key": write_idempotency_key},
                    ),
                ) from exc
            command = f"supa operation get {write_idempotency_key}"
            message = (
                "The write completed, but its result could not be written. "
                f"Do not retry automatically; inspect it with `{command}`."
            )
            raise ClientError(
                message,
                code="output_error",
                exit_code=6,
                envelope=_local_envelope(
                    code="output_error",
                    message=message,
                    retryable=False,
                    details={
                        "idempotency_key": write_idempotency_key,
                        "operation_get_command": command,
                    },
                ),
            ) from exc
        return config.token


def _validate_cc_args(args: argparse.Namespace) -> None:
    """`--cc` takes at most two addresses (the server enforces the same)."""
    cc = getattr(args, "cc", None) or []
    if len({address.lower() for address in cc}) > MAX_CC_ADDRESSES:
        raise UsageError(f"--cc takes at most {MAX_CC_ADDRESSES} addresses.")


def _validate_schedule_args(args: argparse.Namespace) -> None:
    """`draft schedule` needs a body (preferred) or the saved draft's version."""
    if args.command != "draft" or args.draft_command != "schedule":
        return
    if (args.body_file is None) != (args.body_format is None):
        raise UsageError("--body-file and --body-format go together.")
    if args.body_file is None and args.expected_version is None:
        raise UsageError(
            "Give --body-file/--body-format (preferred) or the saved draft's "
            "--expected-version."
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one command and return its stable process exit code."""
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    actual_environ = os.environ if environ is None else environ
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    error_stream = sys.stderr if stderr is None else stderr
    token_for_redaction = actual_environ.get("SUPA_TOKEN")

    try:
        cleaned_argv, output_format = _extract_format(actual_argv)
        parser = _parser(actual_environ)
        with contextlib.redirect_stdout(output_stream), contextlib.redirect_stderr(error_stream):
            try:
                args = parser.parse_args(cleaned_argv)
            except ParserExit as exc:
                return exc.status
        # The parser only sees a cleaned argv; preserve the pre-parsed global option.
        args.format = output_format
        _validate_schedule_args(args)
        _validate_cc_args(args)
        # getpass uses its explicit stream, but _execute does not otherwise print prompts.
        with contextlib.redirect_stderr(error_stream):
            returned_token = _execute(
                args,
                output_format=output_format,
                transport=transport,
                environ=actual_environ,
                stdin=input_stream,
                stdout=output_stream,
            )
        token_for_redaction = returned_token or token_for_redaction
        return 0
    except UsageError as exc:
        message = str(exc)
        for index, argument in enumerate(actual_argv):
            if argument == "--token" and index + 1 < len(actual_argv):
                message = _redact(message, actual_argv[index + 1])
            elif argument.startswith("--token="):
                message = _redact(message, argument.split("=", 1)[1])
        message = _TOKEN_LIKE_RE.sub("[REDACTED]", message)
        _write_json(
            error_stream,
            _local_envelope(code="usage_error", message=message),
            token=token_for_redaction,
        )
        return 2
    except ConfigError as exc:
        _write_json(
            error_stream,
            _local_envelope(code=exc.code, message=str(exc)),
            token=token_for_redaction,
        )
        return exc.exit_code
    except ClientError as exc:
        envelope = exc.envelope or _local_envelope(code=exc.code, message=str(exc))
        _write_json(error_stream, envelope, token=token_for_redaction)
        return exc.exit_code
    except argparse.ArgumentError:
        _write_json(
            error_stream,
            _local_envelope(code="usage_error", message="Invalid command usage."),
            token=token_for_redaction,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
