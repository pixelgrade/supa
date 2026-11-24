#!/usr/bin/env python3
"""Create, list, restrict, and revoke scoped Agent API tokens on the local server."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.agent_auth import (  # noqa: E402
    ALLOWED_SCOPES,
    AgentTokenRegistryError,
    agent_token_registry_lock,
    generate_token,
    get_agent_tokens_path,
    hash_token,
    load_token_records,
    validate_conversation_allowlist,
    write_token_records,
)


def _parse_scopes(value: str) -> list[str]:
    scopes = [scope.strip() for scope in value.split(",") if scope.strip()]
    unknown = set(scopes) - ALLOWED_SCOPES
    if not scopes:
        raise ValueError("at least one scope is required")
    if unknown:
        raise ValueError("unknown scope")
    if len(set(scopes)) != len(scopes):
        raise ValueError("duplicate scope")
    return sorted(scopes)


def _parse_conversations(value: str) -> list[int]:
    """Parse a comma-separated list of Help Scout conversation ids."""
    items = [item.strip() for item in value.split(",")]
    if not any(items):
        raise ValueError("at least one conversation id is required")
    conversations: list[int] = []
    for item in items:
        if not item:
            raise ValueError("empty conversation id")
        if not (item.isascii() and item.isdecimal()) or len(item) > 19:
            raise ValueError("conversation ids must be positive integers")
        conversations.append(int(item))
    return validate_conversation_allowlist(conversations)


def _load(path: Path, *, missing_ok: bool) -> list[dict[str, object]]:
    return load_token_records(path, missing_ok=missing_ok)


def _create(args: argparse.Namespace, path: Path) -> int:
    scopes = _parse_scopes(args.scopes)
    conversations = (
        _parse_conversations(args.conversations)
        if args.conversations is not None
        else None
    )
    name = args.name.strip()
    if not name:
        raise ValueError("name must not be empty")
    with agent_token_registry_lock(path):
        records = _load(path, missing_ok=True)
        if any(record["name"] == name for record in records):
            raise ValueError("token name already exists")

        existing_ids = {str(record["token_id"]) for record in records}
        token_id = str(uuid.uuid4())
        while token_id in existing_ids:
            token_id = str(uuid.uuid4())
        token = generate_token()
        created_at = datetime.now(timezone.utc).isoformat()
        record: dict[str, object] = {
            "token_id": token_id,
            "name": name,
            "hash": hash_token(token),
            "scopes": scopes,
            "created_at": created_at,
        }
        if conversations is not None:
            record["conversations"] = conversations
        records.append(record)
        write_token_records(path, records)
    print(
        json.dumps(
            {
                "token_id": token_id,
                "name": name,
                "scopes": scopes,
                "conversations": conversations,
                "token": token,
            },
            sort_keys=True,
        )
    )
    return 0


def _set_conversations(args: argparse.Namespace, path: Path) -> int:
    """Change a token's conversation allowlist without rotating its secret."""
    conversations = (
        None if args.unrestricted else _parse_conversations(args.conversations)
    )
    with agent_token_registry_lock(path):
        records = _load(path, missing_ok=False)
        matches = [record for record in records if record["name"] == args.name]
        if not matches:
            raise ValueError("token name was not found")
        record = matches[0]
        record.pop("conversations", None)
        if conversations is not None:
            record["conversations"] = conversations
        write_token_records(path, records)
    print(
        json.dumps(
            {"name": args.name, "conversations": conversations},
            sort_keys=True,
        )
    )
    return 0


def _set_scopes(args: argparse.Namespace, path: Path) -> int:
    """Change a token's scopes without rotating its secret (the allowlist is kept)."""
    scopes = _parse_scopes(args.scopes)
    with agent_token_registry_lock(path):
        records = _load(path, missing_ok=False)
        matches = [record for record in records if record["name"] == args.name]
        if not matches:
            raise ValueError("token name was not found")
        matches[0]["scopes"] = scopes
        write_token_records(path, records)
    print(json.dumps({"name": args.name, "scopes": scopes}, sort_keys=True))
    return 0


def _list_tokens(path: Path) -> int:
    records = _load(path, missing_ok=True)
    public_records = [
        {
            "token_id": record["token_id"],
            "name": record["name"],
            "scopes": sorted(record["scopes"]),
            "conversations": record.get("conversations"),
            "created_at": record.get("created_at"),
        }
        for record in records
    ]
    print(json.dumps({"tokens": public_records}, indent=2, sort_keys=True))
    return 0


def _revoke(args: argparse.Namespace, path: Path) -> int:
    with agent_token_registry_lock(path):
        records = _load(path, missing_ok=False)
        retained = [record for record in records if record["name"] != args.name]
        if len(retained) == len(records):
            raise ValueError("token name was not found")
        write_token_records(path, retained)
    print(json.dumps({"revoked": args.name}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    create = subcommands.add_parser("create", help="create a scoped token")
    create.add_argument("--name", required=True, help="stable local identity name")
    create.add_argument(
        "--scopes",
        required=True,
        help="comma-separated scopes: read,draft,send,note,events,task",
    )
    create.add_argument(
        "--conversations",
        help=(
            "comma-separated Help Scout conversation ids this token may touch; "
            "omit for an unrestricted token"
        ),
    )

    restrict = subcommands.add_parser(
        "set-conversations",
        help="change a token's conversation allowlist without rotating it",
    )
    restrict.add_argument("--name", required=True, help="identity name to change")
    target = restrict.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--conversations",
        help="comma-separated Help Scout conversation ids (replaces the list)",
    )
    target.add_argument(
        "--unrestricted",
        action="store_true",
        help="remove the restriction (the token may touch any conversation)",
    )

    rescope = subcommands.add_parser(
        "set-scopes",
        help="change a token's scopes without rotating it",
    )
    rescope.add_argument("--name", required=True, help="identity name to change")
    rescope.add_argument(
        "--scopes",
        required=True,
        help="comma-separated scopes (replaces the list): read,draft,send,note,events,task",
    )

    subcommands.add_parser("list", help="list token identities (never secrets or hashes)")

    revoke = subcommands.add_parser("revoke", help="revoke a token identity by name")
    revoke.add_argument("--name", required=True, help="identity name to revoke")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = get_agent_tokens_path()
    try:
        if args.command == "create":
            return _create(args, path)
        if args.command == "list":
            return _list_tokens(path)
        if args.command == "set-conversations":
            return _set_conversations(args, path)
        if args.command == "set-scopes":
            return _set_scopes(args, path)
        if args.command == "revoke":
            return _revoke(args, path)
    except (AgentTokenRegistryError, OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
