#!/usr/bin/env python3
"""Inspect or resolve agent operations from the local Supa server only."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.agent_operations import (  # noqa: E402
    AgentOperationError,
    get_operation_locally,
    resolve_operation_locally,
)
from services.runtime_paths import get_supa_db_path  # noqa: E402


_FileSignature = tuple[int, int, int, int]


def _lexical_absolute_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(Path(value).expanduser())))


def _path_components(path: Path) -> list[Path]:
    components = [path]
    components.extend(path.parents)
    components.reverse()
    return components


def _validate_local_db_path(path: Path) -> _FileSignature:
    current_uid = os.getuid()
    trusted_owners = {0, current_uid}
    target_stat: os.stat_result | None = None
    for component in _path_components(path):
        try:
            component_stat = component.lstat()
        except OSError as error:
            raise ValueError("database path does not exist") from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValueError("database path must not contain symlinks")
        is_target = component == path
        expected_type = stat.S_ISREG if is_target else stat.S_ISDIR
        if not expected_type(component_stat.st_mode):
            raise ValueError("database path has an invalid component")
        if component_stat.st_uid not in trusted_owners:
            raise ValueError("database path is not owner controlled")
        if is_target and component_stat.st_uid != current_uid:
            raise ValueError("database file must be owned by the current user")
        if stat.S_IMODE(component_stat.st_mode) & 0o022:
            raise ValueError("database path must not be group/world writable")
        if is_target:
            target_stat = component_stat
    assert target_stat is not None
    return (
        target_stat.st_dev,
        target_stat.st_ino,
        target_stat.st_uid,
        stat.S_IFMT(target_stat.st_mode),
    )


def _existing_local_db(value: str | os.PathLike[str] | None) -> Path:
    """Resolve and validate an existing owner-controlled local database."""

    configured = _lexical_absolute_path(
        value if value is not None else get_supa_db_path()
    )
    try:
        before = _validate_local_db_path(configured)
        after = _validate_local_db_path(configured)
    except OSError as error:  # pragma: no cover - validation normalizes OSErrors
        raise ValueError("database path is unavailable") from error
    if before != after:
        raise ValueError("database path changed during validation")
    return configured


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        help="existing local Supa SQLite DB (defaults to SUPA_DB_PATH)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    show = commands.add_parser("show", help="show a sanitized operation record")
    show.add_argument("key", help="idempotency key")

    resolve = commands.add_parser(
        "resolve", help="resolve an unknown or stale operation locally"
    )
    resolve.add_argument("key", help="idempotency key")
    resolve.add_argument(
        "--as",
        dest="outcome",
        choices=("succeeded", "failed"),
        required=True,
        help="definitive locally observed outcome",
    )
    resolve.add_argument(
        "--thread-id",
        help="observed Help Scout thread ID, when one exists",
    )
    resolve.add_argument(
        "--reason",
        required=True,
        help="required audit reason for the manual resolution",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        db_path = _existing_local_db(args.db)
        before = _validate_local_db_path(db_path)
        if args.command == "show":
            operation = get_operation_locally(
                db_path=db_path,
                idempotency_key=args.key,
            )
        elif args.command == "resolve":
            operation = resolve_operation_locally(
                db_path=db_path,
                idempotency_key=args.key,
                outcome=args.outcome,
                reason=args.reason,
                upstream_thread_id=args.thread_id,
            )
        else:  # pragma: no cover - argparse enforces the command set
            raise AssertionError("unreachable")
        if _validate_local_db_path(db_path) != before:
            raise ValueError("database path changed during operation")
    except (AgentOperationError, OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

    print(json.dumps(operation.to_public_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
