"""Scoped bearer-token authentication for the Agent API.

The registry contains token identities and SHA-256 hashes only. Raw bearer
tokens exist only at creation time and in an incoming request.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import json
import os
import secrets
import stat
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from fastapi import Header, HTTPException


ALLOWED_SCOPES = frozenset({"read", "draft", "send", "note", "events", "task", "create", "delete"})
TOKEN_RECORD_KEYS = frozenset({"token_id", "name", "hash", "scopes", "created_at"})
OPTIONAL_TOKEN_RECORD_KEYS = frozenset({"conversations"})
MAX_RESOURCE_ID = (1 << 63) - 1
# A per-case token names a handful of conversations; a long list is a mistake.
MAX_TOKEN_CONVERSATIONS = 100
CONVERSATION_NOT_ALLOWED_CODE = "conversation_not_allowed"
DEFAULT_AGENT_TOKENS_PATH = (
    Path(__file__).resolve().parent.parent / "db" / "agent-tokens.json"
)


class AgentAuthError(Exception):
    """A credential could not be authenticated."""


class AgentTokenRegistryError(Exception):
    """The on-disk token registry is missing or malformed."""


@dataclass(frozen=True)
class AgentIdentity:
    token_id: str
    name: str
    scopes: frozenset[str]
    # None means unrestricted. A frozenset restricts every conversation-scoped
    # operation (reads, writes, events, tasks) to exactly these Help Scout ids.
    conversations: frozenset[int] | None = None

    @property
    def is_conversation_restricted(self) -> bool:
        return self.conversations is not None

    def allows_conversation(self, conversation_id: Any) -> bool:
        """Fail closed: a restricted identity allows only listed positive ints."""
        if self.conversations is None:
            return True
        normalized = normalize_conversation_id(conversation_id)
        return normalized is not None and normalized in self.conversations


def normalize_conversation_id(value: Any) -> int | None:
    """Return a positive conversation id from an int or decimal string, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        candidate = value
    elif isinstance(value, str) and value.isascii() and value.isdecimal():
        if len(value) > 19:
            return None
        candidate = int(value)
    else:
        return None
    if not 1 <= candidate <= MAX_RESOURCE_ID:
        return None
    return candidate


def validate_conversation_allowlist(value: Any) -> list[int]:
    """Validate a stored/requested allowlist: non-empty, unique positive ints."""
    if not isinstance(value, list) or not value:
        raise ValueError("conversations must be a non-empty list")
    if len(value) > MAX_TOKEN_CONVERSATIONS:
        raise ValueError("too many conversations")
    normalized: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("conversation ids must be positive integers")
        if not 1 <= item <= MAX_RESOURCE_ID:
            raise ValueError("conversation ids must be positive integers")
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate conversation id")
    return sorted(normalized)


@dataclass(frozen=True)
class _TokenRecord:
    identity: AgentIdentity
    digest: bytes


def get_agent_tokens_path() -> Path:
    """Return the configured registry path without loading environment files."""
    configured = os.getenv("SUPA_AGENT_TOKENS_FILE", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_AGENT_TOKENS_PATH


def generate_token() -> str:
    """Generate an unpadded URL-safe token from 32 random bytes."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")


def hash_token(token: str) -> str:
    """Return the persistence-safe SHA-256 hex digest for a raw token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _validate_record(record: Any) -> tuple[dict[str, Any], _TokenRecord]:
    if (
        not isinstance(record, dict)
        or not TOKEN_RECORD_KEYS.issubset(record)
        or not set(record).issubset(TOKEN_RECORD_KEYS | OPTIONAL_TOKEN_RECORD_KEYS)
    ):
        raise AgentTokenRegistryError("Agent token registry is invalid")

    token_id = record.get("token_id")
    name = record.get("name")
    digest_hex = record.get("hash")
    scopes_value = record.get("scopes")
    created_at = record.get("created_at")
    if not isinstance(token_id, str) or not token_id.strip():
        raise AgentTokenRegistryError("Agent token registry is invalid")
    if not isinstance(name, str) or not name.strip():
        raise AgentTokenRegistryError("Agent token registry is invalid")
    if not isinstance(digest_hex, str) or len(digest_hex) != 64:
        raise AgentTokenRegistryError("Agent token registry is invalid")
    try:
        digest = bytes.fromhex(digest_hex)
    except ValueError as exc:
        raise AgentTokenRegistryError("Agent token registry is invalid") from exc
    if len(digest) != hashlib.sha256().digest_size:
        raise AgentTokenRegistryError("Agent token registry is invalid")
    if not isinstance(scopes_value, list) or not all(
        isinstance(scope, str) for scope in scopes_value
    ):
        raise AgentTokenRegistryError("Agent token registry is invalid")
    scopes = frozenset(scopes_value)
    if not scopes.issubset(ALLOWED_SCOPES):
        raise AgentTokenRegistryError("Agent token registry contains an unknown scope")
    if len(scopes) != len(scopes_value):
        raise AgentTokenRegistryError("Agent token registry is invalid")
    if not isinstance(created_at, str):
        raise AgentTokenRegistryError("Agent token registry is invalid")
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise AgentTokenRegistryError("Agent token registry is invalid") from exc
    if created.tzinfo is None or created.utcoffset() is None:
        raise AgentTokenRegistryError("Agent token registry is invalid")

    conversations: frozenset[int] | None = None
    conversations_value = record.get("conversations")
    if conversations_value is not None:
        # An empty list is invalid rather than "no access": the operator must
        # say either which conversations or unrestricted (field absent).
        try:
            allowlist = validate_conversation_allowlist(conversations_value)
        except ValueError as exc:
            raise AgentTokenRegistryError("Agent token registry is invalid") from exc
        conversations = frozenset(allowlist)

    normalized = dict(record)
    normalized["token_id"] = token_id.strip()
    normalized["name"] = name.strip()
    normalized["hash"] = digest_hex.lower()
    normalized["scopes"] = list(scopes_value)
    normalized.pop("conversations", None)
    if conversations is not None:
        normalized["conversations"] = sorted(conversations)
    loaded = _TokenRecord(
        identity=AgentIdentity(
            token_id=token_id.strip(),
            name=name.strip(),
            scopes=scopes,
            conversations=conversations,
        ),
        digest=digest,
    )
    return normalized, loaded


_FileSignature = tuple[int, int, int, int, int]


def _secure_open_flags() -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    return flags | getattr(os, "O_NOFOLLOW", 0)


def _validate_private_regular_file(file_stat: os.stat_result) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise AgentTokenRegistryError("Agent token registry is unavailable")
    if file_stat.st_uid != os.getuid():
        raise AgentTokenRegistryError("Agent token registry is unavailable")
    if stat.S_IMODE(file_stat.st_mode) != 0o600:
        raise AgentTokenRegistryError("Agent token registry is unavailable")


def _file_signature(file_stat: os.stat_result) -> _FileSignature:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
        file_stat.st_size,
    )


def _open_registry(
    path: Path, *, missing_ok: bool
) -> tuple[int, _FileSignature] | None:
    try:
        descriptor = os.open(path, _secure_open_flags())
    except FileNotFoundError:
        if missing_ok:
            return None
        raise AgentTokenRegistryError("Agent token registry is unavailable") from None
    except OSError as exc:
        raise AgentTokenRegistryError("Agent token registry is unavailable") from exc
    try:
        file_stat = os.fstat(descriptor)
        _validate_private_regular_file(file_stat)
        return descriptor, _file_signature(file_stat)
    except BaseException:
        os.close(descriptor)
        raise


def _read_registry_descriptor(descriptor: int) -> str:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 64 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AgentTokenRegistryError("Agent token registry is invalid") from exc


def _parse_token_records(raw: str) -> list[dict[str, Any]]:
    """Parse and strictly validate a registry snapshot."""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AgentTokenRegistryError("Agent token registry is invalid") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"tokens"}
        or not isinstance(payload["tokens"], list)
    ):
        raise AgentTokenRegistryError("Agent token registry is invalid")

    records: list[dict[str, Any]] = []
    token_ids: set[str] = set()
    names: set[str] = set()
    digests: set[bytes] = set()
    for raw_record in payload["tokens"]:
        record, loaded = _validate_record(raw_record)
        if (
            loaded.identity.token_id in token_ids
            or loaded.identity.name in names
            or loaded.digest in digests
        ):
            raise AgentTokenRegistryError("Agent token registry contains duplicate identities")
        token_ids.add(loaded.identity.token_id)
        names.add(loaded.identity.name)
        digests.add(loaded.digest)
        records.append(record)
    return records


def load_token_records(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    """Securely load and validate records for local token management."""
    opened = _open_registry(path, missing_ok=missing_ok)
    if opened is None:
        return []
    descriptor, _ = opened
    try:
        return _parse_token_records(_read_registry_descriptor(descriptor))
    finally:
        os.close(descriptor)


def _loaded_records(records: Iterable[dict[str, Any]]) -> tuple[_TokenRecord, ...]:
    return tuple(_validate_record(record)[1] for record in records)


def write_token_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Validate and atomically replace a token registry with mode 0600."""
    normalized: list[dict[str, Any]] = []
    token_ids: set[str] = set()
    names: set[str] = set()
    digests: set[bytes] = set()
    for raw_record in records:
        record, loaded = _validate_record(raw_record)
        if (
            loaded.identity.token_id in token_ids
            or loaded.identity.name in names
            or loaded.digest in digests
        ):
            raise AgentTokenRegistryError("Agent token registry contains duplicate identities")
        token_ids.add(loaded.identity.token_id)
        names.add(loaded.identity.name)
        digests.add(loaded.digest)
        normalized.append(record)

    path.parent.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump({"tokens": normalized}, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        os.close(directory_fd)


@contextmanager
def agent_token_registry_lock(path: Path) -> Iterator[None]:
    """Hold a private sibling advisory lock for a registry transaction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        created = True
    except FileExistsError:
        try:
            descriptor = os.open(
                lock_path,
                os.O_RDWR
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise AgentTokenRegistryError("Agent token registry lock is unavailable") from exc
    except OSError as exc:
        raise AgentTokenRegistryError("Agent token registry lock is unavailable") from exc

    try:
        if created:
            os.fchmod(descriptor, 0o600)
        try:
            lock_stat = os.fstat(descriptor)
            _validate_private_regular_file(lock_stat)
        except (OSError, AgentTokenRegistryError) as exc:
            raise AgentTokenRegistryError("Agent token registry lock is unavailable") from exc
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


class AgentTokenRegistry:
    """An mtime/inode-aware, fail-closed view of the token registry."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else get_agent_tokens_path()
        self._file_state: _FileSignature | None | object = object()
        self._records: tuple[_TokenRecord, ...] = ()
        self._lock = threading.RLock()

    def _reload_if_changed(self) -> None:
        try:
            opened = _open_registry(self.path, missing_ok=False)
        except (AgentTokenRegistryError, OSError, ValueError, TypeError):
            self._file_state = None
            self._records = ()
            return
        if opened is None:  # pragma: no cover - missing_ok=False
            self._file_state = None
            self._records = ()
            return
        descriptor, state = opened
        if state == self._file_state:
            os.close(descriptor)
            return
        self._file_state = state
        self._records = ()
        try:
            records = _parse_token_records(_read_registry_descriptor(descriptor))
            self._records = _loaded_records(records)
        except (AgentTokenRegistryError, OSError, ValueError, TypeError):
            # Authentication must fail closed; malformed local state is never
            # exposed as a request-time stack trace.
            self._records = ()
        finally:
            os.close(descriptor)

    def authenticate(self, token: str) -> AgentIdentity:
        if not isinstance(token, str) or not token:
            raise AgentAuthError("Authentication failed")
        candidate = hashlib.sha256(token.encode("utf-8")).digest()
        with self._lock:
            self._reload_if_changed()
            match: AgentIdentity | None = None
            for record in self._records:
                if hmac.compare_digest(candidate, record.digest):
                    match = record.identity
        if match is None:
            raise AgentAuthError("Authentication failed")
        return match


    def identity_for_token_id(self, token_id: str) -> AgentIdentity | None:
        """The current identity of a registered token, or None once revoked.

        Background work booked by a token (scheduled sends) re-reads the token
        at execution time, so a revoked token or a narrowed conversation list
        applies to work that was queued before the change.
        """
        if not isinstance(token_id, str) or not token_id:
            return None
        with self._lock:
            self._reload_if_changed()
            for record in self._records:
                if record.identity.token_id == token_id:
                    return record.identity
        return None


_registry_cache: dict[Path, AgentTokenRegistry] = {}
_registry_cache_lock = threading.Lock()


def _configured_registry() -> AgentTokenRegistry:
    path = get_agent_tokens_path()
    with _registry_cache_lock:
        registry = _registry_cache.get(path)
        if registry is None:
            registry = AgentTokenRegistry(path)
            _registry_cache[path] = registry
        return registry


def current_identity(token_id: str) -> AgentIdentity | None:
    """Look up a token's current identity in the configured registry."""
    return _configured_registry().identity_for_token_id(token_id)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Authentication failed",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_scopes(*scopes: str):
    """Return a FastAPI dependency requiring every named scope."""
    unknown = set(scopes) - ALLOWED_SCOPES
    if unknown:
        raise ValueError("Unknown agent scope")
    required = frozenset(scopes)

    def authenticate_request(
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> AgentIdentity:
        if authorization is None:
            raise _unauthorized()
        parts = authorization.split(" ")
        if (
            len(parts) != 2
            or parts[0].lower() != "bearer"
            or not parts[1]
            or any(character.isspace() for character in parts[1])
        ):
            raise _unauthorized()
        try:
            identity = _configured_registry().authenticate(parts[1])
        except AgentAuthError:
            raise _unauthorized() from None
        if not required.issubset(identity.scopes):
            raise HTTPException(status_code=403, detail="Insufficient scope")
        return identity

    return authenticate_request
