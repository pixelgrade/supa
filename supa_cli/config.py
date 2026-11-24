"""Private local configuration for the standalone Supa CLI."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import secrets
import stat
import tomllib
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit

import httpx


_CONFIG_MAX_BYTES = 64 * 1024
TOKEN_MAX_BYTES = 16 * 1024
TOKEN_LENGTH = 43
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_LOCK_NAME = ".lock"
_PENDING_NAME = ".pending.json"


class ConfigError(Exception):
    """A local setting is absent, unsafe, or malformed."""

    def __init__(self, message: str, *, code: str, exit_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


@dataclass(frozen=True)
class CLIConfig:
    url: str
    token: str


def _configuration_error(message: str, *, code: str = "invalid_configuration") -> ConfigError:
    return ConfigError(message, code=code, exit_code=2)


def _token_error(message: str, *, code: str = "authentication_required") -> ConfigError:
    return ConfigError(message, code=code, exit_code=3)


def _config_dir(environ: Mapping[str, str]) -> Path:
    home = environ.get("HOME")
    if not isinstance(home, str) or not home.strip():
        raise _configuration_error("HOME is required to locate Supa configuration.")
    return Path(home).expanduser() / ".config" / "supa"


def _private_file_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


@contextmanager
def _closing_descriptor(descriptor: int):
    """Close on every path without replacing an exception already in flight."""
    try:
        yield descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)


def _validate_private_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise _configuration_error("Supa configuration directory does not exist.") from None
    except OSError as exc:
        raise _configuration_error(
            "Supa configuration directory is unavailable."
        ) from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise _configuration_error(
            "Supa configuration directory must be private (mode 0700).",
            code="insecure_configuration",
        )


def _read_private_file(
    path: Path,
    *,
    max_bytes: int,
    token_file: bool,
) -> str | None:
    try:
        _validate_private_directory(path.parent)
    except ConfigError:
        if not path.parent.exists():
            return None
        raise

    try:
        descriptor = os.open(path, _private_file_flags())
    except FileNotFoundError:
        return None
    except OSError as exc:
        code = "insecure_token" if token_file else "insecure_configuration"
        error = _token_error if token_file else _configuration_error
        raise error("Supa configuration file could not be opened.", code=code) from exc

    try:
        with _closing_descriptor(descriptor):
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                code = "insecure_token" if token_file else "insecure_configuration"
                error = _token_error if token_file else _configuration_error
                raise error("Supa configuration file must be private (mode 0600).", code=code)
            if info.st_size > max_bytes:
                error = _token_error if token_file else _configuration_error
                raise error("Supa configuration file is too large.")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    error = _token_error if token_file else _configuration_error
                    raise error("Supa configuration file is too large.")
            try:
                return b"".join(chunks).decode("utf-8")
            except UnicodeDecodeError as exc:
                error = _token_error if token_file else _configuration_error
                raise error("Supa configuration file is not valid UTF-8.") from exc
    except OSError as exc:
        code = "insecure_token" if token_file else "insecure_configuration"
        error = _token_error if token_file else _configuration_error
        raise error("Supa configuration file could not be read.", code=code) from exc


def normalize_url(value: str) -> str:
    """Validate a base origin and return it without a trailing slash."""
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise _configuration_error("SUPA_URL must be a valid URL.")
    if any(ord(character) < 0x20 or character.isspace() for character in value):
        raise _configuration_error("SUPA_URL must be a valid URL.")
    try:
        parts = urlsplit(value)
        port = parts.port
        hostname = parts.hostname
    except ValueError as exc:
        raise _configuration_error("SUPA_URL must be a valid URL.") from exc
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not hostname:
        raise _configuration_error("SUPA_URL must use HTTP or HTTPS.")
    if parts.username is not None or parts.password is not None:
        raise _configuration_error("SUPA_URL must not contain credentials.")
    if parts.query or parts.fragment or parts.path not in {"", "/"}:
        raise _configuration_error("SUPA_URL must contain only an origin.")

    normalized_host = hostname.rstrip(".").lower()
    if not normalized_host:
        raise _configuration_error("SUPA_URL must have a hostname.")
    if scheme == "http":
        loopback = normalized_host == "localhost"
        if not loopback:
            try:
                loopback = ipaddress.ip_address(normalized_host).is_loopback
            except ValueError:
                loopback = False
        if not loopback:
            raise _configuration_error("Remote Supa URLs must use HTTPS.")

    try:
        ip = ipaddress.ip_address(normalized_host)
    except ValueError:
        host_for_url = normalized_host
    else:
        host_for_url = f"[{normalized_host}]" if ip.version == 6 else normalized_host
    netloc = host_for_url if port is None else f"{host_for_url}:{port}"
    normalized = urlunsplit((scheme, netloc, "", "", ""))
    try:
        httpx.URL(normalized)
    except (httpx.InvalidURL, UnicodeError) as exc:
        raise _configuration_error("SUPA_URL must be a valid URL.") from exc
    return normalized


def normalize_token(value: str) -> str:
    if not isinstance(value, str):
        raise _token_error("A Supa agent token is required.")
    if (
        not value
        or len(value) != TOKEN_LENGTH
        or _TOKEN_RE.fullmatch(value) is None
    ):
        raise _token_error("A valid Supa agent token is required.")
    return value


def _strip_one_line_ending(value: str) -> str:
    if value.endswith("\r\n"):
        return value[:-2]
    if value.endswith("\n"):
        return value[:-1]
    return value


def _load_file_url(path: Path) -> str | None:
    raw = _read_private_file(path, max_bytes=_CONFIG_MAX_BYTES, token_file=False)
    if raw is None:
        return None
    try:
        payload = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as exc:
        raise _configuration_error("Supa config.toml is invalid.") from exc
    if not isinstance(payload, dict) or set(payload) != {"url"} or not isinstance(payload["url"], str):
        raise _configuration_error("Supa config.toml must contain only a URL.")
    return payload["url"]


def load_config(environ: Mapping[str, str]) -> CLIConfig:
    """Load each setting from the environment, falling back to private files."""
    need_url = "SUPA_URL" not in environ
    need_token = "SUPA_TOKEN" not in environ
    file_url: str | None = None
    file_token: str | None = None
    if need_url or need_token:
        file_url, file_token = _load_file_settings(
            environ,
            need_url=need_url,
            need_token=need_token,
        )
    raw_url = file_url if need_url else environ["SUPA_URL"]
    if raw_url is None:
        raise _configuration_error("SUPA_URL is not configured.")

    raw_token = file_token if need_token else environ["SUPA_TOKEN"]
    if raw_token is None:
        raise _token_error("SUPA_TOKEN is not configured.")
    return CLIConfig(url=normalize_url(raw_url), token=normalize_token(raw_token))


def _ensure_private_config_dir(environ: Mapping[str, str]) -> Path:
    directory = _config_dir(environ)
    try:
        directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
    except OSError as exc:
        raise _configuration_error(
            "Could not create the Supa configuration directory."
        ) from exc
    _validate_private_directory(directory)
    return directory


def _validate_existing_target(path: Path, *, token_file: bool) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        error = _token_error if token_file else _configuration_error
        code = "insecure_token" if token_file else "insecure_configuration"
        raise error("Refusing to replace an unsafe configuration file.", code=code)


def _validate_private_regular_file(
    info: os.stat_result,
    *,
    token_file: bool,
    message: str,
) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        error = _token_error if token_file else _configuration_error
        code = "insecure_token" if token_file else "insecure_configuration"
        raise error(message, code=code)


@contextmanager
def _configuration_lock(directory: Path):
    """Serialize readers and writers through a private advisory lock."""
    _validate_private_directory(directory)
    path = directory / _LOCK_NAME
    create_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    create_flags |= getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        try:
            descriptor = os.open(path, create_flags, 0o600)
            created = True
        except FileExistsError:
            descriptor = os.open(
                path,
                os.O_RDWR
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
    except OSError as exc:
        raise _configuration_error(
            "Supa configuration lock is unavailable.",
            code="insecure_configuration",
        ) from exc

    try:
        with _closing_descriptor(descriptor):
            if created:
                os.fchmod(descriptor, 0o600)
            _validate_private_regular_file(
                os.fstat(descriptor),
                token_file=False,
                message="Supa configuration lock must be private (mode 0600).",
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            except BaseException:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
                raise
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError as exc:
        raise _configuration_error(
            "Supa configuration lock is unavailable.",
            code="insecure_configuration",
        ) from exc


def _atomic_private_write(path: Path, data: bytes, *, token_file: bool) -> None:
    _validate_existing_target(path, token_file=token_file)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with _closing_descriptor(descriptor):
            os.fchmod(descriptor, 0o600)
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("Configuration write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        with _closing_descriptor(directory_fd):
            os.fsync(directory_fd)
    except OSError as exc:
        error = _token_error if token_file else _configuration_error
        code = "insecure_token" if token_file else "insecure_configuration"
        raise error("Could not write Supa configuration securely.", code=code) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _config_bytes(url: str) -> bytes:
    return f"url = {json.dumps(url, ensure_ascii=True)}\n".encode("utf-8")


def _token_bytes(token: str) -> bytes:
    return f"{token}\n".encode("utf-8")


def _pending_bytes(url: str, token: str) -> bytes:
    return (
        json.dumps(
            {"token": token, "url": url},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _apply_config_pair(directory: Path, *, url: str, token: str) -> None:
    _atomic_private_write(
        directory / "config.toml",
        _config_bytes(url),
        token_file=False,
    )
    _atomic_private_write(directory / "token", _token_bytes(token), token_file=True)


def _clear_pending(directory: Path) -> None:
    pending = directory / _PENDING_NAME
    _validate_existing_target(pending, token_file=True)
    try:
        pending.unlink()
        directory_fd = os.open(
            directory,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        with _closing_descriptor(directory_fd):
            os.fsync(directory_fd)
    except OSError as exc:
        raise _token_error(
            "Could not complete the Supa configuration transaction.",
            code="insecure_token",
        ) from exc


def _recover_pending(directory: Path) -> None:
    raw = _read_private_file(
        directory / _PENDING_NAME,
        max_bytes=_CONFIG_MAX_BYTES,
        token_file=True,
    )
    if raw is None:
        return
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise _token_error(
            "Supa pending configuration is invalid.",
            code="insecure_token",
        ) from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"token", "url"}
        or not isinstance(payload["token"], str)
        or not isinstance(payload["url"], str)
    ):
        raise _token_error(
            "Supa pending configuration is invalid.",
            code="insecure_token",
        )
    url = normalize_url(payload["url"])
    token = normalize_token(payload["token"])
    _apply_config_pair(directory, url=url, token=token)
    _clear_pending(directory)


def _load_file_settings(
    environ: Mapping[str, str],
    *,
    need_url: bool,
    need_token: bool,
) -> tuple[str | None, str | None]:
    directory = _config_dir(environ)
    try:
        _validate_private_directory(directory)
    except ConfigError:
        if not directory.exists():
            return None, None
        raise
    with _configuration_lock(directory):
        _recover_pending(directory)
        raw_url = _load_file_url(directory / "config.toml") if need_url else None
        raw_token = (
            _read_private_file(
                directory / "token",
                max_bytes=TOKEN_MAX_BYTES,
                token_file=True,
            )
            if need_token
            else None
        )
        if raw_token is not None:
            raw_token = _strip_one_line_ending(raw_token)
        return raw_url, raw_token


def write_config(environ: Mapping[str, str], *, url: str, token: str) -> str:
    """Validate and atomically persist the configured origin and raw token."""
    normalized_url = normalize_url(url)
    normalized_token = normalize_token(token)
    directory = _ensure_private_config_dir(environ)
    with _configuration_lock(directory):
        _recover_pending(directory)
        _atomic_private_write(
            directory / _PENDING_NAME,
            _pending_bytes(normalized_url, normalized_token),
            token_file=True,
        )
        _apply_config_pair(
            directory,
            url=normalized_url,
            token=normalized_token,
        )
        _clear_pending(directory)
    return normalized_url
