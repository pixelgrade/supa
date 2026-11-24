"""Bounded HTTP transport for the standalone Supa CLI."""

from __future__ import annotations

import os
from typing import Any, Iterable

import httpx

from . import __version__
from .config import CLIConfig
from .jsonutil import StrictJSONError, loads_object


API_VERSION = "v1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024


class ClientError(Exception):
    """A transport, protocol, or API-envelope failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        exit_code: int,
        envelope: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code
        self.envelope = envelope


def api_error_exit_code(status: int, code: str) -> int:
    if code in {"task_failed", "run_failed", "run_interrupted"}:
        return 7
    if code == "upstream_auth_error":
        return 6
    if status in {401, 403} or code in {
        "authentication_failed",
        "conversation_not_allowed",
        "insufficient_scope",
    }:
        return 3
    if status == 404 or code in {"not_found", "attachment_not_found", "note_not_found"}:
        return 4
    if status in {409, 412} or code in {
        "conflict",
        "conversation_changed",
        "conversation_locked",
        "draft_conflict",
        "external_write_conflict",
        "idempotency_conflict",
        "note_conflict",
        "note_not_editable",
        "precondition_failed",
        "wrong_expected_thread",
    }:
        return 5
    if status in {400, 405, 422} or code in {"invalid_request", "invalid_cursor"}:
        return 2
    return 6


def _bounded_content_length(response: httpx.Response, maximum: int) -> None:
    raw = response.headers.get("content-length")
    if raw is None:
        return
    try:
        length = int(raw)
    except ValueError as exc:
        raise ClientError(
            "The server returned an invalid Content-Length.",
            code="invalid_response",
            exit_code=6,
        ) from exc
    if length < 0:
        raise ClientError(
            "The server returned an invalid Content-Length.",
            code="invalid_response",
            exit_code=6,
        )
    if length > maximum:
        raise ClientError(
            "The server response exceeds the client limit.",
            code="response_too_large",
            exit_code=6,
        )


def _read_bounded(response: httpx.Response, maximum: int) -> bytes:
    _bounded_content_length(response, maximum)
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > maximum:
            raise ClientError(
                "The server response exceeds the client limit.",
                code="response_too_large",
                exit_code=6,
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_json(data: bytes) -> dict[str, Any]:
    try:
        return loads_object(data)
    except StrictJSONError as exc:
        raise ClientError(
            "The server returned invalid JSON.",
            code="invalid_response",
            exit_code=6,
        ) from exc


def _validate_api_version(value: Any) -> None:
    if value != API_VERSION:
        raise ClientError(
            "The server API version is incompatible; upgrade the Supa CLI.",
            code="incompatible_api",
            exit_code=6,
        )


def _redact_secret(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, dict):
        return {
            _redact_secret(key, secret) if isinstance(key, str) else key: _redact_secret(
                item, secret
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_secret(item, secret) for item in value]
    return value


def _error_from_response(
    response: httpx.Response, data: bytes, *, secret: str
) -> ClientError:
    try:
        envelope = _parse_json(data)
        _validate_api_version(envelope.get("api_version"))
        error = envelope.get("error")
        if not isinstance(error, dict):
            raise ClientError("invalid", code="invalid_response", exit_code=6)
        code = error.get("code")
        message = error.get("message")
        retryable = error.get("retryable")
        details = error.get("details")
        if (
            not isinstance(code, str)
            or not code
            or not isinstance(message, str)
            or not isinstance(retryable, bool)
            or not isinstance(details, dict)
            or not isinstance(envelope.get("request_id"), str)
        ):
            raise ClientError("invalid", code="invalid_response", exit_code=6)
    except ClientError as exc:
        if exc.code == "incompatible_api":
            return exc
        return ClientError(
            "The server returned an invalid error response.",
            code="invalid_response",
            exit_code=6,
        )
    safe_envelope = _redact_secret(envelope, secret)
    safe_message = safe_envelope["error"]["message"]
    return ClientError(
        safe_message,
        code=code,
        exit_code=api_error_exit_code(response.status_code, code),
        envelope=safe_envelope,
    )


class SupaClient:
    """Thin, synchronous client for the remote Agent API."""

    def __init__(self, config: CLIConfig, *, transport: httpx.BaseTransport | None = None) -> None:
        self.config = config
        self._client = httpx.Client(
            base_url=config.url,
            headers={
                "Authorization": f"Bearer {config.token}",
                "Accept": "application/json",
                "User-Agent": f"supa-cli/{__version__}",
                "X-Supa-API-Version": API_VERSION,
            },
            timeout=httpx.Timeout(30.0, connect=10.0, read=30.0, write=30.0, pool=10.0),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SupaClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Iterable[tuple[str, str]] | None = None,
        json: dict[str, Any] | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> dict[str, Any]:
        request_options: dict[str, Any] = {}
        if timeout is not None:
            request_options["timeout"] = timeout
        try:
            with self._client.stream(
                method,
                path,
                params=params,
                json=json,
                **request_options,
            ) as response:
                data = _read_bounded(response, MAX_JSON_BYTES)
        except ClientError:
            raise
        except httpx.HTTPError as exc:
            raise ClientError(
                "Could not reach the Supa service.",
                code="transport_error",
                exit_code=6,
            ) from exc
        if not 200 <= response.status_code < 300:
            raise _error_from_response(response, data, secret=self.config.token)
        envelope = _parse_json(data)
        _validate_api_version(envelope.get("api_version"))
        if (
            not isinstance(envelope.get("request_id"), str)
            or "result" not in envelope
            or "error" in envelope
        ):
            raise ClientError(
                "The server returned an invalid response envelope.",
                code="invalid_response",
                exit_code=6,
            )
        return envelope

    def get_json(
        self,
        path: str,
        *,
        params: Iterable[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        return self._request_json("GET", path, params=params)

    def get_long_poll_json(
        self, path: str, *, params: Iterable[tuple[str, str]], wait_seconds: int
    ) -> dict[str, Any]:
        timeout = httpx.Timeout(
            max(30.0, float(wait_seconds) + 10.0),
            connect=10.0, write=30.0, pool=10.0,
        )
        return self._request_json("GET", path, params=params, timeout=timeout)

    def put_json(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        """Make exactly one idempotent-keyed PUT request; callers own retries."""
        return self._request_json("PUT", path, json=json)

    def post_json(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        """Make exactly one POST request; mutation outcomes are never auto-retried."""
        return self._request_json("POST", path, json=json)

    def post_read_json(
        self,
        path: str,
        *,
        json: dict[str, Any],
        wait_seconds: int,
    ) -> dict[str, Any]:
        """Make one POST-shaped read with enough timeout for a bounded long poll."""
        read_timeout = max(30.0, float(wait_seconds) + 10.0)
        timeout = httpx.Timeout(
            read_timeout,
            connect=10.0,
            write=30.0,
            pool=10.0,
        )
        return self._request_json("POST", path, json=json, timeout=timeout)

    def download_to(
        self,
        path: str,
        *,
        descriptor: int,
        maximum: int = MAX_ATTACHMENT_BYTES,
    ) -> dict[str, Any]:
        try:
            with self._client.stream("GET", path) as response:
                if not 200 <= response.status_code < 300:
                    data = _read_bounded(response, MAX_JSON_BYTES)
                    raise _error_from_response(
                        response, data, secret=self.config.token
                    )
                _bounded_content_length(response, maximum)
                _validate_api_version(response.headers.get("x-supa-api-version"))
                request_id = response.headers.get("x-request-id")
                if not request_id or len(request_id) > 256:
                    raise ClientError(
                        "The attachment response has no valid request ID.",
                        code="invalid_response",
                        exit_code=6,
                    )
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > maximum:
                        raise ClientError(
                            "The attachment exceeds the client download limit.",
                            code="response_too_large",
                            exit_code=6,
                        )
                    view = memoryview(chunk)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise OSError("Attachment write made no progress")
                        view = view[written:]
                return {
                    "api_version": API_VERSION,
                    "request_id": request_id,
                    "bytes": size,
                    "content_type": response.headers.get(
                        "content-type", "application/octet-stream"
                    ).split(";", 1)[0].strip()
                    or "application/octet-stream",
                }
        except ClientError:
            raise
        except httpx.HTTPError as exc:
            raise ClientError(
                "Could not reach the Supa service.",
                code="transport_error",
                exit_code=6,
            ) from exc
