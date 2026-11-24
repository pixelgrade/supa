from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import threading
import tomllib
import zipfile
from uuid import UUID, uuid4
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

import supa_cli
import supa_cli.client as cli_client
import supa_cli.config as cli_config
import supa_cli.main as cli_main
from supa_cli.main import main


TOKEN = "A" * 43
ENV_TOKEN = "B" * 43
CONFIGURED_TOKEN = "C" * 43
BASE_URL = "https://supa.example.test"
INVALID_AGENT_TOKENS = [
    "",
    "café",
    "bad\udcfftoken",
    "padded-token=",
    "token.with.punctuation",
    " leading-space",
    "trailing-space ",
    "token with space",
    "token\tcontrol",
    "token\x00control",
    "token\x7fcontrol",
    "A" * 42,
    "A" * 44,
]
COMMAND_SNAPSHOTS = {
    "0.2.1": [
        "attachment.download",
        "capabilities",
        "conversation.create",
        "conversation.discard",
        "conversation.get",
        "conversation.list",
        "conversation.search",
        "draft.get",
        "draft.save",
        "draft.schedule",
        "draft.schedule.cancel",
        "draft.schedule.get",
        "draft.send",
        "events.ack",
        "events.next",
        "events.reset",
        "health",
        "note.add",
        "note.update",
        "operation.get",
        "reply.send",
        "run.get",
        "run.watch",
        "ticket.evaluate",
        "ticket.process",
    ],
}
COMMAND_SNAPSHOTS["0.2.2"] = COMMAND_SNAPSHOTS["0.2.1"] + ["ticket.recover"]
COMMAND_SNAPSHOTS["0.2.3"] = sorted(COMMAND_SNAPSHOTS["0.2.2"] + ["mailbox.operations"])
COMMAND_SNAPSHOTS["0.2.4"] = sorted(COMMAND_SNAPSHOTS["0.2.3"] + ["conversation.state"])
COMMAND_SNAPSHOTS["0.2.5"] = sorted(COMMAND_SNAPSHOTS["0.2.4"] + ["request.add", "request.done"])

COMMAND_SNAPSHOTS["0.2.6"] = sorted(COMMAND_SNAPSHOTS["0.2.5"] + [
    "desk", "conversation.brief", "conversation.precedents", "mailbox.insights"
])


def _success(result: object, request_id: str = "req_test") -> httpx.Response:
    return httpx.Response(
        200,
        json={"api_version": "v1", "request_id": request_id, "result": result},
    )


def _error(
    status: int,
    code: str,
    *,
    retryable: bool = False,
    details: dict | None = None,
) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "api_version": "v1",
            "request_id": "req_error",
            "error": {
                "code": code,
                "message": "Synthetic failure",
                "retryable": retryable,
                "details": details or {},
            },
        },
    )


def _run(
    argv: list[str],
    handler,
    *,
    environ: dict[str, str] | None = None,
    stdin: io.StringIO | None = None,
):
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        argv,
        transport=httpx.MockTransport(handler),
        environ=environ
        or {"SUPA_URL": BASE_URL, "SUPA_TOKEN": TOKEN},
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def _event_result(
    event_id: int,
    *,
    ack_token: str = "D" * 43,
    run_id: str = "run-event-test",
) -> dict[str, object]:
    return {
        "event": {
            "event_id": event_id,
            "type": "supa.run.completed",
            "conversation_id": None,
            "thread_id": None,
            "run_id": run_id,
            "source": "supa",
            "actor_type": "system",
            "occurred_at": "2026-09-25T12:00:00.000000Z",
            "received_at": "2026-09-25T12:00:00.000000Z",
            "summary": {"status": "succeeded"},
            "ack_token": ack_token,
            "lease_expires_at": "2026-09-25T12:01:00.000000Z",
        }
    }


def test_health_uses_agent_route_required_headers_and_compact_json():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == f"{BASE_URL}/agent/v1/health"
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        assert request.headers["accept"] == "application/json"
        assert request.headers["user-agent"].startswith("supa-cli/")
        assert request.headers["x-supa-api-version"] == "v1"
        return _success({"status": "ok", "server_version": "0.1.0"})

    code, stdout, stderr = _run(["health"], handler)

    assert code == 0
    assert stderr == ""
    assert stdout == (
        '{"api_version":"v1","request_id":"req_test",'
        '"result":{"status":"ok","server_version":"0.1.0"}}\n'
    )


def test_ticket_commands_use_fixed_names_scopes_and_idempotency():
    seen = []
    run_id = str(uuid4())
    def handler(request):
        seen.append(json.loads(request.content))
        assert request.url.path == "/agent/v1/tasks"
        return _success({"run_id": run_id, "status": "queued", "delivery_ceiling": "draft"})
    code, stdout, stderr = _run(["ticket", "evaluate", "321", "--idempotency-key", "eval-321"], handler)
    assert code == 0 and stderr == ""
    assert seen[-1] == {"task_name": "ticket.evaluate", "arguments": {"conversation_id": 321}, "idempotency_key": "eval-321"}
    code, stdout, stderr = _run(["ticket", "process", "321", "--idempotency-key", "process-321"], handler)
    assert code == 0
    assert seen[-1]["arguments"] == {"conversation_id": 321, "allow_send": False}
    code, stdout, stderr = _run(["ticket", "process", "321", "--allow-send", "--idempotency-key", "send-321"], handler)
    assert code == 0
    assert seen[-1]["arguments"]["allow_send"] is True
    code, stdout, stderr = _run(["ticket", "recover", "321", "--expected-thread-id", "1",
        "--expected-draft-version", "a"*64, "--expected-conversation-version", "b"*64,
        "--idempotency-key", "recover-321"], handler)
    assert code == 0 and stderr == ""
    assert seen[-1]["task_name"] == "ticket.recover"
    assert seen[-1]["arguments"] == {"conversation_id":321, "expected_thread_id":"1",
        "expected_draft_version":"a"*64, "expected_conversation_version":"b"*64}
    code, _stdout, _stderr = _run(["ticket", "arbitrary", "321"], handler)
    assert code == 2


def test_run_get_and_watch_incremental_terminal_result():
    run_id = str(uuid4())
    calls = []
    def handler(request):
        calls.append((request.url.path, request.url.query))
        if request.url.path.endswith("/events"):
            after = int(request.url.params["after"])
            if after == 0:
                return _success({"run": {"run_id": run_id, "status": "running"}, "events": [{"run_id": run_id, "sequence": 1, "stage": "fetch"}]})
            return _success({"run": {"run_id": run_id, "status": "completed", "result": {"posted": False}}, "events": [{"run_id": run_id, "sequence": 2, "stage": "completed"}]})
        return _success({"run_id": run_id, "status": "completed"})
    code, stdout, stderr = _run(["run", "get", run_id], handler)
    assert code == 0 and stderr == ""
    code, stdout, stderr = _run(["run", "watch", run_id], handler)
    assert code == 0 and stderr == ""
    records = [json.loads(line)["result"] for line in stdout.splitlines()]
    assert [record.get("sequence") for record in records] == [1, 2, None]
    assert records[-1]["status"] == "completed"
    assert [int(query.split(b"after=")[1].split(b"&")[0]) for path, query in calls if path.endswith("/events")] == [0, 1]


def test_run_watch_failed_returns_task_failure_exit_code():
    run_id = str(uuid4())
    def handler(_request):
        return _success({"run": {"run_id": run_id, "status": "failed", "error": {"code": "processing_failed"}}, "events": []})
    code, stdout, stderr = _run(["run", "watch", run_id], handler)
    assert code == 7
    assert json.loads(stdout)["result"]["status"] == "failed"
    assert json.loads(stderr)["error"]["code"] == "run_failed"


def test_http_client_disables_environment_proxy_and_tls_overrides(monkeypatch):
    captured: dict[str, object] = {}

    class RecordingClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(cli_client.httpx, "Client", RecordingClient)

    with cli_client.SupaClient(
        cli_config.CLIConfig(url="http://localhost:8100", token=TOKEN)
    ):
        pass

    assert captured["trust_env"] is False


def test_build_archives_contain_only_cli_and_safe_packaging_metadata(tmp_path):
    build_dir = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "--directory", str(build_dir)],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )

    [sdist] = build_dir.glob("*.tar.gz")
    [wheel] = build_dir.glob("*.whl")
    with tarfile.open(sdist) as archive:
        sdist_paths = {
            "/".join(Path(name).parts[1:]) for name in archive.getnames()
        }
    with zipfile.ZipFile(wheel) as archive:
        wheel_paths = set(archive.namelist())

    allowed_sdist_roots = {
        ".gitignore",
        "PKG-INFO",
        "LICENSE",
        "NOTICE",
        "README.md",
        "pyproject.toml",
        "supa_cli",
    }
    assert all(path.split("/", 1)[0] in allowed_sdist_roots for path in sdist_paths)
    assert ".env" not in sdist_paths
    assert not any(path.startswith(("api/", "services/", "knowledge/", "infra/")) for path in sdist_paths)
    assert all(
        path.startswith("supa_cli/") or ".dist-info/" in path
        for path in wheel_paths
    )


def test_environment_url_and_token_override_protected_files(tmp_path):
    config_dir = tmp_path / ".config" / "supa"
    config_dir.mkdir(parents=True, mode=0o700)
    config_dir.chmod(0o700)
    (config_dir / "config.toml").write_text('url = "https://file.invalid"\n')
    (config_dir / "token").write_text("file-token\n")
    (config_dir / "config.toml").chmod(0o600)
    (config_dir / "token").chmod(0o600)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "env.example.test"
        assert request.headers["authorization"] == f"Bearer {ENV_TOKEN}"
        return _success({"status": "ok"})

    code, _, stderr = _run(
        ["health"],
        handler,
        environ={
            "HOME": str(tmp_path),
            "SUPA_URL": "https://env.example.test/",
            "SUPA_TOKEN": ENV_TOKEN,
        },
    )

    assert code == 0
    assert stderr == ""


@pytest.mark.parametrize("token", INVALID_AGENT_TOKENS)
def test_invalid_environment_token_is_rejected_before_request(token):
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _success({})

    code, stdout, stderr = _run(
        ["health"],
        handler,
        environ={"SUPA_URL": BASE_URL, "SUPA_TOKEN": token},
    )

    assert code == 3
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "authentication_required"
    assert called is False


def test_agent_token_uses_canonical_unpadded_32_byte_base64url_shape():
    assert cli_config.normalize_token("A" * 43) == "A" * 43
    for token in ("A" * 42, "A" * 44, "A" * 42 + "="):
        with pytest.raises(cli_config.ConfigError):
            cli_config.normalize_token(token)


def test_cli_version_fallback_reads_project_metadata(tmp_path, monkeypatch):
    metadata_path = tmp_path / "pyproject.toml"
    metadata_path.write_text('[project]\nversion = "9.8.7"\n', encoding="utf-8")

    def package_not_installed(_name):
        raise supa_cli.PackageNotFoundError("supa")

    monkeypatch.setattr(supa_cli, "_distribution_version", package_not_installed)
    monkeypatch.setattr(supa_cli, "PROJECT_METADATA_PATH", metadata_path)

    assert supa_cli._resolve_version() == "9.8.7"


def test_cli_version_prefers_installed_distribution_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(supa_cli, "_distribution_version", lambda _name: "7.6.5")
    monkeypatch.setattr(
        supa_cli,
        "PROJECT_METADATA_PATH",
        tmp_path / "missing.toml",
    )

    assert supa_cli._resolve_version() == "7.6.5"


def test_cli_implemented_commands_match_version_snapshot():
    with (Path(__file__).resolve().parent.parent / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]

    assert version in COMMAND_SNAPSHOTS, "Record the command snapshot for the new version."
    assert cli_main._implemented_commands({"SUPA_EXTENSIONS": ""}) == COMMAND_SNAPSHOTS[version], (
        "Client commands changed: bump version in pyproject.toml and record its command snapshot."
    )


def test_protected_config_and_token_files_are_used(tmp_path):
    config_dir = tmp_path / ".config" / "supa"
    config_dir.mkdir(parents=True, mode=0o700)
    config_dir.chmod(0o700)
    config_file = config_dir / "config.toml"
    token_file = config_dir / "token"
    config_file.write_text('url = "https://configured.example.test"\n')
    token_file.write_text(f"{CONFIGURED_TOKEN}\n")
    config_file.chmod(0o600)
    token_file.chmod(0o600)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "configured.example.test"
        assert request.headers["authorization"] == f"Bearer {CONFIGURED_TOKEN}"
        return _success({"status": "ok"})

    code, _, stderr = _run(
        ["health"], handler, environ={"HOME": str(tmp_path)}
    )

    assert code == 0
    assert stderr == ""


@pytest.mark.parametrize(
    "token",
    ["", "café", "padded-token=", "token.with.punctuation", "token with space"],
)
def test_invalid_protected_file_token_is_rejected_before_request(tmp_path, token):
    config_dir = tmp_path / ".config" / "supa"
    config_dir.mkdir(parents=True, mode=0o700)
    config_dir.chmod(0o700)
    config_file = config_dir / "config.toml"
    token_file = config_dir / "token"
    config_file.write_text(f'url = "{BASE_URL}"\n')
    token_file.write_text(f"{token}\n")
    config_file.chmod(0o600)
    token_file.chmod(0o600)
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _success({})

    code, stdout, stderr = _run(
        ["health"], handler, environ={"HOME": str(tmp_path)}
    )

    assert code == 3
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "authentication_required"
    assert called is False


@pytest.mark.parametrize(
    "url",
    [
        "http://supa.example.test",
        "ftp://127.0.0.1:8100",
        "https://user:password@supa.example.test",
        "https://supa.example.test/agent/v1",
        "https://supa.example.test?token=secret",
        "https://supa.example.test#fragment",
    ],
)
def test_unsafe_or_ambiguous_base_urls_are_rejected_without_a_request(url):
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _success({})

    code, stdout, stderr = _run(
        ["health"],
        handler,
        environ={"SUPA_URL": url, "SUPA_TOKEN": TOKEN},
    )

    assert code == 2
    assert stdout == ""
    error = json.loads(stderr)
    assert error["error"]["code"] == "invalid_configuration"
    assert TOKEN not in stderr
    assert called is False


def test_http_client_incompatible_hostname_is_stable_configuration_error():
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _success({})

    code, stdout, stderr = _run(
        ["health"],
        handler,
        environ={"SUPA_URL": "https://­.example", "SUPA_TOKEN": TOKEN},
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "invalid_configuration"
    assert TOKEN not in stderr
    assert called is False


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:8100", "http://[::1]:8100", "http://localhost:8100"],
)
def test_plain_http_is_allowed_only_for_loopback(url):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/agent/v1/health"
        return _success({"status": "ok"})

    code, _, stderr = _run(
        ["health"], handler, environ={"SUPA_URL": url, "SUPA_TOKEN": TOKEN}
    )

    assert code == 0
    assert stderr == ""


def test_insecure_config_file_mode_is_rejected(tmp_path):
    config_dir = tmp_path / ".config" / "supa"
    config_dir.mkdir(parents=True, mode=0o700)
    config_dir.chmod(0o700)
    config_file = config_dir / "config.toml"
    config_file.write_text('url = "https://configured.example.test"\n')
    config_file.chmod(0o644)

    code, stdout, stderr = _run(
        ["health"],
        lambda _request: _success({}),
        environ={"HOME": str(tmp_path), "SUPA_TOKEN": TOKEN},
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "insecure_configuration"


def test_insecure_token_file_mode_is_local_auth_failure(tmp_path):
    config_dir = tmp_path / ".config" / "supa"
    config_dir.mkdir(parents=True, mode=0o700)
    config_dir.chmod(0o700)
    token_file = config_dir / "token"
    token_file.write_text(f"{CONFIGURED_TOKEN}\n")
    token_file.chmod(0o644)

    code, stdout, stderr = _run(
        ["health"],
        lambda _request: _success({}),
        environ={"HOME": str(tmp_path), "SUPA_URL": BASE_URL},
    )

    assert code == 3
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "insecure_token"


def test_no_command_line_token_option_exists_and_secret_is_not_printed():
    code, stdout, stderr = _run(
        ["--token", "do-not-print-me", "health"],
        lambda _request: _success({}),
    )

    assert code == 2
    assert stdout == ""
    assert "do-not-print-me" not in stderr
    assert json.loads(stderr)["error"]["code"] == "usage_error"


def test_usage_error_includes_argparse_reason_without_request():
    code, stdout, stderr = _run(
        ["draft", "get", "zero"],
        lambda _request: pytest.fail("usage errors must not make requests"),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["message"] == (
        "Invalid command usage: argument conversation_id: must be an integer"
    )
    assert "supa capabilities" not in stderr


@pytest.mark.parametrize("argv", [["unknown"], ["note", "unknown"]])
def test_unknown_subcommand_includes_stale_client_hint_without_request(argv):
    code, stdout, stderr = _run(
        argv,
        lambda _request: pytest.fail("unknown commands must not make requests"),
    )

    assert code == 2
    assert stdout == ""
    message = json.loads(stderr)["error"]["message"]
    assert "invalid choice: 'unknown'" in message
    assert "your supa client may be older than the server" in message
    assert "run `supa capabilities`" in message


@pytest.mark.parametrize(
    "argv",
    [[TOKEN], ["note", CONFIGURED_TOKEN], ["health", "--token=do-not-print-me"]],
)
def test_usage_error_redacts_tokens_in_argparse_reason(argv):
    code, stdout, stderr = _run(
        argv,
        lambda _request: pytest.fail("usage errors must not make requests"),
    )

    assert code == 2
    assert stdout == ""
    assert TOKEN not in stderr
    assert CONFIGURED_TOKEN not in stderr
    assert "do-not-print-me" not in stderr
    assert "[REDACTED]" in json.loads(stderr)["error"]["message"]


@pytest.mark.parametrize(
    "argv",
    [
        ["configure", "--url", BASE_URL, "--url", "https://other.example.test", "--token-stdin"],
        ["conversation", "list", "--status", "active", "--status", "pending"],
        ["conversation", "list", "--limit", "10", "--limit", "20"],
        [
            "attachment",
            "download",
            "321",
            "701",
            "--output",
            "first.bin",
            "--output",
            "second.bin",
        ],
        [
            "draft",
            "save",
            "321",
            "--body-file",
            "first.txt",
            "--body-file",
            "second.txt",
            "--body-format",
            "text",
        ],
        [
            "draft",
            "send",
            "321",
            "--expected-version",
            "a" * 64,
            "--expected-thread",
            "40",
            "--idempotency-key",
            "first-key",
            "--idempotency-key",
            "second-key",
        ],
    ],
)
def test_duplicate_singleton_options_are_usage_errors(tmp_path, argv):
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _success({})

    code, stdout, stderr = _run(
        argv,
        handler,
        environ={"HOME": str(tmp_path), "SUPA_URL": BASE_URL, "SUPA_TOKEN": TOKEN},
        stdin=io.StringIO(f"{CONFIGURED_TOKEN}\n"),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "usage_error"
    assert called is False
    assert not (tmp_path / ".config" / "supa").exists()


def test_configure_reads_non_tty_stdin_and_writes_private_files(tmp_path):
    code, stdout, stderr = _run(
        ["configure", "--url", BASE_URL, "--token-stdin"],
        lambda _request: pytest.fail("configure must not make a request"),
        environ={"HOME": str(tmp_path)},
        stdin=io.StringIO(f"{CONFIGURED_TOKEN}\n"),
    )

    assert code == 0
    assert stderr == ""
    result = json.loads(stdout)["result"]
    assert result == {"configured": True, "url": BASE_URL}
    config_dir = tmp_path / ".config" / "supa"
    assert stat.S_IMODE(config_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((config_dir / "config.toml").stat().st_mode) == 0o600
    assert stat.S_IMODE((config_dir / "token").stat().st_mode) == 0o600
    assert (config_dir / "token").read_text() == f"{CONFIGURED_TOKEN}\n"
    assert CONFIGURED_TOKEN not in stdout


@pytest.mark.parametrize("token", INVALID_AGENT_TOKENS)
def test_configure_rejects_invalid_token_without_persisting(tmp_path, token):
    code, stdout, stderr = _run(
        ["configure", "--url", BASE_URL, "--token-stdin"],
        lambda _request: pytest.fail("configure must not make a request"),
        environ={"HOME": str(tmp_path)},
        stdin=io.StringIO(f"{token}\n"),
    )

    assert code == 3
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "authentication_required"
    assert not (tmp_path / ".config" / "supa").exists()


def test_configure_bounds_non_tty_token_input_before_validation(tmp_path):
    class BoundedTokenInput(io.StringIO):
        def read(self, size: int = -1) -> str:
            assert size == 16 * 1024 + 1
            return "x" * size

    code, stdout, stderr = _run(
        ["configure", "--url", BASE_URL, "--token-stdin"],
        lambda _request: pytest.fail("configure must not make a request"),
        environ={"HOME": str(tmp_path)},
        stdin=BoundedTokenInput(),
    )

    assert code == 3
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "authentication_required"
    assert not (tmp_path / ".config" / "supa").exists()


def test_configure_filesystem_failure_is_a_stable_config_error(tmp_path, monkeypatch):
    original_mkdir = cli_config.Path.mkdir

    def fail_supa_mkdir(path: Path, *args, **kwargs):
        if path.name == "supa":
            raise PermissionError("synthetic secret path detail")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(cli_config.Path, "mkdir", fail_supa_mkdir)
    code, stdout, stderr = _run(
        ["configure", "--url", BASE_URL, "--token-stdin"],
        lambda _request: pytest.fail("configure must not make a request"),
        environ={"HOME": str(tmp_path)},
        stdin=io.StringIO(f"{CONFIGURED_TOKEN}\n"),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "invalid_configuration"
    assert "synthetic secret path detail" not in stderr
    assert CONFIGURED_TOKEN not in stderr


def test_configure_zero_progress_write_fails_once_and_cleans_temp(tmp_path, monkeypatch):
    write_calls = 0

    def zero_then_fail(_descriptor: int, _data) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return 0
        raise OSError("write loop repeated after zero progress")

    monkeypatch.setattr(cli_config.os, "write", zero_then_fail)

    code, stdout, stderr = _run(
        ["configure", "--url", BASE_URL, "--token-stdin"],
        lambda _request: pytest.fail("configure must not make a request"),
        environ={"HOME": str(tmp_path)},
        stdin=io.StringIO(f"{CONFIGURED_TOKEN}\n"),
    )

    assert write_calls == 1
    assert code == 3
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "insecure_token"
    assert not list((tmp_path / ".config" / "supa").glob("*.tmp"))


def test_configure_close_failure_is_stable_and_cleans_temp(tmp_path, monkeypatch):
    original_open = cli_config.os.open
    original_close = cli_config.os.close
    temporary_descriptors: set[int] = set()

    def recording_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if Path(path).name.endswith(".tmp"):
            temporary_descriptors.add(descriptor)
        return descriptor

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        if descriptor in temporary_descriptors:
            temporary_descriptors.remove(descriptor)
            raise OSError("synthetic close failure")

    monkeypatch.setattr(cli_config.os, "open", recording_open)
    monkeypatch.setattr(cli_config.os, "close", close_then_fail)

    code, stdout, stderr = _run(
        ["configure", "--url", BASE_URL, "--token-stdin"],
        lambda _request: pytest.fail("configure must not make a request"),
        environ={"HOME": str(tmp_path)},
        stdin=io.StringIO(f"{CONFIGURED_TOKEN}\n"),
    )

    assert code == 3
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "insecure_token"
    assert "synthetic close failure" not in stderr
    assert not list((tmp_path / ".config" / "supa").glob("*.tmp"))


def test_configure_cleanup_close_failure_does_not_mask_write_error(
    tmp_path, monkeypatch
):
    original_open = cli_config.os.open
    original_close = cli_config.os.close
    temporary_descriptors: set[int] = set()

    def recording_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if Path(path).name.endswith(".tmp"):
            temporary_descriptors.add(descriptor)
        return descriptor

    def fail_write(_descriptor: int, _data) -> int:
        raise OSError("synthetic primary write failure")

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        if descriptor in temporary_descriptors:
            temporary_descriptors.remove(descriptor)
            raise OSError("synthetic cleanup close failure")

    monkeypatch.setattr(cli_config.os, "open", recording_open)
    monkeypatch.setattr(cli_config.os, "write", fail_write)
    monkeypatch.setattr(cli_config.os, "close", close_then_fail)

    code, stdout, stderr = _run(
        ["configure", "--url", BASE_URL, "--token-stdin"],
        lambda _request: pytest.fail("configure must not make a request"),
        environ={"HOME": str(tmp_path)},
        stdin=io.StringIO(f"{CONFIGURED_TOKEN}\n"),
    )

    assert code == 3
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "insecure_token"
    assert "synthetic cleanup close failure" not in stderr
    assert not list((tmp_path / ".config" / "supa").glob("*.tmp"))


def test_configure_second_write_failure_rolls_forward_for_next_reader(
    tmp_path, monkeypatch
):
    environ = {"HOME": str(tmp_path)}
    old_url = "https://old.example.test"
    new_url = "https://new.example.test"
    old_token = "D" * 43
    new_token = "E" * 43
    cli_config.write_config(environ, url=old_url, token=old_token)
    original_write = cli_config._atomic_private_write
    failed = False

    def fail_first_token_write(path, data, *, token_file):
        nonlocal failed
        if path.name == "token" and not failed:
            failed = True
            raise cli_config.ConfigError(
                "synthetic write failure",
                code="insecure_token",
                exit_code=3,
            )
        return original_write(path, data, token_file=token_file)

    monkeypatch.setattr(cli_config, "_atomic_private_write", fail_first_token_write)

    with pytest.raises(cli_config.ConfigError):
        cli_config.write_config(environ, url=new_url, token=new_token)

    pending = tmp_path / ".config" / "supa" / ".pending.json"
    assert pending.exists()
    recovered = cli_config.load_config(environ)
    assert recovered == cli_config.CLIConfig(url=new_url, token=new_token)
    assert not pending.exists()


def test_concurrent_configure_writers_leave_one_complete_pair(tmp_path, monkeypatch):
    environ = {"HOME": str(tmp_path)}
    first = ("https://first.example.test", "F" * 43)
    second = ("https://second.example.test", "G" * 43)
    original_write = cli_config._atomic_private_write
    first_config_written = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    failures: list[BaseException] = []

    def delay_first_writer(path, data, *, token_file):
        result = original_write(path, data, token_file=token_file)
        if path.name == "config.toml" and b"first.example.test" in data:
            first_config_written.set()
            assert release_first.wait(timeout=2)
        return result

    monkeypatch.setattr(cli_config, "_atomic_private_write", delay_first_writer)

    def configure(pair, finished=None):
        try:
            cli_config.write_config(environ, url=pair[0], token=pair[1])
        except BaseException as exc:
            failures.append(exc)
        finally:
            if finished is not None:
                finished.set()

    first_thread = threading.Thread(target=configure, args=(first,))
    second_thread = threading.Thread(target=configure, args=(second, second_finished))
    first_thread.start()
    assert first_config_written.wait(timeout=2)
    second_thread.start()
    second_finished.wait(timeout=0.1)
    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert failures == []
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    configured = cli_config.load_config(environ)
    assert (configured.url, configured.token) in {first, second}


@pytest.mark.parametrize("unsafe_name", [".lock", ".pending.json"])
def test_config_reader_rejects_symlink_transaction_files(tmp_path, unsafe_name):
    environ = {"HOME": str(tmp_path)}
    cli_config.write_config(environ, url=BASE_URL, token=TOKEN)
    config_dir = tmp_path / ".config" / "supa"
    transaction_path = config_dir / unsafe_name
    if transaction_path.exists():
        transaction_path.unlink()
    target = tmp_path / "outside"
    target.write_text("private")
    target.chmod(0o600)
    transaction_path.symlink_to(target)

    with pytest.raises(cli_config.ConfigError):
        cli_config.load_config(environ)


@pytest.mark.parametrize("unsafe_name", [".lock", ".pending.json"])
def test_config_reader_rejects_non_private_transaction_files(tmp_path, unsafe_name):
    environ = {"HOME": str(tmp_path)}
    cli_config.write_config(environ, url=BASE_URL, token=TOKEN)
    config_dir = tmp_path / ".config" / "supa"
    transaction_path = config_dir / unsafe_name
    if unsafe_name == ".pending.json":
        transaction_path.write_text(
            json.dumps({"token": TOKEN, "url": BASE_URL}),
            encoding="utf-8",
        )
    transaction_path.chmod(0o644)

    with pytest.raises(cli_config.ConfigError):
        cli_config.load_config(environ)


def test_capabilities_text_format_is_a_human_view_while_errors_stay_json():
    code, stdout, stderr = _run(
        ["--format", "text", "capabilities"],
        lambda _request: _success({"scopes": ["read"], "tasks": []}),
    )

    assert code == 0
    assert stderr == ""
    assert '"scopes": [' in stdout
    assert stdout.startswith("{")
    assert "request_id" not in stdout

    code, stdout, stderr = _run(
        ["--format", "text", "health"],
        lambda _request: _error(403, "insufficient_scope"),
    )
    assert code == 3
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "insufficient_scope"


def test_capabilities_json_adds_the_local_cli_version():
    code, stdout, stderr = _run(
        ["capabilities"],
        lambda _request: _success(
            {"api_version": "v1", "server_version": "0.1.0", "scopes": ["read"]}
        ),
    )

    assert code == 0
    assert stderr == ""
    assert json.loads(stdout)["result"]["cli_version"] == supa_cli.__version__
    assert json.loads(stdout)["result"]["client_missing"] == []


@pytest.mark.parametrize("output_format", ["json", "text"])
@pytest.mark.parametrize("missing", [[], ["note.future", "ticket.future"]])
def test_capabilities_detects_client_drift_and_warns_only_when_missing(output_format, missing):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path == "/agent/v1/capabilities"
        return _success({
            "commands": {"available": COMMAND_SNAPSHOTS["0.2.1"] + missing[::-1]},
            "tasks": [{"name": "ticket.evaluate", "required_scopes": ["read", "task"]}],
        })

    code, stdout, stderr = _run(["capabilities", "--format", output_format], handler)

    assert code == 0
    assert len(requests) == 1
    result = json.loads(stdout)
    if output_format == "json":
        result = result["result"]
    assert result["client_missing"] == missing
    assert result["cli_version"] == supa_cli.__version__
    if missing:
        assert len(stderr.splitlines()) == 1
        assert "uv tool install --force --reinstall --no-cache ." in stderr
        assert "Supa checkout" in stderr
        assert "reinstall" in stderr and "uv tool install" in stderr
    else:
        assert stderr == ""


@pytest.mark.parametrize(
    ("argv", "expected_path", "expected_query"),
    [
        (["conversation", "get", "321"], "/agent/v1/conversations/321", {}),
        (
            [
                "conversation",
                "list",
                "--status",
                "pending",
                "--tag",
                "field-journal",
                "--limit",
                "37",
                "--cursor",
                "opaque-cursor",
            ],
            "/agent/v1/conversations",
            {
                "status": ["pending"],
                "tag": ["field-journal"],
                "limit": ["37"],
                "cursor": ["opaque-cursor"],
            },
        ),
        (
            [
                "conversation",
                "search",
                "header spacing",
                "--status",
                "active",
                "--tag",
                "field-journal",
                "--cursor",
                "next-page",
                "--limit",
                "12",
            ],
            "/agent/v1/conversations",
            {
                "query": ["header spacing"],
                "status": ["active"],
                "tag": ["field-journal"],
                "limit": ["12"],
                "cursor": ["next-page"],
            },
        ),
        (
            ["draft", "get", "321"],
            "/agent/v1/conversations/321/draft",
            {},
        ),
    ],
)
def test_read_commands_match_agent_routes(argv, expected_path, expected_query):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == expected_path
        assert parse_qs(request.url.query.decode()) == expected_query
        return _success({"synthetic": True})

    code, stdout, stderr = _run(argv, handler)

    assert code == 0
    assert json.loads(stdout)["result"] == {"synthetic": True}
    assert stderr == ""


@pytest.mark.parametrize("send,body_format", [(False, "text"), (True, "html")])
def test_conversation_create_defaults_to_draft_and_requires_explicit_send(tmp_path, send, body_format):
    body_path = tmp_path / "invitation.txt"
    body_path.write_text("Hello café", encoding="utf-8")

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/agent/v1/conversations"
        assert json.loads(request.content) == {
            "subject": "A private review",
            "mailbox_id": 12,
            "customer_id": 34,
            "body": "Hello café",
            "body_format": body_format,
            "draft": not send,
            "idempotency_key": "create-review-v1",
        }
        return _success({"status": "created", "conversation_id": 321})

    argv = ["conversation", "create", "--subject", "A private review", "--mailbox", "12",
            "--customer", "34", "--body-file", str(body_path),
            "--idempotency-key", "create-review-v1"]
    if send:
        argv += ["--send", "--body-format", "html"]
    code, stdout, stderr = _run(argv, handler)
    assert code == 0
    assert stderr == ""
    assert json.loads(stdout)["result"]["idempotency_key"] == "create-review-v1"


def test_conversation_create_generated_key_and_stdin_body():
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload["idempotency_key"])
        assert payload["body"] == "Private preview"
        assert payload["draft"] is True
        return _success({"status": "created", "conversation_id": 321})

    code, stdout, stderr = _run(
        ["conversation", "create", "--subject", "Review", "--mailbox", "12",
         "--customer", "34", "--body-file", "-"],
        handler, stdin=io.StringIO("Private preview"),
    )
    assert code == 0
    assert stderr == ""
    assert seen[0]
    assert json.loads(stdout)["result"]["idempotency_key"] == seen[0]


def test_conversation_discard_forwards_exact_id_and_version():
    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/agent/v1/conversations/321/discard"
        assert json.loads(request.content) == {
            "expected_version": "a" * 64,
            "idempotency_key": "discard-321-v1",
        }
        return _success({"status": "discarded", "conversation_id": 321})

    code, stdout, stderr = _run(
        ["conversation", "discard", "321", "--expected-version", "a" * 64,
         "--idempotency-key", "discard-321-v1"], handler,
    )
    assert code == 0
    assert stderr == ""
    assert json.loads(stdout)["result"]["idempotency_key"] == "discard-321-v1"


@pytest.mark.parametrize("argv", [
    ["conversation", "discard", "321"],
    ["conversation", "discard", "0", "--expected-version", "a" * 64],
    ["conversation", "discard", "321", "--expected-version", "stale"],
    ["conversation", "delete", "321"],
    ["conversation", "create", "--subject", "Review", "--mailbox", "12", "--customer", "34"],
    ["conversation", "create", "--subject", "Review", "--subject", "Other", "--mailbox", "12",
     "--customer", "34", "--body-file", "-"],
    ["conversation", "create", "--subject", "Review", "--mailbox", "0",
     "--customer", "34", "--body-file", "-"],
])
def test_conversation_mutation_usage_errors_never_request(argv):
    def handler(request):
        pytest.fail("Invalid mutation must not contact the server")

    code, stdout, stderr = _run(argv, handler)
    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "usage_error"


@pytest.mark.parametrize("action,status,error_code,exit_code", [
    ("create", 403, "conversation_not_allowed", 3),
    ("create", 403, "insufficient_scope", 3),
    ("discard", 409, "conversation_changed", 5),
    ("discard", 409, "conversation_not_draft", 5),
])
def test_conversation_mutation_rejections_are_not_retried(action, status, error_code, exit_code):
    seen = []

    def handler(request):
        seen.append(request)
        return _error(status, error_code)

    argv = (["conversation", "create", "--subject", "Review", "--mailbox", "12",
             "--customer", "34", "--body-file", "-"] if action == "create" else
            ["conversation", "discard", "321", "--expected-version", "a" * 64])
    code, stdout, stderr = _run(argv, handler, stdin=io.StringIO("Preview"))
    assert code == exit_code
    assert stdout == ""
    assert len(seen) == 1
    assert json.loads(stderr)["error"]["code"] == error_code


def test_draft_save_reads_utf8_file_and_prints_generated_idempotency_key(
    tmp_path, monkeypatch
):
    body_path = tmp_path / "draft.html"
    body_path.write_bytes("<p>Bună!</p>".encode("utf-8"))
    generated_key = str(UUID(int=72, version=4))
    monkeypatch.setattr(cli_main, "uuid4", lambda: generated_key, raising=False)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "PUT"
        assert request.url.path == "/agent/v1/conversations/321/draft"
        assert json.loads(request.content) == {
            "body": "<p>Bună!</p>",
            "body_format": "html",
            "expected_version": None,
            "idempotency_key": generated_key,
        }
        return _success(
            {
                "action": "created",
                "body_sha256": "a" * 64,
                "conversation_id": 321,
                "thread_id": 701,
                "version": "b" * 64,
            }
        )

    code, stdout, stderr = _run(
        [
            "draft",
            "save",
            "321",
            "--body-file",
            str(body_path),
            "--body-format",
            "html",
        ],
        handler,
    )

    assert code == 0
    assert stderr == ""
    assert len(requests) == 1
    result = json.loads(stdout)["result"]
    assert result["action"] == "created"
    assert result["idempotency_key"] == generated_key


def test_draft_save_preserves_caller_key_and_expected_version(tmp_path):
    body_path = tmp_path / "draft.txt"
    body_path.write_text("Hello\n\nWorld", encoding="utf-8")
    caller_key = "caller-draft-save-1"
    version = "c" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert json.loads(request.content) == {
            "body": "Hello\n\nWorld",
            "body_format": "text",
            "expected_version": version,
            "idempotency_key": caller_key,
        }
        return _success(
            {
                "action": "updated",
                "body_sha256": "d" * 64,
                "conversation_id": 321,
                "thread_id": 701,
                "version": "e" * 64,
            }
        )

    code, stdout, stderr = _run(
        [
            "draft",
            "save",
            "321",
            "--body-file",
            str(body_path),
            "--body-format",
            "text",
            "--expected-version",
            version,
            "--idempotency-key",
            caller_key,
        ],
        handler,
    )

    assert code == 0
    assert stderr == ""
    assert json.loads(stdout)["result"]["idempotency_key"] == caller_key


def test_draft_send_requires_and_forwards_exact_preconditions(monkeypatch):
    generated_key = str(UUID(int=72, version=4))
    monkeypatch.setattr(cli_main, "uuid4", lambda: generated_key, raising=False)
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        assert request.method == "POST"
        assert request.url.path == "/agent/v1/conversations/321/draft/send"
        assert json.loads(request.content) == {
            "expected_version": "a" * 64,
            "expected_latest_thread_id": 40,
            "idempotency_key": generated_key,
        }
        return _success(
            {
                "action": "sent",
                "body_sha256": "b" * 64,
                "conversation_id": 321,
                "status": "sent",
                "thread_id": 990,
                "warnings": [],
            }
        )

    code, stdout, stderr = _run(
        [
            "draft",
            "send",
            "321",
            "--expected-version",
            "a" * 64,
            "--expected-thread",
            "40",
        ],
        handler,
    )

    assert code == 0
    assert stderr == ""
    assert requests == 1
    assert json.loads(stdout)["result"]["idempotency_key"] == generated_key


def test_reply_send_reads_stdin_once_and_makes_one_request(monkeypatch):
    generated_key = str(UUID(int=72, version=4))
    monkeypatch.setattr(cli_main, "uuid4", lambda: generated_key, raising=False)

    class CountingStdin(io.StringIO):
        reads = 0

        def read(self, *args, **kwargs):
            self.reads += 1
            return super().read(*args, **kwargs)

    stdin = CountingStdin("Hello from stdin")
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        assert request.method == "POST"
        assert request.url.path == "/agent/v1/conversations/321/replies"
        assert json.loads(request.content) == {
            "body": "Hello from stdin",
            "body_format": "text",
            "expected_latest_thread_id": 40,
            "idempotency_key": generated_key,
        }
        return _success(
            {
                "action": "sent",
                "body_sha256": "b" * 64,
                "conversation_id": 321,
                "status": "sent_with_warning",
                "thread_id": 991,
                "warnings": ["tag_update_failed"],
            }
        )

    code, stdout, stderr = _run(
        [
            "reply",
            "send",
            "321",
            "--body-file",
            "-",
            "--body-format",
            "text",
            "--expected-thread",
            "40",
        ],
        handler,
        stdin=stdin,
    )

    assert code == 0
    assert stderr == ""
    assert stdin.reads == 1
    assert requests == 1
    assert json.loads(stdout)["result"]["idempotency_key"] == generated_key


def test_operation_get_calls_owner_scoped_status_route():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/agent/v1/operations/draft-save-1"
        return _success(
            {
                "idempotency_key": "draft-save-1",
                "operation_name": "draft.save",
                "state": "succeeded",
                "target_id": "321",
            }
        )

    code, stdout, stderr = _run(
        ["operation", "get", "draft-save-1"],
        handler,
    )

    assert code == 0
    assert stderr == ""
    assert json.loads(stdout)["result"]["state"] == "succeeded"


def test_events_next_forwards_first_start_and_wait_then_acks_after_flush():
    requests: list[tuple[str, str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append((request.method, request.url.path, payload))
        if request.url.path == "/agent/v1/events/next":
            assert request.extensions["timeout"]["read"] >= 65.0
            return _success(_event_result(71))
        return _success({"acknowledged": True, "event_id": 71, "replayed": False})

    code, stdout, stderr = _run(
        [
            "events",
            "next",
            "--consumer",
            "example-reader",
            "--start",
            "71",
            "--wait",
            "55",
        ],
        handler,
    )

    assert code == 0
    assert stderr == ""
    assert stdout == json.dumps(
        {
            "api_version": "v1",
            "request_id": "req_test",
            "result": _event_result(71),
        },
        separators=(",", ":"),
    ) + "\n"
    assert requests == [
        (
            "POST",
            "/agent/v1/events/next",
            {"consumer": "example-reader", "start": 71, "wait": 55},
        ),
        (
            "POST",
            "/agent/v1/events/71/ack",
            {"consumer": "example-reader", "ack_token": "D" * 43},
        ),
    ]


@pytest.mark.parametrize(
    ("result", "extra_args"),
    [
        ({"event": None}, []),
        (_event_result(72), ["--manual-ack"]),
    ],
)
def test_events_next_no_event_and_manual_ack_make_only_one_request(
    result, extra_args
):
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        assert request.url.path == "/agent/v1/events/next"
        return _success(result)

    code, stdout, stderr = _run(
        ["events", "next", "--consumer", "manual-agent", *extra_args],
        handler,
    )

    assert code == 0
    assert stderr == ""
    assert requests == 1
    assert json.loads(stdout)["result"] == result


def test_events_next_does_not_ack_when_stdout_flush_fails():
    requests: list[str] = []

    class BrokenStdout(io.StringIO):
        def flush(self):
            raise BrokenPipeError("private broken pipe")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return _success(_event_result(73))

    stdout = BrokenStdout()
    stderr = io.StringIO()
    code = main(
        ["events", "next", "--consumer", "pipe-agent"],
        transport=httpx.MockTransport(handler),
        environ={"SUPA_URL": BASE_URL, "SUPA_TOKEN": TOKEN},
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 6
    assert requests == ["/agent/v1/events/next"]
    assert json.loads(stderr.getvalue())["error"] == {
        "code": "output_error",
        "message": "The event could not be written; it was not acknowledged.",
        "retryable": False,
        "details": {},
    }


def test_events_next_does_not_ack_after_short_stdout_write():
    requests: list[str] = []

    class ShortStdout(io.StringIO):
        def write(self, value):
            super().write(value[:-1])
            return len(value) - 1

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/agent/v1/events/next":
            return _success(_event_result(73))
        return _success({"acknowledged": True, "event_id": 73, "replayed": False})

    stderr = io.StringIO()
    code = main(
        ["events", "next", "--consumer", "short-write-agent"],
        transport=httpx.MockTransport(handler),
        environ={"SUPA_URL": BASE_URL, "SUPA_TOKEN": TOKEN},
        stdout=ShortStdout(),
        stderr=stderr,
    )

    assert code == 6
    assert requests == ["/agent/v1/events/next"]
    assert json.loads(stderr.getvalue())["error"] == {
        "code": "output_error",
        "message": "The event could not be written; it was not acknowledged.",
        "retryable": False,
        "details": {},
    }


def test_events_ack_and_reset_forward_exact_explicit_mutations():
    requests: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        if request.url.path.endswith("/ack"):
            return _success({"acknowledged": True, "event_id": 74, "replayed": True})
        return _success(
            {
                "acknowledged_event_id": 74,
                "consumer_name": "explicit-agent",
                "start_mode": "event_id",
            }
        )

    ack = _run(
        [
            "events",
            "ack",
            "--consumer",
            "explicit-agent",
            "74",
            "--ack-token",
            "E" * 43,
        ],
        handler,
    )
    reset = _run(
        [
            "events",
            "reset",
            "--consumer",
            "explicit-agent",
            "--to",
            "75",
        ],
        handler,
    )

    assert ack[0] == reset[0] == 0
    assert json.loads(ack[1])["result"]["replayed"] is True
    assert json.loads(reset[1])["result"]["start_mode"] == "event_id"
    assert ack[2] == reset[2] == ""
    assert requests == [
        (
            "/agent/v1/events/74/ack",
            {"consumer": "explicit-agent", "ack_token": "E" * 43},
        ),
        (
            "/agent/v1/events/reset",
            {"consumer": "explicit-agent", "to": 75},
        ),
    ]


def test_events_explicit_ack_output_failure_is_stable_and_safe_to_repeat():
    requests: list[str] = []

    class BrokenStdout(io.StringIO):
        def flush(self):
            raise BrokenPipeError("private broken pipe")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return _success({"acknowledged": True, "event_id": 74, "replayed": False})

    stderr = io.StringIO()
    code = main(
        [
            "events",
            "ack",
            "--consumer",
            "explicit-agent",
            "74",
            "--ack-token",
            "E" * 43,
        ],
        transport=httpx.MockTransport(handler),
        environ={"SUPA_URL": BASE_URL, "SUPA_TOKEN": TOKEN},
        stdout=BrokenStdout(),
        stderr=stderr,
    )

    assert code == 6
    assert requests == ["/agent/v1/events/74/ack"]
    error = json.loads(stderr.getvalue())["error"]
    assert error == {
        "code": "output_error",
        "message": (
            "The acknowledgement completed, but its result could not be written. "
            "Repeating the same acknowledgement is safe."
        ),
        "retryable": False,
        "details": {"event_id": 74},
    }
    assert "E" * 43 not in stderr.getvalue()


def test_events_reset_output_failure_is_stable_and_warns_against_retry():
    requests: list[str] = []

    class BrokenStdout(io.StringIO):
        def flush(self):
            raise BrokenPipeError("private broken pipe")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return _success(
            {
                "acknowledged_event_id": 74,
                "consumer_name": "explicit-agent",
                "start_mode": "now",
            }
        )

    stderr = io.StringIO()
    code = main(
        ["events", "reset", "--consumer", "explicit-agent", "--to", "now"],
        transport=httpx.MockTransport(handler),
        environ={"SUPA_URL": BASE_URL, "SUPA_TOKEN": TOKEN},
        stdout=BrokenStdout(),
        stderr=stderr,
    )

    assert code == 6
    assert requests == ["/agent/v1/events/reset"]
    assert json.loads(stderr.getvalue())["error"] == {
        "code": "output_error",
        "message": (
            "The reset completed, but its result could not be written. "
            "Do not repeat it automatically because doing so may move the cursor again."
        ),
        "retryable": False,
        "details": {"consumer": "explicit-agent"},
    }


def test_events_watch_reconnects_retryable_read_with_backoff_and_sigint_is_success(
    monkeypatch,
):
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("synthetic disconnect")
        if calls == 2:
            assert json.loads(request.content) == {
                "consumer": "watch-agent",
                "start": "now",
                "wait": 55,
            }
            return _success(_event_result(80, ack_token="F" * 43))
        if calls == 3:
            return _success(
                {"acknowledged": True, "event_id": 80, "replayed": False}
            )
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_main, "_watch_sleep", sleeps.append, raising=False)
    code, stdout, stderr = _run(
        ["events", "watch", "--consumer", "watch-agent"],
        handler,
    )

    assert code == 0
    assert stderr == ""
    assert sleeps == [1.0]
    records = [json.loads(line) for line in stdout.splitlines()]
    assert [record["result"]["event"]["event_id"] for record in records] == [80]


def test_events_watch_manual_ack_does_not_suppress_duplicate_event_ids():
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls <= 2:
            return _success(_event_result(81, ack_token="G" * 43))
        raise KeyboardInterrupt

    code, stdout, stderr = _run(
        ["events", "watch", "--consumer", "duplicate-agent", "--manual-ack"],
        handler,
    )

    assert code == 0
    assert stderr == ""
    records = [json.loads(line) for line in stdout.splitlines()]
    assert [record["result"]["event"]["event_id"] for record in records] == [81, 81]
    assert calls == 3


def test_events_cursor_expired_includes_explicit_reset_guidance():
    code, stdout, stderr = _run(
        ["events", "next", "--consumer", "stale-agent"],
        lambda _request: _error(
            409,
            "cursor_expired",
            details={"oldest_retained_event_id": 12, "current_head_event_id": 20},
        ),
    )

    assert code == 5
    assert stdout == ""
    error = json.loads(stderr)["error"]
    assert error["code"] == "cursor_expired"
    assert error["details"]["event_reset_command"] == (
        "supa events reset --consumer stale-agent --to now"
    )
    assert "reset explicitly" in error["message"].lower()


def test_events_reject_malformed_event_enum_without_acknowledging():
    requests = 0
    malformed = _event_result(82)
    malformed["event"]["source"] = ["supa"]

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return _success(malformed)

    code, stdout, stderr = _run(
        ["events", "next", "--consumer", "strict-agent"],
        handler,
    )

    assert code == 6
    assert stdout == ""
    assert requests == 1
    assert json.loads(stderr)["error"]["code"] == "invalid_response"


def test_events_reject_boolean_ack_event_id_without_retrying_ack():
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if request.url.path == "/agent/v1/events/next":
            return _success(_event_result(1))
        return _success({"acknowledged": True, "event_id": True, "replayed": False})

    code, stdout, stderr = _run(
        ["events", "next", "--consumer", "strict-ack-agent"],
        handler,
    )

    assert code == 6
    assert requests == 2
    assert len(stdout.splitlines()) == 1
    assert json.loads(stderr)["error"]["code"] == "invalid_response"


def test_events_reject_malformed_reset_enum_as_invalid_response():
    code, stdout, stderr = _run(
        ["events", "reset", "--consumer", "strict-reset-agent", "--to", "now"],
        lambda _request: _success(
            {
                "acknowledged_event_id": 0,
                "consumer_name": "strict-reset-agent",
                "start_mode": ["now"],
            }
        ),
    )

    assert code == 6
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "invalid_response"


@pytest.mark.parametrize(
    "argv",
    [
        ["events", "next"],
        ["events", "next", "--consumer", "agent", "--wait", "56"],
        ["events", "next", "--consumer", "agent", "--start", "0"],
        ["events", "ack", "--consumer", "agent", "1", "--ack-token", "short"],
        ["events", "reset", "--consumer", "agent", "--to", "future"],
    ],
)
def test_events_reject_invalid_or_incomplete_usage_before_request(argv):
    code, stdout, stderr = _run(
        argv,
        lambda _request: pytest.fail("invalid event usage must not make a request"),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "usage_error"


@pytest.mark.parametrize(
    "argv",
    [
        ["draft", "save", "321", "--body-file", "message.txt"],
        [
            "draft",
            "save",
            "321",
            "--body-file",
            "-",
            "--body-format",
            "text",
            "--body",
            "inline text is forbidden",
        ],
        [
            "draft",
            "save",
            "321",
            "--body-format",
            "text",
        ],
        ["draft", "send", "321", "--expected-thread", "40"],
        ["draft", "send", "321", "--expected-version", "a" * 64],
        [
            "reply",
            "send",
            "321",
            "--body-file",
            "message.txt",
            "--expected-thread",
            "40",
        ],
        [
            "reply",
            "send",
            "321",
            "--body-file",
            "message.txt",
            "--body-format",
            "text",
        ],
    ],
)
def test_write_commands_reject_missing_preconditions_or_inline_body(argv):
    code, stdout, stderr = _run(
        argv,
        lambda _request: pytest.fail("invalid usage must not make a request"),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "usage_error"


def test_body_file_must_be_strict_utf8_and_is_rejected_before_request(tmp_path):
    body_path = tmp_path / "invalid.txt"
    body_path.write_bytes(b"hello\xffworld")

    code, stdout, stderr = _run(
        [
            "reply",
            "send",
            "321",
            "--body-file",
            str(body_path),
            "--body-format",
            "text",
            "--expected-thread",
            "40",
        ],
        lambda _request: pytest.fail("invalid UTF-8 must not make a request"),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "invalid_body_file"
    assert "hello" not in stderr


def test_body_file_is_read_exactly_once(tmp_path, monkeypatch):
    body_path = tmp_path / "one-read.txt"
    body_path.write_text("Read me once", encoding="utf-8")
    original_open = Path.open
    read_calls = 0

    class CountingBodyFile:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.wrapped.close()

        def read(self, *args, **kwargs):
            nonlocal read_calls
            read_calls += 1
            return self.wrapped.read(*args, **kwargs)

    def counting_open(path, *args, **kwargs):
        opened = original_open(path, *args, **kwargs)
        if path == body_path:
            return CountingBodyFile(opened)
        return opened

    monkeypatch.setattr(Path, "open", counting_open)

    code, _, stderr = _run(
        [
            "reply",
            "send",
            "321",
            "--body-file",
            str(body_path),
            "--body-format",
            "text",
            "--expected-thread",
            "40",
            "--idempotency-key",
            "body-read-once",
        ],
        lambda _request: _success({"sent": True}),
    )

    assert code == 0
    assert stderr == ""
    assert read_calls == 1


def test_oversized_body_is_rejected_before_request(tmp_path):
    body_path = tmp_path / "oversized.txt"
    body_path.write_bytes(b"x" * (1024 * 1024 + 1))

    code, stdout, stderr = _run(
        [
            "reply",
            "send",
            "321",
            "--body-file",
            str(body_path),
            "--body-format",
            "text",
            "--expected-thread",
            "40",
        ],
        lambda _request: pytest.fail("oversized body must not make a request"),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "invalid_body_file"


@pytest.mark.parametrize(
    "argv",
    [
        [
            "draft",
            "send",
            "321",
            "--expected-version",
            "A" * 64,
            "--expected-thread",
            "40",
        ],
        [
            "draft",
            "send",
            "321",
            "--expected-version",
            "a" * 63,
            "--expected-thread",
            "40",
        ],
        [
            "reply",
            "send",
            "321",
            "--body-file",
            "-",
            "--body-format",
            "text",
            "--expected-thread",
            "0",
        ],
    ],
)
def test_write_preconditions_are_strictly_validated_before_request(argv):
    code, stdout, stderr = _run(
        argv,
        lambda _request: pytest.fail("invalid precondition must not make a request"),
        stdin=io.StringIO("body"),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "usage_error"


@pytest.mark.parametrize(
    "key",
    [
        "contains/slash",
        "contains space",
        "prefix_" + "A" * 43 + "_suffix",
        "x" * 201,
        "café",
    ],
)
def test_idempotency_keys_must_round_trip_as_safe_path_segments(key):
    code, stdout, stderr = _run(
        ["operation", "get", key],
        lambda _request: pytest.fail("invalid key must not make a request"),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "usage_error"


def test_server_unknown_outcome_preserves_generated_key_for_lookup(monkeypatch):
    generated_key = str(UUID(int=72, version=4))
    monkeypatch.setattr(cli_main, "uuid4", lambda: generated_key)

    code, stdout, stderr = _run(
        [
            "draft",
            "send",
            "321",
            "--expected-version",
            "a" * 64,
            "--expected-thread",
            "40",
        ],
        lambda _request: _error(409, "outcome_unknown"),
    )

    assert code == 5
    assert stdout == ""
    error = json.loads(stderr)["error"]
    assert error["details"]["idempotency_key"] == generated_key
    assert error["details"]["operation_get_command"] == (
        f"supa operation get {generated_key}"
    )


def test_invalid_write_response_preserves_key_and_forbids_blind_retry(monkeypatch):
    generated_key = str(UUID(int=72, version=4))
    monkeypatch.setattr(cli_main, "uuid4", lambda: generated_key)
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, content=b"not-json")

    code, stdout, stderr = _run(
        [
            "draft",
            "send",
            "321",
            "--expected-version",
            "a" * 64,
            "--expected-thread",
            "40",
        ],
        handler,
    )

    assert requests == 1
    assert code == 6
    assert stdout == ""
    error = json.loads(stderr)["error"]
    assert error["code"] == "invalid_response"
    assert error["retryable"] is False
    assert error["details"]["idempotency_key"] == generated_key
    assert error["details"]["operation_get_command"] == (
        f"supa operation get {generated_key}"
    )


@pytest.mark.parametrize(
    "argv",
    [
        [
            "draft",
            "send",
            "321",
            "--expected-version",
            "a" * 64,
            "--expected-thread",
            "40",
        ],
        [
            "reply",
            "send",
            "321",
            "--body-file",
            "-",
            "--body-format",
            "text",
            "--expected-thread",
            "40",
        ],
    ],
)
def test_send_transport_uncertainty_never_retries_and_prints_recovery_command(
    argv, monkeypatch
):
    generated_key = str(UUID(int=72, version=4))
    monkeypatch.setattr(cli_main, "uuid4", lambda: generated_key, raising=False)
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        raise httpx.ReadError("private network failure")

    code, stdout, stderr = _run(
        argv,
        handler,
        stdin=io.StringIO("reply body"),
    )

    assert requests == 1
    assert code == 6
    assert stdout == ""
    error = json.loads(stderr)["error"]
    assert error["code"] == "transport_error"
    assert error["retryable"] is False
    assert error["details"]["idempotency_key"] == generated_key
    assert error["details"]["operation_get_command"] == (
        f"supa operation get {generated_key}"
    )
    assert generated_key in error["message"]
    assert "private network failure" not in stderr


@pytest.mark.parametrize(
    ("failure_point", "caller_key"),
    [
        ("write", None),
        ("flush", "caller-output-key"),
    ],
)
def test_write_output_failure_prints_exact_recovery_key_without_retry(
    failure_point,
    caller_key,
    monkeypatch,
):
    generated_key = str(UUID(int=72, version=4))
    monkeypatch.setattr(cli_main, "uuid4", lambda: generated_key)
    requests = 0

    class FailingStdout(io.StringIO):
        def write(self, value):
            if failure_point == "write":
                raise OSError("private stdout write failure")
            return super().write(value)

        def flush(self):
            if failure_point == "flush":
                raise OSError("private stdout flush failure")
            return super().flush()

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return _success(
            {
                "action": "sent",
                "body_sha256": "a" * 64,
                "conversation_id": 321,
                "status": "sent",
                "thread_id": 990,
                "warnings": [],
            }
        )

    argv = [
        "reply",
        "send",
        "321",
        "--body-file",
        "-",
        "--body-format",
        "text",
        "--expected-thread",
        "40",
    ]
    if caller_key is not None:
        argv.extend(["--idempotency-key", caller_key])
    stdout = FailingStdout()
    stderr = io.StringIO()

    code = main(
        argv,
        transport=httpx.MockTransport(handler),
        environ={"SUPA_URL": BASE_URL, "SUPA_TOKEN": TOKEN},
        stdin=io.StringIO("private reply body"),
        stdout=stdout,
        stderr=stderr,
    )

    expected_key = caller_key or generated_key
    assert requests == 1
    assert code == 6
    error = json.loads(stderr.getvalue())["error"]
    assert error["code"] == "output_error"
    assert error["retryable"] is False
    assert error["details"] == {
        "idempotency_key": expected_key,
        "operation_get_command": f"supa operation get {expected_key}",
    }
    assert expected_key in error["message"]
    assert "private reply body" not in stderr.getvalue()
    assert "private stdout" not in stderr.getvalue()


@pytest.mark.parametrize(
    ("status", "code_name", "expected_exit"),
    [
        (403, "insufficient_scope", 3),
        (404, "operation_not_found", 4),
        (409, "draft_conflict", 5),
        (409, "idempotency_conflict", 5),
        (412, "conversation_locked", 5),
        (429, "upstream_rate_limited", 6),
        (503, "conversation_busy", 6),
    ],
)
def test_write_and_operation_errors_keep_stable_exit_codes(
    status, code_name, expected_exit
):
    code, stdout, stderr = _run(
        ["operation", "get", "draft-save-1"],
        lambda _request: _error(status, code_name, retryable=status >= 429),
    )

    assert code == expected_exit
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == code_name


@pytest.mark.parametrize("option", ["--tag", "--cursor"])
def test_list_rejects_duplicate_scalar_filters(option):
    code, stdout, stderr = _run(
        ["conversation", "list", option, "first", option, "second"],
        lambda _request: pytest.fail("invalid usage must not make a request"),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "usage_error"


@pytest.mark.parametrize(
    ("status", "code_name", "expected_exit"),
    [
        (400, "invalid_request", 2),
        (401, "authentication_failed", 3),
        (403, "insufficient_scope", 3),
        (404, "not_found", 4),
        (409, "draft_conflict", 5),
        (412, "conversation_locked", 5),
        (429, "rate_limited", 6),
        (502, "upstream_auth_error", 6),
        (403, "upstream_auth_error", 6),
        (503, "service_unavailable", 6),
        (500, "task_failed", 7),
    ],
)
def test_api_errors_have_stable_exit_codes(status, code_name, expected_exit):
    code, stdout, stderr = _run(
        ["health"], lambda _request: _error(status, code_name)
    )

    assert code == expected_exit
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == code_name


def test_transport_failure_is_service_exit_and_does_not_expose_token():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed with {request.headers['authorization']}")

    code, stdout, stderr = _run(["health"], handler)

    assert code == 6
    assert stdout == ""
    assert TOKEN not in stderr
    assert json.loads(stderr)["error"]["code"] == "transport_error"


def test_file_token_is_redacted_from_untrusted_api_error(tmp_path):
    config_dir = tmp_path / ".config" / "supa"
    config_dir.mkdir(parents=True, mode=0o700)
    config_dir.chmod(0o700)
    (config_dir / "config.toml").write_text(f'url = "{BASE_URL}"\n')
    (config_dir / "token").write_text(f"{TOKEN}\n")
    (config_dir / "config.toml").chmod(0o600)
    (config_dir / "token").chmod(0o600)

    def handler(_request: httpx.Request) -> httpx.Response:
        response = _error(502, "upstream_error", details={"secret": TOKEN})
        response._content = response.content.replace(b"Synthetic failure", TOKEN.encode())
        return response

    code, stdout, stderr = _run(
        ["health"], handler, environ={"HOME": str(tmp_path)}
    )

    assert code == 6
    assert stdout == ""
    assert TOKEN not in stderr
    assert stderr.count("[REDACTED]") == 2


@pytest.mark.parametrize("redactor", [cli_client._redact_secret, cli_main._redact])
def test_credential_redactors_cover_dictionary_keys(redactor):
    assert redactor({TOKEN: {f"Bearer {TOKEN}": "safe"}}, TOKEN) == {
        "[REDACTED]": {"Bearer [REDACTED]": "safe"}
    }


def test_json_response_is_bounded():
    code, stdout, stderr = _run(
        ["health"],
        lambda _request: httpx.Response(
            200,
            content=b"x" * (8 * 1024 * 1024 + 1),
            headers={"content-type": "application/json"},
        ),
    )

    assert code == 6
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "response_too_large"


@pytest.mark.parametrize(
    "body",
    [
        b'{"api_version":"v1","request_id":"req_test","result":{"value":NaN}}',
        b'{"api_version":"v1","request_id":"req_test","result":{"value":1e999}}',
    ],
)
def test_non_finite_server_json_is_a_stable_invalid_response(body):
    code, stdout, stderr = _run(
        ["health"], lambda _request: httpx.Response(200, content=body)
    )

    assert code == 6
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "invalid_response"


def test_excessively_nested_server_json_is_a_stable_invalid_response():
    depth = 200
    body = (
        '{"api_version":"v1","request_id":"req_test","result":'
        + "[" * depth
        + "0"
        + "]" * depth
        + "}"
    ).encode()

    code, stdout, stderr = _run(
        ["health"], lambda _request: httpx.Response(200, content=body)
    )

    assert code == 6
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "invalid_response"


def test_lone_surrogate_response_is_rejected_before_strict_utf8_stdout():
    raw_stdout = io.BytesIO()
    stdout = io.TextIOWrapper(raw_stdout, encoding="utf-8", errors="strict")
    stderr = io.StringIO()

    code = main(
        ["health"],
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                content=(
                    b'{"api_version":"v1","request_id":"req_test",'
                    b'"result":{"value":"\\udcff"}}'
                ),
            )
        ),
        environ={"SUPA_URL": BASE_URL, "SUPA_TOKEN": TOKEN},
        stdout=stdout,
        stderr=stderr,
    )
    stdout.flush()

    assert code == 6
    assert raw_stdout.getvalue() == b""
    assert json.loads(stderr.getvalue())["error"]["code"] == "invalid_response"


def test_unsupported_success_value_is_a_stable_invalid_response(monkeypatch):
    monkeypatch.setattr(
        cli_client.SupaClient,
        "get_json",
        lambda *_args, **_kwargs: {
            "api_version": "v1",
            "request_id": "req_test",
            "result": {"unsupported": {"set-value"}},
        },
    )

    code, stdout, stderr = _run(
        ["health"], lambda _request: pytest.fail("transport is replaced")
    )

    assert code == 6
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "invalid_response"


def test_valid_unicode_response_is_ascii_safe_for_restricted_stdout():
    raw_stdout = io.BytesIO()
    stdout = io.TextIOWrapper(raw_stdout, encoding="ascii", errors="strict")
    stderr = io.StringIO()

    code = main(
        ["health"],
        transport=httpx.MockTransport(
            lambda _request: _success({"message": "Bună"})
        ),
        environ={"SUPA_URL": BASE_URL, "SUPA_TOKEN": TOKEN},
        stdout=stdout,
        stderr=stderr,
    )
    stdout.flush()

    assert code == 0
    assert stderr.getvalue() == ""
    assert json.loads(raw_stdout.getvalue())["result"] == {"message": "Bună"}


def test_attachment_download_streams_to_new_private_file(tmp_path):
    output = tmp_path / "screen.png"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/agent/v1/conversations/321/attachments/701"
        return httpx.Response(
            200,
            content=b"synthetic-image",
            headers={
                "Content-Type": "image/png",
                "X-Request-ID": "req_attachment",
                "X-Supa-API-Version": "v1",
            },
        )

    code, stdout, stderr = _run(
        [
            "attachment",
            "download",
            "321",
            "701",
            "--output",
            str(output),
        ],
        handler,
    )

    assert code == 0
    assert stderr == ""
    assert output.read_bytes() == b"synthetic-image"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    payload = json.loads(stdout)
    assert payload == {
        "api_version": "v1",
        "request_id": "req_attachment",
        "result": {
            "attachment_id": 701,
            "bytes": 15,
            "content_type": "image/png",
            "conversation_id": 321,
            "output": str(output),
        },
    }


@pytest.mark.parametrize("existing_kind", ["file", "symlink"])
def test_attachment_refuses_existing_or_symlink_output(tmp_path, existing_kind):
    output = tmp_path / "output.bin"
    if existing_kind == "file":
        output.write_bytes(b"preserve")
    else:
        target = tmp_path / "target.bin"
        target.write_bytes(b"preserve")
        output.symlink_to(target)

    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, content=b"replace")

    code, stdout, stderr = _run(
        [
            "attachment",
            "download",
            "321",
            "701",
            "--output",
            str(output),
        ],
        handler,
    )

    assert code == 2
    assert stdout == ""
    assert called is False
    assert json.loads(stderr)["error"]["code"] == "unsafe_output"
    if existing_kind == "file":
        assert output.read_bytes() == b"preserve"
    else:
        assert output.is_symlink()
        assert output.resolve().read_bytes() == b"preserve"


def test_attachment_failure_removes_only_partial_output(tmp_path):
    output = tmp_path / "partial.bin"

    code, stdout, stderr = _run(
        [
            "attachment",
            "download",
            "321",
            "701",
            "--output",
            str(output),
        ],
        lambda _request: _error(404, "attachment_not_found"),
    )

    assert code == 4
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "attachment_not_found"
    assert not output.exists()


def test_attachment_rejects_missing_version_header_and_cleans_partial(tmp_path):
    output = tmp_path / "partial.bin"

    code, stdout, stderr = _run(
        [
            "attachment",
            "download",
            "321",
            "701",
            "--output",
            str(output),
        ],
        lambda _request: httpx.Response(
            200,
            content=b"not-trusted",
            headers={"X-Request-ID": "req_attachment"},
        ),
    )

    assert code == 6
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "incompatible_api"
    assert not output.exists()


def test_attachment_output_setup_failure_does_not_leave_a_file(tmp_path, monkeypatch):
    output = tmp_path / "never-created.bin"

    def fail_fchmod(_descriptor: int, _mode: int) -> None:
        raise OSError("synthetic chmod failure")

    monkeypatch.setattr(cli_main.os, "fchmod", fail_fchmod)
    code, stdout, stderr = _run(
        [
            "attachment",
            "download",
            "321",
            "701",
            "--output",
            str(output),
        ],
        lambda _request: pytest.fail("unsafe output must not make a request"),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "unsafe_output"
    assert not output.exists()


def test_attachment_fstat_failure_closes_descriptor_and_removes_created_file(
    tmp_path, monkeypatch
):
    output = tmp_path / "never-created.bin"
    original_open = cli_main.os.open
    original_fstat = cli_main.os.fstat
    output_descriptors: list[int] = []

    def recording_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if Path(path) == output:
            output_descriptors.append(descriptor)
        return descriptor

    def fail_output_fstat(descriptor):
        if descriptor in output_descriptors:
            raise OSError("synthetic fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(cli_main.os, "open", recording_open)
    monkeypatch.setattr(cli_main.os, "fstat", fail_output_fstat)

    code, stdout, stderr = _run(
        [
            "attachment",
            "download",
            "321",
            "701",
            "--output",
            str(output),
        ],
        lambda _request: pytest.fail("unsafe output must not make a request"),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "unsafe_output"
    assert not output.exists()
    assert len(output_descriptors) == 1
    with pytest.raises(OSError):
        original_fstat(output_descriptors[0])


def test_attachment_parent_fsync_failure_removes_partial_output(tmp_path, monkeypatch):
    output = tmp_path / "partial.bin"
    original_fsync = cli_main.os.fsync
    calls = 0

    def fail_directory_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic directory fsync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(cli_main.os, "fsync", fail_directory_fsync)

    code, stdout, stderr = _run(
        [
            "attachment",
            "download",
            "321",
            "701",
            "--output",
            str(output),
        ],
        lambda _request: httpx.Response(
            200,
            content=b"content",
            headers={
                "X-Request-ID": "req_attachment",
                "X-Supa-API-Version": "v1",
            },
        ),
    )

    assert calls == 2
    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "unsafe_output"
    assert not output.exists()


def test_attachment_disk_write_failure_is_stable_and_removes_partial(tmp_path, monkeypatch):
    output = tmp_path / "partial.bin"

    def fail_write(_descriptor: int, _data) -> int:
        raise OSError("synthetic private disk failure")

    monkeypatch.setattr(cli_client.os, "write", fail_write)
    code, stdout, stderr = _run(
        [
            "attachment",
            "download",
            "321",
            "701",
            "--output",
            str(output),
        ],
        lambda _request: httpx.Response(
            200,
            content=b"content",
            headers={
                "X-Request-ID": "req_attachment",
                "X-Supa-API-Version": "v1",
            },
        ),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "unsafe_output"
    assert "synthetic private disk failure" not in stderr
    assert not output.exists()


def test_attachment_zero_progress_write_fails_once_and_removes_partial(
    tmp_path, monkeypatch
):
    output = tmp_path / "partial.bin"
    write_calls = 0

    def zero_then_fail(_descriptor: int, _data) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return 0
        raise OSError("write loop repeated after zero progress")

    monkeypatch.setattr(cli_client.os, "write", zero_then_fail)
    code, stdout, stderr = _run(
        [
            "attachment",
            "download",
            "321",
            "701",
            "--output",
            str(output),
        ],
        lambda _request: httpx.Response(
            200,
            content=b"content",
            headers={
                "X-Request-ID": "req_attachment",
                "X-Supa-API-Version": "v1",
            },
        ),
    )

    assert write_calls == 1
    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "unsafe_output"
    assert not output.exists()


def test_attachment_success_close_failure_is_stable_and_removes_output(
    tmp_path, monkeypatch
):
    output = tmp_path / "partial.bin"
    original_open = cli_main.os.open
    original_close = cli_main.os.close
    output_descriptors: set[int] = set()

    def recording_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if Path(path) == output:
            output_descriptors.add(descriptor)
        return descriptor

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        if descriptor in output_descriptors:
            output_descriptors.remove(descriptor)
            raise OSError("synthetic output close failure")

    monkeypatch.setattr(cli_main.os, "open", recording_open)
    monkeypatch.setattr(cli_main.os, "close", close_then_fail)

    code, stdout, stderr = _run(
        [
            "attachment",
            "download",
            "321",
            "701",
            "--output",
            str(output),
        ],
        lambda _request: httpx.Response(
            200,
            content=b"content",
            headers={
                "X-Request-ID": "req_attachment",
                "X-Supa-API-Version": "v1",
            },
        ),
    )

    assert code == 2
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "unsafe_output"
    assert "synthetic output close failure" not in stderr
    assert not output.exists()


def test_attachment_cleanup_close_failure_preserves_primary_error(
    tmp_path, monkeypatch
):
    output = tmp_path / "partial.bin"
    original_open = cli_main.os.open
    original_close = cli_main.os.close
    output_descriptors: set[int] = set()

    def recording_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if Path(path) == output:
            output_descriptors.add(descriptor)
        return descriptor

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        if descriptor in output_descriptors:
            output_descriptors.remove(descriptor)
            raise OSError("synthetic cleanup close failure")

    monkeypatch.setattr(cli_main.os, "open", recording_open)
    monkeypatch.setattr(cli_main.os, "close", close_then_fail)

    code, stdout, stderr = _run(
        [
            "attachment",
            "download",
            "321",
            "701",
            "--output",
            str(output),
        ],
        lambda _request: _error(404, "attachment_not_found"),
    )

    assert code == 4
    assert stdout == ""
    assert json.loads(stderr)["error"]["code"] == "attachment_not_found"
    assert "synthetic cleanup close failure" not in stderr
    assert not output.exists()


def test_conversation_timeline_labels_unsent_drafts_and_notes():
    conversation = {
        "id": 321,
        "subject": "Printing proofs",
        "status": "active",
        "threads": [
            {
                "id": 1,
                "type": "customer",
                "state": "published",
                "draft": False,
                "is_internal_note": False,
                "author": {"name": "Morgan"},
                "body_text": "Are the printer proofs ready?",
                "created_at": "2026-10-02T10:06:31Z",
            },
            {
                "id": 2,
                "type": "note",
                "state": "published",
                "draft": False,
                "is_internal_note": True,
                "author": {"name": "Rowan"},
                "body_text": "Check the paper size before replying.",
                "created_at": "2026-10-02T10:16:11Z",
            },
            {
                "id": 3,
                "type": "message",
                "state": "draft",
                "draft": True,
                "is_internal_note": False,
                "author": {"name": "Rowan"},
                "body_text": "Morgan, the proofs are ready to review.",
                "created_at": "2026-10-02T10:51:07Z",
            },
            {
                "id": 4,
                "type": "message",
                "state": "published",
                "draft": False,
                "is_internal_note": False,
                "author": {"name": "Rowan"},
                "body_text": "An earlier update",
                "created_at": "2026-10-01T15:53:41Z",
            },
        ],
    }
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return _success(conversation)

    code, stdout, stderr = _run(["conversation", "timeline", "321"], handler)

    assert code == 0
    assert stderr == ""
    assert seen == ["/agent/v1/conversations/321"]
    lines = stdout.splitlines()
    assert "CUSTOMER" in lines[1] and "Morgan" in lines[1]
    assert "NOTE" in lines[2]
    assert "DRAFT (NOT SENT)" in lines[3] and "thread 3" in lines[3]
    assert "SENT" in lines[4] and "DRAFT" not in lines[4]
    assert "Unsent draft: yes (thread 3)" in stdout


def test_conversation_timeline_without_drafts_says_so():
    conversation = {
        "id": 9,
        "subject": "Hi",
        "status": "closed",
        "threads": [
            {
                "id": 7,
                "type": "message",
                "state": "published",
                "draft": False,
                "is_internal_note": False,
                "author": {"name": "Rowan"},
                "body_text": "Done.",
                "created_at": "2026-10-01T00:00:00Z",
            }
        ],
    }
    code, stdout, _ = _run(
        ["conversation", "timeline", "9"], lambda _r: _success(conversation)
    )

    assert code == 0
    assert "Unsent draft: none" in stdout


def test_conversation_timeline_marks_draft_leftover_after_a_later_sent_reply():
    conversation = {
        "id": 5,
        "subject": "Re",
        "status": "active",
        "threads": [
            {
                "id": 10,
                "type": "message",
                "state": "draft",
                "draft": True,
                "author": {"name": "Rowan"},
                "body_text": "Manual review required before sending a reply.",
                "created_at": "2026-10-02T10:51:07Z",
            },
            {
                "id": 11,
                "type": "message",
                "state": "published",
                "draft": False,
                "author": {"name": "Rowan"},
                "body_text": "Hi",
                "created_at": "2026-10-02T10:59:00Z",
            },
        ],
    }
    code, stdout, _ = _run(
        ["conversation", "timeline", "5"], lambda _r: _success(conversation)
    )

    assert code == 0
    assert "DRAFT LEFTOVER" in stdout.splitlines()[1]
    assert "Unsent draft: none" in stdout


def _timeline_draft_and_reply(*, superseded: bool | None) -> dict:
    draft = {
        "id": 10,
        "type": "message",
        "state": "draft",
        "draft": True,
        "author": {"name": "Rowan"},
        "body_text": "Draft text",
        "created_at": "2026-10-02T10:51:07Z",
    }
    if superseded is not None:
        draft["superseded_by_reply"] = superseded
    return {
        "id": 6,
        "subject": "Re",
        "status": "active",
        "threads": [
            draft,
            {
                "id": 11,
                "type": "message",
                "state": "published",
                "draft": False,
                "author": {"name": "Rowan"},
                "body_text": "Hi",
                # The same second as the draft: the timestamp rule cannot tell.
                "created_at": "2026-10-02T10:51:07Z",
                "superseded_by_reply": False,
            },
        ],
    }


def test_conversation_timeline_prefers_the_servers_superseded_flag():
    code, stdout, _ = _run(
        ["conversation", "timeline", "6"],
        lambda _r: _success(_timeline_draft_and_reply(superseded=True)),
    )

    assert code == 0
    assert "DRAFT LEFTOVER" in stdout.splitlines()[1]
    assert "Unsent draft: none" in stdout


def test_conversation_timeline_server_flag_false_keeps_a_draft_pending():
    conversation = _timeline_draft_and_reply(superseded=False)
    # Even when a sent reply's timestamp is later, the server's answer wins.
    conversation["threads"][1]["created_at"] = "2026-10-02T11:00:00Z"
    code, stdout, _ = _run(
        ["conversation", "timeline", "6"], lambda _r: _success(conversation)
    )

    assert code == 0
    assert "DRAFT (NOT SENT)" in stdout.splitlines()[1]
    assert "Unsent draft: yes (thread 10)" in stdout


def test_conversation_timeline_falls_back_to_timestamps_without_the_flag():
    conversation = _timeline_draft_and_reply(superseded=None)
    conversation["threads"][1].pop("superseded_by_reply")
    conversation["threads"][1]["created_at"] = "2026-10-02T11:00:00Z"
    code, stdout, _ = _run(
        ["conversation", "timeline", "6"], lambda _r: _success(conversation)
    )

    assert code == 0
    assert "DRAFT LEFTOVER" in stdout.splitlines()[1]


@pytest.mark.parametrize("command", [["draft", "send"], ["reply", "send"], ["draft", "schedule"]])
def test_expected_thread_help_names_the_actionable_thread(command):
    code, stdout, stderr = _run(
        [*command, "--help"],
        lambda _request: pytest.fail("help must not make a request"),
    )

    assert code == 0
    assert stderr == ""
    assert (
        "the latest actionable thread id (the customer's latest message, "
        "latest_actionable_thread_id in draft get/conversation get), not the draft's thread id"
    ) in " ".join(stdout.split())


@pytest.mark.parametrize("command", [["draft", "send"], ["reply", "send"], ["draft", "schedule"]])
def test_wrong_expected_thread_is_a_conflict_without_retry(command):
    seen = []
    details = {"latest_actionable_thread_id": 40, "given_thread_id": 60, "draft_thread_id": 60}

    def handler(request):
        seen.append(request)
        return _error(409, "wrong_expected_thread", details=details)

    options = (["--expected-version", "a" * 64] if command == ["draft", "send"] else
               ["--body-file", "-", "--body-format", "text"])
    code, stdout, stderr = _run(
        [*command, "321", *options, "--expected-thread", "60", "--idempotency-key", "wrong-thread"],
        handler, stdin=io.StringIO("Hello"),
    )

    assert code == 5
    assert stdout == ""
    assert len(seen) == 1
    error = json.loads(stderr)["error"]
    assert error["code"] == "wrong_expected_thread"
    assert error["retryable"] is False
    assert error["details"] == {**details, "idempotency_key": "wrong-thread"}


def test_wrong_expected_thread_has_an_explicit_conflict_exit_code():
    assert cli_client.api_error_exit_code(500, "wrong_expected_thread") == 5


def test_mailbox_operations_reads_only_the_supervisor_status_route():
    def handler(request):
        assert request.method == "GET" and request.url == f"{BASE_URL}/agent/v1/mailbox/operations"
        return _success({"provider": "vendor"})
    code, stdout, _ = _run(["mailbox", "operations"], handler)
    assert code == 0 and json.loads(stdout)["result"] == {"provider": "vendor"}


def test_unregistered_stage_preserves_the_server_spelling():
    received = []
    def handler(request):
        received.append(json.loads(request.content)["changes"]["stage"])
        return _success({})
    code, _, _ = _run(
        ["conversation", "state", "321", "--stage", "copy-edit"], handler,
        environ={"SUPA_URL": BASE_URL, "SUPA_TOKEN": TOKEN, "SUPA_EXTENSIONS": ""},
    )
    assert code == 0
    assert received == ["copy-edit"]


def test_request_add_and_done_use_the_request_routes():
    seen = []
    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        return _success({"request_id": 41, "status": "open"})
    code, stdout, stderr = _run(["request", "add", "321", "--text", "Confirm whether old captions are kept.",
                                 "--idempotency-key", "captions-1"], handler)
    assert code == 0 and stderr == "" and json.loads(stdout)["result"]["request_id"] == 41
    assert seen[-1] == ("POST", "/agent/v1/conversations/321/requests",
                        {"text": "Confirm whether old captions are kept.", "idempotency_key": "captions-1"})
    code, _, _ = _run(["request", "done", "321", "41"], handler)
    assert code == 0 and seen[-1] == ("POST", "/agent/v1/conversations/321/requests/41/done", {})
    assert _run(["request", "add", "321", "--idempotency-key", "k"], handler)[0] != 0  # No text, no request.
    assert _run(["request", "add", "321", "--text", "Confirm."], handler)[0] != 0  # No key: a retry would ask twice.
    assert len(seen) == 2
    # After an uncertain answer the advice is to send it again, not to look up an operation that does not exist.
    def down(request):
        raise httpx.ConnectError("down")
    code, _, stderr = _run(["request", "add", "321", "--text", "Confirm.", "--idempotency-key", "captions-1"], down)
    assert code != 0 and "operation get" not in stderr


@pytest.mark.parametrize(("args", "method", "path", "params", "body"), [
    (["desk"], "GET", "/desk", {"p": "1"}, None),
    (["conversation", "brief", "321", "--max-chars", "1600", "--quotes", "--precedents", "2"],
     "GET", "/conversations/321/brief", {"max_chars": "1600", "quotes": "1", "precedents": "2"}, None),
    (["conversation", "precedents", "321"], "GET", "/conversations/321/precedents", {}, None),
    (["conversation", "precedents", "--question", "private question", "--tag", "sample"],
     "POST", "/precedents", {}, {"q": "private question", "tag": "sample"}),
    (["mailbox", "insights", "--months", "6"], "GET", "/mailbox/insights", {"months": "6"}, None),
])
def test_desk_commands_use_private_read_contract(args, method, path, params, body):
    def handler(request):
        assert request.method == method and request.url.path == "/agent/v1" + path
        assert dict(request.url.params) == params
        assert (json.loads(request.content) if request.content else None) == body
        return _success({"example": 1})
    code, output, errors = _run(args, handler)
    assert code == 0 and not errors and json.loads(output)["result"] == {"example": 1}


def test_desk_text_prints_reason_receipt_more_and_unavailable():
    desk = dict(page=1, day="2026-11-24", held_more=True, handled_more=False, unavailable_reason=None,
                counts={"held": 1, "handled": 1}, held=[{"number": 71, "subject": "A\nsubject", "reason": "Review " + TOKEN, "waiting": "2h"}],
                handled=[{"number": 72, "subject": "Delivery", "at": "2026-11-24T09:10:00Z", "actor": "Worker", "action": "Sent"}])
    code, output, err = _run(["desk", "--format", "text"], lambda request: _success(desk))
    assert code == 0 and not err
    assert output.splitlines() == ["Held for you (1+)", "#71 A subject · waiting 2h", "  Review [REDACTED]",
                                   "Handled without you (1 · 2026-11-24 UTC)", "#72 Delivery · 09:10 UTC · Worker · Sent",
                                   "More: supa desk --page 2"]
    desk["unavailable_reason"] = "ledger_unavailable"
    assert "unavailable" in _run(["desk", "--format", "text"], lambda request: _success(desk))[1]


def test_desk_on_an_older_server_preserves_exit_and_explains_capabilities():
    def handler(request):
        if request.url.path.endswith("/capabilities"):
            return _success({"commands": {"available": ["health"]}})
        return _error(404, "not_found")
    code, output, err = _run(["desk"], handler)
    assert code == 4 and not output
    assert "supa capabilities" in json.loads(err)["error"]["message"]
