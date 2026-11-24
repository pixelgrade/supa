import base64
import contextlib
import hashlib
import io
import json
import multiprocessing
import os
import stat
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from scripts import manage_agent_tokens
from services import agent_auth
from services.agent_auth import (
    AgentAuthError,
    AgentIdentity,
    AgentTokenRegistry,
    generate_token,
    hash_token,
    require_scopes,
)


def _record(
    token: str,
    *,
    token_id: str = "token-123",
    name: str = "review-desk",
    scopes: tuple[str, ...] = ("read", "events"),
) -> dict[str, object]:
    return {
        "token_id": token_id,
        "name": name,
        "hash": hash_token(token),
        "scopes": list(scopes),
        "created_at": "2026-09-24T12:00:00+00:00",
    }


def _write_registry(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"tokens": records}), encoding="utf-8")
    path.chmod(0o600)


def _run_cli_process(argv: list[str], result_queue) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = manage_agent_tokens.main(argv)
        result_queue.put(("ok", code, stdout.getvalue(), stderr.getvalue()))
    except BaseException as exc:  # pragma: no cover - reported to the parent assertion
        result_queue.put(("error", repr(exc), stdout.getvalue(), stderr.getvalue()))


def _join_process(process) -> None:
    process.join(10)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail(f"worker {process.name} did not finish")
    assert process.exitcode == 0


def test_generate_token_uses_32_random_bytes_and_urlsafe_encoding(monkeypatch):
    random_bytes = bytes(range(32))
    calls: list[int] = []

    def fake_token_bytes(size: int) -> bytes:
        calls.append(size)
        return random_bytes

    monkeypatch.setattr(agent_auth.secrets, "token_bytes", fake_token_bytes)

    token = generate_token()

    assert calls == [32]
    assert token == base64.urlsafe_b64encode(random_bytes).rstrip(b"=").decode("ascii")
    assert len(token) == 43
    assert "+" not in token and "/" not in token and "=" not in token


def test_hash_token_is_sha256_hex_without_retaining_the_token():
    token = "synthetic-secret-that-must-not-be-persisted"

    digest = hash_token(token)

    assert digest == hashlib.sha256(token.encode("utf-8")).hexdigest()
    assert token not in digest


def test_registry_returns_stable_identity_with_exact_additive_scopes(tmp_path):
    token = "synthetic-valid-token"
    path = tmp_path / "tokens.json"
    _write_registry(path, [_record(token)])

    identity = AgentTokenRegistry(path).authenticate(token)

    assert identity == AgentIdentity(
        token_id="token-123",
        name="review-desk",
        scopes=frozenset({"read", "events"}),
    )
    assert isinstance(identity.scopes, frozenset)
    assert "draft" not in identity.scopes


def test_registry_compares_raw_digest_bytes_with_compare_digest(tmp_path, monkeypatch):
    token = "synthetic-valid-token"
    path = tmp_path / "tokens.json"
    _write_registry(path, [_record(token)])
    compared: list[tuple[bytes, bytes]] = []
    real_compare_digest = agent_auth.hmac.compare_digest

    def recording_compare_digest(left: bytes, right: bytes) -> bool:
        compared.append((left, right))
        return real_compare_digest(left, right)

    monkeypatch.setattr(agent_auth.hmac, "compare_digest", recording_compare_digest)

    AgentTokenRegistry(path).authenticate(token)

    assert len(compared) == 1
    assert all(isinstance(value, bytes) for pair in compared for value in pair)
    assert compared[0][0] == hashlib.sha256(token.encode("utf-8")).digest()


def test_registry_reloads_after_atomic_replacement_even_with_same_mtime_and_size(tmp_path):
    first_token = "first-synthetic-token"
    second_token = "other-synthetic-token"
    path = tmp_path / "tokens.json"
    first_payload = json.dumps({"tokens": [_record(first_token)]})
    second_payload = json.dumps({"tokens": [_record(second_token)]})
    assert len(first_payload) == len(second_payload)
    path.write_text(first_payload, encoding="utf-8")
    path.chmod(0o600)
    registry = AgentTokenRegistry(path)
    assert registry.authenticate(first_token).token_id == "token-123"
    original_mtime_ns = path.stat().st_mtime_ns

    replacement = path.with_name("replacement.json")
    replacement.write_text(second_payload, encoding="utf-8")
    replacement.chmod(0o600)
    os.utime(replacement, ns=(original_mtime_ns, original_mtime_ns))
    os.replace(replacement, path)

    with pytest.raises(AgentAuthError):
        registry.authenticate(first_token)
    assert registry.authenticate(second_token).token_id == "token-123"


def test_registry_revocation_is_effective_on_the_next_authentication(tmp_path):
    token = "synthetic-valid-token"
    path = tmp_path / "tokens.json"
    _write_registry(path, [_record(token)])
    registry = AgentTokenRegistry(path)
    assert registry.authenticate(token).name == "review-desk"

    _write_registry(path, [])

    with pytest.raises(AgentAuthError, match="Authentication failed"):
        registry.authenticate(token)


def test_registry_rejects_duplicate_hashes_without_selecting_an_identity_or_leaking_token(
    tmp_path,
):
    token = "duplicate-credential-that-must-not-leak"
    path = tmp_path / "tokens.json"
    _write_registry(
        path,
        [
            _record(token, token_id="token-first", name="first"),
            _record(token, token_id="token-second", name="second"),
        ],
    )

    with pytest.raises(AgentAuthError) as exc_info:
        AgentTokenRegistry(path).authenticate(token)

    assert str(exc_info.value) == "Authentication failed"
    assert token not in str(exc_info.value)


def test_registry_refuses_symlink_even_when_target_is_private_and_valid(tmp_path):
    token = "symlinked-credential"
    target = tmp_path / "real-tokens.json"
    path = tmp_path / "tokens.json"
    _write_registry(target, [_record(token)])
    path.symlink_to(target)

    with pytest.raises(AgentAuthError, match="Authentication failed"):
        AgentTokenRegistry(path).authenticate(token)


@pytest.mark.parametrize("mode", [0o640, 0o604])
def test_registry_refuses_group_or_world_permissions(tmp_path, mode):
    token = "overexposed-credential"
    path = tmp_path / "tokens.json"
    _write_registry(path, [_record(token)])
    path.chmod(mode)

    with pytest.raises(AgentAuthError, match="Authentication failed"):
        AgentTokenRegistry(path).authenticate(token)


def test_registry_refuses_non_regular_file(tmp_path):
    path = tmp_path / "tokens.json"
    path.mkdir(mode=0o700)

    with pytest.raises(AgentAuthError, match="Authentication failed"):
        AgentTokenRegistry(path).authenticate("synthetic-token")


def test_registry_reads_content_and_signature_from_same_open_file(tmp_path, monkeypatch):
    first_token = "first-race-token"
    second_token = "other-race-token"
    path = tmp_path / "tokens.json"
    replacement = tmp_path / "replacement.json"
    _write_registry(path, [_record(first_token)])
    _write_registry(replacement, [_record(second_token)])
    real_fstat = agent_auth.os.fstat
    replaced = False

    def replace_after_fstat(descriptor: int):
        nonlocal replaced
        file_stat = real_fstat(descriptor)
        if not replaced and stat.S_ISREG(file_stat.st_mode):
            os.replace(replacement, path)
            replaced = True
        return file_stat

    monkeypatch.setattr(agent_auth.os, "fstat", replace_after_fstat)
    registry = AgentTokenRegistry(path)

    assert registry.authenticate(first_token).token_id == "token-123"
    assert replaced is True
    with pytest.raises(AgentAuthError):
        registry.authenticate(first_token)
    assert registry.authenticate(second_token).token_id == "token-123"


@pytest.mark.parametrize(
    "change",
    [
        {"unexpected": "metadata"},
        {"created_at": None},
        {"created_at": "not-a-timestamp"},
        {"created_at": "2026-09-24T12:00:00"},
    ],
)
def test_registry_rejects_unknown_keys_and_invalid_created_at(tmp_path, change):
    token = "schema-credential"
    record = _record(token)
    record.update(change)
    path = tmp_path / "tokens.json"
    _write_registry(path, [record])

    with pytest.raises(AgentAuthError, match="Authentication failed"):
        AgentTokenRegistry(path).authenticate(token)


def test_registry_requires_created_at_key(tmp_path):
    token = "missing-created-at-credential"
    record = _record(token)
    record.pop("created_at")
    path = tmp_path / "tokens.json"
    _write_registry(path, [record])

    with pytest.raises(AgentAuthError, match="Authentication failed"):
        AgentTokenRegistry(path).authenticate(token)


@pytest.mark.parametrize("extra_key", ["metadata", "token"])
def test_registry_rejects_extra_top_level_keys_without_leaking_secrets(
    tmp_path, extra_key
):
    token = "top-level-credential-that-must-not-leak"
    path = tmp_path / "tokens.json"
    payload = {"tokens": [_record(token)], extra_key: token}
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(AgentAuthError) as exc_info:
        AgentTokenRegistry(path).authenticate(token)

    assert str(exc_info.value) == "Authentication failed"
    assert token not in str(exc_info.value)


def test_directory_fsync_setup_failure_does_not_publish_or_leave_temp_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "tokens.json"
    original_token = "original-durable-token"
    replacement_token = "replacement-durable-token"
    _write_registry(path, [_record(original_token)])
    original_bytes = path.read_bytes()
    real_open = agent_auth.os.open

    def fail_parent_directory_open(target, flags, *args, **kwargs):
        if Path(target) == path.parent:
            raise OSError("synthetic directory open failure")
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(agent_auth.os, "open", fail_parent_directory_open)

    with pytest.raises(OSError, match="synthetic directory open failure"):
        agent_auth.write_token_records(path, [_record(replacement_token)])

    assert path.read_bytes() == original_bytes
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "not json",
        json.dumps([]),
        json.dumps({"tokens": "not-a-list"}),
        json.dumps({"tokens": [{"token_id": "x"}]}),
        json.dumps({"tokens": [_record("token", scopes=("read", "unknown"))]}),
    ],
)
def test_missing_or_invalid_registry_fails_closed_without_leaking_credentials(tmp_path, payload):
    token = "credential-that-must-not-leak"
    path = tmp_path / "tokens.json"
    if payload is not None:
        path.write_text(payload, encoding="utf-8")
        path.chmod(0o600)

    with pytest.raises(AgentAuthError) as exc_info:
        AgentTokenRegistry(path).authenticate(token)

    assert str(exc_info.value) == "Authentication failed"
    assert token not in str(exc_info.value)


def _auth_client(monkeypatch, registry_path: Path) -> TestClient:
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(registry_path))
    app = FastAPI()

    @app.get("/read")
    def read_endpoint(identity: AgentIdentity = Depends(require_scopes("read"))):
        return {
            "token_id": identity.token_id,
            "name": identity.name,
            "scopes": sorted(identity.scopes),
        }

    @app.post("/send")
    def send_endpoint(
        identity: AgentIdentity = Depends(require_scopes("read", "send")),
    ):
        return {"token_id": identity.token_id}

    return TestClient(app)


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Basic abc", "Bearer", "Bearer ", "Bearer one two"],
)
def test_missing_or_malformed_bearer_header_returns_401(tmp_path, monkeypatch, authorization):
    path = tmp_path / "tokens.json"
    _write_registry(path, [_record("valid-token")])
    client = _auth_client(monkeypatch, path)
    headers = {} if authorization is None else {"Authorization": authorization}

    response = client.get("/read", headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication failed"}
    assert response.headers["www-authenticate"] == "Bearer"


def test_invalid_bearer_token_returns_401_without_echoing_it(tmp_path, monkeypatch):
    path = tmp_path / "tokens.json"
    _write_registry(path, [_record("valid-token")])
    client = _auth_client(monkeypatch, path)

    response = client.get(
        "/read", headers={"Authorization": "Bearer invalid-secret-value"}
    )

    assert response.status_code == 401
    assert "invalid-secret-value" not in response.text


def test_dependency_requires_every_scope_without_implicit_grants(tmp_path, monkeypatch):
    token = "read-only-token"
    path = tmp_path / "tokens.json"
    _write_registry(path, [_record(token, scopes=("read",))])
    client = _auth_client(monkeypatch, path)

    allowed = client.get("/read", headers={"Authorization": f"Bearer {token}"})
    denied = client.post("/send", headers={"Authorization": f"Bearer {token}"})

    assert allowed.status_code == 200
    assert allowed.json() == {
        "token_id": "token-123",
        "name": "review-desk",
        "scopes": ["read"],
    }
    assert denied.status_code == 403
    assert denied.json() == {"detail": "Insufficient scope"}


def test_require_scopes_rejects_unknown_scope_at_definition_time():
    with pytest.raises(ValueError, match="Unknown agent scope"):
        require_scopes("administrator")


def test_cli_create_persists_only_hash_and_list_never_prints_secrets(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "agent-tokens.json"
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(path))

    assert manage_agent_tokens.main(
        ["create", "--name", "review-desk", "--scopes", "read,events"]
    ) == 0
    create_output = json.loads(capsys.readouterr().out)
    raw_token = create_output["token"]
    payload_text = path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)

    assert create_output["token_id"]
    assert create_output["name"] == "review-desk"
    assert create_output["scopes"] == ["events", "read"]
    assert raw_token not in payload_text
    assert payload["tokens"][0]["hash"] == hash_token(raw_token)
    assert "token" not in payload["tokens"][0]
    assert path.stat().st_mode & 0o777 == 0o600
    lock_path = path.with_name(f".{path.name}.lock")
    lock_stat = lock_path.stat()
    assert stat.S_ISREG(lock_stat.st_mode)
    assert lock_stat.st_uid == os.getuid()
    assert stat.S_IMODE(lock_stat.st_mode) == 0o600
    assert lock_path.read_bytes() == b""

    assert manage_agent_tokens.main(["list"]) == 0
    list_output = capsys.readouterr().out
    assert raw_token not in list_output
    assert payload["tokens"][0]["hash"] not in list_output
    listed = json.loads(list_output)
    assert listed == {
        "tokens": [
            {
                "token_id": create_output["token_id"],
                "name": "review-desk",
                "scopes": ["events", "read"],
                "conversations": None,
                "created_at": payload["tokens"][0]["created_at"],
            }
        ]
    }


def test_cli_rejects_duplicate_names_and_unknown_scopes_without_clobbering(tmp_path, monkeypatch):
    path = tmp_path / "agent-tokens.json"
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(path))
    assert manage_agent_tokens.main(
        ["create", "--name", "review-desk", "--scopes", "read"]
    ) == 0
    original = path.read_bytes()

    with pytest.raises(SystemExit) as duplicate:
        manage_agent_tokens.main(
            ["create", "--name", "review-desk", "--scopes", "events"]
        )
    assert duplicate.value.code != 0
    assert path.read_bytes() == original

    with pytest.raises(SystemExit) as unknown:
        manage_agent_tokens.main(
            ["create", "--name", "other", "--scopes", "read,root"]
        )
    assert unknown.value.code != 0
    assert path.read_bytes() == original


def test_cli_refuses_to_clobber_malformed_json(tmp_path, monkeypatch):
    path = tmp_path / "agent-tokens.json"
    malformed = b'{"tokens": ['
    path.write_bytes(malformed)
    path.chmod(0o600)
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(path))

    with pytest.raises(SystemExit) as exc_info:
        manage_agent_tokens.main(
            ["create", "--name", "review-desk", "--scopes", "read"]
        )

    assert exc_info.value.code != 0
    assert path.read_bytes() == malformed


def test_cli_revoke_removes_identity_and_live_registry_reloads(tmp_path, monkeypatch, capsys):
    path = tmp_path / "agent-tokens.json"
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(path))
    assert manage_agent_tokens.main(
        ["create", "--name", "review-desk", "--scopes", "read,events"]
    ) == 0
    token = json.loads(capsys.readouterr().out)["token"]
    registry = AgentTokenRegistry(path)
    assert registry.authenticate(token).name == "review-desk"

    assert manage_agent_tokens.main(["revoke", "--name", "review-desk"]) == 0
    revoke_output = json.loads(capsys.readouterr().out)

    assert revoke_output == {"revoked": "review-desk"}
    with pytest.raises(AgentAuthError):
        registry.authenticate(token)
    assert json.loads(path.read_text(encoding="utf-8")) == {"tokens": []}


def test_concurrent_create_cannot_overwrite_a_completed_revoke(tmp_path, monkeypatch):
    path = tmp_path / "agent-tokens.json"
    old_token = "credential-being-revoked"
    _write_registry(path, [_record(old_token, token_id="old-id", name="old")])
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(path))
    context = multiprocessing.get_context("fork")
    create_loaded = context.Event()
    allow_create = context.Event()
    revoke_attempted_lock = context.Event()
    real_load = manage_agent_tokens._load
    real_transaction_lock = manage_agent_tokens.agent_token_registry_lock

    @contextlib.contextmanager
    def observed_transaction_lock(registry_path):
        if multiprocessing.current_process().name == "revoke-worker":
            revoke_attempted_lock.set()
        with real_transaction_lock(registry_path):
            yield

    def controlled_load(registry_path, *, missing_ok):
        records = real_load(registry_path, missing_ok=missing_ok)
        process_name = multiprocessing.current_process().name
        if process_name == "create-worker":
            create_loaded.set()
            if not allow_create.wait(10):
                raise RuntimeError("create coordination timed out")
        return records

    monkeypatch.setattr(
        manage_agent_tokens, "agent_token_registry_lock", observed_transaction_lock
    )
    monkeypatch.setattr(manage_agent_tokens, "_load", controlled_load)
    results = context.Queue()
    create = context.Process(
        name="create-worker",
        target=_run_cli_process,
        args=(["create", "--name", "new", "--scopes", "read"], results),
    )
    revoke = context.Process(
        name="revoke-worker",
        target=_run_cli_process,
        args=(["revoke", "--name", "old"], results),
    )
    create.start()
    assert create_loaded.wait(5)
    revoke.start()
    assert revoke_attempted_lock.wait(5)
    allow_create.set()
    _join_process(create)
    _join_process(revoke)

    outcomes = [results.get(timeout=2), results.get(timeout=2)]
    assert all(outcome[0:2] == ("ok", 0) for outcome in outcomes)
    records = agent_auth.load_token_records(path)
    assert [record["name"] for record in records] == ["new"]


def test_concurrent_creates_do_not_lose_either_record(tmp_path, monkeypatch):
    path = tmp_path / "agent-tokens.json"
    monkeypatch.setenv("SUPA_AGENT_TOKENS_FILE", str(path))
    context = multiprocessing.get_context("fork")
    a_loaded = context.Event()
    allow_a = context.Event()
    b_attempted_lock = context.Event()
    real_load = manage_agent_tokens._load
    real_transaction_lock = manage_agent_tokens.agent_token_registry_lock

    @contextlib.contextmanager
    def observed_transaction_lock(registry_path):
        if multiprocessing.current_process().name == "create-b":
            b_attempted_lock.set()
        with real_transaction_lock(registry_path):
            yield

    def controlled_load(registry_path, *, missing_ok):
        records = real_load(registry_path, missing_ok=missing_ok)
        process_name = multiprocessing.current_process().name
        if process_name == "create-a":
            a_loaded.set()
            if not allow_a.wait(10):
                raise RuntimeError("create-a coordination timed out")
        return records

    monkeypatch.setattr(
        manage_agent_tokens, "agent_token_registry_lock", observed_transaction_lock
    )
    monkeypatch.setattr(manage_agent_tokens, "_load", controlled_load)
    results = context.Queue()
    create_a = context.Process(
        name="create-a",
        target=_run_cli_process,
        args=(["create", "--name", "alpha", "--scopes", "read"], results),
    )
    create_b = context.Process(
        name="create-b",
        target=_run_cli_process,
        args=(["create", "--name", "beta", "--scopes", "events"], results),
    )
    create_a.start()
    assert a_loaded.wait(5)
    create_b.start()
    assert b_attempted_lock.wait(5)
    allow_a.set()
    _join_process(create_a)
    _join_process(create_b)

    outcomes = [results.get(timeout=2), results.get(timeout=2)]
    assert all(outcome[0:2] == ("ok", 0) for outcome in outcomes)
    records = agent_auth.load_token_records(path)
    assert {record["name"] for record in records} == {"alpha", "beta"}
