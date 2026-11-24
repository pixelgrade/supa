"""The daily write canary, with an in-memory Agent API: no network, no Help Scout."""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone

import pytest

from services import write_canary
from supa_cli.client import ClientError

TOKEN = "T" * 43
CID = 62


class FakeAgentAPI:
    """Just enough of the Agent API for one canary run; records every call."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.operations: dict[str, dict] = {}
        self.exists = False
        self.version = "a" * 64
        self.draft_version = "b" * 64
        # Knobs for failure scenarios.
        self.create_error: ClientError | None = None
        self.save_error: ClientError | None = None
        self.discard_error: ClientError | None = None
        self.save_result_extra: dict = {}
        self.operation_overrides: dict[str, dict] = {}
        self.survives_discard = False
        self.thread_draft = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def _op(self, key, response):
        record = {"state": "succeeded", "error": None, "response": response}
        record.update(self.operation_overrides.get(key.rsplit(".", 1)[-1], {}))
        self.operations[key] = record

    def post_json(self, path, *, json):
        self.calls.append(("POST", path))
        if path == "/agent/v1/conversations":
            assert json["draft"] is True and json["body_format"] == "html"
            assert json["customer_id"] == write_canary.CUSTOMER_ID
            assert "<p>&nbsp;</p>" in json["body"] and "<br>" in json["body"]
            if self.create_error:
                raise self.create_error
            self.exists = True
            result = {"action": "created", "conversation_id": CID, "status": "created"}
            self._op(json["idempotency_key"], result)
            return {"result": result}
        if path == f"/agent/v1/conversations/{CID}/discard":
            assert json["expected_version"] == self.version
            if self.discard_error:
                raise self.discard_error
            self.exists = self.survives_discard
            result = {"action": "discarded", "conversation_id": CID, "status": "discarded"}
            self._op(json["idempotency_key"], result)
            return {"result": result}
        raise AssertionError(path)

    def put_json(self, path, *, json):
        self.calls.append(("PUT", path))
        assert path == f"/agent/v1/conversations/{CID}/draft"
        assert json["expected_version"] == self.draft_version
        assert "<p>&nbsp;</p>" in json["body"] and "<br>" in json["body"]
        if self.save_error:
            raise self.save_error
        result = {"action": "updated", "conversation_id": CID, "thread_id": 9,
                  "version": "c" * 64, **self.save_result_extra}
        self._op(json["idempotency_key"], result)
        return {"result": result}

    def get_json(self, path, *, params=None):
        self.calls.append(("GET", path))
        if path.startswith("/agent/v1/operations/"):
            return {"result": self.operations[path.rsplit("/", 1)[-1]]}
        assert path == f"/agent/v1/conversations/{CID}"
        if not self.exists:
            raise ClientError("Resource not found", code="not_found", exit_code=4)
        return {"result": {
            "id": CID, "version": self.version, "tags": ["supa-ignore"],
            "customer": {"id": write_canary.CUSTOMER_ID},
            "mailbox": {"id": write_canary.MAILBOX_ID},
            "threads": [
                {"id": 8, "type": "lineitem", "state": None, "draft": False},
                {"id": 9, "type": "message", "state": "draft" if self.thread_draft else "published",
                 "draft": self.thread_draft},
            ],
            "draft": {"thread_id": 9, "version": self.draft_version},
        }}


@pytest.fixture(autouse=True)
def canary_target(monkeypatch):
    monkeypatch.setattr(write_canary.installation, "CANARY_EMAIL", "desk@woodland.example")
    monkeypatch.setattr(write_canary, "MAILBOX_ID", 41)
    monkeypatch.setattr(write_canary, "CUSTOMER_ID", 42)


@pytest.fixture
def token_env(tmp_path):
    path = tmp_path / "token"
    path.write_text(TOKEN + "\n")
    os.chmod(path, 0o600)
    return {"SUPA_WRITE_CANARY_TOKEN_FILE": str(path), "HOME": str(tmp_path)}


def run(api, env):
    alerts: list[tuple[str, str]] = []
    out = io.StringIO()
    code = write_canary.main(
        environ=env,
        client_factory=lambda config: api,
        alert=lambda subject, body: alerts.append((subject, body)) or True,
        now=lambda: datetime(2026, 10, 2, 7, 20, tzinfo=timezone.utc),
        stdout=out,
    )
    return code, out.getvalue(), alerts


def discarded(api):
    return ("POST", f"/agent/v1/conversations/{CID}/discard") in api.calls


def test_success_creates_updates_discards_and_logs_one_line(token_env):
    api = FakeAgentAPI()

    code, out, alerts = run(api, token_env)

    assert code == 0
    assert alerts == []
    assert out.count("\n") == 1 and "write canary ok" in out and str(CID) in out
    assert discarded(api) and not api.exists
    # Readback after the discard proves it is gone.
    assert api.calls[-1] == ("GET", f"/agent/v1/conversations/{CID}")


def test_a_warning_on_the_draft_save_alerts_and_still_discards(token_env):
    api = FakeAgentAPI()
    api.save_result_extra = {"warnings": ["verification_mismatch"]}

    code, out, alerts = run(api, token_env)

    assert code == 1
    assert len(alerts) == 1
    subject, body = alerts[0]
    assert subject == "[Supa] write canary failed"
    assert "verification_mismatch" in body and "Discarded: yes" in body
    assert discarded(api) and not api.exists


def test_a_ledger_operation_that_did_not_succeed_alerts(token_env):
    api = FakeAgentAPI()
    api.operation_overrides = {"create": {"state": "outcome_unknown",
                                          "error": {"code": "outcome_unknown"}}}

    code, _, alerts = run(api, token_env)

    assert code == 1
    assert "outcome_unknown" in alerts[0][1]
    assert discarded(api)


def test_a_write_failure_alerts_and_discards_the_created_draft(token_env):
    api = FakeAgentAPI()
    api.save_error = ClientError("The final Help Scout draft differs from the intended write.",
                                 code="external_write_conflict", exit_code=5)

    code, out, alerts = run(api, token_env)

    assert code == 1
    assert "save: external_write_conflict" in alerts[0][1]
    assert "Discarded: yes" in alerts[0][1]
    assert discarded(api) and not api.exists
    assert "FAILED" in out


def test_a_create_failure_alerts_without_a_conversation_to_discard(token_env):
    api = FakeAgentAPI()
    api.create_error = ClientError("Could not reach the Supa service.",
                                   code="transport_error", exit_code=6)

    code, _, alerts = run(api, token_env)

    assert code == 1
    body = alerts[0][1]
    assert "create: transport_error" in body
    assert "No conversation id came back" in body and write_canary.SUBJECT in body
    assert not discarded(api)


def test_an_unexpected_conversation_shape_alerts_before_saving(token_env):
    api = FakeAgentAPI()
    api.thread_draft = False  # the created thread reads as published

    code, _, alerts = run(api, token_env)

    assert code == 1
    assert "not an unsent draft" in alerts[0][1]
    assert ("PUT", f"/agent/v1/conversations/{CID}/draft") not in api.calls
    assert discarded(api)


def test_a_refused_discard_alerts_that_a_draft_is_left(token_env):
    api = FakeAgentAPI()
    api.discard_error = ClientError("Only unsent draft conversations can be discarded",
                                    code="conversation_not_discardable", exit_code=5)

    code, _, alerts = run(api, token_env)

    assert code == 1
    body = alerts[0][1]
    assert "conversation_not_discardable" in body
    assert "Discarded: NO" in body and f"conversation/{CID}" in body


def test_a_discard_that_leaves_the_conversation_readable_alerts(token_env):
    api = FakeAgentAPI()
    api.survives_discard = True

    code, _, alerts = run(api, token_env)

    assert code == 1
    assert "still exists" in alerts[0][1] and "Discarded: NO" in alerts[0][1]


def test_a_token_file_readable_by_others_is_a_setup_failure(token_env):
    os.chmod(token_env["SUPA_WRITE_CANARY_TOKEN_FILE"], 0o644)
    api = FakeAgentAPI()

    code, _, alerts = run(api, token_env)

    assert code == 1
    assert "setup:" in alerts[0][1] and "chmod 600" in alerts[0][1]
    assert api.calls == []


def test_an_alert_delivery_failure_is_logged_not_raised(token_env):
    api = FakeAgentAPI()
    api.save_result_extra = {"status": "saved_with_warning"}
    out = io.StringIO()

    def broken_alert(subject, body):
        raise OSError("smtp down")

    code = write_canary.main(environ=token_env, client_factory=lambda c: api,
                             alert=broken_alert, stdout=out)

    assert code == 1
    assert "alert delivery failed" in out.getvalue()


def test_the_token_never_reaches_output_or_alerts(token_env):
    api = FakeAgentAPI()
    api.save_error = ClientError("boom", code="internal_error", exit_code=6)

    _, out, alerts = run(api, token_env)

    assert TOKEN not in out and TOKEN not in alerts[0][0] + alerts[0][1]


def test_idempotency_keys_pass_the_servers_token_like_guard():
    import re

    result = write_canary.run_canary(FakeAgentAPI(), key_prefix="write-canary.fake-clock.abcd")
    assert result.ok
    for key in result.keys.values():
        assert re.search(r"[A-Za-z0-9_-]{43,}", key) is None


def test_missing_target_prevents_any_request(monkeypatch):
    monkeypatch.setattr(write_canary, "MAILBOX_ID", 0)
    api = FakeAgentAPI()
    result = write_canary.run_canary(api, key_prefix="fixture")
    assert not result.ok and api.calls == []
