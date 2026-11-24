import agents.provider_alerts as pa


def setup_function(_):
    pa.reset_state()


# --- classification ------------------------------------------------------


def test_classifies_claude_not_logged_in():
    msg = '{"result":"Not logged in · Please run /login"}'
    assert pa.classify_provider_failure(msg) == "needs_login"


def test_classifies_invalid_auth_credentials():
    msg = 'API Error: 401 {"type":"error","error":{"type":"authentication_error"}}'
    assert pa.classify_provider_failure(msg) == "needs_login"


def test_classifies_rate_limit_and_credit():
    assert pa.classify_provider_failure("You've hit your limit · resets 6:30am") == "rate_limited"
    assert pa.classify_provider_failure("Credit balance is too low.") == "rate_limited"


def test_classifies_session_limit_as_rate_limited():
    assert pa.classify_provider_failure("You've hit your session limit") == "rate_limited"


def test_classifies_codex_upgrade_requirement():
    assert (
        pa.classify_provider_failure("requires a newer version of Codex")
        == "needs_upgrade"
    )


def test_codex_upgrade_requirement_wins_marker_collisions():
    message = (
        "This request is unauthorized and over quota because it requires a newer "
        "version of Codex"
    )

    assert pa.classify_provider_failure(message) == "needs_upgrade"


def test_transient_crash_is_not_a_login_failure():
    # The codex 0.136 crash signature must NOT trigger a login alert.
    msg = "Codex CLI failed (exit 1): Reading prompt from stdin... OpenAI Codex v0.136.0"
    assert pa.classify_provider_failure(msg) is None


def test_empty_message_is_none():
    assert pa.classify_provider_failure("") is None
    assert pa.classify_provider_failure(None) is None


# --- alert dispatch + throttle ------------------------------------------


def test_maybe_alert_dispatches_once_then_throttles(monkeypatch):
    calls = []
    monkeypatch.setattr(pa, "_dispatch_alert", lambda subject, body: calls.append(subject))

    now = 1000.0
    r1 = pa.maybe_alert_provider_login("claude", "Not logged in · Please run /login", now=now)
    r2 = pa.maybe_alert_provider_login("claude", "Not logged in · Please run /login", now=now + 60)

    assert r1 == "needs_login"
    assert r2 == "needs_login"          # still classified
    assert len(calls) == 1              # but only dispatched once within the window


def test_throttle_window_expires(monkeypatch):
    calls = []
    monkeypatch.setattr(pa, "_dispatch_alert", lambda subject, body: calls.append(subject))
    monkeypatch.setenv("SUPA_ALERT_MIN_INTERVAL_SECONDS", "100")

    pa.maybe_alert_provider_login("codex", "unauthorized", now=0.0)
    pa.maybe_alert_provider_login("codex", "unauthorized", now=50.0)    # throttled
    pa.maybe_alert_provider_login("codex", "unauthorized", now=200.0)   # window passed
    assert len(calls) == 2


def test_distinct_providers_alert_independently(monkeypatch):
    calls = []
    monkeypatch.setattr(pa, "_dispatch_alert", lambda subject, body: calls.append(subject))
    pa.maybe_alert_provider_login("claude", "Not logged in", now=0.0)
    pa.maybe_alert_provider_login("codex", "Not logged in", now=0.0)
    assert len(calls) == 2


def test_non_login_failure_never_dispatches(monkeypatch):
    calls = []
    monkeypatch.setattr(pa, "_dispatch_alert", lambda subject, body: calls.append(subject))
    assert pa.maybe_alert_provider_login("codex", "exit 1: stream closed") is None
    assert calls == []


def test_upgrade_alert_has_specific_guidance_and_independent_throttle(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pa, "_dispatch_alert", lambda subject, body: calls.append((subject, body))
    )
    detail = "This model requires a newer version of Codex"

    pa.maybe_alert_provider_login("codex", "Not logged in", now=1000.0)
    first_upgrade = pa.maybe_alert_provider_login("codex", detail, now=1000.0)
    second_upgrade = pa.maybe_alert_provider_login("codex", detail, now=1060.0)

    assert first_upgrade == "needs_upgrade"
    assert second_upgrade == "needs_upgrade"
    assert len(calls) == 2
    subject, body = calls[1]
    assert subject == "[Supa] codex CLI upgrade required"
    assert "Upgrade the codex CLI" in body
    assert detail in body


def test_generic_login_alert_body_uses_supported_sign_in_language():
    body = pa._build_body("other", "needs_login", "Not authenticated")

    from services.company_pack import wording
    assert wording("provider_sign_in") in body
    assert "Not authenticated" in body
    assert "/login" not in body


def test_unknown_alert_reason_uses_generic_safe_guidance():
    body = pa._build_body("codex", "unknown_reason", "unexpected failure")

    assert "manual review" in body and "unexpected failure" in body
    assert "upgrade" not in body.lower()


def test_alert_body_redacts_tokens_and_bounds_provider_detail():
    oat_token = "sk-ant-oat01-super-secret-oauth-token"
    api_token = "sk-proj-super-secret-api-token"
    bearer_token = "Bearer eyJhbGciOi.secret.signature"
    detail = (
        f"failure {oat_token} {api_token} {bearer_token} "
        + ("x" * (pa.PROVIDER_DETAIL_LIMIT + 100))
    )

    body = pa._build_body("claude", "rate_limited", detail)
    operator_detail = body.split("Provider detail:\n", 1)[1]

    assert oat_token not in body
    assert api_token not in body
    assert bearer_token not in body
    assert operator_detail.count("[REDACTED]") == 3
    assert len(operator_detail) <= pa.PROVIDER_DETAIL_LIMIT
    assert operator_detail.endswith(pa.PROVIDER_DETAIL_TRUNCATION_MARKER)


# --- delivery channel selection -----------------------------------------


def test_dispatch_logs_only_when_unconfigured(monkeypatch):
    # No webhook + no SMTP -> must not raise, returns having delivered to logs.
    monkeypatch.delenv("SUPA_ALERT_WEBHOOK_URL", raising=False)
    for k in ("SUPA_ALERT_SMTP_HOST", "SUPA_ALERT_SMTP_USER", "SUPA_ALERT_SMTP_PASS"):
        monkeypatch.delenv(k, raising=False)
    # Should be a no-throw path.
    pa._dispatch_alert("[Supa] claude needs login", "body")


def test_dispatch_posts_webhook_when_configured(monkeypatch):
    monkeypatch.setenv("SUPA_ALERT_WEBHOOK_URL", "https://hook.example/test")
    posted = {}
    monkeypatch.setattr(pa, "_post_webhook", lambda url, subject, body: posted.update(url=url, subject=subject))
    # SMTP unconfigured
    for k in ("SUPA_ALERT_SMTP_HOST", "SUPA_ALERT_SMTP_USER", "SUPA_ALERT_SMTP_PASS"):
        monkeypatch.delenv(k, raising=False)
    pa._dispatch_alert("[Supa] claude needs login", "body")
    assert posted["url"] == "https://hook.example/test"


def test_default_email_recipient_is_empty(monkeypatch):
    for k in ("SUPA_ALERT_EMAIL_TO",):
        monkeypatch.delenv(k, raising=False)
    assert pa._email_recipient() == ""


def test_failing_channel_does_not_raise(monkeypatch):
    monkeypatch.setenv("SUPA_ALERT_WEBHOOK_URL", "https://hook.example/test")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(pa, "_post_webhook", boom)
    for k in ("SUPA_ALERT_SMTP_HOST", "SUPA_ALERT_SMTP_USER", "SUPA_ALERT_SMTP_PASS"):
        monkeypatch.delenv(k, raising=False)
    # Must swallow channel errors so ticket processing is never broken by an alert.
    pa._dispatch_alert("[Supa] claude needs login", "body")
