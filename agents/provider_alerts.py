"""Provider login / quota outage alerts.

When the Claude or Codex CLI fails because it needs a human to log in again
(`Not logged in`, expired OAuth token, `401 authentication_error`), because it
ran out of quota (`hit your limit`, `Credit balance is too low`), or because the
CLI must be upgraded, Supa tries an alternate provider and keeps any remaining
failures for manual review.

This module classifies those failures and notifies the operator
through the configured alert channels so the session can be restored quickly. It is
deliberately dependency-light and best-effort: every delivery channel is wrapped
so a failing alert can never break ticket processing.

Delivery channels (all optional, tried in addition to a loud log line):
  - Webhook: POST JSON to ``SUPA_ALERT_WEBHOOK_URL`` (n8n / Slack / Discord …).
  - Email:   SMTP via ``SUPA_ALERT_SMTP_HOST`` (+ USER/PASS/PORT/FROM/STARTTLS).

Throttling: at most one alert per ``provider:reason`` per
``SUPA_ALERT_MIN_INTERVAL_SECONDS`` (default 1800s), so an outage that touches
many tickets sends one message, not one per ticket.
"""

from __future__ import annotations

import json
import logging
import os
import re
import smtplib
import time
import urllib.request
from email.message import EmailMessage
from typing import Optional

from services.company_pack import wording

logger = logging.getLogger(__name__)

DEFAULT_EMAIL_TO = ""
DEFAULT_MIN_INTERVAL_SECONDS = 1800
PROVIDER_DETAIL_LIMIT = 600
PROVIDER_DETAIL_TRUNCATION_MARKER = "… [truncated]"

_BEARER_TOKEN_RE = re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_SK_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]+", re.IGNORECASE)

# Failure signatures that mean "a human must log the provider back in".
_LOGIN_MARKERS = (
    "not logged in",
    "please run /login",
    "please log in",
    "invalid authentication credentials",
    "authentication_error",
    "not authenticated",
    "login required",
    "unauthorized",
    "oauth token has expired",
    "token has expired",
    "session expired",
)

# Failure signatures that mean "the provider is out of quota / rate limited".
_QUOTA_MARKERS = (
    "credit balance is too low",
    "hit your limit",
    "session limit",
    "weekly limit",
    "usage limit",
    "rate limit",
    "rate_limit",
    "quota",
)

# Failure signatures that mean the installed provider CLI must be upgraded.
_UPGRADE_MARKERS = (
    "requires a newer version of codex",
)

_last_alert_at: dict[str, float] = {}


def reset_state() -> None:
    """Test helper: clear throttle state."""
    _last_alert_at.clear()


def classify_provider_failure(message: Optional[str]) -> Optional[str]:
    """Classify actionable auth, quota, and provider-upgrade failures."""
    text = (message or "").lower()
    if not text:
        return None
    if any(marker in text for marker in _UPGRADE_MARKERS):
        return "needs_upgrade"
    if any(marker in text for marker in _LOGIN_MARKERS):
        return "needs_login"
    if any(marker in text for marker in _QUOTA_MARKERS):
        return "rate_limited"
    return None


def sanitize_provider_detail(message: Optional[str]) -> str:
    """Redact provider credentials and bound operator-facing failure details."""
    detail = (message or "").strip()
    detail = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", detail)
    detail = _SK_TOKEN_RE.sub("[REDACTED]", detail)
    if len(detail) <= PROVIDER_DETAIL_LIMIT:
        return detail

    content_limit = PROVIDER_DETAIL_LIMIT - len(PROVIDER_DETAIL_TRUNCATION_MARKER)
    return (
        detail[:content_limit].rstrip()
        + PROVIDER_DETAIL_TRUNCATION_MARKER
    )


def _interval_seconds() -> float:
    raw = (os.getenv("SUPA_ALERT_MIN_INTERVAL_SECONDS", "") or "").strip()
    try:
        value = float(raw)
    except ValueError:
        return float(DEFAULT_MIN_INTERVAL_SECONDS)
    return value if value > 0 else float(DEFAULT_MIN_INTERVAL_SECONDS)


def _should_send(alert_key: str, now: float, interval: float) -> bool:
    last = _last_alert_at.get(alert_key)
    if last is not None and (now - last) < interval:
        return False
    _last_alert_at[alert_key] = now
    return True


def maybe_alert_provider_login(
    provider: str,
    message: Optional[str],
    *,
    now: Optional[float] = None,
) -> Optional[str]:
    """Classify a provider failure and, if it is actionable, dispatch a
    throttled operator alert. Returns the classification (or ``None``).

    Best-effort: never raises — alerting must not break ticket processing.
    """
    try:
        reason = classify_provider_failure(message)
        if reason is None:
            return None

        provider_name = (provider or "unknown").strip().lower() or "unknown"
        alert_key = f"{provider_name}:{reason}"
        moment = time.time() if now is None else now

        if not _should_send(alert_key, moment, _interval_seconds()):
            logger.warning("Provider login alert throttled: %s", alert_key)
            return reason

        if reason == "needs_upgrade":
            subject = f"[Supa] {provider_name} CLI upgrade required"
        else:
            subject = f"[Supa] {provider_name} needs attention ({reason})"
        _dispatch_alert(subject, _build_body(provider_name, reason, message or ""))
        return reason
    except Exception as exc:  # noqa: BLE001 - alerting is best-effort
        logger.error("Provider login alert failed unexpectedly: %s", exc)
        return None


def _build_body(provider: str, reason: str, message: str) -> str:
    if reason == "needs_login":
        headline = (
            f"The {provider} CLI on {wording('service_host')} is not authenticated. Supa is "
            f"sending tickets to manual review until it is logged back in."
        )
        if provider == "claude":
            fix = wording("claude_sign_in")
        elif provider == "codex":
            fix = wording("codex_sign_in")
        else:
            fix = wording("provider_sign_in")
    elif reason == "rate_limited":
        fallback = "Codex" if provider == "claude" else "the alternate provider"
        headline = (
            f"The {provider} CLI on {wording('service_host')} is out of quota / rate limited. "
            f"Supa is using {fallback} fallback where available; a ticket needs "
            f"manual review only if fallback fails."
        )
        fix = "Fix: wait for the quota window to reset or switch the provider / plan."
    elif reason == "needs_upgrade":
        headline = (
            f"The {provider} CLI on {wording('service_host')} is too old for the requested model. "
            f"Supa is degrading affected tickets to manual review."
        )
        fix = (
            f"Fix: Upgrade the {provider} CLI, restart Supa, and confirm the provider "
            f"with a health check."
        )
    else:
        headline = (
            f"The {provider} CLI on {wording('service_host')} failed for an unrecognized "
            f"actionable reason. Supa is degrading affected tickets to manual review."
        )
        fix = (
            "Fix: inspect the provider detail, restore the provider using its "
            "supported recovery flow, then restart and verify Supa."
        )
    detail = sanitize_provider_detail(message)
    return f"{headline}\n\n{fix}\n\nProvider detail:\n{detail}"


# --- delivery channels ---------------------------------------------------


def _email_recipient() -> str:
    return (os.getenv("SUPA_ALERT_EMAIL_TO", "") or "").strip() or DEFAULT_EMAIL_TO


def _smtp_configured() -> bool:
    return bool((os.getenv("SUPA_ALERT_SMTP_HOST", "") or "").strip())


def _dispatch_alert(subject: str, body: str) -> None:
    send_operator_alert(subject, body, log_label="PROVIDER LOGIN ALERT")


def send_operator_alert(subject: str, body: str, *, log_label: str = "SUPA OPERATOR ALERT") -> bool:
    """Deliver an alert to the operator through every configured channel.

    Always logs; webhook and SMTP are attempted only when configured. Channel
    failures are swallowed. Returns whether any channel beyond the log took it.
    Other operational checks (e.g. the daily write canary) reuse this channel.
    """
    logger.error("%s — %s\n%s", log_label, subject, body)

    delivered = False

    webhook = (os.getenv("SUPA_ALERT_WEBHOOK_URL", "") or "").strip()
    if webhook:
        try:
            _post_webhook(webhook, subject, body)
            delivered = True
        except Exception as exc:  # noqa: BLE001
            logger.error("Provider alert webhook delivery failed: %s", exc)

    if _smtp_configured() and _email_recipient():
        try:
            _send_email(subject, body)
            delivered = True
        except Exception as exc:  # noqa: BLE001
            logger.error("Provider alert email delivery failed: %s", exc)

    if not delivered:
        logger.warning(
            "Provider alert delivered to logs only — set SUPA_ALERT_WEBHOOK_URL "
            "or SUPA_ALERT_SMTP_* to reach %s.",
            _email_recipient(),
        )
    return delivered


def _post_webhook(url: str, subject: str, body: str) -> None:
    payload = json.dumps(
        {
            "text": f"{subject}\n\n{body}",  # Slack/Discord/n8n-friendly
            "subject": subject,
            "body": body,
            "recipient": _email_recipient(),
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310 - operator-configured URL
        resp.read()


def _send_email(subject: str, body: str) -> None:
    host = (os.getenv("SUPA_ALERT_SMTP_HOST", "") or "").strip()
    port = int((os.getenv("SUPA_ALERT_SMTP_PORT", "") or "587").strip() or "587")
    user = (os.getenv("SUPA_ALERT_SMTP_USER", "") or "").strip()
    password = (os.getenv("SUPA_ALERT_SMTP_PASS", "") or "").strip()
    sender = (os.getenv("SUPA_ALERT_EMAIL_FROM", "") or "").strip() or user or _email_recipient()
    use_starttls = (os.getenv("SUPA_ALERT_SMTP_STARTTLS", "true") or "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = _email_recipient()
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=15) as server:
        if use_starttls:
            server.starttls()
        if user and password:
            server.login(user, password)
        server.send_message(msg)
