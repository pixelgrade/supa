"""Shared core for re-processing tickets stranded by a provider (Claude/Codex)
login/quota outage.

When a provider is logged out, incoming tickets still run but degrade to 0/10
manual-review drafts. They do NOT auto-replay when the provider recovers (Supa
only acts on new webhook events), so they need a one-time re-run. This module
identifies the *provider-failure casualties* (not ordinary manual-review
tickets) and exposes the small helpers the dashboard endpoint and the CLI
script both use, so the logic lives in exactly one place.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from agents.support_agent import autosend_threshold
from agents.support_agent import customer_waiting  # noqa: F401  (one rule, re-exported)

# Latest audit row per conversation whose failure looks like a provider
# login/quota outage — NOT a legitimate manual review (those lack these
# signatures, e.g. a marketing newsletter that correctly got no draft).
_FAILURE_SQL = """
WITH recent AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY helpscout_id ORDER BY created_at DESC) AS rn
  FROM tickets
  WHERE created_at >= ?
)
SELECT helpscout_id, subject, created_at, confidence_score, confidence_status,
       delivery_mode, risk_flags, confidence_summary
FROM recent
WHERE rn = 1
  AND (
    risk_flags LIKE '%investigator_failure%'
    OR risk_flags LIKE '%evaluator_failure%'
    OR confidence_summary LIKE '%Codex CLI failed%'
    OR confidence_summary LIKE '%Not logged in%'
    OR confidence_summary LIKE '%Credit balance%'
    OR confidence_summary LIKE '%authentication%'
    OR confidence_summary LIKE '%hit your limit%'
  )
ORDER BY created_at
"""


def default_since(hours: int = 24) -> str:
    """UTC 'YYYY-MM-DD HH:MM:SS' for `hours` ago (the lookback window)."""
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")


def find_casualties(db_path: str, since: str) -> list[dict[str, Any]]:
    """Return the latest provider-failure casualty rows at/after `since`."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(_FAILURE_SQL, (since,)).fetchall()]
    finally:
        con.close()


def can_auto_send(*, confidence_score: int | None, delivery_mode: str | None,
                  confidence_status: str | None, hard_blockers: list[str] | None,
                  waiting: bool) -> bool:
    """Supa's standard auto-send gate (same as implement-ticket): the support threshold (8+)."""
    return (
        (confidence_score or 0) >= autosend_threshold()
        and delivery_mode == "auto_send"
        and confidence_status == "accept"
        and not (hard_blockers or [])
        and waiting
    )
