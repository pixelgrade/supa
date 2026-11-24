import sqlite3
from contextlib import closing
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone

from services.runtime_paths import get_supa_db_path


DB_PATH = Path(get_supa_db_path())


def get_db(path: Optional[str] = None) -> sqlite3.Connection:
    """Get database connection with better concurrency support."""
    db_path = path or str(DB_PATH)
    conn = sqlite3.connect(db_path, timeout=30.0)  # Wait up to 30 seconds if locked
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent access
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")  # 30 seconds
    return conn


def init_db(path: Optional[str] = None) -> None:
    """Initialize database tables."""
    conn = get_db(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            helpscout_id TEXT UNIQUE NOT NULL,
            subject TEXT,
            customer_name TEXT,
            classification TEXT,
            status TEXT,
            response_posted INTEGER DEFAULT 0,
            error_message TEXT,
            trigger TEXT DEFAULT 'webhook',
            confidence_score INTEGER,
            confidence_status TEXT,
            confidence_attempts INTEGER DEFAULT 1,
            confidence_summary TEXT,
            confidence_trace TEXT,
            confidence_provider TEXT,
            draft_response_html TEXT,
            evidence_summary TEXT,
            delivery_mode TEXT,
            investigator_provider TEXT,
            evaluator_provider TEXT,
            decision_note TEXT,
            investigation_steps TEXT,
            risk_flags TEXT,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_tickets_created ON tickets(created_at);
        CREATE INDEX IF NOT EXISTS idx_tickets_classification ON tickets(classification);

        CREATE TABLE IF NOT EXISTS ticket_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            helpscout_id TEXT NOT NULL,
            ticket_id INTEGER,
            subject TEXT,
            customer_name TEXT,
            classification TEXT,
            status TEXT,
            response_posted INTEGER DEFAULT 0,
            error_message TEXT,
            trigger TEXT DEFAULT 'webhook',
            confidence_score INTEGER,
            confidence_status TEXT,
            confidence_attempts INTEGER DEFAULT 1,
            confidence_summary TEXT,
            confidence_trace TEXT,
            confidence_provider TEXT,
            draft_response_html TEXT,
            evidence_summary TEXT,
            delivery_mode TEXT,
            investigator_provider TEXT,
            evaluator_provider TEXT,
            decision_note TEXT,
            investigation_steps TEXT,
            risk_flags TEXT,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_ticket_runs_helpscout ON ticket_runs(helpscout_id);
        CREATE INDEX IF NOT EXISTS idx_ticket_runs_created ON ticket_runs(created_at);

        CREATE TABLE IF NOT EXISTS semantic_failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            helpscout_id TEXT NOT NULL,
            subject TEXT,
            theme_tag TEXT,
            customer_message TEXT,
            draft_html TEXT,
            final_reply_html TEXT,
            change_type TEXT,
            summary TEXT,
            wrong_claim TEXT,
            correct_claim TEXT,
            confidence_score INTEGER,
            confidence_status TEXT,
            confidence_provider TEXT,
            confidence_trace TEXT,
            evidence_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_semantic_failures_created ON semantic_failures(created_at);
        CREATE INDEX IF NOT EXISTS idx_semantic_failures_helpscout ON semantic_failures(helpscout_id);
        CREATE INDEX IF NOT EXISTS idx_semantic_failures_theme ON semantic_failures(theme_tag);

        CREATE TABLE IF NOT EXISTS pending_sends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            draft_thread_id TEXT NOT NULL,
            response_html_sha TEXT NOT NULL,
            queued_at TEXT NOT NULL,
            due_at TEXT NOT NULL,
            last_known_customer_thread_id TEXT,
            tags_on_promote TEXT NOT NULL DEFAULT '[]',
            confidence_score INTEGER,
            confidence_provider TEXT,
            state TEXT NOT NULL DEFAULT 'scheduled',
            cancel_reason TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            published_thread_id TEXT,
            sent_at TEXT,
            delivery_verified_at TEXT,
            delivery_receipt_error TEXT,
            cleanup_error TEXT,
            cleanup_updated_at TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_pending_sends_ready ON pending_sends(state, due_at);
        CREATE INDEX IF NOT EXISTS idx_pending_sends_conversation ON pending_sends(conversation_id, state);

        CREATE TABLE IF NOT EXISTS agent_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            conversation_id TEXT,
            thread_id TEXT,
            run_id TEXT,
            source TEXT NOT NULL,
            actor_type TEXT,
            occurred_at TEXT NOT NULL,
            received_at TEXT NOT NULL,
            dedupe_key TEXT NOT NULL,
            summary_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_agent_events_id
            ON agent_events(id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_events_dedupe_key
            ON agent_events(dedupe_key);
        CREATE INDEX IF NOT EXISTS idx_agent_events_conversation_id
            ON agent_events(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_agent_events_received_at
            ON agent_events(received_at);

        CREATE TABLE IF NOT EXISTS agent_event_consumers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_id TEXT NOT NULL,
            consumer_name TEXT NOT NULL,
            acknowledged_event_id INTEGER NOT NULL DEFAULT 0,
            leased_event_id INTEGER,
            lease_expires_at TEXT,
            lease_token_hash TEXT,
            last_ack_event_id INTEGER,
            last_ack_token_hash TEXT,
            start_mode TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_agent_event_consumers_lease_expiry
            ON agent_event_consumers(lease_expires_at);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_event_consumers_identity
            ON agent_event_consumers(token_id, consumer_name);

        CREATE TABLE IF NOT EXISTS agent_operations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idempotency_key TEXT NOT NULL,
            token_id TEXT NOT NULL,
            token_name TEXT NOT NULL,
            operation_name TEXT NOT NULL,
            target_id TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            state TEXT NOT NULL CHECK (
                state IN (
                    'reserved',
                    'in_progress',
                    'succeeded',
                    'sent_with_warning',
                    'failed',
                    'outcome_unknown'
                )
            ),
            request_metadata_json TEXT NOT NULL DEFAULT '{}',
            upstream_thread_id TEXT,
            response_json TEXT,
            error_json TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            upstream_started_at TEXT,
            completed_at TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_operations_idempotency_key
            ON agent_operations(idempotency_key);
        CREATE INDEX IF NOT EXISTS idx_agent_operations_token_created
            ON agent_operations(token_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_agent_operations_state_updated
            ON agent_operations(state, updated_at);

        CREATE TABLE IF NOT EXISTS agent_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idempotency_key TEXT,
            token_id TEXT NOT NULL,
            token_name TEXT NOT NULL,
            scope TEXT NOT NULL,
            operation_name TEXT NOT NULL,
            target_id TEXT NOT NULL,
            request_id TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            result TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_agent_audit_target_created
            ON agent_audit(target_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_agent_audit_operation_created
            ON agent_audit(idempotency_key, created_at);

        CREATE TABLE IF NOT EXISTS agent_runs (
            run_id TEXT NOT NULL PRIMARY KEY,
            idempotency_key TEXT,
            token_id TEXT NOT NULL,
            token_name TEXT NOT NULL,
            task_name TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN (
                    'queued',
                    'running',
                    'completed',
                    'failed',
                    'interrupted'
                )
            ),
            arguments_json TEXT NOT NULL DEFAULT '{}',
            scope_snapshot_json TEXT NOT NULL DEFAULT '[]',
            delivery_ceiling TEXT CHECK (
                delivery_ceiling IS NULL OR delivery_ceiling IN ('draft', 'policy')
            ),
            result_json TEXT,
            error_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_agent_runs_status_created
            ON agent_runs(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation_id
            ON agent_runs(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_agent_runs_token_id
            ON agent_runs(token_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_runs_owner_idempotency
            ON agent_runs(token_id, idempotency_key)
            WHERE idempotency_key IS NOT NULL;

        CREATE TABLE IF NOT EXISTS agent_run_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            sequence INTEGER NOT NULL CHECK (sequence > 0),
            stage TEXT NOT NULL,
            progress_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES agent_runs(run_id)
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_run_events_run_sequence
            ON agent_run_events(run_id, sequence);

        CREATE TRIGGER IF NOT EXISTS agent_run_events_reject_update
        BEFORE UPDATE ON agent_run_events
        BEGIN
            SELECT RAISE(ABORT, 'agent_run_events is append-only');
        END;

        CREATE TRIGGER IF NOT EXISTS agent_run_events_reject_delete
        BEFORE DELETE ON agent_run_events
        BEGIN
            SELECT RAISE(ABORT, 'agent_run_events is append-only');
        END;

        CREATE TRIGGER IF NOT EXISTS agent_audit_reject_replace
        BEFORE INSERT ON agent_audit
        WHEN EXISTS (
            SELECT 1 FROM agent_audit AS existing WHERE existing.id = NEW.id
        )
        BEGIN
            SELECT RAISE(ABORT, 'agent_audit is append-only');
        END;

        CREATE TRIGGER IF NOT EXISTS agent_audit_reject_update
        BEFORE UPDATE ON agent_audit
        BEGIN
            SELECT RAISE(ABORT, 'agent_audit is append-only');
        END;

        CREATE TRIGGER IF NOT EXISTS agent_audit_reject_delete
        BEFORE DELETE ON agent_audit
        BEGIN
            SELECT RAISE(ABORT, 'agent_audit is append-only');
        END;

        CREATE TABLE IF NOT EXISTS agent_conversation_locks (
            conversation_id TEXT NOT NULL PRIMARY KEY,
            owner TEXT NOT NULL,
            acquired_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_agent_conversation_locks_expires
            ON agent_conversation_locks(expires_at);

        -- Agent replies booked for later (supa draft schedule); services/scheduled_sends.py.
        CREATE TABLE IF NOT EXISTS scheduled_sends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            requested_conversation_id INTEGER NOT NULL,
            source TEXT NOT NULL CHECK (source IN ('body', 'draft')),
            body TEXT NOT NULL,
            body_format TEXT NOT NULL CHECK (body_format IN ('html', 'text')),
            body_sha256 TEXT NOT NULL,
            draft_thread_id INTEGER,
            draft_version TEXT,
            expected_latest_thread_id INTEGER NOT NULL,
            customer_message_at TEXT,
            send_at TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('queued', 'sent', 'held', 'cancelled', 'failed')
            ),
            hold_reason TEXT,
            token_id TEXT NOT NULL,
            token_name TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TEXT,
            claimed_at TEXT,
            send_operation_key TEXT,
            last_error TEXT,
            sent_thread_id INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            cc_json TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_scheduled_sends_idempotency
            ON scheduled_sends(token_id, idempotency_key);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_scheduled_sends_one_queued
            ON scheduled_sends(conversation_id) WHERE status = 'queued';
        CREATE INDEX IF NOT EXISTS idx_scheduled_sends_due
            ON scheduled_sends(status, send_at);
    """)
    # Lightweight additive migrations for local SQLite databases.
    migrations = [
        ("trigger", "ALTER TABLE tickets ADD COLUMN trigger TEXT DEFAULT 'webhook'"),
        ("confidence_score", "ALTER TABLE tickets ADD COLUMN confidence_score INTEGER"),
        ("confidence_status", "ALTER TABLE tickets ADD COLUMN confidence_status TEXT"),
        ("confidence_attempts", "ALTER TABLE tickets ADD COLUMN confidence_attempts INTEGER DEFAULT 1"),
        ("confidence_summary", "ALTER TABLE tickets ADD COLUMN confidence_summary TEXT"),
        ("confidence_trace", "ALTER TABLE tickets ADD COLUMN confidence_trace TEXT"),
        ("confidence_provider", "ALTER TABLE tickets ADD COLUMN confidence_provider TEXT"),
        ("draft_response_html", "ALTER TABLE tickets ADD COLUMN draft_response_html TEXT"),
        ("evidence_summary", "ALTER TABLE tickets ADD COLUMN evidence_summary TEXT"),
        ("delivery_mode", "ALTER TABLE tickets ADD COLUMN delivery_mode TEXT"),
        ("investigator_provider", "ALTER TABLE tickets ADD COLUMN investigator_provider TEXT"),
        ("evaluator_provider", "ALTER TABLE tickets ADD COLUMN evaluator_provider TEXT"),
        ("decision_note", "ALTER TABLE tickets ADD COLUMN decision_note TEXT"),
        ("investigation_steps", "ALTER TABLE tickets ADD COLUMN investigation_steps TEXT"),
        ("risk_flags", "ALTER TABLE tickets ADD COLUMN risk_flags TEXT"),
        ("sources", "ALTER TABLE tickets ADD COLUMN sources TEXT"),
        # CC addresses booked with a scheduled reply (supa draft schedule --cc).
        ("scheduled_cc", "ALTER TABLE scheduled_sends ADD COLUMN cc_json TEXT"),
        ("published_thread_id", "ALTER TABLE pending_sends ADD COLUMN published_thread_id TEXT"),
        ("sent_at", "ALTER TABLE pending_sends ADD COLUMN sent_at TEXT"),
        ("delivery_verified_at", "ALTER TABLE pending_sends ADD COLUMN delivery_verified_at TEXT"),
        ("delivery_receipt_error", "ALTER TABLE pending_sends ADD COLUMN delivery_receipt_error TEXT"),
        ("cleanup_error", "ALTER TABLE pending_sends ADD COLUMN cleanup_error TEXT"),
        ("cleanup_updated_at", "ALTER TABLE pending_sends ADD COLUMN cleanup_updated_at TEXT"),
    ]

    for _, statement in migrations:
        try:
            conn.execute(statement)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()
    conn.close()


def ensure_db_initialized(path: Optional[str] = None) -> str:
    """Ensure the SQLite schema exists before read-only access."""
    db_path = path or str(DB_PATH)
    init_db(db_path)
    return db_path


def log_ticket(
    db_path: Optional[str] = None,
    helpscout_id: str = "",
    subject: str = "",
    customer_name: str = "",
    classification: str = "",
    status: str = "",
    response_posted: bool = False,
    error_message: Optional[str] = None,
    trigger: str = "webhook",
    confidence_score: Optional[int] = None,
    confidence_status: str = "",
    confidence_attempts: int = 1,
    confidence_summary: Optional[str] = None,
    confidence_trace: Optional[str] = None,
    confidence_provider: Optional[str] = None,
    draft_response_html: Optional[str] = None,
    evidence_summary: Optional[str] = None,
    delivery_mode: Optional[str] = None,
    investigator_provider: Optional[str] = None,
    evaluator_provider: Optional[str] = None,
    decision_note: Optional[str] = None,
    investigation_steps: Optional[Any] = None,
    risk_flags: Optional[Any] = None,
    sources: Optional[Any] = None,
) -> int:
    """Log a processed ticket."""
    def _serialize_optional_json(value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value)
        except TypeError:
            return json.dumps(str(value))

    serialized_steps = _serialize_optional_json(investigation_steps)
    serialized_risk_flags = _serialize_optional_json(risk_flags)
    serialized_sources = _serialize_optional_json(sources)

    conn = get_db(db_path)
    cursor = conn.execute("""
        INSERT INTO tickets (
            helpscout_id, subject, customer_name, classification, status,
            response_posted, error_message, trigger, confidence_score,
            confidence_status, confidence_attempts, confidence_summary,
            confidence_trace, confidence_provider, draft_response_html,
            evidence_summary, delivery_mode, investigator_provider,
            evaluator_provider, decision_note, investigation_steps,
            risk_flags, sources
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(helpscout_id) DO UPDATE SET
            subject = excluded.subject,
            customer_name = excluded.customer_name,
            classification = excluded.classification,
            status = excluded.status,
            response_posted = excluded.response_posted,
            error_message = excluded.error_message,
            trigger = excluded.trigger,
            confidence_score = excluded.confidence_score,
            confidence_status = excluded.confidence_status,
            confidence_attempts = excluded.confidence_attempts,
            confidence_summary = excluded.confidence_summary,
            confidence_trace = excluded.confidence_trace,
            confidence_provider = excluded.confidence_provider,
            draft_response_html = excluded.draft_response_html,
            evidence_summary = excluded.evidence_summary,
            delivery_mode = excluded.delivery_mode,
            investigator_provider = excluded.investigator_provider,
            evaluator_provider = excluded.evaluator_provider,
            decision_note = excluded.decision_note,
            investigation_steps = excluded.investigation_steps,
            risk_flags = excluded.risk_flags,
            sources = excluded.sources,
            created_at = CURRENT_TIMESTAMP
    """, (
        helpscout_id,
        subject,
        customer_name,
        classification,
        status,
        int(response_posted),
        error_message,
        trigger,
        confidence_score,
        confidence_status,
        confidence_attempts,
        confidence_summary,
        confidence_trace,
        confidence_provider,
        draft_response_html,
        evidence_summary,
        delivery_mode,
        investigator_provider,
        evaluator_provider,
        decision_note,
        serialized_steps,
        serialized_risk_flags,
        serialized_sources,
    ))
    conn.commit()
    ticket_id = conn.execute(
        "SELECT id FROM tickets WHERE helpscout_id = ?",
        (helpscout_id,),
    ).fetchone()["id"]
    conn.execute(
        """
        INSERT INTO ticket_runs (
            helpscout_id, ticket_id, subject, customer_name, classification, status,
            response_posted, error_message, trigger, confidence_score,
            confidence_status, confidence_attempts, confidence_summary,
            confidence_trace, confidence_provider, draft_response_html,
            evidence_summary, delivery_mode, investigator_provider,
            evaluator_provider, decision_note, investigation_steps,
            risk_flags, sources
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            helpscout_id,
            ticket_id,
            subject,
            customer_name,
            classification,
            status,
            int(response_posted),
            error_message,
            trigger,
            confidence_score,
            confidence_status,
            confidence_attempts,
            confidence_summary,
            confidence_trace,
            confidence_provider,
            draft_response_html,
            evidence_summary,
            delivery_mode,
            investigator_provider,
            evaluator_provider,
            decision_note,
            serialized_steps,
            serialized_risk_flags,
            serialized_sources,
        ),
    )
    conn.commit()
    conn.close()
    return ticket_id


def get_recent_tickets(db_path: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent ticket logs."""
    conn = get_db(ensure_db_initialized(db_path))
    cursor = conn.execute("""
        SELECT * FROM tickets ORDER BY created_at DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_ticket_logs_by_helpscout_ids(
    db_path: Optional[str] = None,
    helpscout_ids: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fetch ticket logs keyed by HelpScout id."""
    if not helpscout_ids:
        return {}

    conn = get_db(ensure_db_initialized(db_path))
    placeholders = ", ".join("?" for _ in helpscout_ids)
    cursor = conn.execute(
        f"SELECT * FROM tickets WHERE helpscout_id IN ({placeholders})",
        tuple(helpscout_ids),
    )
    rows = cursor.fetchall()
    conn.close()
    return {row["helpscout_id"]: dict(row) for row in rows}


def get_ticket_log_by_helpscout_id(
    db_path: Optional[str] = None,
    helpscout_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Fetch one ticket log by HelpScout id."""
    if not helpscout_id:
        return None

    conn = get_db(ensure_db_initialized(db_path))
    row = conn.execute(
        "SELECT * FROM tickets WHERE helpscout_id = ?",
        (helpscout_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_ticket_runs_by_helpscout_id(
    db_path: Optional[str] = None,
    helpscout_id: str = "",
) -> List[Dict[str, Any]]:
    """Fetch append-only ticket processing history for one HelpScout conversation."""
    if not helpscout_id:
        return []

    conn = get_db(ensure_db_initialized(db_path))
    rows = conn.execute(
        """
        SELECT * FROM ticket_runs
        WHERE helpscout_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (helpscout_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def log_semantic_failure(
    db_path: Optional[str] = None,
    helpscout_id: str = "",
    subject: str = "",
    theme_tag: str = "",
    customer_message: str = "",
    draft_html: str = "",
    final_reply_html: str = "",
    change_type: str = "",
    summary: str = "",
    wrong_claim: str = "",
    correct_claim: str = "",
    confidence_score: Optional[int] = None,
    confidence_status: str = "",
    confidence_provider: Optional[str] = None,
    confidence_trace: Optional[str] = None,
    evidence_summary: Optional[str] = None,
) -> int:
    """Persist a semantic correction made by the human responder."""
    conn = get_db(db_path)
    cursor = conn.execute(
        """
        INSERT INTO semantic_failures (
            helpscout_id, subject, theme_tag, customer_message, draft_html,
            final_reply_html, change_type, summary, wrong_claim, correct_claim,
            confidence_score, confidence_status, confidence_provider,
            confidence_trace, evidence_summary
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            helpscout_id,
            subject,
            theme_tag,
            customer_message,
            draft_html,
            final_reply_html,
            change_type,
            summary,
            wrong_claim,
            correct_claim,
            confidence_score,
            confidence_status,
            confidence_provider,
            confidence_trace,
            evidence_summary,
        ),
    )
    conn.commit()
    failure_id = cursor.lastrowid
    conn.close()
    return failure_id


def get_semantic_failures_by_helpscout_id(
    db_path: Optional[str] = None,
    helpscout_id: str = "",
) -> List[Dict[str, Any]]:
    """Fetch semantic failures for a specific HelpScout conversation."""
    if not helpscout_id:
        return []

    conn = get_db(ensure_db_initialized(db_path))
    cursor = conn.execute(
        """
        SELECT * FROM semantic_failures
        WHERE helpscout_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (helpscout_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_recent_semantic_failures(
    db_path: Optional[str] = None,
    theme_tag: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Fetch recent semantic failures, optionally filtered by theme."""
    conn = get_db(ensure_db_initialized(db_path))
    try:
        if theme_tag:
            cursor = conn.execute(
                """
                SELECT * FROM semantic_failures
                WHERE theme_tag = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (theme_tag, limit),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM semantic_failures
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
def get_stats(db_path: Optional[str] = None, days: int = 1) -> Dict[str, int]:
    """Get ticket statistics for the last N days."""
    conn = get_db(ensure_db_initialized(db_path))
    since = datetime.now() - timedelta(days=days)

    cursor = conn.execute("""
        SELECT classification, COUNT(*) as count
        FROM tickets
        WHERE created_at >= ?
        GROUP BY classification
    """, (since.isoformat(),))

    stats = {"support": 0, "marketing": 0, "spam": 0, "total": 0}
    for row in cursor.fetchall():
        stats[row["classification"]] = row["count"]
        stats["total"] += row["count"]

    conn.close()
    return stats


def get_extended_stats(db_path: Optional[str] = None, days: int = 7) -> Dict[str, int]:
    """Get extended ticket statistics for the dashboard.

    Returns:
        received: total tickets processed
        auto_closed: marketing + spam (automatically closed)
        drafts_created: support tickets where draft was posted
        manual: support tickets where no draft was posted
    """
    conn = get_db(ensure_db_initialized(db_path))
    since = datetime.now() - timedelta(days=days)

    # Total received
    cursor = conn.execute("""
        SELECT COUNT(*) as count FROM tickets WHERE created_at >= ?
    """, (since.isoformat(),))
    received = cursor.fetchone()["count"]

    # Auto-closed (marketing + spam)
    cursor = conn.execute("""
        SELECT COUNT(*) as count FROM tickets
        WHERE created_at >= ? AND classification IN ('marketing', 'spam')
    """, (since.isoformat(),))
    auto_closed = cursor.fetchone()["count"]

    # Drafts created (support tickets with response_posted = 1)
    cursor = conn.execute("""
        SELECT COUNT(*) as count FROM tickets
        WHERE created_at >= ? AND classification = 'support' AND response_posted = 1
    """, (since.isoformat(),))
    drafts_created = cursor.fetchone()["count"]

    # Manual (support tickets with response_posted = 0)
    cursor = conn.execute("""
        SELECT COUNT(*) as count FROM tickets
        WHERE created_at >= ? AND classification = 'support' AND response_posted = 0
    """, (since.isoformat(),))
    manual = cursor.fetchone()["count"]

    conn.close()
    return {
        "received": received,
        "auto_closed": auto_closed,
        "drafts_created": drafts_created,
        "manual": manual
    }


def get_tickets_by_type(db_path: Optional[str] = None, ticket_type: str = "received", days: int = 7) -> List[Dict[str, Any]]:
    """Get tickets filtered by type for the last N days.

    Types:
        received: all tickets
        auto_closed: marketing + spam
        drafts: support with response_posted = 1
        manual: support with response_posted = 0
    """
    conn = get_db(ensure_db_initialized(db_path))
    since = datetime.now() - timedelta(days=days)

    if ticket_type == "received":
        cursor = conn.execute("""
            SELECT * FROM tickets WHERE created_at >= ?
            ORDER BY created_at DESC
        """, (since.isoformat(),))
    elif ticket_type == "auto_closed":
        cursor = conn.execute("""
            SELECT * FROM tickets
            WHERE created_at >= ? AND classification IN ('marketing', 'spam')
            ORDER BY created_at DESC
        """, (since.isoformat(),))
    elif ticket_type == "drafts":
        cursor = conn.execute("""
            SELECT * FROM tickets
            WHERE created_at >= ? AND classification = 'support' AND response_posted = 1
            ORDER BY created_at DESC
        """, (since.isoformat(),))
    elif ticket_type == "manual":
        cursor = conn.execute("""
            SELECT * FROM tickets
            WHERE created_at >= ? AND classification = 'support' AND response_posted = 0
            ORDER BY created_at DESC
        """, (since.isoformat(),))
    else:
        conn.close()
        return []

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def insert_pending_send(
    *,
    db_path: Optional[str] = None,
    conversation_id: str,
    draft_thread_id: str,
    response_html_sha: str,
    due_at: str,
    last_known_customer_thread_id: Optional[str],
    tags_on_promote: List[str],
    confidence_score: Optional[int],
    confidence_provider: Optional[str],
) -> int:
    """Insert a new scheduled pending_sends row. Returns the new row id."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO pending_sends (
                conversation_id, draft_thread_id, response_html_sha,
                queued_at, due_at, last_known_customer_thread_id,
                tags_on_promote, confidence_score, confidence_provider,
                state, attempts, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'scheduled', 0, ?)
            """,
            (
                conversation_id, draft_thread_id, response_html_sha,
                now, due_at, last_known_customer_thread_id,
                json.dumps(tags_on_promote or []),
                confidence_score, confidence_provider, now,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def _row_to_pending_send(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "draft_thread_id": row["draft_thread_id"],
        "response_html_sha": row["response_html_sha"],
        "queued_at": row["queued_at"],
        "due_at": row["due_at"],
        "last_known_customer_thread_id": row["last_known_customer_thread_id"],
        "tags_on_promote": json.loads(row["tags_on_promote"] or "[]"),
        "confidence_score": row["confidence_score"],
        "confidence_provider": row["confidence_provider"],
        "state": row["state"],
        "cancel_reason": row["cancel_reason"],
        "attempts": row["attempts"],
        "last_error": row["last_error"],
        "published_thread_id": row["published_thread_id"],
        "sent_at": row["sent_at"],
        "delivery_verified_at": row["delivery_verified_at"],
        "delivery_receipt_error": row["delivery_receipt_error"],
        "cleanup_error": row["cleanup_error"],
        "cleanup_updated_at": row["cleanup_updated_at"],
        "updated_at": row["updated_at"],
    }


def fetch_due_pending_sends(
    *,
    db_path: Optional[str] = None,
    now_iso: str,
) -> List[Dict[str, Any]]:
    """Return scheduled rows whose due_at <= now_iso, oldest-due first."""
    conn = get_db(db_path)
    try:
        rows = conn.execute(
            """
            SELECT * FROM pending_sends
            WHERE state = 'scheduled' AND due_at <= ?
            ORDER BY due_at ASC
            """,
            (now_iso,),
        ).fetchall()
        return [_row_to_pending_send(r) for r in rows]
    finally:
        conn.close()


def fetch_scheduled_pending_sends_by_conversation(
    *,
    db_path: Optional[str] = None,
    conversation_id: str,
) -> List[Dict[str, Any]]:
    """Return all scheduled rows for a given conversation."""
    conn = get_db(db_path)
    try:
        rows = conn.execute(
            """
            SELECT * FROM pending_sends
            WHERE conversation_id = ? AND state = 'scheduled'
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
        return [_row_to_pending_send(r) for r in rows]
    finally:
        conn.close()


def update_pending_send_state(
    *,
    db_path: Optional[str] = None,
    row_id: int,
    state: str,
    cancel_reason: Optional[str] = None,
) -> None:
    """Transition a pending_sends row to a terminal state (sent | cancelled | failed)."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db(db_path)
    try:
        conn.execute(
            """
            UPDATE pending_sends
            SET state = ?, cancel_reason = ?, updated_at = ?
            WHERE id = ?
            """,
            (state, cancel_reason, now, row_id),
        )
        conn.commit()
    finally:
        conn.close()


def record_pending_send_delivery(
    *,
    db_path: Optional[str] = None,
    row_id: int,
    published_thread_id: Optional[str] = None,
    sent_at: Optional[str] = None,
    receipt_error: Optional[str] = None,
) -> bool:
    """Fence an acknowledged send from retries, retaining an independently verified receipt.

    A missing readback is a diagnostic on a terminal send, never a failed delivery to retry.
    Old sent rows remain untouched until an actual reconciliation supplies evidence.
    """
    if bool(published_thread_id) != bool(sent_at):
        raise ValueError("A verified delivery needs both the published thread and its timestamp")
    now = datetime.now(timezone.utc).isoformat()
    with closing(get_db(db_path)) as conn:
        if published_thread_id:
            cursor = conn.execute(
                """UPDATE pending_sends SET state='sent', cancel_reason=NULL,
                       published_thread_id=?, sent_at=?, delivery_verified_at=?,
                       cleanup_error=CASE WHEN state='scheduled' THEN 'cleanup_pending' ELSE cleanup_error END,
                       cleanup_updated_at=CASE WHEN state='scheduled' THEN ? ELSE cleanup_updated_at END,
                       delivery_receipt_error=NULL, updated_at=?
                   WHERE id=? AND state IN ('scheduled','sent')
                     AND (published_thread_id IS NULL OR published_thread_id=?)""",
                (str(published_thread_id), sent_at, now, now, now, row_id, str(published_thread_id)),
            )
        else:
            cursor = conn.execute(
                """UPDATE pending_sends SET state='sent', cancel_reason=NULL,
                       cleanup_error=CASE WHEN state='scheduled' THEN 'cleanup_pending' ELSE cleanup_error END,
                       cleanup_updated_at=CASE WHEN state='scheduled' THEN ? ELSE cleanup_updated_at END,
                       delivery_receipt_error=?, updated_at=?
                   WHERE id=? AND state IN ('scheduled','sent') AND delivery_verified_at IS NULL""",
                (now, receipt_error or "awaiting_delivery_readback", now, row_id),
            )
        conn.commit()
        return cursor.rowcount == 1


def fetch_unverified_autosend_receipts(
    *, db_path: Optional[str] = None, before: str, limit: int = 10,
) -> List[Dict[str, Any]]:
    """Bounded readback work for new acknowledged sends; legacy null receipts are not guessed."""
    with closing(get_db(db_path)) as conn:
        rows = conn.execute(
            """SELECT * FROM pending_sends
               WHERE state='sent' AND delivery_verified_at IS NULL
                 AND delivery_receipt_error IS NOT NULL AND updated_at<=?
               ORDER BY updated_at ASC, id ASC LIMIT ?""",
            (before, max(0, min(limit, 100))),
        ).fetchall()
        return [_row_to_pending_send(row) for row in rows]


def record_pending_send_cleanup(*, db_path: Optional[str] = None, row_id: int, error: Optional[str]) -> None:
    """Track post-send housekeeping independently of verified customer delivery."""
    with closing(get_db(db_path)) as conn:
        conn.execute(
            "UPDATE pending_sends SET cleanup_error=?, cleanup_updated_at=? WHERE id=? AND state='sent'",
            (error, datetime.now(timezone.utc).isoformat(), row_id),
        )
        conn.commit()


def fetch_pending_autosend_cleanup(
    *, db_path: Optional[str] = None, before: str, limit: int = 10,
) -> List[Dict[str, Any]]:
    """Retry incomplete cleanup, while preserving explicit operator/ownership holds."""
    with closing(get_db(db_path)) as conn:
        rows = conn.execute(
            """SELECT * FROM pending_sends WHERE state='sent' AND cleanup_error IS NOT NULL
               AND cleanup_updated_at<=? AND cleanup_error NOT IN (
                 'cleanup_draft_modified','cleanup_draft_not_api_owned','cleanup_manual_conversation')
               ORDER BY cleanup_updated_at ASC, id ASC LIMIT ?""",
            (before, max(0, min(limit, 100))),
        ).fetchall()
        return [_row_to_pending_send(row) for row in rows]


def update_pending_send_retry(
    *,
    db_path: Optional[str] = None,
    row_id: int,
    new_due_at: str,
    last_error: str,
) -> None:
    """Bump attempts, update due_at, record last_error. Leaves state as 'scheduled'."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db(db_path)
    try:
        conn.execute(
            """
            UPDATE pending_sends
            SET attempts = attempts + 1,
                due_at = ?,
                last_error = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (new_due_at, last_error, now, row_id),
        )
        conn.commit()
    finally:
        conn.close()


def normalization_recovery_candidates(*, db_path: str, since: str, min_score: int) -> list[dict]:
    """Latest unattempted queue rows with the same still-accepted auto-send audit body."""
    with closing(get_db(db_path)) as conn:
        rows = conn.execute("""
            SELECT p.*, t.draft_response_html AS expected_html
            FROM pending_sends p JOIN tickets t ON t.helpscout_id=p.conversation_id
            WHERE p.queued_at >= ? AND p.attempts=0
              AND (p.state='scheduled' OR (p.state='cancelled' AND p.cancel_reason='draft_modified'))
              AND p.id=(SELECT MAX(n.id) FROM pending_sends n WHERE n.conversation_id=p.conversation_id)
              AND t.delivery_mode='auto_send' AND t.confidence_status='accept'
              AND t.response_posted=1 AND t.confidence_score>=?
            ORDER BY p.id DESC LIMIT 100
        """, (since, min_score)).fetchall()
        return [{**_row_to_pending_send(row), "expected_html": row["expected_html"]} for row in rows]


def repin_normalized_send(*, db_path: str, row: dict, persisted_hash: str, min_score: int) -> bool:
    """CAS a normalization-only hold; never replace timing, audit text or a newer obligation."""
    with closing(get_db(db_path)) as conn:
        result = conn.execute("""
            UPDATE pending_sends SET response_html_sha=?, state='scheduled', cancel_reason=NULL, updated_at=?
            WHERE id=? AND state=? AND response_html_sha=? AND attempts=0
              AND (state='scheduled' OR (state='cancelled' AND cancel_reason='draft_modified'))
              AND id=(SELECT MAX(n.id) FROM pending_sends n WHERE n.conversation_id=pending_sends.conversation_id)
              AND EXISTS(SELECT 1 FROM tickets t WHERE t.helpscout_id=pending_sends.conversation_id
                AND t.draft_response_html=? AND t.delivery_mode='auto_send' AND t.confidence_status='accept'
                AND t.response_posted=1 AND t.confidence_score>=?)
        """, (persisted_hash, datetime.now(timezone.utc).isoformat(), row["id"], row["state"],
              row["response_html_sha"], row["expected_html"], min_score))
        conn.commit()
        return result.rowcount == 1
