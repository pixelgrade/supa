import json
import sqlite3
from typing import Optional, get_type_hints

import pytest

from agents.semantic_review import SemanticReviewResult
from db.database import (
    get_db,
    get_recent_semantic_failures,
    get_ticket_logs_by_helpscout_ids,
    get_ticket_log_by_helpscout_id,
    get_ticket_runs_by_helpscout_id,
    get_extended_stats,
    get_semantic_failures_by_helpscout_id,
    get_recent_tickets,
    get_stats,
    init_db,
    insert_pending_send,
    fetch_due_pending_sends,
    fetch_scheduled_pending_sends_by_conversation,
    update_pending_send_state,
    update_pending_send_retry,
    log_ticket,
    log_semantic_failure,
)
from db.models import TicketLog
from services.semantic_failure_learning import learn_from_sent_reply


def test_init_db_creates_tables(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    db = get_db(db_path)
    cursor = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets'")
    assert cursor.fetchone() is not None
    db.close()


def test_init_db_creates_additive_agent_run_tables_and_indexes(tmp_path):
    db_path = str(tmp_path / "agent-runs.db")
    init_db(db_path)
    init_db(db_path)

    db = get_db(db_path)
    try:
        tables = {
            row["name"]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"agent_runs", "agent_run_events"} <= tables

        run_columns = {
            row["name"]: row
            for row in db.execute("PRAGMA table_info(agent_runs)").fetchall()
        }
        assert set(run_columns) == {
            "run_id",
            "idempotency_key",
            "token_id",
            "token_name",
            "task_name",
            "conversation_id",
            "status",
            "arguments_json",
            "scope_snapshot_json",
            "delivery_ceiling",
            "result_json",
            "error_json",
            "created_at",
            "updated_at",
            "started_at",
            "completed_at",
        }
        assert run_columns["run_id"]["pk"] == 1
        assert all(
            run_columns[name]["notnull"] == 1
            for name in (
                "run_id",
                "token_id",
                "token_name",
                "task_name",
                "conversation_id",
                "status",
                "arguments_json",
                "scope_snapshot_json",
                "created_at",
                "updated_at",
            )
        )

        event_columns = {
            row["name"]: row
            for row in db.execute("PRAGMA table_info(agent_run_events)").fetchall()
        }
        assert set(event_columns) == {
            "id",
            "run_id",
            "sequence",
            "stage",
            "progress_json",
            "created_at",
        }
        assert event_columns["id"]["pk"] == 1
        assert all(
            event_columns[name]["notnull"] == 1
            for name in ("run_id", "sequence", "stage", "progress_json", "created_at")
        )

        expected_indexes = {
            "idx_agent_runs_status_created": ["status", "created_at"],
            "idx_agent_runs_conversation_id": ["conversation_id"],
            "idx_agent_runs_token_id": ["token_id"],
            "idx_agent_run_events_run_sequence": ["run_id", "sequence"],
        }
        for index_name, columns in expected_indexes.items():
            index = db.execute(
                "SELECT * FROM pragma_index_list(?) WHERE name = ?",
                ("agent_run_events" if "run_events" in index_name else "agent_runs", index_name),
            ).fetchone()
            assert index is not None
            assert [
                row["name"]
                for row in db.execute(f"PRAGMA index_info({index_name})").fetchall()
            ] == columns
        event_index = db.execute(
            "SELECT * FROM pragma_index_list('agent_run_events') "
            "WHERE name = 'idx_agent_run_events_run_sequence'"
        ).fetchone()
        assert event_index["unique"] == 1
    finally:
        db.close()

    init_db(db_path)


def test_init_db_creates_agent_event_tables(tmp_path):
    db_path = str(tmp_path / "agent-events.db")
    init_db(db_path)

    db = get_db(db_path)
    try:
        tables = {
            row["name"]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"agent_events", "agent_event_consumers"}.issubset(tables)

        event_columns = {
            row["name"]: row
            for row in db.execute("PRAGMA table_info(agent_events)").fetchall()
        }
        assert set(event_columns) == {
            "id",
            "event_type",
            "conversation_id",
            "thread_id",
            "run_id",
            "source",
            "actor_type",
            "occurred_at",
            "received_at",
            "dedupe_key",
            "summary_json",
        }
        assert event_columns["id"]["pk"] == 1
        assert all(
            event_columns[column]["notnull"] == 1
            for column in (
                "event_type",
                "source",
                "occurred_at",
                "received_at",
                "dedupe_key",
                "summary_json",
            )
        )

        event_indexes = {
            row["name"]: row
            for row in db.execute("PRAGMA index_list(agent_events)").fetchall()
        }
        expected_event_indexes = {
            "idx_agent_events_id": (["id"], False),
            "idx_agent_events_dedupe_key": (["dedupe_key"], True),
            "idx_agent_events_conversation_id": (["conversation_id"], False),
            "idx_agent_events_received_at": (["received_at"], False),
        }
        assert expected_event_indexes.keys() <= event_indexes.keys()
        for index_name, (expected_columns, expected_unique) in (
            expected_event_indexes.items()
        ):
            assert event_indexes[index_name]["unique"] == int(expected_unique)
            actual_columns = [
                row["name"]
                for row in db.execute(f"PRAGMA index_info({index_name})").fetchall()
            ]
            assert actual_columns == expected_columns

        consumer_columns = {
            row["name"]: row
            for row in db.execute(
                "PRAGMA table_info(agent_event_consumers)"
            ).fetchall()
        }
        assert set(consumer_columns) == {
            "id",
            "token_id",
            "consumer_name",
            "acknowledged_event_id",
            "leased_event_id",
            "lease_expires_at",
            "lease_token_hash",
            "last_ack_event_id",
            "last_ack_token_hash",
            "start_mode",
            "created_at",
            "updated_at",
        }
        assert consumer_columns["id"]["pk"] == 1
        assert all(
            consumer_columns[column]["notnull"] == 1
            for column in (
                "token_id",
                "consumer_name",
                "acknowledged_event_id",
                "start_mode",
                "created_at",
                "updated_at",
            )
        )

        consumer_indexes = {
            row["name"]: row
            for row in db.execute(
                "PRAGMA index_list(agent_event_consumers)"
            ).fetchall()
        }
        expected_consumer_indexes = {
            "idx_agent_event_consumers_lease_expiry": (
                ["lease_expires_at"],
                False,
            ),
            "idx_agent_event_consumers_identity": (
                ["token_id", "consumer_name"],
                True,
            ),
        }
        assert expected_consumer_indexes.keys() <= consumer_indexes.keys()
        for index_name, (expected_columns, expected_unique) in (
            expected_consumer_indexes.items()
        ):
            assert consumer_indexes[index_name]["unique"] == int(expected_unique)
            actual_columns = [
                row["name"]
                for row in db.execute(f"PRAGMA index_info({index_name})").fetchall()
            ]
            assert actual_columns == expected_columns

        first_event_id = db.execute(
            """
            INSERT INTO agent_events (
                event_type, conversation_id, thread_id, run_id, source,
                actor_type, occurred_at, received_at, dedupe_key, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "helpscout.customer.replied",
                "603",
                "thread-1",
                None,
                "helpscout",
                "customer",
                "2026-09-24T01:02:03Z",
                "2026-09-24T01:02:04Z",
                "event:first",
                '{"subject":"Menu issue"}',
            ),
        ).lastrowid
        db.commit()
        db.execute("DELETE FROM agent_events WHERE id = ?", (first_event_id,))
        db.commit()

        second_event_id = db.execute(
            """
            INSERT INTO agent_events (
                event_type, conversation_id, thread_id, run_id, source,
                actor_type, occurred_at, received_at, dedupe_key, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "supa.run.completed",
                "603",
                None,
                "run-1",
                "supa",
                "system",
                "2026-09-24T02:03:04Z",
                "2026-09-24T02:03:05Z",
                "event:second",
                '{"status":"succeeded"}',
            ),
        ).lastrowid
        assert second_event_id > first_event_id
        db.commit()

        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """
                INSERT INTO agent_events (
                    event_type, source, occurred_at, received_at,
                    dedupe_key, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "supa.run.completed",
                    "supa",
                    "2026-09-24T02:03:04Z",
                    "2026-09-24T02:03:06Z",
                    "event:second",
                    "{}",
                ),
            )
        db.rollback()

        db.execute(
            """
            INSERT INTO agent_event_consumers (
                token_id, consumer_name, acknowledged_event_id,
                leased_event_id, lease_expires_at, lease_token_hash,
                last_ack_event_id, last_ack_token_hash, start_mode,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "token-a",
                "review-desk",
                second_event_id - 1,
                second_event_id,
                "2026-09-24T02:04:05Z",
                "a" * 64,
                second_event_id - 1,
                "b" * 64,
                "oldest",
                "2026-09-24T02:03:05Z",
                "2026-09-24T02:03:06Z",
            ),
        )
        db.commit()

        consumer = db.execute(
            """
            SELECT acknowledged_event_id, leased_event_id, lease_expires_at,
                   lease_token_hash, last_ack_event_id, last_ack_token_hash,
                   start_mode, created_at, updated_at
            FROM agent_event_consumers
            WHERE token_id = ? AND consumer_name = ?
            """,
            ("token-a", "review-desk"),
        ).fetchone()
        assert tuple(consumer) == (
            second_event_id - 1,
            second_event_id,
            "2026-09-24T02:04:05Z",
            "a" * 64,
            second_event_id - 1,
            "b" * 64,
            "oldest",
            "2026-09-24T02:03:05Z",
            "2026-09-24T02:03:06Z",
        )

        db.executemany(
            """
            INSERT INTO agent_event_consumers (
                token_id, consumer_name, acknowledged_event_id, start_mode
            ) VALUES (?, ?, ?, ?)
            """,
            (
                ("token-a", "build-agent", 0, "now"),
                ("token-b", "review-desk", 0, "now"),
            ),
        )
        db.commit()

        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """
                INSERT INTO agent_event_consumers (
                    token_id, consumer_name, acknowledged_event_id, start_mode
                ) VALUES (?, ?, ?, ?)
                """,
                ("token-a", "review-desk", 0, "now"),
            )
        db.rollback()
    finally:
        db.close()

    init_db(db_path)
    init_db(db_path)

    preserved = get_db(db_path)
    try:
        assert preserved.execute("SELECT COUNT(*) FROM agent_events").fetchone()[0] == 1
        assert (
            preserved.execute(
                "SELECT COUNT(*) FROM agent_event_consumers"
            ).fetchone()[0]
            == 3
        )
    finally:
        preserved.close()


def test_init_db_creates_agent_write_tables(tmp_path):
    db_path = str(tmp_path / "agent-writes.db")

    init_db(db_path)
    init_db(db_path)

    db = get_db(db_path)
    try:
        tables = {
            row["name"]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "agent_operations",
            "agent_audit",
            "agent_conversation_locks",
        }.issubset(tables)

        operation_columns = {
            row["name"]: row
            for row in db.execute("PRAGMA table_info(agent_operations)").fetchall()
        }
        assert set(operation_columns) == {
            "id",
            "idempotency_key",
            "token_id",
            "token_name",
            "operation_name",
            "target_id",
            "request_hash",
            "state",
            "request_metadata_json",
            "upstream_thread_id",
            "response_json",
            "error_json",
            "created_at",
            "updated_at",
            "upstream_started_at",
            "completed_at",
        }
        for required in (
            "idempotency_key",
            "token_id",
            "token_name",
            "operation_name",
            "target_id",
            "request_hash",
            "state",
            "request_metadata_json",
            "created_at",
            "updated_at",
        ):
            assert operation_columns[required]["notnull"] == 1

        operation_indexes = {
            row["name"]: row
            for row in db.execute("PRAGMA index_list(agent_operations)").fetchall()
        }
        assert operation_indexes["idx_agent_operations_idempotency_key"]["unique"] == 1
        expected_operation_indexes = {
            "idx_agent_operations_idempotency_key": ["idempotency_key"],
            "idx_agent_operations_token_created": ["token_id", "created_at"],
            "idx_agent_operations_state_updated": ["state", "updated_at"],
        }
        assert expected_operation_indexes.keys() <= operation_indexes.keys()
        for index_name, expected_columns in expected_operation_indexes.items():
            expected_unique = index_name == "idx_agent_operations_idempotency_key"
            assert operation_indexes[index_name]["unique"] == int(expected_unique)
            actual_columns = [
                row["name"]
                for row in db.execute(f"PRAGMA index_info({index_name})").fetchall()
            ]
            assert actual_columns == expected_columns

        audit_columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(agent_audit)").fetchall()
        }
        assert audit_columns == {
            "id",
            "idempotency_key",
            "token_id",
            "token_name",
            "scope",
            "operation_name",
            "target_id",
            "request_id",
            "request_hash",
            "result",
            "metadata_json",
            "created_at",
        }
        audit_indexes = {
            row["name"]: row
            for row in db.execute("PRAGMA index_list(agent_audit)").fetchall()
        }
        expected_audit_indexes = {
            "idx_agent_audit_target_created": ["target_id", "created_at"],
            "idx_agent_audit_operation_created": [
                "idempotency_key",
                "created_at",
            ],
        }
        assert expected_audit_indexes.keys() <= audit_indexes.keys()
        assert all(row["unique"] == 0 for row in audit_indexes.values())
        for index_name, expected_columns in expected_audit_indexes.items():
            actual_columns = [
                row["name"]
                for row in db.execute(f"PRAGMA index_info({index_name})").fetchall()
            ]
            assert actual_columns == expected_columns

        lock_columns = {
            row["name"]: row
            for row in db.execute(
                "PRAGMA table_info(agent_conversation_locks)"
            ).fetchall()
        }
        assert set(lock_columns) == {
            "conversation_id",
            "owner",
            "acquired_at",
            "expires_at",
        }
        assert lock_columns["conversation_id"]["pk"] == 1
        assert all(
            lock_columns[column]["notnull"] == 1
            for column in ("conversation_id", "owner", "acquired_at", "expires_at")
        )
        lock_indexes = {
            row["name"]: row
            for row in db.execute(
                "PRAGMA index_list(agent_conversation_locks)"
            ).fetchall()
        }
        assert lock_indexes["idx_agent_conversation_locks_expires"]["unique"] == 0
        lock_index_columns = [
            row["name"]
            for row in db.execute(
                "PRAGMA index_info(idx_agent_conversation_locks_expires)"
            ).fetchall()
        ]
        assert lock_index_columns == ["expires_at"]

        operation_values = (
            "same-key",
            "token-a",
            "Agent A",
            "draft.save",
            "603",
            "a" * 64,
            "reserved",
            "{}",
        )
        db.execute(
            """
            INSERT INTO agent_operations (
                idempotency_key, token_id, token_name, operation_name,
                target_id, request_hash, state, request_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            operation_values,
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """
                INSERT INTO agent_operations (
                    idempotency_key, token_id, token_name, operation_name,
                    target_id, request_hash, state, request_metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                operation_values,
            )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """
                INSERT INTO agent_operations (
                    idempotency_key, token_id, token_name, operation_name,
                    target_id, request_hash, state, request_metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "invalid-state-key",
                    "token-a",
                    "Agent A",
                    "draft.save",
                    "603",
                    "b" * 64,
                    "not-a-real-state",
                    "{}",
                ),
            )
    finally:
        db.close()


def test_init_db_agent_write_schema_preserves_existing_rows(tmp_path):
    db_path = str(tmp_path / "legacy-agent-writes.db")
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        CREATE TABLE tickets (
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
        CREATE INDEX idx_tickets_created ON tickets(created_at);
        CREATE INDEX idx_tickets_classification ON tickets(classification);

        CREATE TABLE ticket_runs (
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
        CREATE INDEX idx_ticket_runs_helpscout ON ticket_runs(helpscout_id);
        CREATE INDEX idx_ticket_runs_created ON ticket_runs(created_at);

        CREATE TABLE semantic_failures (
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
        CREATE INDEX idx_semantic_failures_created ON semantic_failures(created_at);
        CREATE INDEX idx_semantic_failures_helpscout ON semantic_failures(helpscout_id);
        CREATE INDEX idx_semantic_failures_theme ON semantic_failures(theme_tag);

        CREATE TABLE pending_sends (
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
            state TEXT NOT NULL,
            cancel_reason TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_pending_sends_ready ON pending_sends(state, due_at);
        CREATE INDEX idx_pending_sends_conversation ON pending_sends(conversation_id, state);

        INSERT INTO tickets (
            helpscout_id, subject, customer_name, classification, status,
            response_posted, error_message, trigger, confidence_score,
            confidence_status, confidence_attempts, confidence_summary,
            confidence_trace, confidence_provider, draft_response_html,
            evidence_summary, delivery_mode, investigator_provider,
            evaluator_provider, decision_note, investigation_steps,
            risk_flags, sources, created_at
        ) VALUES (
            'legacy-ticket', 'Preserve ticket', 'Jane', 'support', 'processed',
            1, NULL, 'webhook', 9, 'accepted', 2, 'summary', '{}', 'claude',
            '<p>Draft</p>', 'evidence', 'draft', 'codex', 'claude', 'decision',
            '["step"]', '["risk"]', '["source"]', '2026-09-24 01:02:03'
        );
        INSERT INTO ticket_runs (
            helpscout_id, ticket_id, subject, customer_name, classification,
            status, response_posted, error_message, trigger, confidence_score,
            confidence_status, confidence_attempts, confidence_summary,
            confidence_trace, confidence_provider, draft_response_html,
            evidence_summary, delivery_mode, investigator_provider,
            evaluator_provider, decision_note, investigation_steps,
            risk_flags, sources, created_at
        ) VALUES (
            'legacy-ticket', 1, 'Preserve run', 'Jane', 'support', 'processed',
            1, NULL, 'dashboard', 10, 'accepted', 1, 'run summary', '{}',
            'claude', '<p>Reply</p>', 'run evidence', 'auto_send', 'codex',
            'claude', 'run decision', '["run-step"]', '[]', '["run-source"]',
            '2026-09-24 02:03:04'
        );
        INSERT INTO semantic_failures (
            helpscout_id, subject, theme_tag, customer_message, draft_html,
            final_reply_html, change_type, summary, wrong_claim, correct_claim,
            confidence_score, confidence_status, confidence_provider,
            confidence_trace, evidence_summary, created_at
        ) VALUES (
            'legacy-ticket', 'Preserve semantic', 'field-journal', 'Question',
            '<p>Draft</p>', '<p>Final</p>', 'claim', 'semantic summary',
            'wrong', 'correct', 8, 'accepted', 'claude', '{}', 'evidence',
            '2026-09-24 03:04:05'
        );
        INSERT INTO pending_sends (
            conversation_id, draft_thread_id, response_html_sha, queued_at,
            due_at, last_known_customer_thread_id, tags_on_promote,
            confidence_score, confidence_provider, state, cancel_reason,
            attempts, last_error, updated_at
        ) VALUES (
            'legacy-ticket', 'draft-1', 'hash', '2026-09-24T04:05:06Z',
            '2026-09-25T04:05:06Z', 'customer-1', '["tag"]', 10, 'claude',
            'scheduled', NULL, 1, NULL, '2026-09-24T04:05:06Z'
        );
        """
    )
    db.commit()

    legacy_objects = {
        "tickets",
        "idx_tickets_created",
        "idx_tickets_classification",
        "ticket_runs",
        "idx_ticket_runs_helpscout",
        "idx_ticket_runs_created",
        "semantic_failures",
        "idx_semantic_failures_created",
        "idx_semantic_failures_helpscout",
        "idx_semantic_failures_theme",
        "pending_sends",
        "idx_pending_sends_ready",
        "idx_pending_sends_conversation",
    }
    placeholders = ",".join("?" for _ in legacy_objects)
    schema_before = {
        row["name"]: (row["type"], row["tbl_name"], row["sql"])
        for row in db.execute(
            f"""
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            WHERE name IN ({placeholders})
            """,
            tuple(sorted(legacy_objects)),
        ).fetchall()
    }
    rows_before = {
        table: [tuple(row) for row in db.execute(f"SELECT * FROM {table}").fetchall()]
        for table in ("tickets", "ticket_runs", "semantic_failures", "pending_sends")
    }
    columns_before = {
        table: [row["name"] for row in db.execute(f"PRAGMA table_info({table})")]
        for table in rows_before
    }
    pending_columns_before = [tuple(row) for row in db.execute("PRAGMA table_info(pending_sends)")]
    db.close()

    init_db(db_path)
    init_db(db_path)

    preserved = get_db(db_path)
    try:
        schema_after = {
            row["name"]: (row["type"], row["tbl_name"], row["sql"])
            for row in preserved.execute(
                f"""
                SELECT type, name, tbl_name, sql
                FROM sqlite_master
                WHERE name IN ({placeholders})
                """,
                tuple(sorted(legacy_objects)),
            ).fetchall()
        }
        rows_after = {
            table: [
                tuple(row)
                for row in preserved.execute(
                    f"SELECT {','.join(columns_before[table])} FROM {table}"
                ).fetchall()
            ]
            for table in (
                "tickets",
                "ticket_runs",
                "semantic_failures",
                "pending_sends",
            )
        }
        pending_columns_after = [tuple(row) for row in preserved.execute("PRAGMA table_info(pending_sends)")]
        receipt = preserved.execute(
            "SELECT published_thread_id,sent_at,delivery_verified_at,delivery_receipt_error,cleanup_error,cleanup_updated_at FROM pending_sends"
        ).fetchone()
    finally:
        preserved.close()

    assert {key: value for key, value in schema_after.items() if key != "pending_sends"} == {
        key: value for key, value in schema_before.items() if key != "pending_sends"
    }
    assert pending_columns_after[:len(pending_columns_before)] == pending_columns_before
    assert [(column[1], column[2], column[3], column[4]) for column in pending_columns_after[len(pending_columns_before):]] == [
        ("published_thread_id", "TEXT", 0, None), ("sent_at", "TEXT", 0, None),
        ("delivery_verified_at", "TEXT", 0, None), ("delivery_receipt_error", "TEXT", 0, None),
        ("cleanup_error", "TEXT", 0, None), ("cleanup_updated_at", "TEXT", 0, None),
    ]
    assert tuple(receipt) == (None, None, None, None, None, None)
    assert rows_after == rows_before


def test_agent_audit_is_append_only_at_database_layer(tmp_path):
    db_path = str(tmp_path / "append-only-audit.db")
    init_db(db_path)
    init_db(db_path)

    db = get_db(db_path)
    try:
        audit_values = (
            "audit-key",
            "token-a",
            "Agent A",
            "draft",
            "draft.save",
            "603",
            "req-audit",
            "a" * 64,
            "reserved",
            "{}",
        )
        cursor = db.execute(
            """
            INSERT INTO agent_audit (
                idempotency_key, token_id, token_name, scope, operation_name,
                target_id, request_id, request_hash, result, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            audit_values,
        )
        audit_id = cursor.lastrowid
        db.commit()

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute(
                "UPDATE agent_audit SET result = ? WHERE id = ?",
                ("changed", audit_id),
            )
        db.rollback()

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM agent_audit WHERE id = ?", (audit_id,))
        db.rollback()

        preserved = db.execute(
            "SELECT result FROM agent_audit WHERE id = ?", (audit_id,)
        ).fetchone()
        second_id = db.execute(
            """
            INSERT INTO agent_audit (
                idempotency_key, token_id, token_name, scope, operation_name,
                target_id, request_id, request_hash, result, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "second-audit-key",
                *audit_values[1:],
            ),
        ).lastrowid
        db.commit()
    finally:
        db.close()

    assert preserved["result"] == "reserved"
    assert second_id != audit_id


def test_agent_audit_insert_or_replace_cannot_replace_existing_row(tmp_path):
    db_path = str(tmp_path / "replace-proof-audit.db")
    init_db(db_path)

    db = get_db(db_path)
    try:
        db.execute("PRAGMA recursive_triggers = OFF")
        audit_id = db.execute(
            """
            INSERT INTO agent_audit (
                idempotency_key, token_id, token_name, scope, operation_name,
                target_id, request_id, request_hash, result, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "original-key",
                "token-a",
                "Agent A",
                "draft",
                "draft.save",
                "603",
                "req-original",
                "a" * 64,
                "reserved",
                "{}",
            ),
        ).lastrowid
        db.commit()

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute(
                """
                INSERT OR REPLACE INTO agent_audit (
                    id, idempotency_key, token_id, token_name, scope,
                    operation_name, target_id, request_id, request_hash,
                    result, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    "replacement-key",
                    "token-b",
                    "Agent B",
                    "send",
                    "reply.send",
                    "604",
                    "req-replacement",
                    "b" * 64,
                    "succeeded",
                    '{"replacement":true}',
                ),
            )
        db.rollback()

        preserved = db.execute(
            """
            SELECT idempotency_key, token_id, operation_name, target_id,
                   request_id, request_hash, result, metadata_json
            FROM agent_audit
            WHERE id = ?
            """,
            (audit_id,),
        ).fetchone()
    finally:
        db.close()

    assert tuple(preserved) == (
        "original-key",
        "token-a",
        "draft.save",
        "603",
        "req-original",
        "a" * 64,
        "reserved",
        "{}",
    )


def test_log_ticket(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    log_ticket(
        db_path=db_path,
        helpscout_id="601",
        subject="Test ticket",
        customer_name="Mica",
        classification="support",
        status="processed",
        response_posted=True
    )

    tickets = get_recent_tickets(db_path=db_path, limit=10)
    assert len(tickets) == 1
    assert tickets[0]["helpscout_id"] == "601"


def test_log_ticket_updates_existing_helpscout_id(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    log_ticket(
        db_path=db_path,
        helpscout_id="601",
        subject="Original subject",
        customer_name="Mica",
        classification="support",
        status="processed",
        response_posted=False,
        trigger="webhook",
    )

    log_ticket(
        db_path=db_path,
        helpscout_id="601",
        subject="Updated subject",
        customer_name="Mica",
        classification="support",
        status="processed",
        response_posted=True,
        trigger="note_trigger",
    )

    tickets = get_recent_tickets(db_path=db_path, limit=10)
    assert len(tickets) == 1
    assert tickets[0]["helpscout_id"] == "601"
    assert tickets[0]["subject"] == "Updated subject"
    assert tickets[0]["response_posted"] == 1
    assert tickets[0]["trigger"] == "note_trigger"


def test_get_stats(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    log_ticket(db_path, "1", "S1", "C1", "support", "processed", True)
    log_ticket(db_path, "2", "S2", "C2", "marketing", "processed", False)
    log_ticket(db_path, "3", "S3", "C3", "support", "processed", True)

    stats = get_stats(db_path=db_path)
    assert stats["support"] == 2
    assert stats["marketing"] == 1


def test_get_extended_stats_initializes_missing_schema(tmp_path):
    db_path = str(tmp_path / "missing.db")

    stats = get_extended_stats(db_path=db_path, days=7)

    assert stats == {
        "received": 0,
        "auto_closed": 0,
        "drafts_created": 0,
        "manual": 0,
    }


def test_log_ticket_persists_confidence_fields(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    log_ticket(
        db_path=db_path,
        helpscout_id="123",
        subject="Map issue",
        customer_name="Mica",
        classification="support",
        status="processed",
        response_posted=True,
        confidence_score=8,
        confidence_status="accepted",
        confidence_attempts=2,
        confidence_summary="Missing site version, but technically plausible",
        confidence_trace='{"attempts":[{"score":6},{"score":8}]}',
        confidence_provider="claude",
        draft_response_html="<p>Draft reply</p>",
        evidence_summary="KB: exact known fix",
    )

    ticket = get_recent_tickets(db_path=db_path, limit=1)[0]
    assert ticket["confidence_score"] == 8
    assert ticket["confidence_status"] == "accepted"
    assert ticket["confidence_attempts"] == 2
    assert ticket["confidence_summary"] == "Missing site version, but technically plausible"
    assert ticket["confidence_trace"] == '{"attempts":[{"score":6},{"score":8}]}'
    assert ticket["confidence_provider"] == "claude"
    assert ticket["draft_response_html"] == "<p>Draft reply</p>"
    assert ticket["evidence_summary"] == "KB: exact known fix"


def test_get_ticket_logs_by_helpscout_ids_returns_rows_by_id(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    log_ticket(
        db_path=db_path,
        helpscout_id="123",
        subject="Map issue",
        customer_name="Mica",
        classification="support",
        status="processed",
        response_posted=True,
        confidence_score=8,
        confidence_status="accepted",
    )

    rows = get_ticket_logs_by_helpscout_ids(db_path, ["123", "456"])

    assert set(rows.keys()) == {"123"}
    assert rows["123"]["confidence_score"] == 8


def test_get_ticket_log_by_helpscout_id_returns_latest_row(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    log_ticket(
        db_path=db_path,
        helpscout_id="321",
        subject="Field Journal header issue",
        customer_name="Mica",
        classification="support",
        status="processed",
        response_posted=True,
        confidence_score=9,
        confidence_status="accepted",
        draft_response_html="<p>Draft</p>",
        evidence_summary="Past answer + doc",
    )

    row = get_ticket_log_by_helpscout_id(db_path, "321")

    assert row is not None
    assert row["helpscout_id"] == "321"
    assert row["draft_response_html"] == "<p>Draft</p>"
    assert row["evidence_summary"] == "Past answer + doc"


def test_log_ticket_appends_ticket_run_history(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    log_ticket(
        db_path=db_path,
        helpscout_id="321",
        subject="Initial subject",
        customer_name="Mica",
        classification="support",
        status="processing",
        response_posted=False,
        trigger="webhook",
    )
    log_ticket(
        db_path=db_path,
        helpscout_id="321",
        subject="Final subject",
        customer_name="Mica",
        classification="support",
        status="processed",
        response_posted=True,
        trigger="dashboard",
        delivery_mode="auto_send",
    )

    latest_row = get_ticket_log_by_helpscout_id(db_path, "321")
    runs = get_ticket_runs_by_helpscout_id(db_path, "321")

    assert latest_row is not None
    assert latest_row["subject"] == "Final subject"
    assert latest_row["trigger"] == "dashboard"
    assert len(runs) == 2
    assert runs[0]["status"] == "processed"
    assert runs[0]["trigger"] == "dashboard"
    assert runs[0]["delivery_mode"] == "auto_send"
    assert runs[1]["status"] == "processing"
    assert runs[1]["trigger"] == "webhook"


def test_log_ticket_persists_agentic_audit_fields(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    investigation_steps = ["Checked docs", "Compared plugin code"]
    risk_flags = ["missing_critical_evidence"]
    sources = [
        "https://woodland.example/docs/field-journal",
        "https://github.com/woodland-example/page-tools",
    ]
    log_ticket(
        db_path=db_path,
        helpscout_id="777",
        subject="Field Journal compatibility question",
        customer_name="Mica",
        classification="support",
        status="processed",
        response_posted=True,
        draft_response_html="<p>Draft reply</p>",
        delivery_mode="auto_send",
        investigator_provider="codex",
        evaluator_provider="claude",
        decision_note="Supa decision note:\nMode: auto_send",
        investigation_steps=investigation_steps,
        risk_flags=risk_flags,
        sources=sources,
    )

    row = get_ticket_log_by_helpscout_id(db_path, "777")

    assert row is not None
    assert row["delivery_mode"] == "auto_send"
    assert row["investigator_provider"] == "codex"
    assert row["evaluator_provider"] == "claude"
    assert row["decision_note"] == "Supa decision note:\nMode: auto_send"
    assert json.loads(row["investigation_steps"]) == investigation_steps
    assert json.loads(row["risk_flags"]) == risk_flags
    assert json.loads(row["sources"]) == sources


def test_init_db_migrates_legacy_ticket_schema_and_persists_new_audit_fields(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE tickets (
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

    init_db(db_path)

    migrated = get_db(db_path)
    columns = {row[1] for row in migrated.execute("PRAGMA table_info(tickets)").fetchall()}
    migrated.close()

    assert "delivery_mode" in columns
    assert "investigator_provider" in columns
    assert "evaluator_provider" in columns
    assert "decision_note" in columns
    assert "investigation_steps" in columns
    assert "risk_flags" in columns
    assert "sources" in columns

    log_ticket(
        db_path=db_path,
        helpscout_id="legacy-1",
        subject="Legacy schema ticket",
        customer_name="Mica",
        classification="support",
        status="processed",
        response_posted=True,
        delivery_mode="draft",
        investigator_provider="codex",
        evaluator_provider="claude",
        decision_note="Supa decision note:\nMode: draft",
        investigation_steps=["Checked docs"],
        risk_flags=["manual_review"],
        sources=["https://woodland.example/docs/example"],
    )
    row = get_ticket_log_by_helpscout_id(db_path=db_path, helpscout_id="legacy-1")
    assert row is not None
    assert row["delivery_mode"] == "draft"
    assert row["investigator_provider"] == "codex"
    assert row["evaluator_provider"] == "claude"
    assert json.loads(row["investigation_steps"]) == ["Checked docs"]
    assert json.loads(row["risk_flags"]) == ["manual_review"]
    assert json.loads(row["sources"]) == ["https://woodland.example/docs/example"]


def test_ticket_log_audit_arrays_are_typed_as_structured_lists():
    hints = get_type_hints(TicketLog)
    assert hints["investigation_steps"] == Optional[list[str]]
    assert hints["risk_flags"] == Optional[list[str]]
    assert hints["sources"] == Optional[list[str]]


@pytest.mark.asyncio
async def test_semantic_learning_reads_existing_fields_from_richer_ticket_rows(tmp_path, monkeypatch):
    class FakeHelpScout:
        async def add_note(self, conversation_id, text):
            return True

    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    log_ticket(
        db_path=db_path,
        helpscout_id="602",
        subject="Ordering a journal index",
        customer_name="Mica",
        classification="support",
        status="processed",
        response_posted=True,
        confidence_score=9,
        confidence_status="accepted",
        confidence_summary="Accepted at 9/10",
        draft_response_html="<p>The index sorts entries by title.</p>",
        evidence_summary="Past answer says unsupported.",
        delivery_mode="draft",
        investigator_provider="codex",
        evaluator_provider="claude",
        decision_note="Supa decision note:\nMode: draft",
        investigation_steps=["Checked docs"],
        risk_flags=["manual_review"],
        sources=["https://woodland.example/docs/field-journal"],
    )

    monkeypatch.setattr(
        "services.semantic_failure_learning.review_sent_reply",
        lambda **_kwargs: SemanticReviewResult(
            same_meaning=True,
            change_type="none",
            summary="Only tone changed.",
            provider="claude",
        ),
    )

    outcome = await learn_from_sent_reply(
        db_path=db_path,
        helpscout_client=FakeHelpScout(),
        conversation_id=602,
        subject="Ordering a journal index",
        customer_message="How do I sort an index by page number?",
        final_reply_html="<p>The index sorts entries by page number.</p>",
        theme_tag="field-journal",
    )

    assert outcome.recorded is False
    assert outcome.note_added is False


def test_log_semantic_failure_persists_and_filters_by_helpscout_id(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    log_semantic_failure(
        db_path=db_path,
        helpscout_id="602",
        subject="Ordering a journal index",
        theme_tag="field-journal",
        customer_message="How do I sort an index by page number?",
        draft_html="<p>Use the built-in Overlay Menu setting.</p>",
        final_reply_html="<p>The index sorts entries by page number.</p>",
        change_type="supported_vs_unsupported",
        summary="Draft invented a built-in setting.",
        wrong_claim="The index sorts entries by title.",
        correct_claim="The index sorts entries by page number.",
        confidence_score=9,
        confidence_status="accepted",
        confidence_provider="claude",
        confidence_trace='[{"attempt":1,"score":9}]',
        evidence_summary="Past answer says unsupported.",
    )

    failures = get_semantic_failures_by_helpscout_id(db_path, "602")

    assert len(failures) == 1
    assert failures[0]["change_type"] == "supported_vs_unsupported"
    assert failures[0]["wrong_claim"] == "The index sorts entries by title."
    assert failures[0]["correct_claim"] == "The index sorts entries by page number."


def test_get_recent_semantic_failures_can_filter_by_theme(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    log_semantic_failure(
        db_path=db_path,
        helpscout_id="1",
        subject="Field Journal menu",
        theme_tag="field-journal",
        wrong_claim="Wrong field-journal claim",
        correct_claim="Correct field-journal claim",
    )
    log_semantic_failure(
        db_path=db_path,
        helpscout_id="2",
        subject="Trail Atlas geocoding",
        theme_tag="trail-atlas",
        wrong_claim="Wrong trail-atlas claim",
        correct_claim="Correct trail-atlas claim",
    )

    field_journal_failures = get_recent_semantic_failures(db_path, theme_tag="field-journal")

    assert len(field_journal_failures) == 1
    assert field_journal_failures[0]["helpscout_id"] == "1"


def test_insert_pending_send_roundtrip(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    row_id = insert_pending_send(
        db_path=db_path,
        conversation_id="111",
        draft_thread_id="222",
        response_html_sha="abc123",
        due_at="2026-04-20T10:00:00+00:00",
        last_known_customer_thread_id="55",
        tags_on_promote=["supa-auto-sent"],
        confidence_score=10,
        confidence_provider="claude",
    )
    assert row_id > 0

    rows = fetch_scheduled_pending_sends_by_conversation(db_path=db_path, conversation_id="111")
    assert len(rows) == 1
    assert rows[0]["state"] == "scheduled"
    assert rows[0]["draft_thread_id"] == "222"
    assert rows[0]["response_html_sha"] == "abc123"
    assert rows[0]["tags_on_promote"] == ["supa-auto-sent"]
    assert rows[0]["attempts"] == 0


def test_fetch_due_pending_sends_filters_by_state_and_due_at(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    insert_pending_send(
        db_path=db_path, conversation_id="1", draft_thread_id="a",
        response_html_sha="h", due_at="2020-01-01T00:00:00+00:00",
        last_known_customer_thread_id=None, tags_on_promote=[],
        confidence_score=10, confidence_provider="claude",
    )
    insert_pending_send(
        db_path=db_path, conversation_id="2", draft_thread_id="b",
        response_html_sha="h", due_at="2099-01-01T00:00:00+00:00",
        last_known_customer_thread_id=None, tags_on_promote=[],
        confidence_score=10, confidence_provider="claude",
    )

    due = fetch_due_pending_sends(db_path=db_path, now_iso="2026-04-20T10:00:00+00:00")
    assert [r["conversation_id"] for r in due] == ["1"]


def test_update_pending_send_state_transitions(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    row_id = insert_pending_send(
        db_path=db_path, conversation_id="1", draft_thread_id="a",
        response_html_sha="h", due_at="2020-01-01T00:00:00+00:00",
        last_known_customer_thread_id=None, tags_on_promote=[],
        confidence_score=10, confidence_provider="claude",
    )

    update_pending_send_state(db_path=db_path, row_id=row_id, state="cancelled", cancel_reason="customer_replied")

    rows = fetch_scheduled_pending_sends_by_conversation(db_path=db_path, conversation_id="1")
    assert rows == []


def test_update_pending_send_retry_bumps_attempts(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    row_id = insert_pending_send(
        db_path=db_path, conversation_id="1", draft_thread_id="a",
        response_html_sha="h", due_at="2020-01-01T00:00:00+00:00",
        last_known_customer_thread_id=None, tags_on_promote=[],
        confidence_score=10, confidence_provider="claude",
    )

    update_pending_send_retry(
        db_path=db_path,
        row_id=row_id,
        new_due_at="2026-04-20T11:00:00+00:00",
        last_error="connection reset",
    )

    db = get_db(db_path)
    row = db.execute("SELECT attempts, due_at, last_error FROM pending_sends WHERE id=?", (row_id,)).fetchone()
    db.close()
    assert row["attempts"] == 1
    assert row["due_at"] == "2026-04-20T11:00:00+00:00"
    assert row["last_error"] == "connection reset"
