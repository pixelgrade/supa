import sqlite3

from db.database import init_db, log_ticket
from services.support_attempts import register_attempt


def test_same_customer_message_cannot_reopen_automatic_delivery(tmp_path):
    path = str(tmp_path / "audit.db")
    init_db(path)
    thread = {"id": 123, "createdAt": "2026-11-17T08:00:00Z"}
    assert register_attempt(path, 99, thread) == 1
    assert register_attempt(path, 99, thread) == 2
    assert register_attempt(path, 99, {**thread, "id": 124}) == 1
    assert register_attempt(path, 100, thread) == 1


def test_predeployment_held_run_is_not_a_fresh_attempt(tmp_path):
    path = str(tmp_path / "audit.db")
    init_db(path)
    log_ticket(db_path=path, helpscout_id="99", classification="support", delivery_mode="draft")
    with sqlite3.connect(path) as con:
        con.execute("UPDATE ticket_runs SET created_at='2026-11-17 08:01:00'")
    assert register_attempt(path, 99, {"id": 123, "createdAt": "2026-11-17T08:00:00Z"}) == 2
    assert register_attempt(path, 99, {"id": 124, "createdAt": "2026-11-17T08:02:00Z"}) == 1


def test_unknown_customer_identity_never_gets_first_pass_authority(tmp_path):
    path = str(tmp_path / "audit.db")
    init_db(path)
    assert register_attempt(path, 99, None) is None
    assert register_attempt(path, 99, {"body": "hello"}) is None


def test_new_message_is_not_old_run_that_finished_after_it_arrived(tmp_path):
    path = str(tmp_path / "audit.db")
    init_db(path)
    log_ticket(db_path=path, helpscout_id="99", classification="support", delivery_mode="draft",
        confidence_trace='[{"customer_thread_id":123,"message_attempt":1}]')
    with sqlite3.connect(path) as con:
        con.execute("UPDATE ticket_runs SET created_at='2026-11-17 08:03:00'")
    assert register_attempt(path, 99, {"id": 124, "createdAt": "2026-11-17T08:02:00Z"}) == 1
