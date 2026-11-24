import sqlite3

from services.reprocess import (
    can_auto_send,
    customer_waiting,
    default_since,
    find_casualties,
)


def _make_db(path, rows):
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE tickets (id INTEGER PRIMARY KEY AUTOINCREMENT, helpscout_id TEXT, "
        "subject TEXT, created_at TEXT, confidence_score INT, confidence_status TEXT, "
        "delivery_mode TEXT, risk_flags TEXT, confidence_summary TEXT)"
    )
    con.executemany(
        "INSERT INTO tickets (helpscout_id, subject, created_at, confidence_score, "
        "confidence_status, delivery_mode, risk_flags, confidence_summary) "
        "VALUES (:helpscout_id,:subject,:created_at,:confidence_score,:confidence_status,"
        ":delivery_mode,:risk_flags,:confidence_summary)",
        rows,
    )
    con.commit()
    con.close()


def _row(**kw):
    base = dict(helpscout_id="1", subject="S", created_at="2026-06-24 12:00:00",
                confidence_score=0, confidence_status="manual_review",
                delivery_mode="draft", risk_flags="[]", confidence_summary="")
    base.update(kw)
    return base


def test_find_casualties_matches_provider_failures(tmp_path):
    db = str(tmp_path / "t.db")
    _make_db(db, [
        _row(helpscout_id="100", risk_flags='["investigator_failure","evaluator_failure"]'),
        _row(helpscout_id="101", confidence_summary="Confidence evaluator failed: Codex CLI failed (exit 1)"),
        _row(helpscout_id="102", confidence_summary="You've hit your limit"),
    ])
    found = {c["helpscout_id"] for c in find_casualties(db, "2026-06-24 00:00:00")}
    assert found == {"100", "101", "102"}


def test_find_casualties_excludes_legit_manual_review(tmp_path):
    db = str(tmp_path / "t.db")
    _make_db(db, [
        # marketing newsletter — 0/manual_review but NO provider-failure signature
        _row(helpscout_id="200", risk_flags='["operator@woodland.example subscribed to a mailing list"]',
             confidence_summary="No draft was generated. This is a marketing newsletter."),
    ])
    assert find_casualties(db, "2026-06-24 00:00:00") == []


def test_find_casualties_keeps_only_latest_row_per_ticket(tmp_path):
    db = str(tmp_path / "t.db")
    _make_db(db, [
        # earlier failure, then a later successful re-run -> should NOT be a casualty anymore
        _row(helpscout_id="300", created_at="2026-06-24 11:00:00",
             risk_flags='["investigator_failure"]'),
        _row(helpscout_id="300", created_at="2026-06-24 15:00:00",
             confidence_score=9, confidence_status="accept", delivery_mode="auto_send",
             risk_flags="[]", confidence_summary="clean"),
    ])
    assert find_casualties(db, "2026-06-24 00:00:00") == []


def test_find_casualties_respects_since(tmp_path):
    db = str(tmp_path / "t.db")
    _make_db(db, [
        _row(helpscout_id="400", created_at="2026-06-20 10:00:00",
             risk_flags='["investigator_failure"]'),
    ])
    assert find_casualties(db, "2026-06-24 00:00:00") == []
    assert len(find_casualties(db, "2026-06-19 00:00:00")) == 1


def test_customer_waiting():
    waiting = [
        {"type": "note", "createdAt": "2026-06-24T13:00:00Z"},
        {"type": "customer", "createdAt": "2026-06-24T12:00:00Z", "createdBy": {"type": "customer"}},
    ]
    assert customer_waiting(waiting) is True
    answered = waiting + [
        {"type": "message", "state": "published", "createdAt": "2026-06-24T14:00:00Z",
         "createdBy": {"type": "user"}},
    ]
    assert customer_waiting(answered) is False


def test_can_auto_send_gate():
    assert can_auto_send(confidence_score=9, delivery_mode="auto_send",
                         confidence_status="accept", hard_blockers=[], waiting=True) is True
    # not waiting -> no
    assert can_auto_send(confidence_score=10, delivery_mode="auto_send",
                         confidence_status="accept", hard_blockers=[], waiting=False) is False
    # blocker -> no
    assert can_auto_send(confidence_score=9, delivery_mode="auto_send",
                         confidence_status="accept", hard_blockers=["x"], waiting=True) is False
    # Eight is the minimum send score.
    assert can_auto_send(confidence_score=8, delivery_mode="auto_send",
                         confidence_status="accept", hard_blockers=[], waiting=True) is True
    # below threshold -> no
    assert can_auto_send(confidence_score=7, delivery_mode="auto_send",
                         confidence_status="accept", hard_blockers=[], waiting=True) is False


def test_default_since_format():
    s = default_since(24)
    assert len(s) == 19 and s[4] == "-" and s[13] == ":"
