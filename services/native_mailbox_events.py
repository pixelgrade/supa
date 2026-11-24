"""Durable native callback acceptance and operator reconciliation in the Supa ledger."""
import argparse
from contextlib import closing
import hashlib
import hmac
import json
import os
import re
import sqlite3
import time

from services import extensions
from services.agent_events import append_event_in_transaction
from services.runtime_paths import get_supa_db_path


class DeliveryConflict(ValueError):
    pass


def signature(secret, key, event, raw):
    signed = f"{key}\n{event}\n".encode() + raw
    return hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()


def connection(path):
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS native_mailbox_jobs(
        id INTEGER PRIMARY KEY, event TEXT NOT NULL, conversation INTEGER NOT NULL, digest TEXT NOT NULL,
        payload BLOB, state TEXT NOT NULL DEFAULT 'queued', accepted INTEGER NOT NULL, finished INTEGER, reason TEXT)""")
    db.execute("""CREATE TABLE IF NOT EXISTS native_mailbox_resolutions(
        job INTEGER NOT NULL, attempt INTEGER NOT NULL, action TEXT NOT NULL, evidence TEXT NOT NULL,
        resolved INTEGER NOT NULL, operator_uid INTEGER NOT NULL, PRIMARY KEY(job,attempt))""")
    return db


def accept(path, key, event_type, cid, raw, journal_event):
    digest = hashlib.sha256(raw).hexdigest()
    with closing(connection(path)) as db, db:
        inserted = db.execute("""
            INSERT INTO native_mailbox_jobs(id,event,conversation,digest,payload,accepted)
            VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING
            """, (key, event_type, cid, digest, raw, int(time.time()))).rowcount
        if inserted:
            append_event_in_transaction(db, journal_event)
            return "queued"
        row = db.execute("SELECT * FROM native_mailbox_jobs WHERE id=?", (key,)).fetchone()
        if (row["event"], row["conversation"], row["digest"]) != (event_type, cid, digest):
            raise DeliveryConflict("Conflicting native delivery identity")
        return row["state"]


def claim(path):
    with closing(connection(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM native_mailbox_jobs WHERE state='queued' ORDER BY id LIMIT 1").fetchone()
        if row:
            db.execute("UPDATE native_mailbox_jobs SET state='processing' WHERE id=?", (row["id"],))
        return row


def finish(path, key, state, reason=None):
    if state not in ("done", "review"):
        raise ValueError("Unknown processing outcome")
    with closing(connection(path)) as db, db:
        db.execute("""UPDATE native_mailbox_jobs SET state=?,finished=?,reason=?,
                   payload=CASE WHEN ?='done' THEN NULL ELSE payload END WHERE id=? AND state='processing'""",
                   (state, int(time.time()), reason, state, key))


def recover(path):
    # A crash may follow a successful downstream write. Only a readback can authorize replay.
    with closing(connection(path)) as db, db:
        db.execute("""UPDATE native_mailbox_jobs SET state='review',reason='interrupted_during_processing'
                      WHERE state='processing'""")


def counts(path):
    with closing(connection(path)) as db:
        return dict(db.execute("SELECT state,count(*) FROM native_mailbox_jobs GROUP BY state"))


def review_list(path):
    with closing(connection(path)) as db:
        rows = db.execute("""SELECT id,event,conversation,digest,reason,
            1+(SELECT count(*) FROM native_mailbox_resolutions WHERE job=j.id) AS attempt
            FROM native_mailbox_jobs j WHERE state='review' ORDER BY id""")
        return [dict(row) for row in rows]


def reconcile(path, key, digest, attempt, action, evidence):
    if action not in ("retry", "finish") or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,127}", evidence):
        raise ValueError("Provide a readback receipt identifier and a valid action")
    with closing(connection(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        job = db.execute("SELECT state,digest FROM native_mailbox_jobs WHERE id=?", (key,)).fetchone()
        if not job or job["digest"] != digest:
            raise DeliveryConflict("Delivery or digest mismatch")
        previous = db.execute("SELECT action,evidence FROM native_mailbox_resolutions WHERE job=? AND attempt=?",
                              (key, attempt)).fetchone()
        if previous:
            if tuple(previous) == (action, evidence):
                return job["state"]
            raise DeliveryConflict("This attempt already has a different resolution")
        expected = db.execute("SELECT count(*)+1 FROM native_mailbox_resolutions WHERE job=?", (key,)).fetchone()[0]
        if job["state"] != "review" or attempt != expected:
            raise DeliveryConflict("Delivery is not held for that attempt")
        db.execute("INSERT INTO native_mailbox_resolutions VALUES(?,?,?,?,?,?)",
                   (key, attempt, action, evidence, int(time.time()), os.getuid()))
        state = "queued" if action == "retry" else "done"
        db.execute("""UPDATE native_mailbox_jobs SET state=?, reason=NULL,
                   finished=CASE WHEN ?='done' THEN ? END, payload=CASE WHEN ?='done' THEN NULL ELSE payload END
                   WHERE id=?""", (state, state, int(time.time()), state, key))
        return state


if __name__ == "__main__":
    extensions.load()
    parser = argparse.ArgumentParser(description="Resolve held callbacks after reading the conversation and send receipts")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    for action in ("retry", "finish"):
        command = sub.add_parser(action)
        command.add_argument("id", type=int)
        command.add_argument("--digest", required=True)
        command.add_argument("--attempt", required=True, type=int)
        command.add_argument("--readback-receipt", required=True)
    args = parser.parse_args()
    try:
        if args.action == "list":
            print(json.dumps(review_list(get_supa_db_path())))
        else:
            print(reconcile(get_supa_db_path(), args.id, args.digest, args.attempt, args.action, args.readback_receipt))
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
