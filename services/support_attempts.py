"""Durable first-pass delivery admission, scoped to one customer message."""

from contextlib import closing
from datetime import datetime, timezone
import json

from db.database import get_db


def register_attempt(db_path, conversation_id, thread):
    """Return the durable attempt number; unknown identity has no send authority.

    Repeated model runs may improve a draft, but cannot sample their way around a
    held first decision. A new customer message gets its own first pass. Old audit
    rows have no thread ID, so conservatively seed from runs since that message.
    """
    if not thread or not str(thread.get("id") or "").isdigit():
        return None
    try:
        created = datetime.fromisoformat(thread["createdAt"].replace("Z", "+00:00"))
        if created.tzinfo is None:
            return None
        created_sql = created.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    with closing(get_db(db_path)) as con, con:
        con.execute("BEGIN IMMEDIATE")
        con.execute("""CREATE TABLE IF NOT EXISTS support_message_attempts (
            conversation_id TEXT NOT NULL, customer_thread_id TEXT NOT NULL,
            attempts INTEGER NOT NULL, PRIMARY KEY (conversation_id, customer_thread_id)
        )""")
        key = (str(conversation_id), str(thread["id"]))
        row = con.execute("SELECT attempts FROM support_message_attempts WHERE conversation_id=? AND customer_thread_id=?", key).fetchone()
        if row:
            attempt = row["attempts"] + 1
            con.execute("UPDATE support_message_attempts SET attempts=? WHERE conversation_id=? AND customer_thread_id=?", (attempt, *key))
        else:
            attempt = 1
            for row in con.execute("""SELECT confidence_trace FROM ticket_runs WHERE helpscout_id=?
                AND classification='support' AND created_at >= ?""", (key[0], created_sql)):
                try:
                    trace = json.loads(row[0] or "[]")
                    bound = [str(t["customer_thread_id"]) for t in trace if isinstance(t, dict) and t.get("customer_thread_id")]
                except (ValueError, TypeError):
                    bound = []
                if not bound or key[1] in bound:
                    attempt += 1
            con.execute("INSERT INTO support_message_attempts VALUES (?, ?, ?)", (*key, attempt))
        return attempt
