"""List active+pending HelpScout conversations as compact JSON for the
process-open-tickets workflow. Each row carries the latest non-system thread
type so the workflow can avoid re-sending to already-answered tickets."""
import asyncio
import json
import os
import sys

# Threads that represent the support back-and-forth (exclude notes/lineitems).
_CUSTOMER_TYPES = {"customer"}
_AGENT_TYPES = {"message"}
_IGNORED_TYPES = {"note", "lineitem", "phone", "chat", "forwardparent", "forwardchild"}


def _latest_thread_type(threads):
    relevant = [
        t for t in (threads or [])
        if t.get("type") not in _IGNORED_TYPES and t.get("createdAt")
    ]
    if not relevant:
        return "unknown"
    # On identical timestamps, resolve to the agent (answered) side so the
    # workflow never re-sends to a ticket that was already answered.
    newest = max(
        relevant,
        key=lambda t: (t.get("createdAt", ""), 1 if t.get("type") in _AGENT_TYPES else 0),
    )
    ttype = newest.get("type")
    if ttype in _CUSTOMER_TYPES:
        return "customer"
    if ttype in _AGENT_TYPES:
        return "agent"
    return "unknown"


def summarize_conversations(conversations):
    """Pure transform: HelpScout conversation dicts -> compact workflow rows."""
    rows = []
    for convo in conversations or []:
        threads = (convo.get("_embedded") or {}).get("threads") or []
        customer = convo.get("primaryCustomer") or {}
        rows.append({
            "id": convo.get("id"),
            "subject": convo.get("subject") or "",
            "latest_thread_type": _latest_thread_type(threads),
            "customer_id": customer.get("id"),
            "existing_tags": [t.get("tag") for t in (convo.get("tags") or []) if t.get("tag")],
        })
    return rows


async def _fetch():
    from dotenv import load_dotenv
    from services.helpscout import HelpScoutClient

    load_dotenv()
    client = HelpScoutClient(
        client_id=os.getenv("HELPSCOUT_CLIENT_ID", ""),
        client_secret=os.getenv("HELPSCOUT_CLIENT_SECRET", ""),
    )
    limit = int(os.getenv("OPEN_TICKETS_LIMIT", "100"))
    return await client.get_recent_conversations(limit=limit, statuses=("active", "pending"))


def main():
    conversations = asyncio.run(_fetch())
    json.dump(summarize_conversations(conversations), sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
