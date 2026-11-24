#!/usr/bin/env python3
"""Reprocess specific tickets - classify and close if non-support."""

import asyncio
import sys
import os
import uuid

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from services import extensions
from services.helpscout import HelpScoutClient
from services.conversation_locks import conversation_lease
from agents.classifier import classify_ticket
import httpx


async def find_and_reprocess_tickets(ticket_numbers: list[int]):
    """Find tickets by number and reprocess them."""

    client = HelpScoutClient(
        client_id=os.getenv("HELPSCOUT_CLIENT_ID", ""),
        client_secret=os.getenv("HELPSCOUT_CLIENT_SECRET", "")
    )

    for number in ticket_numbers:
        print(f"\n--- Looking for ticket #{number} ---")

        try:
            # Search by ticket number
            results = await client.search_conversations(f"number:{number}", status="all", limit=1)

            if not results:
                print(f"Not found")
                continue

            conv = results[0]
            conv_id = conv.get("id")
            subject = conv.get("subject", "")
            print(f"Found: {conv_id} - {subject[:50]}...")

            # Fetch full conversation for message content
            full_conv = await client.get_conversation(conv_id)
            threads = full_conv.get("_embedded", {}).get("threads", [])
            message = ""
            for thread in threads:
                if thread.get("type") == "customer":
                    message = thread.get("body", "")
                    break

            # Classify
            classification = await classify_ticket(subject, message)
            print(f"Classification: {classification}")

            # Close if non-support
            if classification in ["admin", "marketing", "spam"]:
                owner = f"reprocess-script:{conv_id}:{uuid.uuid4().hex}"
                async with conversation_lease(None, conv_id, owner=owner) as lease:
                    # Classification can take time; refresh tags after acquiring the
                    # shared writer lease instead of applying a stale snapshot.
                    full_conv = await client.get_conversation(conv_id)
                    existing_tags = [t.get("tag", "") for t in full_conv.get("tags", [])]
                    tags_to_add = ["supa-processed", classification]
                    all_tags = list(set(existing_tags + tags_to_add))

                    closed = await lease.run_guarded(
                        client.close_conversation, conv_id
                    )
                    tagged = await lease.run_guarded(
                        client.add_tags, conv_id, all_tags, existing_tags
                    )

                print(f"Closed: {closed}, Tagged: {tagged}")
            else:
                print(f"Skipped (classified as {classification}; the processor handles it)")

        except Exception as e:
            print(f"Error: {e}")

    await client.close()
    print("\nDone!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/reprocess_tickets.py <ticket-number> [<ticket-number> ...]")
        sys.exit(1)

    extensions.load()
    tickets = [int(ticket) for ticket in sys.argv[1:]]
    asyncio.run(find_and_reprocess_tickets(tickets))
