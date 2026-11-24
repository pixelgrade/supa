"""Read installation settings used by server and scheduled processes."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Standalone server and worker entry points can import these settings before
# the ticket processor loads the environment. Explicit process settings win.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

PACK = os.environ.get("SUPA_PACK") or "packs/woodland"
VENDOR_INBOX_URL = os.environ.get("SUPA_VENDOR_INBOX_URL") or "https://secure.helpscout.net"
OPERATIONS_ORIGIN = os.environ.get("SUPA_OPERATIONS_ORIGIN") or ""
CODEX_DIRECTORY = os.environ.get("SUPA_CODEX_DIRECTORY") or str(Path(__file__).resolve().parents[1])
OWN_DOMAINS = tuple(v.strip() for v in (os.environ.get("SUPA_OWN_DOMAINS") or "").split(",") if v.strip())
CANARY_MAILBOX = int(os.environ.get("SUPA_CANARY_MAILBOX_ID") or 0)
CANARY_CUSTOMER = int(os.environ.get("SUPA_CANARY_CUSTOMER_ID") or 0)
CANARY_EMAIL = os.environ.get("SUPA_CANARY_EMAIL") or ""
INLINE_IMAGE_HOSTS = tuple(v.strip() for v in os.getenv("SUPA_INLINE_IMAGE_HOSTS", "").split(",") if v.strip())
