"""Configure the mailbox API, OAuth identity and human links as one origin."""

import os
import re
from urllib.parse import urlsplit

from services import installation

VENDOR_WEB = "https://secure.helpscout.net"
VENDOR_API = "https://api.helpscout.net/v2"
VENDOR_INBOX = installation.VENDOR_INBOX_URL


def native_origin(environ=None):
    values = os.environ if environ is None else environ
    value = values.get("SUPA_MAILBOX_ORIGIN", "")
    if not value:
        return None
    # The selector is an operator-provided HTTPS origin, never a URL taken from a ticket.
    try:
        url = urlsplit(value)
        if (
            not re.fullmatch(r"https://[a-zA-Z0-9.-]+(?::[0-9]+)?/?", value)
            or not url.hostname
            or url.port == 0
        ):
            raise ValueError
    except ValueError as exc:
        raise ValueError("Invalid mailbox origin") from exc
    return value.rstrip("/")


def web_origin():
    return native_origin() or VENDOR_WEB


def inbox_url():
    return native_origin() or VENDOR_INBOX


def conversation_url(conversation_id, *, thread_id=None, trailing_slash=False):
    result = f"{web_origin()}/conversation/{conversation_id}"
    if trailing_slash:
        result += "/"
    if thread_id is not None:
        result += f"#thread-{thread_id}"
    return result


def matches_citation(value, conversation_id, thread_id):
    path = f"/conversation/{conversation_id}#thread-{thread_id}"
    return value in (web_origin() + path, VENDOR_WEB + path)


def has_credentials():
    prefix = "SUPA_MAILBOX" if native_origin() else "HELPSCOUT"
    return bool(
        os.getenv(prefix + "_CLIENT_ID") and os.getenv(prefix + "_CLIENT_SECRET")
    )


def connection(client_id, client_secret, environ=None):
    values = os.environ if environ is None else environ
    origin = native_origin(values)
    if origin is None:
        return VENDOR_API, client_id, client_secret
    native_id = values.get("SUPA_MAILBOX_CLIENT_ID")
    native_secret = values.get("SUPA_MAILBOX_CLIENT_SECRET")
    if not native_id or not native_secret:
        raise ValueError("Native mailbox credentials are missing")
    return origin + "/v2", native_id, native_secret
