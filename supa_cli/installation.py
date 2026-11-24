"""Installation vocabulary shared by the standalone client and server."""
import os
from pathlib import Path
import tomllib


def client_defaults():
    try:
        with Path(__file__).with_name("installation.toml").open("rb") as stream:
            return tomllib.load(stream)
    except FileNotFoundError:
        return {}


def waiting_on_choices(environ=None, *, standalone=False):
    environment = os.environ if environ is None else environ
    alias = environment.get("SUPA_OWNER_ALIAS")
    if alias is None and standalone:
        alias = client_defaults().get("SUPA_OWNER_ALIAS")
    return tuple(dict.fromkeys(v for v in ("customer", "owner", alias, "blocked", "nobody") if v))
