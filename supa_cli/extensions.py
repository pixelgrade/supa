"""Extension startup shared by the server and the standalone client wheel."""

from importlib import import_module
import os
from .installation import client_defaults

_loaded: dict[str, None] = {}


def configured(environ=None):
    environment = os.environ if environ is None else environ
    names = environment.get("SUPA_EXTENSIONS", "").split(",")
    return list(dict.fromkeys(name.strip() for name in names if name.strip()))


def load(environ=None):
    for name in configured(environ):
        if name in _loaded:
            continue
        try:
            import_module(name)
        except Exception as exc:
            raise RuntimeError(f"Could not load Supa extension {name!r}") from exc
        _loaded[name] = None


def loaded():
    return list(_loaded)


def reset():
    """Clear the loader's bookkeeping for isolated tests."""
    _loaded.clear()


def client_modules(environ=None):
    """Use an installation default only when the process did not configure extensions."""
    environment = os.environ if environ is None else environ
    if "SUPA_EXTENSIONS" not in environment:
        environment = client_defaults()
    load(environment)
    return [import_module(name) for name in configured(environment)]
