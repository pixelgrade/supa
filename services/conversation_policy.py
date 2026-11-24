"""Pure routing guard shared by webhook and explicitly requested ticket tasks."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from services import extensions
from services.helpscout import is_supa_ignored


def manual_route(
    payload: Mapping[str, Any],
    *,
    ignored_check: Callable[[Mapping[str, Any]], bool] = is_supa_ignored,
) -> tuple[str | None, int | None]:
    """Return the manual route and case ID, without performing any side effect."""
    route = extensions.manual_route(payload)
    if route is not None:
        return route
    if ignored_check(payload):
        return "supa_ignore", None
    return None, None
