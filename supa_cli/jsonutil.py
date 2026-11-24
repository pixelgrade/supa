"""Strict, bounded-shape JSON helpers for untrusted API data."""

from __future__ import annotations

import json
import math
from typing import Any


MAX_JSON_DEPTH = 100


class StrictJSONError(ValueError):
    """A value cannot be safely accepted or emitted as interoperable JSON."""


def _reject_constant(_value: str) -> None:
    raise StrictJSONError("Non-finite JSON numbers are not supported")


def _validate_string(value: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise StrictJSONError("Lone Unicode surrogates are not supported")


def validate_json_value(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise StrictJSONError("JSON nesting exceeds the client limit")
        if current is None or isinstance(current, (bool, int)):
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise StrictJSONError("Non-finite JSON numbers are not supported")
            continue
        if isinstance(current, str):
            _validate_string(current)
            continue
        if isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
            continue
        if isinstance(current, dict):
            for key, item in current.items():
                if not isinstance(key, str):
                    raise StrictJSONError("JSON object keys must be strings")
                _validate_string(key)
                stack.append((item, depth + 1))
            continue
        raise StrictJSONError("Unsupported JSON value")


def loads_object(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data, parse_constant=_reject_constant)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
        StrictJSONError,
        ValueError,
    ) as exc:
        raise StrictJSONError("Invalid JSON") from exc
    validate_json_value(value)
    if not isinstance(value, dict):
        raise StrictJSONError("The JSON root must be an object")
    return value


def dumps(
    value: Any,
    *,
    indent: int | None = None,
    sort_keys: bool = False,
) -> str:
    validate_json_value(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=sort_keys,
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise StrictJSONError("Value cannot be serialized safely") from exc
