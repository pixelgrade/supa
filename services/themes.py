"""Detect products from tags and ticket text using the installation catalogue."""

import re
from typing import Optional

from services.company_pack import product_settings

_PRODUCTS = product_settings()
THEMES = _PRODUCTS["aliases"]
LT_THEMES = set(_PRODUCTS["modern"])

VALID_SLUGS = set(THEMES.values())


def detect_theme(
    tags: list[str] = None,
    subject: str = "",
    message: str = ""
) -> Optional[str]:
    """
    Detect the theme from ticket data.

    Priority:
    1. Tags (most reliable - often pre-tagged)
    2. Subject line
    3. Message body

    Returns the theme slug or None if no theme detected.
    """
    tags = tags or []

    # Prefer explicit LT theme tags when both classic and LT variants are present.
    for tag in tags:
        tag_lower = tag.lower().strip()
        mapped = THEMES.get(tag_lower, tag_lower if tag_lower in VALID_SLUGS else None)
        if mapped in LT_THEMES:
            return mapped

    # 1. Check tags first (most reliable)
    for tag in tags:
        tag_lower = tag.lower().strip()
        if tag_lower in THEMES:
            return THEMES[tag_lower]
        # Check if tag is already a valid slug
        if tag_lower in VALID_SLUGS:
            return tag_lower

    # 2. Check subject
    theme = _find_theme_in_text(subject)
    if theme:
        return theme

    # 3. Check message body
    theme = _find_theme_in_text(message)
    if theme:
        return theme

    return None


def _find_theme_in_text(text: str) -> Optional[str]:
    """Find a theme mention in text."""
    if not text:
        return None

    text_lower = text.lower()

    # Match the most specific name first.
    sorted_themes = sorted(THEMES.keys(), key=len, reverse=True)

    for theme_name in sorted_themes:
        # Use word boundaries to avoid false matches
        # Match whole product names.
        pattern = r'\b' + re.escape(theme_name) + r'\b'
        if re.search(pattern, text_lower):
            return THEMES[theme_name]

    return None


def get_docs_url(theme_slug: str) -> str:
    """Get the documentation URL for a theme."""
    if not theme_slug:
        return _PRODUCTS["docs_url"]
    return f'{_PRODUCTS["docs_url"]}{theme_slug}/'


def is_lt_theme(theme_slug: str) -> bool:
    """Check if a theme is an LT (block) theme."""
    return theme_slug in LT_THEMES
