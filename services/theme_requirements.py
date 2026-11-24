import json
import re
from pathlib import Path
from typing import Any

from services.company_pack import product_settings
from services.themes import LT_THEMES, VALID_SLUGS

_PRODUCTS = product_settings()
THEME_REQUIREMENTS_PATH = _PRODUCTS["theme_requirements_path"]
CLASSIC_UPDATE_KEYWORDS = tuple(_PRODUCTS["classic_update_keywords"])
REQUIREMENT_SNIPPET_PATTERNS = tuple(_PRODUCTS["requirement_snippet_patterns"])


def theme_requirements_url(theme_slug: str) -> str:
    return f'{_PRODUCTS["docs_url"]}{theme_slug}/{THEME_REQUIREMENTS_PATH}'


def is_classic_theme_slug(theme_slug: str | None) -> bool:
    slug = (theme_slug or "").strip().lower()
    return bool(slug and slug in VALID_SLUGS and slug not in LT_THEMES and slug not in _PRODUCTS["skip_classic"])


def should_include_classic_requirements_context(
    *,
    theme_tag: str | None,
    subject: str = "",
    message: str = "",
    internal_note: str = "",
    conversation_history: str = "",
) -> bool:
    if not is_classic_theme_slug(theme_tag):
        return False

    haystack = " ".join(
        part
        for part in (subject, message, internal_note, conversation_history)
        if part
    ).lower()
    if not haystack:
        return False

    return any(keyword in haystack for keyword in CLASSIC_UPDATE_KEYWORDS)


def load_theme_requirements_doc(
    *,
    theme_tag: str | None,
    docs_dir: Path,
) -> dict[str, Any] | None:
    if not is_classic_theme_slug(theme_tag):
        return None

    slug = (theme_tag or "").strip().lower()
    json_file = docs_dir / f"{slug}.json"
    if not json_file.exists():
        return None

    target_url = theme_requirements_url(slug).rstrip("/")
    try:
        docs = json.loads(json_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_url = str(doc.get("url") or doc.get("id") or "").rstrip("/")
        if doc_url == target_url:
            return doc
    return None


def _normalize_requirement_snippet(snippet: str) -> str:
    snippet = re.sub(r"\s+", " ", snippet).strip()
    snippet = re.sub(r"\s+\.", ".", snippet)
    if snippet and snippet[-1] not in ".!?":
        snippet = f"{snippet}."
    return snippet


def _extract_requirement_snippets(content: str) -> list[str]:
    snippets: list[str] = []
    seen: set[str] = set()

    for pattern in REQUIREMENT_SNIPPET_PATTERNS:
        match = re.search(pattern, content, flags=re.IGNORECASE)
        if not match:
            continue

        snippet = _normalize_requirement_snippet(match.group(0))
        key = snippet.lower()
        if snippet and key not in seen:
            snippets.append(snippet)
            seen.add(key)

    return snippets


def format_theme_requirements_context(doc: dict[str, Any] | None) -> str:
    if not doc:
        return ""

    title = str(doc.get("title") or "Theme Requirements (classic)").strip()
    url = str(doc.get("url") or doc.get("id") or "").strip()
    content = re.sub(r"\s+", " ", str(doc.get("content") or "")).strip()
    if not content:
        return ""

    snippets = _extract_requirement_snippets(content)
    requirements = ""
    if snippets:
        requirements = "\nKey requirements:\n" + "\n".join(f"- {snippet}" for snippet in snippets)

    return f"**{title}**\nURL: {url}{requirements}\n\nExcerpt:\n{content[:1200]}..."


def get_classic_theme_requirements_context(
    *,
    theme_tag: str | None,
    subject: str = "",
    message: str = "",
    internal_note: str = "",
    conversation_history: str = "",
    docs_dir: Path,
) -> str:
    if not should_include_classic_requirements_context(
        theme_tag=theme_tag,
        subject=subject,
        message=message,
        internal_note=internal_note,
        conversation_history=conversation_history,
    ):
        return ""

    return format_theme_requirements_context(
        load_theme_requirements_doc(theme_tag=theme_tag, docs_dir=docs_dir)
    )
