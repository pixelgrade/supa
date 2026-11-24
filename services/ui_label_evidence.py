"""Locate UI labels in source translations, block attributes and configuration. Missing labels supply evidence for review. Skip core labels and controls the customer explicitly named."""

import logging
import re
from pathlib import Path
from typing import Iterable, Optional

from services.live_site_evidence import _normalize_for_css as _normalize_html

logger = logging.getLogger(__name__)

# Standard WordPress admin controls can exist independently of product source.
WP_CORE_UI_LABELS = {
    "add new",
    "additional css",
    "advanced",
    "appearance",
    "block",
    "classic editor",
    "colors",
    "columns",
    "cover",
    "customize",
    "customizer",
    "dashboard",
    "delete",
    "editor",
    "featured image",
    "fonts",
    "footer",
    "format",
    "gallery",
    "general",
    "header",
    "home",
    "image",
    "installed plugins",
    "list view",
    "media",
    "menus",
    "pages",
    "paragraph",
    "patterns",
    "permalinks",
    "plugins",
    "posts",
    "publish",
    "reading",
    "replace",
    "save changes",
    "save draft",
    "screen options",
    "settings",
    "site editor",
    "site identity",
    "styles",
    "template parts",
    "tools",
    "update",
    "users",
    "video",
    "widgets",
    "writing",
}

_QUOTED_RE = re.compile(r"[\"“‘']([^\"”’'\n]{2,40})[\"”’']")
_ARROW_SPLIT_RE = re.compile(r"\s*(?:→|->)\s*")
# Bold is how support replies mark UI labels ("open <strong>List View</strong>").
_BOLD_LABEL_RE = re.compile(r"<(?:strong|b)>([^<]{2,60})</(?:strong|b)>", re.IGNORECASE)
# A breadcrumb segment often trails prose ("That goes in Appearance"); the label is the
# trailing run of Title-Case words, with connector words allowed inside the run.
_TITLE_RUN_RE = re.compile(
    r"((?:[A-Z][\w'&-]*)(?:\s+(?:[A-Z][\w'&-]*|and|of|on|to|the|for|&)){0,4})\s*$"
)
_CONNECTORS = {"and", "of", "on", "to", "the", "for", "&"}
# Capitalized words that start prose, not labels ("Go to Appearance", "What is actually
# happening"). A run beginning with one of these is a sentence fragment.
_PROSE_STARTERS = {
    "go", "open", "click", "find", "choose", "select", "visit", "press", "use", "head",
    "navigate", "what", "why", "how", "when", "where", "that", "this", "one", "two",
    "then", "now", "please", "if", "once", "next",
}


def _looks_like_ui_label(candidate: str) -> bool:
    label = candidate.strip()
    if not 2 <= len(label) <= 40:
        return False
    words = label.split()
    if not 1 <= len(words) <= 4:
        return False
    # UI labels are title-ish; error messages and prose quotes ("page not found") are not.
    if not label[0].isupper():
        return False
    # Selectors, URLs, and code fragments quote fine but are not labels.
    if re.search(r"[<>{}();=/\\@#]|https?:", label):
        return False
    if words[0].lower() in _PROSE_STARTERS:
        return False
    return True


def _trailing_title_run(segment: str) -> str:
    """The Title-Case run a breadcrumb segment ends with — the label, minus surrounding prose."""
    segment = re.sub(r"[.,;:!]+\s*$", "", segment.strip())
    match = _TITLE_RUN_RE.search(segment)
    if not match:
        return ""
    words = match.group(1).split()
    while words and words[-1].lower() in _CONNECTORS:
        words.pop()
    while words and words[0].lower() in _PROSE_STARTERS:
        words.pop(0)
        while words and words[0].lower() in _CONNECTORS:
            words.pop(0)
    return " ".join(words)


def extract_ui_label_claims(draft_html: str) -> list[str]:
    """Labels the draft tells the customer to find: quoted, bolded, or in breadcrumb paths."""
    raw = draft_html or ""
    text = _normalize_html(raw)
    candidates: list[str] = []

    for match in _QUOTED_RE.finditer(text):
        candidates.append(match.group(1))

    for match in _BOLD_LABEL_RE.finditer(raw):
        content = match.group(1).strip().strip("\"“”‘’'")
        if "→" in content or "->" in content:
            continue  # breadcrumb inside bold — handled below on the normalized text
        candidates.append(content)

    # Breadcrumb paths: Appearance → Editor → Header Layout. Every segment is a claim, but
    # the first one usually trails prose ("That goes in Appearance") — keep only the label.
    for line in text.splitlines():
        if "→" not in line and "->" not in line:
            continue
        for segment in _ARROW_SPLIT_RE.split(line):
            # Both ends can carry prose ("That goes in Appearance", "Header and click it");
            # a clean label is simply its own trailing Title-Case run.
            candidates.append(_trailing_title_run(segment))

    labels: list[str] = []
    for candidate in candidates:
        label = candidate.strip()
        if not _looks_like_ui_label(label):
            continue
        if label.lower() in WP_CORE_UI_LABELS:
            continue
        labels.append(label)
    return list(dict.fromkeys(labels))


_PLACEHOLDERS = ("%s", "%1$s")


def _locate_composed_label(repo_path, tag: str, label: str, search) -> Optional[str]:
    """Find a label the source builds with sprintf, e.g. "Parent Topic" from
    sprintf( 'Parent %s', $singular ) and $singular = 'Topic'. Both halves must be
    in the same file, so two unrelated strings never add up to a control that does not exist."""
    words = label.split()
    for start in range(len(words)):
        for end in range(start + 1, len(words) + 1):
            if end - start == len(words):
                continue  # the whole label is the literal search that already failed
            part = " ".join(words[start:end])
            for placeholder in _PLACEHOLDERS:
                template = " ".join([*words[:start], placeholder, *words[end:]])
                try:
                    hits = search(repo_path, template, limit_per_term=5)
                except Exception:
                    continue
                for hit in hits:
                    match = re.search(r"([^\s:]+):(\d+):", hit)
                    if not match:
                        continue
                    file_path = Path(repo_path) / match.group(1).replace(f"{repo_path}/", "").lstrip("./")
                    try:
                        source = file_path.read_text(encoding="utf-8", errors="replace").lower()
                    except OSError:
                        continue
                    if f"'{part.lower()}'" in source or f'"{part.lower()}"' in source:
                        relative = file_path.relative_to(repo_path) if file_path.is_relative_to(repo_path) else file_path
                        return f"{tag} {relative}:{match.group(2)} (composed: '{template}' + '{part}')"
    return None


def verify_ui_labels_in_repos(
    labels: Iterable[str], repo_tags: list[str]
) -> tuple[dict[str, str], list[str]]:
    """Locate each label in the given repos, as `repo file:line` for the first repo that has it.

    Also returns which repos were actually searched: a repo we could not check out proves
    nothing about a label, and "not found" may only be claimed against repos that were read.
    """
    from services.repo_evidence import _search_repo, ensure_repo_checkout

    remaining = list(labels)
    confirmed: dict[str, str] = {}
    searched: list[str] = []
    for tag in repo_tags:
        if not remaining:
            break
        try:
            repo_path = ensure_repo_checkout(tag)
        except Exception:
            logger.info("ui-label check: no checkout for %s", tag, exc_info=True)
            continue
        searched.append(tag)
        still_missing: list[str] = []
        for label in remaining:
            location = None
            try:
                hits = _search_repo(repo_path, label, limit_per_term=1)
            except Exception:
                hits = []
            for hit in hits:
                match = re.search(r"([^\s:]+):(\d+):", hit)
                if match:
                    path = match.group(1).replace(f"{repo_path}/", "").lstrip("./")
                    location = f"{tag} {path}:{match.group(2)}"
                    break
            if not location:
                location = _locate_composed_label(repo_path, tag, label, _search_repo)
            if location:
                confirmed[label] = location
            else:
                still_missing.append(label)
        remaining = still_missing
    return confirmed, searched


def verify_draft_ui_labels(
    *,
    draft_html: str,
    product_tag: str,
    context_text: str = "",
) -> str:
    """Evidence block for the draft's UI-label claims, or "" when the draft makes none.

    Labels the customer used themselves are skipped — if they named the control, it exists on
    their screen regardless of whose product renders it.
    """
    from services.repo_catalog import is_product_repo
    from services.repo_evidence import companion_repo_targets, repo_url

    if not is_product_repo(product_tag):
        return ""

    labels = extract_ui_label_claims(draft_html)
    if context_text:
        lowered_context = context_text.lower()
        labels = [label for label in labels if label.lower() not in lowered_context]
    if not labels:
        return ""

    repo_tags = [product_tag, *companion_repo_targets(product_tag, text=context_text)]
    confirmed, searched = verify_ui_labels_in_repos(labels, repo_tags)
    # "Not found" is only a claim about repos that were actually read.
    missing = [label for label in labels if label not in confirmed] if searched else []

    lines: list[str] = []
    if confirmed:
        # Shaped as a `Repo evidence` block on purpose: verified labels should count as real
        # backing wherever the pipeline checks for it.
        lines.extend(
            [
                f"Repo evidence ({product_tag}):",
                f"Source repo: {repo_url(product_tag)}",
                "Verified UI labels used by the draft:",
            ]
        )
        lines.extend(f'"{label}" — {confirmed[label]}' for label in sorted(confirmed))

    if missing:
        # Deliberately NOT shaped as repo evidence: a warning must never satisfy an
        # evidence check. The evaluator decides — third-party UI is legitimate, but a
        # theme panel that is not in the theme's repo usually is not.
        if lines:
            lines.append("")
        lines.append(
            f"UI-label check ({product_tag}): these labels quoted in the draft were NOT found "
            f"in {', '.join(searched)}: " + ", ".join(f'"{label}"' for label in missing) + ". "
            "If a label names a theme or plugin control, verify it exists before instructing "
            "the customer; a nonexistent control costs a full customer round-trip."
        )

    return "\n".join(lines)
