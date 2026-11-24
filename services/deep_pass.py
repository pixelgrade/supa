"""Collect bounded, read-only evidence from customer messages, material images and permitted product checkouts for the deep drafting pass."""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from agents.draft_confidence import DraftConfidenceResult
from agents.support_agent import SupportInvestigationResult
from services.helpscout import strip_html
from services.themes import VALID_SLUGS
from services.repo_catalog import repo_for_theme
from services.repo_evidence import (
    companion_repo_targets,
    ensure_repo_checkout,
    resolve_repo_target,
)

from services.company_pack import product_settings, wording

logger = logging.getLogger(__name__)

DEEP_PASS_ROOT = Path(tempfile.gettempdir()) / "supa-deep-pass"

# Commercial decisions and failed evaluations require review.
SKIP_BLOCKERS = {"concierge_human_review", "money_decision", "investigator_failure", "evaluator_failure"}

# Every deep pass gathers these, whatever the judge listed: they are cheap and they are what
# the stuck tickets lacked (orders, similar answers, the KB beyond the top three hits).
BASELINE_ACTIONS = ("refresh_customer_context", "expand_past_ticket_search", "expand_kb_search")

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
from services import installation

_OWN_DOMAINS = (*installation.OWN_DOMAINS, "helpscout.net")


@dataclass
class DeepEvidence:
    """Context the processor gathered for the deep pass; empty strings mean nothing new."""
    knowledge_context: str = ""
    past_tickets_context: str = ""
    customer_context: str = ""
    image_paths: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# (evaluation, workdir) -> DeepEvidence. Runs in the pipeline's worker thread.
DeepEvidenceCollector = Callable[[DraftConfidenceResult, Path], DeepEvidence]


def deep_pass_skip_reason(
    *,
    investigation: SupportInvestigationResult,
    evaluation: DraftConfidenceResult,
    hard_blockers: list[str],
    target_score: int,
) -> Optional[str]:
    """Why the deep pass should not run for this best result, or None when it should."""
    if evaluation.score >= target_score and not hard_blockers:
        return "first pass already reached the bar"
    skip = SKIP_BLOCKERS.intersection(hard_blockers or []) | SKIP_BLOCKERS.intersection(
        list(investigation.risk_flags or []) + list(evaluation.risk_flags or [])
    )
    if skip:
        return f"blocked by {', '.join(sorted(skip))}"
    if not strip_html(investigation.reply_html or "").strip():
        return "first pass produced no draft"
    return None


def deep_actions(evaluation: DraftConfidenceResult) -> list[str]:
    """The judge's requested actions plus the baseline, in a stable order."""
    return list(dict.fromkeys([*(evaluation.investigation_actions or []), *BASELINE_ACTIONS]))


def other_customer_emails(message: str, customer_email: str, *, limit: int = 3) -> list[str]:
    """Addresses the customer names in their message besides the one they wrote from."""
    own = (customer_email or "").strip().lower()
    found: list[str] = []
    for match in _EMAIL_RE.findall(message or ""):
        email = match.strip(".").lower()
        if email == own or email.endswith(_OWN_DOMAINS) or email in found:
            continue
        found.append(email)
    return found[:limit]


def named_theme_slugs(text: str, *, exclude: str = "", limit: int = 2) -> list[str]:
    """Return known product slugs named in the supplied text."""
    words = set(re.findall(r"[a-z0-9-]+", (text or "").lower()))
    return sorted(slug for slug in VALID_SLUGS if slug in words and slug != exclude)[:limit]


def repo_dirs_for(*, theme_tag: str, subject: str, message: str, extra_text: str = "") -> list[Path]:
    """Checkouts for the deep pass: the product's repo, any other theme the customer names,
    then product repos named in the ticket, the draft or the judge's critique, the theme's
    TGMPA-required plugins and its standing companions. Only catalog product repos
    (services/repo_catalog.py); failures are skipped."""
    target = resolve_repo_target(theme_tag=theme_tag, subject=subject, message=message)
    if not target:
        return []
    dirs: list[Path] = []
    try:
        target_dir: Optional[Path] = ensure_repo_checkout(target)
        dirs.append(target_dir)
    except Exception as exc:  # a missing checkout narrows the evidence, never stops the pass
        logger.warning("Deep pass: no checkout for %s: %s", target, exc)
        target_dir = None
    ticket_text = f"{subject}\n{message}"
    # A product name may be an ordinary word. Check only customer-authored text.
    named_themes = named_theme_slugs(ticket_text, exclude=(theme_tag or "").strip().lower())
    slugs = [
        *(repo for repo in map(repo_for_theme, named_themes) if repo),
        *companion_repo_targets(target, checkout=target_dir, text=f"{ticket_text}\n{extra_text}"),
    ]
    for slug in dict.fromkeys(slug for slug in slugs if slug != target):
        try:
            dirs.append(ensure_repo_checkout(slug))
        except Exception as exc:
            logger.warning("Deep pass: no checkout for %s: %s", slug, exc)
    return dirs


_CITATION_RE = re.compile(
    r"(?P<path>[\w./-]+\.(?:php|js|jsx|ts|tsx|json|css|scss|html|twig|txt|md|xml))(?::|#L)"
    r"(?P<lines>\d+(?:\s*[-\u2013]\s*L?\d+)?(?:\s*,\s*L?\d+(?:\s*[-\u2013]\s*L?\d+)?)*)"
)
_EXCERPT_CONTEXT = 3       # lines around a single cited line
_EXCERPT_MAX_LINES = 60    # per citation
_EXCERPTS_MAX_CHARS = 12000
SOURCE_EXCERPTS_HEADING = (
    wording("source_excerpts")
)


def _resolve_cited_file(raw_path: str, repo_dirs: list[Path]) -> Optional[Path]:
    """The cited file inside one of the checkouts, or None. Never leaves the checkouts."""
    roots = [root.resolve() for root in repo_dirs]
    candidates: list[Path] = []
    path = Path(raw_path)
    if path.is_absolute():
        candidates.append(path)
    else:
        # Repository-qualified basenames also resolve beneath the selected checkout.
        # It may live below inc/, while another product has a file with the same name.
        named_roots = [root for root in roots if path.parts[0] == root.name]
        if named_roots:
            roots = named_roots
            path = Path(*path.parts[1:])
        candidates.extend(root / path for root in roots)
    resolved_candidates: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file() and any(resolved.is_relative_to(root) for root in roots):
            resolved_candidates.add(resolved)
    if resolved_candidates:
        return next(iter(resolved_candidates)) if len(resolved_candidates) == 1 else None
    if not path.is_absolute() and len(path.parts) == 1:
        # Unique within the selected checkout(s), with the same containment check as
        # direct paths. A symlink outside a checkout can never become source evidence.
        matches = {m.resolve() for root in roots for m in root.rglob(path.name)
                   if m.is_file() and m.resolve().is_relative_to(root)}
        if len(matches) == 1:
            return next(iter(matches))
    return None


def _cited_ranges(spec: str) -> list[tuple[int, int]]:
    ranges = []
    for part in spec.replace("L", "").split(","):
        bounds = [int(b) for b in re.split(r"\s*[-\u2013]\s*", part.strip()) if b]
        if not bounds:
            continue
        start, end = bounds[0], bounds[-1]
        if start == end:
            start, end = start - _EXCERPT_CONTEXT, end + _EXCERPT_CONTEXT
        start = max(1, start)
        # Keep later facts in a long cited function. Four bounded chunks per range plus
        # the existing total character budget prevent an enormous range from expanding work.
        end = max(start, min(end, start + 4 * _EXCERPT_MAX_LINES - 1))
        ranges.extend((chunk, min(end, chunk + _EXCERPT_MAX_LINES - 1))
                      for chunk in range(start, end + 1, _EXCERPT_MAX_LINES))
    return ranges


def matching_source_citations(request: str, sources: list[str]) -> list[str]:
    """Retrieve existing citations for a judge's named missing file or specific path topic.

    This is evidence retrieval only: no keyword overlap decides whether a reply may send.
    Every returned citation must still resolve inside the admitted checkouts and be re-read.
    """
    requested = request.lower()
    words = set(re.findall(r"[a-z]{4,}", requested))
    mentioned_paths = re.findall(
        r"[\w./-]+\.(?:php|js|jsx|ts|tsx|json|css|scss|html|twig|txt|md|xml)\b", requested)
    generic = {*product_settings()["generic_source_words"], "blocks", "packages", "library", "block", "editor", "filters",
               "controls", "settings", "index", "source", "repo", "with", "code", "class", "admin"}
    found = []
    for index, source in enumerate(sources):
        for match in _CITATION_RE.finditer(source):
            path = match.group("path").lower()
            specific = set(re.findall(r"[a-z]{4,}", path)) - generic
            # "logo/index.js" must not select every unrelated index.js first.
            # Prefer the explicit suffix, then distinctive path topics such as spacing.
            explicit = max((len(p.split("/")) for p in mentioned_paths
                            if path == p or path.endswith("/" + p)), default=0)
            overlap = len(specific & words)
            if explicit or overlap:
                found.append((explicit, overlap, -index, source))
                break
    return list(dict.fromkeys(item[3] for item in sorted(found, reverse=True)))[:8]


def cited_source_excerpts(text: str, repo_dirs: list[Path], *, existing_context: str = "") -> str:
    """Verbatim lines for every file:line the deep pass cites inside the checkouts.

    The judge sees only what Supa hands it; without these lines a draft the deep pass verified
    in code still reads as an unsupported claim. Supa reads the lines itself, so the judge
    compares the draft with the source, not with the agent's description of it."""
    if not text or not repo_dirs:
        return ""
    blocks: list[str] = []
    seen: set[tuple[str, int, int]] = set()
    total = 0
    for match in _CITATION_RE.finditer(text):
        raw_path = match.group("path")
        # Keep the repository attached to citations qualified by a commit.
        # attribution; a bare functions.php exists in nearly every theme/plugin checkout.
        prefix = re.search(r"([\w-]+)(?:@[0-9a-fA-F]{7,40})?\s+$", text[max(0, match.start() - 100):match.start()])
        if prefix and prefix.group(1) in {root.name for root in repo_dirs}:
            raw_path = f"{prefix.group(1)}/{raw_path}"
        path = _resolve_cited_file(raw_path, repo_dirs)
        if path is None:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        root = next(r for r in (d.resolve() for d in repo_dirs) if path.is_relative_to(r))
        label = f"{root.name}/{path.relative_to(root)}"
        for start, end in _cited_ranges(match.group("lines")):
            key = (label, start, end)
            if key in seen or start > len(lines):
                continue
            seen.add(key)
            body = "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, min(end, len(lines)) + 1))
            block = f"{label}:{start}-{min(end, len(lines))}\n{body}"
            if block.rstrip() in existing_context:
                continue
            if total + len(block) > _EXCERPTS_MAX_CHARS:
                continue
            blocks.append(block)
            total += len(block)
    return _excerpts_section(blocks)


def _excerpts_section(blocks: list[str]) -> str:
    if not blocks:
        return ""
    return (
        SOURCE_EXCERPTS_HEADING + "\n\n" + "\n\n".join(blocks)
    )


def new_workdir() -> Path:
    DEEP_PASS_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=DEEP_PASS_ROOT))


def remove_workdir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
