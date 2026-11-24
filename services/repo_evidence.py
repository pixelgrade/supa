import fcntl
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from services.company_pack import product_settings
from services.helpscout import strip_html
from services.past_tickets import extract_keywords
from services.repo_catalog import (
    RepoNotAllowed,
    catalog,
    companion_repos,
    is_product_repo,
    named_repos,
    product_repo,
    repo_for_theme,
)
from services.themes import VALID_SLUGS


REPO_CACHE_DIR = Path(tempfile.gettempdir()) / "supa-repo-evidence"


def repo_url(product_tag: str) -> str:
    """Clone URL of a product repository; raises RepoNotAllowed for anything else."""
    return product_repo(product_tag).ssh_url


def _is_theme_repo(product_tag: str) -> bool:
    return product_tag in VALID_SLUGS or product_tag in catalog().theme_repos.values()


# TGMPA registrations: `'slug' => 'pixtypes'` inside a theme's required-plugins config.
_TGMPA_SLUG_RE = re.compile(r"""['"]slug['"]\s*=>\s*['"]([a-z0-9-]+)['"]""")


def required_plugin_repos(repo_path: Optional[Path]) -> list[str]:
    """Resolve the required plugin repositories declared by a product's dependency configuration."""
    if repo_path is None or not repo_path.is_dir():
        return []
    found: list[str] = []
    for config in sorted(repo_path.rglob("*required-plugins*.php")):
        if "class-tgm-plugin-activation" in config.name or "vendor" in config.parts:
            continue
        try:
            text = config.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for slug in _TGMPA_SLUG_RE.findall(text):
            if slug not in found and is_product_repo(slug):
                found.append(slug)
    return found


# One primary plus this many others per ticket: each is a shallow clone and a grep.
MAX_EXTRA_REPOS = 5


def companion_repo_targets(product_tag: str, checkout: Optional[Path] = None, text: str = "") -> list[str]:
    """Repos to search besides the primary one, most specific first: product repos the text
    names ("the pix types plugin"), the plugins the theme's TGMPA config requires (when its
    checkout is at hand), then the catalog's standing companions for that product."""
    targets = named_repos(text, exclude=(product_tag,)) if text else []
    if _is_theme_repo(product_tag) and checkout is not None:
        targets.extend(required_plugin_repos(checkout))
    targets.extend(companion_repos(product_tag, is_theme=_is_theme_repo(product_tag)))
    return [name for name in dict.fromkeys(targets) if name != product_tag][:MAX_EXTRA_REPOS]


CAPABILITY_SIGNAL_PATTERNS = product_settings()["capability_signals"]

SEARCH_TERMS = product_settings()["search_terms"]

# Generic support vocabulary. These survive the keyword fallback but match everywhere in any
# WordPress codebase (UI strings, comments, i18n labels), so they manufacture "evidence" that
# looks specific and is not. Dropping them is what lets an empty primary repo read as empty.
GENERIC_SEARCH_STOPWORDS = {
    "click", "clicked", "error", "errors", "found", "help", "install", "installed",
    "issue", "issues", "open", "opens", "page", "pages", "please", "post", "posts",
    "problem", "site", "sites", "thanks", "theme", "themes", "update", "updated",
    "updates", "version", "versions", "website", "wordpress", "work", "working", "works",
}

CODE_TERM_PATTERNS = [
    r"`([^`]+)`",
    r"\badd_filter\(\s*['\"]([^'\"]+)['\"]",
    r"\bapply_filters\(\s*['\"]([^'\"]+)['\"]",
    r"\bdo_action\(\s*['\"]([^'\"]+)['\"]",
    r"\b([a-zA-Z_][a-zA-Z0-9_]*::[a-zA-Z_][a-zA-Z0-9_]*)\b",
    r"\b([a-zA-Z_][a-zA-Z0-9_]*\(\))\b",
]

FEATURE_TERM_PATTERNS = [
    (r"\bwp_query\b", "WP_Query"),
    *product_settings()["feature_terms"],
]


@dataclass
class RepoSourceEvidence:
    """Evidence from one repo, kept separate so every snippet stays attributable."""
    product_tag: str
    repo_url: str
    repo_path: Path
    commit: str = ""
    snippets: list[str] = field(default_factory=list)


@dataclass
class RepoEvidenceResult:
    product_tag: str
    repo_url: str
    repo_path: Path
    commit: str = ""
    terms: list[str] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)
    # Companion-plugin hits, never merged into `snippets` — the theme repo and the plugin repo
    # are different sources and a draft that cites the wrong one is exactly the failure mode
    # this evidence is meant to prevent.
    extra_sources: list[RepoSourceEvidence] = field(default_factory=list)


def resolve_repo_target(*, theme_tag: str = "", subject: str = "", message: str = "") -> Optional[str]:
    """The ticket's primary product repository: its theme's repo, else the first product
    the text names by a catalog alias."""
    normalized_theme = (theme_tag or "").strip().lower()
    theme_repo = repo_for_theme(normalized_theme) if normalized_theme in VALID_SLUGS else None
    if theme_repo:
        return theme_repo
    if normalized_theme and is_product_repo(normalized_theme):
        return normalized_theme
    named = named_repos(f"{subject}\n{message}")
    return named[0] if named else None


def should_use_repo_evidence(*, theme_tag: str = "", subject: str = "", message: str = "", draft_html: str = "") -> bool:
    if not resolve_repo_target(theme_tag=theme_tag, subject=subject, message=message):
        return False

    text = strip_html(f"{subject}\n{message}\n{draft_html}".lower())
    return any(re.search(pattern, text) for pattern in CAPABILITY_SIGNAL_PATTERNS)


def extract_repo_search_terms(*, subject: str = "", message: str = "", draft_html: str = "", max_terms: int = 6) -> list[str]:
    raw_text = strip_html(f"{subject}\n{message}\n{draft_html}")
    text = raw_text.lower()
    terms: list[str] = []

    for pattern in CODE_TERM_PATTERNS:
        for match in re.findall(pattern, raw_text, flags=re.IGNORECASE):
            term = str(match).strip()
            if term:
                terms.append(term)

    for pattern, term in FEATURE_TERM_PATTERNS:
        if re.search(pattern, text):
            terms.append(term)

    for term, patterns in SEARCH_TERMS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            terms.append(term)

    if terms:
        return list(dict.fromkeys(terms))[:max_terms]

    keyword_terms = [
        keyword for keyword in extract_keywords(text, max_keywords=max_terms * 3)
        if len(keyword) >= 4 and keyword.lower() not in GENERIC_SEARCH_STOPWORDS
    ]
    return keyword_terms[:max_terms]


def _valid_repo_checkout(repo_path: Path) -> bool:
    """Reject incomplete or damaged evidence checkouts."""
    if not (repo_path / ".git").exists():
        return False
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--show-toplevel", "HEAD"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        lines = result.stdout.splitlines()
        if len(lines) != 2 or Path(lines[0]).resolve() != repo_path.resolve():
            return False
        tracked = subprocess.run(
            ["git", "-C", str(repo_path), "ls-tree", "-r", "--name-only", "-z", "HEAD"],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout.rstrip("\0").split("\0")
        return bool(tracked[0]) and all(os.path.lexists(repo_path / name) for name in tracked)
    except (subprocess.SubprocessError, OSError):
        return False


def ensure_repo_checkout(product_tag: str, refresh: bool = False) -> Path:
    """Publish validated generations, preserving paths already handed to readers.

    The legacy cache and previous generations are deliberately retained: a deep pass
    can still be reading one after this function returns. Cleanup requires a separate
    quiescent maintenance window, never an evidence lookup.
    """
    clone_url = repo_url(product_tag)
    REPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pointer = REPO_CACHE_DIR / f".{product_tag}-current"
    legacy = REPO_CACHE_DIR / product_tag
    # Each caller opens its own descriptor, serializing both processes and threads.
    # Never unlink lock files: replacing the inode would permit two concurrent owners.
    with (REPO_CACHE_DIR / f".{product_tag}.lock").open("a") as lock:
        deadline = time.monotonic() + 150
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for {product_tag} evidence cache")
                time.sleep(0.05)
        try:
            active = pointer.resolve() if pointer.is_symlink() else legacy
            if not refresh and _valid_repo_checkout(active):
                return active
            generation = Path(tempfile.mkdtemp(prefix=f".{product_tag}-generation-", dir=REPO_CACHE_DIR))
            staged = generation / product_tag
            pending = generation / "current"
            try:
                subprocess.run(
                    ["git", "clone", "--depth=1", clone_url, str(staged)],
                    check=True, capture_output=True, text=True, timeout=120,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                )
                if not _valid_repo_checkout(staged):
                    raise RuntimeError(f"Invalid {product_tag} evidence checkout after clone")
                # Atomic pointer publication; returned paths never follow this mutable pointer.
                pending.symlink_to(staged.resolve())
                os.replace(pending, pointer)
            except BaseException:
                # Only our unpublished generation can be removed, never shared/active paths.
                shutil.rmtree(generation, ignore_errors=True)
                raise
            return staged.resolve()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _parse_grep_blocks(stdout: str, limit_per_term: int) -> list[str]:
    if not stdout.strip():
        return []
    snippets = []
    for block in stdout.strip().split("\n--\n"):
        cleaned = re.sub(r"(?m)^\./", "", block.strip())
        if cleaned:
            snippets.append(cleaned)
        if len(snippets) >= limit_per_term:
            break
    return snippets


# Paths that can never be diagnostic evidence about product behaviour. Without this, generic
# terms ("error", "post", "found") match CI templates, dev docs and minified bundles, and the
# investigator reports confident-looking evidence drawn from AGENTS.md or .github/. Worse, those
# junk hits count as "the primary repo answered", so the companion plugin repos never get tried.
# Matched at any depth: build output is routinely nested (admin/build/, assets/dist/), and an
# excludes list anchored to the repo root silently lets those webpack bundles back in.
# Exclude abandoned copies beside the current source.
NON_SOURCE_DIRS = [".github", "node_modules", "vendor", "bin", "dist", "build", "js-old"]
NON_SOURCE_FILE_GLOBS = [
    "*.md",
    "*.min.js",
    "*.min.css",
    "*.bundle.js",
    "*.map",
    "package-lock.json",
    "composer.lock",
]


def _rg_exclude_globs() -> list[str]:
    globs = [f"!**/{name}/**" for name in NON_SOURCE_DIRS]
    globs.extend(f"!{glob}" for glob in NON_SOURCE_FILE_GLOBS)
    return globs


def _git_grep_exclude_pathspecs() -> list[str]:
    specs = [f":(glob,exclude)**/{name}/**" for name in NON_SOURCE_DIRS]
    specs.extend(f":(glob,exclude)**/{glob}" for glob in NON_SOURCE_FILE_GLOBS)
    # `**/x` does not match a top-level `x` in git's glob pathspecs, so anchor those too.
    specs.extend(f":(glob,exclude){name}/**" for name in NON_SOURCE_DIRS)
    specs.extend(f":(glob,exclude){glob}" for glob in NON_SOURCE_FILE_GLOBS)
    return specs


def _rg_search(repo_path: Path, term: str, limit_per_term: int = 3) -> list[str]:
    globs: list[str] = []
    for glob in _rg_exclude_globs():
        globs.extend(["--glob", glob])
    result = subprocess.run(
        [
            "rg", "-n", "-i", "-m", str(limit_per_term), "-C", "2",
            *globs,
            re.escape(term), ".",
        ],
        check=False, capture_output=True, text=True, cwd=repo_path,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or f"rg failed for term {term}")
    return _parse_grep_blocks(result.stdout, limit_per_term)


def _git_grep_search(repo_path: Path, term: str, limit_per_term: int = 3) -> list[str]:
    """Fallback grep that needs no ripgrep binary. `git grep --no-index` searches
    the working tree like grep, so it works in a checkout (or any directory) as
    long as git is installed."""
    excludes = _git_grep_exclude_pathspecs()
    result = subprocess.run(
        ["git", "grep", "--no-index", "-n", "-i", "-I", "-C", "2", "-e", term, "--", ".", *excludes],
        check=False, capture_output=True, text=True, cwd=repo_path,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or f"git grep failed for term {term}")
    return _parse_grep_blocks(result.stdout, limit_per_term)


def _search_repo(repo_path: Path, term: str, limit_per_term: int = 3) -> list[str]:
    # Prefer ripgrep when a real binary is on PATH; otherwise fall back to
    # git grep (always available — we clone with git). Without this fallback the
    # whole code-grep path silently produced nothing on hosts without rg.
    if shutil.which("rg"):
        return _rg_search(repo_path, term, limit_per_term)
    return _git_grep_search(repo_path, term, limit_per_term)


def _repo_commit(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--short=12", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def repo_full_name(product_tag: str) -> str:
    """Return the full repository name used by the API, or an empty string outside the catalog."""
    name = repo_for_theme(product_tag) or product_tag
    try:
        return product_repo(name).full_name
    except RepoNotAllowed:
        return ""


def extract_issue_references(*, subject: str = "", message: str = "", draft_html: str = "") -> list[int]:
    """GitHub issue numbers cited in the ticket/draft (#254, issue 254, issues/254)."""
    text = strip_html(f"{subject}\n{message}\n{draft_html}")
    pattern = re.compile(
        r"(?:issues?/(\d{1,6})|\bissues?\s*#?\s*(\d{1,6})\b|#(\d{1,6})\b)",
        re.IGNORECASE,
    )
    numbers: list[int] = []
    for match in pattern.finditer(text):
        for group in match.groups():
            if group:
                numbers.append(int(group))
                break
    return list(dict.fromkeys(numbers))


def extract_version_terms(text: str) -> list[str]:
    """Three-part version numbers (e.g. 2.10.8) so the grep can hit changelog
    entries. Two-part (7.4 / 6.2) are skipped to avoid PHP/WP version noise."""
    return list(dict.fromkeys(re.findall(r"\b\d+\.\d+\.\d+\b", strip_html(text or ""))))


def _fetch_github_issue(full_name: str, number: int, timeout: int = 20) -> Optional[dict]:
    """Verify a GitHub issue via `gh api`. Returns its state/title/url, or None
    if it doesn't exist, is a pull request, gh is missing, or the call fails."""
    if not shutil.which("gh"):
        return None
    try:
        result = subprocess.run(
            [
                "gh", "api", f"repos/{full_name}/issues/{number}",
                "--jq",
                "{number: .number, state: .state, title: .title, url: .html_url, "
                "is_pr: (has(\"pull_request\"))}",
            ],
            check=False, capture_output=True, text=True, timeout=timeout,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if data.get("is_pr"):
        return None
    return data


def collect_github_issue_evidence(product_tag: str, numbers: list[int], limit: int = 3) -> list[str]:
    """Verified GitHub-issue snippets for cited issue numbers (recognized as repo
    evidence). Unverifiable / missing issues are skipped, so a hallucinated issue
    number never becomes 'evidence'."""
    full_name = repo_full_name(product_tag)
    if not full_name:
        return []
    snippets: list[str] = []
    for number in dict.fromkeys(numbers):
        issue = _fetch_github_issue(full_name, number)
        if not issue:
            continue
        title = (issue.get("title") or "").strip()
        snippets.append(
            f"GitHub issue #{issue.get('number', number)} "
            f"({issue.get('state', '?')}) in {full_name}: {title} — {issue.get('url', '')}".strip()
        )
        if len(snippets) >= limit:
            break
    return snippets


def _grep_repo_snippets(
    product_tag: str,
    search_terms: list[str],
    *,
    limit: int,
    refresh: bool = False,
) -> tuple[Optional[Path], str, list[str]]:
    """Grep one repo for the search terms, returning (path, commit, deduped snippets).

    Best-effort throughout: a repo we cannot clone or grep yields no snippets rather than
    failing the whole investigation.
    """
    try:
        repo_path = ensure_repo_checkout(product_tag, refresh=refresh)
        commit = _repo_commit(repo_path)
    except Exception:
        return None, "", []

    collected: list[str] = []
    for term in search_terms:
        try:
            collected.extend(_search_repo(repo_path, term))
        except Exception:
            continue
        if len(collected) >= limit:
            break

    deduped: list[str] = []
    seen = set()
    for snippet in collected:
        if snippet not in seen:
            seen.add(snippet)
            deduped.append(snippet)
        if len(deduped) >= limit:
            break

    return repo_path, commit, deduped


def collect_repo_evidence(
    *,
    theme_tag: str = "",
    subject: str = "",
    message: str = "",
    draft_html: str = "",
    refresh: bool = False,
    limit: int = 6,
) -> Optional[RepoEvidenceResult]:
    product_tag = resolve_repo_target(theme_tag=theme_tag, subject=subject, message=message)
    if not product_tag:
        return None

    if not should_use_repo_evidence(
        theme_tag=theme_tag,
        subject=subject,
        message=message,
        draft_html=draft_html,
    ):
        return None

    terms = extract_repo_search_terms(subject=subject, message=message, draft_html=draft_html)
    version_terms = extract_version_terms(f"{message}\n{draft_html}")
    search_terms = list(dict.fromkeys(terms + version_terms))
    issue_numbers = extract_issue_references(subject=subject, message=message, draft_html=draft_html)

    if not search_terms and not issue_numbers:
        return None

    # Code grep (best-effort; survives a missing checkout / grep tool). The theme repo alone is
    # not enough: a theme can legitimately match ("add_theme_support( 'portfolio' )") while the
    # behaviour being asked about is implemented in a companion plugin. So search the companions
    # too and keep every hit attributed to the repo it came from.
    primary_tag = product_tag
    repo_path: Optional[Path] = None
    commit = ""
    deduped: list[str] = []
    extra_sources: list[RepoSourceEvidence] = []
    if search_terms:
        repo_path, commit, deduped = _grep_repo_snippets(
            primary_tag, search_terms, limit=limit, refresh=refresh
        )
        companion_text = f"{subject}\n{message}\n{strip_html(draft_html or '')}"
        for candidate in companion_repo_targets(primary_tag, checkout=repo_path, text=companion_text):
            candidate_path, candidate_commit, snippets = _grep_repo_snippets(
                candidate, search_terms, limit=limit, refresh=refresh
            )
            if not snippets:
                continue
            if not deduped:
                # Primary had nothing — promote the companion so attribution is exact.
                product_tag = candidate
                repo_path = candidate_path
                commit = candidate_commit
                deduped = snippets
                continue
            extra_sources.append(
                RepoSourceEvidence(
                    product_tag=candidate,
                    repo_url=repo_url(candidate),
                    repo_path=candidate_path or (REPO_CACHE_DIR / candidate),
                    commit=candidate_commit,
                    snippets=snippets,
                )
            )

    # Verified GitHub-issue evidence (independent of the checkout/grep).
    issue_snippets = collect_github_issue_evidence(product_tag, issue_numbers) if issue_numbers else []

    all_snippets = deduped + issue_snippets
    if not all_snippets:
        return None

    result_terms = list(search_terms)
    result_terms.extend(f"issue#{number}" for number in issue_numbers if issue_snippets)

    return RepoEvidenceResult(
        product_tag=product_tag,
        repo_url=repo_url(product_tag),
        repo_path=repo_path or (REPO_CACHE_DIR / product_tag),
        commit=commit,
        terms=result_terms,
        snippets=all_snippets,
        extra_sources=extra_sources,
    )


def format_repo_evidence(result: Optional[RepoEvidenceResult]) -> str:
    if result is None:
        return ""

    repo_prefix = f"{result.repo_path}/"
    snippets = [snippet.replace(repo_prefix, "") for snippet in result.snippets]
    parts = [
        f"Repo evidence ({result.product_tag}):",
        f"Source repo: {result.repo_url}",
    ]
    if result.commit:
        parts.append(f"Commit: {result.commit}")
    parts.extend([
        f"Matched terms: {', '.join(result.terms)}",
        "",
        "\n\n---\n\n".join(snippets),
    ])

    for source in result.extra_sources:
        source_prefix = f"{source.repo_path}/"
        source_snippets = [s.replace(source_prefix, "") for s in source.snippets]
        parts.extend(["", f"Companion repo evidence ({source.product_tag}):"])
        parts.append(f"Source repo: {source.repo_url}")
        if source.commit:
            parts.append(f"Commit: {source.commit}")
        parts.extend(["", "\n\n---\n\n".join(source_snippets)])

    return "\n".join(parts).strip()


def get_repo_evidence_context(
    *,
    theme_tag: str = "",
    subject: str = "",
    message: str = "",
    draft_html: str = "",
    refresh: bool = False,
    limit: int = 6,
) -> str:
    return format_repo_evidence(
        collect_repo_evidence(
            theme_tag=theme_tag,
            subject=subject,
            message=message,
            draft_html=draft_html,
            refresh=refresh,
            limit=limit,
        )
    )


def detect_repo_contradiction(*, draft_html: str, repo_context: str) -> Optional[str]:
    draft_text = strip_html(draft_html or "").lower()
    repo_text = (repo_context or "").lower()

    if not draft_text or not repo_text:
        return None

    for rule in product_settings()["repo_contradictions"]:
        if (rule["subject"] in draft_text
                and any(phrase in draft_text for phrase in rule["draft"])
                and any(phrase in repo_text for phrase in rule["evidence"])):
            return rule["verdict"]

    return None


def build_bug_issue_payload(
    *,
    repo_name: str,
    title: str,
    summary: str,
    live_evidence: str,
    expected_behavior: str,
    actual_behavior: str,
    reproduction_steps: list[str],
    file_references: list[str],
    workaround: str = "",
) -> dict[str, str]:
    """Build a GitHub-ready bug issue payload from confirmed repo and live evidence."""
    body_parts = [summary.strip(), ""]

    body_parts.append("## Expected behavior")
    body_parts.append(expected_behavior.strip())
    body_parts.append("")

    body_parts.append("## Actual behavior")
    body_parts.append(actual_behavior.strip())
    body_parts.append("")

    body_parts.append("## Live evidence")
    body_parts.append(live_evidence.strip())
    body_parts.append("")

    body_parts.append("## Reproduction")
    for index, step in enumerate(reproduction_steps, start=1):
        body_parts.append(f"{index}. {step.strip()}")
    body_parts.append("")

    body_parts.append("## Relevant files")
    for file_reference in file_references:
        body_parts.append(f"- `{file_reference.strip()}`")

    if workaround.strip():
        body_parts.append("")
        body_parts.append("## Workaround")
        body_parts.append(workaround.strip())

    return {
        "repo_name": repo_name.strip(),
        "title": title.strip(),
        "body": "\n".join(body_parts).strip(),
    }
