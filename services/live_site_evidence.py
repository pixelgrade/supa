"""Check draft selectors against the pages the customer named. Keep each page's class tokens separate. An unreadable page cannot disprove a selector."""

import html as html_module
import logging
import re
from typing import Iterable, Optional
from urllib.parse import urlparse

import httpx

from services import installation

from services.helpscout import strip_html

logger = logging.getLogger(__name__)

# Domains that appear in these conversations but are never the customer's own site.
NON_CUSTOMER_DOMAINS = {
    *installation.OWN_DOMAINS,
    "helpscout.net",
    "secure.helpscout.net",
    "wordpress.org",
    "wordpress.com",
    "github.com",
    "youtube.com",
    "youtu.be",
    "google.com",
    "gravatar.com",
    "w.org",
    "schema.org",
    # Asset/CDN hosts that show up in HelpScout payloads (avatars, embedded images).
    "cloudfront.net",
    "gstatic.com",
    "googleapis.com",
    "gravatar.com",
    "s.w.org",
    "imgur.com",
}

# A link to an image or script is never the customer's site. Without this the HelpScout
# avatar CDN in every conversation payload gets picked as the site to verify against.
ASSET_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".bmp",
    ".css", ".js", ".json", ".xml", ".pdf", ".zip", ".mp4", ".webm",
)

_URL_RE = re.compile(r"https?://[^\s\"'<>()\[\]]+", re.IGNORECASE)
_CLASS_TOKEN_RE = re.compile(r"\.(-?[_a-zA-Z][\w-]*)")
_SELECTOR_RE = re.compile(r"([^{}\n]+)\{")
# Belt and braces alongside the line-based split: a CSS selector uses only this alphabet, so
# an English sentence that happens to sit on the same line cannot pass as one.
_SELECTOR_SHAPE_RE = re.compile(r"[\w\s.#>+~*\[\]=\"':(),%-]+")
_HTML_CLASS_ATTR_RE = re.compile(r"""class=["']([^"']+)["']""", re.IGNORECASE)


def _registrable_domain(host: str) -> str:
    host = (host or "").lower().lstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def extract_site_urls(*texts: str) -> list[str]:
    """Customer page URLs, preserving paths instead of substituting the homepage.

    Use the first text containing customer URLs (normally the current message), so an old
    staging site in the history cannot validate CSS for the customer's new site.
    """
    for text in texts:
        if not text:
            continue
        urls: list[str] = []
        for raw in _URL_RE.findall(strip_html(text)):
            url = raw.rstrip(".,;:)")
            parsed = urlparse(url)
            host = _registrable_domain(parsed.netloc)
            if not host or "." not in host:
                continue
            if any(host == domain or host.endswith(f".{domain}") for domain in NON_CUSTOMER_DOMAINS):
                continue
            if parsed.path.lower().endswith(ASSET_SUFFIXES):
                continue
            url = parsed._replace(path=parsed.path or "/", fragment="").geturl()
            if url not in urls:
                urls.append(url)
        if urls:
            return urls[:6]
    return []


def extract_site_url(*texts: str) -> Optional[str]:
    """First customer page URL, with its path intact."""
    return next(iter(extract_site_urls(*texts)), None)


def _normalize_for_css(draft_html: str) -> str:
    """Flatten the reply to text while keeping line breaks.

    `strip_html` collapses all whitespace, which merges the prose and the code block onto one
    line. A selector scan then runs from the start of the reply up to the first `{` and reads
    the `.org` in "reader.example.org" as a class name. Line structure is what separates a selector
    from the sentence in front of it, so it has to survive.
    """
    text = re.sub(r"(?i)<br\s*/?>", "\n", draft_html or "")
    text = re.sub(r"(?i)</(p|div|li|pre|code)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_module.unescape(text)
    text = text.replace("\xa0", " ")
    return text


def extract_css_class_tokens(draft_html: str) -> list[str]:
    """Class names the draft's CSS snippets actually depend on.

    Only text in selector position counts, and only when it is shaped like a selector — prose
    that happens to precede a brace must never contribute tokens.
    """
    tokens: list[str] = []
    for match in _SELECTOR_RE.finditer(_normalize_for_css(draft_html)):
        selector = match.group(1).strip()
        if not selector or selector.startswith("@"):
            # At-rules (@media, @supports) carry conditions, not selectors.
            continue
        if not _SELECTOR_SHAPE_RE.fullmatch(selector):
            continue
        tokens.extend(_CLASS_TOKEN_RE.findall(selector))
    return list(dict.fromkeys(tokens))


def extract_html_class_names(html: str) -> set[str]:
    """Every class present in the rendered markup.

    Deliberately reads `class="..."` attributes rather than the page's stylesheets: a class
    that only appears in CSS is not on the page, and hiding it would do nothing.
    """
    names: set[str] = set()
    for match in _HTML_CLASS_ATTR_RE.finditer(html or ""):
        names.update(match.group(1).split())
    return names


def find_missing_classes(tokens: Iterable[str], site_html: str) -> list[str]:
    """Classes the draft targets that do not exist in the fetched markup.

    A descendant selector needs every one of its parts to exist, so a single missing token
    is enough to make the whole rule dead — `.nb-header .site-branding` selected nothing even
    though `.nb-header` was right there.
    """
    present = extract_html_class_names(site_html)
    if not present:
        return []
    return [token for token in tokens if token not in present]


def fetch_site_html(url: str, *, timeout: float = 10.0) -> str:
    """Best-effort fetch. Any failure yields "" so the guard simply does not fire."""
    try:
        response = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SupaSupport/1.0)"},
        )
        if response.status_code != 200:
            return ""
        return response.text
    except Exception:
        logger.info("live-site fetch failed for %s", url, exc_info=True)
        return ""


def detect_dead_css_selectors(*, draft_html: str, site_html: str) -> Optional[str]:
    """Describe CSS in the draft that cannot match the customer's live markup."""
    tokens = extract_css_class_tokens(draft_html)
    if not tokens or not site_html:
        return None

    missing = find_missing_classes(tokens, site_html)
    if not missing:
        return None

    return (
        "draft CSS targets classes that are absent from the customer's live site: "
        + ", ".join(f".{name}" for name in missing)
    )


def _source_rank(location: str) -> tuple[int, int]:
    """Rank a `file:line` hit so the citation points at the rule, not a leftover bundle.

    A class appears in many places — the SCSS that defines it, the PHP that prints it, a
    compiled bundle, an abandoned `js-old/` copy. Citing the stylesheet is what makes the
    evidence checkable by a human; citing `assets/js-old/main.js` looks authoritative and
    tells them nothing.
    """
    path = location.lower()
    if "js-old" in path or "/dist/" in path or "/build/" in path or ".bundle." in path:
        penalty = 2
    elif path.endswith((".min.js", ".min.css")):
        penalty = 2
    else:
        penalty = 0

    for rank, suffixes in enumerate(((".scss", ".sass"), (".css",), (".php",), (".html",))):
        if any(path.split(":")[0].endswith(suffix) for suffix in suffixes):
            return (penalty, rank)
    return (penalty, 9)


def verify_css_classes_in_repo(tokens: Iterable[str], product_tag: str) -> dict[str, str]:
    """Find the draft's classes in checked-out stylesheets and return source-qualified evidence."""
    from services.repo_catalog import is_product_repo
    from services.repo_evidence import _search_repo, ensure_repo_checkout

    if not is_product_repo(product_tag):
        return {}

    try:
        repo_path = ensure_repo_checkout(product_tag)
    except Exception:
        logger.info("css verification: no checkout for %s", product_tag, exc_info=True)
        return {}

    confirmed: dict[str, str] = {}
    for token in tokens:
        try:
            hits = _search_repo(repo_path, token, limit_per_term=5)
        except Exception:
            continue

        locations = []
        for hit in hits:
            match = re.search(r"([^\s:]+):(\d+):", hit)
            if match:
                path = match.group(1).replace(f"{repo_path}/", "").lstrip("./")
                locations.append(f"{path}:{match.group(2)}")
        if locations:
            confirmed[token] = min(locations, key=_source_rank)
    return confirmed


def format_css_repo_verification(
    *,
    product_tag: str,
    confirmed: dict[str, str],
    unconfirmed: Iterable[str],
) -> str:
    """Render as a `Repo evidence (...)` block so the pipeline counts it as real backing."""
    from services.repo_catalog import RepoNotAllowed
    from services.repo_evidence import repo_url

    if not confirmed:
        return ""
    try:
        source_repo = repo_url(product_tag)
    except RepoNotAllowed:
        source_repo = ""

    lines = [
        f"Repo evidence ({product_tag}):",
        f"Source repo: {source_repo}",
        "Verified CSS selectors used by the draft:",
    ]
    lines.extend(f".{name} — {location}" for name, location in sorted(confirmed.items()))

    missing = [name for name in unconfirmed if name not in confirmed]
    if missing:
        lines.append(
            "Not found in this repo (may come from WordPress core or a companion plugin): "
            + ", ".join(f".{name}" for name in sorted(missing))
        )
    return "\n".join(lines)


def verify_draft_css_against_repo(*, draft_html: str, product_tag: str) -> str:
    """Evidence block confirming where the draft's CSS classes are defined, or ""."""
    tokens = extract_css_class_tokens(draft_html)
    if not tokens or not product_tag:
        return ""

    confirmed = verify_css_classes_in_repo(tokens, product_tag)
    return format_css_repo_verification(
        product_tag=product_tag, confirmed=confirmed, unconfirmed=tokens
    )


def detect_dead_css_selectors_for_context(
    *,
    draft_html: str,
    texts: Iterable[str],
    timeout: float = 10.0,
) -> Optional[str]:
    """Guard entry point: only fetches when the draft actually ships CSS classes."""
    if not extract_css_class_tokens(draft_html):
        return None

    verdicts: list[str] = []
    for url in extract_site_urls(*texts):
        site_html = fetch_site_html(url, timeout=timeout)
        if not site_html or not extract_html_class_names(site_html):
            # Failure to read a named page is not proof that its selector is dead.
            return None
        verdict = detect_dead_css_selectors(draft_html=draft_html, site_html=site_html)
        if verdict is None:
            return None
        verdicts.append(f"{url}: {verdict}")
    return "; ".join(verdicts) or None


def live_css_evidence_for_context(*, draft_html: str, texts: Iterable[str]) -> str:
    """Give the writer/judge verified page hooks, including real page IDs for scoped CSS.

    WebFetch often returns Markdown without classes. The live guard already has access to
    raw markup, so expose the small relevant facts instead of asking the writer to guess.
    """
    tokens = extract_css_class_tokens(draft_html)
    if not tokens:
        return ""
    blocks = []
    for url in extract_site_urls(*texts):
        markup = fetch_site_html(url)
        if not markup:
            continue
        present = extract_html_class_names(markup)
        body = re.search(r"<body\b[^>]*>", markup, re.IGNORECASE)
        body_classes = sorted(extract_html_class_names(body.group(0)))[:40] if body else []
        blocks.append("\n".join([
            f"Live CSS evidence read by Supa: {url}",
            "Verified body classes (use the real page ID when scoping this page): " + ", ".join(body_classes),
            "Draft classes present here: " + ", ".join(t for t in tokens if t in present),
            "Draft classes absent here: " + ", ".join(t for t in tokens if t not in present),
        ]))
    return "\n\n".join(blocks)
