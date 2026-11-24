"""Pinned GitHub citations from the deep pass, re-read by Supa.

The deep pass may read code it has no checkout for (a fix branch, a pull request's file, a
WordPress core function) with WebFetch. Its own description of that code is not evidence:
the judge would be trusting the agent. So for every GitHub link with line numbers that the
deep pass cites, Supa resolves the ref to a commit, fetches the file at that commit itself
and hands the judge the exact lines, shaped as a ``Repo evidence`` block.

Only two kinds of source qualify: catalog product repositories (services/repo_catalog.py,
read over SSH, so private product repos work) and the catalog's short third-party list
(read over public HTTPS and labelled third-party). Any other link stays context: a random
fork must never back a claim. A link that cannot be resolved or read yields nothing, which
leaves an unbacked claim exactly as unbacked as before.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from services.repo_catalog import catalog, is_product_repo, is_third_party, product_repo

from services.company_pack import wording

logger = logging.getLogger(__name__)

MAX_CITATIONS = 8
MAX_LINES_PER_CITATION = 60
MAX_TOTAL_CHARS = 12_000
GIT_TIMEOUT = 60
HTTP_TIMEOUT = 20
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# github.com/<owner>/<repo>/blob/<ref and path>#L10-L20  (also #L10, #L10-20)
_BLOB_RE = re.compile(
    r"https?://github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)/blob/(?P<rest>[^\s#?)\]>'\"]+)"
    r"#L(?P<start>\d+)(?:-L?(?P<end>\d+))?"
)
# raw.githubusercontent.com/<owner>/<repo>/<ref and path> followed by :10-20 or " lines 10-20"
_RAW_RE = re.compile(
    r"https?://raw\.githubusercontent\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)/(?P<rest>[^\s#?)\]>'\",]+?)"
    r"(?::|\s+\(?lines?\s+|#L)(?P<start>\d+)(?:\s*[-–]\s*L?(?P<end>\d+))?"
)


@dataclass(frozen=True)
class Citation:
    owner: str
    repo: str
    rest: str      # "<ref>/<path>"; a ref may contain slashes (fix/builder-json-roundtrip)
    start: int
    end: int

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


# Each path segment: letters, digits and . _ @ + - only, and never "." or ".." (an HTTP client
# collapses "../" before the request, which would escape the cited repository).
_SAFE_SEGMENT_RE = re.compile(r"[\w.@+-]+")


def _safe_rest(rest: str) -> bool:
    segments = rest.split("/")
    return len(segments) >= 2 and all(
        _SAFE_SEGMENT_RE.fullmatch(segment) and segment not in {".", ".."} for segment in segments
    )


def find_citations(text: str) -> list[Citation]:
    citations: list[Citation] = []
    for pattern in (_BLOB_RE, _RAW_RE):
        for match in pattern.finditer(text or ""):
            start = int(match.group("start"))
            end = int(match.group("end") or start)
            if end < start:
                start, end = end, start
            if start == end:
                start, end = max(1, start - 3), end + 3
            end = min(end, start + MAX_LINES_PER_CITATION)
            repo = match.group("repo")
            repo = repo[:-4] if repo.endswith(".git") else repo
            rest = match.group("rest").rstrip(".,;:")
            if not _safe_rest(rest):
                continue
            citation = Citation(match.group("owner"), repo, rest, start, end)
            if citation not in citations:
                citations.append(citation)
    return citations[:MAX_CITATIONS]


def _source_kind(citation: Citation) -> Optional[str]:
    """"product", "third_party", or None when the link may not back a claim."""
    if citation.owner.lower() == catalog().owner.lower() and is_product_repo(citation.repo.lower()):
        return "product"
    if is_third_party(citation.full_name):
        return "third_party"
    return None


def _git_env() -> dict[str, str]:
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


def _remote_url(citation: Citation, kind: str) -> str:
    if kind == "product":
        return product_repo(citation.repo.lower()).ssh_url
    return f"https://github.com/{citation.full_name}.git"


def _ls_remote(url: str) -> dict[str, str]:
    """ref name ("main", "fix/x", "v1.2") -> commit sha, for branches and tags."""
    result = subprocess.run(
        ["git", "ls-remote", "--heads", "--tags", url],
        check=True, capture_output=True, text=True, timeout=GIT_TIMEOUT, env=_git_env(),
    )
    refs: dict[str, str] = {}
    for line in result.stdout.splitlines():
        sha, _, name = line.partition("\t")
        for prefix in ("refs/heads/", "refs/tags/"):
            if name.startswith(prefix):
                short = name[len(prefix):]
                if short.endswith("^{}"):  # peeled annotated tag: the commit it points at
                    refs[short[:-3]] = sha
                else:
                    refs.setdefault(short, sha)
    return refs


def resolve_ref(rest: str, refs: dict[str, str]) -> Optional[tuple[str, str, str]]:
    """Split "<ref>/<path>" into (ref, commit sha, path). The longest matching ref wins, so a
    branch named "fix/x" beats a branch named "fix"."""
    first = rest.split("/", 1)[0]
    if _SHA_RE.match(first):
        path = rest.split("/", 1)[1] if "/" in rest else ""
        return (first, first, path) if path else None
    for name in sorted(refs, key=len, reverse=True):
        if rest.startswith(name + "/"):
            return name, refs[name], rest[len(name) + 1:]
    return None


def _read_product_file(url: str, sha: str, path: str) -> str:
    """One file at one commit, without cloning the repository's history or other blobs."""
    workdir = Path(tempfile.mkdtemp(prefix="supa-gh-evidence-"))
    try:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(workdir), *args],
                check=True, capture_output=True, text=True, timeout=GIT_TIMEOUT, env=_git_env(),
            ).stdout

        git("init", "-q")
        git("remote", "add", "origin", url)
        git("config", "remote.origin.promisor", "true")
        git("config", "remote.origin.partialclonefilter", "blob:none")
        git("fetch", "-q", "--depth=1", "--filter=blob:none", "origin", sha)
        return git("show", f"{sha}:{path}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _read_third_party_file(full_name: str, sha: str, path: str) -> str:
    response = httpx.get(
        f"https://raw.githubusercontent.com/{full_name}/{sha}/{path}",
        timeout=HTTP_TIMEOUT,
        follow_redirects=False,
    )
    response.raise_for_status()
    return response.text


def pinned_github_excerpts(text: str) -> tuple[str, list[str]]:
    """Evidence blocks for the cited GitHub lines, and notes on links that could not count."""
    blocks: list[str] = []
    notes: list[str] = []
    total = 0
    refs_cache: dict[str, dict[str, str]] = {}
    files_cache: dict[tuple[str, str, str], list[str]] = {}
    for citation in find_citations(text):
        kind = _source_kind(citation)
        if kind is None:
            notes.append(f"{citation.full_name}: not a catalog product or allowed third-party repo; not evidence")
            continue
        url = _remote_url(citation, kind)
        try:
            if url not in refs_cache:
                refs_cache[url] = _ls_remote(url)
            resolved = resolve_ref(citation.rest, refs_cache[url])
            if resolved is None:
                notes.append(f"{citation.full_name}: could not resolve {citation.rest.split('/', 1)[0]!r} to a commit")
                continue
            ref, sha, path = resolved
            key = (url, sha, path)
            if key not in files_cache:
                content = (
                    _read_product_file(url, sha, path)
                    if kind == "product"
                    else _read_third_party_file(citation.full_name, sha, path)
                )
                files_cache[key] = content.splitlines()
        except Exception as exc:
            logger.info("Pinned GitHub evidence: %s unreadable: %s", citation.full_name, exc)
            notes.append(f"{citation.full_name}: could not be re-read ({type(exc).__name__})")
            continue
        lines = files_cache[key]
        if citation.start > len(lines):
            notes.append(f"{citation.full_name}/{path}: cited line {citation.start} is past the end of the file")
            continue
        end = min(citation.end, len(lines))
        body = "\n".join(f"{n}: {lines[n - 1]}" for n in range(citation.start, end + 1))
        # Third-party lines must not satisfy Supa's code-side "repo evidence present" checks,
        # which guard first-party claims. The judge weighs third-party excerpts.
        heading = (
            f"Repo evidence ({citation.repo.lower()}):"
            if kind == "product"
            else f"Third-party source evidence ({citation.full_name}):"
        )
        header = [
            heading,
            f"Source repo: {url}",
            f"Commit: {sha[:12]} ({ref})",
        ]
        if kind == "third_party":
            header.append(wording("third_party_notice"))
        header.append(f"Lines re-read by Supa at that commit (not the agent's description): {path}:{citation.start}-{end}")
        block = "\n".join(header) + "\n" + body
        if total + len(block) > MAX_TOTAL_CHARS:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks), notes
