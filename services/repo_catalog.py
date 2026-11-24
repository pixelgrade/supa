"""The installation catalog controls which repositories can be cloned or cited. Only listed product repositories qualify as first-party evidence. Third-party repositories have separate rules for citations."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from services.company_pack import PACK_DIR

CATALOG_PATH = PACK_DIR / "repo_catalog.json"

# A shallow clone of anything bigger is not worth a support lookup.
MAX_CLONE_SIZE_KB = 400_000


class RepoNotAllowed(ValueError):
    """The repository is not a product repository in the catalog."""


@dataclass(frozen=True)
class CatalogRepo:
    name: str
    owner: str
    visibility: str
    size_kb: int
    aliases: tuple[str, ...]

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def ssh_url(self) -> str:
        return f"git@github.com:{self.full_name}.git"


@dataclass(frozen=True)
class Catalog:
    owner: str
    repos: dict[str, CatalogRepo]          # product repositories only
    theme_repos: dict[str, str]            # product slug -> repository
    companions: dict[str, tuple[str, ...]]  # repository or "@classic" -> companion repositories
    third_party: tuple[str, ...]           # "Owner/repo", compared case-insensitively


def load_catalog(path: Path = CATALOG_PATH) -> Catalog:
    data = json.loads(path.read_text())
    owner = data["owner"]
    repos = {
        name.lower(): CatalogRepo(
            name=name,
            owner=owner,
            visibility=entry.get("visibility", ""),
            size_kb=int(entry.get("size_kb") or 0),
            aliases=tuple(alias.lower() for alias in entry.get("aliases") or ()),
        )
        for name, entry in (data.get("repos") or {}).items()
        if entry.get("class") == "product"
    }
    companions = {key: tuple(value) for key, value in (data.get("companions") or {}).items()}
    for key, names in companions.items():
        unknown = [name for name in names if name not in repos]
        if unknown:
            raise ValueError(f"repo catalog: companions of {key} are not product repos: {unknown}")
    return Catalog(
        owner=owner,
        repos=repos,
        theme_repos=dict(data.get("theme_repos") or {}),
        companions=companions,
        third_party=tuple(data.get("third_party") or ()),
    )


@lru_cache(maxsize=1)
def catalog() -> Catalog:
    return load_catalog()


def product_repo(name: str) -> CatalogRepo:
    """The catalog entry for a readable repository; raises RepoNotAllowed otherwise."""
    repo = catalog().repos.get((name or "").strip().lower())
    if repo is None:
        raise RepoNotAllowed(f"{name!r} is not a product repository in {PACK_DIR.name}/repo_catalog.json")
    if repo.size_kb > MAX_CLONE_SIZE_KB:
        raise RepoNotAllowed(f"{name!r} is too large to clone for a support lookup ({repo.size_kb} KB)")
    return repo


def is_product_repo(name: str) -> bool:
    try:
        product_repo(name)
    except RepoNotAllowed:
        return False
    return True


def repo_for_theme(theme_slug: str) -> Optional[str]:
    """Resolve an installation product slug to its configured repository."""
    slug = (theme_slug or "").strip().lower()
    if not slug:
        return None
    name = catalog().theme_repos.get(slug, slug)
    return name if is_product_repo(name) else None


def companion_repos(repo: str, *, is_theme: bool) -> list[str]:
    """Look up the additional repositories configured for a product."""
    table = catalog().companions
    if repo in table:
        return list(table[repo])
    if is_theme:
        return list(table.get("@classic", ()))
    return []


def _alias_pattern(alias: str) -> re.Pattern[str]:
    words = [re.escape(word) for word in re.split(r"[\s-]+", alias) if word]
    return re.compile(r"\b" + r"[\s-]?".join(words) + r"s?\b")


def named_repos(text: str, *, exclude: tuple[str, ...] = ()) -> list[str]:
    """Find products named by an explicit catalog alias, in catalog order. Product detection handles names without aliases."""
    lowered = (text or "").lower()
    found: list[str] = []
    for repo in catalog().repos.values():
        if repo.name in exclude or not repo.aliases:
            continue
        if any(_alias_pattern(alias).search(lowered) for alias in repo.aliases):
            found.append(repo.name)
    return found


def is_third_party(full_name: str) -> bool:
    wanted = (full_name or "").strip().lower()
    return any(wanted == allowed.lower() for allowed in catalog().third_party)
