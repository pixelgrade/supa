"""Pinned GitHub citations count only when Supa re-reads the lines itself."""
import pytest

from services.company_pack import wording
from services.repo_catalog import product_repo
from services import github_evidence as gh
from services.github_evidence import find_citations, pinned_github_excerpts, resolve_ref

SHA = "a1b2c3d4" * 5
FILE = "\n".join(f"line {n}" for n in range(1, 201))


@pytest.fixture
def remote(monkeypatch, journal_products):
    calls = {"ls": [], "product": [], "third": []}

    def ls_remote(url):
        calls["ls"].append(url)
        return {"master": SHA, "fix": "f" * 40, "fix/index-order": SHA}

    def read_product(url, sha, path):
        calls["product"].append((url, sha, path))
        return FILE

    def read_third(full_name, sha, path):
        calls["third"].append((full_name, sha, path))
        return FILE

    monkeypatch.setattr(gh, "_ls_remote", ls_remote)
    monkeypatch.setattr(gh, "_read_product_file", read_product)
    monkeypatch.setattr(gh, "_read_third_party_file", read_third)
    return calls


def test_branch_with_slash_resolves_to_its_commit():
    refs = {"fix": "a" * 40, "fix/index-order": SHA}
    assert resolve_ref("fix/index-order/features/init.php", refs) == (
        "fix/index-order", SHA, "features/init.php"
    )
    assert resolve_ref(f"{SHA}/readme.txt", {}) == (SHA, SHA, "readme.txt")
    assert resolve_ref("nope/readme.txt", refs) is None


def test_product_citation_becomes_repo_evidence(remote):
    text = "save -> https://github.com/woodland-example/journal-tools/blob/fix/index-order/features/init.php#L100-L102"
    blocks, notes = pinned_github_excerpts(text)
    assert blocks.startswith("Repo evidence (journal-tools):")
    assert "Source repo: " + product_repo("journal-tools").ssh_url in blocks
    assert "100: line 100" in blocks and "102: line 102" in blocks and "103:" not in blocks
    assert remote["product"] == [(product_repo("journal-tools").ssh_url, SHA, "features/init.php")]
    assert notes == []


def test_third_party_is_labelled_and_read_over_https(remote):
    text = "https://github.com/upstream-guide/page-layout/blob/master/lib/x.php#L5"
    blocks, _ = pinned_github_excerpts(text)
    assert "Third-party source evidence (upstream-guide/page-layout):" in blocks
    # Never shaped as Woodland repo evidence: the code-side gates must not count it.
    assert "Repo evidence (" not in blocks
    assert wording("third_party_notice") in blocks
    assert remote["third"] and not remote["product"]
    # A single cited line comes with three lines of context either side.
    assert "2: line 2" in blocks and "8: line 8" in blocks and "9: line 9" not in blocks


def test_allowlisted_third_party_source_can_back_its_own_behaviour(remote):
    blocks, notes = pinned_github_excerpts(
        "https://github.com/upstream-guide/map-tools/blob/master/render-index.php#L10-L20"
    )
    assert "Third-party source evidence (upstream-guide/map-tools):" in blocks
    assert "Repo evidence (" not in blocks
    assert remote["third"] and not remote["product"]
    assert notes == []


@pytest.mark.parametrize("url", [
    "https://github.com/someone/journal-tools/blob/master/init.php#L1-L3",          # a fork
    "https://github.com/woodland-example/private-index/blob/master/a.php#L1",   # internal
    "https://github.com/woodland-example/internal-catalog/blob/main/CURRENT.md#L1",
])
def test_links_outside_the_catalog_never_count(remote, url):
    blocks, notes = pinned_github_excerpts(url)
    assert blocks == ""
    assert notes and "not evidence" in notes[0]
    assert not remote["ls"] and not remote["product"] and not remote["third"]


def test_unreadable_or_out_of_range_lines_stay_unbacked(remote, monkeypatch):
    blocks, notes = pinned_github_excerpts("https://github.com/woodland-example/journal-tools/blob/master/init.php#L900-L905")
    assert blocks == "" and "past the end" in notes[0]

    def boom(*_args):
        raise RuntimeError("network")

    monkeypatch.setattr(gh, "_read_product_file", boom)
    blocks, notes = pinned_github_excerpts("https://github.com/woodland-example/journal-tools/blob/master/init.php#L1")
    assert blocks == "" and "could not be re-read" in notes[0]


def test_unanchored_links_are_not_citations():
    assert find_citations("see https://github.com/woodland-example/journal-tools/pull/63/files") == []
    assert find_citations("https://github.com/woodland-example/journal-tools/blob/master/init.php") == []
    raw = find_citations("https://raw.githubusercontent.com/woodland-example/journal-tools/master/init.php:10-12")
    assert [(c.repo, c.start, c.end) for c in raw] == [("journal-tools", 10, 12)]


@pytest.mark.parametrize("url", [
    # httpx collapses ../ before the request: this would read WordPress/evil instead of gutenberg.
    "https://github.com/upstream-guide/page-layout/blob/master/lib/../../../evil/repo/main/x.php#L1",
    "https://github.com/upstream-guide/page-layout/blob/master/lib/%2e%2e/x.php#L1",
    "https://github.com/woodland-example/journal-tools/blob/master/./init.php#L1",
])
def test_paths_that_could_escape_the_repo_are_not_citations(remote, url):
    assert find_citations(url) == []
    assert pinned_github_excerpts(url) == ("", [])
    assert not remote["ls"]
