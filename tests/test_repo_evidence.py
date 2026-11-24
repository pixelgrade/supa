from services.repo_evidence import RepoEvidenceResult, extract_repo_search_terms, format_repo_evidence


# --- A: GitHub-issue verification + grep fallback -----------------------

from services.repo_evidence import extract_issue_references, extract_version_terms
import services.repo_evidence as repo_evidence  # noqa: E402


def test_extract_issue_references_variants():
    nums = extract_issue_references(
        subject="Typography bug",
        message="see issue #254 and issues/87",
        draft_html="<p>GitHub issue 254 is open; also #999.</p>",
    )
    assert nums == [254, 87, 999]


def test_extract_issue_references_ignores_non_issue_numbers():
    # version numbers / hex-ish tokens must not become issue refs
    assert extract_issue_references(message="Field Journal 2.10.8 on PHP 7.4") == []


def test_extract_version_terms():
    assert extract_version_terms("broken in 2.10.8, fixed in 2.11.0 (not 7.4)") == ["2.10.8", "2.11.0"]


def test_git_grep_fallback_finds_matches_without_rg(tmp_path, monkeypatch):
    # Force the git-grep path (no rg binary) over a plain directory.
    monkeypatch.setattr(repo_evidence.shutil, "which", lambda name: None)
    (tmp_path / "journal.php").write_text(
        "function journal_typography() {\n  // legacy typography handling\n}\n"
    )
    snippets = repo_evidence._search_repo(tmp_path, "typography")
    assert snippets and any("typography" in s.lower() for s in snippets)


# --- B: companion-repo fallback ----------------------------------------

from services.repo_evidence import GENERIC_SEARCH_STOPWORDS, extract_repo_search_terms


def test_generic_support_words_are_not_used_as_repo_search_terms():
    """"error"/"page"/"found" match every WP codebase and manufacture false evidence."""
    terms = extract_repo_search_terms(
        subject="Projects return page not found errors",
        message="When I click on any of the projects it opens to the error page not found.",
    )

    assert terms, "expected the keyword fallback to still produce something"
    assert not GENERIC_SEARCH_STOPWORDS.intersection(t.lower() for t in terms)
    assert "projects" in [t.lower() for t in terms]


def test_companion_and_primary_sources_keep_their_paths_and_commits(tmp_path):
    from services.repo_evidence import RepoEvidenceResult, RepoSourceEvidence, format_repo_evidence

    primary = tmp_path / "journal"
    companion = tmp_path / "index"
    result = RepoEvidenceResult("journal", "https://source.example/journal", primary,
        commit="aaa111", snippets=[f"{primary}/pages.py:15: title = entry.name"],
        extra_sources=[RepoSourceEvidence("index", "https://source.example/index", companion,
            commit="bbb222", snippets=[f"{companion}/entries.py:22: title = chapter.name"])])
    text = format_repo_evidence(result)
    source, extra = text.split("Companion repo evidence (index):")
    assert "Commit: aaa111" in source and "pages.py:15:" in source and "entries.py" not in source
    assert "Commit: bbb222" in extra and "entries.py:22:" in extra and "pages.py" not in extra
    assert str(tmp_path) not in text
