"""The repo catalog is the only gate for which code may back a customer reply."""
import json

import pytest

from services import repo_catalog, repo_evidence
from services.repo_catalog import (
    RepoNotAllowed,
    load_catalog,
    named_repos,
    product_repo,
    repo_for_theme,
)


def test_unknown_repo_is_refused_at_the_clone_step(tmp_path, monkeypatch):
    """A new private repo that nobody classified must never be cloned as evidence."""
    monkeypatch.setattr(repo_evidence, "REPO_CACHE_DIR", tmp_path)
    with pytest.raises(RepoNotAllowed):
        repo_evidence.ensure_repo_checkout("brand-new-private-repo")
    with pytest.raises(RepoNotAllowed):
        repo_evidence.ensure_repo_checkout("unlisted-notes")
    assert not any(tmp_path.glob("*generation*"))


def test_loader_drops_internal_and_rejects_bad_companions(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({
        "owner": "woodland",
        "repos": {
            "field-journal": {"class": "product", "visibility": "private"},
            "secret": {"class": "internal", "visibility": "private"},
        },
        "companions": {"field-journal": ["field-journal"]},
    }))
    loaded = load_catalog(path)
    assert set(loaded.repos) == {"field-journal"}
    path.write_text(json.dumps({
        "owner": "woodland",
        "repos": {"secret": {"class": "internal"}},
        "companions": {"field-journal": ["secret"]},
    }))
    with pytest.raises(ValueError):
        load_catalog(path)


def test_private_repository_needs_explicit_product_class(tmp_path, monkeypatch):
    data = {"owner": "woodland-example", "repos": {
        "field-journal": {"class": "product", "visibility": "private", "aliases": ["field journal"]},
        "reader-list": {"class": "internal", "visibility": "private"}},
        "theme_repos": {"field-journal": "field-journal"}, "companions": {}}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data))
    catalog = load_catalog(path)
    monkeypatch.setattr(repo_catalog, "catalog", lambda: catalog)
    assert repo_for_theme("field-journal") == "field-journal"
    assert named_repos("Open the Field Journal.") == ["field-journal"]
    for name in ("reader-list", "unclassified"):
        with pytest.raises(RepoNotAllowed):
            product_repo(name)
