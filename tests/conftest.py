"""Use a disposable database and injected site evidence during tests."""

import os
from pathlib import Path
import shutil
import tempfile

import pytest


_TEST_RUNTIME_DIR = Path(tempfile.mkdtemp(prefix="supa-pytest-"))
_TEST_DB_PATH = _TEST_RUNTIME_DIR / "supa.db"
os.environ["SUPA_DB_PATH"] = str(_TEST_DB_PATH)
os.environ["SUPA_SUPPORT_DEEP_PASS"] = "off"

# An installation can supply its own fixtures before application modules import.
if Path(__file__).with_name("private_fixtures.py").is_file():
    from tests import private_fixtures

    private_fixtures.configure(_TEST_RUNTIME_DIR, _TEST_DB_PATH)
    pytest_plugins = ["tests.private_fixtures"]


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_RUNTIME_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def _no_support_site_network(monkeypatch):
    """Support evidence tests inject markup explicitly; unit tests never fetch live sites."""
    monkeypatch.setattr("services.live_site_evidence.fetch_site_html", lambda *a, **kw: "")


@pytest.fixture
def journal_products(monkeypatch, tmp_path):
    """Two invented titles, a companion, and a third-party catalogue for source lookup."""
    import json
    from services import github_evidence, repo_catalog, themes

    aliases = {"field journal": "field-journal", "field-journal": "field-journal",
               "trail atlas": "trail-atlas", "trail-atlas": "trail-atlas"}
    monkeypatch.setattr(themes, "THEMES", aliases)
    monkeypatch.setattr(themes, "VALID_SLUGS", set(aliases.values()))
    monkeypatch.setattr(themes, "LT_THEMES", {"field-journal"})
    data = {"owner": "woodland-example", "repos": {
        "field-journal": {"class": "product", "visibility": "public"},
        "trail-atlas": {"class": "product", "visibility": "public"},
        "journal-tools": {"class": "product", "visibility": "public", "aliases": ["Journal Tools"]}},
        "companions": {"field-journal": ["journal-tools"]},
        "third_party": ["upstream-guide/page-layout", "upstream-guide/map-tools"]}
    config = tmp_path / "catalogue.json"
    config.write_text(json.dumps(data))
    catalogue = repo_catalog.load_catalog(config)
    monkeypatch.setattr(repo_catalog, "catalog", lambda: catalogue)
    monkeypatch.setattr(github_evidence, "catalog", lambda: catalogue)
    return catalogue


@pytest.fixture
def inline_assets(monkeypatch):
    monkeypatch.setattr("services.helpscout.HELPSCOUT_INLINE_IMAGE_HOSTS", {"assets.woodland.example"})
