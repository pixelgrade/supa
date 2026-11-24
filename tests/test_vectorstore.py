import numpy as np
import pytest
from services.vectorstore import VectorStore


@pytest.fixture(autouse=True)
def offline_encoder(monkeypatch):
    class WordCounts:
        def encode(self, text):
            count = text.lower().count("header")
            return np.array([count, 1.0], dtype=np.float32)

    monkeypatch.setattr("services.vectorstore.SentenceTransformer", lambda _: WordCounts())


def test_vectorstore_add_and_search(tmp_path):
    db_path = str(tmp_path / "vectors.db")
    store = VectorStore(db_path)

    # Add documents
    store.add_documents([
        {"id": "1", "content": "How to fix header issues in Field Journal theme", "theme": "field-journal"},
        {"id": "2", "content": "Processing refund requests within 14 days", "theme": "all"},
        {"id": "3", "content": "Chapter numbering", "theme": "field-journal"},
    ])

    # Search
    results = store.search("header not showing", limit=2)

    assert len(results) <= 2
    assert any("header" in r["content"].lower() for r in results)


def test_vectorstore_filter_by_theme(tmp_path):
    db_path = str(tmp_path / "vectors.db")
    store = VectorStore(db_path)

    store.add_documents([
        {"id": "1", "content": "Field Journal header fix", "theme": "field-journal"},
        {"id": "2", "content": "Trail Atlas header fix", "theme": "trail-atlas"},
    ])

    results = store.search("header", theme_filter="field-journal", limit=5)

    assert len(results) == 1
    assert results[0]["theme"] == "field-journal"


def test_vectorstore_count(tmp_path):
    db_path = str(tmp_path / "vectors.db")
    store = VectorStore(db_path)

    assert store.count() == 0

    store.add_documents([
        {"id": "1", "content": "Test document", "theme": "all"},
    ])

    assert store.count() == 1
