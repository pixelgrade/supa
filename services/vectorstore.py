import sqlite3
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer


class VectorStore:
    """Simple vector store using SQLite and sentence-transformers."""

    def __init__(self, db_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self._init_db()

    def _init_db(self):
        """Initialize the database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                theme TEXT,
                source TEXT,
                metadata TEXT,
                embedding BLOB
            )
        """)
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add documents with embeddings."""
        conn = self._get_conn()
        added = 0

        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue

            # Generate embedding
            embedding = self.model.encode(content)
            embedding_bytes = embedding.tobytes()

            # Store
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO documents (id, title, content, theme, source, metadata, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc.get("id", ""),
                    doc.get("title", ""),
                    content,
                    doc.get("theme", "all"),
                    doc.get("source", ""),
                    json.dumps(doc.get("metadata", {})),
                    embedding_bytes
                ))
                added += 1
            except Exception as e:
                print(f"Error adding document {doc.get('id')}: {e}")

        conn.commit()
        conn.close()
        return added

    def search(
        self,
        query: str,
        theme_filter: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        # Encode query
        query_embedding = self.model.encode(query)

        # Get all documents (for small datasets, this is fine)
        conn = self._get_conn()

        if theme_filter:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE theme = ? OR theme = 'all'",
                (theme_filter,)
            )
        else:
            cursor = conn.execute("SELECT * FROM documents")

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        # Calculate similarities
        results = []
        for row in rows:
            doc_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            # Derive URL: for docs, the id IS the URL
            url = row["id"] if row["id"].startswith("http") else ""

            results.append({
                "id": row["id"],
                "title": row["title"] or row["id"],
                "content": row["content"],
                "theme": row["theme"],
                "source": row["source"],
                "url": url,
                "score": float(similarity)
            })

        # Sort by similarity and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def clear(self, source: Optional[str] = None):
        """Clear documents, optionally filtered by source."""
        conn = self._get_conn()
        if source:
            conn.execute("DELETE FROM documents WHERE source = ?", (source,))
        else:
            conn.execute("DELETE FROM documents")
        conn.commit()
        conn.close()

    def count(self) -> int:
        """Count documents."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def count_by_source(self) -> Dict[str, int]:
        """Count documents grouped by source."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT source, COUNT(*) as count
            FROM documents
            GROUP BY source
        """)
        result = {row["source"]: row["count"] for row in cursor.fetchall()}
        conn.close()
        return result
