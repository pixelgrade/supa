import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUPA_DB_PATH = PROJECT_ROOT / "db" / "supa.db"


def get_supa_db_path() -> str:
    """Return the configured audit database, or the production default."""
    return os.getenv("SUPA_DB_PATH") or str(DEFAULT_SUPA_DB_PATH)
