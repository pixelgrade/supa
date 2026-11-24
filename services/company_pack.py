"""Resolve installation-owned support material relative to the selected pack."""

import json
from functools import lru_cache
from pathlib import Path

from services import installation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = PROJECT_ROOT / Path(installation.PACK).expanduser()


def prompt(name: str, *, schema: dict | None = None) -> str:
    text = (PACK_DIR / "prompts" / f"{name}.md").read_text(encoding="utf-8")
    if schema is not None:
        return text.replace("{{output_schema}}", json.dumps(schema, ensure_ascii=False))
    return text


@lru_cache(maxsize=1)
def product_settings() -> dict:
    return json.loads((PACK_DIR / "products.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _wording() -> dict[str, str]:
    return json.loads((PACK_DIR / "wording.json").read_text(encoding="utf-8"))


def wording(key: str) -> str:
    return _wording()[key]
