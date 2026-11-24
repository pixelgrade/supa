"""Standalone client for Supa's versioned Agent API."""

from importlib.metadata import PackageNotFoundError, version as _distribution_version
from pathlib import Path
import tomllib


PROJECT_METADATA_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _resolve_version() -> str:
    try:
        return _distribution_version("supa")
    except PackageNotFoundError:
        try:
            with PROJECT_METADATA_PATH.open("rb") as handle:
                project_version = tomllib.load(handle)["project"]["version"]
        except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
            return "unknown"
        if isinstance(project_version, str) and project_version.strip():
            return project_version.strip()
        return "unknown"


__version__ = _resolve_version()
