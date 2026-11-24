import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "db" / "supa.db"


def _connection_path(connection: sqlite3.Connection) -> Path:
    row = connection.execute("PRAGMA database_list").fetchone()
    return Path(row[2]).resolve()


class _SourceChangedDuringFingerprint(RuntimeError):
    pass


def _database_paths(path: Path) -> tuple[Path, Path, Path]:
    return path, Path(f"{path}-wal"), Path(f"{path}-shm")


def _file_fingerprint(path: Path) -> tuple:
    try:
        stat_before = path.stat()
    except FileNotFoundError:
        if path.exists():
            raise _SourceChangedDuringFingerprint(path)
        return False, None, None, None

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        stat_after = path.stat()
    except FileNotFoundError as exc:
        raise _SourceChangedDuringFingerprint(path) from exc

    before = stat_before.st_size, stat_before.st_mtime_ns
    after = stat_after.st_size, stat_after.st_mtime_ns
    if before != after:
        raise _SourceChangedDuringFingerprint(path)
    return True, stat_after.st_size, stat_after.st_mtime_ns, digest.hexdigest()


def _database_fingerprint(path: Path) -> tuple:
    return tuple(_file_fingerprint(candidate) for candidate in _database_paths(path))


def _snapshot_database(
    path: Path, snapshot_dir: Path, *, attempts: int = 3
) -> Path | None:
    if attempts < 1:
        raise ValueError("attempts must be positive")

    source_main, source_wal, _ = _database_paths(path)
    snapshot_path = snapshot_dir / path.name
    snapshot_wal = Path(f"{snapshot_path}-wal")
    snapshot_shm = Path(f"{snapshot_path}-shm")

    def clear_snapshot() -> None:
        for candidate in (snapshot_path, snapshot_wal, snapshot_shm):
            candidate.unlink(missing_ok=True)

    for _ in range(attempts):
        try:
            before = _database_fingerprint(path)
            if not before[0][0]:
                if before == _database_fingerprint(path):
                    return None
                continue

            snapshot_dir.mkdir(exist_ok=True)
            clear_snapshot()
            shutil.copy2(source_main, snapshot_path)
            if before[1][0]:
                shutil.copy2(source_wal, snapshot_wal)
            after = _database_fingerprint(path)
        except (FileNotFoundError, _SourceChangedDuringFingerprint):
            clear_snapshot()
            continue

        if before == after:
            return snapshot_path
        clear_snapshot()

    raise RuntimeError(f"could not capture a stable database snapshot: {path}")


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_row(path: Path | None) -> tuple | None:
    if path is None:
        return None

    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'tickets'"
        ).fetchone()
        if table is None:
            return None
        row = connection.execute(
            "SELECT * FROM tickets WHERE helpscout_id = ?", ("43",)
        ).fetchone()
        return tuple(row) if row is not None else None
    finally:
        connection.close()


def _observed_database_state(path: Path) -> tuple:
    state = []
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        if not candidate.exists():
            state.append((False, None, None, None))
            continue
        stat = candidate.stat()
        state.append(
            (
                True,
                stat.st_size,
                stat.st_mtime_ns,
                hashlib.sha256(candidate.read_bytes()).hexdigest(),
            )
        )
    return tuple(state)


def test_database_snapshot_captures_committed_wal_without_opening_source(
    monkeypatch, tmp_path
):
    source_path = tmp_path / "source.db"
    writer = sqlite3.connect(source_path)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE tickets (helpscout_id TEXT PRIMARY KEY)")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("INSERT INTO tickets VALUES ('wal-only')")
        writer.commit()

        main_file_only_path = tmp_path / "main-file-only.db"
        shutil.copy2(source_path, main_file_only_path)
        main_file_only = sqlite3.connect(main_file_only_path)
        try:
            assert main_file_only.execute("SELECT * FROM tickets").fetchone() is None
        finally:
            main_file_only.close()

        source_state_before = _observed_database_state(source_path)
        real_connect = sqlite3.connect
        connected_paths = []

        def reject_source_connection(database, *args, **kwargs):
            database_text = os.fspath(database)
            connected_paths.append(database_text)
            if source_path.resolve().as_uri() in database_text:
                raise AssertionError("the mutable source database was opened with SQLite")
            return real_connect(database, *args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", reject_source_connection)

        snapshot_path = _snapshot_database(source_path, tmp_path / "snapshot")

        assert snapshot_path is not None
        snapshot = sqlite3.connect(snapshot_path)
        try:
            assert snapshot.execute("SELECT * FROM tickets").fetchone() == ("wal-only",)
        finally:
            snapshot.close()
        assert _observed_database_state(source_path) == source_state_before
        assert connected_paths == [os.fspath(snapshot_path)]
    finally:
        writer.close()


def test_database_snapshot_does_not_create_source_sidecars(tmp_path):
    source_path = tmp_path / "source.db"
    connection = sqlite3.connect(source_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE tickets (helpscout_id TEXT PRIMARY KEY)")
    connection.commit()
    connection.close()
    assert not Path(f"{source_path}-wal").exists()
    assert not Path(f"{source_path}-shm").exists()
    source_state_before = _observed_database_state(source_path)

    assert _snapshot_database(source_path, tmp_path / "snapshot") is not None

    assert _observed_database_state(source_path) == source_state_before


def test_database_snapshot_retries_then_fails_when_source_never_stabilizes(
    monkeypatch, tmp_path
):
    source_path = tmp_path / "source.db"
    source_path.write_bytes(b"version-0")
    real_copy2 = shutil.copy2
    main_copy_count = 0

    def copy_then_change_source(source, destination):
        nonlocal main_copy_count
        result = real_copy2(source, destination)
        if Path(source) == source_path:
            main_copy_count += 1
            source_path.write_bytes(f"version-{main_copy_count}".encode())
        return result

    monkeypatch.setattr(shutil, "copy2", copy_then_change_source)

    with pytest.raises(RuntimeError, match="stable database snapshot"):
        _snapshot_database(source_path, tmp_path / "snapshot", attempts=2)

    assert main_copy_count == 2
    assert not (tmp_path / "snapshot" / source_path.name).exists()


def test_database_snapshot_is_safe_when_source_is_absent(tmp_path):
    snapshot_dir = tmp_path / "snapshot"

    assert _snapshot_database(tmp_path / "missing.db", snapshot_dir) is None
    assert not snapshot_dir.exists()


def test_supa_db_override_reaches_all_import_time_defaults(monkeypatch):
    expected = Path(os.environ["SUPA_DB_PATH"]).resolve()

    from api import dashboard, webhooks
    from db import database
    from services import processor, startup_recovery

    assert Path(database.DB_PATH).resolve() == expected
    assert Path(dashboard.DB_PATH).resolve() == expected
    assert Path(webhooks.DEFAULT_DB_PATH).resolve() == expected
    assert Path(startup_recovery.DEFAULT_DB_PATH).resolve() == expected

    with (
        patch.object(processor, "VectorStore"),
        patch.object(processor, "init_db") as init_db,
    ):
        processor.TicketProcessor(auto_index=False)
    init_db.assert_called_once()
    assert Path(init_db.call_args.args[0]).resolve() == expected


def test_database_helpers_honor_patched_module_default(monkeypatch, tmp_path):
    from db import database

    patched_default = tmp_path / "patched-default.db"
    env_default = tmp_path / "environment-default.db"
    monkeypatch.setattr(database, "DB_PATH", patched_default)
    monkeypatch.setenv("SUPA_DB_PATH", str(env_default))

    connection = database.get_db()
    try:
        assert _connection_path(connection) == patched_default.resolve()
    finally:
        connection.close()

    assert database.ensure_db_initialized() == str(patched_default)
    assert patched_default.exists()
    assert not env_default.exists()


def test_database_explicit_path_wins(monkeypatch, tmp_path):
    from db.database import get_db

    explicit_db = (tmp_path / "explicit.db").resolve()
    connection = get_db(str(explicit_db))
    try:
        assert _connection_path(connection) == explicit_db
    finally:
        connection.close()

@pytest.mark.asyncio
async def test_autosend_scheduler_default_and_explicit_paths(monkeypatch, tmp_path):
    from services import autosend_scheduler

    captured: list[str] = []

    async def fake_worker_loop(*, db_path, helpscout, config, **kwargs):
        captured.append(db_path)

    monkeypatch.setattr(autosend_scheduler, "run_worker_loop", fake_worker_loop)

    await autosend_scheduler.schedule_autosend_worker(helpscout=object())
    explicit = str((tmp_path / "autosend.db").resolve())
    await autosend_scheduler.schedule_autosend_worker(helpscout=object(), db_path=explicit)

    assert [str(Path(path).resolve()) for path in captured] == [
        str(Path(os.environ["SUPA_DB_PATH"]).resolve()),
        explicit,
    ]


def test_production_default_remains_repo_database():
    env = os.environ.copy()
    env.pop("SUPA_DB_PATH", None)
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from services.runtime_paths import get_supa_db_path; print(get_supa_db_path())",
        ],
        cwd=ROOT,
        env=env,
        text=True,
    ).strip()

    assert Path(output).resolve() == DEFAULT_DB_PATH.resolve()


def test_isolated_dashboard_and_webhook_suite_preserves_repository_database(tmp_path):
    # Refuse to launch the write-heavy subprocess until every cached application
    # default points at this process's isolated test database. This makes the RED
    # run safe even while production modules still hard-code db/supa.db.
    expected = Path(os.environ["SUPA_DB_PATH"]).resolve()
    from api import dashboard, webhooks
    from db import database
    from services import startup_recovery

    assert Path(database.DB_PATH).resolve() == expected
    assert Path(dashboard.DB_PATH).resolve() == expected
    assert Path(webhooks.DEFAULT_DB_PATH).resolve() == expected
    assert Path(startup_recovery.DEFAULT_DB_PATH).resolve() == expected

    # Exercise a real repository-relative default using only source-controlled code
    # and a database created for this test. No developer database is inspected.
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    children = [Path("main.py"), Path("pyproject.toml")]
    children.extend(Path("tests") / name for name in (
        "__init__.py", "conftest.py", "lease_recording.py", "test_dashboard.py", "test_webhooks.py"))
    for directory in ("agents", "api", "db", "services", "supa_cli"):
        children.extend(path.relative_to(ROOT) for path in (ROOT / directory).glob("*.py"))
    for directory in ("packs/woodland", "static"):
        children.extend(path.relative_to(ROOT) for path in (ROOT / directory).rglob("*")
                        if path.is_file() and not path.is_symlink())
    for relative in children:
        source = ROOT / relative
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target, follow_symlinks=False)
    repository_db = checkout / "db" / "supa.db"
    with sqlite3.connect(repository_db) as conn:
        conn.execute("CREATE TABLE tickets (helpscout_id TEXT PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO tickets VALUES ('43', 'an invented message')")

    live_state_before = _observed_database_state(repository_db)
    snapshot_before = _snapshot_database(repository_db, tmp_path / "before")
    checksum_before = _sha256(snapshot_before)
    fixture_before = _fixture_row(snapshot_before)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_dashboard.py",
            "tests/test_webhooks.py",
            "-q",
        ],
        cwd=checkout,
        env={**os.environ, "SUPA_PACK": "packs/woodland", "SUPA_EXTENSIONS": ""},
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    snapshot_after = _snapshot_database(repository_db, tmp_path / "after")
    assert _sha256(snapshot_after) == checksum_before
    assert _fixture_row(snapshot_after) == fixture_before
    assert _observed_database_state(repository_db) == live_state_before
