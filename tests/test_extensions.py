import importlib
import sys

import pytest
from starlette.requests import Request

from services import extensions


def test_loader_registers_each_configured_module_once(tmp_path, monkeypatch):
    monkeypatch.setattr(extensions, "_active", {})
    monkeypatch.setattr(extensions, "_router_sources", [])
    monkeypatch.setattr(extensions, "_manual_routes", [])
    monkeypatch.setattr(extensions, "_cc_checks", [])
    monkeypatch.setattr(extensions, "_voice_sources", [])
    monkeypatch.setattr(extensions, "_stages", [])
    monkeypatch.setattr(extensions, "_agent_tasks", {})
    monkeypatch.setattr(extensions, "_agent_routes", {})
    monkeypatch.setattr(extensions, "_webhooks", [])
    monkeypatch.setattr(extensions, "_workers", {"before_delivery": {}, "after_delivery": {}})
    monkeypatch.setattr(extensions.modules, "_loaded", {})
    name = "woodland_test_extension"
    (tmp_path / f"{name}.py").write_text("registrations = []\nregistrations.append('field-journal')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("SUPA_EXTENSIONS", f" {name}, ,{name}")
    try:
        extensions.load()
        extensions.load()
        assert extensions.loaded() == [name]
        assert importlib.import_module(name).registrations == ["field-journal"]
    finally:
        extensions.reset()
        sys.modules.pop(name, None)


def test_missing_extension_stops_startup(monkeypatch):
    monkeypatch.setenv("SUPA_EXTENSIONS", "woodland_missing_extension")
    with pytest.raises(RuntimeError, match="woodland_missing_extension"):
        extensions.load()


@pytest.mark.asyncio
@pytest.mark.parametrize(("peer", "host", "headers", "local"), [
    ("127.0.0.1", "127.0.0.1:8100", [], True),
    ("::1", "[::1]:8100", [], True),
    ("192.0.2.1", "localhost", [], False),
    ("127.0.0.1", "support.example", [], False),
    ("127.0.0.1", "localhost", [(b"x-forwarded-for", b"127.0.0.1")], False),
    ("127.0.0.1", "localhost", [(b"cf-connecting-ip", b"192.0.2.1")], False),
])
async def test_health_preserves_existing_response_off_loopback(monkeypatch, peer, host, headers, local):
    import main

    monkeypatch.setattr(main.mailbox_provider, "native_origin", lambda: "")
    request = Request({"type": "http", "path": "/health", "client": (peer, 1), "headers": [(b"host", host.encode()), *headers]})
    legacy = {"status": "ok"}
    assert await main.health(request) == ({**legacy, "extensions": extensions.loaded()} if local else legacy)


@pytest.mark.parametrize("command", [
    ["-c", "import main"],
    ["-m", "supa_cli.main", "--help"],
])
def test_configured_extension_loads_through_process_entry_points(tmp_path, command):
    assert_process_loads_extension(tmp_path, command)


def assert_process_loads_extension(tmp_path, command):
    import os
    from pathlib import Path
    import subprocess

    marker = tmp_path / "loaded"
    (tmp_path / "woodland_startup.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    env = {**os.environ, "SUPA_EXTENSIONS": "woodland_startup", "PYTHONPATH": str(tmp_path)}
    subprocess.run([sys.executable, *command], cwd=Path(__file__).resolve().parents[1],
                   capture_output=True, env=env, check=True)
    assert marker.exists()
