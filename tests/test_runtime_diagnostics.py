from pathlib import Path
from unittest.mock import Mock
import signal

from services.runtime_diagnostics import configure_runtime_diagnostics, get_dump_signal


def test_get_dump_signal_normalizes_usr1_name(monkeypatch):
    monkeypatch.delenv("SUPA_STACKDUMP_SIGNAL", raising=False)

    assert get_dump_signal("usr1") == signal.SIGUSR1
    assert get_dump_signal("SIGUSR1") == signal.SIGUSR1


def test_configure_runtime_diagnostics_enables_faulthandler_and_registers_signal(tmp_path, monkeypatch):
    enable = Mock()
    register = Mock()
    unregister = Mock()

    monkeypatch.setattr("services.runtime_diagnostics.faulthandler.enable", enable)
    monkeypatch.setattr("services.runtime_diagnostics.faulthandler.register", register)
    monkeypatch.setattr("services.runtime_diagnostics.faulthandler.unregister", unregister)
    monkeypatch.delenv("SUPA_STACKDUMP_SIGNAL", raising=False)

    log_path = tmp_path / "logs" / "supa.stackdump.log"
    configured_path = configure_runtime_diagnostics(log_path)

    assert configured_path == log_path
    assert log_path.exists()

    enable.assert_called_once()
    file_handle = enable.call_args.kwargs["file"]
    assert Path(file_handle.name) == log_path
    assert enable.call_args.kwargs["all_threads"] is True

    register.assert_called_once_with(
        signal.SIGUSR1,
        file=file_handle,
        all_threads=True,
        chain=False,
    )
