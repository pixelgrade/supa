import faulthandler
import os
import signal
from pathlib import Path
from typing import TextIO


_STACKDUMP_FILE: TextIO | None = None
_REGISTERED_SIGNAL: signal.Signals | None = None


def get_dump_signal(raw_name: str | None = None) -> signal.Signals:
    value = (raw_name or os.getenv("SUPA_STACKDUMP_SIGNAL", "USR1")).strip().upper()
    signal_name = value if value.startswith("SIG") else f"SIG{value}"
    resolved = getattr(signal, signal_name, None)
    if not isinstance(resolved, signal.Signals):
        raise ValueError(f"Unsupported dump signal: {value}")
    return resolved


def configure_runtime_diagnostics(log_path: str | Path) -> Path:
    global _STACKDUMP_FILE, _REGISTERED_SIGNAL

    target_path = Path(log_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if _REGISTERED_SIGNAL is not None:
        try:
            faulthandler.unregister(_REGISTERED_SIGNAL)
        except RuntimeError:
            pass

    if _STACKDUMP_FILE is not None and not _STACKDUMP_FILE.closed:
        _STACKDUMP_FILE.close()

    _STACKDUMP_FILE = target_path.open("a", encoding="utf-8", buffering=1)
    _REGISTERED_SIGNAL = get_dump_signal()

    faulthandler.enable(file=_STACKDUMP_FILE, all_threads=True)
    faulthandler.register(
        _REGISTERED_SIGNAL,
        file=_STACKDUMP_FILE,
        all_threads=True,
        chain=False,
    )
    return target_path
