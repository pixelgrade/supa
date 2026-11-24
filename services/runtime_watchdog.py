from __future__ import annotations

from dataclasses import dataclass
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from urllib import error, request


@dataclass(frozen=True)
class WatchdogConfig:
    app_dir: Path
    health_url: str
    app_launch_agent_label: str
    interval_seconds: int
    failure_threshold: int
    cooldown_seconds: int
    dump_wait_seconds: int
    dump_signal_name: str
    health_timeout_seconds: int
    verify_attempts: int
    verify_delay_seconds: int
    launchd_domain: str = ""


def _int_env(name: str, default: int) -> int:
    raw_value = (os.getenv(name, str(default)) or "").strip()
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


def parse_launch_agent_pid(output: str) -> int | None:
    match = re.search(r"^\s*pid = (\d+)\s*$", output or "", re.MULTILINE)
    if not match:
        return None
    return int(match.group(1))


def load_watchdog_config(app_dir: str | Path | None = None) -> WatchdogConfig:
    root = Path(app_dir) if app_dir is not None else Path(__file__).resolve().parents[1]
    local_port = _int_env("LOCAL_PORT", 8100)
    health_path = (os.getenv("HEALTH_PATH", "/health") or "/health").strip() or "/health"
    if not health_path.startswith("/"):
        health_path = f"/{health_path}"

    dump_signal = (os.getenv("SUPA_STACKDUMP_SIGNAL", "USR1") or "USR1").strip().upper()
    normalized_dump_signal = dump_signal[3:] if dump_signal.startswith("SIG") else dump_signal

    return WatchdogConfig(
        app_dir=root,
        launchd_domain="system" if os.getenv("LAUNCHD_MODE") == "system-daemon" else f"gui/{os.getuid()}",
        health_url=f"http://127.0.0.1:{local_port}{health_path}",
        app_launch_agent_label=(os.getenv("LAUNCH_AGENT_LABEL") or "org.supa.server").strip(),
        interval_seconds=_int_env("WATCHDOG_INTERVAL_SECONDS", 15),
        failure_threshold=_int_env("WATCHDOG_FAILURE_THRESHOLD", 3),
        cooldown_seconds=_int_env("WATCHDOG_COOLDOWN_SECONDS", 60),
        dump_wait_seconds=_int_env("WATCHDOG_DUMP_WAIT_SECONDS", 2),
        dump_signal_name=normalized_dump_signal,
        health_timeout_seconds=_int_env("WATCHDOG_HEALTH_TIMEOUT_SECONDS", 5),
        verify_attempts=_int_env("WATCHDOG_VERIFY_ATTEMPTS", 10),
        verify_delay_seconds=_int_env("WATCHDOG_VERIFY_DELAY_SECONDS", 2),
    )


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_event(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def check_local_health(config: WatchdogConfig) -> bool:
    try:
        with request.urlopen(config.health_url, timeout=config.health_timeout_seconds) as response:
            return 200 <= response.status < 300
    except (error.URLError, TimeoutError, OSError):
        return False


def get_launch_agent_pid(label: str, domain: str = "") -> int | None:
    result = subprocess.run(
        ["launchctl", "print", f"{domain or f'gui/{os.getuid()}'}/{label}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return parse_launch_agent_pid(result.stdout)


def send_dump_signal(pid: int, signal_name: str) -> bool:
    try:
        resolved = getattr(signal, f"SIG{signal_name}")
        os.kill(pid, resolved)
    except (AttributeError, OSError):
        return False
    return True


def restart_launch_agent(label: str, domain: str = "") -> bool:
    result = subprocess.run(
        ["launchctl", "kickstart", "-k", f"{domain or f'gui/{os.getuid()}'}/{label}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # A user-owned system daemon is KeepAlive-managed, but its owner cannot
        # kickstart the system domain. Terminate only the PID resolved from this
        # exact launchd service, never whichever process happens to own the port.
        if domain == "system" and any(text in result.stderr.lower() for text in
                ("permission denied", "operation not permitted", "not privileged")):
            pid = get_launch_agent_pid(label, domain)
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGTERM)
                    return True
                except OSError:
                    pass
        log_event("launchd restart request failed")
        return False
    return True


def sample_process(pid: int, path: Path) -> bool:
    """Collect stacks without sending a signal requiring a registered handler."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["/usr/bin/sample", str(pid), "1", "-file", str(path)],
                                capture_output=True, text=True, check=False, timeout=5)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def wait_for_recovery(config: WatchdogConfig) -> bool:
    for _ in range(config.verify_attempts):
        if check_local_health(config):
            return True
        time.sleep(config.verify_delay_seconds)
    return False


def run_watchdog(config: WatchdogConfig | None = None) -> int:
    current = config or load_watchdog_config()
    failures = 0
    cooldown_until = time.monotonic() + current.cooldown_seconds

    log_event(
        f"Watchdog started for {current.launchd_domain or f'gui/{os.getuid()}'}/{current.app_launch_agent_label} with interval={current.interval_seconds}s "
        f"threshold={current.failure_threshold}"
    )

    while True:
        remaining_cooldown = cooldown_until - time.monotonic()
        if remaining_cooldown > 0:
            time.sleep(min(current.interval_seconds, remaining_cooldown))
            continue

        if check_local_health(current):
            if failures:
                log_event(f"Local health recovered after {failures} consecutive failures")
            failures = 0
            time.sleep(current.interval_seconds)
            continue

        failures += 1
        log_event(
            f"Local health failed for {current.app_launch_agent_label} "
            f"({failures}/{current.failure_threshold})"
        )

        if failures < current.failure_threshold:
            time.sleep(current.interval_seconds)
            continue

        pid = get_launch_agent_pid(current.app_launch_agent_label, current.launchd_domain)
        log_event(
            f"Recovery triggered for {current.app_launch_agent_label}; "
            f"pid={pid if pid is not None else 'unknown'}"
        )

        if pid is not None:
            sampled = sample_process(pid, current.app_dir / "logs" / "supa.watchdog-sample.log")
            log_event("Process sample captured" if sampled else "Process sample unavailable")

        restarted = restart_launch_agent(current.app_launch_agent_label, current.launchd_domain)
        log_event("launchd restart requested" if restarted else "launchd restart request failed")

        recovered = wait_for_recovery(current)
        log_event("Recovery verified via local health" if recovered else "Recovery verification failed")

        failures = 0
        cooldown_until = time.monotonic() + current.cooldown_seconds
        time.sleep(current.interval_seconds)


__all__ = [
    "WatchdogConfig",
    "check_local_health",
    "get_launch_agent_pid",
    "load_watchdog_config",
    "parse_launch_agent_pid",
    "restart_launch_agent",
    "run_watchdog",
    "send_dump_signal",
    "sample_process",
    "wait_for_recovery",
]
