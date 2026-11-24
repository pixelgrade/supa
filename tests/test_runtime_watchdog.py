from pathlib import Path

from services.runtime_watchdog import WatchdogConfig, load_watchdog_config, parse_launch_agent_pid


def test_parse_launch_agent_pid_reads_pid_line():
    launchctl_output = """
gui/501/example.woodland.supa = {
    state = running
    pid = 8070
}
"""

    assert parse_launch_agent_pid(launchctl_output) == 8070


def test_parse_launch_agent_pid_returns_none_without_pid():
    assert parse_launch_agent_pid("state = running") is None


def test_load_watchdog_config_uses_expected_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCAL_PORT", raising=False)
    monkeypatch.delenv("HEALTH_PATH", raising=False)
    monkeypatch.delenv("LAUNCH_AGENT_LABEL", raising=False)
    monkeypatch.delenv("WATCHDOG_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("WATCHDOG_FAILURE_THRESHOLD", raising=False)
    monkeypatch.delenv("WATCHDOG_COOLDOWN_SECONDS", raising=False)
    monkeypatch.delenv("WATCHDOG_DUMP_WAIT_SECONDS", raising=False)
    monkeypatch.delenv("SUPA_STACKDUMP_SIGNAL", raising=False)

    config = load_watchdog_config(app_dir=tmp_path)

    assert isinstance(config, WatchdogConfig)
    assert config.app_dir == tmp_path
    assert config.health_url == "http://127.0.0.1:8100/health"
    assert config.app_launch_agent_label == "org.supa.server"
    assert config.interval_seconds == 15
    assert config.failure_threshold == 3
    assert config.cooldown_seconds == 60
    assert config.dump_wait_seconds == 2
    assert config.dump_signal_name == "USR1"


def test_system_daemon_domain_and_exact_pid_fallback(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from services import runtime_watchdog as watchdog
    monkeypatch.setenv('LAUNCH_AGENT_LABEL', 'example.woodland.supa')
    monkeypatch.setenv('LAUNCHD_MODE', 'system-daemon')
    config = load_watchdog_config(tmp_path)
    assert config.launchd_domain == 'system'
    calls, signals = [], []
    def run(argv, **kw):
        calls.append(argv)
        if argv[1] == 'print':
            return SimpleNamespace(returncode=0, stdout='pid = 1234', stderr='')
        return SimpleNamespace(returncode=1, stdout='', stderr='permission denied')
    monkeypatch.setattr(watchdog.subprocess, 'run', run)
    monkeypatch.setattr(watchdog.os, 'kill', lambda pid, sig: signals.append((pid, sig)))
    assert watchdog.get_launch_agent_pid(config.app_launch_agent_label, config.launchd_domain) == 1234
    assert watchdog.restart_launch_agent(config.app_launch_agent_label, config.launchd_domain)
    assert calls[0][-1] == 'system/example.woodland.supa'
    assert signals == [(1234, watchdog.signal.SIGTERM)]


def test_restart_never_kills_an_unknown_pid(monkeypatch):
    from types import SimpleNamespace
    from services import runtime_watchdog as watchdog
    monkeypatch.setattr(watchdog.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1, stdout='', stderr='permission denied'))
    monkeypatch.setattr(watchdog.os, 'kill', lambda *a: (_ for _ in ()).throw(AssertionError('must not kill')))
    assert not watchdog.restart_launch_agent('service', 'system')


def test_diagnostics_sample_does_not_signal_unregistered_handler(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from services import runtime_watchdog as watchdog
    calls = []
    monkeypatch.setattr(watchdog.subprocess, 'run', lambda argv, **kw: calls.append((argv, kw)) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(watchdog.os, 'kill', lambda *a: (_ for _ in ()).throw(AssertionError('unsafe signal')))
    assert watchdog.sample_process(1234, tmp_path / 'sample.log')
    assert calls[0][0][:3] == ['/usr/bin/sample', '1234', '1']
    assert calls[0][1]['timeout'] == 5


def test_watchdog_recovery_uses_system_domain_and_native_sample(monkeypatch, tmp_path):
    import dataclasses
    import pytest
    from services import runtime_watchdog as watchdog
    config = dataclasses.replace(load_watchdog_config(tmp_path), launchd_domain='system',
                                 cooldown_seconds=0, failure_threshold=1)
    checks, calls = iter([False,True]), []
    monkeypatch.setattr(watchdog, 'check_local_health', lambda cfg: next(checks))
    monkeypatch.setattr(watchdog, 'get_launch_agent_pid', lambda label, domain: calls.append(('pid',domain)) or 1234)
    monkeypatch.setattr(watchdog, 'sample_process', lambda pid, path: calls.append(('sample',pid)) or True)
    monkeypatch.setattr(watchdog, 'restart_launch_agent', lambda label, domain: calls.append(('restart',domain)) or True)
    monkeypatch.setattr(watchdog, 'send_dump_signal', lambda *a: (_ for _ in ()).throw(AssertionError('unsafe signal')))
    class Finished(Exception):
        pass
    monkeypatch.setattr(watchdog.time, 'sleep', lambda seconds: (_ for _ in ()).throw(Finished()))
    with pytest.raises(Finished):
        watchdog.run_watchdog(config)
    assert calls == [('pid','system'),('sample',1234),('restart','system')]
