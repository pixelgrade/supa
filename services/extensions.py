"""Load configured extensions and register their server-side integrations."""

from dataclasses import dataclass
from collections.abc import Callable, Iterable
from importlib import import_module

from supa_cli import extensions as modules

configured = modules.configured
_active: dict[str, None] = {}
_workers = {"before_delivery": {}, "after_delivery": {}}
_agent_routes = {}
_agent_tasks = {}
_stages = []
_voice_sources = []
_cc_checks = []
_manual_routes = []
_webhooks = []
_router_sources: list[Callable[[], Iterable]] = []


def register_routers(source):
    """Register a factory so command-line processes need not import web routers."""
    if source not in _router_sources:
        _router_sources.append(source)


def routers():
    return [router for source in _router_sources for router in source()]


def load(environ=None):
    modules.load(environ)
    for name in configured(environ):
        if name in _active:
            continue
        register = getattr(import_module(name), "register_server", None)
        if register is not None:
            try:
                register()
            except Exception as exc:
                raise RuntimeError(f"Could not register Supa extension {name!r}") from exc
        _active[name] = None


def loaded():
    return list(_active)


def reset():
    """Clear server registrations and import bookkeeping for isolated tests."""
    _active.clear()
    _router_sources.clear()
    _manual_routes.clear()
    _cc_checks.clear()
    _voice_sources.clear()
    _stages.clear()
    _agent_tasks.clear()
    _agent_routes.clear()
    _webhooks.clear()
    for group in _workers.values():
        group.clear()
    modules.reset()


def register_worker(phase, name, label, start):
    """Register a startup function returning a task, or None when disabled."""
    _workers[phase][name] = (label, start)


def worker_names(phase):
    return tuple(_workers[phase])


def start_workers(phase, attach):
    for name, (label, start) in _workers[phase].items():
        task = start()
        if task is not None:
            attach(name, task, label=label)


def register_manual_route(route, hold):
    if (route, hold) not in _manual_routes:
        _manual_routes.append((route, hold))


def manual_hold(payload):
    load()
    return any(hold(payload) for _, hold in _manual_routes)


def manual_route(payload):
    load()
    for route, _ in _manual_routes:
        result = route(payload)
        if result is not None:
            return result
    return None


def register_webhook(intercept):
    if intercept not in _webhooks:
        _webhooks.append(intercept)


def intercept_webhook(event, ticket_id, payload, background_tasks, route, case_id):
    for intercept in _webhooks:
        response = intercept(event, ticket_id, payload, background_tasks, route, case_id)
        if response is not None:
            return response
    return None


def register_cc_check(check):
    if check not in _cc_checks:
        _cc_checks.append(check)


def cc_block_reason(address):
    load()
    for check in _cc_checks:
        reason = check(address)
        if reason:
            return reason
    return None


def register_voice(source):
    if source not in _voice_sources:
        _voice_sources.append(source)


def voice_guidance(role, *, wins):
    load()
    if not _voice_sources:
        from services.company_pack import PACK_DIR
        try:
            return (PACK_DIR / "voice.md").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""
    return "\n\n".join(style for source in _voice_sources if (style := source(role, wins=wins)))


@dataclass(frozen=True)
class AgentTask:
    required_scopes: frozenset[str]
    runner_name: str
    run: Callable


def register_agent_task(name, scopes, runner_name, run):
    task = AgentTask(frozenset(scopes), runner_name, run)
    if name in _agent_tasks and _agent_tasks[name] != task:
        raise ValueError(f"Duplicate extension task: {name}")
    _agent_tasks[name] = task


def agent_tasks():
    load()
    return _agent_tasks.copy()


def register_agent_routes(commands, install):
    _agent_routes[install] = tuple(commands)


def agent_commands():
    load()
    return [cmd for commands in _agent_routes.values() for cmd in commands]


def install_agent_routes(router):
    load()
    for install in _agent_routes:
        install(router)


def register_stages(values):
    for value in values:
        if value not in _stages:
            _stages.append(value)


def stages():
    load()
    return tuple(_stages)
