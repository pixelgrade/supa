from html import escape
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
from urllib.parse import urlsplit
import asyncio
import logging
import os

from services import extensions, mailbox_provider

from api.webhooks import native_worker, router as webhooks_router
from api.dashboard import router as dashboard_router
from api.agent import (
    AgentAPIBoundaryMiddleware,
    close_agent_run_tasks,
    close_conversation_gateway,
    get_conversation_gateway,
    recover_agent_runs,
    router as agent_router,
)
from db.database import init_db
from services.agent_events import schedule_event_retention_worker
from services.agent_operations import RecoverySummary, recover_stale_operations
from services.autosend_scheduler import schedule_autosend_worker
from services.scheduled_sends import schedule_worker as schedule_scheduled_send_worker
from services.helpscout import HelpScoutClient
from services.runtime_paths import get_supa_db_path
from services.startup_recovery import schedule_periodic_recovery, schedule_startup_recovery

from services.company_pack import wording

extensions.load()

def source_url() -> str:
    default = "https://github.com/pixelgrade/supa"
    value = os.environ.get("SUPA_SOURCE_URL") or default
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise ValueError("SUPA_SOURCE_URL must be an HTTP(S) source URL")
    return value


app = FastAPI(title="Supa", description=wording("api_description"),
              license_info={"name": "AGPL-3.0-or-later", "url": source_url()})
logger = logging.getLogger(__name__)
app.add_middleware(AgentAPIBoundaryMiddleware)

app.include_router(webhooks_router)
app.include_router(dashboard_router)
app.include_router(agent_router)
for router in extensions.routers():
    app.include_router(router)

STATIC_DIR = Path(__file__).parent / "static"
RECOVERY_TASK_NAMES = (
    "native_mailbox_task",
    "startup_recovery_task",
    "periodic_recovery_task",
    "autosend_worker_task",
    *extensions.worker_names("before_delivery"),
    "event_retention_task",
    "scheduled_send_task",
    *extensions.worker_names("after_delivery"),
)


def get_cache_version():
    """Get cache version based on static file modification times."""
    css_file = STATIC_DIR / "style.css"
    js_file = STATIC_DIR / "app.js"
    css_mtime = int(css_file.stat().st_mtime) if css_file.exists() else 0
    js_mtime = int(js_file.stat().st_mtime) if js_file.exists() else 0
    return max(css_mtime, js_mtime)


def recover_agent_operations_on_startup(
    db_path: str | os.PathLike[str] | None = None,
) -> RecoverySummary:
    """Initialize and recover the selected ledger before write workers start."""

    path = get_supa_db_path() if db_path is None else os.fspath(db_path)
    init_db(path)
    return recover_stale_operations(db_path=path)


@app.on_event("startup")
async def startup_recovery() -> None:
    def _attach_recovery_task(name: str, task: asyncio.Task, *, label: str) -> None:
        def _log_task_outcome(completed_task: asyncio.Task) -> None:
            try:
                result = completed_task.result()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("%s failed", label)
            else:
                logger.info("%s finished: %s", label, result)

        setattr(app.state, name, task)
        task.add_done_callback(_log_task_outcome)

    operation_recovery = recover_agent_operations_on_startup()
    if mailbox_provider.native_origin():
        _attach_recovery_task(
            "native_mailbox_task", asyncio.create_task(native_worker(get_supa_db_path())), label="Native mailbox delivery"
        )
    logger.info("Agent operation recovery finished: %s", operation_recovery)
    interrupted, queued = recover_agent_runs(get_supa_db_path())
    logger.info("Agent task recovery: interrupted=%s queued=%s", interrupted, queued)

    _attach_recovery_task(
        "event_retention_task",
        schedule_event_retention_worker(get_supa_db_path()),
        label="Agent event retention",
    )

    _attach_recovery_task(
        "startup_recovery_task",
        schedule_startup_recovery(),
        label="Startup recovery",
    )
    _attach_recovery_task(
        "periodic_recovery_task",
        schedule_periodic_recovery(),
        label="Periodic recovery",
    )
    extensions.start_workers("before_delivery", _attach_recovery_task)
    autosend_helpscout = HelpScoutClient(
        client_id=os.getenv("HELPSCOUT_CLIENT_ID", ""),
        client_secret=os.getenv("HELPSCOUT_CLIENT_SECRET", ""),
    )
    _attach_recovery_task(
        "autosend_worker_task",
        schedule_autosend_worker(helpscout=autosend_helpscout),
        label="Autosend worker",
    )
    # Booked agent replies (supa draft schedule) go out through the Agent API gateway.
    _attach_recovery_task(
        "scheduled_send_task",
        schedule_scheduled_send_worker(gateway_factory=get_conversation_gateway),
        label="Scheduled send worker",
    )

    extensions.start_workers("after_delivery", _attach_recovery_task)


@app.on_event("shutdown")
async def shutdown_recovery_tasks() -> None:
    for task_name in RECOVERY_TASK_NAMES:
        task = getattr(app.state, task_name, None)
        if task is None:
            continue
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Recovery task %s failed before shutdown", task_name)
        finally:
            setattr(app.state, task_name, None)
    await close_agent_run_tasks()
    try:
        await close_conversation_gateway()
    except Exception:
        logger.exception("Agent conversation gateway failed to close during shutdown")


@app.get("/static/index.html", include_in_schema=False)
@app.get("/")
async def root():
    """Serve index.html with automatic cache busting."""
    index_file = STATIC_DIR / "index.html"
    html = index_file.read_text()

    # Replace version placeholder with actual file mtime
    version = get_cache_version()
    html = html.replace("__VERSION__", str(version))
    html = html.replace("__MAILBOX_WEB__", escape(mailbox_provider.web_origin(), quote=True))
    html = html.replace("__MAILBOX_INBOX__", escape(mailbox_provider.inbox_url(), quote=True))

    html = html.replace("__SOURCE_URL__", escape(source_url(), quote=True))
    html = html.replace("__DASHBOARD_TAGLINE__", escape(wording("dashboard_tagline")))
    html = html.replace("__RUN_HINT__", escape(wording("run_hint"), quote=True))
    return HTMLResponse(content=html)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
async def health(request: Request = None):
    details = {}
    if (request is not None and request.client
            and request.client.host in {"127.0.0.1", "::1"}
            and request.url.hostname in {"127.0.0.1", "::1", "localhost"}
            and not any(key == "forwarded" or key.startswith(("x-forwarded-", "cf-"))
                        for key in request.headers)):
        details["extensions"] = extensions.loaded()
        if request.headers.get("x-supa-verify-settings") == "1":
            from services.installation_readback import explicit_settings_match
            details["explicit_settings_match"] = explicit_settings_match()
    if mailbox_provider.native_origin():
        from services.native_mailbox_events import counts
        task = getattr(app.state, "native_mailbox_task", None)
        if task is None or task.done():
            return JSONResponse({"status": "unavailable", "mailbox_worker": "stopped", **details}, status_code=503)
        return {"status": "ok", "mailbox_jobs": counts(get_supa_db_path()), **details}
    return {"status": "ok", **details}
