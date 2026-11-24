import json
import os
import asyncio
import uuid
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from pathlib import Path

from db.database import (
    get_extended_stats,
    get_recent_semantic_failures,
    get_recent_tickets,
    get_stats,
    get_ticket_logs_by_helpscout_ids,
    get_tickets_by_type,
    log_ticket,
)
from services.company_pack import PACK_DIR

from services.theme_history import theme_hint_for_ticket
from services.processor import (
    _get_support_autosend_threshold,
    _get_support_provider_env,
    deep_evidence_collector,
    get_processor,
    merge_context_sections,
)
from services.repo_evidence import get_repo_evidence_context
from services.helpscout import get_draft_thread_ids, is_supa_ignored, parse_api_conversation
from services.past_tickets import find_similar_tickets, format_past_tickets_context
from services.support_pipeline import SupportPipelineContext, run_support_pipeline
from agents.classifier import classify_ticket, generate_subject
from agents.claude_cli import get_claude_warning
from services.customer import fetch_customer_data, format_customer_context
from services.conversation_locks import conversation_lease
from services.live_runs import get_live_runs_registry
from services.runtime_paths import get_supa_db_path
from services.ticket_runs import run_ticket_job
from services.theme_requirements import get_classic_theme_requirements_context
from services.reprocess import (
    can_auto_send,
    customer_waiting,
    default_since,
    find_casualties,
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

DB_PATH = get_supa_db_path()


def verify_admin_key(x_admin_key: Optional[str] = Header(None)):
    """Verify admin API key."""
    admin_key = os.getenv("ADMIN_API_KEY")
    if admin_key and x_admin_key != admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return True


class StatsResponse(BaseModel):
    support: int
    marketing: int
    spam: int
    total: int


class TicketLogResponse(BaseModel):
    helpscout_id: str
    subject: str
    customer_name: str
    classification: str
    status: str
    response_posted: bool
    created_at: str
    error_message: Optional[str]


class ActivityResponse(BaseModel):
    stats: StatsResponse
    recent: List[TicketLogResponse]
    knowledge_count: int


class QueueItem(BaseModel):
    id: int
    subject: str
    theme: str
    url: str
    confidence_score: Optional[int] = None
    confidence_status: Optional[str] = None
    confidence_attempts: int = 1
    confidence_summary: Optional[str] = None
    # True when the draft scored high enough to auto-send but was withheld because the
    # support agent runs in draft_only mode. Without surfacing it these sit in the queue
    # looking like ordinary low-confidence drafts and go unanswered for days.
    auto_send_ready: bool = False


class DraftDetail(BaseModel):
    id: int
    subject: str
    theme: str
    url: str
    customer_name: str
    customer_email: str
    customer_message: str
    draft_content: str
    draft_thread_id: Optional[int] = None
    tags: List[str] = []
    confidence_score: Optional[int] = None
    confidence_status: Optional[str] = None
    confidence_attempts: int = 1
    confidence_summary: Optional[str] = None
    confidence_trace: List[Dict[str, Any]] = []
    confidence_provider: Optional[str] = None


class SendReplyRequest(BaseModel):
    conversation_id: int
    reply_html: str
    draft_thread_id: Optional[int] = None
    new_subject: Optional[str] = None
    theme_tag: Optional[str] = None
    existing_tags: List[str] = []


class ExtendedStats(BaseModel):
    received: int
    auto_closed: int
    drafts_created: int
    manual: int


class KnowledgeStats(BaseModel):
    docs: int = 0
    examples: int = 0
    internal: int = 0


class QueueWarning(BaseModel):
    code: str
    message: str
    detail: Optional[str] = None
    occurred_at: str


class QueueResponse(BaseModel):
    drafts: List[QueueItem]
    follow_ups: List[QueueItem]
    stats: ExtendedStats
    knowledge: KnowledgeStats
    warning: Optional[QueueWarning] = None
    stuck_count: int = 0  # tickets stranded by a provider outage, awaiting recovery


class LiveRunItem(BaseModel):
    run_id: str
    ticket_id: int
    source: str
    stage: str
    status: str


class LiveRunsResponse(BaseModel):
    runs: List[LiveRunItem]


class LiveRunEventsResponse(BaseModel):
    events: List[Dict[str, Any]]
    run: Optional[Dict[str, Any]] = None


class TestTicketRequest(BaseModel):
    conversation_id: int


class RunTicketResponse(BaseModel):
    status: str
    run_id: str


class PostCachedRequest(BaseModel):
    conversation_id: int
    customer_id: Optional[int] = None
    response_html: str
    new_subject: Optional[str] = None
    theme_tag: Optional[str] = None
    existing_tags: List[str] = []
    classification: Optional[str] = None
    subject: Optional[str] = None
    customer_name: Optional[str] = None
    confidence_score: Optional[int] = None
    confidence_status: Optional[str] = None
    confidence_attempts: int = 1
    confidence_summary: Optional[str] = None
    confidence_trace: List[Dict[str, Any]] = []
    confidence_provider: Optional[str] = None
    evidence_summary: Optional[str] = None
    delivery_mode: Optional[str] = None
    investigator_provider: Optional[str] = None
    evaluator_provider: Optional[str] = None
    decision_note: Optional[str] = None
    investigation_steps: List[str] = []
    risk_flags: List[str] = []
    sources: List[str] = []
    send: bool = False


class ReprocessRequest(BaseModel):
    since: Optional[str] = None   # UTC 'YYYY-MM-DD HH:MM:SS'; default: 24h ago
    send: bool = False            # dry-run by default


class TestTicketResponse(BaseModel):
    ticket: Dict[str, Any]
    classification: str
    new_subject: Optional[str]
    knowledge_results: List[Dict[str, Any]]
    similar_tickets: List[Dict[str, Any]]
    response_html: Optional[str]
    posted: bool
    customer_data: Optional[Dict[str, Any]] = None
    confidence_score: Optional[int] = None
    confidence_status: Optional[str] = None
    confidence_attempts: int = 1
    confidence_summary: Optional[str] = None
    confidence_trace: List[Dict[str, Any]] = []
    confidence_provider: Optional[str] = None
    delivery_mode: Optional[str] = None
    decision_note: Optional[str] = None
    investigator_provider: Optional[str] = None
    evaluator_provider: Optional[str] = None
    evidence_summary: Optional[str] = None
    investigation_steps: List[str] = []
    risk_flags: List[str] = []
    hard_blockers: List[str] = []
    sources: List[str] = []


@router.get("/stats", response_model=StatsResponse)
async def get_dashboard_stats(_: bool = Depends(verify_admin_key)):
    """Get ticket statistics for today."""
    stats = get_stats(db_path=DB_PATH, days=1)
    return StatsResponse(**stats)


@router.get("/activity", response_model=ActivityResponse)
async def get_activity(_: bool = Depends(verify_admin_key)):
    """Get stats and recent activity."""
    stats = get_stats(db_path=DB_PATH, days=1)
    recent = get_recent_tickets(db_path=DB_PATH, limit=20)

    processor = get_processor()
    knowledge_count = processor.get_knowledge_count()

    return ActivityResponse(
        stats=StatsResponse(**stats),
        recent=[TicketLogResponse(
            helpscout_id=t["helpscout_id"],
            subject=t["subject"] or "",
            customer_name=t["customer_name"] or "",
            classification=t["classification"] or "",
            status=t["status"] or "",
            response_posted=bool(t["response_posted"]),
            created_at=t["created_at"] or "",
            error_message=t["error_message"]
        ) for t in recent],
        knowledge_count=knowledge_count
    )


def _is_auto_send_ready(local: dict) -> bool:
    """Whether this draft cleared the auto-send bar and is only waiting on delivery policy."""
    score = local.get("confidence_score")
    if not isinstance(score, int):
        return False
    if score < _get_support_autosend_threshold():
        return False
    if "\nBlocked by:" in (local.get("decision_note") or ""):
        return False
    return (local.get("confidence_status") or "").strip().lower() in ("", "accept")


@router.get("/queue", response_model=QueueResponse)
async def get_queue(_: bool = Depends(verify_admin_key)):
    """Get support queue: drafts to review, follow-ups, and 7-day stats."""
    processor = get_processor()

    # Get drafts and follow-ups from HelpScout
    drafts = await processor.helpscout.get_conversations_with_drafts(limit=10)
    follow_ups = await processor.helpscout.get_follow_ups(limit=10)

    # Get extended stats for last 7 days
    stats = get_extended_stats(db_path=DB_PATH, days=7)

    draft_logs = get_ticket_logs_by_helpscout_ids(
        db_path=DB_PATH,
        helpscout_ids=[str(d["id"]) for d in drafts],
    )
    for draft in drafts:
        local = draft_logs.get(str(draft["id"]), {})
        draft.update({
            "confidence_score": local.get("confidence_score"),
            "confidence_status": local.get("confidence_status"),
            "confidence_attempts": local.get("confidence_attempts") or 1,
            "confidence_summary": local.get("confidence_summary"),
            "auto_send_ready": _is_auto_send_ready(local),
        })

    # Get knowledge base stats by source
    kb_stats = processor.vector_store.count_by_source()
    warning = get_claude_warning()

    # Cheap count (DB only, no pipeline re-runs) so the dashboard can show the
    # "Recover stuck tickets" button only when there's actually something to recover.
    try:
        stuck_count = len(find_casualties(DB_PATH, default_since()))
    except Exception:  # noqa: BLE001 - never let this break the queue
        stuck_count = 0

    return QueueResponse(
        drafts=[QueueItem(**d) for d in drafts],
        follow_ups=[QueueItem(**f) for f in follow_ups],
        stats=ExtendedStats(**stats),
        knowledge=KnowledgeStats(
            docs=kb_stats.get("docs", 0),
            examples=kb_stats.get("examples", 0),
            internal=kb_stats.get("internal", 0)
        ),
        warning=QueueWarning(**warning) if warning else None,
        stuck_count=stuck_count,
    )


@router.post("/refresh-knowledge")
async def refresh_knowledge(_: bool = Depends(verify_admin_key)):
    """Re-index knowledge base (examples and internal notes)."""
    processor = get_processor()
    count = processor.index_knowledge()
    return {"status": "ok", "indexed": count}


@router.post("/index-docs")
async def index_docs(_: bool = Depends(verify_admin_key)):
    """Index documentation from pre-crawled docs.json file."""
    processor = get_processor()
    count = processor.index_docs()
    return {"status": "ok", "indexed": count}


@router.get("/search")
async def search_knowledge(q: str, theme: Optional[str] = None, _: bool = Depends(verify_admin_key)):
    """Search the knowledge base."""
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    processor = get_processor()
    results = processor.vector_store.search(query=q, theme_filter=theme, limit=10)
    return {"results": results}


@router.get("/live-runs", response_model=LiveRunsResponse)
async def get_live_runs(_: bool = Depends(verify_admin_key)):
    registry = get_live_runs_registry()
    return LiveRunsResponse(runs=[LiveRunItem(**run) for run in registry.list_active_runs()])


@router.get("/live-runs/{run_id}/events", response_model=LiveRunEventsResponse)
async def get_live_run_events(run_id: str, after: int = -1, _: bool = Depends(verify_admin_key)):
    registry = get_live_runs_registry()
    return LiveRunEventsResponse(
        events=registry.get_events(run_id, after_index=after),
        run=registry.get_run(run_id),
    )


@router.post("/run-ticket", response_model=RunTicketResponse)
async def run_ticket(request: TestTicketRequest, _: bool = Depends(verify_admin_key)):
    registry = get_live_runs_registry()
    run_id = registry.start_run(ticket_id=request.conversation_id, source="run")
    asyncio.create_task(run_ticket_job(request.conversation_id, run_id))
    return RunTicketResponse(status="queued", run_id=run_id)


@router.post("/test-ticket", response_model=TestTicketResponse)
async def test_ticket(request: TestTicketRequest, _: bool = Depends(verify_admin_key)):
    """Test processing a specific HelpScout ticket.

    Set dry_run=false to actually post the response to HelpScout.
    """
    return await evaluate_ticket(request.conversation_id)


async def evaluate_ticket(conversation_id: int) -> TestTicketResponse:
    """Run the dashboard's read-only ticket evaluation for either caller."""
    processor = get_processor()

    # Fetch conversation from HelpScout
    try:
        conversation = await processor.helpscout.get_conversation(conversation_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not fetch conversation: {e}")

    # Parse the API response
    ticket = parse_api_conversation(conversation)

    # Classify
    classification = await classify_ticket(
        subject=ticket["subject"],
        message=ticket["message"]
    )

    # Generate improved subject
    new_subject = await generate_subject(
        subject=ticket["subject"],
        message=ticket["message"],
        theme_tag=ticket["theme_tag"]
    )
    # Only keep if different from original
    if new_subject == ticket["subject"]:
        new_subject = None

    # Search knowledge base (sync) + customer data & past tickets (async, in parallel)
    import asyncio

    customer_task = asyncio.create_task(
        fetch_customer_data(ticket.get("customer_email", ""))
    )
    # Same as live processing (services/processor.py step 3a): a theme inferred from the
    # customer's history opens repos and labels prompts, never narrows KB/past tickets.
    theme_hint = await theme_hint_for_ticket(
        ticket=ticket, helpscout=processor.helpscout, customer_data_task=customer_task
    )
    evidence_theme = ticket["theme_tag"] or (theme_hint.slug if theme_hint else "")
    similar_task = asyncio.create_task(
        find_similar_tickets(
            helpscout_client=processor.helpscout,
            theme_tag=ticket["theme_tag"],
            subject=ticket["subject"],
            message=ticket["message"],
            exclude_id=ticket["id"],
            limit=2
        )
    )

    knowledge_results = processor.vector_store.search(
        query=f"{ticket['subject']} {ticket['message']}",
        theme_filter=ticket["theme_tag"] if ticket["theme_tag"] else None,
        limit=3
    )

    knowledge_context = "\n\n---\n\n".join([
        f"**{r.get('title', 'Document')}**\n{'URL: ' + r['url'] if r.get('url') else '(no URL available)'}\n{r['content'][:500]}..."
        for r in knowledge_results
    ]) if knowledge_results else ""

    customer_data, similar_tickets = await asyncio.gather(customer_task, similar_task)
    customer_context = format_customer_context(customer_data) if customer_data else ""
    if theme_hint:
        customer_context = merge_context_sections(theme_hint.context_block(), customer_context)
    past_tickets_context = format_past_tickets_context(similar_tickets)
    conversation_history_context = json.dumps(ticket.get("conversation_history", []))
    requirements_context = get_classic_theme_requirements_context(
        # Adds the (inferred) theme's requirements article; unlike the KB filter it hides nothing.
        theme_tag=evidence_theme,
        subject=ticket["subject"],
        message=ticket["message"],
        internal_note=ticket["internal_note"],
        conversation_history=conversation_history_context,
        docs_dir=PACK_DIR / "docs",
    )
    repo_context = get_repo_evidence_context(
        theme_tag=evidence_theme,
        subject=ticket["subject"],
        message=ticket["message"],
    )
    knowledge_context = merge_context_sections(requirements_context, knowledge_context)
    combined_knowledge_context = merge_context_sections(knowledge_context, repo_context)

    response_html = None
    confidence_score = None
    confidence_status = None
    confidence_attempts = 1
    confidence_summary = None
    confidence_trace: List[Dict[str, Any]] = []
    confidence_provider = None
    delivery_mode = None
    decision_note = None
    investigator_provider = None
    evaluator_provider = None
    evidence_summary = None
    investigation_steps: List[str] = []
    risk_flags: List[str] = []
    hard_blockers: List[str] = []
    sources: List[str] = []

    if classification == "support":
        semantic_failure_matches = get_recent_semantic_failures(
            db_path=DB_PATH,
            theme_tag=ticket["theme_tag"] or None,
            limit=20,
        )
        first_name = ticket["customer_name"].split()[0] if ticket["customer_name"] else "there"
        support_context = SupportPipelineContext(
            customer_name=first_name,
            subject=ticket["subject"],
            message=ticket["message"],
            theme_tag=evidence_theme,
            theme_source=theme_hint.source if theme_hint else "",
            knowledge_context=combined_knowledge_context,
            past_tickets_context=past_tickets_context,
            customer_context=customer_context,
            internal_note=ticket["internal_note"],
            conversation_history=conversation_history_context,
            semantic_failure_matches=semantic_failure_matches,
            attachments=ticket.get("attachments") or [],
        )
        # run_support_pipeline runs the support agent's drafting (Opus, then Sonnet, then
        # Codex) + the send-confidence judge via blocking subprocess.run (minutes). Offload to a thread so the single uvicorn
        # worker keeps answering /health; otherwise the runtime watchdog times out and
        # restarts the app mid-request. Mirrors processor.process_ticket.
        pipeline_result = await asyncio.to_thread(
            run_support_pipeline,
            context=support_context,
            evaluator_provider=_get_support_provider_env("SUPA_SUPPORT_EVALUATOR", "claude"),
            target_score=_get_support_autosend_threshold(),
            deep_evidence=deep_evidence_collector(processor=processor, ticket=ticket),
        )
        response_html = pipeline_result.reply_html
        confidence_score = pipeline_result.confidence_score
        confidence_status = pipeline_result.confidence_verdict
        confidence_attempts = pipeline_result.attempts
        confidence_summary = pipeline_result.evaluator_result.summary
        confidence_trace = pipeline_result.trace
        confidence_provider = pipeline_result.evaluator_provider
        delivery_mode = pipeline_result.delivery_mode
        decision_note = pipeline_result.decision_note
        investigator_provider = pipeline_result.investigator_provider
        evaluator_provider = pipeline_result.evaluator_provider
        evidence_summary = pipeline_result.evidence_summary
        investigation_steps = list(pipeline_result.investigation_steps)
        risk_flags = list(pipeline_result.risk_flags)
        hard_blockers = list(pipeline_result.hard_blockers)
        sources = list(pipeline_result.investigator_result.sources)

    return TestTicketResponse(
        ticket=ticket,
        classification=classification,
        new_subject=new_subject,
        knowledge_results=knowledge_results,
        similar_tickets=similar_tickets,
        response_html=response_html,
        posted=False,
        customer_data=customer_data,
        confidence_score=confidence_score,
        confidence_status=confidence_status,
        confidence_attempts=confidence_attempts,
        confidence_summary=confidence_summary,
        confidence_trace=confidence_trace,
        confidence_provider=confidence_provider,
        delivery_mode=delivery_mode,
        decision_note=decision_note,
        investigator_provider=investigator_provider,
        evaluator_provider=evaluator_provider,
        evidence_summary=evidence_summary,
        investigation_steps=investigation_steps,
        risk_flags=risk_flags,
        hard_blockers=hard_blockers,
        sources=sources,
    )


@router.post("/post-cached")
async def post_cached(request: PostCachedRequest, _: bool = Depends(verify_admin_key)):
    """Post a previously tested ticket result to HelpScout."""
    processor = get_processor()

    # Build tags to add
    tags_to_add = ["supa-processed"]
    if request.theme_tag and request.theme_tag not in request.existing_tags:
        tags_to_add.append(request.theme_tag)
    if request.confidence_status == "manual_review" and "supa-manual-review" not in tags_to_add:
        tags_to_add.append("supa-manual-review")

    owner = f"dashboard-post-cached:{request.conversation_id}:{uuid.uuid4().hex}"
    async with conversation_lease(DB_PATH, request.conversation_id, owner=owner) as lease:
        existing_tags = list(request.existing_tags)
        get_conversation = getattr(processor.helpscout, "get_conversation", None)
        if callable(get_conversation):
            current = await get_conversation(request.conversation_id)
            current_tags = current.get("tags") if isinstance(current, dict) else None
            if isinstance(current_tags, list):
                existing_tags = [
                    tag.get("tag") if isinstance(tag, dict) else tag
                    for tag in current_tags
                    if isinstance(tag, str)
                    or (isinstance(tag, dict) and isinstance(tag.get("tag"), str))
                ]
        # Update subject if improved
        if request.new_subject:
            await lease.run_guarded(
                processor.helpscout.update_subject,
                request.conversation_id,
                request.new_subject,
            )

        # Live-send for auto-send outcomes, otherwise post as draft.
        # Surface HelpScout write failures (e.g. 412 conversation-locked) as a non-200
        # so callers — including CLI agents — never report success for a
        # reply/draft that never actually landed.
        if request.send:
            ok = await lease.run_guarded(
                processor.helpscout.send_reply,
                request.conversation_id,
                request.response_html,
                customer_id=request.customer_id,
            )
            if not ok:
                raise HTTPException(status_code=502, detail="HelpScout send reply failed")
        else:
            ok = await lease.run_guarded(
                processor.helpscout.add_draft,
                request.conversation_id,
                request.response_html,
                customer_id=request.customer_id,
            )
            if not ok:
                raise HTTPException(status_code=502, detail="HelpScout draft save failed")

        if request.decision_note:
            await lease.run_guarded(
                processor.helpscout.add_note,
                request.conversation_id,
                request.decision_note,
            )

        # Add tags while the same conversation lease still owns the sequence.
        await lease.run_guarded(
            processor.helpscout.add_tags,
            request.conversation_id,
            tags_to_add,
            existing_tags,
        )

    confidence_trace = None
    if request.confidence_trace:
        confidence_trace = json.dumps(request.confidence_trace)

    log_ticket(
        db_path=DB_PATH,
        helpscout_id=str(request.conversation_id),
        subject=request.subject or request.new_subject or "",
        customer_name=request.customer_name or "",
        classification=request.classification or "",
        status="processed",
        response_posted=True,
        trigger="dashboard",
        confidence_score=request.confidence_score,
        confidence_status=request.confidence_status or "",
        confidence_attempts=request.confidence_attempts,
        confidence_summary=request.confidence_summary,
        confidence_trace=confidence_trace,
        confidence_provider=request.confidence_provider,
        draft_response_html=request.response_html,
        evidence_summary=request.evidence_summary,
        delivery_mode=request.delivery_mode,
        investigator_provider=request.investigator_provider,
        evaluator_provider=request.evaluator_provider,
        decision_note=request.decision_note,
        investigation_steps=request.investigation_steps,
        risk_flags=request.risk_flags,
        sources=request.sources,
    )

    return {"status": "ok", "posted": True}


@router.post("/reprocess-stuck")
async def reprocess_stuck(request: ReprocessRequest, _: bool = Depends(verify_admin_key)):
    """Recover tickets stranded by a Claude/Codex login/quota outage.

    Finds provider-failure casualties in the audit DB, re-runs each through the
    canonical pipeline now that the provider is healthy. send=true saves drafts
    under a hard per-run draft ceiling; it never publishes a rerun. Dry-run by default (send=false).
    """
    processor = get_processor()
    since = request.since or default_since()
    casualties = find_casualties(DB_PATH, since)

    results: List[Dict[str, Any]] = []
    for c in casualties:
        cid = int(c["helpscout_id"])
        item: Dict[str, Any] = {
            "conversation_id": cid,
            "subject": c.get("subject") or "",
            "prior_score": c.get("confidence_score"),
        }
        try:
            conv = await processor.helpscout.get_conversation(cid)
        except Exception as e:  # noqa: BLE001
            item.update(action="skipped", reason=f"fetch failed: {e}")
            results.append(item)
            continue

        if is_supa_ignored(conv):
            item.update(action="skipped", reason="supa-ignore: manual handling")
            results.append(item)
            continue

        status = conv.get("status")
        threads = conv.get("_embedded", {}).get("threads", []) or []
        waiting = customer_waiting(threads)
        if status in ("closed", "spam"):
            item.update(action="skipped", reason=f"already {status}")
            results.append(item)
            continue
        if not waiting:
            item.update(action="skipped", reason="agent already replied")
            results.append(item)
            continue

        if request.send:
            # Bulk recovery prepares reviewed work; it cannot sample a previously
            # held obligation into an immediate send through post_cached.
            outcome = await processor.process_ticket(conv, trigger="manual_run", delivery_ceiling="draft")
            item.update(new_score=outcome.get("confidence_score"), would="draft",
                        action="drafted" if outcome.get("response_posted") else "skipped",
                        reason=outcome.get("reason") or outcome.get("error"))
        else:
            fresh = await test_ticket(TestTicketRequest(conversation_id=cid), _=True)
            item.update(new_score=fresh.confidence_score, would="draft", action="dry_run")
        results.append(item)

    return {
        "since": since,
        "send": request.send,
        "count": len(casualties),
        "results": results,
    }


@router.get("/drafts", response_model=List[DraftDetail])
async def get_drafts(_: bool = Depends(verify_admin_key)):
    """Get drafts with full details for review."""
    processor = get_processor()
    drafts = await processor.helpscout.get_conversations_with_drafts(
        limit=10,
        include_details=True
    )
    draft_logs = get_ticket_logs_by_helpscout_ids(
        db_path=DB_PATH,
        helpscout_ids=[str(d["id"]) for d in drafts],
    )
    for draft in drafts:
        local = draft_logs.get(str(draft["id"]), {})
        draft["confidence_score"] = local.get("confidence_score")
        draft["confidence_status"] = local.get("confidence_status")
        draft["confidence_attempts"] = local.get("confidence_attempts") or 1
        draft["confidence_summary"] = local.get("confidence_summary")
        draft["confidence_provider"] = local.get("confidence_provider")
        trace_raw = local.get("confidence_trace")
        draft["confidence_trace"] = []
        if trace_raw:
            try:
                import json
                draft["confidence_trace"] = json.loads(trace_raw)
            except Exception:
                draft["confidence_trace"] = []
    return [DraftDetail(**d) for d in drafts]


@router.post("/send-reply")
async def send_reply(request: SendReplyRequest, _: bool = Depends(verify_admin_key)):
    """Send a reply directly (not as draft) with status set to pending.

    If there's an existing draft, it will be deleted first.
    Also indexes the Q&A pair for future learning.
    """
    processor = get_processor()

    owner = f"dashboard-send-reply:{request.conversation_id}:{uuid.uuid4().hex}"
    async with conversation_lease(DB_PATH, request.conversation_id, owner=owner) as lease:
        # Fetch under the lease so draft IDs, tags, and customer identity are current.
        conversation = await processor.helpscout.get_conversation(request.conversation_id)
        conv_data = parse_api_conversation(conversation)
        current_tags = list(conv_data.get("tags", []))
        draft_thread_ids = get_draft_thread_ids(
            conversation.get("_embedded", {}).get("threads", [])
        )
        if request.draft_thread_id and request.draft_thread_id not in draft_thread_ids:
            draft_thread_ids.append(request.draft_thread_id)

        # Delete all known draft threads before sending the final reply.
        for draft_thread_id in draft_thread_ids:
            await lease.run_guarded(
                processor.helpscout.delete_draft,
                request.conversation_id,
                draft_thread_id,
            )

        if request.new_subject:
            await lease.run_guarded(
                processor.helpscout.update_subject,
                request.conversation_id,
                request.new_subject,
            )

        sent = await lease.run_guarded(
            processor.helpscout.send_reply,
            request.conversation_id,
            request.reply_html,
            status="pending",
            customer_id=conv_data.get("customer_id"),
        )
        if not sent:
            raise HTTPException(status_code=502, detail="HelpScout send reply failed")

        pending_result = await lease.run_guarded(
            processor.helpscout.ensure_conversation_status,
            request.conversation_id,
            "pending",
        )
        if not pending_result.get("verified"):
            raise HTTPException(status_code=502, detail="HelpScout pending status update failed")

        cleaned_existing_tags = [
            tag for tag in current_tags if tag != "supa-manual-review"
        ]
        tags_to_add = ["supa-processed"]
        if request.theme_tag and request.theme_tag not in cleaned_existing_tags:
            tags_to_add.append(request.theme_tag)
        if "supa-auto-sent" not in cleaned_existing_tags:
            tags_to_add.append("supa-auto-sent")

        await lease.run_guarded(
            processor.helpscout.add_tags,
            request.conversation_id,
            tags_to_add,
            cleaned_existing_tags,
        )

    log_ticket(
        db_path=DB_PATH,
        helpscout_id=str(request.conversation_id),
        subject=request.new_subject or conv_data.get("subject", ""),
        customer_name=conv_data.get("customer_name", ""),
        classification="support",
        status="processed",
        response_posted=True,
        trigger="dashboard",
        draft_response_html=request.reply_html,
        delivery_mode="auto_send",
    )

    # Index the sent reply for future learning
    processor.index_sent_reply(
        conversation_id=request.conversation_id,
        subject=request.new_subject or conv_data.get("subject", ""),
        customer_message=conv_data.get("message", ""),
        reply_text=request.reply_html,
        theme_tag=request.theme_tag
    )

    return {"status": "ok", "sent": True, "learned": True}


@router.get("/list/{ticket_type}")
async def list_tickets(ticket_type: str, _: bool = Depends(verify_admin_key)):
    """List tickets by type for the last 7 days.

    Types: received, auto_closed, drafts, manual
    """
    valid_types = ["received", "auto_closed", "drafts", "manual"]
    if ticket_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid type. Use: {', '.join(valid_types)}")

    tickets = get_tickets_by_type(db_path=DB_PATH, ticket_type=ticket_type, days=7)

    return {
        "type": ticket_type,
        "count": len(tickets),
        "tickets": [
            {
                "id": t["helpscout_id"],
                "subject": t["subject"] or "No subject",
                "customer": t["customer_name"] or "Unknown",
                "classification": t["classification"] or "",
                "status": t["status"] or "",
                "response_posted": bool(t["response_posted"]),
                "created_at": t["created_at"]
            }
            for t in tickets
        ]
    }
