import json
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agents.draft_confidence import DraftConfidenceResult


RETRY_THRESHOLD = int(os.getenv("SUPA_CONFIDENCE_RETRY_THRESHOLD", "8"))
TARGET_SCORE = int(os.getenv("SUPA_CONFIDENCE_TARGET_SCORE", "9"))
MAX_ATTEMPTS = int(os.getenv("SUPA_CONFIDENCE_MAX_ATTEMPTS", "3"))


@dataclass
class DraftLoopResult:
    final_html: str | None
    final_score: int
    final_status: str
    attempts: int
    summary: str
    trace: list[dict[str, Any]]
    provider: str = "claude"
    final_context: dict[str, Any] = field(default_factory=dict)


def _trace_entry(attempt: int, draft_html: str | None, evaluation: DraftConfidenceResult) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "draft_html": draft_html,
        "score": evaluation.score,
        "verdict": evaluation.verdict,
        "summary": evaluation.summary,
        "missing_information": evaluation.missing_information,
        "investigation_actions": evaluation.investigation_actions,
        "rewrite_brief": evaluation.rewrite_brief,
        "risk_flags": evaluation.risk_flags,
        "provider": evaluation.provider,
    }


async def run_draft_confidence_loop(
    *,
    initial_html: str | None,
    initial_context: dict[str, Any],
    evaluate_fn: Callable[..., DraftConfidenceResult],
    investigate_fn: Callable[..., Awaitable[dict[str, Any]]],
    rewrite_fn: Callable[..., Awaitable[str | None]],
    guard_fn: Callable[..., DraftConfidenceResult] | None = None,
    retry_threshold: int = RETRY_THRESHOLD,
    target_score: int = TARGET_SCORE,
    max_attempts: int = MAX_ATTEMPTS,
) -> DraftLoopResult:
    """Evaluate a draft, optionally investigate more, rewrite, and retry."""

    current_html = initial_html
    current_context = dict(initial_context)
    best_html = initial_html
    best_evaluation: DraftConfidenceResult | None = None
    trace: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        evaluation = evaluate_fn(draft_html=current_html, **current_context)
        if guard_fn is not None:
            evaluation = guard_fn(draft_html=current_html, evaluation=evaluation, **current_context)
        trace.append(_trace_entry(attempt, current_html, evaluation))

        if best_evaluation is None or evaluation.score >= best_evaluation.score:
            best_evaluation = evaluation
            best_html = current_html

        if evaluation.score >= target_score:
            return DraftLoopResult(
                final_html=current_html,
                final_score=evaluation.score,
                final_status="accepted",
                attempts=attempt,
                summary=evaluation.summary,
                trace=trace,
                provider=evaluation.provider,
                final_context=current_context,
            )

        if evaluation.score >= retry_threshold:
            return DraftLoopResult(
                final_html=current_html,
                final_score=evaluation.score,
                final_status="accepted",
                attempts=attempt,
                summary=evaluation.summary,
                trace=trace,
                provider=evaluation.provider,
                final_context=current_context,
            )

        if attempt == max_attempts:
            assert best_evaluation is not None
            return DraftLoopResult(
                final_html=best_html,
                final_score=best_evaluation.score,
                final_status="manual_review",
                attempts=attempt,
                summary=best_evaluation.summary,
                trace=trace,
                provider=best_evaluation.provider,
                final_context=current_context,
            )

        investigation = await investigate_fn(
            evaluation=evaluation,
            attempt=attempt,
            context=current_context,
        )
        current_context = {
            **current_context,
            **(investigation or {}),
        }

        additional_context = ""
        if investigation:
            additional_context = investigation.get("additional_context", "")
            if not additional_context:
                additional_context = json.dumps(investigation, ensure_ascii=True, sort_keys=True)

        rewrite_context = dict(current_context)
        rewrite_context.pop("additional_context", None)

        rewritten = await rewrite_fn(
            original_draft=current_html or "",
            rewrite_brief=evaluation.rewrite_brief or evaluation.summary,
            additional_context=additional_context,
            **rewrite_context,
        )
        if rewritten:
            current_html = rewritten

    assert best_evaluation is not None
    return DraftLoopResult(
        final_html=best_html,
        final_score=best_evaluation.score,
        final_status="manual_review",
        attempts=max_attempts,
        summary=best_evaluation.summary,
        trace=trace,
        provider=best_evaluation.provider,
        final_context=current_context,
    )
