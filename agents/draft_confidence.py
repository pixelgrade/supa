import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from services.company_pack import prompt
from agents.claude_cli import invoke_claude
from agents.codex_cli import invoke_codex, scoped_eval_working_dir
from agents.openclaw_cli import invoke_openclaw
from agents.support_agent import load_rules


# The send-confidence judge is its own role, separate from the support agent that writes
# the draft. Its model is configurable.
JUDGE_MODEL_ENV = "SUPA_SUPPORT_JUDGE_MODEL"
DEFAULT_JUDGE_MODEL = "sonnet"


def judge_model() -> str:
    return (os.getenv(JUDGE_MODEL_ENV) or "").strip() or DEFAULT_JUDGE_MODEL


CONFIDENCE_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "score",
        "verdict",
        "summary",
        "missing_information",
        "investigation_actions",
        "rewrite_brief",
        "risk_flags",
        "hard_blockers",
    ],
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "verdict": {"type": "string"},
        "summary": {"type": "string"},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "investigation_actions": {"type": "array", "items": {"type": "string"}},
        "rewrite_brief": {"type": "string"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "hard_blockers": {"type": "array", "items": {"type": "string"}},
    },
}


CONFIDENCE_SYSTEM_PROMPT = prompt("confidence_system_prompt", schema=CONFIDENCE_OUTPUT_SCHEMA)


@dataclass
class DraftConfidenceResult:
    score: int
    verdict: str
    summary: str
    missing_information: list[str] = field(default_factory=list)
    investigation_actions: list[str] = field(default_factory=list)
    rewrite_brief: str = ""
    risk_flags: list[str] = field(default_factory=list)
    hard_blockers: list[str] = field(default_factory=list)
    provider: str = "claude"


def _require_string_list(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected list[str] field: {field_name}")

    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Expected list[str] field: {field_name}")
        stripped = item.strip()
        if stripped:
            cleaned.append(stripped)
    return cleaned


def _strip_code_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text.strip()


def _normalize_verdict(verdict: str, score: int) -> str:
    normalized = (verdict or "").strip().lower()
    if normalized in {"accept", "retry", "manual_review"}:
        return normalized
    if score >= 8:
        return "accept"
    return "retry"


def parse_confidence_result(raw: str, provider: str = "claude") -> DraftConfidenceResult:
    data = json.loads(_strip_code_fences(raw))
    if not isinstance(data, dict):
        raise ValueError("Confidence evaluator output must be a JSON object")

    score = int(data.get("score", 0))
    score = max(0, min(10, score))

    return DraftConfidenceResult(
        score=score,
        verdict=_normalize_verdict(data.get("verdict", ""), score),
        summary=str(data.get("summary", "")).strip(),
        missing_information=_require_string_list(
            data.get("missing_information", []),
            "missing_information",
        ),
        investigation_actions=_require_string_list(
            data.get("investigation_actions", []),
            "investigation_actions",
        ),
        rewrite_brief=str(data.get("rewrite_brief", "")).strip(),
        risk_flags=_require_string_list(data.get("risk_flags", []), "risk_flags"),
        hard_blockers=_require_string_list(data.get("hard_blockers", []), "hard_blockers"),
        provider=provider,
    )


def fallback_confidence_result(provider: str, summary: str) -> DraftConfidenceResult:
    return DraftConfidenceResult(
        score=0,
        verdict="manual_review",
        summary=summary,
        provider=provider,
        risk_flags=["evaluator_failure"],
        hard_blockers=["malformed_provider_output"],
    )


def build_confidence_prompt(
    *,
    customer_name: str,
    subject: str,
    message: str,
    draft_html: str,
    theme_tag: str = "",
    knowledge_context: str = "",
    past_tickets_context: str = "",
    customer_context: str = "",
    conversation_history: str = "",
    internal_note: str = "",
    investigation_summary: str = "",
    evidence_summary: str = "",
    investigation_steps: list[str] | None = None,
    investigator_risk_flags: list[str] | None = None,
    investigator_sources: list[str] | None = None,
) -> str:
    parts = [
        f"Customer name: {customer_name}",
        f"Subject: {subject}",
        f"Theme: {theme_tag or 'not specified'}",
        "",
        "Latest customer message:",
        message,
        "",
        "Draft reply HTML:",
        draft_html,
        "",
        "Knowledge context:",
        knowledge_context or "(none)",
    ]

    if past_tickets_context:
        parts.extend(["", "Past tickets context:", past_tickets_context])
    if customer_context:
        parts.extend(["", "Customer context:", customer_context])
    if conversation_history:
        parts.extend(["", "Conversation history:", conversation_history])
    if internal_note:
        parts.extend(["", "Internal note:", internal_note])
    if investigation_summary:
        parts.extend(["", "Investigator summary:", investigation_summary])
    if evidence_summary:
        parts.extend(["", "Investigator evidence summary:", evidence_summary])
    if investigation_steps:
        parts.extend(["", "Investigator steps:", "\n".join(f"- {step}" for step in investigation_steps)])
    if investigator_risk_flags:
        parts.extend(["", "Investigator risk flags:", "\n".join(f"- {flag}" for flag in investigator_risk_flags)])
    if investigator_sources:
        parts.extend(["", "Investigator sources:", "\n".join(f"- {source}" for source in investigator_sources)])

    parts.extend([
        "",
        "Audit the draft against the support agent's authoritative rules:",
        load_rules(),
        "",
        "Evaluate whether the draft is accurate, direct, and supported by the available evidence.",
        "Score it and suggest the smallest useful next investigation/rewrite steps.",
    ])
    return "\n".join(parts)


def evaluate_draft(
    *,
    customer_name: str,
    subject: str,
    message: str,
    draft_html: str,
    theme_tag: str = "",
    knowledge_context: str = "",
    past_tickets_context: str = "",
    customer_context: str = "",
    conversation_history: str = "",
    internal_note: str = "",
    investigation_summary: str = "",
    evidence_summary: str = "",
    investigation_steps: list[str] | None = None,
    investigator_risk_flags: list[str] | None = None,
    investigator_sources: list[str] | None = None,
    provider: str = "claude",
    timeout: int = 180,
) -> DraftConfidenceResult:
    provider = (provider or "claude").strip().lower()
    prompt = build_confidence_prompt(
        customer_name=customer_name,
        subject=subject,
        message=message,
        draft_html=draft_html,
        theme_tag=theme_tag,
        knowledge_context=knowledge_context,
        past_tickets_context=past_tickets_context,
        customer_context=customer_context,
        conversation_history=conversation_history,
        internal_note=internal_note,
        investigation_summary=investigation_summary,
        evidence_summary=evidence_summary,
        investigation_steps=investigation_steps,
        investigator_risk_flags=investigator_risk_flags,
        investigator_sources=investigator_sources,
    )

    try:
        if provider == "openclaw":
            raw = invoke_openclaw(prompt, timeout=timeout)
        elif provider == "codex":
            # The evaluator only scores a draft — it needs no repo access, so run
            # Codex in a scoped scratch dir to skip loading project context.
            raw = invoke_codex(
                prompt=prompt,
                output_schema=CONFIDENCE_OUTPUT_SCHEMA,
                timeout=timeout,
                working_dir=scoped_eval_working_dir(),
                text_only=True,
            )
        elif provider == "claude":
            raw = invoke_claude(
                prompt=prompt,
                model=judge_model(),
                system_prompt=CONFIDENCE_SYSTEM_PROMPT,
                timeout=timeout,
            )
        else:
            return fallback_confidence_result(provider, f"Unsupported confidence provider: {provider}")

        return parse_confidence_result(raw, provider=provider)
    except Exception as exc:
        return fallback_confidence_result(provider, f"Confidence evaluator failed: {exc}")
