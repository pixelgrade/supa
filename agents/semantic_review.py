import json
import re
from dataclasses import dataclass

from agents.claude_cli import invoke_claude
from agents.support_agent import LEARN_MODEL


SEMANTIC_REVIEW_SYSTEM_PROMPT = """You compare a support draft against the final human reply.

Return ONLY valid JSON with these fields:
- same_meaning: boolean
- change_type: one of "none", "supported_vs_unsupported", "built_in_vs_custom", "required_vs_optional", "wrong_root_cause", "missing_critical_constraint", "unknown"
- summary: short string
- wrong_claim: short string
- correct_claim: short string

Rules:
- Use same_meaning=true when the final human reply only changes tone, wording, formatting, or brevity.
- Use same_meaning=false only when the final human reply materially changes the technical meaning.
- If same_meaning=true, set change_type to "none" and leave wrong_claim/correct_claim empty.
- Do not include markdown fences or extra commentary."""


@dataclass(eq=True)
class SemanticReviewResult:
    same_meaning: bool
    change_type: str
    summary: str
    wrong_claim: str = ""
    correct_claim: str = ""
    provider: str = "claude"


def _strip_code_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text.strip()


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def parse_semantic_review_result(raw: str, provider: str = "claude") -> SemanticReviewResult:
    data = json.loads(_strip_code_fences(raw))

    same_meaning = _normalize_bool(data.get("same_meaning", True))
    summary = str(data.get("summary", "")).strip()

    if same_meaning:
        return SemanticReviewResult(
            same_meaning=True,
            change_type="none",
            summary=summary,
            wrong_claim="",
            correct_claim="",
            provider=provider,
        )

    change_type = str(data.get("change_type", "unknown")).strip() or "unknown"
    return SemanticReviewResult(
        same_meaning=False,
        change_type=change_type,
        summary=summary,
        wrong_claim=str(data.get("wrong_claim", "")).strip(),
        correct_claim=str(data.get("correct_claim", "")).strip(),
        provider=provider,
    )


def fallback_semantic_review_result(provider: str, summary: str) -> SemanticReviewResult:
    return SemanticReviewResult(
        same_meaning=True,
        change_type="unknown",
        summary=summary,
        provider=provider,
    )


def build_semantic_review_prompt(
    *,
    subject: str,
    customer_message: str,
    supa_draft_html: str,
    final_reply_html: str,
    theme_tag: str = "",
    confidence_summary: str = "",
    evidence_summary: str = "",
) -> str:
    parts = [
        f"Subject: {subject}",
        f"Theme: {theme_tag or 'not specified'}",
        "",
        "Customer message:",
        customer_message,
        "",
        "Supa draft reply:",
        supa_draft_html,
        "",
        "Final human reply:",
        final_reply_html,
    ]

    if confidence_summary:
        parts.extend(["", "Confidence summary:", confidence_summary])
    if evidence_summary:
        parts.extend(["", "Evidence summary:", evidence_summary])

    parts.extend([
        "",
        "Decide whether the final human reply materially changed the meaning of the Supa draft.",
    ])
    return "\n".join(parts)


def review_sent_reply(
    *,
    subject: str,
    customer_message: str,
    supa_draft_html: str,
    final_reply_html: str,
    theme_tag: str = "",
    confidence_summary: str = "",
    evidence_summary: str = "",
    provider: str = "claude",
    timeout: int = 180,
) -> SemanticReviewResult:
    prompt = build_semantic_review_prompt(
        subject=subject,
        customer_message=customer_message,
        supa_draft_html=supa_draft_html,
        final_reply_html=final_reply_html,
        theme_tag=theme_tag,
        confidence_summary=confidence_summary,
        evidence_summary=evidence_summary,
    )

    provider = (provider or "claude").strip().lower()
    try:
        if provider != "claude":
            return fallback_semantic_review_result(provider, f"Unsupported semantic review provider: {provider}")

        raw = invoke_claude(
            prompt=prompt,
            model=LEARN_MODEL,
            system_prompt=SEMANTIC_REVIEW_SYSTEM_PROMPT,
            timeout=timeout,
        )
        return parse_semantic_review_result(raw, provider=provider)
    except Exception as exc:
        return fallback_semantic_review_result(provider, f"Semantic review failed: {exc}")
