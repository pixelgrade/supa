"""Provider-agnostic triage agent for incoming HelpScout tickets.

This upgrades the bare `classify_ticket` string classifier into a small triage
agent that returns a structured, confidence-scored decision the processor can
act on conservatively. It deliberately reuses the existing CLI provider
wrappers (`invoke_claude`, `invoke_codex`) so it matches the rest of the
project's "providers are interchangeable infrastructure" pattern.

Safety model (the processor enforces the writes):
  - support / uncertain  -> process_support (continue the normal pipeline)
  - non-support, high confidence -> close (and tag)
  - non-support, low/medium confidence -> review (never close; tag for review)
  - any provider failure -> support (fail open to a human/pipeline)
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field

from agents.classifier import _normalize_classification
from services.company_pack import prompt
from agents.claude_cli import invoke_claude
from agents.codex_cli import invoke_codex

TRIAGE_SYSTEM_PROMPT = prompt("triage_system_prompt")


TRIAGE_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["classification", "confidence", "reason"],
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["support", "admin", "marketing", "spam"],
        },
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
}

VALID_MODES = {"shadow", "live"}
DEFAULT_AUTOCLOSE_MIN_CONFIDENCE = 0.90
REVIEW_TAG = "supa-triage-review"
WOULD_CLOSE_TAG = "supa-triage-would-close"


@dataclass(frozen=True)
class TriageConfig:
    mode: str = "shadow"
    autoclose_min_confidence: float = DEFAULT_AUTOCLOSE_MIN_CONFIDENCE
    provider: str = "claude"


@dataclass
class TriageResult:
    classification: str  # support | admin | marketing | spam
    confidence: float  # 0.0 - 1.0
    reason: str
    recommended_action: str  # process_support | close | review
    tags: list = field(default_factory=list)


def load_triage_config() -> TriageConfig:
    """Read triage settings from the environment with conservative defaults."""
    mode = (os.getenv("SUPA_TRIAGE_MODE", "shadow") or "shadow").strip().lower()
    if mode not in VALID_MODES:
        mode = "shadow"

    raw_threshold = (os.getenv("SUPA_TRIAGE_AUTOCLOSE_MIN_CONFIDENCE", "") or "").strip()
    try:
        threshold = float(raw_threshold) if raw_threshold else DEFAULT_AUTOCLOSE_MIN_CONFIDENCE
    except ValueError:
        threshold = DEFAULT_AUTOCLOSE_MIN_CONFIDENCE
    threshold = max(0.0, min(1.0, threshold))

    provider = (os.getenv("SUPA_TRIAGE_PROVIDER", "claude") or "claude").strip().lower() or "claude"

    return TriageConfig(mode=mode, autoclose_min_confidence=threshold, provider=provider)


def _coerce_confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid triage confidence: {value!r}")
    return max(0.0, min(1.0, confidence))


def _parse_triage_payload(raw: str) -> tuple[str, float, str]:
    """Parse a provider's structured triage output into (classification, confidence, reason)."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in triage output: {raw!r}")

    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Triage output must be a JSON object")

    classification = _normalize_classification(str(data.get("classification", "")))
    confidence = _coerce_confidence(data.get("confidence"))
    reason = str(data.get("reason", "") or "").strip()
    return classification, confidence, reason


def _triage_user_prompt(subject: str, message: str) -> str:
    return f"""Subject: {subject}

Message:
{message}"""


def _codex_triage_prompt(user_prompt: str) -> str:
    return (
        f"{TRIAGE_SYSTEM_PROMPT}\n\n"
        "Return strict JSON matching the required schema.\n\n"
        f"{user_prompt}"
    )


def _triage_with_claude(user_prompt: str) -> tuple[str, float, str]:
    raw = invoke_claude(
        prompt=user_prompt,
        model="haiku",
        system_prompt=TRIAGE_SYSTEM_PROMPT,
        timeout=60,
    )
    return _parse_triage_payload(raw)


def _triage_with_codex(user_prompt: str) -> tuple[str, float, str]:
    raw = invoke_codex(
        prompt=_codex_triage_prompt(user_prompt),
        output_schema=TRIAGE_OUTPUT_SCHEMA,
        timeout=60,
    )
    return _parse_triage_payload(raw)


def _run_primary_provider(provider: str, user_prompt: str) -> tuple[str, float, str]:
    # Only the CLI providers are wired today; "openai" has no clean active
    # pattern in this project, so anything other than codex uses claude.
    if provider == "codex":
        return _triage_with_codex(user_prompt)
    return _triage_with_claude(user_prompt)


def decide_action(classification: str, confidence: float, *, autoclose_min: float) -> tuple[str, list]:
    """Map a classification + confidence onto a conservative action and tags."""
    if classification == "support":
        return "process_support", []
    if confidence >= autoclose_min:
        return "close", [classification]
    return "review", [REVIEW_TAG]


async def triage_ticket(
    *,
    subject: str,
    message: str,
    config: TriageConfig | None = None,
) -> TriageResult:
    """Triage a ticket with the configured provider, falling back to Codex then support."""
    cfg = config or load_triage_config()
    user_prompt = _triage_user_prompt(subject, message)

    try:
        classification, confidence, reason = await asyncio.to_thread(
            _run_primary_provider, cfg.provider, user_prompt
        )
    except Exception as primary_error:
        print(f"Triage error: {primary_error}")
        try:
            classification, confidence, reason = await asyncio.to_thread(
                _triage_with_codex, user_prompt
            )
        except Exception as codex_error:
            print(f"Triage codex fallback error: {codex_error}")
            return TriageResult(
                classification="support",
                confidence=0.0,
                reason="Triage providers unavailable; defaulting to support.",
                recommended_action="process_support",
                tags=[],
            )

    action, tags = decide_action(
        classification, confidence, autoclose_min=cfg.autoclose_min_confidence
    )
    return TriageResult(
        classification=classification,
        confidence=confidence,
        reason=reason,
        recommended_action=action,
        tags=tags,
    )
