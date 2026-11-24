import asyncio
import json

from services.company_pack import prompt
from agents.claude_cli import invoke_claude, should_skip_claude
from agents.codex_cli import invoke_codex
from agents.support_agent import CLASSIFY_MODEL

SUBJECT_GENERATOR_PROMPT = prompt("subject_generator_prompt")

CLASSIFIER_PROMPT = prompt("classifier_prompt")

CATEGORIES = ("support", "admin", "marketing", "spam", "wrong_address")

WRONG_ADDRESS_CHECK_PROMPT = prompt("wrong_address_check_prompt")

WRONG_ADDRESS_CHECK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["wrong_address", "confidence", "reason"],
    "properties": {
        "wrong_address": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
}


CLASSIFIER_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["classification"],
    "properties": {
        "classification": {
            "type": "string",
            "enum": list(CATEGORIES),
        }
    },
}


SUBJECT_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["subject"],
    "properties": {
        "subject": {"type": "string"},
    },
}


def _normalize_classification(result: str) -> str:
    normalized = (result or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in CATEGORIES:
        return normalized
    raise ValueError(f"Invalid classification: {result!r}")


def _normalize_subject(result: str) -> str:
    normalized = (result or "").strip().strip('"\'')
    if 5 < len(normalized) < 100:
        return normalized
    raise ValueError(f"Invalid generated subject: {result!r}")


def _codex_classifier_prompt(user_prompt: str) -> str:
    return (
        f"{CLASSIFIER_PROMPT}\n\n"
        "Return strict JSON matching the required schema.\n\n"
        f"{user_prompt}"
    )


def _codex_subject_prompt(user_prompt: str) -> str:
    return (
        f"{SUBJECT_GENERATOR_PROMPT}\n\n"
        "Return strict JSON matching the required schema.\n\n"
        f"{user_prompt}"
    )


def _classify_with_codex(user_prompt: str) -> str:
    raw = invoke_codex(
        prompt=_codex_classifier_prompt(user_prompt),
        output_schema=CLASSIFIER_OUTPUT_SCHEMA,
        timeout=60,
    )
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Codex classifier output must be a JSON object")
    return _normalize_classification(str(data.get("classification", "")))


def _subject_with_codex(user_prompt: str) -> str:
    raw = invoke_codex(
        prompt=_codex_subject_prompt(user_prompt),
        output_schema=SUBJECT_OUTPUT_SCHEMA,
        timeout=60,
    )
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Codex subject output must be a JSON object")
    return _normalize_subject(str(data.get("subject", "")))


async def classify_ticket(subject: str, message: str) -> str:
    """Classify a ticket using Claude first, then Codex as fallback."""

    user_prompt = f"""Subject: {subject}

Message:
{message}"""

    try:
        if should_skip_claude():
            raise RuntimeError("Claude is in outage cooldown")
        result = await asyncio.to_thread(
            invoke_claude,
            prompt=user_prompt,
            model=CLASSIFY_MODEL,
            system_prompt=CLASSIFIER_PROMPT,
            timeout=60,
        )
        return _normalize_classification(result)

    except Exception as e:
        print(f"Classification error: {e}")
        try:
            return await asyncio.to_thread(_classify_with_codex, user_prompt)
        except Exception as codex_error:
            print(f"Classification codex fallback error: {codex_error}")
            return "support"


def _parse_wrong_address_check(raw: str) -> tuple[bool, float, str]:
    import re

    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        raise ValueError("No JSON object in wrong-address check output")
    data = json.loads(match.group(0))
    if not isinstance(data, dict) or type(data.get("wrong_address")) is not bool:
        raise ValueError("Invalid wrong-address check output")
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        raise ValueError("Invalid wrong-address confidence")
    confidence = float(confidence)
    return data["wrong_address"], confidence, str(data.get("reason") or "").strip()[:300]


async def confirm_wrong_address(subject: str, message: str) -> tuple[bool | None, float, str]:
    """Second, independent reading of a message the classifier called wrong_address.

    Returns (is_wrong_address, confidence, reason). When no provider could read it the
    first value is None: the caller must hold, neither sending the reply nor treating the
    silence as a "no".
    """
    user_prompt = f"""Subject: {subject}

Message:
{message[:6000]}"""
    try:
        if should_skip_claude():
            raise RuntimeError("Claude is in outage cooldown")
        raw = await asyncio.to_thread(
            invoke_claude,
            prompt=user_prompt,
            model=CLASSIFY_MODEL,
            system_prompt=WRONG_ADDRESS_CHECK_PROMPT,
            timeout=60,
        )
        return _parse_wrong_address_check(raw)
    except Exception as e:
        print(f"Wrong-address check error: {e}")
        try:
            raw = await asyncio.to_thread(
                invoke_codex,
                prompt=f"{WRONG_ADDRESS_CHECK_PROMPT}\n\nReturn strict JSON matching the required schema.\n\n{user_prompt}",
                output_schema=WRONG_ADDRESS_CHECK_SCHEMA,
                timeout=60,
            )
            return _parse_wrong_address_check(raw)
        except Exception as codex_error:
            print(f"Wrong-address check codex fallback error: {codex_error}")
            return None, 0.0, "Check unavailable."


async def generate_subject(subject: str, message: str, theme_tag: str = "") -> str:
    """Generate a better, searchable subject with Claude first, then Codex."""

    generic_subjects = ["help", "question", "issue", "problem", "support", "hi", "hello", "re:"]
    subject_lower = subject.lower().strip()
    if len(subject) > 30 and not any(g in subject_lower for g in generic_subjects):
        return subject

    context = f"Theme: {theme_tag}\n" if theme_tag else ""
    user_prompt = f"""{context}Original subject: {subject}

Message:
{message[:500]}"""

    try:
        if should_skip_claude():
            raise RuntimeError("Claude is in outage cooldown")
        new_subject = await asyncio.to_thread(
            invoke_claude,
            prompt=user_prompt,
            model=CLASSIFY_MODEL,
            system_prompt=SUBJECT_GENERATOR_PROMPT,
            timeout=60,
        )
        return _normalize_subject(new_subject)

    except Exception as e:
        print(f"Subject generation error: {e}")
        try:
            return await asyncio.to_thread(_subject_with_codex, user_prompt)
        except Exception as codex_error:
            print(f"Subject generation codex fallback error: {codex_error}")
            return subject
