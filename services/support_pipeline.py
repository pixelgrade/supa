import json
import logging
import re
import time
from html import unescape
from dataclasses import replace
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from agents.draft_confidence import DraftConfidenceResult, evaluate_draft
from agents.responder import extract_context_urls, finalize_response_html
from agents.support_agent import (
    DeepPassConfig,
    DeepPassCritique,
    SupportInvestigationRequest,
    SupportInvestigationResult,
    autosend_threshold,
    deep_draft_reply,
    draft_reply,
    load_deep_pass_config,
    theme_label,
)
from services.deep_pass import (
    DeepEvidence,
    DeepEvidenceCollector,
    SOURCE_EXCERPTS_HEADING,
    cited_source_excerpts,
    matching_source_citations,
    deep_pass_skip_reason,
    new_workdir,
    remove_workdir,
    repo_dirs_for,
)
from services.github_evidence import pinned_github_excerpts
from services.helpscout import describe_attachments, material_customer_images, strip_html
from services.live_site_evidence import (
    detect_dead_css_selectors_for_context,
    live_css_evidence_for_context,
    verify_draft_css_against_repo,
)
from services.ui_label_evidence import verify_draft_ui_labels
from services.repo_evidence import (
    detect_repo_contradiction,
    extract_repo_search_terms,
    get_repo_evidence_context,
    resolve_repo_target,
)

from services.company_pack import wording, product_settings

_PRODUCTS = product_settings()
logger = logging.getLogger(__name__)


def _mentions_concierge(reply_html: str) -> bool:
    text = unescape(strip_html(reply_html or ""))
    link_pattern = _PRODUCTS["concierge_link"]
    text_pattern = _PRODUCTS["concierge_text"]
    return bool((text_pattern and re.search(text_pattern, text, re.IGNORECASE))
                or (link_pattern and re.search(link_pattern, unescape(reply_html or ""), re.IGNORECASE)))


@dataclass
class SupportPipelineContext:
    customer_name: str
    subject: str
    message: str
    theme_tag: str = ""
    # "" when the ticket names the theme; "licence"/"past tickets" when Supa inferred it
    # (services/theme_history.py): then it is a hypothesis every prompt labels as such.
    theme_source: str = ""
    knowledge_context: str = ""
    past_tickets_context: str = ""
    customer_context: str = ""
    internal_note: str = ""
    conversation_history: str = ""
    semantic_failure_matches: list[dict[str, Any]] = field(default_factory=list)
    # Metadata for the latest customer message's attachments (services/helpscout.py
    # extract_customer_attachments). Screenshots are primary evidence; the pipeline must
    # know they exist to insist they are analyzed.
    attachments: list[dict[str, Any]] = field(default_factory=list)
    duplicate_conversation_ids: list[int] = field(default_factory=list)


@dataclass
class SupportPipelineResult:
    delivery_mode: str
    confidence_score: int
    confidence_verdict: str
    reply_html: str
    decision_summary: str
    evidence_summary: str
    investigation_steps: list[str]
    risk_flags: list[str]
    hard_blockers: list[str]
    decision_note: str | None
    manual_review: bool
    attempts: int
    investigator_provider: str
    evaluator_provider: str
    investigator_result: SupportInvestigationResult
    evaluator_result: DraftConfidenceResult
    trace: list[dict[str, Any]] = field(default_factory=list)


def _contains_token(values: list[str], tokens: tuple[str, ...]) -> bool:
    for value in values:
        lowered = value.lower()
        if any(token in lowered for token in tokens):
            return True
    return False


def _has_unsupported_package_instructions(
    *,
    context: SupportPipelineContext,
    reply_html: str,
) -> bool:
    policy = _PRODUCTS.get("package_guard")
    if not policy:
        return False
    text = strip_html(reply_html or "").lower()
    if not text:
        return False

    context_text = " ".join(
        [
            context.theme_tag or "",
            context.subject or "",
            context.message or "",
            context.knowledge_context or "",
        ]
    ).lower()
    family_theme = (context.theme_tag or "").strip().lower() in policy["family_tags"]
    family_context = any(
        token in context_text
        for token in policy["family_names"]
    ) or (
        family_theme
        and any(token in context_text for token in policy["base_package_names"])
    )
    install_context = any(
        token in f"{context_text} {text}"
        for token in ("download", "install", "activate", "upload", "unzip", "theme package")
    )
    if not family_context or not install_context or policy["base_name"] not in text:
        return False

    if re.search(policy["invalid_archive"], text):
        return True

    if "two separate folders" in text and re.search(policy["child_slug"], text):
        return True

    child_theme_install = (
        "child theme" in text
        and re.search(r"\b(install|activate|upload)\b", text)
        and re.search(policy["child_name"], text)
    )
    return bool(child_theme_install)


def _has_unsafe_migration_instructions(
    *,
    context: SupportPipelineContext,
    reply_html: str,
) -> bool:
    policy = _PRODUCTS.get("migration_guard")
    if not policy:
        return False
    draft_text = strip_html(reply_html or "").lower()
    if not draft_text:
        return False

    context_text = strip_html(
        "\n".join(
            [
                context.theme_tag or "",
                context.subject or "",
                context.message or "",
                context.knowledge_context or "",
                context.past_tickets_context or "",
                context.conversation_history or "",
            ]
        )
    ).lower()
    current_product = re.search(policy["current_product"], context_text)
    classic_context = any(
        token in context_text
        for token in policy["legacy_terms"]
    )
    if not current_product or not classic_context:
        return False

    corrective_guidance = (
        "not a normal in-place" in draft_text
        or "not an in-place" in draft_text
        or "not automatic or in-place" in draft_text
        or (
            any(token in draft_text for token in ("clean", "fresh", "staging"))
            and any(token in draft_text for token in ("rebuild", "manually recreat", "manual recreation"))
        )
    )
    if corrective_guidance:
        return False

    return bool(
        re.search(policy["switch_back"], draft_text)
        or re.search(policy["activate_on_current"], draft_text)
        or re.search(policy["current_then_activate"], draft_text)
        or "in-place update" in draft_text
        or "automatic migration" in draft_text
    )


def _text_has_repo_evidence(value: str) -> bool:
    text = value or ""
    return ("Repo evidence (" in text and "Source repo:" in text) or (
        SOURCE_EXCERPTS_HEADING in text
        and bool(re.search(r"[\w./-]+:\d+-\d+\n\d+: ", text))
    )


def _text_has_uncertainty(value: str) -> bool:
    normalized = (value or "").lower()
    return any(
        token in normalized
        for token in (
            "uncertain",
            "plausible",
            "not verified",
            "unverified",
            "cannot verify",
            "missing critical",
            "critical evidence",
            "may not exist",
            "needs repo evidence",
            "missing repo evidence",
        )
    )


def _customer_asks_code_level_question(context: SupportPipelineContext) -> bool:
    text = strip_html(f"{context.subject}\n{context.message}").lower()
    return bool(
        re.search(r"\b(does|do|can|is|are|will|would)\b.{0,80}\b(function|query|widget|hook|filter|shortcode|block)\b", text)
        and re.search(r"\b(exclude|support|use|uses|include|return|show|hide|filter)\b", text)
    )


def _has_risky_implementation_claim(
    *,
    context: SupportPipelineContext,
    investigation: SupportInvestigationResult,
) -> bool:
    draft_text = strip_html(investigation.reply_html or "")
    combined = f"{context.subject}\n{context.message}\n{draft_text}"
    lowered = combined.lower()

    if re.search(r"\b(add_filter|add_action|apply_filters|do_action|wp_query|php|code snippet|custom code)\b", lowered):
        return True
    if re.search(r"\b(hook|filter|function|template|query args?|widget query|compatib(?:le|ility))\b", lowered):
        return True
    if extract_repo_search_terms(
        subject=context.subject,
        message=context.message,
        draft_html=investigation.reply_html,
        max_terms=3,
    ):
        return _customer_asks_code_level_question(context) or any(
            token in lowered
            for token in ("hook", "filter", "function", "widget", "query", "compatible")
        )
    return False


def _repo_evidence_supports_claim(
    *,
    context: SupportPipelineContext,
    investigation: SupportInvestigationResult,
) -> bool:
    evidence_text = "\n".join(
        part
        for part in (
            context.knowledge_context,
            investigation.evidence_summary,
            "\n".join(investigation.sources or []),
        )
        if part
    )
    if not _text_has_repo_evidence(evidence_text):
        return False

    terms = extract_repo_search_terms(
        subject="",
        message="",
        draft_html=investigation.reply_html,
        max_terms=8,
    )
    draft_text = unescape(strip_html(investigation.reply_html or ""))
    # Retrieval aliases are leads rather than literal claims. Conversely,
    # a literal identifier remains a claim even if the writer omitted backticks.
    terms.extend(re.findall(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b", draft_text))
    code_terms = [
        term
        for term in terms
        if re.search(r"(?<![\w])" + re.escape(term) + r"(?![\w])",
                     draft_text, re.IGNORECASE)
        and (
            "_" in term
            or "::" in term
            or term.endswith("()")
            or term in _PRODUCTS["literal_feature_claims"]
        )
    ]
    if not code_terms:
        return True

    lowered_evidence = evidence_text.lower()
    normalized_evidence = re.sub(r"[^a-z0-9]+", "", lowered_evidence)
    return any(
        term.lower() in lowered_evidence
        or re.sub(r"[^a-z0-9]+", "", term.lower()) in normalized_evidence
        for term in code_terms
    )


def _repo_lookup_reason(
    *,
    context: SupportPipelineContext,
    investigation: SupportInvestigationResult,
    evaluation: DraftConfidenceResult,
    hard_blockers: list[str],
) -> str:
    combined_evaluator_text = " ".join(
        [
            evaluation.summary or "",
            evaluation.rewrite_brief or "",
            " ".join(evaluation.missing_information or []),
            " ".join(evaluation.risk_flags or []),
            " ".join(evaluation.hard_blockers or []),
            " ".join(hard_blockers or []),
        ]
    )

    if detect_repo_contradiction(
        draft_html=investigation.reply_html,
        repo_context=context.knowledge_context,
    ):
        return "Repo evidence contradicts the current draft; rewrite the draft instead of defending the workaround."

    if _has_unreviewed_screenshots(context=context, investigation=investigation):
        return (
            "The customer attached screenshots that the investigation never analyzed. Open every "
            "attachment and diagnose from what they show before drafting; do not answer around them."
        )

    dead_css = _detect_dead_css(context=context, reply_html=investigation.reply_html)
    if dead_css:
        return (
            f"The draft's CSS cannot work: {dead_css}. Re-derive the selector from the customer's "
            "live markup rather than from a past ticket, an internal note, or another theme."
        )

    if _contains_token(hard_blockers + list(evaluation.missing_information or []), ("missing_critical_evidence", "missing critical", "critical evidence")):
        return wording("evidence_missing")

    if 6 <= evaluation.score <= 8 and _text_has_uncertainty(combined_evaluator_text):
        return wording("score_uncertain")

    if _customer_asks_code_level_question(context) and not _text_has_repo_evidence(context.knowledge_context):
        return wording("source_question")

    if _has_risky_implementation_claim(context=context, investigation=investigation) and not _repo_evidence_supports_claim(
        context=context,
        investigation=investigation,
    ):
        return wording("claim_needs_repo")

    return ""


def _cap_for_missing_repo_evidence(
    *,
    evaluation: DraftConfidenceResult,
    reason: str,
    ceiling: int,
) -> DraftConfidenceResult:
    """Hold a risky implementation claim without repo support below the auto-send bar:
    ``ceiling`` is one under it, plus the missing_repo_evidence blocker."""
    if evaluation.score <= ceiling and "missing_repo_evidence" in (evaluation.hard_blockers or []):
        return evaluation

    summary = (evaluation.summary or "").strip()
    if summary:
        summary = f"{summary} Guard: {reason}"
    else:
        summary = f"Guard: {reason}"

    return replace(
        evaluation,
        score=min(evaluation.score, ceiling),
        verdict="retry" if evaluation.verdict == "accept" else evaluation.verdict,
        summary=summary,
        investigation_actions=_unique(list(evaluation.investigation_actions or []) + [wording("check_repo_action")]),
        rewrite_brief=evaluation.rewrite_brief or wording("rewrite_repo_claim"),
        risk_flags=_unique(list(evaluation.risk_flags or []) + ["missing_repo_evidence"]),
        hard_blockers=_unique(list(evaluation.hard_blockers or []) + ["missing_repo_evidence"]),
    )


def _has_semantic_failure_match(reply_html: str, failures: list[dict[str, Any]]) -> bool:
    """Catch near-verbatim repetition; the independent judge assesses paraphrased claims.

    Shared topic words are retrieval hints, never proof of repeating a mistake. Compare
    individual sentences so unrelated caveats cannot negate an otherwise repeated claim.
    """
    draft_text = strip_html(reply_html or "")
    if not draft_text or not failures:
        return False

    for failure in failures:
        wrong_claim = str((failure or {}).get("wrong_claim") or "").strip()
        for claim in re.split(r"(?<=[.!?;])\s+", wrong_claim):
            claim_words = re.findall(r"\b\w+\b", claim.lower())
            if len(claim_words) < 6:
                continue
            for sentence in re.split(r"(?<=[.!?;])\s+", draft_text):
                if _has_negation(claim) != _has_negation(sentence):
                    continue
                words = re.findall(r"\b\w+\b", sentence.lower())
                if SequenceMatcher(None, claim_words, words, autojunk=False).ratio() >= 0.8:
                    return True
    return False


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _has_negation(text: str) -> bool:
    normalized = strip_html(text or "").lower().replace("’", "'")
    normalized = f" {normalized} "
    negation_markers = (
        " not ",
        " does not ",
        " doesn't ",
        " cannot ",
        " can't ",
        " isn't ",
        " aren't ",
        " wasn't ",
        " weren't ",
        " won't ",
        " don't ",
        " didn't ",
        " no longer ",
        " unsupported ",
        " not supported ",
    )
    return any(marker in normalized for marker in negation_markers)


def _normalize_hard_blocker_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    if not normalized:
        return ""

    if "manual_review" in normalized:
        return "manual_review"

    if (
        "malformed" in normalized
        or "invalid_json" in normalized
        or ("provider" in normalized and "output" in normalized)
    ):
        return "malformed_provider_output"

    if (
        "missing_critical" in normalized
        or ("critical" in normalized and "evidence" in normalized)
        or "cannot_verify" in normalized
        or "unverified" in normalized
    ):
        return "missing_critical_evidence"

    if any(token in normalized for token in ("contradict", "conflict", "inconsistent")):
        return "contradiction_detected"

    if "semantic_failure" in normalized:
        return "semantic_failure_match"

    return normalized


def _is_soft_style_risk(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    if not normalized:
        return False

    soft_tokens = (
        "tone",
        "tutorial",
        "verbose",
        "verbosity",
        "reply_length",
        "too_long",
        "operator_note",
        "hand_holding",
        "hand_holding",
        "over_explain",
        "overexplaining",
        "noise",
        "question_complexity",
        "customer_tone",
        "escalation",
        "conclusive",
        "direct",
    )
    return any(token in normalized for token in soft_tokens)


# Self-review flags a drafter raises only because the reply corrects an earlier reply of ours.
# An evidence-backed correction accepted by the judge follows ordinary send rules.
_CORRECTION_FLAG_TOKENS = (
    "corrects_earlier",
    "correction",
    "retract",
    "withdraw",
    "walk_back",
    "walks_back",
    "previous_diagnosis",
    "earlier_diagnosis",
    "prior_diagnosis",
    "previous_reply",
    "earlier_reply",
    "prior_reply",
    "previous_explanation",
    "earlier_explanation",
)


def _is_correction_flag(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return bool(normalized) and any(token in normalized for token in _CORRECTION_FLAG_TOKENS)


def _release_correction_self_review(
    *,
    investigation: SupportInvestigationResult,
    evaluation: DraftConfidenceResult,
) -> SupportInvestigationResult:
    """Drop the drafter's own manual-review request when it only concerns correcting an
    earlier reply and the judge accepted the draft. Any other self-flag keeps the hold; the
    judge's verdict, score and blockers still decide delivery."""
    flags = list(investigation.risk_flags or [])
    if (
        not investigation.requires_manual_review
        or evaluation.verdict != "accept"
        or not any(_is_correction_flag(flag) for flag in flags)
        or not all(_is_correction_flag(flag) or _is_soft_style_risk(flag) for flag in flags)
    ):
        return investigation
    return replace(investigation, requires_manual_review=False)


def _is_soft_manual_review_case(
    *,
    investigation: SupportInvestigationResult,
    evaluation: DraftConfidenceResult,
    hard_blockers: list[str],
) -> bool:
    if evaluation.verdict != "manual_review":
        return False

    if investigation.requires_manual_review:
        return False

    normalized_blockers = [_normalize_hard_blocker_name(blocker) for blocker in hard_blockers if blocker]
    non_manual_blockers = [blocker for blocker in normalized_blockers if blocker and blocker != "manual_review"]
    if non_manual_blockers:
        return False

    combined_risk_flags = list(investigation.risk_flags or []) + list(evaluation.risk_flags or [])
    if not combined_risk_flags:
        return False

    return all(_is_soft_style_risk(flag) for flag in combined_risk_flags)


def _has_unreviewed_screenshots(
    *,
    context: SupportPipelineContext,
    investigation: SupportInvestigationResult,
) -> bool:
    """Whether real screenshots were absent from the actual investigator input.

    Material images must be inspected before automatic delivery.
    Blank sub-1KB transport artifacts do not count — there is nothing in them to review (the
    right response to those is telling the customer their images never arrived).
    """
    real_images = material_customer_images(context.attachments)
    if not real_images:
        return False

    # Text-only first passes cannot inspect pixels. A partial download cannot prove the
    # omitted images were inspected either. Only the CLI wrapper supplies this receipt;
    # it is absent from the model's output schema, so mentioning screenshots proves nothing.
    submitted = getattr(investigation, "image_input_count", 0)
    return type(submitted) is not int or submitted < len(real_images)


def _contradicts_image_receipt(context: SupportPipelineContext, reply_html: str) -> bool:
    present = any(not a.get("is_blank") and not a.get("is_signature")
                  and (a.get("is_image") or a.get("inline"))
                  for a in context.attachments or [])
    if not present:
        return False
    text = unescape(strip_html(reply_html or ""))
    return bool(re.search(
        r"\b(?:your|the)\s+(?:screenshots?|images?|attachments?)\s+"
        r"(?:(?:did\s+not|didn['’]t|haven['’]t|hasn['’]t)\s+(?:come\s+through|arriv\w*|reach\w*)|"
        r"(?:is|are|was|were)\s+(?:missing|not\s+attached))",
        text, re.IGNORECASE))


def _detect_dead_css(*, context: SupportPipelineContext, reply_html: str) -> str | None:
    """Whether the draft's CSS targets classes the customer's live site does not have.

    Only fires when the draft actually ships class selectors and the conversation names a
    customer domain, so the fetch stays off the path for the vast majority of tickets. Any
    failure returns None: an unreachable site is missing knowledge, not proof of a bad
    selector.
    """
    try:
        return detect_dead_css_selectors_for_context(
            draft_html=reply_html,
            texts=(
                context.message,
                context.conversation_history,
                context.customer_context,
            ),
        )
    except Exception:
        logger.info("dead-CSS guard failed; treating as inconclusive", exc_info=True)
        return None


# Requests or promises involving money require operator review.
_CUSTOMER_MONEY_REQUEST = re.compile(
    # English
    r"\brefund|\bmoney\s+back\b|\bcharge\s?back|\bdisput(?:e|ed|ing)\b|\breimburse|\bdiscount|\bcoupon|\bpromo\s+code|"
    r"\bpro-?rat(?:e|ed|ion)\b|\bpartial\s+credit\b|\b(?:charged|billed)\s+(?:me\s+)?(?:twice|double)\b|"
    r"\bdouble[-\s](?:charged|billed|charge)\b|\bcheaper\s+price\b|\breverse\s+the\s+charge\b|"
    r"\b(?:free|complimentary)\s+(?:theme\s+|product\s+|licen[cs]e\s+|plan\s+|account\s+)?upgrade\b"
    r"(?!\s+(?:(?:to|of)\s+)?(?:wordpress|php|gutenberg|mysql|mariadb|woocommerce)\b)|"
    r"\b(?:get|have|receive|claim|provide|give|offer)\s+(?:me\s+|us\s+)?(?:an?\s+)?"
    r"(?:free|complimentary)\s+(?:theme\s+)?(?:licen[cs]e|theme|renewal)\b|"
    r"\b(?:upgrade|switch|move)\s+(?:my|our|the|your)\s+(?:licen[cs]e|plan|account)\b"
    r"[^.!?\n]{0,60}\bfor\s+free\b|"
    rf"\b(?:get|receive|have|claim)\s+(?:an?\s+|the\s+)?upgrade\s+to\s+(?:{re.escape(_PRODUCTS['brand'])}\s+)?plus\b|"
    # DE / FR / ES / IT / RO / NL / PL
    r"r[üu]ckerstatt|erstatt|geld\s+zur[üu]ck|\brabatt|"
    r"rembours|\bune\s+remise\b|\bremise\s+(?:de\s+\d|sur|commerciale)|\bune\s+r[ée]duction\b|\brabais\b|"
    r"reembols|devoluci[óo]n|devolver\s+(?:el|mi)\s+dinero|\bdescuento|"
    r"rimbors|\bsconto\b|"
    r"ramburs|banii\s+[îi]napoi|returnarea\s+banilor|\breducere\b|"
    r"terugbetal|geld\s+terug|\bkorting\b|"
    r"\bzwrot|\bzni[żz]k|\brabat\b",
    re.IGNORECASE,
)
_DRAFT_MONEY_OFFER = re.compile(
    r"\brefund|\bmoney\s+back\b|\breimburse|\bdiscount|\bcoupon|\bpro-?rat(?:e|ed|ion)\b|\bwaiv(?:e|ed|ing)\b|"
    r"\bfree\s+of\s+charge\b|\bat\s+no\s+(?:extra\s+|additional\s+)?(?:cost|charge)\b|\bno\s+charge\b|\bcomplimentary\b|"
    r"\bfree\s+(?:licen[cs]e|copy|month|months|year|extension|renewal)\b|\bextra\s+(?:month|months|year)\b|"
    r"\bextend\w*\b[^.\n]{0,40}\bfor\s+free\b|\bon\s+(?:us|the\s+house)\b|\b\d{1,3}\s?%\s+off\b|\bcredit(?:ed)?\s+(?:to\s+)?your\s+account\b|"
    r"\breverse\s+the\s+charge\b",
    re.IGNORECASE,
)


def _customer_history_text(conversation_history: str) -> str:
    """The customer's earlier turns from the processor's JSON history ([{role, message}])."""
    try:
        turns = json.loads(conversation_history or "[]")
    except (TypeError, ValueError):
        return ""
    if not isinstance(turns, list):
        return ""
    return "\n".join(
        str(turn.get("message") or "")
        for turn in turns
        if isinstance(turn, dict) and turn.get("role") == "customer"
    )


def _money_decision(*, context: SupportPipelineContext, reply_html: str) -> bool:
    """Return whether the conversation or draft asks for a financial decision requiring operator review."""
    customer_text = unescape(strip_html("\n".join(
        (context.subject or "", context.message or "", _customer_history_text(context.conversation_history))
    )))
    draft_text = unescape(strip_html(reply_html or ""))
    # A purchased lifetime entitlement plus a request to receive a product upgrade is
    # a commercial decision even when "free" is implicit. Do not confuse installing
    # newer WordPress/PHP or updating an already-owned theme with granting a new licence.
    lifetime_upgrade_request = (
        re.search(r"\b(?:lifetime|for\s+life|perpetual)\b", customer_text, re.IGNORECASE)
        and re.search(r"\b(?:bought|purchas\w*|paid\s+for|licen[cs]e)\b", customer_text, re.IGNORECASE)
        and re.search(
            r"\b(?:get|receive|have|claim)\s+(?:an?\s+|the\s+)?"
            r"(?:theme\s+|product\s+|licen[cs]e\s+)?upgrade\b"
            r"(?!\s+(?:(?:to|of)\s+)?(?:wordpress|php|gutenberg|mysql|mariadb|woocommerce)\b)",
            customer_text,
            re.IGNORECASE,
        )
    )
    return bool(
        _CUSTOMER_MONEY_REQUEST.search(customer_text)
        or lifetime_upgrade_request
        or _DRAFT_MONEY_OFFER.search(draft_text)
        or _CUSTOMER_MONEY_REQUEST.search(draft_text)
    )


def _collect_hard_blockers(
    *,
    context: SupportPipelineContext,
    investigation: SupportInvestigationResult,
    evaluation: DraftConfidenceResult,
) -> list[str]:
    blockers: list[str] = []
    if context.duplicate_conversation_ids:
        blockers.append("duplicate_conversation_same_customer")

    # Promotional introductions never inherit ordinary support's send authority,
    # even when both models omit the required manual-review signal.
    if (_mentions_concierge(investigation.reply_html)
            or "concierge_human_review" in (investigation.risk_flags or [])
            or "concierge_human_review" in (evaluation.risk_flags or [])):
        blockers.append("concierge_human_review")

    if evaluation.verdict == "manual_review" or investigation.requires_manual_review:
        blockers.append("manual_review")

    explicit_evaluator_blockers = []
    for blocker in list(evaluation.hard_blockers or []):
        normalized = _normalize_hard_blocker_name(blocker)
        if normalized:
            explicit_evaluator_blockers.append(normalized)
    explicit_evaluator_blockers = _unique(explicit_evaluator_blockers)
    blockers.extend(explicit_evaluator_blockers)

    combined_flags = _unique(
        list(investigation.risk_flags or [])
        + list(evaluation.risk_flags or [])
        + explicit_evaluator_blockers
    )

    for hold in ("duplicate_conversation_same_customer", "known_data_loss_risk"):
        if hold in combined_flags:
            blockers.append(hold)

    if "cancellation_not_verified" in combined_flags:
        reply_text = unescape(strip_html(investigation.reply_html or ""))
        completed_cancellation = re.search(
            r"\b(?:i|we)(?:(?:['’]ve| have)\s+|\s+)(?:already\s+)?cancel(?:l)?ed\b"
            r"|\b(?:renewal|subscription)\s+(?:has|have)\s+been\s+cancel(?:l)?ed\b",
            reply_text,
            flags=re.IGNORECASE,
        )
        if completed_cancellation:
            blockers.append("unverified_account_action_claim")

    if "investigator_failure" in combined_flags or "evaluator_failure" in combined_flags:
        blockers.append("malformed_provider_output")

    contradiction_flags = {
        "contradiction_detected", "contradictory_evidence", "conflicting_evidence",
        "inconsistent_evidence", "evidence_conflict", "repo_evidence_contradiction",
    }
    normalized_flags = {re.sub(r"[^a-z0-9]+", "_", f.lower()).strip("_") for f in combined_flags}
    if normalized_flags & contradiction_flags:
        blockers.append("contradiction_detected")

    repo_contradiction = detect_repo_contradiction(
        draft_html=investigation.reply_html,
        repo_context=context.knowledge_context,
    )
    if repo_contradiction:
        blockers.append("repo_evidence_contradiction")

    if _detect_dead_css(context=context, reply_html=investigation.reply_html):
        blockers.append("dead_css_selector")

    if _has_unreviewed_screenshots(context=context, investigation=investigation):
        blockers.append("unreviewed_screenshots")
    if _contradicts_image_receipt(context, investigation.reply_html):
        blockers.append("image_receipt_contradiction")

    # The judge decides whether a missing fact makes the reply unsafe. A diagnostic
    # question can be sendable precisely because the cause is still unknown. Only an
    # explicit critical-evidence hold is a code veto, not arbitrary "unverified" prose.
    if any(re.sub(r"[^a-z0-9]+", "_", flag.lower()).strip("_") == "missing_critical_evidence"
           for flag in combined_flags):
        blockers.append("missing_critical_evidence")

    if _has_semantic_failure_match(investigation.reply_html, context.semantic_failure_matches):
        blockers.append("semantic_failure_match")

    if _has_unsupported_package_instructions(context=context, reply_html=investigation.reply_html):
        blockers.append(_PRODUCTS["package_guard"]["blocker"])

    if _has_unsafe_migration_instructions(context=context, reply_html=investigation.reply_html):
        blockers.append(_PRODUCTS["migration_guard"]["blocker"])

    if _money_decision(context=context, reply_html=investigation.reply_html):
        blockers.append("money_decision")

    return _unique(blockers)


def _build_retry_feedback(evaluation: DraftConfidenceResult) -> str:
    lines = ["Evaluator feedback:", f"Summary: {evaluation.summary or 'needs stronger evidence'}"]
    if evaluation.missing_information:
        lines.append(f"Missing: {', '.join(evaluation.missing_information[:3])}")
    if evaluation.investigation_actions:
        lines.append(f"Actions: {', '.join(evaluation.investigation_actions[:3])}")
    if evaluation.rewrite_brief:
        lines.append(f"Rewrite: {evaluation.rewrite_brief}")
    return "\n".join(lines).strip()


def _append_note(base: str, addition: str) -> str:
    if not base.strip():
        return addition.strip()
    return f"{base.strip()}\n\n{addition.strip()}".strip()


def _compact_note_text(value: str, *, max_chars: int = 360) -> str:
    text = re.sub(r"\s+", " ", strip_html(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip(" ,.;:") + "..."


def _format_note_list(values: list[str], *, limit: int = 3, max_chars: int = 260) -> str:
    cleaned = [_compact_note_text(value, max_chars=max_chars) for value in _unique(values or [])]
    return "; ".join(cleaned[:limit])


def _merge_knowledge_context(base: str, addition: str) -> str:
    if not addition.strip():
        return base
    if addition.strip() in (base or ""):
        return base
    if not (base or "").strip():
        return addition.strip()
    return f"{base.rstrip()}\n\n---\n\n{addition.strip()}"


def _lookup_repo_evidence_for_retry(
    *,
    context: SupportPipelineContext,
    investigation: SupportInvestigationResult,
) -> str:
    repo_context = get_repo_evidence_context(
        theme_tag=context.theme_tag,
        subject=context.subject,
        message=context.message,
        draft_html=investigation.reply_html,
    )

    # A CSS snippet is an implementation claim, and the keyword-based grep rarely backs it up
    # on its own — which is what caps accurate CSS drafts at manual review. Confirming each
    # class against the product's own stylesheets turns that into checkable evidence.
    return _merge_knowledge_context(repo_context, _verify_draft_css(context=context, investigation=investigation))


def _ground_context_with_draft_evidence(
    *,
    context: SupportPipelineContext,
    investigation: SupportInvestigationResult,
) -> SupportPipelineContext:
    """Attach deterministic verification of the draft's claims before it is evaluated.

    Until now this evidence (CSS classes located in the product's stylesheets, UI labels
    located in its i18n strings) was only gathered on retry — after the evaluator had already
    capped an accurate first draft for lacking repo backing. Grounding first means a correct
    draft arrives at the evaluator with its proof attached, and a draft naming a nonexistent
    control arrives with the warning attached. Cheap (grep over cached checkouts) and
    best-effort: any failure leaves the context unchanged.
    """
    additions = [
        _verify_draft_css(context=context, investigation=investigation),
    ]
    try:
        additions.append(live_css_evidence_for_context(draft_html=investigation.reply_html,
            texts=(context.message, context.conversation_history, context.customer_context)))
    except Exception:
        logger.info("Live CSS evidence collection failed; continuing without it", exc_info=True)
    try:
        product_tag = resolve_repo_target(
            theme_tag=context.theme_tag,
            subject=context.subject,
            message=context.message,
        )
        if product_tag:
            additions.append(
                verify_draft_ui_labels(
                    draft_html=investigation.reply_html,
                    product_tag=product_tag,
                    # Controls the customer named themselves exist on their screen,
                    # whoever renders them — never second-guess those.
                    context_text=f"{context.message}\n{context.conversation_history}",
                )
            )
    except Exception:
        logger.info("UI-label verification failed; continuing without it", exc_info=True)

    merged = context.knowledge_context
    for addition in additions:
        if addition:
            merged = _merge_knowledge_context(merged, addition)
    if merged == context.knowledge_context:
        return context
    return replace(context, knowledge_context=merged)


def _verify_draft_css(
    *,
    context: SupportPipelineContext,
    investigation: SupportInvestigationResult,
) -> str:
    try:
        product_tag = resolve_repo_target(
            theme_tag=context.theme_tag,
            subject=context.subject,
            message=context.message,
        )
        if not product_tag:
            return ""
        return verify_draft_css_against_repo(
            draft_html=investigation.reply_html,
            product_tag=product_tag,
        )
    except Exception:
        logger.info("CSS repo verification failed; continuing without it", exc_info=True)
        return ""


def _result_rank(
    *,
    investigation: SupportInvestigationResult,
    evaluation: DraftConfidenceResult,
    hard_blockers: list[str],
) -> tuple[int, int, int, int]:
    has_reply = 1 if strip_html(investigation.reply_html or "") else 0
    blocker_free = 1 if not hard_blockers else 0
    not_manual_review = 1 if evaluation.verdict != "manual_review" and not investigation.requires_manual_review else 0
    return (blocker_free, not_manual_review, evaluation.score, has_reply)


def _sanitize_investigation_reply(
    *,
    context: SupportPipelineContext,
    investigation: SupportInvestigationResult,
) -> SupportInvestigationResult:
    # Preserve the review hold even if sanitization removes a malformed/link-only
    # invitation. Only the published assessment URL gains explicit link context.
    if _mentions_concierge(investigation.reply_html):
        investigation = replace(investigation, requires_manual_review=True,
            risk_flags=_unique([*investigation.risk_flags, "concierge_human_review"]))
    allowed_urls = extract_context_urls(
        context.knowledge_context,
        context.past_tickets_context,
        extra_urls=investigation.sources,
    )
    concierge_url = _PRODUCTS["concierge_url"]
    if concierge_url and concierge_url in (investigation.reply_html or ""):
        allowed_urls.add(concierge_url)
    sanitized_html = finalize_response_html(
        investigation.reply_html or "",
        context.knowledge_context,
        context.past_tickets_context,
        allowed_urls=allowed_urls,
    ).replace("\x1a", " → ")
    sanitized_html = re.sub(r"[\x00-\x09\x0b-\x1f\x7f]", "", sanitized_html)

    if sanitized_html == investigation.reply_html:
        return investigation

    return replace(investigation, reply_html=sanitized_html)


def _is_evaluator_failure(evaluation: DraftConfidenceResult) -> bool:
    return "evaluator_failure" in (evaluation.risk_flags or [])


def _is_investigator_failure(investigation: SupportInvestigationResult) -> bool:
    return "investigator_failure" in (investigation.risk_flags or [])


def _evaluate_with_provider_fallback(
    *,
    context: SupportPipelineContext,
    investigation: SupportInvestigationResult,
    evaluator_provider: str,
    evaluator_timeout: int,
    original_internal_note: str,
) -> DraftConfidenceResult:
    judge_knowledge = context.knowledge_context
    if context.semantic_failure_matches:
        prior_claims = [
            {"wrong_claim": f.get("wrong_claim", ""), "correct_claim": f.get("correct_claim", "")}
            for f in context.semantic_failure_matches[:20] if f.get("wrong_claim")
        ]
        judge_knowledge = _merge_knowledge_context(
            judge_knowledge,
            "Past mistakes to assess against this draft (shared topic words are not repetition):\n"
            + json.dumps(prior_claims, ensure_ascii=False),
        )
    evaluation = evaluate_draft(
        customer_name=context.customer_name,
        subject=context.subject,
        message=context.message,
        draft_html=investigation.reply_html,
        theme_tag=theme_label(context.theme_tag, context.theme_source),
        knowledge_context=judge_knowledge,
        past_tickets_context=context.past_tickets_context,
        customer_context=context.customer_context,
        conversation_history=context.conversation_history,
        internal_note=original_internal_note,
        investigation_summary=investigation.decision_summary,
        evidence_summary=investigation.evidence_summary,
        investigation_steps=investigation.investigation_steps,
        investigator_risk_flags=investigation.risk_flags,
        investigator_sources=investigation.sources,
        provider=evaluator_provider,
        timeout=evaluator_timeout,
    )

    if evaluator_provider == "claude" and _is_evaluator_failure(evaluation):
        codex_evaluation = evaluate_draft(
            customer_name=context.customer_name,
            subject=context.subject,
            message=context.message,
            draft_html=investigation.reply_html,
            theme_tag=context.theme_tag,
            knowledge_context=judge_knowledge,
            past_tickets_context=context.past_tickets_context,
            customer_context=context.customer_context,
            conversation_history=context.conversation_history,
            internal_note=original_internal_note,
            investigation_summary=investigation.decision_summary,
            evidence_summary=investigation.evidence_summary,
            investigation_steps=investigation.investigation_steps,
            investigator_risk_flags=investigation.risk_flags,
            investigator_sources=investigation.sources,
            provider="codex",
            timeout=evaluator_timeout,
        )
        return codex_evaluation

    return evaluation


@dataclass
class _DeepPassOutcome:
    trace: dict[str, Any]
    investigation: SupportInvestigationResult | None = None
    evaluation: DraftConfidenceResult | None = None
    hard_blockers: list[str] = field(default_factory=list)


def _judge_draft(
    *,
    context: SupportPipelineContext,
    investigation: SupportInvestigationResult,
    evaluator_provider: str,
    evaluator_timeout: int,
    original_internal_note: str,
    target_score: int,
) -> tuple[DraftConfidenceResult, list[str], SupportInvestigationResult]:
    """Score one draft the way the attempt loop does: judge, correction self-review release,
    hard blockers, repo contradiction, and the cap on risky implementation claims without
    repo support. Returns the investigation too, since the release may change it."""
    evaluation = _evaluate_with_provider_fallback(
        context=context,
        investigation=investigation,
        evaluator_provider=evaluator_provider,
        evaluator_timeout=evaluator_timeout,
        original_internal_note=original_internal_note,
    )
    investigation = _release_correction_self_review(investigation=investigation, evaluation=evaluation)
    hard_blockers = _collect_hard_blockers(context=context, investigation=investigation, evaluation=evaluation)
    if detect_repo_contradiction(draft_html=investigation.reply_html, repo_context=context.knowledge_context):
        hard_blockers = _unique(hard_blockers + ["contradiction_detected"])
    if (
        evaluation.score >= target_score
        and _has_risky_implementation_claim(context=context, investigation=investigation)
        and not _repo_evidence_supports_claim(context=context, investigation=investigation)
    ):
        evaluation = _cap_for_missing_repo_evidence(
            evaluation=evaluation,
            reason="Risky implementation claim is not backed by repo evidence.",
            ceiling=target_score - 1,
        )
        hard_blockers = _collect_hard_blockers(context=context, investigation=investigation, evaluation=evaluation)
    return evaluation, hard_blockers, investigation


def _run_deep_pass(
    *,
    context: SupportPipelineContext,
    investigation: SupportInvestigationResult,
    evaluation: DraftConfidenceResult,
    evaluator_provider: str,
    evaluator_timeout: int,
    original_internal_note: str,
    deep_evidence: DeepEvidenceCollector | None,
    config: DeepPassConfig,
    target_score: int,
) -> _DeepPassOutcome:
    """One deep investigation of the best first-pass draft. Never raises: a failure is
    recorded in the trace and the first draft stands."""
    started = time.monotonic()
    trace: dict[str, Any] = {
        "attempt": "deep",
        "investigator_provider": "claude",
        "investigator_model": "opus-deep",
        "effort": config.effort,
        "from_score": evaluation.score,
        "kept": False,
    }
    workdir = new_workdir()
    try:
        evidence = DeepEvidence()
        if deep_evidence is not None:
            try:
                evidence = deep_evidence(evaluation, workdir)
            except Exception as exc:
                logger.warning("Deep pass evidence collection failed: %s", exc)
                evidence = DeepEvidence(notes=[f"Evidence collection failed: {exc}"])
        deep_context = replace(
            context,
            knowledge_context=_merge_knowledge_context(context.knowledge_context, evidence.knowledge_context),
            past_tickets_context=_merge_knowledge_context(context.past_tickets_context, evidence.past_tickets_context),
            customer_context=_merge_knowledge_context(context.customer_context, evidence.customer_context),
        )
        repo_dirs = repo_dirs_for(
            theme_tag=context.theme_tag,
            subject=context.subject,
            message=context.message,
            # The draft and the judge name the plugin they could not verify ("PixTypes").
            extra_text="\n".join([
                strip_html(investigation.reply_html or ""),
                evaluation.summary or "",
                *(evaluation.missing_information or []),
            ]),
        )
        trace["evidence_notes"] = list(evidence.notes)
        trace["repos"] = [path.name for path in repo_dirs]
        trace["images"] = len(evidence.image_paths)

        request = SupportInvestigationRequest(
            customer_name=deep_context.customer_name,
            subject=deep_context.subject,
            message=deep_context.message,
            theme_tag=deep_context.theme_tag,
            theme_source=deep_context.theme_source,
            knowledge_context=deep_context.knowledge_context,
            past_tickets_context=deep_context.past_tickets_context,
            customer_context=deep_context.customer_context,
            internal_note=original_internal_note,
            conversation_history=deep_context.conversation_history,
        )
        critique = DeepPassCritique(
            previous_draft_html=investigation.reply_html,
            score=evaluation.score,
            summary=evaluation.summary,
            missing_information=list(evaluation.missing_information or []),
            risk_flags=list(evaluation.risk_flags or []),
            rewrite_brief=evaluation.rewrite_brief,
        )
        deep_investigation = deep_draft_reply(
            request,
            critique,
            workdir=workdir,
            repo_dirs=repo_dirs,
            image_paths=evidence.image_paths,
            config=config,
        )
        trace["image_input_count"] = getattr(deep_investigation, "image_input_count", 0)
        if _is_investigator_failure(deep_investigation):
            trace["error"] = _compact_note_text(deep_investigation.decision_summary, max_chars=200)
            return _DeepPassOutcome(trace=trace)
        deep_investigation = _sanitize_investigation_reply(context=deep_context, investigation=deep_investigation)
        excerpts = cited_source_excerpts(
            "\n".join([
                deep_investigation.evidence_summary or "",
                *(deep_investigation.sources or []),
                *(deep_investigation.investigation_steps or []),
            ]),
            repo_dirs,
        )
        trace["cited_excerpts"] = excerpts.count("\n\n") if excerpts else 0
        # GitHub links the deep pass read with WebFetch (a fix branch, WordPress core): Supa
        # re-reads the cited lines at a resolved commit, so they back a claim like a checkout.
        pinned, pinned_notes = pinned_github_excerpts(
            "\n".join([
                deep_investigation.evidence_summary or "",
                *(deep_investigation.sources or []),
                *(deep_investigation.investigation_steps or []),
            ])
        )
        trace["pinned_github"] = pinned.count("\nLines re-read by Supa at that commit") if pinned else 0
        if pinned_notes:
            trace["pinned_github_notes"] = pinned_notes[:5]
        excerpts = "\n\n".join(part for part in (excerpts, pinned) if part)
        if excerpts:
            deep_context = replace(
                deep_context,
                knowledge_context=_merge_knowledge_context(deep_context.knowledge_context, excerpts),
            )
        deep_context = _ground_context_with_draft_evidence(context=deep_context, investigation=deep_investigation)
        deep_evaluation, hard_blockers, deep_investigation = _judge_draft(
            context=deep_context,
            investigation=deep_investigation,
            evaluator_provider=evaluator_provider,
            evaluator_timeout=evaluator_timeout,
            original_internal_note=original_internal_note,
            target_score=target_score,
        )
        # The judge can name an exact source range omitted by the initial excerpt budget.
        # Read those lines and rejudge the same draft once, only with genuinely new evidence.
        if {"missing_repo_evidence", "missing_critical_evidence"} & set(hard_blockers):
            requested_sources = "\n".join([
                *(deep_evaluation.missing_information or []), deep_evaluation.rewrite_brief or "",
                *matching_source_citations("\n".join(deep_evaluation.missing_information or []),
                                           deep_investigation.sources or []),
            ])
            requested = cited_source_excerpts(requested_sources, repo_dirs,
                                             existing_context=deep_context.knowledge_context)
            if requested and requested not in deep_context.knowledge_context:
                repaired_context = replace(deep_context, knowledge_context=_merge_knowledge_context(
                    deep_context.knowledge_context, requested))
                rechecked, rechecked_blockers, rechecked_investigation = _judge_draft(
                    context=repaired_context, investigation=deep_investigation,
                    evaluator_provider=evaluator_provider, evaluator_timeout=evaluator_timeout,
                    original_internal_note=original_internal_note, target_score=target_score,
                )
                kept = _result_rank(investigation=rechecked_investigation, evaluation=rechecked,
                    hard_blockers=rechecked_blockers) > _result_rank(investigation=deep_investigation,
                    evaluation=deep_evaluation, hard_blockers=hard_blockers)
                trace["evidence_recheck"] = {"from_score": deep_evaluation.score,
                    "score": rechecked.score, "hard_blockers": rechecked_blockers, "kept": kept,
                    "judge_summary": _compact_note_text(rechecked.summary, max_chars=600),
                    "still_missing": list(rechecked.missing_information or [])[:5],
                    "requested_sources": requested_sources[:2000]}
                # Source facts remain useful to the factual rewrite even when the same
                # draft's score is unchanged. Keep the candidate decision separately.
                deep_context = repaired_context
                if kept:
                    deep_evaluation, hard_blockers, deep_investigation = rechecked, rechecked_blockers, rechecked_investigation
        # Near-bar evidence/CSS holds get one factual correction, then every gate again.
        # The writer removes unsupported extras rather than pretending to verify them.
        # Neither a commercial hold nor an explicit manual-review decision can enter here.
        if (hard_blockers and set(hard_blockers) <= {"dead_css_selector", "missing_repo_evidence"}
                and deep_evaluation.verdict in {"accept", "retry"}
                and deep_evaluation.score >= target_score - 1 and not deep_investigation.requires_manual_review):
            css_reason = _detect_dead_css(context=deep_context, reply_html=deep_investigation.reply_html)
            correction = (
                f"Final evidence correction. Remaining holds: {', '.join(hard_blockers)}.\n"
                f"Live CSS finding: {css_reason or 'none'}.\n"
                "Missing evidence identified by the independent judge:\n"
                + "\n".join(deep_evaluation.missing_information or [])
                + "\nMake the smallest factual correction. Remove unsupported extra claims, exact numeric ranges "
                "and optional CSS unless the supplied source/live markup verifies them. Do not repeat an earlier "
                "CSS rule merely because it was in the conversation. Keep the verified core and answer each question "
                "as far as the evidence permits; ask briefly for facts only the customer can supply. Do not add new "
                "technical claims or describe missing evidence as verified. Return the normal JSON contract.\nPreviously judged reply:\n"
                + deep_investigation.reply_html + "\nEvidence summary:\n" + deep_investigation.evidence_summary
            )
            rewrite_request = replace(request, knowledge_context=deep_context.knowledge_context,
                                      internal_note=_merge_knowledge_context(original_internal_note, correction))
            rewritten = draft_reply(rewrite_request, timeout=min(config.timeout, 180))
            rewrite_label = "css_rewrite" if set(hard_blockers) == {"dead_css_selector"} else "evidence_rewrite"
            trace[rewrite_label] = {"from_score": deep_evaluation.score, "kept": False,
                "investigator_provider": rewritten.provider_name, "investigator_model": getattr(rewritten, "model", "")}
            if not _is_investigator_failure(rewritten):
                rewritten = _sanitize_investigation_reply(context=deep_context, investigation=rewritten)
                rewrite_context = _ground_context_with_draft_evidence(context=deep_context, investigation=rewritten)
                checked, checked_blockers, rewritten = _judge_draft(
                    context=rewrite_context, investigation=rewritten, evaluator_provider=evaluator_provider,
                    evaluator_timeout=evaluator_timeout, original_internal_note=original_internal_note, target_score=target_score)
                kept = _result_rank(investigation=rewritten, evaluation=checked, hard_blockers=checked_blockers) > _result_rank(
                    investigation=deep_investigation, evaluation=deep_evaluation, hard_blockers=hard_blockers)
                trace[rewrite_label].update(score=checked.score, hard_blockers=checked_blockers, kept=kept,
                    judge_summary=_compact_note_text(checked.summary, max_chars=600),
                    still_missing=list(checked.missing_information or [])[:5])
                if kept:
                    deep_investigation, deep_evaluation, hard_blockers = rewritten, checked, checked_blockers
            else:
                trace[rewrite_label]["error"] = _compact_note_text(rewritten.decision_summary, max_chars=200)
        trace.update(
            evaluator_provider=deep_evaluation.provider,
            score=deep_evaluation.score,
            verdict=deep_evaluation.verdict,
            hard_blockers=list(hard_blockers),
            # What the deep pass did and what the judge still wanted, kept even when the first
            # draft wins, so a stuck ticket shows where the evidence ran out.
            steps=list(deep_investigation.investigation_steps or [])[:12],
            judge_summary=_compact_note_text(deep_evaluation.summary, max_chars=600),
            still_missing=list(deep_evaluation.missing_information or [])[:5],
            reply_html=deep_investigation.reply_html,
        )
        return _DeepPassOutcome(
            trace=trace,
            investigation=deep_investigation,
            evaluation=deep_evaluation,
            hard_blockers=hard_blockers,
        )
    except Exception as exc:  # the deep pass is an improvement, never a new way to fail a ticket
        logger.exception("Deep pass failed")
        trace["error"] = _compact_note_text(str(exc), max_chars=200)
        return _DeepPassOutcome(trace=trace)
    finally:
        trace["duration_s"] = round(time.monotonic() - started)
        remove_workdir(workdir)


def _deep_pass_note_line(trace: dict[str, Any]) -> str:
    minutes = max(1, round((trace.get("duration_s") or 0) / 60))
    head = f"Deep pass ({trace.get('investigator_model', 'opus-deep')}, effort {trace.get('effort')}, ~{minutes} min)"
    if trace.get("error"):
        return f"{head}: failed ({trace['error']}); first draft kept."
    outcome = "deep draft kept" if trace.get("kept") else "first draft kept"
    checked = ", ".join(trace.get("repos") or []) or "no repo checkout"
    line = f"{head}: {trace.get('from_score')}/10 -> {trace.get('score')}/10, {outcome}. Repos: {checked}."
    if trace.get("pinned_github"):
        line += f" GitHub citations re-read by Supa: {trace['pinned_github']}."
    if not trace.get("kept") and trace.get("still_missing"):
        line += f" Still missing after it: {_format_note_list(trace['still_missing'], limit=2)}"
    return line


def build_decision_note(
    *,
    delivery_mode: str,
    confidence_score: int,
    evaluator_summary: str,
    evidence_summary: str = "",
    missing_information: list[str] | None = None,
    investigation_actions: list[str] | None = None,
    rewrite_brief: str = "",
    trace: list[dict[str, Any]] | None = None,
    hard_blockers: list[str],
    investigator_provider: str,
    evaluator_provider: str,
    sources: list[str],
    target_score: int,
    deep_pass_line: str = "",
) -> str | None:
    meaningful = (
        delivery_mode == "auto_send"
        or confidence_score < target_score
        or bool(hard_blockers)
    )
    if not meaningful:
        return None

    reason = _compact_note_text(evaluator_summary, max_chars=520)
    if not reason and hard_blockers:
        reason = f"Blocked by {', '.join(hard_blockers[:3])}."
    if not reason:
        reason = "Decision based on evaluator score."

    evidence_parts = []
    if evidence_summary:
        evidence_parts.append(_compact_note_text(evidence_summary, max_chars=360))
    source_summary = _format_note_list(sources or [], limit=3, max_chars=180)
    if source_summary:
        evidence_parts.append(f"Sources: {source_summary}")
    evidence = " | ".join(evidence_parts) or "none"

    confidence_line = f"Confidence: {confidence_score}/10"
    if delivery_mode == "draft":
        confidence_line += f" (auto-send needs {target_score}/10 and no blockers)"

    lines = [
        "Supa decision note:",
        f"Mode: {delivery_mode}",
        confidence_line,
        f"Why: {reason}",
    ]
    if deep_pass_line:
        lines.append(deep_pass_line)

    if hard_blockers:
        lines.append(f"Blocked by: {', '.join(hard_blockers[:3])}")

    missing = _format_note_list(missing_information or [], limit=3)
    if missing:
        lines.append(f"Missing: {missing}")

    next_steps = _format_note_list(
        list(investigation_actions or []) + ([rewrite_brief] if rewrite_brief else []),
        limit=3,
    )
    if next_steps:
        lines.append(f"Next: {next_steps}")

    if trace and len(trace) > 1:
        scores = [
            str(item.get("score"))
            for item in trace
            if isinstance(item, dict) and item.get("score") is not None
        ]
        if scores:
            lines.append(f"Attempts: {len(trace)} evaluator runs; scores {', '.join(scores)}")

    lines.extend(
        [
            f"Evidence: {evidence}",
            f"Provider: investigator={investigator_provider}, evaluator={evaluator_provider}",
        ]
    )
    return "\n".join(lines)


DECISION_TITLES = {
    "auto_send": "Sent automatically",
    "draft": "Draft left for review",
    "manual_review": "Held for manual review",
}


def decision_record(result: "SupportPipelineResult", note: str) -> dict[str, Any]:
    """The decision as data for the owned mailbox's agent ledger.

    The note text stays the fallback wording; this carries the same facts unabridged, so nobody
    has to read them back out of prose. The mode comes from the note's final Mode line, because a
    late hold (draft-only mode, a customer who is not waiting) rewrites that line last.
    """
    mode_line = next((line[6:].strip() for line in str(note or "").split("\n") if line.startswith("Mode: ")), "")
    mode, _, mode_note = (mode_line or str(result.delivery_mode)).partition(" ")
    target = re.search(r"(?m)^Confidence: [^\n]{0,40}needs (\d{1,2})/10", str(note or ""))
    scheduled = re.search(r"(?m)^Scheduled send: (.+)$", str(note or ""))
    evaluator = result.evaluator_result
    investigator = result.investigator_result
    blockers = [str(item) for item in result.hard_blockers or []]
    title = DECISION_TITLES.get(mode, mode.replace("_", " ").capitalize())
    if mode == "auto_send" and scheduled:
        title = "Send booked"  # A delayed auto-send is not sent yet.
    title += f" · {result.confidence_score}/10" + (" · blocked" if blockers else "")
    data = {
        "mode": mode,
        "mode_note": mode_note.strip("() "),
        "confidence": result.confidence_score,
        "target": int(target[1]) if target else None,
        "verdict": result.confidence_verdict,
        "manual_review": bool(result.manual_review),
        "why": getattr(evaluator, "summary", "") or result.decision_summary,
        "blockers": blockers,
        "risk_flags": [str(item) for item in result.risk_flags or []],
        "missing": "; ".join(getattr(evaluator, "missing_information", None) or []),
        "next": "; ".join(
            list(getattr(evaluator, "investigation_actions", None) or [])
            + ([evaluator.rewrite_brief] if getattr(evaluator, "rewrite_brief", "") else [])
        ),
        "scores": [item.get("score") for item in result.trace or []
                   if isinstance(item, dict) and item.get("score") is not None],
        "evidence": result.evidence_summary,
        "sources": [str(item) for item in getattr(investigator, "sources", None) or []][:12],
        "providers": {"investigator": result.investigator_provider, "evaluator": result.evaluator_provider},
        "scheduled": scheduled[1] if scheduled else None,
        "text": str(note or ""),
    }
    return {
        "kind": "decision",
        "actor": "supa",
        "title": title[:240],
        "data": {key: value for key, value in data.items() if value not in (None, "", [], {})},
    }


def run_support_pipeline(
    *,
    context: SupportPipelineContext,
    evaluator_provider: str = "claude",
    target_score: int | None = None,
    max_attempts: int = 2,
    investigator_timeout: int | None = None,
    evaluator_timeout: int = 180,
    deep_evidence: DeepEvidenceCollector | None = None,
    deep_pass_config: DeepPassConfig | None = None,
) -> SupportPipelineResult:
    """Draft with the support agent (agents/support_agent.py: Opus, then Sonnet, then Codex on
    the same prompt and rules), score with the separate send-confidence judge, apply the hard
    blockers and retry once. When the best draft is still below ``target_score`` the deep pass
    runs once (``_run_deep_pass``): more evidence from ``deep_evidence``, the same agent with
    read-only tools at a higher effort, the same judge; the better-ranked draft wins and the
    usual delivery rules apply to it. ``investigator_timeout`` defaults to the agent's
    configured per-model timeout."""
    evaluator_name = (evaluator_provider or "claude").strip().lower() or "claude"
    if target_score is None:
        target_score = autosend_threshold()

    original_internal_note = context.internal_note or ""
    investigator_internal_note = original_internal_note
    active_context = context

    # Screenshots are primary evidence — the investigator must know they exist. The blocker
    # layer then insists real images were actually engaged with before a diagnosis stands.
    attachment_summary = describe_attachments(context.attachments or [])
    if attachment_summary:
        active_context = replace(
            active_context,
            knowledge_context=_merge_knowledge_context(active_context.knowledge_context, attachment_summary),
        )
    trace: list[dict[str, Any]] = []
    final_investigation: SupportInvestigationResult | None = None
    final_evaluation: DraftConfidenceResult | None = None
    final_hard_blockers: list[str] = []
    best_investigation: SupportInvestigationResult | None = None
    best_evaluation: DraftConfidenceResult | None = None
    best_hard_blockers: list[str] = []

    for attempt in range(1, max_attempts + 1):
        request = SupportInvestigationRequest(
            customer_name=active_context.customer_name,
            subject=active_context.subject,
            message=active_context.message,
            theme_tag=active_context.theme_tag,
            theme_source=active_context.theme_source,
            knowledge_context=active_context.knowledge_context,
            past_tickets_context=active_context.past_tickets_context,
            customer_context=active_context.customer_context,
            internal_note=investigator_internal_note,
            conversation_history=active_context.conversation_history,
        )
        investigation = draft_reply(request, timeout=investigator_timeout)
        investigation = _sanitize_investigation_reply(
            context=active_context,
            investigation=investigation,
        )
        active_context = _ground_context_with_draft_evidence(
            context=active_context,
            investigation=investigation,
        )
        evaluation = _evaluate_with_provider_fallback(
            context=active_context,
            investigation=investigation,
            evaluator_provider=evaluator_name,
            evaluator_timeout=evaluator_timeout,
            original_internal_note=original_internal_note,
        )
        investigation = _release_correction_self_review(investigation=investigation, evaluation=evaluation)
        hard_blockers = _collect_hard_blockers(
            context=active_context,
            investigation=investigation,
            evaluation=evaluation,
        )
        repo_contradiction = detect_repo_contradiction(
            draft_html=investigation.reply_html,
            repo_context=active_context.knowledge_context,
        )
        if repo_contradiction:
            hard_blockers = _unique(hard_blockers + ["contradiction_detected"])

        repo_retry_reason = _repo_lookup_reason(
            context=active_context,
            investigation=investigation,
            evaluation=evaluation,
            hard_blockers=hard_blockers,
        )
        repo_context_for_retry = ""
        if (
            repo_retry_reason
            and attempt < max_attempts
            and not repo_retry_reason.startswith("Repo evidence contradicts")
        ):
            repo_context_for_retry = _lookup_repo_evidence_for_retry(
                context=active_context,
                investigation=investigation,
            )

        if (
            evaluation.score >= target_score
            and _has_risky_implementation_claim(context=active_context, investigation=investigation)
            and not _repo_evidence_supports_claim(context=active_context, investigation=investigation)
        ):
            evaluation = _cap_for_missing_repo_evidence(
                evaluation=evaluation,
                reason=repo_retry_reason or "Risky implementation claim is not backed by repo evidence.",
                ceiling=target_score - 1,
            )
            hard_blockers = _collect_hard_blockers(
                context=active_context,
                investigation=investigation,
                evaluation=evaluation,
            )
            repo_retry_reason = repo_retry_reason or wording("claim_needs_repo")

        trace.append(
            {
                "attempt": attempt,
                "investigator_provider": investigation.provider_name,
                "investigator_model": getattr(investigation, "model", ""),
                "evaluator_provider": evaluation.provider,
                "score": evaluation.score,
                "verdict": evaluation.verdict,
                "hard_blockers": list(hard_blockers),
                "repo_retry_reason": repo_retry_reason,
            }
        )

        final_investigation = investigation
        final_evaluation = evaluation
        final_hard_blockers = hard_blockers
        if best_investigation is None or _result_rank(
            investigation=investigation,
            evaluation=evaluation,
            hard_blockers=hard_blockers,
        ) > _result_rank(
            investigation=best_investigation,
            evaluation=best_evaluation,
            hard_blockers=best_hard_blockers,
        ):
            best_investigation = investigation
            best_evaluation = evaluation
            best_hard_blockers = list(hard_blockers)

        below_target = evaluation.score < target_score
        manual_review = evaluation.verdict == "manual_review" or investigation.requires_manual_review
        soft_manual_review = _is_soft_manual_review_case(
            investigation=investigation,
            evaluation=evaluation,
            hard_blockers=hard_blockers,
        )
        should_retry = (
            attempt < max_attempts
            and (below_target or bool(repo_context_for_retry) or bool(repo_contradiction))
            and (not manual_review or soft_manual_review)
            and (not hard_blockers or soft_manual_review or bool(repo_context_for_retry) or bool(repo_contradiction))
        )
        if should_retry:
            if repo_context_for_retry:
                active_context = replace(
                    active_context,
                    knowledge_context=_merge_knowledge_context(
                        active_context.knowledge_context,
                        repo_context_for_retry,
                    ),
                )
            investigator_internal_note = _append_note(
                investigator_internal_note,
                _build_retry_feedback(evaluation),
            )
            if repo_contradiction:
                investigator_internal_note = _append_note(
                    investigator_internal_note,
                    f"Repo evidence contradicts the current draft: {repo_contradiction}. Rewrite the draft around the verified repo behavior.",
                )
            elif repo_retry_reason:
                investigator_internal_note = _append_note(
                    investigator_internal_note,
                    repo_retry_reason,
                )
            continue

        break

    assert final_investigation is not None
    assert final_evaluation is not None
    assert best_investigation is not None
    assert best_evaluation is not None

    deep_pass_line = ""
    deep_cfg = deep_pass_config or load_deep_pass_config()
    deep_skip = deep_pass_skip_reason(
        investigation=best_investigation,
        evaluation=best_evaluation,
        hard_blockers=best_hard_blockers,
        target_score=target_score,
    )
    if deep_cfg.enabled and deep_skip is None:
        deep = _run_deep_pass(
            context=active_context,
            investigation=best_investigation,
            evaluation=best_evaluation,
            evaluator_provider=evaluator_name,
            evaluator_timeout=evaluator_timeout,
            original_internal_note=original_internal_note,
            deep_evidence=deep_evidence,
            config=deep_cfg,
            target_score=target_score,
        )
        trace.append(deep.trace)
        if deep.investigation is not None and deep.evaluation is not None and _result_rank(
            investigation=deep.investigation,
            evaluation=deep.evaluation,
            hard_blockers=deep.hard_blockers,
        ) > _result_rank(
            investigation=best_investigation,
            evaluation=best_evaluation,
            hard_blockers=best_hard_blockers,
        ):
            best_investigation = deep.investigation
            best_evaluation = deep.evaluation
            best_hard_blockers = list(deep.hard_blockers)
            deep.trace["kept"] = True
        deep_pass_line = _deep_pass_note_line(deep.trace)

    final_investigation = best_investigation
    final_evaluation = best_evaluation
    final_hard_blockers = best_hard_blockers
    if "missing_repo_evidence" in final_hard_blockers and final_evaluation.score >= target_score:
        final_evaluation = replace(
            final_evaluation,
            score=target_score - 1,
            verdict="retry" if final_evaluation.verdict == "accept" else final_evaluation.verdict,
            summary=_append_note(
                final_evaluation.summary,
                f"Risky implementation claim needs repo evidence before it can reach the {target_score}/10 auto-send bar.",
            ),
        )

    delivery_mode = "draft"
    if (
        final_evaluation.score >= target_score
        # Only a draft the judge accepted: an 8 with "retry" is one it wanted improved (2026-11-03).
        and final_evaluation.verdict == "accept"
        and not final_investigation.requires_manual_review
        and not final_hard_blockers
    ):
        delivery_mode = "auto_send"

    decision_note = build_decision_note(
        delivery_mode=delivery_mode,
        confidence_score=final_evaluation.score,
        evaluator_summary=final_evaluation.summary,
        evidence_summary=final_investigation.evidence_summary,
        missing_information=final_evaluation.missing_information,
        investigation_actions=final_evaluation.investigation_actions,
        rewrite_brief=final_evaluation.rewrite_brief,
        trace=trace,
        hard_blockers=final_hard_blockers,
        investigator_provider=final_investigation.provider_name,
        evaluator_provider=final_evaluation.provider,
        sources=final_investigation.sources,
        target_score=target_score,
        deep_pass_line=deep_pass_line,
    )

    return SupportPipelineResult(
        delivery_mode=delivery_mode,
        confidence_score=final_evaluation.score,
        confidence_verdict=final_evaluation.verdict,
        reply_html=final_investigation.reply_html,
        decision_summary=final_investigation.decision_summary,
        evidence_summary=final_investigation.evidence_summary,
        investigation_steps=list(final_investigation.investigation_steps),
        risk_flags=_unique(list(final_investigation.risk_flags) + list(final_evaluation.risk_flags)),
        hard_blockers=final_hard_blockers,
        decision_note=decision_note,
        manual_review=(
            final_evaluation.verdict == "manual_review"
            or final_investigation.requires_manual_review
            or "concierge_human_review" in final_hard_blockers
        ),
        attempts=len(trace),
        investigator_provider=final_investigation.provider_name,
        evaluator_provider=final_evaluation.provider,
        investigator_result=final_investigation,
        evaluator_result=final_evaluation,
        trace=trace,
    )
