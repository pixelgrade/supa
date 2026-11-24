"""Infer the likely product from licences and earlier ticket tags. Prefer a purchase record, then the most recent relevant tag. The drafter confirms the result with the customer."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from services.themes import THEMES, VALID_SLUGS, _find_theme_in_text, detect_theme

logger = logging.getLogger(__name__)

PAST_TICKET_LIMIT = 10


@dataclass(frozen=True)
class ThemeHint:
    slug: str
    source: str   # "licence" or "past tickets"
    detail: str   # what the drafter is told

    def context_block(self) -> str:
        return (
            f"Theme not named in this ticket. Likely theme: {self.slug} ({self.detail}). "
            "This is a hypothesis from the customer's history, not something they said: confirm it "
            f"in one short clause (e.g. \"I'm assuming you're still on {self.slug.replace('-', ' ').title()}\") "
            "and never state it as fact."
        )


def _licence_themes(customer_data: Optional[dict[str, Any]]) -> list[tuple[str, str]]:
    """(theme slug, product title) per licence, newest order first."""
    licences = (customer_data or {}).get("licenses") or {}
    if not isinstance(licences, dict):
        return []
    found: list[tuple[str, str]] = []

    def order_key(item: tuple[str, Any]) -> int:
        return int(item[0]) if str(item[0]).isdigit() else 0

    for _order_id, order in sorted(licences.items(), key=order_key, reverse=True):
        for licence in (order or {}).get("licenses") or []:
            title = (licence or {}).get("product_title") or ""
            slug = _find_theme_in_text(title)
            if slug and slug in VALID_SLUGS:
                found.append((slug, title))
    return found


def _ticket_tag_themes(conversations: list[dict[str, Any]], exclude_id: Any) -> list[tuple[str, str, int]]:
    """(theme slug, created date, ticket number) per earlier ticket with a theme tag, newest first."""
    rows = sorted(conversations or [], key=lambda c: c.get("createdAt") or "", reverse=True)
    found: list[tuple[str, str, int]] = []
    for conversation in rows:
        if str(conversation.get("id")) == str(exclude_id):
            continue
        tags = [t.get("tag", "") if isinstance(t, dict) else str(t) for t in conversation.get("tags") or []]
        # Tags only: the subject of an old ticket is not this customer's theme.
        slug = detect_theme(tags=[t for t in tags if t.lower().strip() in THEMES or t.lower().strip() in VALID_SLUGS])
        if slug:
            found.append((slug, (conversation.get("createdAt") or "")[:10], conversation.get("number") or 0))
    return found


async def infer_theme_from_history(
    *,
    helpscout: Any,
    customer_email: str,
    exclude_id: Any,
    customer_data: Optional[dict[str, Any]],
) -> Optional[ThemeHint]:
    licences = _licence_themes(customer_data)
    tickets: list[tuple[str, str, int]] = []
    if customer_email:
        try:
            conversations = await helpscout.search_conversations(
                f'email:"{customer_email}"', status="all", limit=PAST_TICKET_LIMIT
            )
            tickets = _ticket_tag_themes(conversations, exclude_id)
        except Exception:
            logger.info("Theme history: past-ticket search failed", exc_info=True)

    if licences:
        slug, title = licences[0]
        others = sorted({s for s, _ in licences[1:] if s != slug})
        detail = f"their newest licence is {title}"
        if others:
            detail += f"; they also hold licences for {', '.join(others)}"
        if tickets:
            detail += f"; earlier tickets were tagged {tickets[0][0]} ({tickets[0][1]})"
        return ThemeHint(slug=slug, source="licence", detail=detail)
    if tickets:
        slug, created, number = tickets[0]
        detail = f"their earlier ticket #{number} ({created}) was tagged {slug}; the licence lookup named no theme"
        return ThemeHint(slug=slug, source="past tickets", detail=detail)
    return None


async def theme_hint_for_ticket(*, ticket: dict[str, Any], helpscout: Any, customer_data_task: Any) -> Optional[ThemeHint]:
    """The hint for a parsed ticket that names no theme, or None. Never raises: a failed
    lookup leaves the ticket exactly as it was. Shared by live processing and the read-only
    evaluation, so an evaluation shows what production would do."""
    if ticket.get("theme_tag"):
        return None
    try:
        return await infer_theme_from_history(
            helpscout=helpscout,
            customer_email=ticket.get("customer_email", ""),
            exclude_id=ticket.get("id"),
            customer_data=await customer_data_task,
        )
    except Exception:
        logger.info("Theme history lookup failed", exc_info=True)
        return None
