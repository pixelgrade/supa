"""Search past HelpScout tickets for similar issues and responses."""

import logging
import re
from typing import List, Dict, Any, Optional

import httpx


logger = logging.getLogger(__name__)

# Common words to ignore when extracting keywords
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "although", "though", "after", "before",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "been", "being", "having", "doing",
    "hi", "hello", "hey", "thanks", "thank", "please", "help", "issue",
    "problem", "question", "want", "like", "using", "use", "trying", "try",
    "get", "got", "getting", "make", "made", "making", "theme", "website",
    "site", "page", "wordpress", "wp", "plugin", "work", "working", "doesn",
    "doesn't", "don't", "can't", "won't", "isn't", "aren't", "wasn't",
}

# Theme-related keywords to boost in search
THEME_KEYWORDS = {
    "menu", "header", "footer", "sidebar", "widget", "logo", "font", "color",
    "background", "image", "slider", "gallery", "portfolio", "blog", "post",
    "page", "home", "homepage", "contact", "form", "social", "icon", "button",
    "link", "navigation", "nav", "mobile", "responsive", "css", "style",
    "customize", "customizer", "woocommerce", "shop", "cart", "checkout",
    "product", "category", "tag", "archive", "search", "404", "error",
    "loading", "slow", "speed", "seo", "meta", "title", "description",
    "featured", "thumbnail", "excerpt", "content", "text", "paragraph",
    "heading", "h1", "h2", "h3", "spacing", "margin", "padding", "width",
    "height", "size", "layout", "grid", "column", "row", "section", "block",
    "gutenberg", "editor", "classic", "elementor", "builder", "template",
    "child", "update", "install", "activate", "license", "demo", "import",
    "export", "backup", "restore", "reset", "remove", "hide", "show",
    "change", "edit", "add", "delete", "custom", "default", "credits",
}


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract meaningful keywords from text for search."""
    if not text:
        return []

    # Lowercase and extract words
    words = re.findall(r'\b[a-z]+\b', text.lower())

    # Filter and score words
    keyword_scores = {}
    for word in words:
        if len(word) < 3:
            continue
        if word in STOP_WORDS:
            continue

        # Score based on relevance
        score = 1
        if word in THEME_KEYWORDS:
            score = 3  # Boost theme-related keywords

        if word not in keyword_scores:
            keyword_scores[word] = 0
        keyword_scores[word] += score

    # Sort by score and return top keywords
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: -x[1])
    return [kw for kw, score in sorted_keywords[:max_keywords]]


def build_search_query(theme_tag: str, keywords: List[str]) -> str:
    """Build HelpScout search query from theme and keywords."""
    parts = []

    if theme_tag:
        parts.append(f"tag:{theme_tag}")

    # Add keywords
    if keywords:
        parts.append(" ".join(keywords[:3]))  # Use top 3 keywords

    return " ".join(parts)


async def find_similar_tickets(
    helpscout_client,
    theme_tag: str,
    subject: str,
    message: str,
    exclude_id: int = None,
    limit: int = 3
) -> List[Dict[str, Any]]:
    """Find similar past tickets with agent responses.

    Returns list of relevant past tickets with their agent replies.
    """
    # The owned mailbox ranks answered conversations itself. Ask it first; fall back to the
    # keyword search when it is not selected, fails or knows no similar answer.
    ranked = getattr(helpscout_client, "precedents", None)
    if callable(ranked):
        try:
            found = await ranked(
                conversation_id=exclude_id,
                text=f"{subject}\n{message}",
                tag=theme_tag or "",
                limit=limit,
            )
        except Exception as exc:
            logger.warning("Ranked past-ticket lookup failed: %s", type(exc).__name__)
            found = None
        # The theme stays a filter, as in the keyword search: an answer about another theme
        # is no evidence that this question has been answered before.
        if found and theme_tag:
            found = [item for item in found if theme_tag.lower() in item.get("shared_tags", [])]
        if found:
            return found[:limit]

    # Extract keywords from subject and message
    combined_text = f"{subject} {message}"
    keywords = extract_keywords(combined_text)

    if not keywords and not theme_tag:
        return []

    # Build search query
    query = build_search_query(theme_tag, keywords)

    if not query.strip():
        return []

    # Search for similar conversations
    try:
        conversations = await helpscout_client.search_conversations(
            query=query,
            status="all",
            limit=limit + 2  # Get a few extra in case we need to filter
        )
    except httpx.HTTPError as exc:
        logger.warning("HelpScout past-ticket search failed for query %r: %s", query, exc)
        return []

    similar_tickets = []
    for conv in conversations:
        conv_id = conv.get("id")

        # Skip the current ticket
        if exclude_id and conv_id == exclude_id:
            continue

        # Skip if no subject
        if not conv.get("subject"):
            continue

        # Get full conversation with replies
        try:
            full_conv = await helpscout_client.get_conversation_with_replies(conv_id)

            # Only include if there are agent replies
            if full_conv.get("agent_replies"):
                similar_tickets.append({
                    "id": conv_id,
                    "subject": full_conv["subject"],
                    "replies": full_conv["agent_replies"]
                })

                if len(similar_tickets) >= limit:
                    break

        except Exception as e:
            print(f"Error fetching conversation {conv_id}: {e}")
            continue

    return similar_tickets


def format_past_tickets_context(similar_tickets: List[Dict[str, Any]]) -> str:
    """Format similar tickets into context for the AI."""
    if not similar_tickets:
        return ""

    parts = ["=== SIMILAR PAST TICKETS (use these as reference) ==="]

    for ticket in similar_tickets:
        parts.append(f"\n**Past ticket: {ticket['subject']}**")

        # Include the most recent/relevant reply (usually the first one that solved it)
        for reply in ticket["replies"][:1]:  # Just use the first reply
            # Truncate long replies
            reply_text = reply["body"]
            if len(reply_text) > 800:
                reply_text = reply_text[:800] + "..."
            parts.append(f"Agent response: {reply_text}")

    parts.append("\n=== END PAST TICKETS ===")

    return "\n".join(parts)
