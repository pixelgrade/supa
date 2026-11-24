import re

from services.company_pack import prompt, wording
from agents.claude_cli import invoke_claude
from services.email_html import ensure_email_paragraph_spacing


def add_paragraph_spacing(html: str) -> str:
    """Keep the agent and Help Scout write paths on the same stable formatter."""
    return ensure_email_paragraph_spacing(html)


def _normalize_urls(urls: set[str]) -> set[str]:
    cleaned = set()
    for url in urls:
        url = (url or "").strip().rstrip('.,;:')
        if not url.startswith(("http://", "https://")):
            continue
        cleaned.add(url)
        cleaned.add(url.rstrip("/"))
    return cleaned


def extract_context_urls(
    knowledge_context: str,
    past_tickets_context: str,
    extra_urls: list[str] | None = None,
) -> set:
    """Extract all real URLs from the context provided to the model."""
    all_text = f"{knowledge_context}\n{past_tickets_context}"
    context_urls = set(re.findall(r'https?://[^\s<>"\')\]]+', all_text))
    extra_url_set = set(extra_urls or [])
    return _normalize_urls(context_urls | extra_url_set)


def validate_response_urls(html: str, allowed_urls: set) -> str:
    """Remove links to URLs not found in the provided context.

    Keeps the link text but removes the <a> tag wrapping.
    """
    if not allowed_urls:
        # If no context URLs, strip ALL links to be safe
        return re.sub(r'<a\s+[^>]*href="[^"]*"[^>]*>(.*?)</a>', r'\1', html)

    def check_link(match):
        full_tag = match.group(0)
        href = match.group(1)
        link_text = match.group(2)

        # Check if this URL (or a prefix of it) is in the allowed set
        href_clean = href.rstrip('/')
        for allowed in allowed_urls:
            if href_clean == allowed.rstrip('/') or href_clean.startswith(allowed.rstrip('/')):
                return full_tag  # Keep the link
            if allowed.rstrip('/').startswith(href_clean):
                return full_tag  # Keep the link

        # URL not in context — remove the link, keep the text
        print(f"Removed hallucinated URL from draft: {href}")
        return link_text

    return re.sub(r'<a\s+[^>]*href="([^"]*)"[^>]*>(.*?)</a>', check_link, html)

RESPONDER_SYSTEM_PROMPT = prompt("responder_system_prompt")

REWRITE_SYSTEM_PROMPT = RESPONDER_SYSTEM_PROMPT + """

REWRITE MODE:
- You are revising an existing support draft, not writing from scratch unless the original draft is unusable
- Keep the parts that are already correct
- Fix the specific weaknesses called out in the rewrite brief
- Stay concise and avoid adding new speculative troubleshooting steps"""


def finalize_response_html(content: str, knowledge_context: str, past_tickets_context: str, allowed_urls: set = None) -> str:
    """Normalize and sanitize responder HTML before it is stored or sent."""
    content = content.strip()

    # Strip markdown code block markers if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
    if content.endswith("```"):
        content = content[:-3].rstrip()

    context_urls = allowed_urls if allowed_urls else extract_context_urls(
        knowledge_context, past_tickets_context
    )
    content = validate_response_urls(content, context_urls)
    return add_paragraph_spacing(content)


async def generate_response(
    customer_name: str,
    subject: str,
    message: str,
    theme_tag: str,
    knowledge_context: str,
    past_tickets_context: str = "",
    internal_note: str = "",
    is_classic_theme: bool = False,
    conversation_history: list = None,
    allowed_urls: set = None,
    customer_context: str = ""
) -> str:
    """Generate a response using Claude CLI, knowledge context, and past tickets."""

    # Determine theme type
    theme_type = wording("classic_product_kind" if is_classic_theme else "modern_product_kind")

    # Build the user prompt
    parts = [
        f"Customer name: {customer_name}",
        f"Subject: {subject}",
        f"Theme: {theme_tag or 'not specified'} ({theme_type})",
    ]

    # Add conversation history if this is a follow-up
    if conversation_history and len(conversation_history) > 1:
        parts.append("")
        parts.append("=== CONVERSATION HISTORY (oldest to newest) ===")
        for msg in conversation_history[:-1]:  # Exclude the latest message (shown separately)
            role_label = "Customer" if msg["role"] == "customer" else "Support (you)"
            parts.append(f"[{role_label}]: {msg['message']}")
        parts.append("=== END HISTORY ===")
        parts.append("")
        parts.append("Latest customer message (respond to this):")
        parts.append(message)
    else:
        parts.append("")
        parts.append("Customer message:")
        parts.append(message)

    parts.extend([
        "",
        "Relevant knowledge/documentation:",
        knowledge_context or "(No specific documentation found)",
    ])

    if past_tickets_context:
        parts.extend([
            "",
            past_tickets_context,
        ])

    if customer_context:
        parts.extend(["", customer_context])

    if internal_note:
        parts.extend([
            "",
            "*** INTERNAL INSTRUCTIONS (prioritize these): ***",
            internal_note
        ])

    parts.extend([
        "",
        "Write the reply in HTML format."
    ])

    user_prompt = "\n".join(parts)

    try:
        content = invoke_claude(
            prompt=user_prompt,
            model="sonnet",
            system_prompt=RESPONDER_SYSTEM_PROMPT,
            timeout=600,
        )
        return finalize_response_html(content, knowledge_context, past_tickets_context, allowed_urls)

    except Exception as e:
        print(f"Response generation error: {e}")
        return None


async def rewrite_response(
    customer_name: str,
    subject: str,
    message: str,
    original_draft: str,
    rewrite_brief: str,
    additional_context: str = "",
    theme_tag: str = "",
    knowledge_context: str = "",
    past_tickets_context: str = "",
    internal_note: str = "",
    allowed_urls: set = None,
) -> str:
    """Rewrite an existing response using targeted evaluator guidance."""

    parts = [
        f"Customer name: {customer_name}",
        f"Subject: {subject}",
        f"Theme: {theme_tag or 'not specified'}",
        "",
        "Latest customer message:",
        message,
        "",
        "Current draft to improve:",
        original_draft,
        "",
        "Rewrite brief:",
        rewrite_brief,
    ]

    if additional_context:
        parts.extend(["", "Additional investigation context:", additional_context])
    if knowledge_context:
        parts.extend(["", "Relevant knowledge/documentation:", knowledge_context])
    if past_tickets_context:
        parts.extend(["", past_tickets_context])
    if internal_note:
        parts.extend([
            "",
            "*** INTERNAL INSTRUCTIONS (prioritize these): ***",
            internal_note,
        ])

    parts.extend([
        "",
        "Revise the draft in HTML format.",
    ])

    user_prompt = "\n".join(parts)

    try:
        content = invoke_claude(
            prompt=user_prompt,
            model="sonnet",
            system_prompt=REWRITE_SYSTEM_PROMPT,
            timeout=600,
        )
        return finalize_response_html(content, knowledge_context, past_tickets_context, allowed_urls)
    except Exception as e:
        print(f"Response rewrite error: {e}")
        return None


# --- Original OpenAI implementation (kept for rollback) ---
#
# import os
# from openai import OpenAI
#
# async def generate_response(
#     customer_name, subject, message, theme_tag, knowledge_context,
#     past_tickets_context="", internal_note="", is_classic_theme=False,
#     conversation_history=None, allowed_urls=None, customer_context=""
# ) -> str:
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         print("Warning: OPENAI_API_KEY not set")
#         return None
#     client = OpenAI(api_key=api_key)
#     theme_type = wording("classic_product_kind" if is_classic_theme else "modern_product_kind")
#     parts = [f"Customer name: {customer_name}", f"Subject: {subject}", f"Theme: {theme_tag or 'not specified'} ({theme_type})"]
#     # ... (same prompt building as above) ...
#     user_prompt = "\n".join(parts)
#     try:
#         response = client.responses.create(
#             model="gpt-5.2",
#             max_output_tokens=1024,
#             instructions=RESPONDER_SYSTEM_PROMPT,
#             input=user_prompt,
#         )
#         content = response.output_text.strip()
#         # ... (same post-processing as above) ...
#         return content
#     except Exception as e:
#         print(f"Response generation error: {e}")
#         return None
