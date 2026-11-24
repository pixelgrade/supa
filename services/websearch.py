"""Web search fallback for finding solutions not in knowledge base."""

import os
import httpx
from services.company_pack import product_settings
from typing import List, Dict, Any, Optional


async def search_web(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Search the web for solutions using Tavily API.

    Returns list of results with title, url, and content snippet.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "include_domains": product_settings()["search_domains"],
                    "max_results": limit
                }
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")[:500]
                })
            return results

    except Exception as e:
        print(f"Web search error: {e}")
        return []


def format_web_results(results: List[Dict[str, Any]]) -> str:
    """Format web search results for inclusion in prompt context."""
    if not results:
        return ""

    parts = ["**Web Search Results:**"]
    for r in results:
        parts.append(f"\n- **{r['title']}**\n  URL: {r['url']}\n  {r['content']}")

    return "\n".join(parts)


def build_search_query(subject: str, message: str, theme_tag: Optional[str] = None) -> str:
    """
    Build a focused search query from ticket details.

    Extracts error messages, plugin names, and key terms.
    """
    # Look for PHP/WordPress errors
    error_patterns = [
        "Fatal error:",
        "TypeError:",
        "Error message:",
        "Uncaught",
        "critical error"
    ]

    # Check if message contains an error
    has_error = any(pattern.lower() in message.lower() for pattern in error_patterns)

    if has_error:
        # Extract the core error message
        lines = message.split('\n')
        for line in lines:
            if any(pattern.lower() in line.lower() for pattern in error_patterns):
                # Clean and truncate for search
                query = line.strip()[:200]
                if theme_tag:
                    query = f"{theme_tag} {query}"
                return product_settings()["search_prefix"] + query

    # Default: use subject + theme
    query = subject
    if theme_tag:
        query = f"{theme_tag} {query}"

    return product_settings()["search_prefix"] + query
