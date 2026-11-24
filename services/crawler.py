import re
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup

from services.company_pack import product_settings

_DOCS_ORIGIN = product_settings()["docs_url"].removesuffix("/docs/")


def extract_text_from_html(html: str) -> str:
    """Extract clean text from HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    # Get text
    text = soup.get_text(separator=" ", strip=True)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def get_url_depth(url: str) -> int:
    """Get the depth of a docs URL.

    /docs/theme/ = 2
    /docs/theme/category/ = 3
    /docs/theme/category/article/ = 4
    """
    path = url.replace(_DOCS_ORIGIN, "").strip("/")
    return len(path.split("/"))


def is_category_page(url: str) -> bool:
    """Check if URL is a category/index page (not an actual article)."""
    return get_url_depth(url) <= 3


MIN_CONTENT_LENGTH = 300  # Minimum chars for a valid article


def parse_docs_page(html: str, url: str) -> Dict[str, Any]:
    """Parse a documentation page."""
    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True).split(" - ")[0]

    # Extract main content
    article = soup.find("article") or soup.find("main") or soup.find("body")
    content = extract_text_from_html(str(article)) if article else ""

    # Extract theme from URL
    # The product slug is the path segment after docs.
    theme = "all"
    match = re.search(r'/docs/([^/]+)/', url)
    if match:
        theme = match.group(1)

    return {
        "id": url,
        "title": title,
        "content": content,
        "theme": theme,
        "source": "docs",
        "url": url
    }


async def crawl_theme_docs(
    theme_slug: str,
    base_url: str = _DOCS_ORIGIN,
    seed_urls: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Crawl all docs pages for a theme."""
    docs = []
    docs_url = f"{base_url}/docs/{theme_slug}/"
    visited = set()
    to_visit = [docs_url]

    if seed_urls:
        for url in seed_urls:
            if url not in to_visit:
                to_visit.append(url)

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        while to_visit and len(visited) < 100:  # Limit to 100 pages per theme
            url = to_visit.pop(0)

            if url in visited:
                continue
            visited.add(url)

            try:
                response = await client.get(url)
                if response.status_code != 200:
                    continue

                html = response.text
                doc = parse_docs_page(html, url)

                # Skip category pages and short content
                if is_category_page(url):
                    pass  # Still crawl links, but don't add to docs
                elif len(doc["content"]) < MIN_CONTENT_LENGTH:
                    pass  # Skip pages with too little content
                elif doc["content"]:
                    docs.append(doc)

                # Find links to other docs pages
                soup = BeautifulSoup(html, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    # Handle both relative and absolute URLs
                    if href.startswith("/docs/" + theme_slug):
                        full_url = base_url + href
                    elif href.startswith(base_url + "/docs/" + theme_slug):
                        full_url = href
                    else:
                        continue

                    # Skip feed URLs
                    if "/feed/" in full_url:
                        continue

                    if full_url not in visited:
                        to_visit.append(full_url)

            except Exception as e:
                print(f"Error crawling {url}: {e}")

    return docs


async def crawl_all_themes(themes: List[str]) -> List[Dict[str, Any]]:
    """Crawl docs for multiple themes."""
    all_docs = []

    for theme in themes:
        print(f"Crawling docs for {theme}...")
        docs = await crawl_theme_docs(theme)
        all_docs.extend(docs)
        print(f"  Found {len(docs)} pages")

    return all_docs
