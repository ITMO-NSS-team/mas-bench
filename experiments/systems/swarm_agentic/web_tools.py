"""Web search / page extraction for SwarmAgentic tool roles.

Tool-parity shims: these hit the *same* backends as AutoMAS's MCP web toolset
(``automas.mcp.servers.web``) — the same SearXNG instance with the same default
engine list, and the same markitdown URL→Markdown pipeline with the same
browser-like session headers — so on search benchmarks neither system has an
information-access advantage. If the AutoMAS servers change their defaults,
mirror the change here.
"""

from __future__ import annotations

import os

import httpx

# Same default engine list as automas.mcp.servers.web.searxng_server.
_DEFAULT_ENGINES = "bing,duckduckgo,brave,mullvadleta,mullvadleta brave,yahoo,presearch"

# Same browser-like headers as automas.mcp.servers.web.server's extract().
_EXTRACT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

WEB_SEARCH_MAX_RESULTS = 10
# Extracted pages flow into every downstream role's prompt, so cap them
# (AutoMAS agents manage their own context and read pages untruncated).
EXTRACT_MAX_LINES = int(os.environ.get("SWARM_EXTRACT_MAX_LINES", "300"))


def _truncate_text(text: str, max_lines: int | None) -> str:
    """Truncate text to max_lines with indicator (mirrors automas content_utils)."""
    if max_lines is None:
        return text
    lines = text.split("\n")
    if len(lines) > max_lines:
        truncated = "\n".join(lines[:max_lines])
        return f"{truncated}\n\n... (truncated, showing {max_lines} of {len(lines)} lines)"
    return text


def do_web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> str:
    """Search the web via the shared SearXNG instance; return results as text.

    Same request shape as the AutoMAS searxng MCP tool's defaults:
    general category, moderate safesearch, the same engine list.
    """
    instance_url = os.environ.get("SEARXNG_URL", "http://localhost:8888")
    params = {
        "q": query,
        "format": "json",
        "categories": "general",
        "safesearch": 1,
        "engines": _DEFAULT_ENGINES,
    }
    try:
        response = httpx.get(f"{instance_url}/search", params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as e:
        return f"SearXNG HTTP error {e.response.status_code}: {e.response.text[:200]}"
    except httpx.RequestError as e:
        return (
            f"SearXNG connection error: {e}. "
            f"Check if SearXNG is running at {instance_url}"
        )

    results = data.get("results", [])[:max_results]
    if not results:
        return f"No search results for: {query}"

    parts = [f"Search results for: {query}"]
    for infobox in data.get("infoboxes", []):
        content = infobox.get("content")
        if content:
            parts.append(f"[Infobox] {content}")
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("content", "")
        parts.append(f"{i}. {title}\n   {url}\n   {snippet}")
    return "\n".join(parts)


def do_web_extract(url: str, max_lines: int | None = EXTRACT_MAX_LINES) -> str:
    """Convert a web page (HTML, PDF, ...) to Markdown, like AutoMAS's extract tool."""
    try:
        from markitdown import MarkItDown
    except ImportError:
        return (
            "web_extract unavailable: the 'markitdown' package is not installed "
            "(install automas-research or add markitdown to the environment)."
        )
    try:
        import requests

        session = requests.Session()
        session.headers.update(_EXTRACT_HEADERS)
        md = MarkItDown(requests_session=session)
        result = md.convert(url)
        return _truncate_text(result.text_content, max_lines)
    except Exception as e:
        return f"Content extraction failed: {e}"
