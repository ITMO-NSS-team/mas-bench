"""Web search / page extraction for SwarmAgentic tool roles.

Tool-parity shims: these hit the *same* backends as AutoMAS's MCP web toolset
(``automas.mcp.servers.web``) — search goes through ``benchlib.search``, which
both systems call, and extraction uses the same markitdown URL→Markdown pipeline
with the same browser-like session headers — so on search benchmarks neither
system has an information-access advantage. If the AutoMAS servers change their
extraction defaults, mirror the change here.
"""

from __future__ import annotations

import os
import json
import time

from benchlib.search import SearchError, web_search

# Same browser-like headers as automas.mcp.servers.web.server's extract().
_EXTRACT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
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
_EXTRACT_CACHE: dict[str, str] = {}
_LAST_WIKIPEDIA_REQUEST = 0.0


def _truncate_text(text: str, max_lines: int | None) -> str:
    """Truncate text to max_lines with indicator (mirrors automas content_utils)."""
    if max_lines is None:
        return text
    lines = text.split("\n")
    if len(lines) > max_lines:
        truncated = "\n".join(lines[:max_lines])
        return f"{truncated}\n\n... (truncated, showing {max_lines} of {len(lines)} lines)"
    return text


def valid_extraction(content: str) -> bool:
    """Reject error pages and corrupt/binary text before it enters the cache."""
    stripped = content.strip()
    lowered = stripped.lower()
    if len(stripped) < 200 or any(
        marker in lowered
        for marker in ("403", "404", "access denied", "not found", "content extraction failed")
    ):
        return False
    if stripped.startswith(("%PDF", "PK\x03\x04", "\x89PNG")):
        return False
    printable = sum(char.isprintable() or char in "\n\r\t" for char in stripped)
    return stripped.count("�") / len(stripped) < 0.01 and printable / len(stripped) > 0.9


def do_web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> str:
    """Search the web via the shared backend; return results as a JSON string."""
    try:
        results = web_search(query, max_results)
    except SearchError as e:
        return str(e)
    return json.dumps({"query": query, "results": results})


def do_web_extract(url: str, max_lines: int | None = EXTRACT_MAX_LINES) -> str:
    """Convert a web page (HTML, PDF, ...) to Markdown, like AutoMAS's extract tool."""
    try:
        from markitdown import MarkItDown
    except ImportError:
        return (
            "web_extract unavailable: the 'markitdown' package is not installed "
            "(install automas-research or add markitdown to the environment)."
        )
    if url in _EXTRACT_CACHE:
        return _EXTRACT_CACHE[url]
    try:
        import requests

        global _LAST_WIKIPEDIA_REQUEST
        if "wikipedia.org" in url.lower():
            remaining = 1.0 - (time.monotonic() - _LAST_WIKIPEDIA_REQUEST)
            if remaining > 0:
                time.sleep(remaining)
            _LAST_WIKIPEDIA_REQUEST = time.monotonic()
        session = requests.Session()
        session.headers.update(_EXTRACT_HEADERS)
        request = session.request
        session.request = lambda *args, **kwargs: request(  # type: ignore[method-assign]
            *args, timeout=20, **kwargs
        )
        md = MarkItDown(requests_session=session)
        result = md.convert(url)
        content = _truncate_text(result.text_content, max_lines)
        if valid_extraction(content):
            _EXTRACT_CACHE[url] = content
        return content
    except Exception as e:
        return f"Content extraction failed: {e}"
