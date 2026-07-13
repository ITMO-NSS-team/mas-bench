"""The web-search backend shared by every system under test.

Both systems must see the *same* result set for the same query, or a benchmark
measures their search backends rather than their multi-agent structure. So the
backend lives here, and each system's tool layer delegates to it: SwarmAgentic's
``web_tools.do_web_search`` and AutoMAS's ``web-search`` MCP server (the latter
runs in a subprocess launched with ``sys.executable``, so it imports this too).

Tavily is used when ``TAVILY_API_KEY`` is set, otherwise a self-hosted SearXNG at
``SEARXNG_URL``. SearXNG's free scraping engines throttle, CAPTCHA, and time out
at benchmark request rates, and an engine that fails returns *no results* rather
than an error -- which is indistinguishable from a question the web cannot
answer, and silently scores as a wrong answer. Prefer the keyed API.
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

import httpx

TAVILY_URL = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 10
_TIMEOUT = 30.0

# Same engine list the AutoMAS SearXNG server has always requested; kept so the
# fallback path stays identical to the pre-Tavily behaviour.
SEARXNG_ENGINES = "bing,duckduckgo,brave,mullvadleta,mullvadleta brave,yahoo,presearch"


class SearchResult(TypedDict):
    """One search hit, in the shape both systems' tool layers expect."""

    title: str
    url: str
    snippet: str


class SearchError(RuntimeError):
    """The search backend could not be reached or rejected the request."""


def backend() -> str:
    """Which backend a search would use right now: ``tavily`` or ``searxng``."""
    return "tavily" if os.environ.get("TAVILY_API_KEY") else "searxng"


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[SearchResult]:
    """Search the web, returning at most ``max_results`` hits.

    Raises :class:`SearchError` if the backend fails. An empty list means the
    backend answered and had nothing -- callers must not conflate the two.
    """
    if os.environ.get("TAVILY_API_KEY"):
        return _tavily(query, max_results)
    return _searxng(query, max_results)


def _tavily(query: str, max_results: int) -> list[SearchResult]:
    try:
        response = httpx.post(
            TAVILY_URL,
            json={
                "query": query,
                "max_results": max_results,
                # "advanced" costs 2 API credits instead of 1, but returns
                # relevance-ranked snippets long enough to judge a source from,
                # which is exactly what the research roles do with them.
                "search_depth": "advanced",
            },
            headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        raise SearchError(
            f"Tavily HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc
    except httpx.RequestError as exc:
        raise SearchError(f"Tavily connection error: {exc}") from exc

    return [
        SearchResult(
            title=str(hit.get("title", "")),
            url=str(hit["url"]),
            snippet=str(hit.get("content", "")),
        )
        for hit in data.get("results", [])
        if hit.get("url")
    ][:max_results]


def _searxng(query: str, max_results: int) -> list[SearchResult]:
    instance_url = os.environ.get("SEARXNG_URL", "http://localhost:8888")
    params: dict[str, Any] = {
        "q": query,
        "format": "json",
        "categories": "general",
        "safesearch": 1,
        "engines": SEARXNG_ENGINES,
    }
    try:
        response = httpx.get(f"{instance_url}/search", params=params, timeout=_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        raise SearchError(
            f"SearXNG HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc
    except httpx.RequestError as exc:
        raise SearchError(
            f"SearXNG connection error: {exc}. Is it running at {instance_url}?"
        ) from exc

    return [
        SearchResult(
            title=str(hit.get("title", "")),
            url=str(hit["url"]),
            snippet=str(hit.get("content", "")),
        )
        for hit in data.get("results", [])[:max_results]
        if hit.get("url")
    ]
