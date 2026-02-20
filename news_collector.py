"""
idle/news_collector.py
Author: Yang
Description: Fetch recent news headlines for all watchlist stocks via DuckDuckGo.
             Results cached in module-level _news_cache for the sentiment runner.

⚡ CUSTOMISE: Replace _fetch with a Playwright scraper for richer content.
"""

import time
import logging
import requests

from config import STOCKS
from tools.heartbeat import send_idle_report

logger = logging.getLogger(__name__)

# Module-level cache shared with idle/sentiment_runner.py
_news_cache: dict[str, list[str]] = {}

_REQUEST_DELAY = 0.5   # seconds between ticker requests


def _fetch(ticker: str) -> list[str]:
    """
    Retrieve up to 5 recent headlines for *ticker* from DuckDuckGo.
    Returns empty list on failure.
    ⚡ Replace with Playwright for more reliable extraction.
    """
    query = f"{ticker} stock news earnings today"
    try:
        r    = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1"},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        data = r.json()
        headlines: list[str] = []
        abstract = data.get("AbstractText", "")
        if abstract:
            headlines.append(abstract[:200])
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict) and topic.get("Text"):
                headlines.append(topic["Text"][:200])
        return headlines[:5]
    except Exception as e:
        logger.warning("News fetch failed for %s: %s", ticker, e)
        return []


def collect_all_news() -> dict[str, list[str]]:
    """
    Fetch headlines for every ticker in STOCKS.
    Updates _news_cache in-place and sends a WeChat report.

    Returns:
        {ticker: [headline, ...]}
    """
    logger.info("Idle: collecting news for %d tickers", len(STOCKS))
    results: dict[str, list[str]] = {}
    hits = 0

    for ticker in STOCKS:
        headlines       = _fetch(ticker)
        results[ticker] = headlines
        if headlines:
            hits += 1
        time.sleep(_REQUEST_DELAY)

    _news_cache.clear()
    _news_cache.update(results)

    summary_lines = []
    for ticker, hl in results.items():
        preview = hl[0][:60] + "..." if hl else "no news"
        summary_lines.append(f"> **{ticker}** ({len(hl)} headlines): {preview}")

    send_idle_report(
        "News Collection",
        f"> Fetched {hits}/{len(STOCKS)} tickers\n\n" + "\n".join(summary_lines[:8]),
    )
    return results
