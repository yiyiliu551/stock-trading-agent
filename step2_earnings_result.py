"""
pipeline/step2_earnings_result.py
Author: Yang
Description: Detect whether a stock beat earnings expectations.
             Fetches news text via DuckDuckGo, then calls Claude to classify.

⚡ CUSTOMISE: Replace _fetch_news() with a Playwright browser scraper
              for more reliable headline extraction.
"""

import logging
import requests
from datetime import datetime

from config import EPS_BEAT_THRESHOLD
from ai.earnings_analyzer import analyze_earnings_beat

logger = logging.getLogger(__name__)


# ── News fetch (⚡ replace with Playwright for better results) ─────────────────

def _fetch_news(ticker: str) -> str:
    """
    Search DuckDuckGo for the latest earnings news for *ticker*.
    ⚡ Simplified: swap this function for Playwright scraping.
    """
    quarter = f"Q{(datetime.now().month - 1) // 3 + 1}"
    year    = datetime.now().year
    query   = f"{ticker} earnings {quarter} {year} EPS beat miss"
    try:
        r    = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1"},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        data = r.json()
        text = data.get("AbstractText", "")
        if not text:
            # Fallback to first RelatedTopic
            topics = data.get("RelatedTopics", [])
            for t in topics:
                if isinstance(t, dict) and t.get("Text"):
                    text = t["Text"]
                    break
        return text or f"No news found for {ticker} {quarter} {year}"
    except Exception as e:
        logger.warning("News fetch failed for %s: %s", ticker, e)
        return f"Search error: {e}"


# ── Pipeline entry point ───────────────────────────────────────────────────────

def check_earnings_beat(ticker: str) -> dict:
    """
    Determine whether *ticker* beat EPS by at least EPS_BEAT_THRESHOLD%.

    Returns:
        {ticker, beat, beat_pct, confidence, reason, qualifies}
        qualifies = True only when beat=True AND beat_pct >= EPS_BEAT_THRESHOLD
    """
    news_text = _fetch_news(ticker)
    result    = analyze_earnings_beat(ticker, news_text)
    result["ticker"]    = ticker
    result["qualifies"] = (
        result.get("beat", False)
        and result.get("beat_pct", 0) >= EPS_BEAT_THRESHOLD
    )
    logger.info("Step 2 %s: qualifies=%s (beat_pct=%.1f%%)",
                ticker, result["qualifies"], result.get("beat_pct", 0))
    return result
