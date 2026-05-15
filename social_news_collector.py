"""
idle/social_news_collector.py
Author: Yang
Description: Unified social media + news collector.
             Aggregates Reddit, Xiaohongshu, and DuckDuckGo news.
             All data is stored in ChromaDB (news_raw collection).
             Replaces the original news_collector.py as the main idle task.

Data flow:
    DuckDuckGo news   ┐
    Reddit posts      ├──→ ChromaDB (news_raw) ──→ sentiment analysis
    XHS notes         ┘
"""

import time
import logging
import hashlib
from datetime import datetime

import requests
from config import STOCKS, CHROMA_PATH
from chroma_utils import upsert_docs, query_docs
from tools.heartbeat import send_idle_report

# Import source collectors
from idle.reddit_collector import collect_reddit_for_ticker
from idle.xiaohongshu_collector import collect_xhs_for_ticker

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 800
_REQUEST_DELAY = 1.0

# ── DuckDuckGo (original source, kept as base layer) ──────────────────────────

def _fetch_duckduckgo(ticker: str) -> list[dict]:
    """Fetch news from DuckDuckGo API (original implementation, enhanced)."""
    query = f"{ticker} stock news earnings today"
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1"},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        data = r.json()
        items = []
        abstract = data.get("AbstractText", "")
        if abstract:
            items.append({
                "title": f"{ticker} - DuckDuckGo abstract",
                "text": abstract[:MAX_TEXT_LENGTH],
                "url": data.get("AbstractURL", ""),
                "source": "duckduckgo",
            })
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict) and topic.get("Text"):
                items.append({
                    "title": topic["Text"][:200],
                    "text": topic["Text"][:MAX_TEXT_LENGTH],
                    "url": topic.get("FirstURL", ""),
                    "source": "duckduckgo",
                })
        return items
    except Exception as e:
        logger.warning("DuckDuckGo fetch failed for %s: %s", ticker, e)
        return []


def _store_news_to_chroma(ticker: str, items: list[dict], source: str) -> int:
    """Upsert news items into ChromaDB news_raw collection."""
    if not items:
        return 0
    col = _get_collection()
    ids, docs, metas = [], [], []
    now_iso = datetime.utcnow().isoformat()

    for item in items:
        full_text = item.get("title", "")
        if item.get("text"):
            full_text += "\n" + item["text"]
        full_text = full_text[:MAX_TEXT_LENGTH]
        if not full_text.strip():
            continue

        raw = f"{ticker}::{source}::{item.get('url', '')}::{item.get('title', '')}"
        doc_id = hashlib.md5(raw.encode()).hexdigest()[:16]
        ids.append(doc_id)
        docs.append(full_text)
        metas.append({
            "ticker":       ticker,
            "source":       source,
            "url":          item.get("url", "")[:500],
            "collected_at": now_iso,
        })

    if not ids:
        return 0

    return upsert_docs(CHROMA_PATH, ids, docs, metas)


# ── Public API ─────────────────────────────────────────────────────────────────

def collect_all_social_news() -> dict[str, list[str]]:
    """
    Main idle task: collect from all sources for all tickers.
    Each ticker gets Reddit + XHS + DuckDuckGo data stored in ChromaDB.
    Returns {ticker: [headline, ...]} for backward compatibility with sentiment_runner.
    """
    logger.info("Social news collection — %d tickers × 3 sources", len(STOCKS))
    results: dict[str, list[str]] = {}

    reddit_hits = 0
    xhs_hits = 0
    ddg_hits = 0

    for ticker in STOCKS:
        all_headlines = []

        # 1. Reddit
        try:
            reddit_titles = collect_reddit_for_ticker(ticker)
            all_headlines.extend(reddit_titles)
            if reddit_titles:
                reddit_hits += 1
        except Exception as e:
            logger.error("Reddit collection failed for %s: %s", ticker, e)

        # 2. Xiaohongshu
        try:
            xhs_titles = collect_xhs_for_ticker(ticker)
            all_headlines.extend(xhs_titles)
            if xhs_titles:
                xhs_hits += 1
        except Exception as e:
            logger.error("XHS collection failed for %s: %s", ticker, e)

        # 3. DuckDuckGo (baseline — always works)
        try:
            ddg_items = _fetch_duckduckgo(ticker)
            _store_news_to_chroma(ticker, ddg_items, "duckduckgo")
            ddg_titles = [item["title"] for item in ddg_items]
            all_headlines.extend(ddg_titles)
            if ddg_items:
                ddg_hits += 1
        except Exception as e:
            logger.error("DuckDuckGo collection failed for %s: %s", ticker, e)

        results[ticker] = all_headlines
        time.sleep(_REQUEST_DELAY)

    # WeChat report
    summary_lines = []
    for ticker, hl in list(results.items())[:8]:
        preview = hl[0][:60] + "..." if hl else "no news"
        summary_lines.append(f"> **{ticker}** ({len(hl)} total): {preview}")

    send_idle_report(
        "Social News Collection V3",
        (
            f"> Sources: Reddit({reddit_hits}) + XHS({xhs_hits}) + DDG({ddg_hits}) / {len(STOCKS)} tickers\n"
            f"> Storage: ChromaDB (news_raw collection)\n\n"
            + "\n".join(summary_lines)
        ),
    )
    return results


def query_all_context(ticker: str, n_results: int = 10) -> list[str]:
    """
    Retrieve all stored social/news context for *ticker* from ChromaDB.
    Used by sentiment analysis and AI validation nodes.
    """
    try:
        col = _get_collection()
        results = col.query(
            query_texts=[f"{ticker} stock earnings news"],
            n_results=n_results,
            where={"ticker": {"$eq": ticker}},
        )
        docs = results.get("documents", [[]])[0]
        return docs
    except Exception as e:
        logger.warning("ChromaDB context query failed for %s: %s", ticker, e)
        return []


def query_by_source(ticker: str, source: str, n_results: int = 5) -> list[str]:
    """
    Query ChromaDB filtered by specific source: 'reddit', 'xiaohongshu', 'duckduckgo'.
    """
    try:
        col = _get_collection()
        results = col.query(
            query_texts=[f"{ticker} stock"],
            n_results=n_results,
            where={"$and": [
                {"ticker": {"$eq": ticker}},
                {"source": {"$eq": source}},
            ]},
        )
        return results.get("documents", [[]])[0]
    except Exception as e:
        logger.warning("ChromaDB source query failed (%s/%s): %s", ticker, source, e)
        return []


# ── Backward-compatible wrapper ────────────────────────────────────────────────
# sentiment_runner.py imports collect_all_news from news_collector.py
# Add this alias so old imports still work during migration

def collect_all_news() -> dict[str, list[str]]:
    """Alias for collect_all_social_news() — backward compatible with sentiment_runner."""
    return collect_all_social_news()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print("=== Social News Collector Test ===\n")
    # Test single ticker
    from idle.reddit_collector import collect_reddit_for_ticker
    from idle.xiaohongshu_collector import collect_xhs_for_ticker

    print("Testing NVDA...")
    reddit = collect_reddit_for_ticker("NVDA")
    print(f"Reddit: {len(reddit)} posts")
    for t in reddit[:3]:
        print(f"  - {t[:80]}")

    xhs = collect_xhs_for_ticker("NVDA")
    print(f"\nXHS: {len(xhs)} posts")
    for t in xhs[:3]:
        print(f"  - {t[:80]}")

    print("\nQuerying ChromaDB context...")
    ctx = query_all_context("NVDA", n_results=5)
    print(f"Retrieved {len(ctx)} docs from ChromaDB")
