"""
idle/reddit_collector.py
Author: Yang
Description: Fetch stock-related posts from Reddit (r/wallstreetbets, r/stocks, r/investing)
             using old.reddit.com JSON endpoints — no API key required.
             Results are stored in ChromaDB for later RAG retrieval.

Data flow:
    Reddit JSON API → parse posts/comments → chunk text → ChromaDB (news_raw collection)
"""

import time
import logging
import hashlib
from datetime import datetime
from typing import Optional

import requests

from config import STOCKS, CHROMA_PATH

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "StockMarket",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.reddit.com/",
    "X-Requested-With": "XMLHttpRequest",
}

REQUEST_DELAY = 2.0          # seconds between requests (respect Reddit rate limit)
MAX_POSTS_PER_SUB = 10       # top posts per subreddit search
MAX_TEXT_LENGTH = 800        # max chars per chunk stored in ChromaDB


# ── ChromaDB via chroma_utils (handles ONNX fallback) ─────────────────────────
from chroma_utils import upsert_docs, query_docs


# ── Reddit fetch ───────────────────────────────────────────────────────────────

def _search_subreddit(subreddit: str, query: str) -> list[dict]:
    """
    Search one subreddit for posts matching query.
    Returns list of {title, text, score, url, created_utc}.
    Uses old.reddit.com JSON endpoint (no OAuth required).
    """
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": query,
        "restrict_sr": "on",
        "sort": "relevance",
        "t": "week",        # posts from last 7 days
        "limit": MAX_POSTS_PER_SUB,
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        posts = data.get("data", {}).get("children", [])
        results = []
        for post in posts:
            p = post.get("data", {})
            title = p.get("title", "")
            selftext = p.get("selftext", "")
            # Skip deleted/empty posts
            if not title or selftext in ("[deleted]", "[removed]", ""):
                selftext = ""
            results.append({
                "title": title[:300],
                "text": selftext[:MAX_TEXT_LENGTH],
                "score": p.get("score", 0),
                "url": f"https://reddit.com{p.get('permalink', '')}",
                "created_utc": p.get("created_utc", 0),
                "subreddit": subreddit,
            })
        logger.info("r/%s → %d posts for '%s'", subreddit, len(results), query)
        return results
    except Exception as e:
        logger.warning("Reddit fetch failed (r/%s, query='%s'): %s", subreddit, query, e)
        return []


def _doc_id(ticker: str, url: str) -> str:
    """Stable unique ID = hash of ticker + url."""
    raw = f"{ticker}::{url}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _store_to_chroma(ticker: str, posts: list[dict]) -> int:
    """
    Upsert Reddit posts into ChromaDB news_raw collection.
    Returns number of documents upserted.
    """
    if not posts:
        return 0
    ids, docs, metas = [], [], []
    now_iso = datetime.utcnow().isoformat()

    for p in posts:
        # Combine title + body as the document text for embedding
        full_text = p["title"]
        if p["text"]:
            full_text += "\n" + p["text"]
        full_text = full_text[:MAX_TEXT_LENGTH]

        doc_id = _doc_id(ticker, p["url"])
        ids.append(doc_id)
        docs.append(full_text)
        metas.append({
            "ticker":      ticker,
            "source":      "reddit",
            "subreddit":   p["subreddit"],
            "score":       int(p["score"]),
            "url":         p["url"],
            "collected_at": now_iso,
            "created_utc": int(p["created_utc"]),
        })

    return upsert_docs(CHROMA_PATH, ids, docs, metas)


# ── Public API ─────────────────────────────────────────────────────────────────

def collect_reddit_for_ticker(ticker: str) -> list[str]:
    """
    Collect Reddit posts about *ticker* from all configured subreddits.
    Stores results in ChromaDB and returns list of post titles (for logging).
    """
    all_posts = []
    query = f"{ticker} stock"

    for sub in SUBREDDITS:
        posts = _search_subreddit(sub, query)
        all_posts.extend(posts)
        time.sleep(REQUEST_DELAY)

    # Deduplicate by URL
    seen_urls = set()
    unique_posts = []
    for p in all_posts:
        if p["url"] not in seen_urls:
            seen_urls.add(p["url"])
            unique_posts.append(p)

    # Sort by Reddit score (most upvoted first)
    unique_posts.sort(key=lambda x: x["score"], reverse=True)

    _store_to_chroma(ticker, unique_posts)
    return [p["title"] for p in unique_posts]


def collect_all_reddit() -> dict[str, list[str]]:
    """
    Collect Reddit posts for every ticker in STOCKS watchlist.
    Returns {ticker: [post_title, ...]} for use in sentiment analysis.
    """
    logger.info("Reddit collection starting — %d tickers", len(STOCKS))
    results: dict[str, list[str]] = {}

    for ticker in STOCKS:
        titles = collect_reddit_for_ticker(ticker)
        results[ticker] = titles
        logger.info("%s → %d Reddit posts", ticker, len(titles))
        time.sleep(REQUEST_DELAY)

    return results


def query_reddit_context(ticker: str, n_results: int = 5) -> list[str]:
    """
    Retrieve stored Reddit posts for *ticker* from ChromaDB.
    Returns list of document texts for RAG context injection.
    """
    return query_docs(
        CHROMA_PATH,
        f"{ticker} stock market",
        n_results=n_results,
        where={"$and": [{"source": {"$eq": "reddit"}}, {"ticker": {"$eq": ticker}}]},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print("=== Reddit Collector Test ===")
    titles = collect_reddit_for_ticker("NVDA")
    print(f"\nNVDA Reddit posts ({len(titles)} total):")
    for i, t in enumerate(titles[:5], 1):
        print(f"  {i}. {t}")

    print("\nQuerying ChromaDB...")
    docs = query_reddit_context("NVDA", n_results=3)
    for i, d in enumerate(docs, 1):
        print(f"\n--- Doc {i} ---\n{d[:200]}...")
