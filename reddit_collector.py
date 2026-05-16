"""
idle/reddit_collector.py
Author: Yang
Description: Fetch stock-related posts from Reddit.

Two fetch modes (auto-selected):
  1. Reddit JSON API  — official endpoint, no key required (preferred)
  2. Playwright       — browser simulation fallback if API blocked

Why keep Playwright as fallback:
  - Reddit occasionally blocks datacenter IPs on the JSON API
  - Playwright simulates real browser, harder to block
  - Same pattern as xiaohongshu_collector for consistency

Data flow:
    Reddit JSON API → parse posts → ChromaDB (news_raw)
    ↳ fallback: Playwright browser if API returns 403/429
"""

import time
import asyncio
import logging
import hashlib
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from config import STOCKS, CHROMA_PATH
from chroma_utils import upsert_docs, query_docs
import random

logger = logging.getLogger(__name__)

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_POSTS       = 10
MAX_TEXT_LENGTH = 800


def _human_delay(min_s: float = 1.5, max_s: float = 4.5):
    """Random delay mimicking human reading rhythm.
    Humans don't click at fixed intervals — they read, pause, move on."""
    t = random.uniform(min_s, max_s)
    if random.random() < 0.1:
        t += random.uniform(2.0, 6.0)   # occasional longer pause
    logger.debug("Human delay: %.1fs", t)
    time.sleep(t)


async def _human_delay_async(min_s: float = 0.5, max_s: float = 2.5):
    """Async version for Playwright browser steps."""
    t = random.uniform(min_s, max_s)
    await asyncio.sleep(t)


# ── Mode 1: Reddit JSON API ────────────────────────────────────────────────────

def _search_subreddit_api(subreddit: str, query: str) -> list[dict]:
    """
    Search Reddit via official JSON API endpoint.
    No OAuth required — public search only.
    """
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {"q": query, "restrict_sr": "on", "sort": "relevance",
              "t": "week", "limit": MAX_POSTS}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        posts = r.json().get("data", {}).get("children", [])
        results = []
        for post in posts:
            p = post.get("data", {})
            title    = p.get("title", "")
            selftext = p.get("selftext", "")
            if selftext in ("[deleted]", "[removed]", ""):
                selftext = ""
            if title:
                results.append({
                    "title":       title[:300],
                    "text":        selftext[:MAX_TEXT_LENGTH],
                    "score":       p.get("score", 0),
                    "url":         f"https://reddit.com{p.get('permalink', '')}",
                    "created_utc": p.get("created_utc", 0),
                    "subreddit":   subreddit,
                    "fetch_mode":  "api",
                })
        logger.info("Reddit API r/%s → %d posts for '%s'", subreddit, len(results), query)
        return results
    except requests.HTTPError as e:
        status = e.response.status_code if e.response else 0
        if status in (403, 429):
            logger.warning("Reddit API blocked (status %d) for r/%s — will try Playwright", status, subreddit)
            raise  # signal caller to switch to Playwright
        logger.warning("Reddit API HTTP error r/%s: %s", subreddit, e)
        return []
    except Exception as e:
        logger.warning("Reddit API failed r/%s: %s", subreddit, e)
        return []


# ── Mode 2: Playwright browser fallback ───────────────────────────────────────

async def _search_subreddit_playwright(subreddit: str, query: str) -> list[dict]:
    """
    Browser simulation fallback for when Reddit API is blocked.
    Visits old.reddit.com (simpler HTML, easier to parse).
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning("Playwright not installed — skip browser fallback")
        return []

    search_url = (
        f"https://old.reddit.com/r/{subreddit}/search"
        f"?q={requests.utils.quote(query)}&restrict_sr=on&sort=relevance&t=week"
    )
    posts = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        page = await context.new_page()
        try:
            logger.info("Playwright Reddit: r/%s query='%s'", subreddit, query)
            await page.goto(search_url, timeout=20000, wait_until="domcontentloaded")
            await _human_delay_async(1.0, 3.0)

            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")

            # old.reddit.com uses <div class="search-result-link"> structure
            for item in soup.select("div.search-result-link")[:MAX_POSTS]:
                title_el = item.select_one("a.search-title")
                score_el = item.select_one("span.search-score")
                title  = title_el.get_text(strip=True)[:300] if title_el else ""
                url    = title_el.get("href", "") if title_el else ""
                score  = int(score_el.get_text(strip=True).replace(",","").split()[0]) if score_el else 0
                if title:
                    posts.append({
                        "title": title, "text": "", "score": score,
                        "url": url, "created_utc": 0,
                        "subreddit": subreddit, "fetch_mode": "playwright",
                    })

            logger.info("Playwright Reddit r/%s → %d posts", subreddit, len(posts))
        except Exception as e:
            logger.warning("Playwright Reddit r/%s failed: %s", subreddit, e)
        finally:
            await browser.close()

    return posts


# ── Auto-select fetch mode ─────────────────────────────────────────────────────

def _search_subreddit(subreddit: str, query: str) -> list[dict]:
    """
    Try Reddit JSON API first. If blocked (403/429), fall back to Playwright.
    """
    try:
        results = _search_subreddit_api(subreddit, query)
        return results
    except requests.HTTPError:
        # API blocked — try Playwright
        try:
            return asyncio.run(_search_subreddit_playwright(subreddit, query))
        except Exception as e:
            logger.warning("Playwright fallback also failed for r/%s: %s", subreddit, e)
            return []


# ── ChromaDB storage ───────────────────────────────────────────────────────────

def _doc_id(ticker: str, url: str) -> str:
    return hashlib.md5(f"{ticker}::{url}".encode()).hexdigest()[:16]


def _store_to_chroma(ticker: str, posts: list[dict]) -> int:
    if not posts:
        return 0
    ids, docs, metas = [], [], []
    now_iso = datetime.utcnow().isoformat()
    for p in posts:
        full_text = p["title"]
        if p.get("text"):
            full_text += "\n" + p["text"]
        full_text = full_text[:MAX_TEXT_LENGTH]
        ids.append(_doc_id(ticker, p["url"]))
        docs.append(full_text)
        metas.append({
            "ticker":      ticker,
            "source":      "reddit",
            "subreddit":   p["subreddit"],
            "score":       int(p["score"]),
            "url":         p["url"],
            "collected_at": now_iso,
            "created_utc": int(p["created_utc"]),
            "fetch_mode":  p.get("fetch_mode", "api"),
        })
    return upsert_docs(CHROMA_PATH, ids, docs, metas)


# ── Public API ─────────────────────────────────────────────────────────────────

def collect_reddit_for_ticker(ticker: str) -> list[str]:
    all_posts, seen_urls = [], set()
    for sub in SUBREDDITS:
        for p in _search_subreddit(sub, f"{ticker} stock"):
            if p["url"] not in seen_urls:
                seen_urls.add(p["url"])
                all_posts.append(p)
        _human_delay()

    all_posts.sort(key=lambda x: x["score"], reverse=True)
    _store_to_chroma(ticker, all_posts)
    return [p["title"] for p in all_posts]


def collect_all_reddit() -> dict[str, list[str]]:
    logger.info("Reddit collection starting — %d tickers", len(STOCKS))
    results = {}
    for ticker in STOCKS:
        results[ticker] = collect_reddit_for_ticker(ticker)
        logger.info("%s → %d Reddit posts", ticker, len(results[ticker]))
        _human_delay()
    return results


def query_reddit_context(ticker: str, n_results: int = 5) -> list[str]:
    return query_docs(
        CHROMA_PATH, f"{ticker} stock market", n_results=n_results,
        where={"$and": [{"source": {"$eq": "reddit"}}, {"ticker": {"$eq": ticker}}]},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print("=== Reddit Collector Test ===\n")
    titles = collect_reddit_for_ticker("NVDA")
    print(f"NVDA: {len(titles)} posts")
    for i, t in enumerate(titles[:5], 1):
        print(f"  {i}. {t}")
    docs = query_reddit_context("NVDA", n_results=3)
    print(f"\nChromaDB query: {len(docs)} docs")
