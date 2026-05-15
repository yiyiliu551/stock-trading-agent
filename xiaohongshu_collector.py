"""
idle/xiaohongshu_collector.py
Author: Yang
Description: Fetch stock-related notes from Xiaohongshu (小红书) search page.
             Uses requests + BeautifulSoup to scrape the public search results.
             Results are stored in ChromaDB for RAG retrieval.

⚠️  IMPORTANT NOTES:
    - 小红书 does NOT have a public API. This scrapes public search pages.
    - XHS blocks scrapers aggressively — rate limit strictly (≥3s between requests).
    - Login wall: only public post titles/summaries are visible without login.
    - If scraping fails, the collector logs a warning and returns empty (graceful degradation).
    - MCP Fetch tool (when available) can replace _fetch_search_page() for more reliable extraction.

Data flow:
    XHS search page → BeautifulSoup → extract titles/summaries → ChromaDB (news_raw collection)
"""

import time
import logging
import hashlib
import re
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config import STOCKS, CHROMA_PATH

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
XHS_SEARCH_URL = "https://www.xiaohongshu.com/search_result"

# Rotate user-agents to reduce block rate
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]

REQUEST_DELAY = 3.5          # XHS is strict — don't go below 3s
MAX_TEXT_LENGTH = 600
_ua_index = 0


def _next_ua() -> str:
    """Round-robin user-agent rotation."""
    global _ua_index
    ua = USER_AGENTS[_ua_index % len(USER_AGENTS)]
    _ua_index += 1
    return ua


# ── ChromaDB via chroma_utils ─────────────────────────────────────────────────
from chroma_utils import upsert_docs, query_docs


# ── XHS fetch ─────────────────────────────────────────────────────────────────

def _build_search_query(ticker: str) -> str:
    """
    Map US stock tickers to Chinese search terms.
    XHS users mainly discuss stocks in Chinese.
    """
    ticker_map = {
        "TSLA": "特斯拉 股票",
        "AAPL": "苹果 AAPL 股票",
        "NVDA": "英伟达 NVDA 股票",
        "META": "Meta 脸书 股票",
        "GOOGL": "谷歌 GOOGL 股票",
        "MSFT": "微软 MSFT 股票",
        "AMZN": "亚马逊 AMZN 股票",
        "AMD":  "AMD 芯片 股票",
        "QCOM": "高通 QCOM 股票",
        "WDC":  "西部数据 WDC 股票",
        "CRM":  "Salesforce CRM 股票",
        "PANW": "Palo Alto PANW 网络安全 股票",
    }
    return ticker_map.get(ticker, f"{ticker} 美股")


def _fetch_search_page(query: str) -> list[dict]:
    """
    Fetch XHS search results for *query*.
    Returns list of {title, summary, url}.

    NOTE: XHS renders content via JavaScript, so static requests often return
    an empty skeleton. This method tries to extract any server-side rendered text.
    For better results, swap this function body with an MCP Fetch call or
    a Playwright async scraper.
    """
    headers = {
        "User-Agent": _next_ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://www.xiaohongshu.com/",
    }
    params = {
        "keyword": query,
        "source": "web_search_result_notes",
    }
    try:
        r = requests.get(
            XHS_SEARCH_URL,
            params=params,
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        posts = []

        # Try to extract from JSON embedded in page (XHS embeds __INITIAL_STATE__)
        scripts = soup.find_all("script")
        for script in scripts:
            if script.string and "__INITIAL_STATE__" in script.string:
                # Extract JSON-like content
                match = re.search(r'"title"\s*:\s*"([^"]{5,200})"', script.string)
                if match:
                    logger.debug("Found XHS embedded data via script tag")

        # Fallback: try meta tags and visible text
        # XHS meta og:description sometimes has content preview
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            posts.append({
                "title": query,
                "summary": og_desc["content"][:MAX_TEXT_LENGTH],
                "url": r.url,
            })

        # Try note card elements (may be present if SSR is enabled)
        note_cards = soup.select("[class*='note-item'], [class*='search-item'], [class*='feeds-container']")
        for card in note_cards[:8]:
            title_el = card.select_one("span[class*='title'], a[class*='title'], h3")
            desc_el  = card.select_one("span[class*='desc'], p[class*='desc']")
            link_el  = card.select_one("a[href]")

            title   = title_el.get_text(strip=True)[:300] if title_el else ""
            summary = desc_el.get_text(strip=True)[:MAX_TEXT_LENGTH] if desc_el else ""
            url     = link_el["href"] if link_el else ""
            if not url.startswith("http"):
                url = f"https://www.xiaohongshu.com{url}"

            if title:
                posts.append({"title": title, "summary": summary, "url": url})

        if posts:
            logger.info("XHS search '%s' → %d items", query, len(posts))
        else:
            logger.warning(
                "XHS returned no parseable content for '%s' "
                "(JS rendering blocked — consider MCP Fetch for this source)", query
            )
        return posts

    except requests.HTTPError as e:
        if e.response and e.response.status_code in (403, 429):
            logger.warning("XHS blocked request (status %s) — backing off", e.response.status_code)
        else:
            logger.warning("XHS HTTP error for '%s': %s", query, e)
        return []
    except Exception as e:
        logger.warning("XHS fetch failed for '%s': %s", query, e)
        return []


def _doc_id(ticker: str, url: str, title: str) -> str:
    """Stable unique ID = hash of ticker + url + title."""
    raw = f"{ticker}::{url}::{title}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _store_to_chroma(ticker: str, posts: list[dict]) -> int:
    """Upsert XHS posts into ChromaDB news_raw collection."""
    if not posts:
        return 0
    ids, docs, metas = [], [], []
    now_iso = datetime.utcnow().isoformat()

    for p in posts:
        full_text = p["title"]
        if p.get("summary"):
            full_text += "\n" + p["summary"]
        full_text = full_text[:MAX_TEXT_LENGTH]

        doc_id = _doc_id(ticker, p.get("url", ""), p["title"])
        ids.append(doc_id)
        docs.append(full_text)
        metas.append({
            "ticker":       ticker,
            "source":       "xiaohongshu",
            "url":          p.get("url", ""),
            "collected_at": now_iso,
        })

    return upsert_docs(CHROMA_PATH, ids, docs, metas)


# ── Public API ─────────────────────────────────────────────────────────────────

def collect_xhs_for_ticker(ticker: str) -> list[str]:
    """
    Collect XHS posts about *ticker*.
    Stores to ChromaDB. Returns list of post titles for logging.
    """
    query = _build_search_query(ticker)
    posts = _fetch_search_page(query)
    _store_to_chroma(ticker, posts)
    time.sleep(REQUEST_DELAY)
    return [p["title"] for p in posts]


def collect_all_xhs() -> dict[str, list[str]]:
    """
    Collect XHS posts for every ticker in STOCKS watchlist.
    Returns {ticker: [post_title, ...]}.
    """
    logger.info("XHS collection starting — %d tickers", len(STOCKS))
    results: dict[str, list[str]] = {}

    for ticker in STOCKS:
        titles = collect_xhs_for_ticker(ticker)
        results[ticker] = titles
        logger.info("%s → %d XHS posts", ticker, len(titles))

    return results


def query_xhs_context(ticker: str, n_results: int = 5) -> list[str]:
    """
    Retrieve stored XHS posts for *ticker* from ChromaDB.
    Returns list of document texts for RAG context injection.
    """
    return query_docs(
        CHROMA_PATH,
        f"{ticker} 美股 股票",
        n_results=n_results,
        where={"$and": [{"source": {"$eq": "xiaohongshu"}}, {"ticker": {"$eq": ticker}}]},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print("=== XHS Collector Test ===")
    print("NOTE: XHS uses heavy JS rendering. Static fetch may return 0 results.")
    print("      In that case, integrate MCP Fetch tool for full page rendering.\n")
    titles = collect_xhs_for_ticker("TSLA")
    print(f"TSLA XHS posts ({len(titles)}):")
    for i, t in enumerate(titles[:5], 1):
        print(f"  {i}. {t}")
