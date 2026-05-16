"""
idle/xiaohongshu_collector.py
Author: Yang
Description: Fetch stock-related notes from Xiaohongshu (小红书).

Two fetch modes (auto-selected):
  1. Playwright  — simulates real browser, handles JS rendering (preferred)
  2. requests    — static fallback, often blocked by XHS JS wall

Why Playwright:
  - XHS renders content via JavaScript, static requests return empty skeleton
  - Playwright controls real Chromium browser, executes JS, waits for content
  - Simulates human behavior: scroll, wait, extract — harder to block
  - No API key required

Data flow:
    Playwright (browser) → extract titles/summaries → ChromaDB (news_raw)
    ↳ fallback: requests + BeautifulSoup if Playwright unavailable
"""

import time
import asyncio
import logging
import hashlib
import re
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from config import STOCKS, CHROMA_PATH
from chroma_utils import upsert_docs, query_docs
import random

logger = logging.getLogger(__name__)

XHS_SEARCH_URL  = "https://www.xiaohongshu.com/search_result"
MAX_TEXT_LENGTH = 600
_ua_index       = 0


def _human_delay(min_s: float = 2.5, max_s: float = 6.0):
    """Random delay mimicking human browsing — not a fixed interval.
    15% chance of a longer pause (like human got distracted)."""
    t = random.uniform(min_s, max_s)
    if random.random() < 0.15:
        t += random.uniform(3.0, 9.0)
    logger.debug("Human delay: %.1fs", t)
    time.sleep(t)


async def _human_delay_async(min_s: float = 0.8, max_s: float = 3.5):
    """Async version for Playwright browser steps."""
    t = random.uniform(min_s, max_s)
    await asyncio.sleep(t)

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Version/17.0 Mobile/15E148 Safari/604.1",
]


def _next_ua() -> str:
    global _ua_index
    ua = USER_AGENTS[_ua_index % len(USER_AGENTS)]
    _ua_index += 1
    return ua


def _build_search_query(ticker: str) -> str:
    ticker_map = {
        "TSLA": "特斯拉 股票", "AAPL": "苹果 AAPL 股票",
        "NVDA": "英伟达 NVDA 股票", "META": "Meta 脸书 股票",
        "GOOGL": "谷歌 GOOGL 股票", "MSFT": "微软 MSFT 股票",
        "AMZN": "亚马逊 AMZN 股票", "AMD": "AMD 芯片 股票",
        "QCOM": "高通 QCOM 股票", "WDC": "西部数据 WDC 股票",
        "SNDK": "西部数据 闪迪 股票", "CRM": "Salesforce CRM 股票",
        "PANW": "Palo Alto PANW 网络安全 股票",
    }
    return ticker_map.get(ticker, f"{ticker} 美股")


# ── Mode 1: Playwright browser simulation ──────────────────────────────────────

async def _fetch_with_playwright(query: str) -> list[dict]:
    """
    Simulate real human browser visiting XHS search page.
    Steps: navigate → wait for JS render → scroll → extract DOM.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise ImportError("playwright not installed — run: pip install playwright && playwright install chromium")

    search_url = (
        f"{XHS_SEARCH_URL}?keyword={requests.utils.quote(query)}"
        f"&source=web_search_result_notes"
    )
    posts = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=_next_ua(),
            locale="zh-CN",
            viewport={"width": 1280, "height": 800},
        )
        # Hide webdriver flag from XHS bot detection
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            "Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN','zh','en']});"
        )
        page = await context.new_page()
        try:
            logger.info("Playwright: navigating XHS for '%s'", query)
            await page.goto(search_url, timeout=20000, wait_until="domcontentloaded")

            # Wait for JS-rendered note cards
            try:
                await page.wait_for_selector(
                    "[class*='note-item'], [class*='search-item']", timeout=8000
                )
            except Exception:
                logger.warning("Playwright: note cards timeout for '%s'", query)

            # Scroll to trigger lazy loading (simulate human behavior)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            await _human_delay_async(1.0, 2.5)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await _human_delay_async(0.8, 2.0)

            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")

            # Extract note cards from rendered DOM
            note_cards = soup.select(
                "[class*='note-item'], [class*='search-item'], "
                "[class*='note-card'], [class*='feeds-container'] > div"
            )
            for card in note_cards[:10]:
                title_el = card.select_one("span[class*='title'], a[class*='title'], h3")
                desc_el  = card.select_one("span[class*='desc'], p[class*='desc']")
                link_el  = card.select_one("a[href]")
                title    = title_el.get_text(strip=True)[:300] if title_el else ""
                summary  = desc_el.get_text(strip=True)[:MAX_TEXT_LENGTH] if desc_el else ""
                url      = link_el.get("href", "") if link_el else ""
                if url and not url.startswith("http"):
                    url = f"https://www.xiaohongshu.com{url}"
                if title and len(title) > 4:
                    posts.append({"title": title, "summary": summary, "url": url})

            # Fallback: parse __INITIAL_STATE__ JSON embedded in page script
            if not posts:
                for script in soup.find_all("script"):
                    if script.string and "__INITIAL_STATE__" in script.string:
                        titles = re.findall(r'"title"\s*:\s*"([^"]{5,200})"', script.string)
                        descs  = re.findall(r'"desc"\s*:\s*"([^"]{5,200})"',  script.string)
                        nids   = re.findall(r'"noteId"\s*:\s*"([a-f0-9]{24})"', script.string)
                        for i, t in enumerate(titles[:8]):
                            posts.append({
                                "title":   t,
                                "summary": descs[i] if i < len(descs) else "",
                                "url":     f"https://www.xiaohongshu.com/explore/{nids[i]}" if i < len(nids) else "",
                            })
                        break

            logger.info("Playwright XHS '%s' → %d posts", query, len(posts))
        except Exception as e:
            logger.warning("Playwright XHS failed for '%s': %s", query, e)
        finally:
            await browser.close()

    return posts


# ── Mode 2: Static requests fallback ──────────────────────────────────────────

def _fetch_static(query: str) -> list[dict]:
    """Static fallback — limited by XHS JS rendering, often returns empty."""
    headers = {
        "User-Agent": _next_ua(),
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://www.xiaohongshu.com/",
    }
    try:
        r = requests.get(
            XHS_SEARCH_URL,
            params={"keyword": query, "source": "web_search_result_notes"},
            headers=headers, timeout=15,
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        posts = []
        og = soup.find("meta", property="og:description")
        if og and og.get("content"):
            posts.append({"title": query, "summary": og["content"][:MAX_TEXT_LENGTH], "url": r.url})
        for card in soup.select("[class*='note-item'], [class*='search-item']")[:8]:
            t = card.select_one("span[class*='title'], h3")
            d = card.select_one("span[class*='desc']")
            a = card.select_one("a[href]")
            title = t.get_text(strip=True)[:300] if t else ""
            url   = a["href"] if a else ""
            if url and not url.startswith("http"):
                url = f"https://www.xiaohongshu.com{url}"
            if title:
                posts.append({"title": title, "summary": d.get_text(strip=True) if d else "", "url": url})
        if not posts:
            logger.warning("Static XHS: JS wall blocked content for '%s'", query)
        return posts
    except Exception as e:
        logger.warning("Static XHS failed for '%s': %s", query, e)
        return []


# ── Auto-select fetch mode ─────────────────────────────────────────────────────

def _fetch_search_page(query: str) -> list[dict]:
    """
    Try Playwright first (JS rendering), fall back to static requests.
    """
    try:
        posts = asyncio.run(_fetch_with_playwright(query))
        if posts:
            return posts
        logger.info("Playwright returned 0 results, trying static fallback")
    except ImportError:
        logger.info("Playwright not installed — using static fetch")
    except Exception as e:
        logger.warning("Playwright error: %s — using static fallback", e)
    return _fetch_static(query)


# ── ChromaDB storage ───────────────────────────────────────────────────────────

def _doc_id(ticker: str, url: str, title: str) -> str:
    return hashlib.md5(f"{ticker}::{url}::{title}".encode()).hexdigest()[:16]


def _store_to_chroma(ticker: str, posts: list[dict]) -> int:
    if not posts:
        return 0
    ids, docs, metas = [], [], []
    now_iso = datetime.utcnow().isoformat()
    for p in posts:
        full_text = (p["title"] + ("\n" + p["summary"] if p.get("summary") else ""))[:MAX_TEXT_LENGTH]
        ids.append(_doc_id(ticker, p.get("url", ""), p["title"]))
        docs.append(full_text)
        metas.append({"ticker": ticker, "source": "xiaohongshu",
                      "url": p.get("url", ""), "collected_at": now_iso})
    return upsert_docs(CHROMA_PATH, ids, docs, metas)


# ── Public API ─────────────────────────────────────────────────────────────────

def collect_xhs_for_ticker(ticker: str) -> list[str]:
    query = _build_search_query(ticker)
    posts = _fetch_search_page(query)
    _store_to_chroma(ticker, posts)
    _human_delay()
    return [p["title"] for p in posts]


def collect_all_xhs() -> dict[str, list[str]]:
    logger.info("XHS collection starting — %d tickers", len(STOCKS))
    results = {}
    for ticker in STOCKS:
        results[ticker] = collect_xhs_for_ticker(ticker)
        logger.info("%s → %d XHS posts", ticker, len(results[ticker]))
    return results


def query_xhs_context(ticker: str, n_results: int = 5) -> list[str]:
    return query_docs(
        CHROMA_PATH, f"{ticker} 美股 股票", n_results=n_results,
        where={"$and": [{"source": {"$eq": "xiaohongshu"}}, {"ticker": {"$eq": ticker}}]},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print("=== XHS Collector (Playwright mode) ===\n")
    titles = collect_xhs_for_ticker("NVDA")
    print(f"NVDA: {len(titles)} posts")
    for i, t in enumerate(titles[:5], 1):
        print(f"  {i}. {t}")
