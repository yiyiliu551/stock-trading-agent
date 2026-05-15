"""
idle/news_pipeline.py
Author: Yang
Description: Full news ingestion pipeline connecting all components.

Pipeline stages:
  Stage 1 — Fetch
    reddit_collector  → raw Reddit posts  (list of dicts)
    xiaohongshu_collector → raw XHS posts

  Stage 2 — LLM Filter (news_filter.py)
    LLM reviews each post: useful / borderline / noise
    Returns structured filter_result with extracted facts

  Stage 3 — User Verification
    format_for_verification() prints what will be stored
    Caller can inspect before committing to ChromaDB

  Stage 4 — GraphRAG Storage
    Useful + borderline posts → GraphEvent triples
    Stored with E5 embeddings in ChromaDB graph_rag_events collection
    Each event is time-stamped and isolated (anti-hallucination)

Usage:
    # Full pipeline for all tickers
    from idle.news_pipeline import run_pipeline
    results = run_pipeline(confirm=True)   # confirm=True auto-stores after showing verification
    
    # Single ticker
    from idle.news_pipeline import run_pipeline_for_ticker
    result = run_pipeline_for_ticker("NVDA", confirm=False)
    # inspect result["verification_text"] then call commit_to_graph(result)

    # Query stored events (for LLM injection)
    from idle.news_pipeline import get_context_for_llm
    context = get_context_for_llm("NVDA", "earnings beat guidance")
"""

import logging
import time
from datetime import datetime
from typing import Optional

from config import STOCKS
from tools.heartbeat import send_idle_report

from idle.reddit_collector import collect_reddit_for_ticker
from idle.xiaohongshu_collector import collect_xhs_for_ticker
from idle.news_filter import filter_posts, format_for_verification
from tools.graph_rag_store import (
    store_events,
    query_events,
    format_events_for_llm,
    extract_events_from_filter,
)

logger = logging.getLogger(__name__)

# ── Stage 1: Fetch ─────────────────────────────────────────────────────────────

def _fetch_raw_posts(ticker: str) -> list[dict]:
    """
    Collect raw posts from Reddit + XHS for one ticker.
    Returns unified list of post dicts with 'source' field.
    """
    all_posts = []

    # Reddit
    try:
        from idle.reddit_collector import _search_subreddit, SUBREDDITS
        for sub in SUBREDDITS:
            posts = _search_subreddit(sub, f"{ticker} stock")
            for p in posts:
                all_posts.append({
                    "id":     f"reddit_{p['url'].split('/')[-2] if '/' in p['url'] else len(all_posts)}",
                    "title":  p["title"],
                    "text":   p.get("text", ""),
                    "source": "reddit",
                    "url":    p["url"],
                    "score":  p.get("score", 0),
                })
            time.sleep(1.5)
        logger.info("%s: fetched %d Reddit posts", ticker, len(all_posts))
    except Exception as e:
        logger.warning("Reddit fetch failed for %s: %s", ticker, e)

    xhs_start = len(all_posts)

    # XHS
    try:
        from idle.xiaohongshu_collector import _fetch_search_page, _build_search_query
        query = _build_search_query(ticker)
        xhs_posts = _fetch_search_page(query)
        for p in xhs_posts:
            all_posts.append({
                "id":     f"xhs_{len(all_posts)}",
                "title":  p["title"],
                "text":   p.get("summary", ""),
                "source": "xiaohongshu",
                "url":    p.get("url", ""),
                "score":  0,
            })
        logger.info("%s: fetched %d XHS posts", ticker, len(all_posts) - xhs_start)
    except Exception as e:
        logger.warning("XHS fetch failed for %s: %s", ticker, e)

    return all_posts


# ── Stage 2+3: Filter + Verify ─────────────────────────────────────────────────

def run_pipeline_for_ticker(ticker: str, confirm: bool = False) -> dict:
    """
    Run full pipeline for one ticker.

    Args:
        ticker:  stock symbol
        confirm: if True, automatically store after showing verification output
                 if False, returns result without storing (caller must call commit)

    Returns:
        {
          "ticker":            str,
          "raw_count":         int,        # total posts fetched
          "filter_result":     dict,       # from news_filter.py
          "verification_text": str,        # human-readable review
          "events":            list,       # GraphEvent objects ready to store
          "stored":            bool,       # True if already committed to ChromaDB
          "stored_count":      int,
        }
    """
    logger.info("Pipeline starting for %s", ticker)

    # Stage 1: Fetch
    raw_posts = _fetch_raw_posts(ticker)

    if not raw_posts:
        logger.warning("%s: no raw posts fetched", ticker)
        return {
            "ticker": ticker,
            "raw_count": 0,
            "filter_result": {},
            "verification_text": f"[{ticker}] No posts fetched from Reddit/XHS.",
            "events": [],
            "stored": False,
            "stored_count": 0,
        }

    # Stage 2: LLM filter
    filter_result = filter_posts(ticker, raw_posts)

    # Stage 3: Format for user verification
    verification_text = format_for_verification(filter_result)
    print(verification_text)   # always print so user can review

    # Extract GraphEvents from filter output
    events = extract_events_from_filter(filter_result)

    result = {
        "ticker":            ticker,
        "raw_count":         len(raw_posts),
        "filter_result":     filter_result,
        "verification_text": verification_text,
        "events":            events,
        "stored":            False,
        "stored_count":      0,
    }

    # Stage 4: Store (if confirm=True)
    if confirm and events:
        n = commit_to_graph(result)
        result["stored"]       = True
        result["stored_count"] = n

    return result


def commit_to_graph(pipeline_result: dict) -> int:
    """
    Stage 4: Commit filtered events to ChromaDB GraphRAG store.
    Call this after user has verified the pipeline_result.
    Returns count of stored events.
    """
    events = pipeline_result.get("events", [])
    if not events:
        logger.info("%s: no events to commit", pipeline_result.get("ticker"))
        return 0

    n = store_events(events)
    logger.info("%s: committed %d events to GraphRAG", pipeline_result.get("ticker"), n)
    return n


# ── Full pipeline for all tickers ─────────────────────────────────────────────

def run_pipeline(tickers: Optional[list[str]] = None, confirm: bool = True) -> dict:
    """
    Run full pipeline for all tickers (or a subset).
    
    Args:
        tickers: list of tickers, defaults to config.STOCKS
        confirm: auto-store after filter (default True for idle scheduler)
    
    Returns:
        {ticker: pipeline_result}
    """
    tickers = tickers or STOCKS
    logger.info("News pipeline starting: %d tickers, confirm=%s", len(tickers), confirm)

    results = {}
    total_raw     = 0
    total_useful  = 0
    total_stored  = 0

    for ticker in tickers:
        r = run_pipeline_for_ticker(ticker, confirm=confirm)
        results[ticker] = r
        total_raw    += r["raw_count"]
        total_useful += len(r.get("filter_result", {}).get("useful", []))
        total_stored += r["stored_count"]
        time.sleep(2.0)   # rate limit between tickers

    # WeChat summary
    summary_lines = []
    for ticker, r in list(results.items())[:8]:
        fr = r.get("filter_result", {})
        u  = len(fr.get("useful", []))
        n  = len(fr.get("noise", []))
        summary_lines.append(f"> **{ticker}**: {r['raw_count']} posts → {u} useful / {n} noise")

    send_idle_report(
        "News Pipeline",
        (
            f"> Tickers: {len(tickers)} | Raw: {total_raw} | "
            f"Useful: {total_useful} | Stored: {total_stored}\n\n"
            + "\n".join(summary_lines)
        ),
    )

    return results


# ── LLM context retrieval ──────────────────────────────────────────────────────

def get_context_for_llm(
    ticker: str,
    query:  str = "",
    start_time: Optional[str] = None,
    end_time:   Optional[str] = None,
    n: int = 10,
) -> str:
    """
    Get formatted GraphRAG context for injection into LLM prompt.
    
    This is the function called by nodes.py / react_verifier.py
    to give the LLM time-series news context with anti-hallucination formatting.
    
    Args:
        ticker: stock symbol
        query:  what to search for, e.g. "earnings guidance analyst"
        n:      max events to return
    
    Returns:
        Formatted string ready to inject into LLM system prompt or user message.
    """
    events = query_events(
        ticker     = ticker,
        query_text = query or f"{ticker} stock news",
        start_time = start_time,
        end_time   = end_time,
        n_results  = n,
    )
    return format_events_for_llm(events)


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, types, os

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Mock config + heartbeat
    cfg = types.ModuleType("config")
    cfg.STOCKS = ["NVDA", "TSLA"]
    cfg.CHROMA_PATH = "/tmp/test_pipeline"
    sys.modules["config"] = cfg

    hb = types.ModuleType("tools.heartbeat")
    hb.send_idle_report = lambda *a, **kw: None
    sys.modules["tools"] = types.ModuleType("tools")
    sys.modules["tools.heartbeat"] = hb

    os.makedirs("/tmp/test_pipeline", exist_ok=True)
    sys.path.insert(0, ".")

    print("=== News Pipeline Test (offline mock) ===\n")

    # Simulate filter_result directly (no real network/API calls)
    from idle.news_filter import _empty_result
    from tools.graph_rag_store import GraphEvent, store_events, query_events, format_events_for_llm

    mock_filter = {
        "ticker": "NVDA",
        "timestamp": "2026-05-15T10:00:00",
        "total": 3,
        "useful": [
            {"title": "NVDA Q2 EPS $23.41 beats by 60%", "text": "Data center doubled",
             "source": "reddit", "url": "https://r.com/1",
             "verdict": "useful", "reason": "earnings beat",
             "facts": ["EPS beat +60%", "data center revenue doubled"],
             "sentiment": "bullish", "event_type": "earnings",
             "filtered_at": "2026-05-15T10:00:01"},
            {"title": "英伟达财报大超预期 AI需求强劲", "text": "EPS比预期高60%",
             "source": "xiaohongshu", "url": "https://xhs.com/1",
             "verdict": "useful", "reason": "earnings news Chinese",
             "facts": ["earnings beat"], "sentiment": "bullish",
             "event_type": "earnings", "filtered_at": "2026-05-15T10:00:02"},
        ],
        "borderline": [],
        "noise": [
            {"title": "NVDA to the moon 🚀", "text": "trust me bro",
             "source": "reddit", "url": "https://r.com/2",
             "verdict": "noise", "reason": "meme no data",
             "facts": [], "sentiment": "n/a", "event_type": "noise",
             "filtered_at": "2026-05-15T10:00:03"},
        ],
        "verdicts": {}
    }

    print("── Stage 3: Verification output ──")
    from idle.news_filter import format_for_verification
    print(format_for_verification(mock_filter))

    print("\n── Stage 4: Extract events + store ──")
    events = extract_events_from_filter(mock_filter)
    print(f"Extracted {len(events)} events from filter output")
    n = store_events(events)
    print(f"Stored {n} events in GraphRAG ChromaDB")

    print("\n── Context for LLM ──")
    ctx = get_context_for_llm("NVDA", "earnings beat guidance", n=5)
    print(ctx)
    print("✅ Pipeline test complete")
