"""
idle/news_filter.py
Author: Yang
Description: LLM-based filter that decides whether raw social media posts
             (Reddit / XHS) contain useful stock-relevant information
             before writing them into ChromaDB.

Why this layer exists:
  - Reddit and XHS have a lot of noise: memes, off-topic posts, spam
  - Storing garbage degrades RAG retrieval quality (low cosine scores)
  - One LLM call batch-processes all posts for a ticker → cheap, fast

Filter verdict per post:
  useful     → store in ChromaDB with extracted structured facts
  borderline → store but mark low-confidence
  noise      → discard, log only

Output for user verification:
  Returns a structured dict the caller can print/display before final storage.
  User sees: what was found, what was kept, what was discarded.
"""

import json
import logging
from datetime import datetime

from ai.base import call_claude, parse_json_response

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

_FILTER_PROMPT = """You are a financial analyst assistant screening social media posts about stock {ticker}.
Review each post and classify it.

Posts to review:
{posts_block}

For each post ID respond with:
  verdict:   "useful" | "borderline" | "noise"
  reason:    one short phrase (max 8 words)
  facts:     list of extracted financial facts (empty if noise)
             e.g. ["EPS beat +60%", "guidance raised", "CEO resigned"]
  sentiment: "bullish" | "bearish" | "neutral" | "n/a"
  event_type: "earnings" | "management" | "analyst" | "product" | "macro" | "social" | "noise"

Classify as USEFUL if the post contains:
  - Earnings / revenue / EPS numbers
  - Management changes (CEO/CFO hire or resign)
  - Analyst upgrades/downgrades with price targets
  - Product launches with market impact
  - Regulatory or legal news
  - Short squeeze or unusual options activity

Classify as NOISE if:
  - Memes, jokes, venting without facts
  - Generic opinions without data ("I think NVDA will go up")
  - Off-topic (unrelated to stock performance)
  - Duplicate / very similar to another post in this batch

Answer ONLY in JSON, no prose, no markdown fences:
{{
  "P001": {{"verdict": "useful", "reason": "earnings beat reported", "facts": ["EPS +60%"], "sentiment": "bullish", "event_type": "earnings"}},
  "P002": {{"verdict": "noise",  "reason": "generic opinion no data",  "facts": [],          "sentiment": "n/a",    "event_type": "noise"}},
  ...
}}"""


# ── Core filter function ───────────────────────────────────────────────────────

def filter_posts(ticker: str, posts: list[dict]) -> dict:
    """
    Run LLM filter on a list of raw social media posts for one ticker.

    Args:
        ticker: stock symbol e.g. "NVDA"
        posts:  list of {"id": str, "title": str, "text": str, "source": str, "url": str}

    Returns:
        {
          "ticker": "NVDA",
          "timestamp": "...",
          "total": 10,
          "useful": [...],       # posts classified as useful
          "borderline": [...],   # posts classified as borderline
          "noise": [...],        # posts classified as noise
          "verdicts": {...}      # raw LLM output per post ID
        }
    """
    if not posts:
        return _empty_result(ticker)

    # Build numbered posts block for the prompt
    lines = []
    id_map = {}  # P001 → original post dict
    for i, post in enumerate(posts[:20]):  # max 20 posts per batch
        pid = f"P{i+1:03d}"
        id_map[pid] = post
        title = post.get("title", "")[:200]
        text  = post.get("text", "")[:300]
        src   = post.get("source", "unknown")
        lines.append(f"[{pid}] ({src}) {title}")
        if text and text != title:
            lines.append(f"       {text}")
    posts_block = "\n".join(lines)

    prompt = _FILTER_PROMPT.format(ticker=ticker, posts_block=posts_block)

    try:
        raw      = call_claude(prompt, max_tokens=1500)
        verdicts = parse_json_response(raw, {})
    except Exception as e:
        logger.error("LLM filter failed for %s: %s", ticker, e)
        verdicts = {}

    # Bucket posts by verdict
    useful, borderline, noise = [], [], []
    for pid, post in id_map.items():
        v = verdicts.get(pid, {})
        verdict = v.get("verdict", "noise")
        enriched = {
            **post,
            "verdict":    verdict,
            "reason":     v.get("reason", "no classification"),
            "facts":      v.get("facts", []),
            "sentiment":  v.get("sentiment", "n/a"),
            "event_type": v.get("event_type", "noise"),
            "filtered_at": datetime.utcnow().isoformat(),
        }
        if verdict == "useful":
            useful.append(enriched)
        elif verdict == "borderline":
            borderline.append(enriched)
        else:
            noise.append(enriched)

    result = {
        "ticker":     ticker,
        "timestamp":  datetime.utcnow().isoformat(),
        "total":      len(posts),
        "useful":     useful,
        "borderline": borderline,
        "noise":      noise,
        "verdicts":   verdicts,
    }

    logger.info(
        "%s filter: %d total → %d useful / %d borderline / %d noise",
        ticker, len(posts), len(useful), len(borderline), len(noise)
    )
    return result


def _empty_result(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "total": 0, "useful": [], "borderline": [], "noise": [], "verdicts": {}
    }


# ── User-facing verification formatter ────────────────────────────────────────

def format_for_verification(filter_result: dict) -> str:
    """
    Format filter output for user review before final ChromaDB storage.
    Shows what the LLM decided to keep vs discard.
    
    Returns a human-readable string suitable for WeChat report or terminal.
    """
    ticker = filter_result["ticker"]
    total  = filter_result["total"]
    useful = filter_result["useful"]
    border = filter_result["borderline"]
    noise  = filter_result["noise"]

    lines = [
        f"┌─ {ticker} News Filter Result ─────────────────",
        f"│  Total posts reviewed : {total}",
        f"│  ✅ Useful (store)     : {len(useful)}",
        f"│  ⚠️  Borderline (store) : {len(border)}",
        f"│  ❌ Noise (discard)    : {len(noise)}",
        f"├─ USEFUL POSTS ──────────────────────────────",
    ]

    for p in useful:
        src   = p.get("source", "?")
        title = p.get("title", "")[:80]
        facts = ", ".join(p.get("facts", []))
        lines.append(f"│  [{src}] {title}")
        if facts:
            lines.append(f"│    Facts: {facts}")
        lines.append(f"│    Sentiment: {p.get('sentiment')}  Type: {p.get('event_type')}")
        lines.append("│")

    if border:
        lines.append(f"├─ BORDERLINE POSTS ──────────────────────────")
        for p in border:
            lines.append(f"│  [{p.get('source','?')}] {p.get('title','')[:80]}")
            lines.append(f"│    Reason: {p.get('reason','')}")
            lines.append("│")

    lines.append(f"└─ {len(noise)} noise posts discarded ──────────────────")
    return "\n".join(lines)


# ── Batch filter for multiple tickers ─────────────────────────────────────────

def filter_all_tickers(posts_by_ticker: dict[str, list[dict]]) -> dict[str, dict]:
    """
    Run filter for every ticker.
    
    Args:
        posts_by_ticker: {ticker: [post, ...]}
    
    Returns:
        {ticker: filter_result}
    """
    results = {}
    for ticker, posts in posts_by_ticker.items():
        results[ticker] = filter_posts(ticker, posts)
    return results


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    logging.basicConfig(level=logging.INFO)

    # Mock claude if no API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("No API key — using mock filter for test\n")

        def mock_filter(ticker, posts):
            result = _empty_result(ticker)
            result["total"] = len(posts)
            for i, p in enumerate(posts):
                verdict = "useful" if i % 3 != 2 else "noise"
                enriched = {**p, "verdict": verdict, "reason": "mock",
                            "facts": ["mock fact"], "sentiment": "bullish",
                            "event_type": "earnings", "filtered_at": "2026-01-01"}
                result["useful" if verdict == "useful" else "noise"].append(enriched)
            return result

        test_posts = [
            {"id": "r1", "title": "NVDA Q2 EPS $23.41 beats by 60%!", "text": "Data center revenue doubled YoY. Strong guidance.", "source": "reddit", "url": "https://reddit.com/1"},
            {"id": "r2", "title": "NVDA to the moon 🚀🚀🚀", "text": "Trust me bro", "source": "reddit", "url": "https://reddit.com/2"},
            {"id": "x1", "title": "英伟达财报大超预期", "text": "EPS比预期高出60%，数据中心收入创历史新高", "source": "xiaohongshu", "url": "https://xhs.com/1"},
        ]
        result = mock_filter("NVDA", test_posts)
        print(format_for_verification(result))
    else:
        sys.path.insert(0, ".")
        test_posts = [
            {"id": "r1", "title": "NVDA Q2 EPS $23.41 beats by 60%!", "text": "Data center revenue doubled.", "source": "reddit", "url": "https://reddit.com/1"},
            {"id": "r2", "title": "NVDA to the moon 🚀🚀🚀", "text": "Trust me bro", "source": "reddit", "url": "https://reddit.com/2"},
            {"id": "x1", "title": "英伟达财报大超预期", "text": "EPS比预期高出60%", "source": "xiaohongshu", "url": "https://xhs.com/1"},
        ]
        result = filter_posts("NVDA", test_posts)
        print(format_for_verification(result))
