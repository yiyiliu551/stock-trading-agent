"""
ai/earnings_analyzer.py
Author: Yang
Description: Claude-powered earnings beat/miss classifier.
             Input: raw search text about a stock's quarterly earnings.
             Output: structured JSON with beat flag, percentage, and confidence.
"""

import logging
from ai.base import call_claude, parse_json_response

logger = logging.getLogger(__name__)

_FALLBACK = {"beat": False, "beat_pct": 0.0, "confidence": 0, "reason": "parse error"}

_PROMPT = """Did {ticker} beat Wall Street EPS expectations this quarter?

Source text:
{search_text}

Answer ONLY in JSON (no prose, no markdown fences):
{{"beat": true/false, "beat_pct": <float>, "confidence": <0-100>, "reason": "<1 sentence>"}}

beat_pct = how much above consensus in %, e.g. 15.0 means 15% beat.
If the beat is unclear, set beat=false and confidence low."""


def analyze_earnings_beat(ticker: str, search_text: str) -> dict:
    """
    Ask Claude whether *ticker* beat earnings expectations.

    Args:
        ticker:      Stock symbol, e.g. "NVDA"
        search_text: Raw text from news search about the earnings release.

    Returns:
        {beat: bool, beat_pct: float, confidence: int, reason: str}
    """
    prompt = _PROMPT.format(ticker=ticker, search_text=search_text[:1500])
    try:
        raw = call_claude(prompt, max_tokens=200)
        result = parse_json_response(raw, _FALLBACK.copy())
        logger.info("Earnings analysis %s: beat=%s pct=%.1f conf=%d",
                    ticker, result.get("beat"), result.get("beat_pct", 0),
                    result.get("confidence", 0))
        return result
    except Exception as e:
        logger.error("analyze_earnings_beat(%s) API error: %s", ticker, e)
        return _FALLBACK.copy()
