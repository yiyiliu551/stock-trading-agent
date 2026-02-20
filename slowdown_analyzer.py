"""
ai/slowdown_analyzer.py
Author: Yang
Description: Claude-powered post-earnings surge slowdown detector.
             Receives recent 5-min price/volume bars and asks Claude whether
             momentum is exhausting.
"""

import logging
from ai.base import call_claude, parse_json_response

logger = logging.getLogger(__name__)

_FALLBACK = {"slowing": False, "confidence": 0, "reasoning": "error"}

_PROMPT = """Is the post-earnings surge in {ticker} SLOWING DOWN and reversing?

Recent 5-min prices (newest last): {prices}
Recent 5-min volumes:              {volumes}
Today's intraday high:             ${today_high:.2f}
Current price:                     ${current_price:.2f}

Look for: flattening momentum, volume exhaustion, pullback from peak.

Answer ONLY in JSON:
{{"slowing": true/false, "confidence": <0-100>, "reasoning": "<2 sentences>"}}"""


def analyze_slowdown(ticker: str, price_data: dict) -> dict:
    """
    Ask Claude whether the surge in *ticker* is running out of steam.

    Args:
        ticker:     Stock symbol.
        price_data: Dict with keys: prices, volumes, today_high, current_price.

    Returns:
        {slowing: bool, confidence: int, reasoning: str}
    """
    prices    = [round(p, 2) for p in price_data.get("prices", [])[-12:]]
    volumes   = [int(v)       for v in price_data.get("volumes", [])[-12:]]
    today_high    = price_data.get("today_high", 0.0)
    current_price = price_data.get("current_price", 0.0)

    prompt = _PROMPT.format(
        ticker=ticker,
        prices=prices,
        volumes=volumes,
        today_high=today_high,
        current_price=current_price,
    )
    try:
        raw    = call_claude(prompt, max_tokens=200)
        result = parse_json_response(raw, _FALLBACK.copy())
        logger.info("Slowdown analysis %s: slowing=%s conf=%d",
                    ticker, result.get("slowing"), result.get("confidence", 0))
        return result
    except Exception as e:
        logger.error("analyze_slowdown(%s) API error: %s", ticker, e)
        return _FALLBACK.copy()
