"""
ai/news_sentiment.py
Author: Yang
Description: Claude-powered batch sentiment analysis for watchlist news.
             One API call processes all tickers to minimise token spend.
"""

import logging
from ai.base import call_claude, parse_json_response

logger = logging.getLogger(__name__)

_PROMPT = """Analyse the following stock news headlines and rate the sentiment for each ticker.

{news_block}

For each ticker respond with:
  sentiment: "bullish" | "bearish" | "neutral"
  score:     float from -1.0 (very bearish) to +1.0 (very bullish)
  summary:   one sentence, max 15 words, English only

Answer ONLY in JSON, no prose, no markdown:
{{
  "TICKER": {{"sentiment": "...", "score": 0.0, "summary": "..."}},
  ...
}}"""


def analyze_batch_sentiment(news_map: dict[str, list[str]]) -> dict:
    """
    Analyse news sentiment for multiple tickers in a single Claude call.

    Args:
        news_map: {ticker: [headline, headline, ...]}

    Returns:
        {ticker: {sentiment, score, summary}}
        Empty dict if news_map is empty or all values are empty lists.
    """
    # Filter to tickers that actually have headlines
    filtered = {t: hl for t, hl in news_map.items() if hl}
    if not filtered:
        logger.info("No news to analyse — skipping sentiment call")
        return {}

    # Build readable news block
    lines = []
    for ticker, headlines in filtered.items():
        lines.append(f"[{ticker}]")
        for h in headlines[:5]:
            lines.append(f"  - {h[:200]}")
    news_block = "\n".join(lines)

    fallback = {}
    try:
        raw    = call_claude(_PROMPT.format(news_block=news_block), max_tokens=1000)
        result = parse_json_response(raw, fallback)
        logger.info("Sentiment analysis complete: %d tickers", len(result))
        return result
    except Exception as e:
        logger.error("analyze_batch_sentiment API error: %s", e)
        return fallback
