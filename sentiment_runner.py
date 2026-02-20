"""
idle/sentiment_runner.py
Author: Yang
Description: Run batch sentiment analysis on cached headlines and
             log any high-conviction signals to the heartbeat buffer.
"""

import logging

from idle.news_collector import _news_cache
from ai.news_sentiment import analyze_batch_sentiment
from tools.heartbeat import send_idle_report, log_signal

logger = logging.getLogger(__name__)

# Module-level cache shared with idle/memory_updater.py
_sentiment_cache: dict[str, dict] = {}

# Threshold above which we emit a signal to the heartbeat feed
_STRONG_SIGNAL_THRESHOLD = 0.6


def run_sentiment() -> dict[str, dict]:
    """
    Analyse news sentiment for all tickers in _news_cache.
    Updates _sentiment_cache and sends a WeChat report.

    Returns:
        {ticker: {sentiment, score, summary}}
    """
    if not _news_cache:
        logger.info("Idle sentiment: no news in cache — skipping")
        return {}

    logger.info("Idle: running sentiment analysis on %d tickers", len(_news_cache))
    results = analyze_batch_sentiment(_news_cache)

    _sentiment_cache.clear()
    _sentiment_cache.update(results)

    # Emit strong signals to heartbeat
    for ticker, v in results.items():
        if abs(v.get("score", 0)) >= _STRONG_SIGNAL_THRESHOLD:
            log_signal(ticker, "news_alert",
                       f"{v['sentiment']} score={v['score']:+.1f}: {v.get('summary', '')}")

    # WeChat report
    bullish = [t for t, v in results.items() if v.get("sentiment") == "bullish"]
    bearish = [t for t, v in results.items() if v.get("sentiment") == "bearish"]
    neutral = [t for t, v in results.items() if v.get("sentiment") == "neutral"]
    detail  = "\n".join(
        f"> **{t}** `{v.get('score', 0):+.1f}` {v.get('summary', '')}"
        for t, v in results.items()
    )
    send_idle_report(
        "Sentiment Analysis",
        (
            f"> Bullish ({len(bullish)}): {', '.join(bullish) or 'none'}\n"
            f"> Bearish ({len(bearish)}): {', '.join(bearish) or 'none'}\n"
            f"> Neutral ({len(neutral)}): {', '.join(neutral) or 'none'}\n\n"
            + detail
        ),
    )
    return results
