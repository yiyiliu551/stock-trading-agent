"""
ai/trade_reflector.py
Author: Yang
Description: Post-trade reflection — Claude extracts 3 reusable lessons
             from a completed trade record.
"""

import json
import logging
from ai.base import call_claude

logger = logging.getLogger(__name__)

_PROMPT = """Review the following completed short trade and extract exactly 3 lessons
that could improve future trades. Be specific and actionable.

Trade record:
{trade_json}

Format:
Lesson 1: ...
Lesson 2: ...
Lesson 3: ..."""


def generate_reflection(trade: dict) -> str:
    """
    Ask Claude to produce 3 trade lessons.

    Args:
        trade: Completed trade dict with keys:
               ticker, short_price, cover_price, profit_loss, days_held, outcome.

    Returns:
        Plain-text reflection string. Falls back to a generic message on error.
    """
    prompt = _PROMPT.format(trade_json=json.dumps(trade, indent=2))
    try:
        reflection = call_claude(prompt, max_tokens=300)
        logger.info("Trade reflection generated for %s", trade.get("ticker"))
        return reflection.strip()
    except Exception as e:
        logger.error("generate_reflection(%s) API error: %s", trade.get("ticker"), e)
        return "Lesson 1: Log error — reflection unavailable."
