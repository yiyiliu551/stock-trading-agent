"""
pipeline/step10_memory.py
Author: Yang
Description: Persist a completed trade to ChromaDB + MEMORY.md with an
             AI-generated reflection for continuous learning.
"""

import logging
from datetime import datetime

from tools.memory_store import save_trade_to_chroma, append_trade_to_markdown
from ai.trade_reflector import generate_reflection

logger = logging.getLogger(__name__)


def record_trade(trade: dict) -> str:
    """
    Save trade to ChromaDB and append a human-readable entry + AI reflection
    to MEMORY.md.

    Args:
        trade: Completed trade dict, expected keys:
               ticker, short_price, cover_price, total_shares,
               profit_loss, days_held, outcome

    Returns:
        AI reflection text (3 lessons). Empty string on failure.
    """
    trade.setdefault("timestamp", datetime.now().isoformat())

    # Derive outcome label if missing
    if "outcome" not in trade:
        trade["outcome"] = "profit" if trade.get("profit_loss", 0) >= 0 else "loss"

    # Generate AI reflection
    reflection = generate_reflection(trade)

    # Persist to both storage backends
    save_trade_to_chroma(trade)
    append_trade_to_markdown(trade, reflection)

    logger.info("Step 10 %s: trade recorded | P&L=$%.2f | outcome=%s",
                trade.get("ticker"), trade.get("profit_loss", 0), trade.get("outcome"))

    return reflection
