"""
pipeline/step9_cover.py
Author: Yang
Description: Cover the short position in three batches and notify via WeChat.
"""

import logging

from tools.broker import cover_in_batches
from tools.heartbeat import send_trade_notification

logger = logging.getLogger(__name__)


def execute_cover(
    ticker: str,
    short_price: float,
    total_shares: int,
    monitor_result: dict,
) -> dict:
    """
    Cover the short position and dispatch a WeChat completion notification.

    Args:
        ticker:         Stock symbol.
        short_price:    Original average fill price when the short was opened.
        total_shares:   Total shares to cover (from step 7 result).
        monitor_result: Output from step 8 monitor_position():
                        {action, price, days_held}

    Returns:
        Cover result dict: {success, avg_cover_price, batches, profit_loss, days_held}
    """
    reason = monitor_result.get("action", "unknown")
    logger.info("Step 9: covering %s — reason=%s", ticker, reason)

    result = cover_in_batches(ticker, total_shares, reason)

    cover_price = result.get("avg_cover_price", 0.0)
    profit_loss = round((short_price - cover_price) * total_shares, 2)

    result["profit_loss"] = profit_loss
    result["days_held"]   = monitor_result.get("days_held", 0.0)

    if result.get("success"):
        send_trade_notification({
            "event":       "covered",
            "ticker":      ticker,
            "short_price": short_price,
            "cover_price": cover_price,
            "shares":      total_shares,
            "profit_loss": profit_loss,
            "days_held":   result["days_held"],
        })
        logger.info("Step 9 %s: covered @ $%.2f P&L=$%.2f",
                    ticker, cover_price, profit_loss)
    else:
        logger.error("Step 9 %s: cover failed", ticker)

    return result
