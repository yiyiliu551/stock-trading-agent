"""
pipeline/step7_short_sell.py
Author: Yang
Description: Execute the short entry in three batches via IBKR.
             After fill, dispatches WeChat open-position notification.
"""

import logging

from tools.broker import short_in_batches
from tools.heartbeat import send_trade_notification

logger = logging.getLogger(__name__)


def execute_short(ticker: str, entry_price: float, stop_loss: float) -> dict:
    """
    Place a short position in 3 batches (30% / 30% / 40%).

    Args:
        ticker:      Stock symbol.
        entry_price: Current market price for sizing.
        stop_loss:   Stop-loss level for the WeChat notification.

    Returns:
        Broker result dict from short_in_batches():
        {success, ticker, total_shares_shorted, avg_fill_price, batches}
    """
    logger.info("Step 7: executing short %s @ $%.2f", ticker, entry_price)
    result = short_in_batches(ticker, entry_price)

    if result.get("success"):
        send_trade_notification({
            "event":       "opened",
            "ticker":      ticker,
            "short_price": result.get("avg_fill_price", entry_price),
            "shares":      result.get("total_shares_shorted", 0),
            "stop_loss":   stop_loss,
        })
        logger.info("Step 7 %s: shorted %d shares @ $%.2f",
                    ticker,
                    result.get("total_shares_shorted", 0),
                    result.get("avg_fill_price", 0))
    else:
        logger.error("Step 7 %s: short failed — %s", ticker, result.get("reason"))

    return result
