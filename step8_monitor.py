"""
pipeline/step8_monitor.py
Author: Yang
Description: Monitor an open short position every 5 minutes.
             Exits when: take-profit target hit, stop-loss breached, or timeout.

⚡ CUSTOMISE:
    _take_profit_target() — replace fixed 3% with dynamic trailing stop,
                            volatility-adjusted target, or time-decay scaling.
"""

import time
import logging

from config import MAX_DAYS_WAIT_COVER
from tools.market_data import get_current_price
from tools.notifier import send_sms

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SEC = 300   # 5 minutes


# ── ⚡ Profit target (replace with dynamic logic) ──────────────────────────────

def _take_profit_target(short_price: float) -> float:
    """
    ⚡ CUSTOMISE — Calculate the take-profit cover price.

    Current: fixed 3% below the short-entry price.
    Suggested upgrade:
      - Trailing stop that moves down with price
      - Volatility-adjusted (ATR-based) target
      - Time-weighted: tighter target as days_held increases
    """
    return short_price * 0.97


# ── Monitor loop ───────────────────────────────────────────────────────────────

def monitor_position(ticker: str, short_price: float, stop_loss: float) -> dict:
    """
    Poll the position until an exit condition is met.

    Exit conditions (in priority order):
        1. current_price >= stop_loss  → stop_loss exit
        2. current_price <= take_profit_target → take_profit exit
        3. Elapsed time >= MAX_DAYS_WAIT_COVER → timeout exit

    Returns:
        {action: "take_profit"|"stop_loss"|"timeout", price, days_held}
    """
    target    = _take_profit_target(short_price)
    start     = time.time()
    max_secs  = MAX_DAYS_WAIT_COVER * 86_400

    logger.info("Step 8 monitoring %s | short=$%.2f stop=$%.2f target=$%.2f",
                ticker, short_price, stop_loss, target)

    while time.time() - start < max_secs:
        current     = get_current_price(ticker)
        elapsed_days = (time.time() - start) / 86_400

        if current <= 0:
            logger.warning("Price fetch returned 0 for %s — retrying", ticker)
            time.sleep(_POLL_INTERVAL_SEC)
            continue

        if current >= stop_loss:
            msg = f"STOP LOSS: {ticker} @ ${current:.2f} (stop=${stop_loss:.2f})"
            send_sms(msg)
            logger.warning(msg)
            return {"action": "stop_loss", "price": current, "days_held": elapsed_days}

        if current <= target:
            msg = f"TAKE PROFIT: {ticker} @ ${current:.2f} (target=${target:.2f})"
            send_sms(msg)
            logger.info(msg)
            return {"action": "take_profit", "price": current, "days_held": elapsed_days}

        logger.info("Monitoring %s: $%.2f | day %.1f", ticker, current, elapsed_days)
        time.sleep(_POLL_INTERVAL_SEC)

    # Timeout — force cover
    current = get_current_price(ticker)
    send_sms(f"TIMEOUT: {ticker} covering now @ ${current:.2f}")
    return {"action": "timeout", "price": current, "days_held": float(MAX_DAYS_WAIT_COVER)}
