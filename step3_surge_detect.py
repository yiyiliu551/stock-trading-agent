"""
pipeline/step3_surge_detect.py
Author: Yang
Description: Detect a significant intraday surge above the pre-earnings baseline
             AND verify that broad market conditions are healthy.
"""

import logging

from config import SURGE_THRESHOLD
from tools.market_data import get_current_price, get_index_change

logger = logging.getLogger(__name__)


def check_market_health() -> dict:
    """
    Return True if SPY and QQQ are both down less than 2% on the day.
    A weak broad market increases short-squeeze risk.

    Returns:
        {healthy: bool, spy_change: float, qqq_change: float}
    """
    spy = get_index_change("SPY")
    qqq = get_index_change("QQQ")
    healthy = spy > -2.0 and qqq > -2.0
    logger.info("Market health: SPY=%.2f%% QQQ=%.2f%% healthy=%s", spy, qqq, healthy)
    return {"healthy": healthy, "spy_change": round(spy, 2), "qqq_change": round(qqq, 2)}


def detect_surge(ticker: str, pre_earnings_price: float) -> dict:
    """
    Return True if the stock has surged >= SURGE_THRESHOLD% above pre-earnings close.

    Args:
        ticker:             Stock symbol.
        pre_earnings_price: Closing price before the earnings release.

    Returns:
        {surging, surge_pct, current_price, pre_earnings_price}
    """
    if pre_earnings_price <= 0:
        return {"surging": False, "surge_pct": 0.0,
                "current_price": 0.0, "pre_earnings_price": pre_earnings_price}

    current = get_current_price(ticker)
    if current <= 0:
        return {"surging": False, "surge_pct": 0.0,
                "current_price": 0.0, "pre_earnings_price": pre_earnings_price}

    surge_pct = (current - pre_earnings_price) / pre_earnings_price * 100
    surging   = surge_pct >= SURGE_THRESHOLD

    logger.info("Surge check %s: %.2f%% surging=%s", ticker, surge_pct, surging)
    return {
        "surging":            surging,
        "surge_pct":          round(surge_pct, 2),
        "current_price":      current,
        "pre_earnings_price": pre_earnings_price,
    }


def run_step3(ticker: str, pre_earnings_price: float) -> dict:
    """
    Full step 3: market health check + surge detection.

    Returns:
        {proceed: bool, surge: dict, market: dict, abort_reason: str}
    """
    market = check_market_health()
    if not market["healthy"]:
        return {"proceed": False, "surge": {}, "market": market,
                "abort_reason": f"Market unhealthy (SPY {market['spy_change']}%)"}

    surge = detect_surge(ticker, pre_earnings_price)
    if not surge["surging"]:
        return {"proceed": False, "surge": surge, "market": market,
                "abort_reason": f"No surge detected ({surge['surge_pct']:.2f}%)"}

    return {"proceed": True, "surge": surge, "market": market, "abort_reason": ""}
