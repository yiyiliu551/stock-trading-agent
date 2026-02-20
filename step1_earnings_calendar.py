"""
pipeline/step1_earnings_calendar.py
Author: Yang
Description: Scan the watchlist for earnings events within the next 7 days.
             Returns a list of upcoming tickers with their expected dates.
"""

import logging
from datetime import datetime

from config import STOCKS
from tools.market_data import get_earnings_calendar, get_pre_earnings_price

logger = logging.getLogger(__name__)


def get_earnings_within_7_days() -> list[dict]:
    """
    Scan STOCKS and return those with earnings in the next 7 calendar days.

    Returns:
        List of dicts: {ticker, earnings_date, days_until, pre_earnings_price}
    """
    today    = datetime.now().replace(tzinfo=None)
    upcoming = []

    for ticker in STOCKS:
        earnings_date = get_earnings_calendar(ticker)
        if earnings_date is None:
            continue
        days_until = (earnings_date - today).days
        if 0 <= days_until <= 7:
            price = get_pre_earnings_price(ticker)
            upcoming.append({
                "ticker":             ticker,
                "earnings_date":      earnings_date,
                "days_until":         days_until,
                "pre_earnings_price": price,
            })
            logger.info("Earnings in %d days: %s @ $%.2f", days_until, ticker, price)

    logger.info("Step 1 complete: %d upcoming earnings found", len(upcoming))
    return upcoming
