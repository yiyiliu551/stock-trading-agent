"""
tools/market_data.py
Author: Yang
Description: Market data helpers — all yfinance calls are isolated here.
             No AI logic. Returns plain dicts or primitives.
"""

import logging
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ── Price helpers ──────────────────────────────────────────────────────────────

def get_current_price(ticker: str) -> float:
    """Return the latest 5-minute bar close price. Returns 0.0 on failure."""
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="5m")
        if hist.empty:
            return 0.0
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning("get_current_price(%s) failed: %s", ticker, e)
        return 0.0


def get_pre_earnings_price(ticker: str) -> float:
    """Return the most recent daily close (proxy for pre-earnings baseline)."""
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return 0.0
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning("get_pre_earnings_price(%s) failed: %s", ticker, e)
        return 0.0


def get_recent_intraday_data(ticker: str) -> dict:
    """
    Return intraday 5-minute data for the current session.
    Shape: {prices, volumes, today_high, current_price}
    Returns {} on failure or insufficient bars (<6).
    """
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="5m")
        if hist.empty or len(hist) < 6:
            logger.warning("Insufficient intraday bars for %s (got %d)", ticker, len(hist))
            return {}
        return {
            "prices":        hist["Close"].tolist(),
            "volumes":       hist["Volume"].tolist(),
            "today_high":    float(hist["High"].max()),
            "current_price": float(hist["Close"].iloc[-1]),
        }
    except Exception as e:
        logger.warning("get_recent_intraday_data(%s) failed: %s", ticker, e)
        return {}


# ── Historical data ────────────────────────────────────────────────────────────

def get_daily_closes(ticker: str, period: str = "30d") -> list[float]:
    """Return list of daily close prices for volatility calculations."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        return hist["Close"].tolist() if not hist.empty else []
    except Exception as e:
        logger.warning("get_daily_closes(%s) failed: %s", ticker, e)
        return []


def get_historical_volatility(ticker: str, period: str = "30d") -> float:
    """
    Return annualised daily volatility as a percentage.
    E.g. 2.5 means daily moves average 2.5%.
    Falls back to 2.0 (mid-range) on error.
    """
    closes = get_daily_closes(ticker, period)
    if len(closes) < 5:
        return 2.0
    try:
        pct_changes = np.diff(closes) / closes[:-1] * 100
        return float(np.std(pct_changes))
    except Exception as e:
        logger.warning("get_historical_volatility(%s) failed: %s", ticker, e)
        return 2.0


# ── Earnings calendar ──────────────────────────────────────────────────────────

def get_earnings_calendar(ticker: str) -> Optional[datetime]:
    """
    Return the next earnings date for *ticker*, or None if unavailable.
    Strips timezone info for safe comparison.
    """
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return None
        earnings_date = None
        if hasattr(cal, "columns") and "Earnings Date" in cal.columns:
            earnings_date = cal["Earnings Date"].iloc[0]
        elif isinstance(cal, dict) and "Earnings Date" in cal:
            earnings_date = cal["Earnings Date"]
            if isinstance(earnings_date, list):
                earnings_date = earnings_date[0]
        if earnings_date is None:
            return None
        if hasattr(earnings_date, "tzinfo") and earnings_date.tzinfo:
            earnings_date = earnings_date.replace(tzinfo=None)
        return earnings_date
    except Exception as e:
        logger.warning("get_earnings_calendar(%s) failed: %s", ticker, e)
        return None


# ── Market index health ────────────────────────────────────────────────────────

def get_index_change(ticker: str) -> float:
    """Return the day-over-day % change of an index ETF (SPY, QQQ, etc.)."""
    try:
        hist = yf.Ticker(ticker).history(period="2d")
        if len(hist) < 2:
            return 0.0
        return float(
            (hist["Close"].iloc[-1] - hist["Close"].iloc[-2])
            / hist["Close"].iloc[-2] * 100
        )
    except Exception as e:
        logger.warning("get_index_change(%s) failed: %s", ticker, e)
        return 0.0
