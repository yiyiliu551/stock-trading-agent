"""
pipeline/step4_slowdown_detect.py
Author: Yang
Description: Detect that the post-earnings surge is exhausting.
             Combines three hard rules with AI analysis.

⚡ CUSTOMISE:
    find_surge_peak()    — replace simple max() with momentum-reversal detection
    calculate_stop_loss() — add VIX / implied-vol inputs
    check_hard_rules()   — tune thresholds after back-testing
"""

import logging
import numpy as np

from config import (
    SLOWDOWN_PRICE_CHANGE, VOLUME_DROP_THRESHOLD, PULLBACK_FROM_HIGH,
    AI_CONFIDENCE_THRESHOLD, PRICE_GUARD_MIN_GAIN,
    STOP_LOSS_HIGH_VOL, STOP_LOSS_MED_VOL, STOP_LOSS_LOW_VOL,
)
from tools.market_data import get_recent_intraday_data, get_historical_volatility
from ai.slowdown_analyzer import analyze_slowdown

logger = logging.getLogger(__name__)


# ── ⚡ Core strategy: find the surge peak ──────────────────────────────────────

def find_surge_peak(price_data: dict) -> float:
    """
    ⚡ CUSTOMISE — Identify the intraday surge high.

    Current implementation: simple max of the last 6 bars (30 min).
    Recommended upgrade: rolling-window momentum-reversal detection,
    optionally weighted by volume (draws on time-series work from Weibo/Qunar).

    Args:
        price_data: Dict with 'prices' key (list of 5-min closes).

    Returns:
        Peak price as float. 0.0 if insufficient data.
    """
    prices = price_data.get("prices", [])
    if not prices:
        return 0.0
    window = prices[-6:] if len(prices) >= 6 else prices
    return float(max(window))


# ── Hard rules (⚡ tune thresholds) ────────────────────────────────────────────

def check_hard_rules(price_data: dict) -> dict:
    """
    Evaluate three quantitative slowdown criteria.

    Rule 1 — Momentum slow:  |last 5-min change| < SLOWDOWN_PRICE_CHANGE (0.3%)
    Rule 2 — Volume drop:    current bar volume < prior 6-bar average * (1 - VOLUME_DROP_THRESHOLD)
    Rule 3 — Pullback:       current price < surge_peak * (1 - PULLBACK_FROM_HIGH/100)

    Returns:
        {passed, rules_met, rule1_momentum_slow, rule2_volume_drop,
         rule3_pullback, surge_peak_used}
    """
    prices       = price_data.get("prices", [])
    volumes      = price_data.get("volumes", [])
    current      = price_data.get("current_price", 0.0)
    surge_peak   = find_surge_peak(price_data)

    rule1 = rule2 = rule3 = False

    # Rule 1: price momentum below threshold
    if len(prices) >= 2 and prices[-2] != 0:
        last_move = abs((prices[-1] - prices[-2]) / prices[-2] * 100)
        rule1     = last_move < SLOWDOWN_PRICE_CHANGE

    # Rule 2: volume contraction vs prior 6-bar average
    if len(volumes) >= 7:
        prior_avg = float(np.mean(volumes[-7:-1]))
        if prior_avg > 0:
            drop  = (prior_avg - volumes[-1]) / prior_avg
            rule2 = drop >= VOLUME_DROP_THRESHOLD

    # Rule 3: price pulled back >= PULLBACK_FROM_HIGH% from surge peak
    if surge_peak > 0 and current > 0:
        pullback = (surge_peak - current) / surge_peak * 100
        rule3    = pullback >= PULLBACK_FROM_HIGH

    rules_met = int(rule1) + int(rule2) + int(rule3)
    return {
        "passed":              rules_met >= 2,
        "rules_met":           rules_met,
        "rule1_momentum_slow": rule1,
        "rule2_volume_drop":   rule2,
        "rule3_pullback":      rule3,
        "surge_peak_used":     surge_peak,
    }


# ── ⚡ Stop loss (tune with VIX / IV) ─────────────────────────────────────────

def calculate_stop_loss(ticker: str, short_price: float) -> float:
    """
    ⚡ CUSTOMISE — Volatility-based stop loss.

    Tiers based on 30-day historical daily volatility:
        > 3%  → 8% hard stop
        2-3%  → 6% hard stop
        < 2%  → 5% hard stop

    Recommended upgrade: incorporate VIX regime and options implied volatility.

    Returns:
        Stop-loss price (above the short entry).
        Falls back to 6% stop on data errors.
    """
    try:
        vol = get_historical_volatility(ticker, period="30d")
        if vol > 3.0:
            pct = STOP_LOSS_HIGH_VOL
        elif vol > 2.0:
            pct = STOP_LOSS_MED_VOL
        else:
            pct = STOP_LOSS_LOW_VOL
        return round(short_price * (1 + pct), 2)
    except Exception as e:
        logger.warning("calculate_stop_loss(%s) error: %s — using 6%% default", ticker, e)
        return round(short_price * (1 + STOP_LOSS_MED_VOL), 2)


# ── Pipeline entry point ───────────────────────────────────────────────────────

def detect_slowdown(ticker: str, pre_earnings_price: float) -> dict:
    """
    Full step 4: load intraday data → price guard → hard rules → AI analysis.

    Returns:
        {trigger, current_price, hard_rules, ai_analysis, stop_loss,
         abort_reason}
    """
    price_data = get_recent_intraday_data(ticker)
    if not price_data:
        return {"trigger": False, "abort_reason": "No intraday data"}

    current = price_data.get("current_price", 0.0)

    # Price guard: minimum absolute gain required to justify the trade
    gain = current - pre_earnings_price
    if gain < PRICE_GUARD_MIN_GAIN:
        return {
            "trigger":      False,
            "current_price": current,
            "abort_reason": f"Price guard: only ${gain:.2f} gain (need ${PRICE_GUARD_MIN_GAIN})",
        }

    hard_rules = check_hard_rules(price_data)
    ai_result  = analyze_slowdown(ticker, price_data)
    ai_ok      = ai_result.get("confidence", 0) >= AI_CONFIDENCE_THRESHOLD
    trigger    = hard_rules["passed"] and ai_ok
    stop_loss  = calculate_stop_loss(ticker, current) if trigger else 0.0

    logger.info("Step 4 %s: trigger=%s rules=%d/3 ai_conf=%d",
                ticker, trigger, hard_rules["rules_met"], ai_result.get("confidence", 0))

    return {
        "trigger":       trigger,
        "current_price": current,
        "hard_rules":    hard_rules,
        "ai_analysis":   ai_result,
        "stop_loss":     stop_loss,
        "abort_reason":  "" if trigger else "Slowdown conditions not met",
    }
