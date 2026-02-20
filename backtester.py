"""
idle/backtester.py
Author: Yang
Description: Simple overnight back-test — compares yesterday's signal entries
             against today's close to measure directional accuracy.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional

from tools.market_data import get_daily_closes
from tools.heartbeat import send_idle_report, log_error

logger = logging.getLogger(__name__)


# ── Load yesterday's signals from MEMORY.md ───────────────────────────────────

def _load_yesterdays_signals(memory_file: str = "MEMORY.md") -> list[dict]:
    """
    Parse MEMORY.md for trade entries logged yesterday.
    Expected line format:  "## TICKER | YYYY-MM-DDTHH:MM:SS"
    followed by:            "- Short: $NNN.NN | ..."
    Returns list of {ticker, short_price}.
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    signals   = []
    try:
        with open(memory_file, "r", encoding="utf-8") as f:
            content = f.read()
        pattern = rf"## (\w+) \| ({yesterday}[^\n]*)\n- Short: \$([0-9.]+)"
        for m in re.finditer(pattern, content):
            signals.append({
                "ticker":      m.group(1),
                "short_price": float(m.group(3)),
            })
    except FileNotFoundError:
        logger.info("MEMORY.md not found — no signals to back-test")
    except Exception as e:
        logger.warning("load_yesterdays_signals failed: %s", e)
    return signals


# ── Back-test logic ────────────────────────────────────────────────────────────

def run_backtest(memory_file: str = "MEMORY.md") -> dict:
    """
    For each signal from yesterday, fetch the latest close and determine
    whether the short direction was correct (price fell).

    Sends a WeChat report with accuracy metrics.

    Returns:
        {accuracy, correct, total, details}
    """
    signals = _load_yesterdays_signals(memory_file)

    if not signals:
        logger.info("Idle backtest: no yesterday signals found")
        send_idle_report("Backtest", "> No signals from yesterday to evaluate")
        return {"accuracy": None, "correct": 0, "total": 0, "details": []}

    details = []
    for sig in signals:
        ticker      = sig["ticker"]
        short_price = sig["short_price"]
        closes      = get_daily_closes(ticker, period="2d")
        if not closes:
            continue
        close_price = closes[-1]
        correct     = close_price < short_price
        pnl_pct     = (short_price - close_price) / short_price * 100
        details.append({
            "ticker":      ticker,
            "short_price": short_price,
            "close":       close_price,
            "correct":     correct,
            "pnl_pct":     round(pnl_pct, 2),
        })

    if not details:
        return {"accuracy": None, "correct": 0, "total": 0, "details": []}

    correct_count = sum(1 for d in details if d["correct"])
    accuracy      = correct_count / len(details) * 100

    # WeChat report
    lines = [f"> **Accuracy: {accuracy:.0f}%** ({correct_count}/{len(details)})"]
    for d in details:
        icon = "pass" if d["correct"] else "fail"
        s    = f"+{d['pnl_pct']:.1f}%" if d["pnl_pct"] >= 0 else f"{d['pnl_pct']:.1f}%"
        lines.append(
            f"> **{d['ticker']}** short=${d['short_price']:.2f} "
            f"close=${d['close']:.2f} {s}"
        )
    send_idle_report("Backtest Results", "\n".join(lines))

    logger.info("Idle backtest: accuracy=%.0f%% (%d/%d)", accuracy, correct_count, len(details))
    return {"accuracy": accuracy, "correct": correct_count,
            "total": len(details), "details": details}
