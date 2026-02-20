"""
graph/state.py
Author: Yang
Description: LangGraph state definition shared across all graph nodes.
             Each field corresponds to one pipeline step's output.
"""

from typing import TypedDict


class AgentState(TypedDict):
    # Step 1
    earnings_list:     list   # [{ticker, earnings_date, days_until, pre_earnings_price}]
    ticker:            str
    pre_earnings_price: float

    # Step 2
    earnings_beat:     dict   # {beat, beat_pct, confidence, reason, qualifies}

    # Step 3
    surge_result:      dict   # {surging, surge_pct, current_price, ...}
    market_health:     dict   # {healthy, spy_change, qqq_change}

    # Step 4
    slowdown_result:   dict   # {trigger, current_price, hard_rules, ai_analysis, stop_loss}

    # Step 5
    verify_result:     dict   # {proceed, confirmed, confidence, risk_factors, ...}

    # Step 6
    approved:          bool

    # Step 7
    short_result:      dict   # {success, total_shares_shorted, avg_fill_price, ...}

    # Step 8-9
    monitor_result:    dict   # {action, price, days_held}
    cover_result:      dict   # {success, avg_cover_price, profit_loss, ...}

    # Global abort
    abort_reason:      str
