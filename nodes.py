"""
graph/nodes.py
Author: Yang
Description: LangGraph node functions and graph construction.
             Each node wraps one pipeline step and updates AgentState.
"""

import logging
from langgraph.graph import StateGraph, END

from graph.state import AgentState
from pipeline.step1_earnings_calendar import get_earnings_within_7_days
from pipeline.step2_earnings_result   import check_earnings_beat
from pipeline.step3_surge_detect      import run_step3
from pipeline.step4_slowdown_detect   import detect_slowdown
from pipeline.step5_react_verify      import run_step5
from pipeline.step6_notify            import notify_and_wait_approval
from pipeline.step7_short_sell        import execute_short
from pipeline.step8_monitor           import monitor_position
from pipeline.step9_cover             import execute_cover
from pipeline.step10_memory           import record_trade

logger = logging.getLogger(__name__)


# ── Node functions ─────────────────────────────────────────────────────────────

def node_step1(state: AgentState) -> AgentState:
    earnings = get_earnings_within_7_days()
    state["earnings_list"] = earnings
    if earnings:
        state["ticker"]             = earnings[0]["ticker"]
        state["pre_earnings_price"] = earnings[0]["pre_earnings_price"]
    else:
        state["abort_reason"] = "Step 1: no upcoming earnings"
    return state


def node_step2(state: AgentState) -> AgentState:
    result = check_earnings_beat(state["ticker"])
    state["earnings_beat"] = result
    if not result.get("qualifies"):
        state["abort_reason"] = (
            f"Step 2: earnings did not qualify "
            f"(beat={result.get('beat')} pct={result.get('beat_pct',0):.1f}%)"
        )
    return state


def node_step3(state: AgentState) -> AgentState:
    result = run_step3(state["ticker"], state["pre_earnings_price"])
    state["surge_result"]  = result.get("surge", {})
    state["market_health"] = result.get("market", {})
    if not result.get("proceed"):
        state["abort_reason"] = f"Step 3: {result.get('abort_reason')}"
    return state


def node_step4(state: AgentState) -> AgentState:
    result = detect_slowdown(state["ticker"], state["pre_earnings_price"])
    state["slowdown_result"] = result
    if not result.get("trigger"):
        state["abort_reason"] = f"Step 4: {result.get('abort_reason')}"
    return state


def node_step5(state: AgentState) -> AgentState:
    result = run_step5(state["ticker"], state["slowdown_result"])
    state["verify_result"] = result
    if not result.get("proceed"):
        state["abort_reason"] = "Step 5: ReAct verification failed"
    return state


def node_step6(state: AgentState) -> AgentState:
    slowdown = state["slowdown_result"]
    verify   = state["verify_result"]
    approved = notify_and_wait_approval(
        ticker      = state["ticker"],
        entry_price = slowdown.get("current_price", 0),
        stop_loss   = slowdown.get("stop_loss", 0),
        confidence  = verify.get("confidence", 0),
        rules_met   = slowdown.get("hard_rules", {}).get("rules_met", 0),
    )
    state["approved"] = approved
    if not approved:
        state["abort_reason"] = "Step 6: user rejected or timed out"
    return state


def node_step7(state: AgentState) -> AgentState:
    slowdown = state["slowdown_result"]
    result   = execute_short(
        ticker      = state["ticker"],
        entry_price = slowdown.get("current_price", 0),
        stop_loss   = slowdown.get("stop_loss", 0),
    )
    state["short_result"] = result
    if not result.get("success"):
        state["abort_reason"] = f"Step 7: short execution failed — {result.get('reason')}"
    return state


def node_step8_9(state: AgentState) -> AgentState:
    short    = state["short_result"]
    slowdown = state["slowdown_result"]

    monitor = monitor_position(
        ticker      = state["ticker"],
        short_price = short.get("avg_fill_price", 0),
        stop_loss   = slowdown.get("stop_loss", 0),
    )
    state["monitor_result"] = monitor

    cover = execute_cover(
        ticker         = state["ticker"],
        short_price    = short.get("avg_fill_price", 0),
        total_shares   = short.get("total_shares_shorted", 0),
        monitor_result = monitor,
    )
    state["cover_result"] = cover
    return state


def node_step10(state: AgentState) -> AgentState:
    short = state["short_result"]
    cover = state["cover_result"]
    record_trade({
        "ticker":       state["ticker"],
        "short_price":  short.get("avg_fill_price", 0),
        "cover_price":  cover.get("avg_cover_price", 0),
        "total_shares": short.get("total_shares_shorted", 0),
        "profit_loss":  cover.get("profit_loss", 0),
        "days_held":    cover.get("days_held", 0),
        "outcome":      "profit" if cover.get("profit_loss", 0) >= 0 else "loss",
    })
    return state


# ── Graph routing helpers ──────────────────────────────────────────────────────

def _should_continue(state: AgentState) -> str:
    """Route to END if an abort_reason has been set, otherwise continue."""
    return END if state.get("abort_reason") else "continue"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph():
    """
    Construct the 10-step LangGraph pipeline.
    Each step either advances to the next node or routes to END on abort.
    """
    g = StateGraph(AgentState)

    g.add_node("step1", node_step1)
    g.add_node("step2", node_step2)
    g.add_node("step3", node_step3)
    g.add_node("step4", node_step4)
    g.add_node("step5", node_step5)
    g.add_node("step6", node_step6)
    g.add_node("step7", node_step7)
    g.add_node("step8_9", node_step8_9)
    g.add_node("step10", node_step10)

    g.set_entry_point("step1")

    node_sequence = ["step1", "step2", "step3", "step4",
                     "step5", "step6", "step7", "step8_9", "step10"]

    for i, node in enumerate(node_sequence[:-1]):
        next_node = node_sequence[i + 1]
        g.add_conditional_edges(
            node,
            _should_continue,
            {"continue": next_node, END: END},
        )

    g.add_edge("step10", END)
    return g.compile()
