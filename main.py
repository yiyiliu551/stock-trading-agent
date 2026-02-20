"""
main.py
Author: Yang
Description: 7x24 main loop — integrates the trading pipeline, heartbeat
             scheduler, and idle task scheduler.

Run modes:
    python main.py            — 7x24 continuous loop
    python main.py --once     — single pipeline run (testing)
    python main.py --idle     — run idle tasks once and exit
"""

import asyncio
import logging
import time
import sys
from datetime import datetime

from config import LOG_FILE
from graph.nodes import build_graph
from graph.state import AgentState
from scheduler.market_hours import is_weekend, is_market_hours
from scheduler.idle_scheduler import IdleTaskScheduler
from tools.heartbeat import HeartbeatScheduler, log_error, clear_daily_signals

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("main")


# ── Pipeline runner ────────────────────────────────────────────────────────────

async def run_pipeline() -> dict:
    """Build and invoke the LangGraph pipeline. Returns the final state."""
    app = build_graph()
    initial: AgentState = {
        "earnings_list":      [],
        "ticker":             "",
        "pre_earnings_price": 0.0,
        "earnings_beat":      {},
        "surge_result":       {},
        "market_health":      {},
        "slowdown_result":    {},
        "verify_result":      {},
        "approved":           False,
        "short_result":       {},
        "monitor_result":     {},
        "cover_result":       {},
        "abort_reason":       "",
    }
    final = await app.ainvoke(initial)
    if final.get("abort_reason"):
        logger.info("Pipeline aborted: %s", final["abort_reason"])
    return final


# ── 7x24 loop ─────────────────────────────────────────────────────────────────

def main_loop() -> None:
    """
    Continuous 7x24 loop with three operating states:
        Market hours  → run pipeline every 5 min, heartbeat every 30 min
        Off-hours     → run idle tasks, heartbeat every 2 h
        Weekend       → run idle tasks, heartbeat every 4 h (same as off-hours)
    """
    logger.info("=== Stock Agent starting — 7x24 mode ===")

    hb_sched   = HeartbeatScheduler()
    idle_sched = IdleTaskScheduler()

    current_positions: list = []
    last_reset_day:    str  = ""

    while True:
        try:
            now       = datetime.now()
            today_str = now.strftime("%Y-%m-%d")

            # Reset daily signal buffer at midnight
            if today_str != last_reset_day:
                clear_daily_signals()
                last_reset_day = today_str
                logger.info("Daily reset complete: %s", today_str)

            has_position = bool(current_positions)

            # Heartbeat tick — dispatches only when interval elapsed
            hb_sched.tick(
                positions       = current_positions,
                is_market_hours = is_market_hours(),
                has_position    = has_position,
            )

            if is_market_hours():
                logger.info("Market open — running pipeline")
                try:
                    result = asyncio.run(run_pipeline())
                    if result.get("current_positions"):
                        current_positions = result["current_positions"]
                except Exception as e:
                    log_error(f"Pipeline error: {e}")
                    logger.error("Pipeline error: %s", e, exc_info=True)
                time.sleep(300)    # re-check every 5 min

            else:
                logger.info("Market closed — idle mode")
                idle_sched.tick(has_position=has_position)
                sleep_secs = 3600 * 4 if is_weekend() else 1800
                time.sleep(sleep_secs)

        except KeyboardInterrupt:
            logger.info("Agent stopped by keyboard interrupt")
            break
        except Exception as e:
            log_error(f"Main loop error: {e}")
            logger.error("Main loop error: %s", e, exc_info=True)
            time.sleep(60)


# ── CLI entry points ───────────────────────────────────────────────────────────

def run_once() -> None:
    """Execute one full pipeline run and print the result."""
    logger.info("=== Single pipeline run ===")
    result = asyncio.run(run_pipeline())
    logger.info("Final state: ticker=%s abort=%s",
                result.get("ticker"), result.get("abort_reason"))


def run_idle_once() -> None:
    """Execute all overdue idle tasks once and exit."""
    from idle.news_collector  import collect_all_news
    from idle.sentiment_runner import run_sentiment
    from idle.memory_updater  import update_memory
    from idle.backtester      import run_backtest
    logger.info("=== Running idle tasks once ===")
    collect_all_news()
    run_sentiment()
    update_memory()
    run_backtest()
    logger.info("Idle tasks complete")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--once" in args:
        run_once()
    elif "--idle" in args:
        run_idle_once()
    else:
        main_loop()
