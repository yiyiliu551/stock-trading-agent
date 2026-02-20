"""
scheduler/idle_scheduler.py
Author: Yang
Description: Rate-limited scheduler for background idle tasks.
             Runs tasks only when there is no active position (to avoid
             competing with the hot path).

Task schedule:
    news       — every 1 h
    sentiment  — every 1 h  (depends on news cache)
    memory     — every 2 h  (depends on sentiment cache)
    backtest   — once per day, between 00:00 and 02:00 local time
"""

import time
import logging
from datetime import datetime

from tools.heartbeat import log_error

logger = logging.getLogger(__name__)

# ── Minimum intervals (seconds) ───────────────────────────────────────────────
TASK_INTERVALS: dict[str, int] = {
    "news":      1 * 3600,
    "sentiment": 1 * 3600,
    "memory":    2 * 3600,
    "backtest":  24 * 3600,
}


class IdleTaskScheduler:
    """
    Call scheduler.tick(has_position=...) on every main-loop iteration.
    Tasks execute only when they are due and no position is open.
    """

    def __init__(self) -> None:
        self._last_run: dict[str, float] = {k: 0.0 for k in TASK_INTERVALS}
        self._is_running: bool = False

    def _due(self, task: str) -> bool:
        return time.time() - self._last_run[task] >= TASK_INTERVALS[task]

    def _mark(self, task: str) -> None:
        self._last_run[task] = time.time()

    def tick(self, has_position: bool = False) -> list[str]:
        """
        Check and run overdue idle tasks.

        Args:
            has_position: Skip all tasks when a position is actively monitored.

        Returns:
            List of task names that were executed this tick.
        """
        if has_position or self._is_running:
            return []

        # Lazy import to avoid circular dependencies at module load time
        from idle.news_collector    import collect_all_news,   _news_cache
        from idle.sentiment_runner  import run_sentiment,      _sentiment_cache
        from idle.memory_updater    import update_memory
        from idle.backtester        import run_backtest

        self._is_running = True
        ran: list[str] = []

        try:
            if self._due("news"):
                collect_all_news()
                self._mark("news")
                ran.append("news")

            if self._due("sentiment") and _news_cache:
                run_sentiment()
                self._mark("sentiment")
                ran.append("sentiment")

            if self._due("memory") and _sentiment_cache:
                update_memory()
                self._mark("memory")
                ran.append("memory")

            if self._due("backtest"):
                if 0 <= datetime.now().hour < 2:
                    run_backtest()
                    self._mark("backtest")
                    ran.append("backtest")

        except Exception as e:
            log_error(f"IdleTaskScheduler error: {e}")
            logger.error("IdleTaskScheduler error: %s", e, exc_info=True)
        finally:
            self._is_running = False

        if ran:
            logger.info("Idle tasks ran: %s", ran)
        return ran
