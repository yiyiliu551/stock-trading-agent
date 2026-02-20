"""
idle/memory_updater.py
Author: Yang
Description: Append today's sentiment snapshot to MEMORY.md for
             long-term context and back-testing reference.
"""

import logging

from idle.sentiment_runner import _sentiment_cache
from tools.memory_store import append_sentiment_snapshot
from tools.heartbeat import send_idle_report

logger = logging.getLogger(__name__)


def update_memory() -> bool:
    """
    Write the current _sentiment_cache to MEMORY.md.
    Sends a WeChat confirmation on success.

    Returns:
        True if the file was updated, False otherwise.
    """
    if not _sentiment_cache:
        logger.info("Idle memory: no sentiment data — skipping")
        return False

    ok = append_sentiment_snapshot(_sentiment_cache)

    if ok:
        send_idle_report(
            "Memory Updated",
            f"> Appended sentiment for {len(_sentiment_cache)} tickers to MEMORY.md",
        )
        logger.info("Idle: MEMORY.md updated with %d tickers", len(_sentiment_cache))
    return ok
