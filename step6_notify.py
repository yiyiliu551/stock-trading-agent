"""
pipeline/step6_notify.py
Author: Yang
Description: Send an SMS trade alert and wait for user approval (YES/NO reply).
             Also dispatches a WeChat notification for real-time visibility.
"""

import time
import logging

from tools.notifier import send_sms, wait_for_approval
from tools.heartbeat import send_trade_notification

logger = logging.getLogger(__name__)


def notify_and_wait_approval(
    ticker: str,
    entry_price: float,
    stop_loss: float,
    confidence: int,
    rules_met: int,
) -> bool:
    """
    1. Send an SMS with trade details.
    2. Optionally post a WeChat message.
    3. Poll for YES/NO reply (5-minute window).

    Returns True if user approved, False if rejected or timed out.
    """
    message = (
        f"TRADE ALERT: Short {ticker}\n"
        f"Entry: ${entry_price:.2f} | Stop: ${stop_loss:.2f}\n"
        f"AI confidence: {confidence}% | Rules met: {rules_met}/3\n"
        f"Reply YES to confirm or NO to abort (5 min)"
    )

    sent = send_sms(message)
    if not sent:
        logger.error("Step 6: SMS failed — aborting trade")
        return False

    sent_at = time.time()

    # WeChat parallel notification
    send_trade_notification({
        "event":       "pending_approval",
        "ticker":      ticker,
        "short_price": entry_price,
        "stop_loss":   stop_loss,
        "shares":      0,
    })

    approved = wait_for_approval(sent_at)
    logger.info("Step 6 %s: approved=%s", ticker, approved)
    return approved
