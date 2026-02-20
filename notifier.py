"""
tools/notifier.py
Author: Yang
Description: Twilio SMS wrapper — send alerts and poll for user approval replies.
             No AI logic. Returns bool / str.
"""

import time
import logging
from twilio.rest import Client

from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, USER_PHONE, TWILIO_FROM_PHONE

logger = logging.getLogger(__name__)

_APPROVAL_TIMEOUT_SEC = 300   # 5-minute window for YES/NO reply
_POLL_INTERVAL_SEC    = 15


# ── Send ───────────────────────────────────────────────────────────────────────

def send_sms(message: str) -> bool:
    """
    Send an SMS via Twilio.
    Returns True on success, False on any error.
    """
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_FROM_PHONE,
            to=USER_PHONE,
        )
        logger.info("SMS sent: %s", message[:60])
        return True
    except Exception as e:
        logger.error("SMS send failed: %s", e)
        return False


# ── Wait for approval ──────────────────────────────────────────────────────────

def wait_for_approval(sent_at: float) -> bool:
    """
    Poll inbound messages for a YES or NO reply received after *sent_at* timestamp.
    Times out after APPROVAL_TIMEOUT_SEC and returns False (abort).
    """
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    deadline = sent_at + _APPROVAL_TIMEOUT_SEC

    while time.time() < deadline:
        time.sleep(_POLL_INTERVAL_SEC)
        try:
            msgs = client.messages.list(to=TWILIO_FROM_PHONE, limit=5)
            for msg in msgs:
                if (
                    msg.date_sent
                    and msg.date_sent.timestamp() > sent_at
                    and msg.direction == "inbound"
                ):
                    body = msg.body.strip().upper()
                    if body.startswith("YES"):
                        logger.info("User approved trade")
                        return True
                    if body.startswith("NO"):
                        logger.info("User rejected trade")
                        return False
        except Exception as e:
            logger.warning("Polling SMS inbox failed: %s", e)

    logger.warning("Approval timeout — aborting trade")
    send_sms("Timeout: trade automatically aborted (no reply within 5 min)")
    return False
