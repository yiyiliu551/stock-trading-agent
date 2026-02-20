"""
pipeline/step5_react_verify.py
Author: Yang
Description: ReAct self-verification — Claude plays devil's advocate before
             the trade proceeds to the notification step.
"""

import logging
from ai.react_verifier import verify_trade
from config import AI_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


def run_step5(ticker: str, slowdown_result: dict) -> dict:
    """
    Ask Claude to verify the trade using ReAct reasoning.

    Args:
        ticker:          Stock symbol.
        slowdown_result: Full output from step4 detect_slowdown().

    Returns:
        {proceed, verified, confidence, risk_factors, final_reasoning}
    """
    verified = verify_trade(ticker, slowdown_result)

    proceed = (
        verified.get("confirmed", False)
        and verified.get("confidence", 0) >= AI_CONFIDENCE_THRESHOLD
    )

    logger.info("Step 5 %s: proceed=%s confidence=%d",
                ticker, proceed, verified.get("confidence", 0))

    return {
        "proceed":          proceed,
        "confirmed":        verified.get("confirmed", False),
        "confidence":       verified.get("confidence", 0),
        "risk_factors":     verified.get("risk_factors", []),
        "final_reasoning":  verified.get("final_reasoning", ""),
    }
