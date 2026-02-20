"""
ai/react_verifier.py
Author: Yang
Description: Claude ReAct (Reason + Act) self-verification step.
             Claude plays devil's advocate before a trade is approved.
"""

import json
import logging
from ai.base import call_claude, parse_json_response

logger = logging.getLogger(__name__)

_FALLBACK = {"confirmed": False, "confidence": 0,
             "risk_factors": ["verification error"], "final_reasoning": "abort"}

_PROMPT = """You are reviewing a short-sell decision. Apply two rounds of reasoning:

Round 1 (Support): Why is this trade safe and well-timed?
Round 2 (Devil's Advocate): What could go wrong? List specific risks.

Trade data:
{decision_json}

After both rounds, give a final verdict.

Answer ONLY in JSON:
{{
  "confirmed": true/false,
  "confidence": <0-100>,
  "risk_factors": ["<risk1>", "<risk2>"],
  "final_reasoning": "<1-2 sentences>"
}}"""


def verify_trade(ticker: str, decision_data: dict) -> dict:
    """
    Run Claude ReAct verification on the proposed short trade.

    Args:
        ticker:        Stock symbol.
        decision_data: Full slowdown/hard-rules result dict.

    Returns:
        {confirmed: bool, confidence: int, risk_factors: list, final_reasoning: str}
    """
    prompt = _PROMPT.format(decision_json=json.dumps(decision_data, indent=2)[:1500])
    try:
        raw    = call_claude(prompt, max_tokens=400)
        result = parse_json_response(raw, _FALLBACK.copy())
        logger.info("ReAct verify %s: confirmed=%s conf=%d",
                    ticker, result.get("confirmed"), result.get("confidence", 0))
        return result
    except Exception as e:
        logger.error("verify_trade(%s) API error: %s", ticker, e)
        return _FALLBACK.copy()
