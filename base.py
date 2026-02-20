"""
ai/base.py
Author: Yang
Description: Shared Anthropic client singleton and JSON parse helper.
             All ai/* modules import from here so the client is created once.
"""

import re
import json
import logging
import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS

logger = logging.getLogger(__name__)

# Singleton client shared across all ai/* modules
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def call_claude(prompt: str, max_tokens: int = CLAUDE_MAX_TOKENS) -> str:
    """
    Send a single-turn message to Claude and return the raw text response.
    Raises on API error — callers should handle exceptions.
    """
    response = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def parse_json_response(text: str, fallback: dict) -> dict:
    """
    Strip optional ```json fences and parse JSON from Claude's response.
    Returns *fallback* dict on any parse error.
    """
    try:
        clean = re.sub(r"```json|```", "", text).strip()
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("JSON parse failed: %s | raw=%s", e, text[:120])
        return fallback
