"""
config.py
Author: Yang
Project: Post-Earnings Short Selling Agent
Description: Centralised configuration — environment variables, thresholds, constants.
             All tuneable parameters live here; nothing else should call os.getenv directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API credentials ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN", "")
USER_PHONE          = os.getenv("USER_PHONE", "")
TWILIO_FROM_PHONE   = os.getenv("TWILIO_FROM_PHONE", "")
WECHAT_WEBHOOK_URL  = os.getenv("WECHAT_WEBHOOK_URL", "")

# ── IBKR ──────────────────────────────────────────────────────────────────────
IBKR_HOST           = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT           = int(os.getenv("IBKR_PORT", "7497"))
IBKR_CLIENT_ID      = int(os.getenv("IBKR_CLIENT_ID", "1"))
IBKR_ACCOUNT_ID     = os.getenv("IBKR_ACCOUNT_ID", "")

# ── Watchlist ──────────────────────────────────────────────────────────────────
STOCKS: list[str] = [
    "TSLA", "AAPL", "NVDA", "META", "GOOGL",
    "MSFT", "AMZN", "AMD",  "QCOM", "WDC", "CRM", "PANW",
]

# ── Position sizing ────────────────────────────────────────────────────────────
MAX_SHORT_POSITION_PER_STOCK = float(os.getenv("MAX_SHORT_POSITION", "10000"))
BATCH_SIZES: list[float]     = [0.30, 0.30, 0.40]   # 3-batch entry/exit ratio
PRICE_GUARD_MIN_GAIN         = float(os.getenv("PRICE_GUARD_MIN_GAIN", "40"))
MAX_DAYS_WAIT_COVER          = int(os.getenv("MAX_DAYS_WAIT_COVER", "7"))
DAILY_LOSS_LIMIT             = float(os.getenv("DAILY_LOSS_LIMIT", "2000"))

# ── Signal thresholds ──────────────────────────────────────────────────────────
# ⚡ Tune these after back-testing
EPS_BEAT_THRESHOLD       = 10.0   # minimum % EPS beat to consider
SURGE_THRESHOLD          = 8.0    # minimum % intraday surge above pre-earnings close
SLOWDOWN_PRICE_CHANGE    = 0.3    # max 5-min price move (%) that counts as "slowing"
VOLUME_DROP_THRESHOLD    = 0.4    # volume must drop >40% vs prior 30-min average
PULLBACK_FROM_HIGH       = 1.5    # price must pull back >1.5% from the surge peak
AI_CONFIDENCE_THRESHOLD  = 70     # Claude confidence score needed to proceed

# ── Volatility-based stop loss ─────────────────────────────────────────────────
STOP_LOSS_HIGH_VOL = 0.08   # daily vol > 3%  → 8% hard stop
STOP_LOSS_MED_VOL  = 0.06   # daily vol 2-3% → 6% hard stop
STOP_LOSS_LOW_VOL  = 0.05   # daily vol < 2%  → 5% hard stop

# ── LLM ───────────────────────────────────────────────────────────────────────
CLAUDE_MODEL      = "claude-opus-4-6"
CLAUDE_MAX_TOKENS = 400

# ── Paths ─────────────────────────────────────────────────────────────────────
MEMORY_FILE  = "MEMORY.md"
CHROMA_PATH  = "./chroma_db"
LOG_FILE     = "agent.log"
