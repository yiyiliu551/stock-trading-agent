"""
tools/heartbeat.py
Author: Yang
Description: WeChat Work (Enterprise WeChat) webhook reporter.
             Handles heartbeat scheduling, signal buffering, and alert dispatch.
             No AI logic. Pure HTTP + timing.

Intervals:
    has_position  → every 15 min  (closely monitored)
    market hours  → every 30 min
    off-hours     → every 2 h
    alert         → immediate
"""

import os
import time
import psutil
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

from config import WECHAT_WEBHOOK_URL

logger = logging.getLogger(__name__)

# ── Intervals (seconds) ────────────────────────────────────────────────────────
HEARTBEAT_INTERVAL_POSITION  = 15 * 60
HEARTBEAT_INTERVAL_MARKET    = 30 * 60
HEARTBEAT_INTERVAL_OFFHOURS  = 2 * 3600

# ── Module-level signal buffers ────────────────────────────────────────────────
_agent_start: float   = time.time()
_today_signals: list  = []
_error_buffer: list   = []


# ── Public logging API ─────────────────────────────────────────────────────────

def log_signal(ticker: str, signal_type: str, detail: str) -> None:
    """Append one intraday signal to the daily buffer (shown in heartbeat)."""
    _today_signals.append({
        "time":   datetime.now().strftime("%H:%M"),
        "ticker": ticker,
        "type":   signal_type,
        "detail": detail,
    })


def log_error(msg: str) -> None:
    """Record an error and immediately dispatch an alert to WeChat."""
    ts = datetime.now().strftime("%H:%M:%S")
    _error_buffer.append(f"[{ts}] {msg}")
    logger.error("heartbeat error logged: %s", msg)
    send_alert(msg)


def clear_daily_signals() -> None:
    """Reset signal and error buffers — call once per trading day at open."""
    _today_signals.clear()
    _error_buffer.clear()
    logger.info("Daily signal buffers cleared")


# ── Message builders ───────────────────────────────────────────────────────────

def _system_info() -> dict:
    try:
        proc    = psutil.Process(os.getpid())
        mem_mb  = proc.memory_info().rss / 1024 / 1024
        cpu     = proc.cpu_percent(interval=0.3)
        uptime  = str(timedelta(seconds=int(time.time() - _agent_start)))
        return {"cpu": f"{cpu:.1f}%", "mem": f"{mem_mb:.0f} MB",
                "uptime": uptime, "status": "Running"}
    except Exception:
        return {"cpu": "N/A", "mem": "N/A", "uptime": "N/A", "status": "Unknown"}


def _positions_block(positions: Optional[list]) -> str:
    if not positions:
        return "> No open positions"
    lines = []
    total = 0.0
    for p in positions:
        pnl = (p.get("short_price", 0) - p.get("current_price", 0)) * p.get("shares", 0)
        total += pnl
        sign  = "+" if pnl >= 0 else ""
        icon  = "green_circle" if pnl >= 0 else "red_circle"
        lines.append(
            f"> **{p['ticker']}** short=${p.get('short_price',0):.2f} "
            f"now=${p.get('current_price',0):.2f} "
            f"({p.get('shares',0)} shares) **{sign}${pnl:.0f}**"
        )
    total_s = f"+${total:.0f}" if total >= 0 else f"-${abs(total):.0f}"
    lines.append(f"> **Total P&L: {total_s}**")
    return "\n".join(lines)


def _signals_block() -> str:
    if not _today_signals:
        return "> No signals today"
    _icons = {
        "surge_detected": "rocket",
        "slowdown":       "chart_down",
        "trade_entered":  "moneybag",
        "trade_covered":  "checkered_flag",
        "no_trade":       "next_track",
        "news_alert":     "newspaper",
    }
    shown = _today_signals[-10:]
    lines = [
        f"> `{s['time']}` **{s['ticker']}** [{s['type']}] {s['detail']}"
        for s in shown
    ]
    if len(_today_signals) > 10:
        lines.append(f"> _...and {len(_today_signals) - 10} more_")
    return "\n".join(lines)


def _build_heartbeat(positions: Optional[list]) -> str:
    now = datetime.now().strftime("%m/%d %H:%M")
    sys = _system_info()
    msg = (
        f"# Agent Heartbeat `{now}`\n\n"
        f"## System\n"
        f"> Status: **{sys['status']}** | CPU: `{sys['cpu']}` | "
        f"Mem: `{sys['mem']}` | Up: `{sys['uptime']}`\n\n"
        f"## Positions\n{_positions_block(positions)}\n\n"
        f"## Signals Today\n{_signals_block()}"
    )
    if _error_buffer:
        recent = "\n".join(f"> **{e}**" for e in _error_buffer[-3:])
        msg += f"\n\n## Errors (last 3)\n{recent}"
    return msg


def _build_alert(error_msg: str) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    return f"# ALERT `{ts}`\n\n> {error_msg}\n\nCheck agent immediately."


def _build_trade_notification(trade: dict) -> str:
    now   = datetime.now().strftime("%H:%M")
    event = trade.get("event", "")
    t     = trade.get("ticker", "?")
    if event == "opened":
        return (
            f"# Short Opened `{now}`\n\n"
            f"> **{t}** @ ${trade.get('short_price', 0):.2f} | "
            f"{trade.get('shares', 0)} shares | stop ${trade.get('stop_loss', 0):.2f}"
        )
    if event == "covered":
        pnl = trade.get("profit_loss", 0)
        s   = f"+${pnl:.0f}" if pnl >= 0 else f"-${abs(pnl):.0f}"
        return (
            f"# Short Covered `{now}`\n\n"
            f"> **{t}** {s} | "
            f"entry ${trade.get('short_price', 0):.2f} → "
            f"cover ${trade.get('cover_price', 0):.2f} | "
            f"{trade.get('days_held', 0):.1f}d"
        )
    return f"# Trade Event `{now}`\n\n> {t}: {trade}"


def _build_idle_report(task: str, summary: str) -> str:
    now = datetime.now().strftime("%H:%M")
    return f"# Idle Task: {task} `{now}`\n\n{summary}"


# ── WeChat HTTP dispatch ───────────────────────────────────────────────────────

def _post(markdown: str) -> bool:
    """POST a Markdown message to the WeChat Work webhook. Returns True on success."""
    if not WECHAT_WEBHOOK_URL:
        logger.debug("WECHAT_WEBHOOK_URL not set — skipping send")
        return False
    try:
        r = requests.post(
            WECHAT_WEBHOOK_URL,
            json={"msgtype": "markdown", "markdown": {"content": markdown}},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") == 0:
            return True
        logger.error("WeChat API errcode=%s msg=%s", data.get("errcode"), data.get("errmsg"))
        return False
    except requests.exceptions.Timeout:
        logger.error("WeChat webhook timeout")
    except requests.exceptions.ConnectionError:
        logger.error("WeChat webhook connection error")
    except Exception as e:
        logger.error("WeChat post failed: %s", e)
    return False


# ── Public send API ────────────────────────────────────────────────────────────

def send_heartbeat(positions: Optional[list] = None) -> bool:
    return _post(_build_heartbeat(positions))


def send_alert(error_msg: str) -> bool:
    return _post(_build_alert(error_msg))


def send_trade_notification(trade: dict) -> bool:
    return _post(_build_trade_notification(trade))


def send_idle_report(task: str, summary: str) -> bool:
    return _post(_build_idle_report(task, summary))


# ── Scheduler ─────────────────────────────────────────────────────────────────

class HeartbeatScheduler:
    """
    Rate-limited heartbeat dispatcher.
    Call scheduler.tick(...) on every main-loop iteration.
    """

    def __init__(self) -> None:
        self._last_sent:  float = 0.0
        self._send_count: int   = 0

    def get_interval(self, is_market_hours: bool, has_position: bool) -> int:
        if has_position:
            return HEARTBEAT_INTERVAL_POSITION
        if is_market_hours:
            return HEARTBEAT_INTERVAL_MARKET
        return HEARTBEAT_INTERVAL_OFFHOURS

    def tick(
        self,
        positions: Optional[list] = None,
        is_market_hours: bool = False,
        has_position: bool = False,
    ) -> bool:
        """
        Send a heartbeat if the configured interval has elapsed.
        Returns True when a heartbeat was dispatched.
        """
        interval = self.get_interval(is_market_hours, has_position)
        if time.time() - self._last_sent >= interval:
            ok = send_heartbeat(positions)
            self._last_sent  = time.time()
            self._send_count += 1
            logger.info(
                "Heartbeat #%d sent (interval=%dmin)",
                self._send_count, interval // 60,
            )
            return True
        return False
