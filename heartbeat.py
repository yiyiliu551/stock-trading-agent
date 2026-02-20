"""
heartbeat.py  ——  企业微信机器人心跳上报
═══════════════════════════════════════════════════════════════════

上报频率：
  - 交易时段 (9:30-16:00 ET)  : 每 30 分钟
  - 盘后/盘前                 : 每 2 小时
  - 有持仓时                  : 每 15 分钟 (更频繁)
  - 错误告警                  : 立即上报

消息内容：
  ❤️ 心跳     → 运行状态 / CPU / 内存 / 上线时长
  📊 持仓状态  → ticker / 数量 / 当前P&L / 止损线
  🔍 信号摘要  → 今日监控到的所有信号
  🚨 错误告警  → API失败 / 断线 / 异常

企业微信webhook文档：
  https://developer.work.weixin.qq.com/document/path/91770
"""

import os
import time
import psutil
import requests
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List

logger = logging.getLogger("heartbeat")

# ── 企业微信 Webhook ────────────────────────────────────────
WECHAT_WEBHOOK_URL = os.getenv("WECHAT_WEBHOOK_URL", "")
# 格式: https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY

# ── 上报频率配置 ────────────────────────────────────────────
HEARTBEAT_INTERVAL_MARKET   = 30 * 60    # 交易时段: 30分钟
HEARTBEAT_INTERVAL_OFFHOURS = 2 * 60 * 60  # 盘后: 2小时
HEARTBEAT_INTERVAL_POSITION = 15 * 60   # 有持仓: 15分钟

# ── 启动时间 (用于计算上线时长) ────────────────────────────
_AGENT_START_TIME = time.time()

# ── 今日信号缓存 ────────────────────────────────────────────
_today_signals: List[Dict] = []
_error_buffer: List[str] = []


# ═══════════════════════════════════════════════════════════
# 公共接口
# ═══════════════════════════════════════════════════════════

def log_signal(ticker: str, signal_type: str, detail: str):
    """记录一条今日信号（供心跳摘要使用）"""
    _today_signals.append({
        "time": datetime.now().strftime("%H:%M"),
        "ticker": ticker,
        "type": signal_type,   # 例如: surge_detected / slowdown / no_trade
        "detail": detail
    })

def log_error(msg: str):
    """记录一条错误（立即触发告警心跳）"""
    _error_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    logger.error(f"ERROR logged for heartbeat: {msg}")
    # 立即发送告警
    send_alert(msg)

def clear_daily_signals():
    """每天开盘前清空信号缓存"""
    _today_signals.clear()
    _error_buffer.clear()
    logger.info("Daily signals buffer cleared")


# ═══════════════════════════════════════════════════════════
# 心跳内容构建
# ═══════════════════════════════════════════════════════════

def _get_system_status() -> Dict:
    """获取系统运行状态"""
    try:
        proc = psutil.Process(os.getpid())
        mem_mb = proc.memory_info().rss / 1024 / 1024
        cpu_pct = proc.cpu_percent(interval=0.5)
        uptime_sec = time.time() - _AGENT_START_TIME
        uptime_str = str(timedelta(seconds=int(uptime_sec)))

        return {
            "cpu": f"{cpu_pct:.1f}%",
            "mem_mb": f"{mem_mb:.0f} MB",
            "uptime": uptime_str,
            "pid": os.getpid(),
            "status": "🟢 正常运行"
        }
    except Exception as e:
        return {"cpu": "N/A", "mem_mb": "N/A", "uptime": "N/A", "status": f"⚠️ {e}"}


def _get_positions_summary(positions: Optional[List[Dict]] = None) -> str:
    """格式化持仓状态"""
    if not positions:
        return "> 📭 当前无持仓"

    lines = []
    total_pnl = 0.0
    for p in positions:
        ticker     = p.get("ticker", "?")
        short_price = p.get("short_price", 0)
        cur_price  = p.get("current_price", 0)
        shares     = p.get("shares", 0)
        stop_loss  = p.get("stop_loss", 0)
        pnl        = (short_price - cur_price) * shares
        total_pnl += pnl
        pnl_str    = f"+${pnl:.0f}" if pnl > 0 else f"-${abs(pnl):.0f}"
        emoji      = "✅" if pnl >= 0 else "🔴"

        lines.append(
            f"> {emoji} **{ticker}** | 空: ${short_price:.2f} → 现: ${cur_price:.2f} "
            f"| {shares}股 | **{pnl_str}** | 止损: ${stop_loss:.2f}"
        )

    total_str = f"+${total_pnl:.0f}" if total_pnl >= 0 else f"-${abs(total_pnl):.0f}"
    lines.append(f"> 💰 **合计P&L: {total_str}**")
    return "\n".join(lines)


def _get_signals_summary() -> str:
    """格式化今日信号摘要"""
    if not _today_signals:
        return "> 📊 今日暂无信号"

    lines = []
    for s in _today_signals[-10:]:  # 最多显示最近10条
        emoji = {
            "surge_detected": "🚀",
            "slowdown": "📉",
            "trade_entered": "💹",
            "trade_covered": "🏁",
            "no_trade": "⏭️",
            "news_alert": "📰",
        }.get(s["type"], "•")
        lines.append(f"> {emoji} `{s['time']}` **{s['ticker']}** — {s['detail']}")

    if len(_today_signals) > 10:
        lines.append(f"> _...还有 {len(_today_signals)-10} 条信号_")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Markdown 消息格式化
# ═══════════════════════════════════════════════════════════

def _build_heartbeat_message(positions: Optional[List[Dict]] = None) -> str:
    """构建完整的心跳 Markdown 消息"""
    now = datetime.now().strftime("%m/%d %H:%M")
    sys = _get_system_status()

    msg = f"""# ❤️ Agent 心跳  `{now}`

## 🖥️ 运行状态
> {sys['status']}
> CPU: `{sys['cpu']}` | 内存: `{sys['mem_mb']}` | 上线: `{sys['uptime']}`

## 📊 持仓状态
{_get_positions_summary(positions)}

## 🔍 今日信号摘要
{_get_signals_summary()}"""

    if _error_buffer:
        errors = "\n".join(f"> 🚨 {e}" for e in _error_buffer[-3:])
        msg += f"\n\n## ⚠️ 错误告警（最近3条）\n{errors}"

    return msg


def _build_alert_message(error_msg: str) -> str:
    """紧急告警消息"""
    now = datetime.now().strftime("%H:%M:%S")
    return f"""# 🚨 Agent 告警  `{now}`

> **错误**: {error_msg}

请检查 agent 状态。"""


def _build_idle_report(task_name: str, result_summary: str) -> str:
    """空闲任务完成上报"""
    now = datetime.now().strftime("%H:%M")
    return f"""# 🌙 空闲任务完成  `{now}`

**任务**: {task_name}
**结果**:
{result_summary}"""


# ═══════════════════════════════════════════════════════════
# 发送到企业微信
# ═══════════════════════════════════════════════════════════

def _send_to_wechat(markdown_text: str) -> bool:
    """发送 Markdown 消息到企业微信机器人"""
    if not WECHAT_WEBHOOK_URL:
        logger.warning("WECHAT_WEBHOOK_URL not set, skipping send")
        print("[HEARTBEAT PREVIEW]\n" + markdown_text[:300] + "...\n")
        return False

    payload = {
        "msgtype": "markdown",
        "markdown": {
            "content": markdown_text
        }
    }
    try:
        r = requests.post(WECHAT_WEBHOOK_URL, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") == 0:
            logger.info("✅ WeChat heartbeat sent successfully")
            return True
        else:
            logger.error(f"WeChat API error: {data}")
            return False
    except requests.exceptions.Timeout:
        logger.error("WeChat webhook timeout")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("WeChat webhook connection error (network?)")
        return False
    except Exception as e:
        logger.error(f"WeChat send failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# 公开发送函数
# ═══════════════════════════════════════════════════════════

def send_heartbeat(positions: Optional[List[Dict]] = None) -> bool:
    """
    发送完整心跳消息
    
    Args:
        positions: 当前持仓列表，每个元素包含:
                   {ticker, short_price, current_price, shares, stop_loss}
    """
    msg = _build_heartbeat_message(positions)
    logger.info(f"Sending heartbeat | positions={len(positions) if positions else 0}")
    return _send_to_wechat(msg)


def send_alert(error_msg: str) -> bool:
    """立即发送错误告警"""
    msg = _build_alert_message(error_msg)
    logger.warning(f"Sending alert: {error_msg}")
    return _send_to_wechat(msg)


def send_idle_report(task_name: str, result_summary: str) -> bool:
    """空闲任务完成后上报"""
    msg = _build_idle_report(task_name, result_summary)
    return _send_to_wechat(msg)


def send_trade_notification(trade: Dict) -> bool:
    """
    发送交易执行通知（开仓/平仓均调用）
    
    trade = {
        event: "opened" | "covered",
        ticker, short_price, cover_price(可选),
        shares, profit_loss(平仓时有), stop_loss
    }
    """
    now = datetime.now().strftime("%H:%M")
    event = trade.get("event", "unknown")
    ticker = trade.get("ticker", "?")

    if event == "opened":
        msg = f"""# 💹 开空仓  `{now}`

> **{ticker}** | 空仓价: `${trade.get('short_price', 0):.2f}`
> 股数: {trade.get('shares', 0)} | 止损: `${trade.get('stop_loss', 0):.2f}`"""

    elif event == "covered":
        pnl = trade.get("profit_loss", 0)
        pnl_str = f"+${pnl:.0f}" if pnl >= 0 else f"-${abs(pnl):.0f}"
        emoji = "✅ 盈利" if pnl >= 0 else "❌ 亏损"
        msg = f"""# 🏁 平仓  `{now}`

> **{ticker}** | {emoji} **{pnl_str}**
> 空仓: `${trade.get('short_price',0):.2f}` → 平仓: `${trade.get('cover_price',0):.2f}`
> 持有 {trade.get('days_held', 0):.1f} 天 | {trade.get('shares', 0)} 股"""

    else:
        msg = f"# ℹ️ 交易事件 `{now}`\n> {ticker}: {trade}"

    return _send_to_wechat(msg)


# ═══════════════════════════════════════════════════════════
# 心跳调度器 (在 main_loop 中调用)
# ═══════════════════════════════════════════════════════════

class HeartbeatScheduler:
    """
    管理心跳发送时机，避免过于频繁
    
    用法:
        scheduler = HeartbeatScheduler()
        
        # 在主循环中:
        scheduler.tick(positions=current_positions, 
                       is_market_hours=True,
                       has_position=bool(current_positions))
    """
    def __init__(self):
        self._last_sent = 0.0
        self._send_count = 0

    def get_interval(self, is_market_hours: bool, has_position: bool) -> int:
        """根据当前状态决定心跳间隔（秒）"""
        if has_position:
            return HEARTBEAT_INTERVAL_POSITION   # 15分钟
        elif is_market_hours:
            return HEARTBEAT_INTERVAL_MARKET     # 30分钟
        else:
            return HEARTBEAT_INTERVAL_OFFHOURS   # 2小时

    def tick(self, positions: Optional[List[Dict]] = None,
             is_market_hours: bool = False,
             has_position: bool = False) -> bool:
        """
        检查是否该发心跳，是的话发送并返回 True
        在主循环每次迭代中调用即可
        """
        interval = self.get_interval(is_market_hours, has_position)
        now = time.time()

        if now - self._last_sent >= interval:
            ok = send_heartbeat(positions)
            if ok or not WECHAT_WEBHOOK_URL:
                self._last_sent = now
                self._send_count += 1
                logger.info(f"Heartbeat #{self._send_count} sent (interval={interval//60}min)")
            return True
        return False


# ═══════════════════════════════════════════════════════════
# 本地测试
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 模拟一些信号
    log_signal("NVDA", "surge_detected", "盘前+12.3%，EPS超预期18%")
    log_signal("TSLA", "no_trade", "市场健康检查失败，SPY -2.3%")
    log_signal("META", "trade_entered", "空仓 $580.20，止损 $598.30")

    # 模拟持仓
    positions = [
        {"ticker": "META", "short_price": 580.20, "current_price": 572.50,
         "shares": 17, "stop_loss": 598.30},
    ]

    print("=== 心跳消息预览 ===\n")
    print(_build_heartbeat_message(positions))
    print("\n=== 告警消息预览 ===\n")
    print(_build_alert_message("IBKR API connection lost after 3 retries"))
    print("\n=== 空闲任务上报预览 ===\n")
    print(_build_idle_report(
        "新闻情绪分析",
        "> 📈 NVDA: Bullish（3篇正面报道，数据中心需求强劲）\n"
        "> 📉 TSLA: Bearish（价格战持续，欧洲销量下滑）\n"
        "> ➖ AAPL: Neutral（无重大新闻）"
    ))
