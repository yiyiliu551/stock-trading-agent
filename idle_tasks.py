"""
idle_tasks.py  ——  空闲时段任务调度器
═══════════════════════════════════════════════════════════════════

触发条件：
  1. 非交易时段 (盘前/盘后/深夜)
  2. 当前无持仓 (有持仓时不做后台任务，专注监控)
  3. 距上次执行 >= 最小间隔

任务列表：
  Task A  新闻采集        每 1 小时  采集12只股票最新新闻标题
  Task B  情绪分析        每 1 小时  Claude分析每只股票 bullish/bearish/neutral
  Task C  MEMORY.md更新  每 2 小时  把情绪+新闻写入知识库
  Task D  昨日回测        每天 00:00  回测昨日策略的实际准确率

结果通过 heartbeat.send_idle_report() 上报到企业微信
"""

import re
import time
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import anthropic

from config import STOCKS, ANTHROPIC_API_KEY
from heartbeat import send_idle_report, log_signal, log_error

logger = logging.getLogger("idle_tasks")

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── 任务执行时间追踪 ────────────────────────────────────────
_last_run: Dict[str, float] = {
    "news":    0.0,
    "sentiment": 0.0,
    "memory":  0.0,
    "backtest": 0.0,
}

# 最小间隔 (秒)
TASK_INTERVALS = {
    "news":      1 * 3600,    # 1小时
    "sentiment": 1 * 3600,    # 1小时
    "memory":    2 * 3600,    # 2小时
    "backtest":  24 * 3600,   # 每天一次
}

# ── 新闻 + 情绪缓存 ────────────────────────────────────────
_news_cache: Dict[str, List[str]] = {}      # ticker → [headline1, ...]
_sentiment_cache: Dict[str, Dict] = {}      # ticker → {sentiment, score, summary}


# ═══════════════════════════════════════════════════════════
# Task A  新闻采集
# ═══════════════════════════════════════════════════════════

def _fetch_news_for_ticker(ticker: str) -> List[str]:
    """
    用 DuckDuckGo 搜索最新新闻标题
    返回最多5条标题
    """
    try:
        query = f"{ticker} stock news today earnings"
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1"
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()

        headlines = []
        # AbstractText 是摘要
        if data.get("AbstractText"):
            headlines.append(data["AbstractText"][:200])

        # RelatedTopics 里有更多
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict) and topic.get("Text"):
                headlines.append(topic["Text"][:200])

        return headlines[:5]

    except Exception as e:
        logger.warning(f"News fetch failed for {ticker}: {e}")
        return []


def run_news_collection() -> Dict[str, List[str]]:
    """
    Task A: 采集所有12只股票新闻
    返回 {ticker: [headlines]}
    """
    logger.info("📰 Task A: Starting news collection...")
    results = {}
    success = 0

    for ticker in STOCKS:
        headlines = _fetch_news_for_ticker(ticker)
        results[ticker] = headlines
        if headlines:
            success += 1
        time.sleep(0.5)   # 避免请求过快

    _news_cache.update(results)
    logger.info(f"News collection done: {success}/{len(STOCKS)} stocks got news")

    summary_lines = []
    for ticker, hl in results.items():
        count = len(hl)
        preview = hl[0][:60] + "..." if hl else "无新闻"
        summary_lines.append(f"> 📌 **{ticker}** ({count}条): {preview}")

    send_idle_report("📰 新闻采集完成", "\n".join(summary_lines[:8]))
    return results


# ═══════════════════════════════════════════════════════════
# Task B  情绪分析
# ═══════════════════════════════════════════════════════════

def _analyze_sentiment_batch(news_map: Dict[str, List[str]]) -> Dict[str, Dict]:
    """
    Claude批量分析12只股票情绪
    一次API调用处理全部（省token）
    """
    if not any(news_map.values()):
        logger.warning("No news to analyze")
        return {}

    # 构建输入文本
    news_text = ""
    for ticker, headlines in news_map.items():
        if headlines:
            news_text += f"\n[{ticker}]\n"
            for h in headlines:
                news_text += f"- {h}\n"

    prompt = f"""分析以下股票新闻的情绪，对每只股票给出：
1. sentiment: bullish / bearish / neutral
2. score: -1.0 到 1.0 (bullish=正, bearish=负)
3. summary: 一句话原因 (中文, ≤30字)

新闻内容:
{news_text}

返回JSON格式，例如：
{{
  "NVDA": {{"sentiment": "bullish", "score": 0.8, "summary": "数据中心需求强劲，Q4超预期"}},
  "TSLA": {{"sentiment": "bearish", "score": -0.6, "summary": "价格战持续，欧洲销量下滑"}}
}}

只返回JSON，不要其他内容。"""

    try:
        resp = claude.messages.create(
            model="claude-opus-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        logger.info(f"Sentiment analysis done: {len(result)} stocks")
        return result

    except json.JSONDecodeError as e:
        log_error(f"Sentiment JSON parse failed: {e}")
        return {}
    except Exception as e:
        log_error(f"Sentiment analysis API error: {e}")
        return {}


def run_sentiment_analysis() -> Dict[str, Dict]:
    """
    Task B: 分析情绪
    依赖 _news_cache，建议在 run_news_collection() 之后调用
    """
    logger.info("🧠 Task B: Starting sentiment analysis...")

    news_to_analyze = _news_cache if _news_cache else {}
    if not news_to_analyze:
        logger.warning("No news in cache, skipping sentiment")
        return {}

    results = _analyze_sentiment_batch(news_to_analyze)
    _sentiment_cache.update(results)

    # 格式化上报
    bullish  = [t for t, v in results.items() if v.get("sentiment") == "bullish"]
    bearish  = [t for t, v in results.items() if v.get("sentiment") == "bearish"]
    neutral  = [t for t, v in results.items() if v.get("sentiment") == "neutral"]

    lines = [
        f"> 📈 **Bullish** ({len(bullish)}只): {', '.join(bullish) or '无'}",
        f"> 📉 **Bearish** ({len(bearish)}只): {', '.join(bearish) or '无'}",
        f"> ➖ **Neutral** ({len(neutral)}只): {', '.join(neutral) or '无'}",
        "",
    ]
    # 详情
    for ticker, v in results.items():
        emoji = "📈" if v.get("sentiment") == "bullish" else \
                "📉" if v.get("sentiment") == "bearish" else "➖"
        score = v.get("score", 0)
        lines.append(f"> {emoji} **{ticker}** `{score:+.1f}` — {v.get('summary','')}")

    send_idle_report("🧠 情绪分析完成", "\n".join(lines))

    # 同时记录信号
    for ticker, v in results.items():
        if abs(v.get("score", 0)) >= 0.6:
            log_signal(ticker, "news_alert",
                       f"{v['sentiment']} score={v['score']:+.1f}: {v.get('summary','')}")

    return results


# ═══════════════════════════════════════════════════════════
# Task C  更新 MEMORY.md 知识库
# ═══════════════════════════════════════════════════════════

def run_memory_update(sentiment_map: Optional[Dict] = None,
                      memory_file: str = "MEMORY.md") -> bool:
    """
    Task C: 将今日新闻情绪写入 MEMORY.md
    追加到文件末尾，保留历史记录
    """
    logger.info("📝 Task C: Updating MEMORY.md...")
    sm = sentiment_map or _sentiment_cache

    if not sm:
        logger.warning("No sentiment data to write to memory")
        return False

    today_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"\n## 📰 {today_str} 市场情绪快照\n",
        f"*记录时间: {datetime.now().strftime('%H:%M')}*\n\n",
        "| 股票 | 情绪 | 分数 | 摘要 |\n",
        "|------|------|------|------|\n",
    ]
    for ticker in sorted(sm.keys()):
        v = sm[ticker]
        sentiment = v.get("sentiment", "neutral")
        score     = v.get("score", 0.0)
        summary   = v.get("summary", "")
        emoji     = "📈" if sentiment == "bullish" else \
                    "📉" if sentiment == "bearish" else "➖"
        lines.append(f"| **{ticker}** | {emoji} {sentiment} | `{score:+.2f}` | {summary} |\n")

    # 追加写入
    try:
        with open(memory_file, "a", encoding="utf-8") as f:
            f.writelines(lines)
        logger.info(f"MEMORY.md updated with {len(sm)} stocks")

        send_idle_report(
            "📝 MEMORY.md 已更新",
            f"> 写入 {len(sm)} 只股票今日情绪\n> 文件: `{memory_file}`"
        )
        return True
    except Exception as e:
        log_error(f"MEMORY.md write failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# Task D  昨日回测
# ═══════════════════════════════════════════════════════════

def _load_yesterdays_signals(memory_file: str = "MEMORY.md") -> List[Dict]:
    """
    从 MEMORY.md 读取昨日记录的 trade_entered 信号
    （格式: 上次交易记录的 ticker, short_price 等）
    """
    signals = []
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        with open(memory_file, "r", encoding="utf-8") as f:
            content = f.read()
        # 简单解析: 找 "## YYYY-MM-DD Trade:" 格式的行
        pattern = rf"## {yesterday}.*?ticker=(\w+).*?short_price=\$([\d.]+)"
        for match in re.finditer(pattern, content):
            signals.append({
                "ticker": match.group(1),
                "short_price": float(match.group(2))
            })
    except FileNotFoundError:
        logger.info("MEMORY.md not found for backtest")
    except Exception as e:
        logger.warning(f"Load signals failed: {e}")
    return signals


def _get_yesterdays_close(ticker: str) -> Optional[float]:
    """用 yfinance 获取昨日收盘价"""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="2d")
        if len(hist) >= 1:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"yfinance failed for {ticker}: {e}")
    return None


def run_backtest() -> Dict:
    """
    Task D: 回测昨日策略准确率
    
    逻辑：
    - 读取昨日进入的空仓信号 (from MEMORY.md)
    - 获取今日收盘价
    - 如果收盘价 < 空仓价 → 方向正确 (profit)
    - 如果收盘价 > 空仓价 → 方向错误 (loss)
    - 计算整体准确率
    """
    logger.info("🔬 Task D: Running backtest...")

    yesterday_signals = _load_yesterdays_signals()
    if not yesterday_signals:
        logger.info("No signals from yesterday to backtest")
        send_idle_report("🔬 昨日回测", "> 昨日无交易信号可回测")
        return {"accuracy": None, "total": 0}

    results = []
    for sig in yesterday_signals:
        ticker      = sig["ticker"]
        short_price = sig["short_price"]
        close_price = _get_yesterdays_close(ticker)

        if close_price is None:
            continue

        correct = close_price < short_price   # 空仓后下跌 = 正确
        pnl_pct = (short_price - close_price) / short_price * 100
        results.append({
            "ticker": ticker,
            "short_price": short_price,
            "close": close_price,
            "correct": correct,
            "pnl_pct": pnl_pct
        })

    if not results:
        return {"accuracy": None, "total": 0}

    correct_count = sum(1 for r in results if r["correct"])
    accuracy      = correct_count / len(results) * 100

    # 格式化上报
    lines = [f"> **准确率: {accuracy:.0f}%** ({correct_count}/{len(results)})\n"]
    for r in results:
        emoji   = "✅" if r["correct"] else "❌"
        pnl_str = f"+{r['pnl_pct']:.1f}%" if r["pnl_pct"] >= 0 else f"{r['pnl_pct']:.1f}%"
        lines.append(
            f"> {emoji} **{r['ticker']}** | "
            f"空: ${r['short_price']:.2f} → 收: ${r['close']:.2f} | {pnl_str}"
        )

    send_idle_report("🔬 昨日策略回测", "\n".join(lines))

    return {
        "accuracy": accuracy,
        "correct":  correct_count,
        "total":    len(results),
        "details":  results
    }


# ═══════════════════════════════════════════════════════════
# 空闲任务调度器  ——  在主循环中调用
# ═══════════════════════════════════════════════════════════

class IdleTaskScheduler:
    """
    在主循环空闲时按时间间隔运行后台任务
    
    用法:
        scheduler = IdleTaskScheduler()
        
        # 在主循环判断空闲时:
        if not has_active_position and not is_market_hours:
            scheduler.tick()
    """

    def __init__(self):
        self._last_run = {k: 0.0 for k in TASK_INTERVALS}
        self._is_running = False

    def _should_run(self, task: str) -> bool:
        elapsed = time.time() - self._last_run[task]
        return elapsed >= TASK_INTERVALS[task]

    def _mark_done(self, task: str):
        self._last_run[task] = time.time()

    def tick(self, has_position: bool = False) -> List[str]:
        """
        检查并运行到期的空闲任务
        
        Args:
            has_position: 是否有活跃持仓（有持仓时跳过所有任务）
        
        Returns:
            本次运行的任务名列表
        """
        if has_position:
            logger.debug("Has active position, skipping idle tasks")
            return []

        if self._is_running:
            logger.debug("Idle tasks already running, skipping")
            return []

        self._is_running = True
        ran = []

        try:
            # Task A: 新闻采集
            if self._should_run("news"):
                logger.info("⏰ Running idle task: news collection")
                run_news_collection()
                self._mark_done("news")
                ran.append("news")

            # Task B: 情绪分析 (依赖新闻缓存)
            if self._should_run("sentiment") and _news_cache:
                logger.info("⏰ Running idle task: sentiment analysis")
                run_sentiment_analysis()
                self._mark_done("sentiment")
                ran.append("sentiment")

            # Task C: 更新 MEMORY.md
            if self._should_run("memory") and _sentiment_cache:
                logger.info("⏰ Running idle task: memory update")
                run_memory_update()
                self._mark_done("memory")
                ran.append("memory")

            # Task D: 昨日回测 (每天午夜跑一次)
            if self._should_run("backtest"):
                now_hour = datetime.now().hour
                if 0 <= now_hour < 2:   # 只在 00:00-02:00 运行
                    logger.info("⏰ Running idle task: backtest")
                    run_backtest()
                    self._mark_done("backtest")
                    ran.append("backtest")

        except Exception as e:
            log_error(f"Idle task failed: {e}")
        finally:
            self._is_running = False

        if ran:
            logger.info(f"Idle tasks completed: {ran}")
        return ran


# ═══════════════════════════════════════════════════════════
# 本地测试
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== 空闲任务调度器测试 ===\n")

    # 测试情绪分析格式 (不调用真实API)
    mock_sentiment = {
        "NVDA": {"sentiment": "bullish",  "score": 0.85, "summary": "数据中心需求暴增"},
        "TSLA": {"sentiment": "bearish",  "score": -0.70, "summary": "降价竞争损害利润"},
        "AAPL": {"sentiment": "neutral",  "score": 0.10, "summary": "无重大新闻"},
        "META": {"sentiment": "bullish",  "score": 0.60, "summary": "广告业务超预期"},
        "MSFT": {"sentiment": "bullish",  "score": 0.55, "summary": "Azure云增长强劲"},
        "AMD":  {"sentiment": "bearish",  "score": -0.40, "summary": "PC端需求疲软"},
    }

    print("模拟情绪分析结果:")
    for t, v in mock_sentiment.items():
        emoji = "📈" if v["sentiment"] == "bullish" else "📉" if v["sentiment"] == "bearish" else "➖"
        print(f"  {emoji} {t:6s} score={v['score']:+.2f}  {v['summary']}")

    print("\n模拟 MEMORY.md 写入...")
    run_memory_update(mock_sentiment, memory_file="/tmp/MEMORY_TEST.md")
    print("已写入 /tmp/MEMORY_TEST.md")

    print("\n=== 调度器状态 ===")
    scheduler = IdleTaskScheduler()
    print("tasks:", list(scheduler._last_run.keys()))
    print("all should_run:", [t for t in TASK_INTERVALS if scheduler._should_run(t)])
