"""
Stock Trading Agent - Complete Version
策略标注版：
  ⚡ = 简化逻辑，能跑，但建议Yang用自己的策略替换
  🔒 = 固定逻辑，不需要修改
  📅 = 时间/调度相关
"""
import asyncio
import json
import re
import time
import math
import logging
import requests
import numpy as np
from datetime import datetime
from typing import TypedDict
from langgraph.graph import StateGraph, END
import yfinance as yf
import anthropic
import chromadb
from twilio.rest import Client
from ib_insync import IB, Stock, MarketOrder
from config import *
from heartbeat import HeartbeatScheduler, log_signal, log_error, send_trade_notification, clear_daily_signals
from idle_tasks import IdleTaskScheduler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("trade_memory")


# ═══════════════════════════════════════════════════════════
# 📅 7×24 主循环 + 心跳
# ═══════════════════════════════════════════════════════════

def is_weekend() -> bool:
    return datetime.now().weekday() >= 5

def is_market_hours() -> bool:
    """美股交易时间: 周一到周五 9:30-16:00 ET"""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    hour = now.hour
    minute = now.minute
    # 假设服务器是UTC时间，ET = UTC-5
    return (14, 30) <= (hour, minute) <= (21, 0)

def main_loop():
    """
    📅 7×24 主循环  ——  集成心跳 + 空闲任务
    
    三种状态：
      交易时段  → 每5分钟运行交易 pipeline，心跳每30分钟
      盘后盘前  → 空闲任务（新闻/情绪/回测），心跳每2小时
      周末      → 同盘后，心跳每4小时
    """
    logger.info("🚀 Stock Agent started - 7x24 mode")

    hb_scheduler   = HeartbeatScheduler()    # 企业微信心跳
    idle_scheduler  = IdleTaskScheduler()    # 后台空闲任务
    current_positions: list = []             # 持仓状态缓存
    last_day_reset: str = ""                 # 用于每日清空信号缓存

    while True:
        try:
            now = datetime.now()

            # ── 每日开盘前重置信号缓存 ──────────────────────
            today_str = now.strftime("%Y-%m-%d")
            if today_str != last_day_reset:
                clear_daily_signals()
                last_day_reset = today_str
                logger.info(f"Daily reset: {today_str}")

            has_position = bool(current_positions)

            # ── 心跳上报 ────────────────────────────────────
            hb_scheduler.tick(
                positions=current_positions,
                is_market_hours=is_market_hours(),
                has_position=has_position
            )

            # ── 状态分支 ────────────────────────────────────
            if is_weekend():
                logger.info("📰 Weekend: idle tasks + heartbeat every 4h")
                idle_scheduler.tick(has_position=has_position)
                time.sleep(3600 * 4)

            elif is_market_hours():
                logger.info("📈 Market hours: running trading pipeline...")
                try:
                    result = asyncio.run(run_agent())
                    # 从 pipeline 结果更新持仓缓存
                    if result and result.get("current_positions"):
                        current_positions = result["current_positions"]
                except Exception as e:
                    log_error(f"Trading pipeline error: {e}")
                    logger.error(f"Pipeline error: {e}", exc_info=True)
                time.sleep(300)   # 每5分钟

            else:
                logger.info("💤 Market closed: running idle tasks...")
                idle_scheduler.tick(has_position=has_position)
                time.sleep(1800)  # 每30分钟

        except KeyboardInterrupt:
            logger.info("Agent stopped by user")
            break
        except Exception as e:
            log_error(f"Main loop unexpected error: {e}")
            logger.error(f"Main loop error: {e}", exc_info=True)
            time.sleep(60)  # 出错后等1分钟再继续


# ═══════════════════════════════════════════════════════════
# 📅 周末新闻分析
# ═══════════════════════════════════════════════════════════

def weekend_news_analysis():
    """
    ⚡ 周末分析12只股票的新闻，写入MEMORY.md
    简化版：搜索新闻标题，Claude分析情绪
    """
    logger.info("📰 Weekend news analysis starting...")
    summaries = []

    for ticker in STOCKS:
        try:
            # 搜索新闻
            url = f"https://api.duckduckgo.com/?q={ticker}+stock+news&format=json&no_html=1"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            news_text = data.get("AbstractText", "") or f"No news found for {ticker}"

            # Claude分析
            response = claude.messages.create(
                model="claude-opus-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": f"Analyze sentiment for {ticker}: {news_text}. Reply: bullish/bearish/neutral + 1 sentence reason"}]
            )
            summary = response.content[0].text
            summaries.append(f"- {ticker}: {summary}")
            logger.info(f"  {ticker}: analyzed")
        except Exception as e:
            summaries.append(f"- {ticker}: error - {e}")

    # 写入MEMORY.md
    with open("MEMORY.md", "a") as f:
        f.write(f"\n## Weekend Analysis: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("\n".join(summaries))
        f.write("\n")

    logger.info("✅ Weekend analysis complete")


# ═══════════════════════════════════════════════════════════
# STEP 1: EARNINGS CALENDAR 🔒
# ═══════════════════════════════════════════════════════════

def get_earnings_within_7_days() -> list:
    """🔒 标准逻辑，不需要修改"""
    upcoming = []
    today = datetime.now().replace(tzinfo=None)
    for ticker in STOCKS:
        try:
            stock = yf.Ticker(ticker)
            cal = stock.calendar
            if cal is None:
                continue
            earnings_date = None
            if hasattr(cal, 'columns') and 'Earnings Date' in cal.columns:
                earnings_date = cal['Earnings Date'].iloc[0]
            elif isinstance(cal, dict) and 'Earnings Date' in cal:
                earnings_date = cal['Earnings Date']
                if isinstance(earnings_date, list):
                    earnings_date = earnings_date[0]
            if earnings_date is None:
                continue
            if hasattr(earnings_date, 'tzinfo') and earnings_date.tzinfo:
                earnings_date = earnings_date.replace(tzinfo=None)
            days_until = (earnings_date - today).days
            if 0 <= days_until <= 7:
                upcoming.append({'ticker': ticker, 'earnings_date': earnings_date, 'days_until': days_until})
                logger.info(f"✓ {ticker}: earnings in {days_until} days")
        except Exception as e:
            logger.warning(f"Calendar fetch failed {ticker}: {e}")
    return upcoming


def get_pre_earnings_price(ticker: str) -> float:
    """🔒 标准逻辑"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        return float(hist['Close'].iloc[-1]) if not hist.empty else 0.0
    except:
        return 0.0


# ═══════════════════════════════════════════════════════════
# STEP 2: EARNINGS RESULT DETECTION
# ═══════════════════════════════════════════════════════════

def search_earnings_result(ticker: str) -> str:
    """
    ⚡ 简化版：用DuckDuckGo搜索
    Yang可以替换成Playwright浏览器搜索，效果更好
    替换点：把requests调用改成Playwright页面抓取
    """
    quarter = f"Q{(datetime.now().month - 1) // 3 + 1}"
    year = datetime.now().year
    query = f"{ticker} earnings {quarter} {year} EPS beat"
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        return data.get("AbstractText", "") or f"Search: {query}"
    except Exception as e:
        return f"Search failed: {e}"


def analyze_earnings_with_llm(ticker: str, search_text: str) -> dict:
    """🔒 Claude分析，不需要修改"""
    prompt = f"""Did {ticker} beat EPS expectations?
Info: {search_text}
Reply ONLY in JSON: {{"beat": true/false, "beat_pct": 0.0, "confidence": 0-100, "reason": "brief"}}"""
    try:
        response = claude.messages.create(
            model="claude-opus-4-6", max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        text = re.sub(r'```json|```', '', response.content[0].text).strip()
        return json.loads(text)
    except:
        return {"beat": False, "beat_pct": 0.0, "confidence": 0, "reason": "parse error"}


def check_earnings_beat(ticker: str) -> dict:
    """🔒"""
    search_text = search_earnings_result(ticker)
    result = analyze_earnings_with_llm(ticker, search_text)
    result['ticker'] = ticker
    return result


# ═══════════════════════════════════════════════════════════
# STEP 3: SURGE DETECTION
# ═══════════════════════════════════════════════════════════

def check_market_health() -> dict:
    """🔒 SPY/QQQ健康检查"""
    try:
        spy_hist = yf.Ticker("SPY").history(period="2d")
        qqq_hist = yf.Ticker("QQQ").history(period="2d")
        spy_change = ((spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[-2]) / spy_hist['Close'].iloc[-2]) * 100
        qqq_change = ((qqq_hist['Close'].iloc[-1] - qqq_hist['Close'].iloc[-2]) / qqq_hist['Close'].iloc[-2]) * 100
        return {"healthy": spy_change > -2.0 and qqq_change > -2.0,
                "spy_change": round(spy_change, 2), "qqq_change": round(qqq_change, 2)}
    except:
        return {"healthy": False, "spy_change": 0, "qqq_change": 0}


def get_current_price(ticker: str) -> float:
    """🔒"""
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="5m")
        return float(hist['Close'].iloc[-1]) if not hist.empty else 0.0
    except:
        return 0.0


def detect_surge(ticker: str, pre_earnings_price: float) -> dict:
    """🔒 暴涨检测"""
    current_price = get_current_price(ticker)
    if current_price == 0 or pre_earnings_price == 0:
        return {"surging": False, "surge_pct": 0, "current_price": 0}
    surge_pct = ((current_price - pre_earnings_price) / pre_earnings_price) * 100
    return {
        "surging": surge_pct >= SURGE_THRESHOLD,
        "surge_pct": round(surge_pct, 2),
        "current_price": current_price,
        "pre_earnings_price": pre_earnings_price
    }


# ═══════════════════════════════════════════════════════════
# STEP 4: SLOWDOWN DETECTION
# ═══════════════════════════════════════════════════════════

def get_recent_price_data(ticker: str) -> dict:
    """🔒"""
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="5m")
        if hist.empty or len(hist) < 6:
            return {}
        return {
            "prices": hist['Close'].tolist(),
            "volumes": hist['Volume'].tolist(),
            "today_high": float(hist['High'].max()),
            "current_price": float(hist['Close'].iloc[-1])
        }
    except:
        return {}


# ⚡⚡⚡ Yang的核心策略函数 ⚡⚡⚡
def find_surge_peak(price_data: dict) -> float:
    """
    ⚡ 【Yang自己的策略】找到暴涨高点
    
    现在用的是简单max()，但你应该设计更智能的高点判断：
    思路建议：
    - 用滚动窗口找动量反转点
    - 结合volume加权
    - 参考你在Weibo推荐系统里用过的时序特征方法
    
    !!!! 这里是最核心的策略函数，定义了做空的timing !!!!
    """
    prices = price_data.get("prices", [])
    if not prices:
        return 0.0
    # ⚡ 简化版：用最近30分钟（6个5分钟bar）的最高点
    recent_prices = prices[-6:] if len(prices) >= 6 else prices
    return max(recent_prices)


def check_hard_rules(price_data: dict) -> dict:
    """
    ⚡ 【Yang可以调整阈值】3个硬规则
    
    Rule 1: 价格动量 < SLOWDOWN_PRICE_CHANGE (0.3%) - ⚡ 阈值可以调
    Rule 2: 成交量下降 > VOLUME_DROP_THRESHOLD (40%) - ⚡ 阈值可以调
    Rule 3: 从高点回落 > PULLBACK_FROM_HIGH (1.5%) - ⚡ 阈值可以调，用find_surge_peak
    """
    prices = price_data.get("prices", [])
    volumes = price_data.get("volumes", [])
    current_price = price_data.get("current_price", 0)
    surge_peak = find_surge_peak(price_data)  # ⚡ 使用你自己的高点

    rule1 = False
    rule2 = False
    rule3 = False

    # Rule 1: 最近5分钟涨幅 < 0.3%
    if len(prices) >= 2:
        last_change = abs((prices[-1] - prices[-2]) / prices[-2] * 100)
        rule1 = last_change < SLOWDOWN_PRICE_CHANGE

    # Rule 2: 成交量较之前30分钟均值下降40%
    if len(volumes) >= 7:
        recent_vol = volumes[-1]
        prior_avg = np.mean(volumes[-7:-1])
        if prior_avg > 0:
            rule2 = (prior_avg - recent_vol) / prior_avg >= VOLUME_DROP_THRESHOLD

    # Rule 3: 从surge高点回落1.5%
    if surge_peak > 0 and current_price > 0:
        pullback = (surge_peak - current_price) / surge_peak * 100
        rule3 = pullback >= PULLBACK_FROM_HIGH

    rules_met = sum([rule1, rule2, rule3])
    return {
        "passed": rules_met >= 2,
        "rules_met": rules_met,
        "rule1_momentum_slow": rule1,
        "rule2_volume_drop": rule2,
        "rule3_pullback": rule3,
        "surge_peak_used": surge_peak  # 方便debug
    }


def ai_slowdown_analysis(ticker: str, price_data: dict) -> dict:
    """🔒 Claude分析放缓"""
    prices = price_data.get("prices", [])[-12:]
    volumes = price_data.get("volumes", [])[-12:]
    prompt = f"""Is {ticker}'s post-earnings surge SLOWING DOWN?
Prices (5min): {[round(p,2) for p in prices]}
Volumes: {[int(v) for v in volumes]}
Today high: ${price_data.get('today_high',0):.2f}
Current: ${price_data.get('current_price',0):.2f}
Reply ONLY JSON: {{"slowing": true/false, "confidence": 0-100, "reasoning": "brief"}}"""
    try:
        response = claude.messages.create(
            model="claude-opus-4-6", max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        text = re.sub(r'```json|```', '', response.content[0].text).strip()
        return json.loads(text)
    except:
        return {"slowing": False, "confidence": 0, "reasoning": "error"}


def calculate_stop_loss(ticker: str, short_price: float) -> float:
    """
    ⚡ 【Yang可以优化】动态止损
    
    现在用30日历史波动率，三档止损：
    - 高波动(>3%): 8% stop
    - 中波动(2-3%): 6% stop  
    - 低波动(<2%): 5% stop
    
    ⚡ Yang可以加入：VIX指数、期权隐含波动率等更精准的指标
    """
    try:
        hist = yf.Ticker(ticker).history(period="30d")
        volatility = float(hist['Close'].pct_change().dropna().std() * 100)
        if volatility > 3.0:
            pct = STOP_LOSS_HIGH_VOL
        elif volatility > 2.0:
            pct = STOP_LOSS_MED_VOL
        else:
            pct = STOP_LOSS_LOW_VOL
        return round(short_price * (1 + pct), 2)
    except:
        return round(short_price * 1.06, 2)


def detect_slowdown(ticker: str, pre_earnings_price: float) -> dict:
    """🔒 组合所有子函数"""
    price_data = get_recent_price_data(ticker)
    if not price_data:
        return {"trigger": False, "reason": "No price data"}
    current_price = price_data.get("current_price", 0)
    if current_price - pre_earnings_price < PRICE_GUARD_MIN_GAIN:
        return {"trigger": False, "reason": f"Price guard: only ${current_price - pre_earnings_price:.2f} gain"}
    hard_rules = check_hard_rules(price_data)
    ai_result = ai_slowdown_analysis(ticker, price_data)
    ai_ok = ai_result.get("confidence", 0) >= AI_CONFIDENCE_THRESHOLD
    trigger = hard_rules["passed"] and ai_ok
    stop_loss = calculate_stop_loss(ticker, current_price) if trigger else 0
    return {
        "trigger": trigger,
        "current_price": current_price,
        "hard_rules": hard_rules,
        "ai_analysis": ai_result,
        "stop_loss": stop_loss
    }


# ═══════════════════════════════════════════════════════════
# STEP 5: REACT SELF-VERIFICATION 🔒
# ═══════════════════════════════════════════════════════════

def react_verify(ticker: str, decision_data: dict) -> dict:
    """🔒"""
    prompt = f"""Review this short-sell decision for {ticker}:
{json.dumps(decision_data, indent=2)}

Round 1: Is this safe?
Round 2: Devil's advocate risks?
Reply ONLY JSON: {{"confirmed": true/false, "confidence": 0-100, "risk_factors": [], "final_reasoning": "brief"}}"""
    try:
        response = claude.messages.create(
            model="claude-opus-4-6", max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        text = re.sub(r'```json|```', '', response.content[0].text).strip()
        return json.loads(text)
    except:
        return {"confirmed": False, "confidence": 0, "risk_factors": ["error"], "final_reasoning": "abort"}


# ═══════════════════════════════════════════════════════════
# STEP 6: SMS 🔒
# ═══════════════════════════════════════════════════════════

def send_sms(message: str) -> bool:
    """🔒"""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(body=message, from_=TWILIO_FROM_PHONE, to=USER_PHONE)
        return True
    except Exception as e:
        logger.error(f"SMS failed: {e}")
        return False


def notify_and_wait_approval(ticker: str, price: float, stop_loss: float,
                              confidence: int, rules_met: int) -> bool:
    """🔒"""
    message = (f"🚨 TRADE ALERT: Short {ticker}\n"
               f"Price: ${price:.2f} | Stop: ${stop_loss:.2f}\n"
               f"AI: {confidence}% | Rules: {rules_met}/3\n"
               f"Reply YES/NO (5min timeout)")
    sent_time = time.time()
    if not send_sms(message):
        return False
    while time.time() - sent_time < 300:
        time.sleep(15)
        try:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            msgs = client.messages.list(to=TWILIO_FROM_PHONE, limit=5)
            for msg in msgs:
                if msg.date_sent and msg.date_sent.timestamp() > sent_time and msg.direction == "inbound":
                    if msg.body.strip().upper().startswith("YES"):
                        return True
                    elif msg.body.strip().upper().startswith("NO"):
                        return False
        except:
            pass
    send_sms(f"⏰ Timeout: {ticker} trade aborted")
    return False


# ═══════════════════════════════════════════════════════════
# STEP 7: IBKR SHORT SELL 🔒
# ═══════════════════════════════════════════════════════════

def connect_ibkr() -> IB:
    """🔒"""
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1)
        return ib
    except Exception as e:
        logger.error(f"IBKR connect failed: {e}")
        return None


def short_in_batches(ticker: str, entry_price: float) -> dict:
    """🔒 三批做空"""
    if entry_price <= 0:
        return {"success": False, "reason": "Invalid entry price (zero or negative)"}
    total_shares = int(MAX_SHORT_POSITION_PER_STOCK / entry_price)
    if total_shares < 3:
        return {"success": False, "reason": "Too few shares"}
    ib = connect_ibkr()
    if not ib:
        return {"success": False, "reason": "IBKR not connected"}
    results = []
    total_filled = 0
    for i, pct in enumerate(BATCH_SIZES):
        shares = math.floor(total_shares * pct)
        try:
            contract = Stock(ticker, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            trade = ib.placeOrder(contract, MarketOrder('SELL', shares))
            ib.sleep(3)
            fill_price = trade.orderStatus.avgFillPrice or 0
            results.append({"batch": i+1, "shares": shares, "fill_price": fill_price})
            total_filled += shares
            logger.info(f"Batch {i+1}: Shorted {shares} @ ${fill_price:.2f}")
        except Exception as e:
            results.append({"batch": i+1, "shares": 0, "fill_price": 0, "error": str(e)})
        if i < 2:
            time.sleep(300)
    ib.disconnect()
    avg = sum(r['fill_price'] for r in results if r['fill_price'] > 0) / max(len([r for r in results if r['fill_price'] > 0]), 1)
    return {"success": True, "ticker": ticker, "total_shares_shorted": total_filled,
            "avg_fill_price": round(avg, 2), "batches": results}


# ═══════════════════════════════════════════════════════════
# STEP 8 & 9: MONITOR + COVER
# ═══════════════════════════════════════════════════════════

def monitor_position(ticker: str, short_price: float, stop_loss: float) -> dict:
    """
    ⚡ 【Yang可以优化】监控逻辑
    
    现在的止盈是固定3%回落，Yang可以改成：
    - 动态止盈：根据波动率调整
    - 追踪止损：随价格下跌动态调整stop loss
    - 时间加权：持仓越久止盈目标越保守
    """
    pullback_target = short_price * 0.97  # ⚡ 固定3%，Yang可以改成动态
    start_time = time.time()
    max_seconds = MAX_DAYS_WAIT_COVER * 86400

    while time.time() - start_time < max_seconds:
        try:
            hist = yf.Ticker(ticker).history(period="1d", interval="5m")
            if hist.empty:
                time.sleep(300)
                continue
            current_price = float(hist['Close'].iloc[-1])
            elapsed_days = (time.time() - start_time) / 86400

            if current_price >= stop_loss:
                send_sms(f"🚨 STOP LOSS: {ticker} @ ${current_price:.2f}")
                return {"action": "stop_loss", "price": current_price, "days_held": elapsed_days}
            if current_price <= pullback_target:
                send_sms(f"✅ TAKE PROFIT: {ticker} @ ${current_price:.2f}")
                return {"action": "take_profit", "price": current_price, "days_held": elapsed_days}

            logger.info(f"  {ticker}: ${current_price:.2f} | day {elapsed_days:.1f}")
            time.sleep(300)
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            time.sleep(300)

    send_sms(f"⏰ TIMEOUT: {ticker} covering now")
    return {"action": "timeout", "price": 0, "days_held": MAX_DAYS_WAIT_COVER}


def cover_short_in_batches(ticker: str, total_shares: int, reason: str) -> dict:
    """🔒"""
    ib = connect_ibkr()
    if not ib:
        return {"success": False}
    results = []
    for i, pct in enumerate(BATCH_SIZES):
        shares = math.floor(total_shares * pct)
        try:
            contract = Stock(ticker, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            trade = ib.placeOrder(contract, MarketOrder('BUY', shares))
            ib.sleep(3)
            fill_price = trade.orderStatus.avgFillPrice or 0
            results.append({"batch": i+1, "shares": shares, "fill_price": fill_price})
        except Exception as e:
            results.append({"batch": i+1, "error": str(e)})
        if i < 2:
            time.sleep(300)
    ib.disconnect()
    avg = sum(r['fill_price'] for r in results if r.get('fill_price', 0) > 0) / max(len([r for r in results if r.get('fill_price', 0) > 0]), 1)
    return {"success": True, "avg_cover_price": round(avg, 2)}


# ═══════════════════════════════════════════════════════════
# STEP 10: MEMORY 🔒
# ═══════════════════════════════════════════════════════════

def record_trade(trade_record: dict):
    """🔒"""
    if 'timestamp' not in trade_record:
        trade_record['timestamp'] = datetime.now().isoformat()
    try:
        collection.upsert(ids=[f"{trade_record['ticker']}_{trade_record['timestamp']}"],
                          documents=[json.dumps(trade_record)],
                          metadatas=[{"ticker": trade_record.get("ticker", ""),
                                      "profit_loss": trade_record.get("profit_loss", 0)}])
    except Exception as e:
        logger.error(f"ChromaDB error: {e}")

    response = claude.messages.create(
        model="claude-opus-4-6", max_tokens=300,
        messages=[{"role": "user", "content": f"Review this trade and give 3 lessons: {json.dumps(trade_record)}"}]
    )
    reflection = response.content[0].text

    with open("MEMORY.md", "a") as f:
        f.write(f"\n---\n## {trade_record['ticker']} | {trade_record['timestamp']}\n")
        f.write(f"P&L: ${trade_record.get('profit_loss', 0):.2f}\n\n{reflection}\n")

    return reflection


# ═══════════════════════════════════════════════════════════
# LANGGRAPH STATE MACHINE 🔒
# ═══════════════════════════════════════════════════════════

class AgentState(TypedDict):
    ticker: str
    pre_earnings_price: float
    earnings_list: list
    earnings_beat: dict
    surge_result: dict
    slowdown_result: dict
    verify_result: dict
    short_result: dict
    monitor_result: dict
    cover_result: dict
    approved: bool
    abort_reason: str


def node_check_earnings(state):
    earnings_list = get_earnings_within_7_days()
    state['earnings_list'] = earnings_list
    if earnings_list:
        ticker = earnings_list[0]['ticker']
        state['ticker'] = ticker
        state['pre_earnings_price'] = get_pre_earnings_price(ticker)
    else:
        state['abort_reason'] = "No upcoming earnings"
    return state

def node_detect_earnings_result(state):
    state['earnings_beat'] = check_earnings_beat(state.get('ticker', ''))
    return state

def node_detect_surge(state):
    market = check_market_health()
    if not market['healthy']:
        state['abort_reason'] = "Market unhealthy"
        state['surge_result'] = {"surging": False}
    else:
        state['surge_result'] = detect_surge(state.get('ticker', ''), state.get('pre_earnings_price', 0))
    return state

def node_detect_slowdown(state):
    state['slowdown_result'] = detect_slowdown(state.get('ticker', ''), state.get('pre_earnings_price', 0))
    return state

def node_react_verify(state):
    state['verify_result'] = react_verify(state.get('ticker', ''), state.get('slowdown_result', {}))
    return state

def node_notify_user(state):
    slowdown = state.get('slowdown_result', {})
    verify = state.get('verify_result', {})
    approved = notify_and_wait_approval(
        state.get('ticker', ''), slowdown.get('current_price', 0),
        slowdown.get('stop_loss', 0), verify.get('confidence', 0),
        slowdown.get('hard_rules', {}).get('rules_met', 0)
    )
    state['approved'] = approved
    if not approved:
        state['abort_reason'] = "User rejected"
    return state

def node_execute_short(state):
    state['short_result'] = short_in_batches(
        state.get('ticker', ''), state.get('slowdown_result', {}).get('current_price', 0)
    )
    return state

def node_monitor_and_cover(state):
    short_result = state.get('short_result', {})
    slowdown = state.get('slowdown_result', {})
    monitor_result = monitor_position(
        state.get('ticker', ''), short_result.get('avg_fill_price', 0), slowdown.get('stop_loss', 0)
    )
    state['monitor_result'] = monitor_result
    state['cover_result'] = cover_short_in_batches(
        state.get('ticker', ''), short_result.get('total_shares_shorted', 0), monitor_result.get('action', '')
    )
    return state

def node_record_memory(state):
    short_result = state.get('short_result', {})
    cover_result = state.get('cover_result', {})
    sp = short_result.get('avg_fill_price', 0)
    cp = cover_result.get('avg_cover_price', 0)
    shares = short_result.get('total_shares_shorted', 0)
    record_trade({
        "ticker": state.get('ticker', ''),
        "short_price": sp, "cover_price": cp,
        "total_shares": shares,
        "profit_loss": round((sp - cp) * shares, 2),
        "days_held": state.get('monitor_result', {}).get('days_held', 0),
        "outcome": "profit" if sp > cp else "loss"
    })
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("check_earnings", node_check_earnings)
    g.add_node("check_earnings_result", node_detect_earnings_result)
    g.add_node("detect_surge", node_detect_surge)
    g.add_node("detect_slowdown", node_detect_slowdown)
    g.add_node("react_verify", node_react_verify)
    g.add_node("notify_user", node_notify_user)
    g.add_node("execute_short", node_execute_short)
    g.add_node("monitor_and_cover", node_monitor_and_cover)
    g.add_node("record_memory", node_record_memory)
    g.set_entry_point("check_earnings")
    g.add_conditional_edges("check_earnings", lambda s: "check_earnings_result" if s.get('earnings_list') else END)
    g.add_conditional_edges("check_earnings_result", lambda s: "detect_surge" if s.get('earnings_beat', {}).get('beat') and s.get('earnings_beat', {}).get('beat_pct', 0) >= EPS_BEAT_THRESHOLD else END)
    g.add_conditional_edges("detect_surge", lambda s: "detect_slowdown" if s.get('surge_result', {}).get('surging') and not s.get('abort_reason') else END)
    g.add_conditional_edges("detect_slowdown", lambda s: "react_verify" if s.get('slowdown_result', {}).get('trigger') else END)
    g.add_conditional_edges("react_verify", lambda s: "notify_user" if s.get('verify_result', {}).get('confirmed') else END)
    g.add_conditional_edges("notify_user", lambda s: "execute_short" if s.get('approved') else END)
    g.add_conditional_edges("execute_short", lambda s: "monitor_and_cover" if s.get('short_result', {}).get('success') else END)
    g.add_edge("monitor_and_cover", "record_memory")
    g.add_edge("record_memory", END)
    return g.compile()


async def run_agent():
    app = build_graph()
    initial_state = AgentState(
        ticker="", pre_earnings_price=0.0, earnings_list=[],
        earnings_beat={}, surge_result={}, slowdown_result={},
        verify_result={}, short_result={}, monitor_result={},
        cover_result={}, approved=False, abort_reason=""
    )
    final_state = await app.ainvoke(initial_state)
    if final_state.get('abort_reason'):
        logger.info(f"⏹️  Stopped: {final_state['abort_reason']}")
    return final_state


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("  Stock Agent - 7x24 Mode")
    print("="*60)
    print("\n标注说明:")
    print("  🔒 = 固定逻辑，不需要修改")
    print("  ⚡ = 简化版，Yang可以替换成自己的策略")
    print("  📅 = 时间/调度相关")
    print("\n需要Yang自己优化的函数:")
    print("  1. find_surge_peak()     - 找暴涨高点 (最重要!)")
    print("  2. search_earnings_result() - 换成Playwright")
    print("  3. monitor_position()    - 动态止盈逻辑")
    print("  4. calculate_stop_loss() - 可加VIX等指标")
    print("  5. check_hard_rules()    - 阈值微调")
    print("\n启动7x24模式:")
    print("  main_loop()")
    print("\n单次运行:")
    print("  asyncio.run(run_agent())")
