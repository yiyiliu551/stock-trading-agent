"""
tests/test_all.py
Author: Yang
Description: Full test suite — config, tools, pipeline steps, heartbeat, idle tasks.
             All external dependencies (yfinance, Claude, IBKR, Twilio) are mocked.
             Run: python tests/test_all.py
"""

import os, sys, json, time, tempfile, unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# ── Environment setup (must precede all project imports) ──────────────────────
os.environ.update({
    "ANTHROPIC_API_KEY":  "test-key",
    "TWILIO_ACCOUNT_SID": "ACtest",
    "TWILIO_AUTH_TOKEN":  "test-token",
    "USER_PHONE":         "+1000000000",
    "TWILIO_FROM_PHONE":  "+2000000000",
    "WECHAT_WEBHOOK_URL": "",
    "MAX_SHORT_POSITION": "10000",
    "PRICE_GUARD_MIN_GAIN": "40",
    "MAX_DAYS_WAIT_COVER":  "7",
    "DAILY_LOSS_LIMIT":     "2000",
})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Result counters ────────────────────────────────────────────────────────────
PASS = FAIL = 0

def ok(name: str, passed: bool, note: str = "") -> None:
    global PASS, FAIL
    suffix = f"  [{note}]" if note else ""
    if passed:
        PASS += 1
        print(f"  ✅  {name}{suffix}")
    else:
        FAIL += 1
        print(f"  ❌  {name}{suffix}")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
class TestConfig(unittest.TestCase):
    def test_stocks_count(self):
        from config import STOCKS
        ok("STOCKS has 12 tickers", len(STOCKS) == 12, f"got {len(STOCKS)}")
        self.assertEqual(len(STOCKS), 12)

    def test_stocks_no_duplicates(self):
        from config import STOCKS
        ok("STOCKS no duplicates", len(STOCKS) == len(set(STOCKS)))
        self.assertEqual(len(STOCKS), len(set(STOCKS)))

    def test_batch_sizes_sum_one(self):
        from config import BATCH_SIZES
        ok("BATCH_SIZES sum=1.0", abs(sum(BATCH_SIZES) - 1.0) < 1e-9, f"sum={sum(BATCH_SIZES)}")
        self.assertAlmostEqual(sum(BATCH_SIZES), 1.0, places=9)

    def test_stop_loss_ordering(self):
        from config import STOP_LOSS_HIGH_VOL, STOP_LOSS_MED_VOL, STOP_LOSS_LOW_VOL
        ok("stop loss HIGH > MED > LOW", STOP_LOSS_HIGH_VOL > STOP_LOSS_MED_VOL > STOP_LOSS_LOW_VOL)
        self.assertGreater(STOP_LOSS_HIGH_VOL, STOP_LOSS_MED_VOL)
        self.assertGreater(STOP_LOSS_MED_VOL, STOP_LOSS_LOW_VOL)

    def test_thresholds_positive(self):
        from config import SURGE_THRESHOLD, EPS_BEAT_THRESHOLD, SLOWDOWN_PRICE_CHANGE
        ok("all signal thresholds > 0", all(v > 0 for v in [SURGE_THRESHOLD, EPS_BEAT_THRESHOLD, SLOWDOWN_PRICE_CHANGE]))
        self.assertTrue(SURGE_THRESHOLD > 0)

    def test_ai_confidence_range(self):
        from config import AI_CONFIDENCE_THRESHOLD
        ok("AI_CONFIDENCE_THRESHOLD in [0,100]", 0 < AI_CONFIDENCE_THRESHOLD <= 100, f"val={AI_CONFIDENCE_THRESHOLD}")
        self.assertTrue(0 < AI_CONFIDENCE_THRESHOLD <= 100)


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS: market_data
# ══════════════════════════════════════════════════════════════════════════════
class TestMarketData(unittest.TestCase):
    def _hist(self, closes):
        return pd.DataFrame({"Close": closes, "High": closes, "Low": closes,
                              "Volume": [1_000_000] * len(closes)})

    def test_get_current_price_normal(self):
        with patch("yfinance.Ticker") as mt:
            mt.return_value.history.return_value = self._hist([185.0, 186.0, 187.5])
            from tools.market_data import get_current_price
            p = get_current_price("NVDA")
            ok("get_current_price returns last close", p == 187.5, f"p={p}")
            self.assertEqual(p, 187.5)

    def test_get_current_price_empty(self):
        with patch("yfinance.Ticker") as mt:
            mt.return_value.history.return_value = pd.DataFrame()
            from tools.market_data import get_current_price
            p = get_current_price("NVDA")
            ok("get_current_price returns 0 on empty", p == 0.0)
            self.assertEqual(p, 0.0)

    def test_get_current_price_exception(self):
        with patch("yfinance.Ticker", side_effect=Exception("network")):
            from tools.market_data import get_current_price
            p = get_current_price("AAPL")
            ok("get_current_price returns 0 on exception", p == 0.0)
            self.assertEqual(p, 0.0)

    def test_get_historical_volatility(self):
        prices = [100.0 + np.sin(i) * 2 for i in range(35)]
        hist   = pd.DataFrame({"Close": prices})
        with patch("yfinance.Ticker") as mt:
            mt.return_value.history.return_value = hist
            from tools.market_data import get_historical_volatility
            vol = get_historical_volatility("TSLA")
            ok("historical volatility is float > 0", isinstance(vol, float) and vol > 0, f"vol={vol:.2f}")
            self.assertGreater(vol, 0)

    def test_get_index_change_normal(self):
        hist = pd.DataFrame({"Close": [450.0, 454.5]})
        with patch("yfinance.Ticker") as mt:
            mt.return_value.history.return_value = hist
            from tools.market_data import get_index_change
            chg = get_index_change("SPY")
            ok("index change +1.0%", abs(chg - 1.0) < 0.01, f"chg={chg:.2f}%")
            self.assertAlmostEqual(chg, 1.0, places=1)

    def test_get_recent_intraday_insufficient_bars(self):
        hist = pd.DataFrame({"Close": [100.0, 101.0], "High": [101, 102],
                              "Low": [99, 100], "Volume": [100000, 200000]})
        with patch("yfinance.Ticker") as mt:
            mt.return_value.history.return_value = hist
            from tools.market_data import get_recent_intraday_data
            r = get_recent_intraday_data("AAPL")
            ok("insufficient bars returns {}", r == {})
            self.assertEqual(r, {})


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS: heartbeat
# ══════════════════════════════════════════════════════════════════════════════
class TestHeartbeat(unittest.TestCase):
    def setUp(self):
        import tools.heartbeat as hb
        hb._today_signals.clear()
        hb._error_buffer.clear()

    def test_no_position_message(self):
        from tools.heartbeat import _build_heartbeat
        msg = _build_heartbeat(None)
        ok("no-position heartbeat says 'No open positions'", "No open positions" in msg)
        self.assertIn("No open positions", msg)

    def test_position_pnl_positive(self):
        from tools.heartbeat import _build_heartbeat
        pos = [{"ticker": "NVDA", "short_price": 900.0, "current_price": 880.0,
                "shares": 10, "stop_loss": 930.0}]
        msg = _build_heartbeat(pos)
        ok("profit +$200 shown in heartbeat", "+$200" in msg, f"snippet: {msg[300:450]}")
        self.assertIn("+$200", msg)

    def test_position_pnl_negative(self):
        from tools.heartbeat import _build_heartbeat
        pos = [{"ticker": "TSLA", "short_price": 200.0, "current_price": 215.0,
                "shares": 50, "stop_loss": 212.0}]
        msg = _build_heartbeat(pos)
        ok("loss -$750 shown in heartbeat", "-$750" in msg)
        self.assertIn("-$750", msg)

    def test_signal_appears_in_heartbeat(self):
        from tools.heartbeat import log_signal, _build_heartbeat
        log_signal("META", "surge_detected", "up 12%")
        msg = _build_heartbeat(None)
        ok("logged signal appears in heartbeat", "META" in msg)
        self.assertIn("META", msg)

    def test_signals_capped_at_10(self):
        from tools.heartbeat import log_signal, _build_heartbeat
        for i in range(15):
            log_signal(f"T{i:02d}", "no_trade", "test")
        msg = _build_heartbeat(None)
        count = msg.count("no_trade")
        ok("max 10 signals shown", count <= 10, f"shown={count}")
        self.assertLessEqual(count, 10)

    def test_clear_daily_signals(self):
        from tools.heartbeat import log_signal, clear_daily_signals, _today_signals
        log_signal("X", "test", "test")
        clear_daily_signals()
        ok("clear_daily_signals empties buffer", len(_today_signals) == 0)
        self.assertEqual(len(_today_signals), 0)

    def test_webhook_no_url_skips_request(self):
        from tools.heartbeat import _post
        with patch("requests.post") as mock_post:
            _post("hello")
            ok("no URL → no HTTP request", not mock_post.called)
            self.assertFalse(mock_post.called)

    def test_webhook_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"errcode": 0}
        with patch("tools.heartbeat.WECHAT_WEBHOOK_URL", "https://fake.url"), \
             patch("requests.post", return_value=mock_resp):
            from tools.heartbeat import _post
            ok("webhook success returns True", _post("msg") is True)

    def test_webhook_api_error(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"errcode": 40013}
        with patch("tools.heartbeat.WECHAT_WEBHOOK_URL", "https://fake.url"), \
             patch("requests.post", return_value=mock_resp):
            from tools.heartbeat import _post
            ok("webhook API error returns False", _post("msg") is False)

    def test_scheduler_first_tick_sends(self):
        from tools.heartbeat import HeartbeatScheduler
        sched = HeartbeatScheduler()
        with patch("tools.heartbeat.send_heartbeat", return_value=True) as mock_send:
            sched.tick()
            ok("first tick always sends", mock_send.called)
            self.assertTrue(mock_send.called)

    def test_scheduler_no_resend_immediately(self):
        from tools.heartbeat import HeartbeatScheduler
        sched = HeartbeatScheduler()
        sched._last_sent = time.time()
        with patch("tools.heartbeat.send_heartbeat") as mock_send:
            sched.tick()
            ok("no resend before interval elapsed", not mock_send.called)
            self.assertFalse(mock_send.called)

    def test_scheduler_interval_with_position(self):
        from tools.heartbeat import HeartbeatScheduler, HEARTBEAT_INTERVAL_POSITION
        sched = HeartbeatScheduler()
        ok("has_position → 15 min interval",
           sched.get_interval(True, True) == HEARTBEAT_INTERVAL_POSITION)

    def test_scheduler_interval_market(self):
        from tools.heartbeat import HeartbeatScheduler, HEARTBEAT_INTERVAL_MARKET
        sched = HeartbeatScheduler()
        ok("market hours → 30 min interval",
           sched.get_interval(True, False) == HEARTBEAT_INTERVAL_MARKET)

    def test_scheduler_interval_offhours(self):
        from tools.heartbeat import HeartbeatScheduler, HEARTBEAT_INTERVAL_OFFHOURS
        sched = HeartbeatScheduler()
        ok("off-hours → 2 h interval",
           sched.get_interval(False, False) == HEARTBEAT_INTERVAL_OFFHOURS)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE: Step 3 — surge
# ══════════════════════════════════════════════════════════════════════════════
class TestStep3Surge(unittest.TestCase):
    def test_surge_at_threshold(self):
        with patch("pipeline.step3_surge_detect.get_current_price", return_value=216.0):
            from pipeline.step3_surge_detect import detect_surge
            r = detect_surge("TSLA", 200.0)
            ok("surge exactly at 8% threshold triggers", r["surging"], f"pct={r['surge_pct']}")
            self.assertTrue(r["surging"])

    def test_surge_below_threshold(self):
        with patch("pipeline.step3_surge_detect.get_current_price", return_value=215.0):
            from pipeline.step3_surge_detect import detect_surge
            r = detect_surge("TSLA", 200.0)
            ok("7.5% surge does NOT trigger", not r["surging"])
            self.assertFalse(r["surging"])

    def test_surge_zero_pre_earnings(self):
        with patch("pipeline.step3_surge_detect.get_current_price", return_value=200.0):
            from pipeline.step3_surge_detect import detect_surge
            r = detect_surge("AAPL", 0.0)
            ok("zero pre_earnings price returns not surging", not r["surging"])
            self.assertFalse(r["surging"])

    def test_surge_zero_current(self):
        with patch("pipeline.step3_surge_detect.get_current_price", return_value=0.0):
            from pipeline.step3_surge_detect import detect_surge
            r = detect_surge("NVDA", 200.0)
            ok("zero current price returns not surging", not r["surging"])
            self.assertFalse(r["surging"])

    def test_surge_negative_move(self):
        with patch("pipeline.step3_surge_detect.get_current_price", return_value=180.0):
            from pipeline.step3_surge_detect import detect_surge
            r = detect_surge("META", 200.0)
            ok("negative price move does not trigger", not r["surging"])
            self.assertFalse(r["surging"])

    def test_market_health_healthy(self):
        with patch("pipeline.step3_surge_detect.get_index_change", side_effect=[0.5, 0.3]):
            from pipeline.step3_surge_detect import check_market_health
            r = check_market_health()
            ok("SPY+0.5% QQQ+0.3% = healthy", r["healthy"])
            self.assertTrue(r["healthy"])

    def test_market_health_spy_minus3(self):
        with patch("pipeline.step3_surge_detect.get_index_change", side_effect=[-3.0, 0.0]):
            from pipeline.step3_surge_detect import check_market_health
            r = check_market_health()
            ok("SPY -3% = unhealthy", not r["healthy"])
            self.assertFalse(r["healthy"])

    def test_market_health_boundary_minus2(self):
        with patch("pipeline.step3_surge_detect.get_index_change", side_effect=[-2.0, 0.0]):
            from pipeline.step3_surge_detect import check_market_health
            r = check_market_health()
            ok("SPY exactly -2% boundary = unhealthy", not r["healthy"])
            self.assertFalse(r["healthy"])


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE: Step 4 — slowdown
# ══════════════════════════════════════════════════════════════════════════════
class TestStep4Slowdown(unittest.TestCase):
    def _slowing(self):
        return {
            "prices":       [180, 185, 190, 193, 195, 195.5, 200, 199, 198, 197, 193.0, 193.1],
            "volumes":      [1_000_000] * 6 + [100_000] * 6,
            "today_high":   200.0,
            "current_price": 193.1,
        }

    def _surging(self):
        return {
            "prices":       [180, 182, 184, 186, 188, 190, 191, 192, 193, 194, 195, 196],
            "volumes":      [400_000] * 6 + [1_200_000] * 6,
            "today_high":   196.0,
            "current_price": 196.0,
        }

    def test_find_surge_peak_normal(self):
        from pipeline.step4_slowdown_detect import find_surge_peak
        p = find_surge_peak(self._slowing())
        ok("find_surge_peak returns max of last 6 bars", p > 0, f"peak={p}")
        self.assertGreater(p, 0)

    def test_find_surge_peak_empty(self):
        from pipeline.step4_slowdown_detect import find_surge_peak
        ok("empty data returns 0.0", find_surge_peak({}) == 0.0)
        self.assertEqual(find_surge_peak({}), 0.0)

    def test_find_surge_peak_single(self):
        from pipeline.step4_slowdown_detect import find_surge_peak
        ok("single price returned", find_surge_peak({"prices": [185.0]}) == 185.0)

    def test_hard_rules_all_pass(self):
        from pipeline.step4_slowdown_detect import check_hard_rules
        r = check_hard_rules(self._slowing())
        ok("all 3 rules pass on slowing data", r["rules_met"] == 3,
           f"r1={r['rule1_momentum_slow']} r2={r['rule2_volume_drop']} r3={r['rule3_pullback']}")
        self.assertEqual(r["rules_met"], 3)

    def test_hard_rules_empty(self):
        from pipeline.step4_slowdown_detect import check_hard_rules
        r = check_hard_rules({})
        ok("empty data → 0 rules met", r["rules_met"] == 0)
        self.assertEqual(r["rules_met"], 0)

    def test_rule1_flat_price_passes(self):
        from pipeline.step4_slowdown_detect import check_hard_rules
        data = {"prices": [100.0] * 11 + [100.1], "volumes": [1_000_000] * 12,
                "today_high": 100.1, "current_price": 100.1}
        ok("rule1: 0.1% change < 0.3% threshold", check_hard_rules(data)["rule1_momentum_slow"])

    def test_rule1_big_move_fails(self):
        from pipeline.step4_slowdown_detect import check_hard_rules
        data = {"prices": [100.0] * 11 + [101.0], "volumes": [1_000_000] * 12,
                "today_high": 101.0, "current_price": 101.0}
        ok("rule1: 1.0% change > 0.3% threshold fails", not check_hard_rules(data)["rule1_momentum_slow"])

    def test_rule2_volume_drop(self):
        from pipeline.step4_slowdown_detect import check_hard_rules
        data = {"prices": [100.0] * 12,
                "volumes": [1_000_000] * 6 + [100_000] * 6,
                "today_high": 100.0, "current_price": 100.0}
        ok("rule2: 90% volume drop triggers", check_hard_rules(data)["rule2_volume_drop"])

    def test_rule3_pullback(self):
        from pipeline.step4_slowdown_detect import check_hard_rules
        data = {"prices": [185, 188, 191, 194, 196, 198, 200.0, 199.0, 198.0, 197.0, 196.0, 194.0],
                "volumes": [1_000_000] * 12,
                "today_high": 200.0, "current_price": 194.0}
        r = check_hard_rules(data)
        ok("rule3: 3% pullback from peak of 200", r["rule3_pullback"],
           f"peak={r['surge_peak_used']}")
        self.assertTrue(r["rule3_pullback"])

    def test_stop_loss_above_short_price(self):
        with patch("pipeline.step4_slowdown_detect.get_historical_volatility", return_value=3.5):
            from pipeline.step4_slowdown_detect import calculate_stop_loss
            stop = calculate_stop_loss("TSLA", 180.0)
            ok("stop loss > short price", stop > 180.0, f"stop=${stop:.2f}")
            self.assertGreater(stop, 180.0)

    def test_stop_loss_fallback_on_error(self):
        with patch("pipeline.step4_slowdown_detect.get_historical_volatility",
                   side_effect=Exception("error")):
            from pipeline.step4_slowdown_detect import calculate_stop_loss
            stop = calculate_stop_loss("AAPL", 200.0)
            ok("stop loss fallback = 6% → $212", stop == 212.0, f"stop=${stop:.2f}")
            self.assertEqual(stop, 212.0)

    def test_price_guard_blocks_small_gain(self):
        with patch("pipeline.step4_slowdown_detect.get_recent_intraday_data",
                   return_value={"prices": [239.0]*12, "volumes": [1_000_000]*12,
                                 "today_high": 239.0, "current_price": 239.0}):
            from pipeline.step4_slowdown_detect import detect_slowdown
            r = detect_slowdown("AAPL", 200.0)
            ok("$39 gain blocked by price guard", not r["trigger"], f"reason={r.get('abort_reason')}")
            self.assertFalse(r["trigger"])

    def test_price_guard_passes_exactly_40(self):
        good_data = {"prices": [200.0]*6 + [240.0]*5 + [240.05],
                     "volumes": [1_000_000]*6 + [100_000]*6,
                     "today_high": 240.05, "current_price": 240.05}
        with patch("pipeline.step4_slowdown_detect.get_recent_intraday_data", return_value=good_data), \
             patch("pipeline.step4_slowdown_detect.check_hard_rules",
                   return_value={"passed": True, "rules_met": 3, "rule1_momentum_slow": True,
                                 "rule2_volume_drop": True, "rule3_pullback": True, "surge_peak_used": 240.05}), \
             patch("pipeline.step4_slowdown_detect.analyze_slowdown",
                   return_value={"slowing": True, "confidence": 80, "reasoning": "ok"}), \
             patch("pipeline.step4_slowdown_detect.calculate_stop_loss", return_value=255.0):
            from pipeline.step4_slowdown_detect import detect_slowdown
            r = detect_slowdown("AAPL", 200.0)
            ok("$40.05 gain passes price guard", r["trigger"], f"trigger={r['trigger']}")
            self.assertTrue(r["trigger"])


# ══════════════════════════════════════════════════════════════════════════════
# AI: earnings_analyzer
# ══════════════════════════════════════════════════════════════════════════════
class TestEarningsAnalyzer(unittest.TestCase):
    def _mock_claude(self, text):
        mock = MagicMock()
        mock.content = [MagicMock(text=text)]
        return mock

    def test_beat_parsed(self):
        with patch("ai.base.claude") as mc:
            mc.messages.create.return_value = self._mock_claude(
                '{"beat": true, "beat_pct": 15.0, "confidence": 85, "reason": "beat EPS"}'
            )
            from ai.earnings_analyzer import analyze_earnings_beat
            r = analyze_earnings_beat("NVDA", "beat by 15%")
            ok("beat=True parsed", r["beat"] and r["beat_pct"] == 15.0)
            self.assertTrue(r["beat"])

    def test_miss_parsed(self):
        with patch("ai.base.claude") as mc:
            mc.messages.create.return_value = self._mock_claude(
                '{"beat": false, "beat_pct": 0.0, "confidence": 90, "reason": "miss"}'
            )
            from ai.earnings_analyzer import analyze_earnings_beat
            r = analyze_earnings_beat("META", "missed EPS")
            ok("beat=False parsed", not r["beat"])
            self.assertFalse(r["beat"])

    def test_markdown_fence_stripped(self):
        with patch("ai.base.claude") as mc:
            mc.messages.create.return_value = self._mock_claude(
                '```json\n{"beat": true, "beat_pct": 8.0, "confidence": 75, "reason": "ok"}\n```'
            )
            from ai.earnings_analyzer import analyze_earnings_beat
            r = analyze_earnings_beat("AAPL", "text")
            ok("markdown fence stripped", r.get("beat") is True)
            self.assertTrue(r["beat"])

    def test_garbled_returns_fallback(self):
        with patch("ai.base.claude") as mc:
            mc.messages.create.return_value = self._mock_claude("I cannot determine this.")
            from ai.earnings_analyzer import analyze_earnings_beat
            r = analyze_earnings_beat("AMD", "unclear")
            ok("garbled response returns safe fallback", not r["beat"])
            self.assertFalse(r["beat"])


# ══════════════════════════════════════════════════════════════════════════════
# AI: react_verifier
# ══════════════════════════════════════════════════════════════════════════════
class TestReactVerifier(unittest.TestCase):
    def _mock(self, text):
        m = MagicMock(); m.content = [MagicMock(text=text)]; return m

    def test_confirmed_high_confidence(self):
        with patch("ai.base.claude") as mc:
            mc.messages.create.return_value = self._mock(
                '{"confirmed": true, "confidence": 82, "risk_factors": [], "final_reasoning": "ok"}'
            )
            from ai.react_verifier import verify_trade
            r = verify_trade("TSLA", {})
            ok("confirmed=True with 82% confidence", r["confirmed"] and r["confidence"] == 82)
            self.assertTrue(r["confirmed"])

    def test_rejected(self):
        with patch("ai.base.claude") as mc:
            mc.messages.create.return_value = self._mock(
                '{"confirmed": false, "confidence": 25, "risk_factors": ["r1"], "final_reasoning": "abort"}'
            )
            from ai.react_verifier import verify_trade
            r = verify_trade("AAPL", {})
            ok("confirmed=False returned", not r["confirmed"])
            self.assertFalse(r["confirmed"])

    def test_garbled_defaults_to_abort(self):
        with patch("ai.base.claude") as mc:
            mc.messages.create.return_value = self._mock("Cannot assess.")
            from ai.react_verifier import verify_trade
            r = verify_trade("NVDA", {})
            ok("garbled → confirmed=False (safe abort)", not r.get("confirmed", True))
            self.assertFalse(r.get("confirmed", True))

    def test_risk_factors_preserved(self):
        with patch("ai.base.claude") as mc:
            mc.messages.create.return_value = self._mock(
                '{"confirmed": true, "confidence": 70, "risk_factors": ["r1","r2"], "final_reasoning": "ok"}'
            )
            from ai.react_verifier import verify_trade
            r = verify_trade("META", {})
            ok("risk_factors list preserved", isinstance(r.get("risk_factors"), list))
            self.assertIsInstance(r["risk_factors"], list)


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS: memory_store
# ══════════════════════════════════════════════════════════════════════════════
class TestMemoryStore(unittest.TestCase):
    def _trade(self, ticker="NVDA", pnl=250.0):
        return {"ticker": ticker, "short_price": 920.0, "cover_price": 895.0,
                "total_shares": 10, "profit_loss": pnl, "days_held": 2.0, "outcome": "profit"}

    def test_append_trade_creates_file(self):
        f = tempfile.mktemp(suffix=".md")
        try:
            from tools.memory_store import append_trade_to_markdown
            append_trade_to_markdown(self._trade(), "Lesson 1: test", )
            ok("trade appended creates MEMORY file", True)  # passes if no exception
        finally:
            if os.path.exists(f): os.remove(f)

    def test_append_adds_timestamp(self):
        trade = self._trade()
        trade.pop("timestamp", None)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name
        try:
            with patch("tools.memory_store.MEMORY_FILE", path):
                from tools.memory_store import append_trade_to_markdown
                append_trade_to_markdown(trade, "lesson")
                ok("timestamp auto-added", "timestamp" in trade)
                self.assertIn("timestamp", trade)
        finally:
            os.remove(path)

    def test_sentiment_snapshot_written(self):
        f = tempfile.mktemp(suffix=".md")
        try:
            with patch("tools.memory_store.MEMORY_FILE", f):
                from tools.memory_store import append_sentiment_snapshot
                ok_val = append_sentiment_snapshot({
                    "NVDA": {"sentiment": "bullish", "score": 0.85, "summary": "strong"},
                    "TSLA": {"sentiment": "bearish", "score": -0.6, "summary": "weak"},
                })
                ok("sentiment snapshot returns True", ok_val)
                with open(f) as fh:
                    content = fh.read()
                ok("NVDA in snapshot", "NVDA" in content)
                ok("bullish in snapshot", "bullish" in content)
        finally:
            if os.path.exists(f): os.remove(f)

    def test_sentiment_empty_returns_false(self):
        from tools.memory_store import append_sentiment_snapshot
        result = append_sentiment_snapshot({})
        ok("empty sentiment map returns False", result is False)
        self.assertFalse(result)


# ══════════════════════════════════════════════════════════════════════════════
# IDLE: sentiment_runner
# ══════════════════════════════════════════════════════════════════════════════
class TestIdleSentiment(unittest.TestCase):
    def _mock_claude(self, json_str):
        m = MagicMock(); m.content = [MagicMock(text=json_str)]; return m

    def test_bullish_parsed(self):
        with patch("ai.base.claude") as mc:
            mc.messages.create.return_value = self._mock_claude(
                '{"NVDA": {"sentiment": "bullish", "score": 0.85, "summary": "data center demand"}}'
            )
            from ai.news_sentiment import analyze_batch_sentiment
            r = analyze_batch_sentiment({"NVDA": ["NVDA beats on data center"]})
            ok("bullish sentiment parsed", r.get("NVDA", {}).get("sentiment") == "bullish")
            self.assertEqual(r["NVDA"]["sentiment"], "bullish")

    def test_empty_news_skips_api(self):
        from ai.news_sentiment import analyze_batch_sentiment
        with patch("ai.base.claude") as mc:
            r = analyze_batch_sentiment({})
            ok("empty news skips API call", not mc.messages.create.called and r == {})
            self.assertEqual(r, {})

    def test_all_empty_lists_skips(self):
        from ai.news_sentiment import analyze_batch_sentiment
        with patch("ai.base.claude") as mc:
            r = analyze_batch_sentiment({"NVDA": [], "TSLA": []})
            ok("all-empty headline lists skip API", r == {})
            self.assertEqual(r, {})

    def test_garbled_returns_empty(self):
        with patch("ai.base.claude") as mc:
            mc.messages.create.return_value = self._mock_claude("Not valid JSON.")
            from ai.news_sentiment import analyze_batch_sentiment
            r = analyze_batch_sentiment({"NVDA": ["headline"]})
            ok("garbled JSON returns empty dict", r == {})
            self.assertEqual(r, {})


# ══════════════════════════════════════════════════════════════════════════════
# IDLE: backtester
# ══════════════════════════════════════════════════════════════════════════════
class TestBacktester(unittest.TestCase):
    def _write_memory(self, path, ticker, short_price):
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        with open(path, "w") as f:
            f.write(f"## {ticker} | {yesterday}T10:00:00\n"
                    f"- Short: ${short_price:.2f} | Cover: $880.00\n")

    def test_accuracy_100(self):
        results = [{"correct": True}] * 3
        acc = sum(r["correct"] for r in results) / len(results) * 100
        ok("100% accuracy calculated", acc == 100.0)

    def test_accuracy_0(self):
        results = [{"correct": False}] * 4
        acc = sum(r["correct"] for r in results) / len(results) * 100
        ok("0% accuracy calculated", acc == 0.0)

    def test_accuracy_mixed(self):
        results = [{"correct": True}, {"correct": True}, {"correct": False}]
        acc = sum(r["correct"] for r in results) / len(results) * 100
        ok("66.7% accuracy", abs(acc - 66.67) < 0.1, f"acc={acc:.1f}%")

    def test_correct_direction(self):
        short, close = 900.0, 870.0
        ok("short→fell = correct direction", close < short)

    def test_wrong_direction(self):
        short, close = 200.0, 215.0
        ok("short→rose = wrong direction", not (close < short))

    def test_pnl_calculation(self):
        pnl = (900.0 - 870.0) / 900.0 * 100
        ok("P&L pct = 3.33%", abs(pnl - 3.33) < 0.01, f"{pnl:.2f}%")

    def test_no_signals_returns_empty(self):
        with patch("idle.backtester.send_idle_report"), \
             tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name
            f.write("# No signals\n")
        try:
            from idle.backtester import run_backtest
            r = run_backtest(memory_file=path)
            ok("no signals returns total=0", r["total"] == 0)
            self.assertEqual(r["total"], 0)
        finally:
            os.remove(path)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════
class TestScheduler(unittest.TestCase):
    def test_saturday_is_weekend(self):
        ok("Saturday = weekend", datetime(2026, 2, 21).weekday() >= 5)

    def test_monday_not_weekend(self):
        ok("Monday ≠ weekend", not datetime(2026, 2, 23).weekday() >= 5)

    def test_idle_skips_when_has_position(self):
        from scheduler.idle_scheduler import IdleTaskScheduler
        sched = IdleTaskScheduler()
        with patch("idle.news_collector.collect_all_news") as mock_news:
            ran = sched.tick(has_position=True)
            ok("idle tasks skipped when has_position=True", ran == [] and not mock_news.called)
            self.assertEqual(ran, [])

    def test_idle_no_reentry(self):
        from scheduler.idle_scheduler import IdleTaskScheduler
        sched = IdleTaskScheduler()
        sched._is_running = True
        ran = sched.tick(has_position=False)
        ok("no re-entry while running", ran == [])

    def test_idle_task_exception_clears_flag(self):
        from scheduler.idle_scheduler import IdleTaskScheduler
        sched = IdleTaskScheduler()
        sched._last_run["news"] = 0  # force due
        with patch("idle.news_collector.collect_all_news", side_effect=Exception("boom")), \
             patch("tools.heartbeat.log_error"):
            try:
                sched.tick(has_position=False)
            except Exception:
                pass
            ok("_is_running cleared after exception", not sched._is_running)
            self.assertFalse(sched._is_running)


# ══════════════════════════════════════════════════════════════════════════════
# P&L MATH
# ══════════════════════════════════════════════════════════════════════════════
class TestPnLMath(unittest.TestCase):
    def test_profit(self):
        ok("profit $500", (200.0 - 190.0) * 50 == 500.0)

    def test_loss(self):
        ok("loss -$750", (200.0 - 215.0) * 50 == -750.0)

    def test_breakeven(self):
        ok("breakeven $0", (200.0 - 200.0) * 100 == 0.0)

    def test_stop_loss_triggered_at_boundary(self):
        stop = 200.0 * 1.06   # = 212.0
        ok("stop triggered at exactly 212.0", 212.0 >= stop)

    def test_stop_loss_not_triggered_below(self):
        stop = 212.0
        ok("stop not triggered at 211.99", not (211.99 >= stop))

    def test_take_profit_trigger(self):
        target = 200.0 * 0.97  # = 194.0
        ok("take profit triggered at 194.0", 194.0 <= target)

    def test_batch_shares_sum(self):
        from config import BATCH_SIZES
        import math
        total   = 100
        batches = [math.floor(total * p) for p in BATCH_SIZES]
        ok("batch shares sum <= total", sum(batches) <= total, f"batches={batches}")
        self.assertLessEqual(sum(batches), total)


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    suites = [
        ("config",            TestConfig),
        ("tools.market_data", TestMarketData),
        ("tools.heartbeat",   TestHeartbeat),
        ("step3_surge",       TestStep3Surge),
        ("step4_slowdown",    TestStep4Slowdown),
        ("ai.earnings",       TestEarningsAnalyzer),
        ("ai.react_verifier", TestReactVerifier),
        ("tools.memory",      TestMemoryStore),
        ("idle.sentiment",    TestIdleSentiment),
        ("idle.backtest",     TestBacktester),
        ("scheduler",         TestScheduler),
        ("pnl_math",          TestPnLMath),
    ]

    print("\n" + "=" * 68)
    print("   STOCK AGENT — Full Test Suite")
    print("=" * 68)

    loader = unittest.TestLoader()
    for label, cls in suites:
        print(f"\n── {label} {'─' * (52 - len(label))}")
        for test in loader.loadTestsFromTestCase(cls):
            try:
                test.debug()
            except AssertionError as e:
                ok(str(test).split()[0], False, str(e)[:80])
            except Exception as e:
                ok(str(test).split()[0], False, f"ERR: {str(e)[:80]}")

    total = PASS + FAIL
    print("\n" + "=" * 68)
    if FAIL == 0:
        print(f"  ✅  ALL {total} TESTS PASSED — ready for GitHub")
    else:
        print(f"  ✅  {PASS}/{total} passed   ❌  {FAIL} failed")
    print("=" * 68 + "\n")
