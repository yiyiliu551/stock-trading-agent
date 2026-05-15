# I gave an LLM a brain, memory, hands — and now a news nervous system

> Design Document (v8): https://github.com/yiyiliu551/stock-trading-agent/blob/main/stock_agent_v8.pdf

## What this is

An AI trading agent built around a single metaphor: a human trader.

- 🧠 **Brain** — Claude (LLM). Reasons, validates, decides when to act and when not to.
- 🤝 **Hands** — Custom tools built for this system: surge detection, volatility stop-loss, batch execution.
- 💓 **Heartbeat** — Background scheduler. Keeps the agent alive 7×24, sends WeChat reports.
- 🗂️ **Memory** — Two-tier: session log + long-term ChromaDB (RAG retrieval).
- 📰 **News nervous system** — Reddit + Xiaohongshu + DuckDuckGo → LLM filter → GraphRAG time-series storage.

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                    main.py  (7×24 loop)                              ║
║         market hours → trading pipeline                              ║
║         off-hours    → idle_scheduler.tick()                         ║
╚══════════════════╤═══════════════════════════╤═══════════════════════╝
                   │                           │
         Market open                     Market closed
                   │                           │
                   ▼                           ▼
   ┌───────────────────────────┐   ┌─────────────────────────────────┐
   │  Trading pipeline         │   │  idle_scheduler.tick()  ✏️      │
   │  step1 → step10           │   └──────────────┬──────────────────┘
   └──────────────┬────────────┘                  │ every 1h
                  │                               ▼
                  ▼                   ┌─────────────────────────────────┐
   ┌───────────────────────────┐      │  news_pipeline.py  (new)        │
   │  step5                    │      │  run_pipeline(confirm=True)      │
   │  react_verifier.py  ✏️    │      └──────┬──────────┬──────────┬───┘
   │  calls get_context_for_   │             │          │          │
   │  llm() ← GraphRAG query   │◀ ─ ─ ─ ─ ─ │   Stage 1: Fetch   │
   └───────────────────────────┘             │          │          │
                                             ▼          ▼          ▼
                                    ┌──────────────┐ ┌──────────┐ ┌──────────────┐
                                    │reddit_       │ │xhs_      │ │duckduckgo    │
                                    │collector     │ │collector │ │              │
                                    └──────┬───────┘ └────┬─────┘ └──────┬───────┘
                                           └──────────────┴──────────────┘
                                                          │
                                                Stage 2: LLM Filter
                                                          │
                                                          ▼
                                           ┌──────────────────────────┐
                                           │  news_filter.py  (new)   │
                                           │  useful / borderline /   │
                                           │  noise                   │
                                           └──────────────┬───────────┘
                                                          │
                                               Stage 3: User Verification
                                                          │
                                                          ▼
                                           ┌──────────────────────────┐
                                           │  format_for_verification()│
                                           │  prints to terminal /    │
                                           │  WeChat before storing   │
                                           └──────────────┬───────────┘
                                                          │
                                                  Stage 4: Store
                                                          │
                                                          ▼
                                           ┌──────────────────────────┐
                                           │  graph_rag_store.py (new)│
                                           │  GraphEvent triples      │
                                           │  + E5 embeddings         │
                                           │  (multilingual-e5-base)  │
                                           └──────────────┬───────────┘
                                                          │
                                                          ▼
                                           ┌──────────────────────────┐
                                           │  ChromaDB                │
                                           │  collection:             │
                                           │  graph_rag_events        │
                                           │  [TICKER][TIME] triple   │
                                           └──────────────────────────┘
                                                          │
                  ┌───────────────────────────────────────┘
                  │  _news_cache also updated (backward compat)
                  ▼
   ┌───────────────────────────┐
   │  sentiment_runner         │
   │  (unchanged)              │
   └───────────────────────────┘

✏️ = modified file     (new) = new file
```

---

## Hallucination problem and fix

**Problem**: Traditional RAG mixes news from different events.
LLM sees "SNDK earnings beat" and "SNDK CEO resigned" as a single blob
and may say "CEO resigned *because* earnings beat" — factually wrong.

**Fix — GraphRAG with time-series isolation**:

Each news event is stored as an independent triple:

```
[SNDK][2026-05-01 16:30]  SNDK Q2       → beat expectations  → EPS $23.41 (+60%)   (Reuters)
[SNDK][2026-05-03 09:00]  CEO Goeckeler → resigned           → effective immediately (Bloomberg)
[SNDK][2026-05-07 10:00]  Mizuho        → raised price target→ $1,220 → $1,625       (TipRanks)
```

When injected into the LLM prompt, each event carries its own timestamp
and the prompt contains a hard instruction:

```
⚠️  CRITICAL: Each event below is INDEPENDENT.
⚠️  DO NOT combine or mix information across different events.
⚠️  Each event has its own timestamp and causal context.
```

**Fix — E5 multilingual embedding**:

Original ChromaDB used ONNX MiniLM — low recall for financial text.
Replaced with `intfloat/multilingual-e5-base`:

- Handles Chinese (XHS) + English (Reddit) natively in one model
- Asymmetric retrieval: `"passage: "` prefix for indexing, `"query: "` for search
- Significantly higher RAGAS Context Recall on financial queries
- Falls back to hash embedding if model unavailable (offline mode)

---

## Strategy

Short-sell after post-earnings surge slows down.

```
Wait for earnings surge
  → detect slowdown (dual validation)
  → short in 3 batches
  → wait for pullback
  → cover short in 3 batches
  → take profit
```

---

## Trading pipeline (10 steps)

```
Step 1   Earnings calendar scan
Step 2   Earnings beat check          (>20% EPS beat required)
Step 3   Surge detection              (price +5% post-earnings)
Step 4   Slowdown detection           (dual: price + volume)
Step 5   ReAct self-verification  ←── GraphRAG news context injected here
Step 6   Human-in-the-loop            (WeChat approval required)
Step 7   Short execution              (3 batches)
Step 8   Position monitoring          (stop-loss + trailing)
Step 9   Cover execution              (3 batches)
Step 10  Memory + reflection          (ChromaDB + MEMORY.md)
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Claude (Anthropic API) |
| Embedding | intfloat/multilingual-e5-base (E5) |
| Vector DB | ChromaDB (persistent, cosine similarity) |
| News sources | Reddit · Xiaohongshu (小红书) · DuckDuckGo |
| Broker | IBKR Client Portal API (margin account) |
| Market data | yfinance |
| Notifications | Twilio (WeChat via SMS) |
| Runtime | Python · asyncio · Docker sandbox |

---

## Project structure

```
main.py                     7×24 main loop  ✏️
nodes.py                    LangGraph node definitions
state.py                    AgentState schema

step1_earnings_calendar.py  )
step2_earnings_result.py    )
step3_surge_detect.py       )  trading pipeline steps
step4_slowdown_detect.py    )
step5_react_verify.py       )
step6_notify.py             )
step7_short_sell.py         )
step8_monitor.py            )
step9_cover.py              )
step10_memory.py            )

idle_scheduler.py           background task scheduler  ✏️
news_pipeline.py            full news ingestion pipeline  (new)
news_filter.py              LLM filter: useful / borderline / noise  (new)
reddit_collector.py         Reddit post collector  (new)
xiaohongshu_collector.py    Xiaohongshu scraper  (new)
social_news_collector.py    unified collector wrapper  (new)

graph_rag_store.py          GraphRAG storage — ChromaDB + JSONL  (new)
e5_embedder.py              E5 multilingual embedding wrapper  (new)
chroma_utils.py             ChromaDB helper with embedding fallback  (new)

react_verifier.py           ReAct self-verification  ✏️  (injects news context)
news_sentiment.py           batch sentiment analysis  ✏️
memory_store.py             trade memory ChromaDB layer
memory_updater.py           long-term memory updater
heartbeat.py                WeChat heartbeat + signal buffer
```

---

## How to run

```bash
# Install dependencies
pip install -r requirements.txt
# First run downloads E5 model (~560MB from HuggingFace, cached after that)

# 7×24 continuous loop
python main.py

# Single trading pipeline run (for testing)
python main.py --once

# Run idle tasks once (legacy)
python main.py --idle

# Run news pipeline once: fetch → LLM filter → verify → GraphRAG store
python main.py --pipeline
```

---

## Status

| Module | Status |
|---|---|
| Core trading pipeline (step1–10) | ✅ Complete |
| GraphRAG news memory | ✅ Complete |
| Reddit + XHS collection | ✅ Complete |
| LLM news filter | ✅ Complete |
| E5 multilingual embedding | ✅ Complete |
| Live trading (broker API) | 🔧 Pending |

---

## Author

Yang Liu · github.com/yiyiliu551
