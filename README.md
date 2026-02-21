 # Design Document ： 
 # https://github.com/yiyiliu551/stock-trading-agent/blob/main/stock_agent_v8.pdf

# I gave an LLM a brain. Then I built it hands（tools/Skills） --AI trading agent 
#  I initially developed an AI agent using LangGraph for task orchestration. 

# 🧠 The brain: Claude (LLM) Reasoning. Validating. Deciding when to act — and when not to.

# 🤝 The hands: Custom-designed tools Not borrowed. Built specifically for this system: surge detection · volatility stop loss · batch execution · trade memory + reflection

# ⚙️ LangGraph turns the entire trading workflow into a pluggable directed graph: each step is implemented as an independent node, transitions between nodes are controlled by conditional edges, failures trigger an early stop (abort_reason), and successful executions complete the loop and return control to the scheduler. 

#LangGraph
#Scheduler
#↓
#Event Detection
#↓
#Signal Filtering
#↓
#AI Validation
#↓
#Human-in-the-Loop
#↓
#Execution Engine
#↓
#Risk Monitor
#↓
#Memory + Reflection

# stock-trading-agent

# Stock Trading Agent — Design Document

This repository contains the full system design for an AI-powered 
stock trading agent focused on US tech stocks around earnings events.

## Strategy
Short selling after post-earnings surge slows down.  
Wait for surge → detect slowdown (dual validation) → short in 3 batches 
→ wait for pullback → cover short in 3 batches → take profit.

## Architecture
- **Orchestration**: LangGraph (10-step flow)
- **AI Analysis**: Claude (ReAct self-verification, fixed 2 iterations)
- **Memory**: Two-tier — session log + long-term MEMORY.md (RAG retrieval)
- **Security**: Docker sandbox + Guardrails + MFA (SMS confirmation)
- **Broker**: IBKR Client Portal API (Margin account)

## Tech Stack
Python · LangGraph · Anthropic API · yfinance · ib_insync · 
Twilio · ChromaDB · asyncio

## Status
Design complete. Code implementation in progress.

## Author
Yang Liu · github.com/yiyiliu551
