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
