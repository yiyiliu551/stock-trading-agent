"""
tools/memory_store.py
Author: Yang
Description: Persistent memory layer — ChromaDB for vector retrieval,
             MEMORY.md for human-readable append-only log.
             No AI logic. Pure storage I/O.
"""

import json
import logging
from datetime import datetime
from typing import Optional

import chromadb

from config import MEMORY_FILE, CHROMA_PATH

logger = logging.getLogger(__name__)

# ── ChromaDB singleton ─────────────────────────────────────────────────────────
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection     = None


def _get_collection():
    """Lazy-initialise ChromaDB client and collection."""
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection    = _chroma_client.get_or_create_collection("trade_memory")
    return _collection


# ── Trade storage ──────────────────────────────────────────────────────────────

def save_trade_to_chroma(trade: dict) -> bool:
    """
    Upsert one trade record into ChromaDB.
    Uses "{ticker}_{timestamp}" as the document ID.
    Returns True on success.
    """
    trade.setdefault("timestamp", datetime.now().isoformat())
    doc_id = f"{trade.get('ticker', 'UNKNOWN')}_{trade['timestamp']}"
    try:
        _get_collection().upsert(
            ids=[doc_id],
            documents=[json.dumps(trade)],
            metadatas=[{
                "ticker":      str(trade.get("ticker", "")),
                "profit_loss": float(trade.get("profit_loss", 0)),
            }],
        )
        logger.info("ChromaDB upsert: %s", doc_id)
        return True
    except Exception as e:
        logger.error("ChromaDB upsert failed: %s", e)
        return False


def append_trade_to_markdown(trade: dict, reflection: str) -> bool:
    """
    Append a formatted trade record + AI reflection to MEMORY.md.
    Returns True on success.
    """
    trade.setdefault("timestamp", datetime.now().isoformat())
    line_sep = "\n---\n"
    entry = (
        f"{line_sep}"
        f"## {trade.get('ticker', '?')} | {trade['timestamp']}\n"
        f"- Short: ${trade.get('short_price', 0):.2f} | "
        f"Cover: ${trade.get('cover_price', 0):.2f} | "
        f"Shares: {trade.get('total_shares', 0)} | "
        f"P&L: ${trade.get('profit_loss', 0):.2f}\n"
        f"- Outcome: {trade.get('outcome', '?')} | "
        f"Days held: {trade.get('days_held', 0):.1f}\n\n"
        f"{reflection}\n"
    )
    try:
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
        logger.info("MEMORY.md appended: %s", trade.get("ticker"))
        return True
    except Exception as e:
        logger.error("MEMORY.md write failed: %s", e)
        return False


# ── Sentiment snapshot ─────────────────────────────────────────────────────────

def append_sentiment_snapshot(sentiment_map: dict) -> bool:
    """
    Append today's news sentiment table to MEMORY.md.
    sentiment_map: {ticker: {sentiment, score, summary}}
    """
    if not sentiment_map:
        return False
    today  = datetime.now().strftime("%Y-%m-%d")
    header = (
        f"\n## Sentiment Snapshot — {today}\n"
        f"*Recorded: {datetime.now().strftime('%H:%M')}*\n\n"
        "| Ticker | Sentiment | Score | Summary |\n"
        "|--------|-----------|-------|---------|\n"
    )
    rows = []
    for ticker in sorted(sentiment_map.keys()):
        v       = sentiment_map[ticker]
        rows.append(
            f"| **{ticker}** | {v.get('sentiment', '?')} | "
            f"`{v.get('score', 0):+.2f}` | {v.get('summary', '')} |"
        )
    try:
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(header + "\n".join(rows) + "\n")
        return True
    except Exception as e:
        logger.error("MEMORY.md sentiment append failed: %s", e)
        return False


# ── Retrieval ──────────────────────────────────────────────────────────────────

def query_similar_trades(ticker: str, n_results: int = 3) -> list[dict]:
    """
    Retrieve the n most similar past trade records from ChromaDB.
    Returns a list of trade dicts.
    """
    try:
        results = _get_collection().query(
            query_texts=[ticker],
            n_results=n_results,
        )
        docs = results.get("documents", [[]])[0]
        return [json.loads(d) for d in docs]
    except Exception as e:
        logger.warning("ChromaDB query failed: %s", e)
        return []
