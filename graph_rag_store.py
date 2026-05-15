"""
tools/graph_rag_store.py
Author: Yang
Description: Persistent GraphRAG storage combining:
  1. ChromaDB (E5 embeddings) — semantic similarity search
  2. In-memory knowledge graph — entity-relation-time triples
  3. JSONL append log — human-readable audit trail

Solves TWO problems:
  Problem A: Hallucination (事件混淆)
    LLM is told "SNDK Q2 earnings beat" and "SNDK CEO resigned".
    Without isolation, it may say "CEO resigned because earnings beat".
    Fix: each event is stored as an INDEPENDENT triple with timestamp.
         format_for_llm() injects hard separators + anti-hallucination warning.

  Problem B: Low recall (新闻在库里但RAG检索不到)
    Fix: E5 multilingual embeddings with proper passage/query prefixes.
         Financial chunk prefix: "[TICKER][TIMESTAMP] " ensures
         metadata is embedded INTO the vector, not just as a filter.

Storage schema per event:
  ChromaDB document : "[TICKER][TIMESTAMP] SUBJECT PREDICATE OBJECT"
  ChromaDB metadata : {ticker, timestamp, event_type, impact, source, subject, predicate, object}
  JSONL line        : full event JSON

Time-series retrieval:
  query_events(ticker, start_time, end_time) → sorted list
  Each event has its own timestamp — LLM sees them independently.
"""

import json
import logging
import os
import hashlib
from datetime import datetime
from typing import Optional

import chromadb

from config import CHROMA_PATH
from e5_embedder import embed_passages, embed_query

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
GRAPH_COLLECTION  = "graph_rag_events"
EVENTS_LOG_PATH   = os.path.join(os.path.dirname(CHROMA_PATH), "graph_events.jsonl")

# ── ChromaDB singleton ─────────────────────────────────────────────────────────
_client: Optional[chromadb.PersistentClient] = None
_col    = None


def _get_col():
    """Lazy-init ChromaDB collection for graph events (E5 dim=768)."""
    global _client, _col
    if _col is not None:
        return _col

    os.makedirs(CHROMA_PATH, exist_ok=True)
    _client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Create collection with cosine distance (E5 vectors are normalized)
    _col = _client.get_or_create_collection(
        name=GRAPH_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("GraphRAG ChromaDB collection ready: %s", GRAPH_COLLECTION)
    return _col


# ── Event model ────────────────────────────────────────────────────────────────

class GraphEvent:
    """
    One isolated knowledge-graph event triple with timestamp.
    
    Example:
        subject   = "SNDK Q2 Earnings"
        predicate = "beat expectations"
        object_   = "EPS $23.41 vs $14.62 (+60%)"
        → stored as: "[SNDK][2026-05-01 16:30] SNDK Q2 Earnings beat expectations EPS $23.41..."
    """
    def __init__(
        self,
        ticker:     str,
        timestamp:  str,   # ISO format: "2026-05-01 16:30:00"
        event_type: str,   # earnings / management / analyst / product / macro / social
        subject:    str,
        predicate:  str,
        object_:    str,
        impact:     str,   # positive / negative / neutral
        source:     str,   # reddit / xiaohongshu / duckduckgo / Reuters / etc.
        raw_text:   str,   # original post/headline text
    ):
        self.ticker     = ticker.upper()
        self.timestamp  = timestamp
        self.event_type = event_type
        self.subject    = subject
        self.predicate  = predicate
        self.object_    = object_
        self.impact     = impact
        self.source     = source
        self.raw_text   = raw_text
        # Stable ID: hash of ticker+timestamp+subject (deduplication)
        raw = f"{self.ticker}::{timestamp}::{subject}::{predicate}"
        self.event_id = hashlib.md5(raw.encode()).hexdigest()[:16]

    def to_document(self) -> str:
        """
        Text stored in ChromaDB.
        Prefix [TICKER][TIME] embeds metadata INTO the vector — 
        this improves E5 recall vs relying only on metadata filters.
        """
        return (
            f"[{self.ticker}][{self.timestamp}] "
            f"{self.subject} {self.predicate} {self.object_}. "
            f"Impact: {self.impact}. Source: {self.source}."
        )

    def to_metadata(self) -> dict:
        """Stored as ChromaDB metadata (for filtering by ticker/time/type)."""
        return {
            "ticker":     self.ticker,
            "timestamp":  self.timestamp,
            "event_type": self.event_type,
            "impact":     self.impact,
            "source":     self.source,
            "subject":    self.subject[:100],
            "predicate":  self.predicate[:100],
            "object_":    self.object_[:200],
        }

    def to_dict(self) -> dict:
        """Full JSON representation for JSONL log."""
        return {
            "event_id":   self.event_id,
            "ticker":     self.ticker,
            "timestamp":  self.timestamp,
            "event_type": self.event_type,
            "subject":    self.subject,
            "predicate":  self.predicate,
            "object_":    self.object_,
            "impact":     self.impact,
            "source":     self.source,
            "raw_text":   self.raw_text[:500],
            "stored_at":  datetime.utcnow().isoformat(),
        }


# ── Storage ────────────────────────────────────────────────────────────────────

def store_event(event: GraphEvent) -> bool:
    """
    Store one GraphEvent into ChromaDB + JSONL log.
    Uses E5 passage embedding for the document vector.
    Returns True on success.
    """
    try:
        col = _get_col()
        doc = event.to_document()

        # E5 passage embedding
        embedding = embed_passages([doc])[0]

        col.upsert(
            ids=[event.event_id],
            documents=[doc],
            embeddings=[embedding],
            metadatas=[event.to_metadata()],
        )

        # Append to JSONL audit log
        _append_to_log(event)

        logger.debug("GraphRAG stored: [%s][%s] %s %s",
                     event.ticker, event.timestamp, event.subject, event.predicate)
        return True
    except Exception as e:
        logger.error("GraphRAG store failed: %s", e)
        return False


def store_events(events: list[GraphEvent]) -> int:
    """Batch store. Returns count of successfully stored events."""
    if not events:
        return 0
    try:
        col   = _get_col()
        docs  = [e.to_document() for e in events]
        embs  = embed_passages(docs)   # one E5 call for whole batch
        ids   = [e.event_id for e in events]
        metas = [e.to_metadata() for e in events]

        col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

        for e in events:
            _append_to_log(e)

        logger.info("GraphRAG batch stored: %d events", len(events))
        return len(events)
    except Exception as e:
        logger.error("GraphRAG batch store failed: %s", e)
        return 0


def _append_to_log(event: GraphEvent):
    """Append event to JSONL audit log."""
    try:
        os.makedirs(os.path.dirname(EVENTS_LOG_PATH), exist_ok=True)
        with open(EVENTS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("JSONL log write failed: %s", e)


# ── Retrieval ──────────────────────────────────────────────────────────────────

def query_events(
    ticker: str,
    query_text: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time:   Optional[str] = None,
    event_type: Optional[str] = None,
    n_results:  int = 10,
) -> list[dict]:
    """
    Retrieve graph events for a ticker.
    
    Args:
        ticker:     stock symbol
        query_text: semantic search query (E5 embedding used)
                    e.g. "earnings beat" or "CEO management change"
                    If None, uses ticker as query.
        start_time: ISO datetime filter (inclusive)
        end_time:   ISO datetime filter (inclusive)
        event_type: filter by type (earnings/management/analyst/etc.)
        n_results:  max events to return

    Returns:
        List of event dicts sorted chronologically (oldest first).
        Each dict has: ticker, timestamp, event_type, subject, predicate,
                       object_, impact, source
    """
    try:
        col = _get_col()

        # Build ChromaDB where filter
        where_clauses = [{"ticker": {"$eq": ticker.upper()}}]
        if event_type:
            where_clauses.append({"event_type": {"$eq": event_type}})
        where = where_clauses[0] if len(where_clauses) == 1 else {"$and": where_clauses}

        # E5 query embedding
        q = query_text or f"{ticker} stock news"
        q_embedding = embed_query(q)

        results = col.query(
            query_embeddings=[q_embedding],
            n_results=min(n_results, 50),
            where=where,
        )

        docs   = results.get("documents", [[]])[0]
        metas  = results.get("metadatas", [[]])[0]
        events = []

        for meta in metas:
            # Time range filter (ChromaDB doesn't support range on string timestamps natively)
            ts = meta.get("timestamp", "")
            if start_time and ts < start_time:
                continue
            if end_time and ts > end_time:
                continue
            events.append(dict(meta))

        # Sort chronologically — critical for anti-hallucination
        events.sort(key=lambda x: x.get("timestamp", ""))
        return events[:n_results]

    except Exception as e:
        logger.warning("GraphRAG query failed for %s: %s", ticker, e)
        return []


# ── LLM formatting ─────────────────────────────────────────────────────────────

def format_events_for_llm(events: list[dict]) -> str:
    """
    Format graph events for injection into LLM prompt.
    
    Design principles (anti-hallucination):
      1. Each event is numbered and visually separated
      2. Timestamp is shown prominently
      3. Hard warning: "DO NOT mix events from different times"
      4. Triple format makes subject-predicate-object explicit
         → LLM cannot confuse "CEO resigned" with "earnings beat"
    """
    if not events:
        return "No knowledge graph events found for this ticker."

    lines = [
        "=" * 60,
        "📊 KNOWLEDGE GRAPH EVENTS — CHRONOLOGICAL TIME SERIES",
        "=" * 60,
        "⚠️  CRITICAL: Each event below is INDEPENDENT.",
        "⚠️  DO NOT combine or mix information across different events.",
        "⚠️  Each event has its own timestamp and causal context.",
        "",
    ]

    for i, e in enumerate(events, 1):
        lines += [
            f"── Event #{i} ──────────────────────────────────────────",
            f"  ⏰ Time    : {e.get('timestamp', 'unknown')}",
            f"  📈 Ticker  : {e.get('ticker', '?')}",
            f"  📌 Type    : {e.get('event_type', '?')}",
            f"  🔗 Triple  : {e.get('subject','?')}",
            f"              → {e.get('predicate','?')}",
            f"              → {e.get('object_','?')}",
            f"  💹 Impact  : {e.get('impact', '?')}",
            f"  📰 Source  : {e.get('source', '?')}",
            "",
        ]

    lines += [
        "=" * 60,
        "⚠️  Reminder: treat each event as isolated fact.",
        "   Only connect events if there is explicit causal evidence.",
        "=" * 60,
    ]

    return "\n".join(lines)


# ── Extract events from filter output ─────────────────────────────────────────

def extract_events_from_filter(filter_result: dict) -> list[GraphEvent]:
    """
    Convert news_filter.py output → list of GraphEvents ready for storage.
    Only converts 'useful' and 'borderline' posts.
    
    Extracts structured triples from LLM-identified facts.
    """
    ticker    = filter_result.get("ticker", "UNKNOWN")
    timestamp = filter_result.get("timestamp", datetime.utcnow().isoformat())
    events    = []

    for post in filter_result.get("useful", []) + filter_result.get("borderline", []):
        facts     = post.get("facts", [])
        source    = post.get("source", "unknown")
        title     = post.get("title", "")
        text      = post.get("text", "")
        event_type = post.get("event_type", "social")
        impact    = post.get("sentiment", "neutral")
        if impact == "n/a":
            impact = "neutral"

        if facts:
            # Create one event per extracted fact (prevents fact-mixing)
            for fact in facts[:3]:   # max 3 facts per post
                e = GraphEvent(
                    ticker     = ticker,
                    timestamp  = post.get("filtered_at", timestamp)[:19],
                    event_type = event_type,
                    subject    = ticker,
                    predicate  = fact,
                    object_    = title[:150],
                    impact     = impact,
                    source     = source,
                    raw_text   = f"{title}\n{text}"[:500],
                )
                events.append(e)
        else:
            # No structured facts — store the title as a generic event
            e = GraphEvent(
                ticker     = ticker,
                timestamp  = post.get("filtered_at", timestamp)[:19],
                event_type = event_type,
                subject    = ticker,
                predicate  = "mentioned",
                object_    = title[:200],
                impact     = impact,
                source     = source,
                raw_text   = f"{title}\n{text}"[:500],
            )
            events.append(e)

    return events


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, types

    logging.basicConfig(level=logging.INFO)

    # Mock config
    cfg = types.ModuleType("config")
    cfg.CHROMA_PATH = "/tmp/test_graph_rag"
    sys.modules["config"] = cfg
    os.makedirs("/tmp/test_graph_rag", exist_ok=True)

    print("=== GraphRAG Store Test ===\n")

    # Create test events (the classic hallucination scenario)
    events = [
        GraphEvent("SNDK", "2026-05-01 16:30:00", "earnings",
                   "SNDK Q2 Earnings", "beat expectations",
                   "EPS $23.41 vs $14.62 (+60%)", "positive", "Reuters",
                   "SanDisk Q2 EPS beat by 60%"),
        GraphEvent("SNDK", "2026-05-03 09:00:00", "management",
                   "CEO David Goeckeler", "resigned from",
                   "SanDisk effective immediately", "negative", "Bloomberg",
                   "SanDisk CEO resigned effective immediately"),
        GraphEvent("SNDK", "2026-05-07 10:00:00", "analyst",
                   "Mizuho analyst", "raised price target",
                   "$1,220 → $1,625 (+33%)", "positive", "TipRanks",
                   "Mizuho raised SNDK target to $1,625"),
    ]

    n = store_events(events)
    print(f"Stored {n} events\n")

    # Query and format
    retrieved = query_events("SNDK", query_text="SNDK earnings CEO management")
    print(format_events_for_llm(retrieved))
    print(f"\n✅ Retrieved {len(retrieved)} events from ChromaDB")
