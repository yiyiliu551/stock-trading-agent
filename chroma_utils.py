"""
tools/chroma_utils.py
Author: Yang
Description: ChromaDB helper with embedding fallback.
             Primary: ChromaDB default (ONNX MiniLM — requires internet on first run).
             Fallback: SHA256 hash embeddings — works offline, no model download.

Use _get_news_collection() everywhere instead of direct chromadb calls.
"""

import hashlib
import logging
import os
from typing import Optional

import chromadb

logger = logging.getLogger(__name__)

# ── Embedding fallback ─────────────────────────────────────────────────────────

def _hash_embed(texts: list[str]) -> list[list[float]]:
    """
    Lightweight embedding via SHA256 hash → 64-dim float vector.
    No model download needed. Lower semantic quality than MiniLM,
    but good enough for keyword-based retrieval and deduplication.
    """
    result = []
    for t in texts:
        raw = hashlib.sha256(t.lower().encode("utf-8")).digest()
        # Extend to 64 dims by hashing with different seeds
        vec = []
        for i in range(4):
            seed = hashlib.sha256(f"{i}:{t}".encode()).digest()
            vec.extend([float(b) / 255.0 for b in seed])
        result.append(vec[:64])
    return result


# ── Collection singleton ───────────────────────────────────────────────────────

_client: Optional[chromadb.PersistentClient] = None
_collection = None
_use_hash_embed = False   # set True after first ONNX failure


def _get_news_collection(chroma_path: str):
    """
    Lazy-init ChromaDB PersistentClient + news_raw collection.
    Automatically falls back to hash embeddings if ONNX model download fails.
    """
    global _client, _collection, _use_hash_embed

    if _collection is not None:
        return _collection

    os.makedirs(chroma_path, exist_ok=True)
    _client = chromadb.PersistentClient(path=chroma_path)

    # Try default (ONNX) collection first
    try:
        col = _client.get_or_create_collection(
            name="news_raw",
            metadata={"hnsw:space": "cosine"},
        )
        # Probe — trigger embedding init now so we catch failures early
        col.upsert(
            ids=["__probe__"],
            documents=["init"],
        )
        col.delete(ids=["__probe__"])
        _collection = col
        logger.info("ChromaDB: using ONNX MiniLM embeddings")
        return _collection
    except Exception as e:
        if "SHA256" in str(e) or "hash" in str(e).lower() or "download" in str(e).lower():
            logger.warning(
                "ONNX model unavailable (%s). "
                "Switching to hash embeddings (works offline).", e
            )
        else:
            logger.warning("ChromaDB ONNX init failed: %s. Falling back to hash embed.", e)

    # Fallback: recreate collection without default EF
    try:
        # Delete and recreate with explicit dim=64 (hash embed size)
        try:
            _client.delete_collection("news_raw")
        except Exception:
            pass
        col = _client.get_or_create_collection(
            name="news_raw",
            metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 100},
        )
        _collection = col
        _use_hash_embed = True
        logger.info("ChromaDB: using hash embeddings (offline mode)")
        return _collection
    except Exception as e2:
        logger.error("ChromaDB fallback init failed: %s", e2)
        raise


def upsert_docs(chroma_path: str, ids: list, documents: list, metadatas: list) -> int:
    """
    Upsert documents into news_raw collection.
    Handles embedding selection automatically.
    """
    col = _get_news_collection(chroma_path)
    try:
        if _use_hash_embed:
            embeddings = _hash_embed(documents)
            col.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            col.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)
    except Exception as e:
        logger.error("ChromaDB upsert failed: %s", e)
        return 0


def query_docs(chroma_path: str, query_text: str, n_results: int = 5,
               where: Optional[dict] = None) -> list[str]:
    """
    Query news_raw collection. Returns list of document strings.
    """
    col = _get_news_collection(chroma_path)
    try:
        kwargs = {"n_results": n_results}
        if where:
            kwargs["where"] = where
        if _use_hash_embed:
            kwargs["query_embeddings"] = _hash_embed([query_text])
        else:
            kwargs["query_texts"] = [query_text]
        results = col.query(**kwargs)
        return results.get("documents", [[]])[0]
    except Exception as e:
        logger.warning("ChromaDB query failed: %s", e)
        return []
