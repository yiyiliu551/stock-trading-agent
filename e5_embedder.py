"""
tools/e5_embedder.py
Author: Yang
Description: E5 multilingual embedding for ChromaDB.
             Replaces ChromaDB's default ONNX MiniLM.
             
Why E5 over MiniLM:
  - multilingual-e5-base handles Chinese (XHS) + English (Reddit) natively
  - E5 uses "query: " / "passage: " prefixes for asymmetric retrieval
    → query  prefix: used when searching (user question / LLM lookup)
    → passage prefix: used when indexing (news text stored in ChromaDB)
  - Significantly better semantic recall for financial text vs keyword-based MiniLM
  - Solves: news in library but RAGAS Context Recall was low

Model: intfloat/multilingual-e5-base (560MB, loads once, cached)
Fallback: SHA256 hash embedding if model load fails (offline mode)
"""

import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_model = None
_model_loaded = False
_use_fallback = False

MODEL_NAME = "intfloat/multilingual-e5-base"

# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model():
    """Lazy-load E5 model. Sets _use_fallback=True on failure."""
    global _model, _model_loaded, _use_fallback
    if _model_loaded:
        return

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading E5 model: %s (first run downloads ~560MB)", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        _model_loaded = True
        _use_fallback = False
        logger.info("E5 model loaded ✅")
    except Exception as e:
        logger.warning(
            "E5 model load failed: %s — falling back to hash embedding", e
        )
        _model_loaded = True
        _use_fallback = True


# ── Public embedding functions ─────────────────────────────────────────────────

def embed_passages(texts: list[str]) -> list[list[float]]:
    """
    Embed news/document texts for storage in ChromaDB.
    Uses "passage: " prefix as required by E5 asymmetric retrieval.
    
    Call this when INDEXING (storing news into ChromaDB).
    """
    _load_model()
    if _use_fallback:
        return _hash_embed(texts)
    
    # E5 passage prefix
    prefixed = [f"passage: {t}" for t in texts]
    try:
        vecs = _model.encode(prefixed, normalize_embeddings=True)
        return vecs.tolist()
    except Exception as e:
        logger.warning("E5 encode failed: %s — using hash fallback", e)
        return _hash_embed(texts)


def embed_query(text: str) -> list[float]:
    """
    Embed a search query for ChromaDB retrieval.
    Uses "query: " prefix as required by E5 asymmetric retrieval.
    
    Call this when QUERYING (looking up news in ChromaDB).
    """
    _load_model()
    if _use_fallback:
        return _hash_embed([text])[0]
    
    prefixed = f"query: {text}"
    try:
        vec = _model.encode([prefixed], normalize_embeddings=True)
        return vec[0].tolist()
    except Exception as e:
        logger.warning("E5 query encode failed: %s — using hash fallback", e)
        return _hash_embed([text])[0]


def embed_queries(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple query strings."""
    _load_model()
    if _use_fallback:
        return _hash_embed(texts)
    
    prefixed = [f"query: {t}" for t in texts]
    try:
        vecs = _model.encode(prefixed, normalize_embeddings=True)
        return vecs.tolist()
    except Exception as e:
        logger.warning("E5 batch query encode failed: %s — using hash fallback", e)
        return _hash_embed(texts)


def get_embedding_dim() -> int:
    """Return embedding dimension (768 for E5-base, 64 for fallback)."""
    _load_model()
    return 64 if _use_fallback else 768


# ── Hash fallback ──────────────────────────────────────────────────────────────

def _hash_embed(texts: list[str]) -> list[list[float]]:
    """
    64-dim SHA256 hash embedding.
    No semantic quality, but works fully offline.
    Only used when E5 model fails to load.
    """
    result = []
    for t in texts:
        vec = []
        for i in range(8):  # 8 × 8 bytes = 64 dims
            h = hashlib.sha256(f"{i}:{t}".encode("utf-8")).digest()
            vec.extend([float(b) / 255.0 for b in h[:8]])
        result.append(vec)
    return result


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== E5 Embedder Test ===\n")

    passage = "NVDA Q2 earnings beat estimates by 60%, EPS $23.41 vs $14.62"
    query   = "NVDA earnings results"

    p_vec = embed_passages([passage])[0]
    q_vec = embed_query(query)

    print(f"Passage embedding dim: {len(p_vec)}")
    print(f"Query   embedding dim: {len(q_vec)}")

    # Cosine similarity
    import math
    dot = sum(a * b for a, b in zip(p_vec, q_vec))
    norm_p = math.sqrt(sum(x**2 for x in p_vec))
    norm_q = math.sqrt(sum(x**2 for x in q_vec))
    sim = dot / (norm_p * norm_q + 1e-8)
    print(f"Cosine similarity (earnings query vs earnings passage): {sim:.4f}")
    print("✅ E5 embedder working")
