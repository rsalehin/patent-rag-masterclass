"""Late interaction, hybrid fusion, reranking, filtering (SPEC Parts XIII–XVI).

Lexical (BM25) and dense retrieval fail on different queries; fusing their *rankings* with RRF
is robust and calibration-free. A cross-encoder then rescoring the shortlist gives the biggest
single-stage quality jump. Metadata filters constrain the candidate set by real biblio fields.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
                           ) -> list[tuple[str, float]]:
    r"""Fuse ranked id-lists: score(d) = Σ_r w_r / (k + rank_r(d)), rank 1-based.

    Rank-based (not score-based) fusion needs no score normalization across retrievers.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def fuse_named(named_rankings: dict[str, list[str]], k: int = 60) -> list[tuple[str, float]]:
    return reciprocal_rank_fusion(list(named_rankings.values()), k=k)


# ---------------------------------------------------------------------------
# ColBERT-style late interaction (educational)
# ---------------------------------------------------------------------------

def token_embeddings(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Per-token contextual embeddings (n_tokens × d), L2-normalized — the ColBERT primitive."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    global _tok, _mod
    try:
        _tok, _mod  # type: ignore[name-defined]
    except NameError:
        _tok = AutoTokenizer.from_pretrained(model_name)
        _mod = AutoModel.from_pretrained(model_name).eval()
    enc = _tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = _mod(**enc).last_hidden_state[0]  # (n_tok, d)
    mask = enc["attention_mask"][0].bool()
    v = out[mask].numpy()
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def maxsim(query_tok: np.ndarray, doc_tok: np.ndarray) -> float:
    r"""ColBERT MaxSim late interaction: Σ_{q} max_{d} cos(q, d).

    Each query token contributes its best match against any document token, preserving
    token-level detail a single pooled vector discards.
    """
    sim = query_tok @ doc_tok.T  # (nq, nd), already normalized
    return float(sim.max(axis=1).sum())


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Rerank a shortlist by joint (query, passage) scoring — O(candidates) model calls."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._ce = None

    def _load(self):
        if self._ce is None:
            from sentence_transformers import CrossEncoder

            self._ce = CrossEncoder(self.model_name, device="cpu")
        return self._ce

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int | None = None
               ) -> list[tuple[str, float]]:
        """candidates: list of (chunk_id, text). Returns (chunk_id, score) sorted desc."""
        if not candidates:
            return []
        ce = self._load()
        scores = ce.predict([(query, text) for _, text in candidates])
        ranked = sorted(((cid, float(s)) for (cid, _), s in zip(candidates, scores)), key=lambda x: -x[1])
        return ranked[:top_k] if top_k else ranked


# ---------------------------------------------------------------------------
# Metadata filtering (pre- vs post-filter)
# ---------------------------------------------------------------------------

def make_filter(cpc_section: str | None = None, language: str | None = None,
                section_kind: str | None = None, min_pub_year: int | None = None):
    """Build a predicate over a (chunk, document) pair for metadata filtering (Ch12/Part XVI)."""

    def pred(chunk, doc) -> bool:  # noqa: ANN001
        if cpc_section and cpc_section not in doc.cpc_sections:
            return False
        if language and doc.language != language:
            return False
        if section_kind and getattr(chunk.section_kind, "value", chunk.section_kind) != section_kind:
            return False
        if min_pub_year and (not doc.publication_date or doc.publication_date.year < min_pub_year):
            return False
        return True

    return pred


def post_filter(results: list[tuple[str, float]], chunks_by_id: dict, docs_by_id: dict, pred,
                k: int) -> list[tuple[str, float]]:
    """ANN → filter: retrieve first, drop non-matching (may under-fill k if selective)."""
    out = []
    for cid, score in results:
        c = chunks_by_id[cid]
        if pred(c, docs_by_id[c.document_id]):
            out.append((cid, score))
        if len(out) >= k:
            break
    return out


def pre_filter_ids(chunks: list, docs_by_id: dict, pred) -> list[str]:
    """filter → ANN: restrict the candidate universe *before* searching."""
    return [c.chunk_id for c in chunks if pred(c, docs_by_id[c.document_id])]
