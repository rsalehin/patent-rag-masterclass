"""Sparse retrieval: inverted index, TF-IDF, BM25 (SPEC Part IX).

BM25 stays indispensable for patents: exact technical terminology, chemical identifiers,
component numbers, claim language and rare terms are exactly where lexical matching beats
dense embeddings. We build the inverted index and BM25 from scratch (to *see* the mechanism),
then use the maintained `rank-bm25` for the production path and confirm they agree in ranking.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens, keeping intra-token '.', '/', '-' so patent identifiers
    like 'G06F16/2457', 'ph-value' or '3.5' survive as single tokens."""
    return [t.lower() for t in _TOKEN.findall(text)]


# ---------------------------------------------------------------------------
# Educational: inverted index with positional postings
# ---------------------------------------------------------------------------

@dataclass
class Posting:
    tf: int = 0
    positions: list[int] = field(default_factory=list)


class InvertedIndex:
    """A tiny positional inverted index: term -> {doc_id -> Posting}."""

    def __init__(self) -> None:
        self.postings: dict[str, dict[str, Posting]] = defaultdict(dict)
        self.doc_len: dict[str, int] = {}
        self.docs: list[str] = []

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        self.doc_len[doc_id] = len(toks)
        self.docs.append(doc_id)
        for pos, tok in enumerate(toks):
            p = self.postings[tok].setdefault(doc_id, Posting())
            p.tf += 1
            p.positions.append(pos)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, {}))

    def boolean_and(self, *terms: str) -> set[str]:
        sets = [set(self.postings.get(tokenize(t)[0], {})) for t in terms if tokenize(t)]
        return set.intersection(*sets) if sets else set()

    def tfidf_search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        n = len(self.docs)
        scores: dict[str, float] = defaultdict(float)
        for term in tokenize(query):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log((n + 1) / (self.df(term) + 1)) + 1  # smoothed idf
            for doc_id, p in plist.items():
                tf = 1 + math.log(p.tf)  # sublinear tf
                scores[doc_id] += tf * idf
        return sorted(scores.items(), key=lambda x: -x[1])[:k]


# ---------------------------------------------------------------------------
# Educational: BM25 from scratch
# ---------------------------------------------------------------------------

def bm25_idf(n_docs: int, df: int) -> float:
    r"""Smoothed BM25 IDF: ln(1 + (N - df + 0.5)/(df + 0.5)) — always non-negative."""
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


class BM25:
    r"""BM25 (Okapi) implemented directly from the formula.

    score(q, d) = Σ_t IDF(t) · ( f(t,d)·(k1+1) ) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) )
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.index = InvertedIndex()
        self.avgdl = 0.0

    def fit(self, doc_ids: list[str], texts: list[str]) -> "BM25":
        for did, txt in zip(doc_ids, texts):
            self.index.add(did, txt)
        self.avgdl = sum(self.index.doc_len.values()) / max(1, len(self.index.doc_len))
        return self

    def score(self, query: str, doc_id: str) -> float:
        n = len(self.index.docs)
        dl = self.index.doc_len.get(doc_id, 0)
        s = 0.0
        for term in tokenize(query):
            plist = self.index.postings.get(term)
            if not plist or doc_id not in plist:
                continue
            f = plist[doc_id].tf
            idf = bm25_idf(n, len(plist))
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s += idf * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        cand: set[str] = set()
        for term in tokenize(query):
            cand.update(self.index.postings.get(term, {}))
        scored = [(d, self.score(query, d)) for d in cand]
        return sorted(scored, key=lambda x: -x[1])[:k]


# ---------------------------------------------------------------------------
# Production: rank-bm25
# ---------------------------------------------------------------------------

class BM25Retriever:
    """Production sparse retriever backed by `rank_bm25.BM25Okapi` over the chunk corpus."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunk_ids: list[str] = []
        self._bm25 = None

    def fit(self, chunk_ids: list[str], texts: list[str]) -> "BM25Retriever":
        from rank_bm25 import BM25Okapi

        self.chunk_ids = list(chunk_ids)
        self._corpus_tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._corpus_tokens, k1=self.k1, b=self.b)
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in order]


# ---- artifact stage registration ------------------------------------------

def build_bm25_index() -> BM25Retriever:
    from . import bootstrap as bs

    chunks = bs.ensure("chunks")
    return BM25Retriever().fit([c.chunk_id for c in chunks], [c.for_index() for c in chunks])


def _register():
    from . import bootstrap as bs

    bs.register_stage("bm25_index", ("chunks",), build_bm25_index)


_register()
