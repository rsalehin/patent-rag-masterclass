"""Dense retrieval, ANN/HNSW, and vector quantization (SPEC Parts X–XII).

Embeddings map text → a dense vector so semantically related passages sit close even without
shared words. We build brute-force (exact) cosine retrieval first, then a real FAISS HNSW index
(measuring the recall/latency trade-off), then product- and scalar-quantized indexes (measuring
the accuracy/memory trade-off). Inference is forced to CPU for determinism and portability.
"""
from __future__ import annotations

import numpy as np

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = None


def get_embedder(model_name: str = _EMBED_MODEL):
    """Lazily load a CPU-pinned SentenceTransformer (cached module-singleton)."""
    global _embedder
    if _embedder is None or getattr(_embedder, "_pr_name", None) != model_name:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(model_name, device="cpu")
        m._pr_name = model_name
        _embedder = m
    return _embedder


def embed(texts: list[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    """Encode texts to a float32 matrix (L2-normalized rows by default → dot == cosine)."""
    m = get_embedder()
    v = m.encode(list(texts), batch_size=batch_size, normalize_embeddings=normalize,
                 convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between row-sets a (n×d) and b (m×d)."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


# ---------------------------------------------------------------------------
# Brute-force (exact) dense retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Exact cosine retrieval via a single matrix multiply against normalized embeddings."""

    def __init__(self, chunk_ids: list[str], matrix: np.ndarray) -> None:
        self.chunk_ids = list(chunk_ids)
        self.matrix = matrix.astype(np.float32)

    def search(self, query: str | np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        q = embed([query])[0] if isinstance(query, str) else query
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        return [(self.chunk_ids[i], float(sims[i])) for i in order]

    def exact_topk(self, q: np.ndarray, k: int) -> list[int]:
        return list(np.argsort(-(self.matrix @ q))[:k])


# ---------------------------------------------------------------------------
# HNSW (FAISS)
# ---------------------------------------------------------------------------

class HNSWIndex:
    """FAISS `IndexHNSWFlat`. On L2-normalized vectors, L2-NN == cosine-NN."""

    def __init__(self, dim: int, M: int = 32, ef_construction: int = 200) -> None:
        import faiss

        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.chunk_ids: list[str] = []

    def add(self, chunk_ids: list[str], matrix: np.ndarray) -> "HNSWIndex":
        self.chunk_ids = list(chunk_ids)
        self.index.add(matrix.astype(np.float32))
        return self

    def set_ef_search(self, ef: int) -> None:
        self.index.hnsw.efSearch = ef

    def search(self, query: str | np.ndarray, k: int = 10, ef_search: int | None = None):
        if ef_search is not None:
            self.set_ef_search(ef_search)
        q = embed([query]) if isinstance(query, str) else query.reshape(1, -1)
        dist, idx = self.index.search(q.astype(np.float32), k)
        out = []
        for j, i in enumerate(idx[0]):
            if i == -1:
                continue
            out.append((self.chunk_ids[i], float(dist[0][j])))  # smaller L2 = closer
        return out

    # FAISS objects are not natively picklable; serialize via faiss for artifact caching.
    def __getstate__(self):
        import faiss

        return {"blob": bytes(faiss.serialize_index(self.index)), "chunk_ids": self.chunk_ids}

    def __setstate__(self, state):
        import faiss
        import numpy as _np

        self.index = faiss.deserialize_index(_np.frombuffer(state["blob"], dtype=_np.uint8))
        self.chunk_ids = state["chunk_ids"]


def recall_at_k(approx_ids: list, exact_ids: list) -> float:
    """Fraction of the exact top-k that the approximate search also returned."""
    if not exact_ids:
        return 1.0
    return len(set(approx_ids) & set(exact_ids)) / len(exact_ids)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def memory_bytes(n_vectors: int, dim: int, dtype_bytes: int) -> int:
    return n_vectors * dim * dtype_bytes


def memory_report(n_vectors: int = 100_000_000, dim: int = 384) -> dict[str, float]:
    """Memory (GB) to store n vectors of `dim` under several encodings."""
    gb = lambda b: b / 1e9  # noqa: E731
    return {
        "float32_GB": gb(memory_bytes(n_vectors, dim, 4)),
        "float16_GB": gb(memory_bytes(n_vectors, dim, 2)),
        "int8_GB": gb(memory_bytes(n_vectors, dim, 1)),
        "pq_m48_GB": gb(n_vectors * 48),  # 48 bytes/vector (m=48 subquantizers, 8 bits)
    }


class PQIndex:
    """FAISS product-quantization index (IndexPQ). `m` subquantizers, `nbits` each."""

    def __init__(self, dim: int, m: int = 48, nbits: int = 8) -> None:
        import faiss

        self.index = faiss.IndexPQ(dim, m, nbits)
        self.chunk_ids: list[str] = []
        self.bytes_per_vector = m  # nbits=8 → 1 byte per subquantizer

    def train_add(self, chunk_ids: list[str], matrix: np.ndarray) -> "PQIndex":
        self.chunk_ids = list(chunk_ids)
        m = matrix.astype(np.float32)
        self.index.train(m)
        self.index.add(m)
        return self

    def search(self, q: np.ndarray, k: int = 10) -> list[int]:
        _, idx = self.index.search(q.astype(np.float32).reshape(1, -1), k)
        return [i for i in idx[0] if i != -1]


class ScalarInt8Index:
    """FAISS 8-bit scalar-quantized flat index (4× smaller than float32, near-exact)."""

    def __init__(self, dim: int) -> None:
        import faiss

        self.index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit)
        self.chunk_ids: list[str] = []
        self.bytes_per_vector = dim  # 1 byte/dim

    def train_add(self, chunk_ids: list[str], matrix: np.ndarray) -> "ScalarInt8Index":
        self.chunk_ids = list(chunk_ids)
        m = matrix.astype(np.float32)
        self.index.train(m)
        self.index.add(m)
        return self

    def search(self, q: np.ndarray, k: int = 10) -> list[int]:
        _, idx = self.index.search(q.astype(np.float32).reshape(1, -1), k)
        return [i for i in idx[0] if i != -1]


# ---- artifact stage registration ------------------------------------------

def build_embeddings() -> dict:
    from . import bootstrap as bs

    chunks = bs.ensure("chunks")
    ids = [c.chunk_id for c in chunks]
    mat = embed([c.for_index() for c in chunks])
    return {"chunk_ids": ids, "matrix": mat, "model": _EMBED_MODEL, "dim": int(mat.shape[1])}


def build_ann_index() -> HNSWIndex:
    from . import bootstrap as bs

    emb = bs.ensure("embeddings")
    return HNSWIndex(emb["dim"]).add(emb["chunk_ids"], emb["matrix"])


def _register():
    from . import bootstrap as bs

    bs.register_stage("embeddings", ("chunks",), build_embeddings)
    bs.register_stage("ann_index", ("embeddings",), build_ann_index)


_register()
