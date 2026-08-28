"""Request tracing + the end-to-end pipeline (SPEC Parts XXIX, XXXI).

Every request produces a structured trace: one span per stage with latency, the ids/scores that
flowed through, and the guardrail decisions taken. Traces never carry raw PII. The
:class:`PatentRAGPipeline` is the single ``patent_rag.query()`` entry point that composes
ingestion-time artifacts with retrieval, fusion, reranking, generation and the input/output
guardrails, exposing every intermediate state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Trace model
# ---------------------------------------------------------------------------

@dataclass
class Span:
    stage: str
    latency_ms: float
    data: dict = field(default_factory=dict)


@dataclass
class RequestTrace:
    query: str
    spans: list[Span] = field(default_factory=list)
    response: Optional[Any] = None

    def add(self, stage: str, latency_ms: float, **data: Any) -> None:
        self.spans.append(Span(stage, latency_ms, data))

    @property
    def total_ms(self) -> float:
        return sum(s.latency_ms for s in self.spans)

    def to_dict(self) -> dict:
        return {"query": self.query, "total_ms": round(self.total_ms, 1),
                "spans": [{"stage": s.stage, "latency_ms": round(s.latency_ms, 1), **s.data} for s in self.spans]}

    def pretty(self) -> str:
        lines = [f"QUERY: {self.query}"]
        for s in self.spans:
            extra = " ".join(f"{k}={v}" for k, v in s.data.items())
            lines.append(f"  ↓ {s.stage:22} {s.latency_ms:7.1f} ms  {extra}")
        lines.append(f"  = TOTAL {self.total_ms:.1f} ms")
        return "\n".join(lines)


class _Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t0) * 1000


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    answer: Any
    trace: RequestTrace
    blocked: bool = False
    block_reason: str = ""
    context: list = field(default_factory=list)


class PatentRAGPipeline:
    """Compose the whole system behind one ``query()`` call, tracing each stage."""

    def __init__(self, provider=None, user_role: str = "researcher") -> None:  # noqa: ANN001
        from . import bootstrap as bs
        from .dense import DenseRetriever
        from .generation import default_provider
        from .guardrails import ContentSafety, InjectionDetector, PIIGuard, ScopeValidator

        self.docs = {d.doc_id: d for d in bs.ensure("docs_canonical")}
        self.chunks = bs.ensure("chunks")
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = bs.ensure("bm25_index")
        emb = bs.ensure("embeddings")
        self.dense = DenseRetriever(emb["chunk_ids"], emb["matrix"])
        self.provider = provider or default_provider()
        self.user_role = user_role
        self.pii = PIIGuard()
        self.injection = InjectionDetector()
        self.safety = ContentSafety()
        self.scope = ScopeValidator()
        self._reranker = None

    def _rerank(self):
        if self._reranker is None:
            from .fusion import CrossEncoderReranker

            self._reranker = CrossEncoderReranker()
        return self._reranker

    def query(self, user_query: str, top_k: int = 5, use_reranker: bool = True,
              check_scope: bool = True) -> PipelineResult:
        from .evaluation import ranked_docs_from_chunks  # noqa: F401
        from .fusion import reciprocal_rank_fusion
        from .generation import generate_answer
        from .guardrails import (authorize_documents, check_groundedness, verify_citations)

        trace = RequestTrace(query=user_query)

        # --- INPUT GUARDRAILS ---
        with _Timer() as t:
            inj = self.injection.detect(user_query)
            safe = self.safety.check(user_query)
            scope = self.scope.check(user_query) if check_scope else None
        blocked = inj.blocked or safe.blocked or (scope.blocked if scope else False)
        trace.add("input_guardrails", t.ms, injection=inj.blocked, unsafe=safe.blocked,
                  out_of_scope=(scope.blocked if scope else False))
        if blocked:
            reason = inj.reason or safe.reason or (scope.reason if scope else "blocked")
            trace.response = None
            return PipelineResult(answer=None, trace=trace, blocked=True, block_reason=reason)

        # --- SPARSE + DENSE RETRIEVAL ---
        with _Timer() as t:
            bm = self.bm25.search(user_query, 40)
        trace.add("sparse_retrieval", t.ms, candidates=len(bm))
        with _Timer() as t:
            dn = self.dense.search(user_query, 40)
        trace.add("dense_retrieval", t.ms, candidates=len(dn))

        # --- FUSION ---
        with _Timer() as t:
            fused = reciprocal_rank_fusion([[c for c, _ in bm], [c for c, _ in dn]])
        trace.add("fusion_rrf", t.ms, unique=len(fused))
        cand_ids = [cid for cid, _ in fused[:30]]

        # --- RERANK ---
        if use_reranker:
            with _Timer() as t:
                ranked = self._rerank().rerank(
                    user_query, [(cid, self.by_id[cid].for_index()) for cid in cand_ids], top_k=top_k * 3)
            trace.add("rerank", t.ms, kept=len(ranked))
            ranked_ids = [cid for cid, _ in ranked]
        else:
            ranked_ids = cand_ids[: top_k * 3]

        # --- RETRIEVAL AUTHORIZATION ---
        with _Timer() as t:
            ranked_docs = list({self.by_id[cid].document_id for cid in ranked_ids})
            allowed, denied = authorize_documents([self.docs[d] for d in ranked_docs], self.user_role)
            allowed_ids = {d.doc_id for d in allowed}
            ranked_ids = [cid for cid in ranked_ids if self.by_id[cid].document_id in allowed_ids]
        trace.add("authorization", t.ms, denied=len(denied))

        ranked_chunks = [self.by_id[cid] for cid in ranked_ids][: top_k * 2]

        # --- CONTEXT + GENERATION ---
        with _Timer() as t:
            answer = generate_answer(user_query, ranked_chunks, provider=self.provider)
        trace.add("generation", t.ms, provider=self.provider.name, citations=len(answer.citations))

        # --- OUTPUT GUARDRAILS ---
        with _Timer() as t:
            evidence = [self.by_id[c.chunk_id].text for c in answer.citations if c.chunk_id in self.by_id]
            ground = check_groundedness(answer.answer, evidence)
            checks = verify_citations(answer, set(ranked_ids), self.by_id)
            masked, _ = self.pii.mask(answer.answer_without_tags)
            out_safe = self.safety.check(answer.answer)
        valid_cites = sum(1 for c in checks if c.valid)
        trace.add("output_guardrails", t.ms, groundedness=round(ground.score, 2),
                  citations_valid=f"{valid_cites}/{len(checks)}", output_pii=(masked != answer.answer_without_tags),
                  output_unsafe=out_safe.blocked)
        answer.confidence = round(0.5 * ground.score + 0.5 * (valid_cites / max(1, len(checks))), 3)
        trace.response = answer
        return PipelineResult(answer=answer, trace=trace, context=ranked_chunks)


# ---------------------------------------------------------------------------
# Provenance chain resolution (SPEC §20) — answer claim → … → bbox
# ---------------------------------------------------------------------------

def resolve_provenance(citation, chunks_by_id: dict, docs_by_id: dict) -> dict:
    """Resolve a citation all the way back to its source, returning each hop of the chain."""
    from .normalize import normalize_document

    chunk = chunks_by_id.get(citation.chunk_id)
    if not chunk:
        return {"resolved": False, "reason": "chunk not found"}
    doc = docs_by_id.get(chunk.document_id)
    anchor = chunk.anchor
    chain: dict[str, Any] = {
        "resolved": True,
        "answer_cites": citation.label(),
        "chunk_id": chunk.chunk_id,
        "normalized_offsets": (anchor.norm_start, anchor.norm_end) if anchor else None,
        "original_offsets": (anchor.orig_start, anchor.orig_end) if anchor else None,
        "section": chunk.section,
        "claim_number": chunk.claim_number,
        "publication_number": chunk.publication_number,
    }
    # Recover the exact original substring via the OffsetMap (proves the offsets resolve).
    if doc and anchor and anchor.section_id and citation.claim_number is None:
        nd = normalize_document(doc)
        try:
            nsec = nd.section(anchor.section_id)
            chain["recovered_original_text"] = nsec.offset_map.recover_original(
                anchor.norm_start, anchor.norm_end)[:160]
        except KeyError:
            pass
    chain["pdf_page_or_xml"] = "PDF page/XML node anchor available via SourceAnchor" if anchor else None
    chain["bbox"] = anchor.bbox.model_dump() if (anchor and anchor.bbox) else None
    return chain
