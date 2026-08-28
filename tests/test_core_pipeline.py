"""Fast, dependency-light invariants: models, normalization/offsets, chunking provenance,
sparse retrieval, RRF, metrics, filters, tool schemas, permissions, citation validation.

Heavier model-backed paths (embeddings, reranker, PII, e2e) are in test_ml_pipeline.py.
"""
from __future__ import annotations

import unicodedata

import pytest

from patentrag import bootstrap as bs
from patentrag.chunking import chunk_document, fixed_token_chunks
from patentrag.evaluation import (dcg_at_k, f1, mean_reciprocal_rank, ndcg_at_k,
                                  precision_at_k, recall_at_k, reciprocal_rank, tool_prf,
                                  trajectory_match)
from patentrag.fusion import make_filter, reciprocal_rank_fusion
from patentrag.models import PatentDocument, SectionKind, stable_chunk_id
from patentrag.normalize import OffsetMap, normalize_document, normalize_text
from patentrag.sparse import BM25, InvertedIndex, tokenize


# ----- models -----

def test_document_loads_and_views(docs):
    d = docs[0]
    assert d.doc_id.startswith("US")
    assert d.claims and all(c.number >= 1 for c in d.claims)
    indep_nums = {c.number for c in d.independent_claims}
    assert indep_nums.issubset({c.number for c in d.claims})
    assert d.full_text()


def test_section_kind_inference():
    assert SectionKind.infer("DETAILED DESCRIPTION") == SectionKind.DETAILED_DESCRIPTION
    assert SectionKind.infer("PRIORITY CLAIM") == SectionKind.OTHER  # not the claims section
    assert SectionKind.infer("What is claimed is") == SectionKind.CLAIMS


def test_stable_chunk_id_deterministic():
    a = stable_chunk_id("US1", "sec", 0, "hello")
    b = stable_chunk_id("US1", "sec", 0, "hello")
    assert a == b and len(a) == 12


# ----- normalization + offsets (SPEC §11-12) -----

@pytest.mark.parametrize("form,text", [
    ("NFC", "café résumé"),
    ("NFKC", "½ ﬁle x²"),
    ("NFC", "Mert Öz, Herwig Häle"),
])
def test_offsetmap_exact_reverse(form, text):
    om = OffsetMap.build(text, form)
    assert om.normalized == normalize_text(text, form)
    for a in range(len(om.normalized) + 1):
        for b in range(a, len(om.normalized) + 1):
            o0, o1 = om.to_original_span(a, b)
            assert 0 <= o0 <= o1 <= len(text)
            if a != b:
                assert om.normalized[a:b] in unicodedata.normalize(form, text[o0:o1])


def test_offsetmap_expands_and_contracts():
    # ligature expands: one original char -> two normalized, both map back to it
    om = OffsetMap.build("ﬁx", "NFKC")
    assert om.recover_original(0, 1) == "ﬁ"
    # decomposed accent contracts: two original chars -> one normalized
    om2 = OffsetMap.build("é", "NFC")
    assert om2.to_original_span(0, 1) == (0, 2)


# ----- chunking provenance -----

def test_chunk_provenance_roundtrip(docs):
    d = next(x for x in docs if x.sections)
    nd = normalize_document(d)
    for c in chunk_document(d, nd):
        if c.claim_number is None and c.anchor and c.anchor.section_id.endswith(tuple("0123456789")):
            ns = nd.section(c.anchor.section_id)
            rec = ns.offset_map.recover_original(c.anchor.norm_start, c.anchor.norm_end)
            assert rec == ns.original[c.anchor.orig_start:c.anchor.orig_end]


def test_chunking_claim_and_section(docs):
    d = docs[0]
    cs = chunk_document(d)
    assert any(c.claim_number is not None for c in cs)
    assert any(c.claim_number is None for c in cs)
    assert all(c.token_estimate > 0 for c in cs)


def test_fixed_chunks_cover_tokens():
    text = " ".join(str(i) for i in range(100))
    assert len(fixed_token_chunks(text, 20, 0)) == 5


# ----- sparse retrieval -----

def test_tokenize_keeps_identifiers():
    assert tokenize("G06F16/2457 ph-value 3.5") == ["g06f16/2457", "ph-value", "3.5"]


def test_bm25_ranks_relevant_first():
    ids = ["a", "b", "c"]
    texts = ["vector nearest neighbor index", "recipe for soup", "vector similarity search"]
    bm = BM25().fit(ids, texts)
    top = bm.search("vector search", 3)
    assert top[0][0] in {"a", "c"}
    assert top[0][1] >= top[-1][1]


def test_inverted_index_boolean():
    idx = InvertedIndex()
    idx.add("d1", "alpha beta gamma")
    idx.add("d2", "beta delta")
    assert idx.boolean_and("beta") == {"d1", "d2"}
    assert idx.df("gamma") == 1


# ----- fusion + filters -----

def test_rrf_orders_by_agreement():
    r1 = ["x", "y", "z"]
    r2 = ["y", "x", "w"]
    fused = dict(reciprocal_rank_fusion([r1, r2]))
    assert fused["y"] >= fused["z"]


def test_metadata_filter(docs):
    d = docs[0]
    c = chunk_document(d)[0]
    pred = make_filter(cpc_section=d.cpc_sections[0] if d.cpc_sections else None)
    assert pred(c, d) is True
    pred_bad = make_filter(cpc_section="Z")
    assert pred_bad(c, d) is False


# ----- metrics (SPEC §37-41) -----

def test_precision_recall():
    ranked = ["a", "b", "c", "d"]
    rel = {"a", "c"}
    assert precision_at_k(ranked, rel, 2) == 0.5
    assert recall_at_k(ranked, rel, 4) == 1.0
    assert 0.0 <= recall_at_k(ranked, rel, 1) <= 1.0


def test_mrr_and_rr():
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert mean_reciprocal_rank([["a"], ["b"]], [{"a"}, set()]) == 0.5


def test_ndcg_bounds():
    rel = {"a": 1.0, "b": 1.0}
    good = ndcg_at_k(["a", "b", "c"], rel, 3)
    bad = ndcg_at_k(["c", "d", "a"], rel, 3)
    assert 0.0 <= bad <= good <= 1.0
    assert dcg_at_k([1, 1], 2) > 0
    assert f1(0.5, 0.5) == 0.5


# ----- agent tool schemas + trajectory + permission metrics -----

def test_tool_schema_validation():
    from patentrag.agent import FetchPatentArgs, SearchPatentsArgs

    assert SearchPatentsArgs(query="vector search").top_k == 5
    with pytest.raises(Exception):
        SearchPatentsArgs(query="x", top_k=999)  # top_k > 50
    with pytest.raises(Exception):
        FetchPatentArgs(publication_number="not-a-number")


def test_tool_prf_and_trajectory():
    m = tool_prf(["a", "b", "b"], ["a", "b"])
    assert 0 <= m["f1"] <= 1
    assert trajectory_match(["a", "b"], ["a", "b"], "exact")
    assert trajectory_match(["b", "a"], ["a", "b"], "unordered")
    assert not trajectory_match(["a"], ["a", "b"], "exact")
