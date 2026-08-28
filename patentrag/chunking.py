"""Chunking + contextual enrichment (SPEC Parts VII–VIII).

In a patent system chunking is half an information-retrieval problem and half a *citability*
problem: a chunk is the unit you retrieve **and** the unit you cite, so its boundaries must
respect structure (sections, claims) and it must carry provenance (offsets → OffsetMap →
original → XML/PDF anchor).

This module exposes the naive strategies (fixed / overlap / paragraph) for teaching
comparison, and the production `chunk_document` (section- and claim-aware, anchor-preserving),
plus contextual enrichment.
"""
from __future__ import annotations

import re

from .models import PatentChunk, PatentDocument, SectionKind, SourceAnchor, stable_chunk_id
from .normalize import NormalizedDocument, OffsetMap, normalize_document

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (word/punct pieces ≈ subword tokens within ~15%)."""
    return len(_WORD.findall(text))


# ---------------------------------------------------------------------------
# Naive strategies (for the Chapter-05 comparison; not used in production paths)
# ---------------------------------------------------------------------------

def fixed_token_chunks(text: str, size: int = 180, overlap: int = 0) -> list[str]:
    """Split into fixed-size token windows (optionally overlapping). Ignores structure."""
    toks = _WORD.findall(text)
    if size <= 0:
        raise ValueError("size must be > 0")
    step = max(1, size - overlap)
    return [" ".join(toks[i : i + size]) for i in range(0, len(toks), step) if toks[i : i + size]]


def paragraph_chunks(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# ---------------------------------------------------------------------------
# Production: section- and claim-aware chunking with full provenance
# ---------------------------------------------------------------------------

def _pack_paragraphs(paragraphs: list[tuple[int, int, str]], target: int, overlap_paras: int
                     ) -> list[tuple[int, int, str]]:
    """Pack (start, end, text) paragraph spans into chunks near `target` tokens.

    Returns (norm_start, norm_end, text) spans in the SAME coordinate system as the inputs.
    """
    chunks: list[tuple[int, int, str]] = []
    i = 0
    n = len(paragraphs)
    while i < n:
        j = i
        tok = 0
        while j < n and (tok == 0 or tok + estimate_tokens(paragraphs[j][2]) <= target):
            tok += estimate_tokens(paragraphs[j][2])
            j += 1
        start = paragraphs[i][0]
        end = paragraphs[j - 1][1]
        text = "\n\n".join(p[2] for p in paragraphs[i:j])
        chunks.append((start, end, text))
        if j >= n:
            break
        i = max(i + 1, j - overlap_paras)
    return chunks


def _paragraph_spans(shadow: str) -> list[tuple[int, int, str]]:
    """Locate paragraph spans (start, end, text) within the normalized section text."""
    spans: list[tuple[int, int, str]] = []
    for m in re.finditer(r"[^\n](?:.|\n(?!\s*\n))*", shadow):
        seg = m.group(0)
        if seg.strip():
            spans.append((m.start(), m.end(), seg.strip()))
    return spans


def chunk_document(
    doc: PatentDocument,
    norm_doc: NormalizedDocument | None = None,
    target_tokens: int = 180,
    overlap_paras: int = 1,
    enrich: bool = True,
) -> list[PatentChunk]:
    """Section-aware description chunks + one chunk per claim, all anchor-preserving."""
    norm_doc = norm_doc or normalize_document(doc)
    chunks: list[PatentChunk] = []

    # --- description / abstract sections ---
    for nsec in norm_doc.sections:
        om = nsec.offset_map
        spans = _paragraph_spans(nsec.shadow)
        if not spans:
            continue
        packed = _pack_paragraphs(spans, target_tokens, overlap_paras)
        for (ns, ne, text) in packed:
            o0, o1 = om.to_original_span(ns, ne)
            anchor = SourceAnchor(
                document_id=doc.doc_id, section_id=nsec.section_id,
                norm_start=ns, norm_end=ne, orig_start=o0, orig_end=o1,
            )
            cid = stable_chunk_id(doc.doc_id, nsec.section_id, ns, text)
            chunks.append(
                PatentChunk(
                    chunk_id=cid, document_id=doc.doc_id, publication_number=doc.publication_number,
                    section=nsec.heading, section_kind=SectionKind(nsec.kind) if nsec.kind in SectionKind._value2member_map_ else SectionKind.OTHER,
                    text=text, norm_start=ns, norm_end=ne, anchor=anchor,
                    token_estimate=estimate_tokens(text),
                )
            )

    # --- claims: one chunk per claim (claim-aware granularity) ---
    for claim in doc.claims:
        om = OffsetMap.build(claim.text)
        anchor = SourceAnchor(
            document_id=doc.doc_id, section_id=f"{doc.doc_id}:claim{claim.number}",
            norm_start=0, norm_end=len(om.normalized), orig_start=0, orig_end=len(claim.text),
        )
        cid = stable_chunk_id(doc.doc_id, f"claim{claim.number}", 0, claim.text)
        chunks.append(
            PatentChunk(
                chunk_id=cid, document_id=doc.doc_id, publication_number=doc.publication_number,
                section="Claims", section_kind=SectionKind.CLAIMS, claim_number=claim.number,
                text=claim.text, norm_start=0, norm_end=len(om.normalized), anchor=anchor,
                token_estimate=estimate_tokens(claim.text),
            )
        )

    if enrich:
        for c in chunks:
            c.enriched_text = enrich_chunk(c, doc)
    return chunks


def enrich_chunk(chunk: PatentChunk, doc: PatentDocument) -> str:
    """Contextual enrichment: prepend disambiguating metadata so a fragment like
    'wherein the second component…' is interpretable and retrievable on its own.

    Kept deliberately compact — a header line, not a dump of the whole patent.
    """
    head = [f"Patent {doc.publication_number or doc.doc_id}: {doc.title}"]
    if chunk.claim_number is not None:
        kind = "independent claim" if _is_independent(doc, chunk.claim_number) else "dependent claim"
        head.append(f"Claim {chunk.claim_number} ({kind})")
    elif chunk.section:
        head.append(f"Section: {chunk.section}")
    if doc.cpc:
        head.append(f"CPC: {doc.cpc[0]}")
    return " | ".join(head) + "\n" + chunk.text


def _is_independent(doc: PatentDocument, number: int) -> bool:
    for c in doc.claims:
        if c.number == number:
            return c.is_independent
    return True


# ---- artifact stage registration ------------------------------------------

def build_all_chunks(target_tokens: int = 180, enrich: bool = True) -> list[PatentChunk]:
    from . import bootstrap as bs

    docs = bs.ensure("docs_canonical")
    norm = {nd.doc_id: nd for nd in bs.ensure("docs_normalized")}
    out: list[PatentChunk] = []
    for d in docs:
        out.extend(chunk_document(d, norm.get(d.doc_id), target_tokens=target_tokens, enrich=enrich))
    return out


def _register():
    from . import bootstrap as bs

    bs.register_stage("chunks", ("docs_normalized",), lambda: build_all_chunks())


_register()
