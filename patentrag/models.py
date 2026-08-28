"""Canonical patent document representation (SPEC Part II).

A patent is *structured data*, not a flat text blob. Flattening destroys the boundaries
(claims vs description, independent vs dependent, section provenance) that a patent-search
system depends on for both retrieval and — critically — *citability*. These Pydantic models
are the shared vocabulary every later chapter speaks.

Design notes
------------
* Every unit of text that can be retrieved or cited carries enough anchoring to resolve back
  to its origin: :class:`SourceAnchor` (offsets + XML/PDF location) and :class:`BoundingBox`.
* Offsets on chunks are into the document's *normalized* text; the :class:`SourceAnchor`
  additionally records the *original* offsets so an :class:`~patentrag.normalize.OffsetMap`
  (Chapter 04) can round-trip normalized→original.
* The models are deliberately permissive about optional biblio fields — real corpora have
  gaps — while remaining strict about the structural relationships that matter.
"""
from __future__ import annotations

import hashlib
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SectionKind(str, Enum):
    """Coarse, retrieval-relevant classification of a patent section."""

    ABSTRACT = "abstract"
    TECHNICAL_FIELD = "technical_field"
    BACKGROUND = "background"
    SUMMARY = "summary"
    BRIEF_DESCRIPTION_DRAWINGS = "brief_description_drawings"
    DETAILED_DESCRIPTION = "detailed_description"
    CLAIMS = "claims"
    OTHER = "other"

    @classmethod
    def infer(cls, heading: str) -> "SectionKind":
        """Map a raw heading string to a :class:`SectionKind` (best-effort, ordered)."""
        h = (heading or "").lower().strip()
        # Exact-ish claims match first (avoid catching 'PRIORITY CLAIM', 'CROSS REFERENCE').
        if h in {"claims", "claim"} or "what is claimed" in h or h.startswith("claims"):
            return cls.CLAIMS
        table = [
            (cls.TECHNICAL_FIELD, ("technical field", "field of")),
            (cls.BACKGROUND, ("background",)),
            (cls.SUMMARY, ("summary", "brief summary")),
            (cls.BRIEF_DESCRIPTION_DRAWINGS, ("brief description of the draw", "description of the draw")),
            (cls.DETAILED_DESCRIPTION, ("detailed description", "detailed desc", "embodiment")),
            (cls.ABSTRACT, ("abstract",)),
        ]
        for kind, needles in table:
            if any(n in h for n in needles):
                return kind
        return cls.OTHER


class BoundingBox(BaseModel):
    """A rectangle on a PDF page, in PDF points (origin top-left as reported by PyMuPDF)."""

    page: int = Field(ge=0, description="0-based page index")
    x0: float
    y0: float
    x1: float
    y1: float

    @field_validator("x1")
    @classmethod
    def _x_order(cls, v: float, info) -> float:  # noqa: ANN001
        if "x0" in info.data and v < info.data["x0"]:
            raise ValueError("x1 must be >= x0")
        return v

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class SourceAnchor(BaseModel):
    """Everything needed to navigate a piece of text back to its immutable origin.

    An anchor may point into XML (``xpath``/``node_id``), a PDF page (``page``/``bbox``), or
    both. ``orig_start``/``orig_end`` are offsets into the *original* (pre-normalization)
    section text; ``norm_start``/``norm_end`` into the normalized text.
    """

    document_id: str
    section_id: Optional[str] = None
    xpath: Optional[str] = None
    node_id: Optional[str] = None
    page: Optional[int] = None
    bbox: Optional[BoundingBox] = None
    orig_start: Optional[int] = None
    orig_end: Optional[int] = None
    norm_start: Optional[int] = None
    norm_end: Optional[int] = None


class Citation(BaseModel):
    """A machine-checkable claim-of-support: 'this text is backed by that source span'."""

    document_id: str
    publication_number: Optional[str] = None
    section: Optional[str] = None
    claim_number: Optional[int] = None
    chunk_id: Optional[str] = None
    anchor: Optional[SourceAnchor] = None
    quote: Optional[str] = None

    def label(self) -> str:
        """Stable, human-readable citation tag used in prompts and answers."""
        parts = [f"PATENT={self.publication_number or self.document_id}"]
        if self.section:
            parts.append(f"SECTION={self.section}")
        if self.claim_number is not None:
            parts.append(f"CLAIM={self.claim_number}")
        if self.chunk_id:
            parts.append(f"CHUNK={self.chunk_id}")
        return "[" + " | ".join(parts) + "]"


class PatentClaim(BaseModel):
    number: int = Field(ge=1)
    text: str
    dependent_on: Optional[int] = None

    @property
    def is_independent(self) -> bool:
        return self.dependent_on is None


class PatentSection(BaseModel):
    section_id: str
    kind: SectionKind = SectionKind.OTHER
    heading: str = ""
    text: str = ""
    order: int = 0
    anchor: Optional[SourceAnchor] = None


class PatentChunk(BaseModel):
    """A retrieval/citation unit carved from one document, with full provenance preserved."""

    chunk_id: str
    document_id: str
    publication_number: Optional[str] = None
    section: Optional[str] = None
    section_kind: SectionKind = SectionKind.OTHER
    claim_number: Optional[int] = None
    text: str
    # enrichment (Chapter 05/08): context prepended for retrieval, kept separate from `text`
    enriched_text: Optional[str] = None
    norm_start: Optional[int] = None
    norm_end: Optional[int] = None
    anchor: Optional[SourceAnchor] = None
    token_estimate: int = 0

    def for_index(self) -> str:
        """Text a retriever should index/embed: enriched if present, else raw."""
        return self.enriched_text or self.text


class PatentDocument(BaseModel):
    """The canonical, structured representation of one patent."""

    doc_id: str
    publication_number: Optional[str] = None
    application_number: Optional[str] = None
    title: str = ""
    abstract: str = ""
    assignees: list[str] = Field(default_factory=list)
    inventors: list[str] = Field(default_factory=list)
    priority_date: Optional[date] = None
    filing_date: Optional[date] = None
    publication_date: Optional[date] = None
    cpc: list[str] = Field(default_factory=list)
    ipc: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    language: str = "en"
    claims: list[PatentClaim] = Field(default_factory=list)
    sections: list[PatentSection] = Field(default_factory=list)
    access_level: str = "public"  # used by the retrieval-authorization guardrail (Ch12)
    source: dict = Field(default_factory=dict)

    # ---- convenience views -------------------------------------------------
    @property
    def independent_claims(self) -> list[PatentClaim]:
        return [c for c in self.claims if c.is_independent]

    @property
    def cpc_sections(self) -> list[str]:
        """Top-level CPC sections present (e.g. ['G','H'])."""
        return sorted({c[0] for c in self.cpc if c})

    def full_text(self) -> str:
        """Concatenated human-readable text (title→abstract→sections→claims)."""
        buf = [self.title, self.abstract]
        buf += [f"{s.heading}\n{s.text}" for s in self.sections]
        buf += [f"Claim {c.number}. {c.text}" for c in self.claims]
        return "\n\n".join(t for t in buf if t)

    # ---- construction from the bundled corpus JSON -------------------------
    @classmethod
    def from_corpus_json(cls, d: dict) -> "PatentDocument":
        """Build a document from one ``data/corpus/*.json`` record (real Google-Patents data)."""

        def _date(v):  # noqa: ANN001
            if not v:
                return None
            try:
                return date.fromisoformat(v[:10])
            except ValueError:
                return None

        sections: list[PatentSection] = []
        order = 0
        if d.get("abstract"):
            sections.append(
                PatentSection(
                    section_id=f"{d['doc_id']}:abstract",
                    kind=SectionKind.ABSTRACT,
                    heading="ABSTRACT",
                    text=d["abstract"],
                    order=order,
                )
            )
            order += 1
        for sec in d.get("description", []):
            heading = sec.get("heading", "SECTION")
            body = "\n\n".join(sec.get("paragraphs", []))
            sections.append(
                PatentSection(
                    section_id=f"{d['doc_id']}:sec{order}",
                    kind=SectionKind.infer(heading),
                    heading=heading,
                    text=body,
                    order=order,
                )
            )
            order += 1

        claims = [
            PatentClaim(number=c["number"], text=c["text"], dependent_on=c.get("dependent_on"))
            for c in d.get("claims", [])
        ]
        return cls(
            doc_id=d["doc_id"],
            publication_number=d.get("publication_number"),
            application_number=d.get("application_number"),
            title=d.get("title", ""),
            abstract=d.get("abstract", ""),
            assignees=d.get("assignees", []),
            inventors=d.get("inventors", []),
            priority_date=_date(d.get("priority_date")),
            filing_date=_date(d.get("filing_date")),
            publication_date=_date(d.get("publication_date")),
            cpc=d.get("cpc", []),
            ipc=d.get("ipc", []),
            citations=d.get("citations", []),
            language=d.get("language", "en"),
            claims=claims,
            sections=sections,
            source=d.get("source", {}),
        )


def stable_chunk_id(document_id: str, section: str, start: int, text: str) -> str:
    """Deterministic short id for a chunk (content-addressed → reproducible across runs)."""
    h = hashlib.sha1(f"{document_id}|{section}|{start}|{text}".encode("utf-8")).hexdigest()
    return h[:12]
