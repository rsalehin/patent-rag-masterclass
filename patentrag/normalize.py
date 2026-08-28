"""Unicode normalization, the OffsetMap, boilerplate handling, language ID, shadow text
(SPEC Parts V–VI).

The load-bearing idea of this module is **provenance under transformation**. A search system
normalizes text so lexical/semantic matching behaves; but a citation must still resolve to the
*exact original* character span (and thence to an XML node or PDF bounding box). The
:class:`OffsetMap` makes normalization reversible at character granularity.

The normalized representation is the system's *shadow text*; the original is immutable.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Literal

Form = Literal["NFC", "NFD", "NFKC", "NFKD"]


def normalize_text(text: str, form: Form = "NFC") -> str:
    """Thin wrapper over :func:`unicodedata.normalize` (kept for symmetry + discoverability)."""
    return unicodedata.normalize(form, text)


def _clusters(text: str) -> list[tuple[int, int]]:
    """Split text into normalization clusters: a starter (combining class 0) plus any trailing
    combining marks. Normalization never reorders across a starter boundary for well-formed
    text, so per-cluster normalization is a sound basis for an exact reverse map.
    """
    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(text[j]) != 0:
            j += 1
        spans.append((i, j))
        i = j
    return spans


@dataclass
class OffsetMap:
    """A reversible map between original text and its normalized (shadow) form.

    Built per cluster: every character of the normalized text records the *original* span
    (start, end) of the cluster it came from. Reverse lookup of a normalized span returns the
    minimal original span that covers it — never losing provenance, even when normalization
    expands (½→1⁄2, ﬁ→fi) or contracts (e+◌́→é) characters.
    """

    form: Form
    original: str
    normalized: str
    _orig_start: list[int] = field(repr=False, default_factory=list)
    _orig_end: list[int] = field(repr=False, default_factory=list)

    @classmethod
    def build(cls, original: str, form: Form = "NFC") -> "OffsetMap":
        norm_chars: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        for a, b in _clusters(original):
            piece = unicodedata.normalize(form, original[a:b])
            for ch in piece:
                norm_chars.append(ch)
                starts.append(a)
                ends.append(b)
        return cls(form=form, original=original, normalized="".join(norm_chars),
                   _orig_start=starts, _orig_end=ends)

    def __len__(self) -> int:
        return len(self.normalized)

    def to_original_span(self, norm_start: int, norm_end: int) -> tuple[int, int]:
        """Map a normalized [start, end) span to the covering original [start, end) span."""
        if norm_start < 0 or norm_end > len(self.normalized) or norm_start > norm_end:
            raise IndexError(f"span ({norm_start},{norm_end}) out of range for len {len(self.normalized)}")
        if norm_start == norm_end:
            # zero-width: anchor at the boundary
            if norm_start < len(self._orig_start):
                o = self._orig_start[norm_start]
            elif self._orig_end:
                o = self._orig_end[-1]
            else:
                o = 0
            return (o, o)
        o0 = min(self._orig_start[norm_start:norm_end])
        o1 = max(self._orig_end[norm_start:norm_end])
        return (o0, o1)

    def recover_original(self, norm_start: int, norm_end: int) -> str:
        o0, o1 = self.to_original_span(norm_start, norm_end)
        return self.original[o0:o1]

    def to_original_index(self, norm_index: int) -> int:
        """Original index corresponding to a normalized index (cluster start)."""
        return self.to_original_span(norm_index, norm_index)[0]


# ---------------------------------------------------------------------------
# Boilerplate handling (structural, provenance-preserving)
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = (
    "cross reference to related application",
    "this application claims priority",
    "the entire contents of which are incorporated by reference",
    "all rights reserved",
)


def is_boilerplate(paragraph: str) -> bool:
    """Heuristic: does a paragraph look like legal/priority boilerplate rather than substance?"""
    p = paragraph.lower()
    return any(pat in p for pat in _BOILERPLATE_PATTERNS)


# ---------------------------------------------------------------------------
# Language identification
# ---------------------------------------------------------------------------

def detect_language(text: str, seed: int = 20240817) -> str:
    """Deterministic language id (langdetect with a fixed seed). Returns an ISO-639-1 code."""
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = seed
    text = text.strip()
    if len(text) < 3:
        return "unknown"
    try:
        return detect(text)
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Shadow-text view for a whole document
# ---------------------------------------------------------------------------

@dataclass
class NormalizedSection:
    section_id: str
    heading: str
    kind: str
    original: str
    shadow: str
    offset_map: OffsetMap


@dataclass
class NormalizedDocument:
    doc_id: str
    sections: list[NormalizedSection]

    def section(self, section_id: str) -> NormalizedSection:
        for s in self.sections:
            if s.section_id == section_id:
                return s
        raise KeyError(section_id)


def normalize_document(doc, form: Form = "NFC") -> NormalizedDocument:  # noqa: ANN001
    """Produce the shadow-text representation of a :class:`~patentrag.models.PatentDocument`."""
    secs: list[NormalizedSection] = []
    for s in doc.sections:
        om = OffsetMap.build(s.text, form)
        secs.append(
            NormalizedSection(
                section_id=s.section_id,
                heading=s.heading,
                kind=s.kind.value if hasattr(s.kind, "value") else str(s.kind),
                original=s.text,
                shadow=om.normalized,
                offset_map=om,
            )
        )
    return NormalizedDocument(doc_id=doc.doc_id, sections=secs)


# ---- artifact stage registration ------------------------------------------

def _build_docs_normalized():
    from . import bootstrap as bs

    docs = bs.ensure("docs_canonical")
    return [normalize_document(d) for d in docs]


def _register():
    from . import bootstrap as bs

    bs.register_stage("docs_normalized", ("docs_canonical",), _build_docs_normalized)


_register()
