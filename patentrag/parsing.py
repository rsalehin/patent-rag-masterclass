"""XML/ST.36/ST.96 parsing + PDF/OCR extraction (SPEC Parts III–IV).

XML is where patents keep their structure; PDF is where they keep their layout. This module
parses the genuine WIPO ST.96 example (with real XSD validation and a version-migration demo),
shows compact DTD validation for the ST.36 lineage, and extracts text/words/bounding-boxes from
a real patent PDF — wiring boxes into :class:`~patentrag.models.SourceAnchor` so a citation can
eventually highlight the exact source region.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from lxml import etree

from .models import BoundingBox, SourceAnchor

ST96 = {
    "pat": "http://www.wipo.int/standards/XMLSchema/ST96/Patent",
    "com": "http://www.wipo.int/standards/XMLSchema/ST96/Common",
}


# ===========================================================================
# XML fundamentals + ST.96
# ===========================================================================

def load_xml(path: str | Path) -> etree._ElementTree:
    return etree.parse(str(path))


def _first_text(root, localname: str) -> str | None:
    for e in root.iter():
        if isinstance(e.tag, str) and etree.QName(e).localname == localname:
            return (e.text or "").strip() or None
    return None


def st96_bibliographic(tree: etree._ElementTree) -> dict:
    """Extract core bibliographic fields from an ST.96 PatentPublication instance."""
    r = tree.getroot()
    return {
        "publication_number": _first_text(r, "PublicationNumber"),
        "kind_code": _first_text(r, "PatentDocumentKindCode"),
        "publication_date": _first_text(r, "PublicationDate"),
        "application_number": _first_text(r, "ApplicationNumberText"),
        "title": _first_text(r, "InventionTitle"),
        "ip_office": _first_text(r, "IPOfficeCode"),
        "st96_version": r.get(f"{{{ST96['com']}}}st96Version"),
    }


def st96_claims(tree: etree._ElementTree) -> list[str]:
    """Return each claim's text (ST.96 Claims/Claim/ClaimText/…)."""
    r = tree.getroot()
    claims = []
    for claim in r.iter(f"{{{ST96['pat']}}}Claim"):
        txt = " ".join(t.strip() for t in claim.itertext() if t.strip())
        if txt:
            claims.append(re.sub(r"\s+", " ", txt))
    return claims


def xpath(tree: etree._ElementTree, expr: str, namespaces: dict | None = None) -> list:
    return tree.xpath(expr, namespaces=namespaces or ST96)


# --- XSD validation against the genuine flattened WIPO schema ---

def extract_st96_schema(dest_dir: str | Path, zip_path: str | Path) -> Path:
    """Unzip the bundled flattened ST.96 schema package once; return the extraction dir."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if not any(dest.glob("*.xsd")):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
    return dest


def validate_xsd(tree: etree._ElementTree, xsd_path: str | Path) -> tuple[bool, list[str]]:
    schema = etree.XMLSchema(etree.parse(str(xsd_path)))
    ok = schema.validate(tree)
    errors = [f"{e.line}: {e.message}" for e in schema.error_log]
    return ok, errors


def migrate_st96_version(xml_bytes: bytes, to_version: str = "V9_0") -> bytes:
    """Migrate a genuine V8_0 instance to V9_0: bump the fixed st96Version attributes and
    upgrade the MathML3 namespace to the plain MathML namespace the v9.0 schema expects.

    This is exactly the kind of transformation an ingestion pipeline runs when a standards
    body increments a version; here it turns the two real validation errors into a clean pass.
    """
    text = xml_bytes.decode("utf-8")
    text = re.sub(r'st96Version="V8_0"', 'st96Version="V9_0"', text)
    text = text.replace("http://www.w3.org/1998/Math/MathML3", "http://www.w3.org/1998/Math/MathML")
    return text.encode("utf-8")


# ===========================================================================
# DTD (ST.36 lineage) — compact teaching schema
# ===========================================================================

ST36_DTD = """
<!ELEMENT patent-document (bibliographic-data, abstract?, description?, claims?)>
<!ATTLIST patent-document lang CDATA #REQUIRED country CDATA #REQUIRED
                          doc-number CDATA #REQUIRED kind CDATA #IMPLIED>
<!ELEMENT bibliographic-data (invention-title, publication-date?)>
<!ELEMENT invention-title (#PCDATA)>
<!ELEMENT publication-date (#PCDATA)>
<!ELEMENT abstract (p+)>
<!ELEMENT description (p+)>
<!ELEMENT claims (claim+)>
<!ELEMENT claim (claim-text)>
<!ATTLIST claim num CDATA #REQUIRED>
<!ELEMENT claim-text (#PCDATA)>
<!ELEMENT p (#PCDATA)>
""".strip()


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def patentdoc_to_st36_xml(doc) -> bytes:  # noqa: ANN001
    """Emit a compact ST.36-style XML instance from a PatentDocument (real content)."""
    country, number = "US", (doc.publication_number or doc.doc_id)
    lines = [
        f'<patent-document lang="en" country="{country}" doc-number="{_xml_escape(number)}" kind="B2">',
        "  <bibliographic-data>",
        f"    <invention-title>{_xml_escape(doc.title)}</invention-title>",
        f"    <publication-date>{doc.publication_date or ''}</publication-date>",
        "  </bibliographic-data>",
    ]
    if doc.abstract:
        lines += ["  <abstract>", f"    <p>{_xml_escape(doc.abstract)}</p>", "  </abstract>"]
    if doc.claims:
        lines.append("  <claims>")
        for c in doc.claims[:5]:
            lines.append(f'    <claim num="{c.number}"><claim-text>{_xml_escape(c.text)}</claim-text></claim>')
        lines.append("  </claims>")
    lines.append("</patent-document>")
    return ("\n".join(lines)).encode("utf-8")


def validate_dtd(xml_bytes: bytes, dtd_str: str) -> tuple[bool, list[str]]:
    dtd = etree.DTD(io.StringIO(dtd_str))
    root = etree.fromstring(xml_bytes)
    ok = dtd.validate(root)
    return ok, [str(e) for e in dtd.error_log]


# ===========================================================================
# PDF text / words / boxes + OCR
# ===========================================================================

def pdf_page_count(path: str | Path) -> int:
    import fitz

    with fitz.open(str(path)) as d:
        return d.page_count


def pdf_page_text(path: str | Path, page: int = 0) -> str:
    import fitz

    with fitz.open(str(path)) as d:
        return d[page].get_text("text")


def pdf_words(path: str | Path, page: int = 0) -> list[tuple[str, float, float, float, float]]:
    """Words with bounding boxes: (text, x0, y0, x1, y1) in PDF points."""
    import fitz

    with fitz.open(str(path)) as d:
        return [(w[4], w[0], w[1], w[2], w[3]) for w in d[page].get_text("words")]


def word_anchor(document_id: str, page: int, word_box: tuple) -> SourceAnchor:
    _, x0, y0, x1, y1 = word_box
    return SourceAnchor(document_id=document_id, page=page,
                        bbox=BoundingBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1))


def rasterize_page(path: str | Path, page: int = 0, dpi: int = 150):
    """Render a PDF page to a PIL image (for OCR)."""
    import fitz
    from PIL import Image

    with fitz.open(str(path)) as d:
        pix = d[page].get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def ocr_available() -> bool:
    from . import bootstrap as bs

    return bs.find_tesseract() is not None


def ocr_image(image) -> dict:
    """OCR a PIL image with tesseract; returns text + per-word confidence/boxes."""
    import pytesseract
    from pytesseract import Output

    from . import bootstrap as bs

    cmd = bs.find_tesseract()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    text = pytesseract.image_to_string(image)
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    words = [
        {"text": t, "conf": float(c), "box": (data["left"][i], data["top"][i], data["width"][i], data["height"][i])}
        for i, (t, c) in enumerate(zip(data["text"], data["conf"]))
        if t.strip() and float(c) >= 0
    ]
    return {"text": text, "words": words, "mean_conf": (sum(w["conf"] for w in words) / len(words) if words else 0.0)}


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein-based CER between the PDF text layer and OCR output (edit distance / |ref|)."""
    ref = re.sub(r"\s+", " ", reference).strip()
    hyp = re.sub(r"\s+", " ", hypothesis).strip()
    if not ref:
        return 0.0
    m, n = len(ref), len(hyp)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[n] / m


# ---- artifact stage registration ------------------------------------------

def _bundled_pdf() -> Path | None:
    from . import bootstrap as bs

    pdfs = sorted((bs.DATA / "pdf").glob("*.pdf"))
    return pdfs[0] if pdfs else None


def build_pdf_extractions() -> dict:
    path = _bundled_pdf()
    if not path:
        return {"available": False, "reason": "no bundled PDF"}
    words = pdf_words(path, 0)
    return {
        "available": True,
        "path": str(path),
        "doc_id": path.stem,
        "page0_text": pdf_page_text(path, 0),
        "page0_words": words[:400],
        "n_words_page0": len(words),
        "page_count": pdf_page_count(path),
    }


def _register():
    from . import bootstrap as bs

    bs.register_stage("pdf_extractions", ("corpus_raw",), build_pdf_extractions)


_register()
