#!/usr/bin/env python3
"""Build the bundled patent corpus from real, public patent data (Google Patents).

Run ONCE at authoring time; the JSON output is committed under data/corpus/ so that
Colab / offline runs never depend on network access. Patent text is public data;
Google Patents is used purely as an aggregator of that public record. Provenance for
every document (source URL, fetch date) is recorded in each JSON file and in
data/PROVENANCE.md.

Usage:
    python scripts/build_corpus.py            # build ~30 on-theme US patents
    python scripts/build_corpus.py --n 24     # cap document count
"""
from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from lxml import html as LH

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "corpus"
PDF_DIR = REPO / "data" / "pdf"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
CTX = ssl.create_default_context()

# On-theme queries: the corpus is deliberately about the very technologies the
# notebook series teaches (retrieval, indexing, embeddings, ANN, OCR, NLP), so that
# realistic patent-search queries have genuine relevant documents.
QUERIES = [
    "approximate nearest neighbor vector index search",
    "dense vector embedding semantic document retrieval",
    "inverted index information retrieval ranking",
    "product quantization compressed vector search",
    "neural network transformer language model",
    "optical character recognition document layout analysis",
    "bloom filter hashing data structure lookup",
    "knowledge graph question answering system",
    "locality sensitive hashing similarity search",
    "text classification natural language processing",
    "recommendation system collaborative filtering embedding",
    "database query optimization index structure",
    "speech recognition acoustic model decoding",
    "image feature extraction convolutional network",
]
# Explicit seeds guarantee coverage of a few well-structured computing patents.
SEEDS = ["US9081550B2", "US9720934B1", "US8930304B2", "US10083169B1"]


def get(url: str, tries: int = 5, timeout: int = 40) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, context=CTX, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3.0 * (i + 1))  # polite backoff for GP 503 rate-limiting
    raise last  # type: ignore[misc]


def search_us_patents(query: str, want: int = 8) -> list[str]:
    inner = urllib.parse.quote(f"q={query}")
    url = f"https://patents.google.com/xhr/query?url={inner}&exp="
    try:
        data = json.loads(get(url))
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for cluster in data.get("results", {}).get("cluster", []):
        for r in cluster.get("result", []):
            pn = r.get("patent", {}).get("publication_number", "")
            if pn.startswith("US") and re.match(r"US\d+B[12]$", pn):
                out.append(pn)
            if len(out) >= want:
                break
    return out


def _txt(nodes) -> str:
    return re.sub(r"\s+", " ", " ".join(t for t in nodes if t)).strip()


def parse_patent(pn: str) -> dict | None:
    raw = get(f"https://patents.google.com/patent/{pn}/en")
    H = raw.decode("utf-8", "replace")
    doc = LH.fromstring(raw)

    def meta(name: str) -> str | None:
        m = re.search(r'<meta name="%s"[^>]*content="([^"]*)"' % re.escape(name), H)
        return m.group(1).strip() if m else None

    title = meta("DC.title")
    if not title:
        return None
    abstract = _txt(doc.xpath('//div[@class="abstract"]//text()')) or (meta("DC.description") or "")

    # Claims: preserve number, text, and (independent vs dependent) via reference parse.
    claims = []
    for c in doc.xpath('//section[@itemprop="claims"]//div[@num and contains(@class,"claim")]'):
        num_raw = c.get("num") or ""
        m = re.match(r"0*(\d+)", num_raw)
        if not m:
            continue
        num = int(m.group(1))
        text = _txt(c.xpath("./div//text()")) or _txt(c.xpath(".//text()"))
        text = re.sub(r"^\d+\s*\.\s*", "", text).strip()
        if not text:
            continue
        deps = [int(x) for x in re.findall(r"claim\s+(\d+)", text)]
        dep = min(deps) if deps else None
        if not any(cl["number"] == num for cl in claims):
            claims.append({"number": num, "text": text, "dependent_on": dep})
    claims.sort(key=lambda x: x["number"])

    # Description: group paragraphs under their most recent heading (real sections).
    sections: list[dict] = []
    cur = {"heading": "DESCRIPTION", "paragraphs": []}
    desc = doc.xpath('//section[@itemprop="description"]')
    if desc:
        for el in desc[0].iter():
            tag = str(el.tag).lower()
            if tag == "heading":
                if cur["paragraphs"]:
                    sections.append(cur)
                cur = {"heading": _txt(el.xpath(".//text()")) or "SECTION", "paragraphs": []}
            elif tag in ("div",) and (el.get("class") or "").find("description-paragraph") >= 0:
                p = _txt(el.xpath(".//text()"))
                if p:
                    cur["paragraphs"].append(p)
        if cur["paragraphs"]:
            sections.append(cur)
    # Cap description size (keep it bundled-small) while keeping full section structure.
    kept, budget = [], 60
    for s in sections:
        if budget <= 0:
            break
        s = {"heading": s["heading"], "paragraphs": s["paragraphs"][:budget]}
        budget -= len(s["paragraphs"])
        kept.append(s)

    cpc = []
    for code in doc.xpath('//span[@itemprop="Code"]/text()'):
        code = code.strip()
        if re.match(r"^[A-H]\d\d[A-Z]", code) and code not in cpc:  # specific CPC (e.g. G06F3/167)
            cpc.append(code)

    def times(prop: str) -> str | None:
        v = doc.xpath(f'//time[@itemprop="{prop}"]/@datetime') or doc.xpath(f'//time[@itemprop="{prop}"]/text()')
        return v[0].strip() if v else None

    inventors = [i.strip() for i in doc.xpath('//dd[@itemprop="inventor"]/text()') if i.strip()]
    assignees = [a.strip() for a in doc.xpath('//dd[@itemprop="assigneeOriginal"]/text() | //dd[@itemprop="assigneeCurrent"]/text()') if a.strip()]
    citations = []
    for cnode in doc.xpath('//tr[contains(@itemprop,"backwardReferences")]//span[@itemprop="publicationNumber"]/text()'):
        cnode = cnode.strip()
        if cnode and cnode not in citations:
            citations.append(cnode)

    return {
        "doc_id": pn,
        "publication_number": pn,
        "title": title,
        "abstract": abstract,
        "assignees": assignees[:3],
        "inventors": inventors[:8],
        "priority_date": times("priorityDate"),
        "filing_date": times("filingDate"),
        "publication_date": times("publicationDate"),
        "cpc": cpc[:12],
        "citations": citations[:20],
        "language": "en",
        "claims": claims,
        "description": kept,
        "pdf_url": meta("citation_pdf_url"),
        "source": {
            "provider": "Google Patents",
            "url": f"https://patents.google.com/patent/{pn}/en",
            "fetched": date.today().isoformat(),
            "note": "Patent text is public record; Google Patents used as aggregator.",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    CORPUS.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Deterministic selection: seeds first, then query hits in query order.
    pool: list[str] = list(SEEDS)
    for q in QUERIES:
        for pn in search_us_patents(q):
            if pn not in pool:
                pool.append(pn)
        time.sleep(0.5)
    print(f"candidate pool: {len(pool)} patents")

    # Carry forward any patents already built (skip-existing top-up mode).
    built: list[dict] = []
    for existing in sorted(CORPUS.glob("US*.json")):
        try:
            built.append(json.loads(existing.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    have = {d["doc_id"] for d in built}
    print(f"already have {len(have)} patents; topping up to {args.n}")

    for pn in pool:
        if len(built) >= args.n:
            break
        if pn in have:
            continue
        try:
            d = parse_patent(pn)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {pn}: {e!r}"[:120])
            continue
        if not d or len(d["claims"]) < 2 or sum(len(s["paragraphs"]) for s in d["description"]) < 5:
            print(f"  skip {pn}: too little structure")
            continue
        (CORPUS / f"{pn}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        built.append(d)
        print(f"  ok {pn}: {len(d['claims'])} claims, {len(d['description'])} sections, {len(d['inventors'])} inventors")
        time.sleep(2.0)

    # Download one genuine born-digital patent PDF for the PDF/OCR chapter.
    pdf_doc = next((d for d in built if d.get("pdf_url")), None)
    if pdf_doc:
        try:
            raw = get(pdf_doc["pdf_url"])
            (PDF_DIR / f"{pdf_doc['doc_id']}.pdf").write_bytes(raw)
            print(f"PDF: saved genuine {pdf_doc['doc_id']}.pdf ({len(raw)//1024} KB)")
        except Exception as e:  # noqa: BLE001
            print(f"PDF download failed: {e!r}")

    index = [
        {"doc_id": d["doc_id"], "title": d["title"], "cpc": d["cpc"][:2], "n_claims": len(d["claims"])}
        for d in built
    ]
    (CORPUS / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nBuilt {len(built)} patents into {CORPUS}")
    return 0 if len(built) >= 12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
