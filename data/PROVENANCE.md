# Data provenance

All data bundled here is derived from the **public patent record**. Patent documents are
public. Sources and exact retrieval steps are recorded below so the corpus is fully
reproducible via `scripts/build_corpus.py`. Colab/offline runs never re-fetch — they read
these bundled files.

## `corpus/` — main RAG corpus (real US patents)

- **Source:** Google Patents (`https://patents.google.com/patent/<PUB>/en`), used purely as an
  aggregator of the public patent record.
- **Selection:** deliberately on-theme with the technologies this series teaches
  (information retrieval, ANN/vector search, embeddings, indexing, OCR, NLP). Built by
  `scripts/build_corpus.py` from a set of topic queries plus explicit seeds; the exact
  publication numbers selected are the filenames in this directory, and each JSON records its
  own `source.url` and `source.fetched` date (2026-08-28).
- **Fields (all real):** title, abstract, inventors (including genuine Unicode names, e.g.
  "Mert Öz", "Herwig Häle" — used in the Unicode-normalization chapter), assignees, CPC codes,
  priority/filing/publication dates, backward citations, full claim set with
  independent/dependent structure, and the description split into its real sections
  (TECHNICAL FIELD, BACKGROUND, SUMMARY, DETAILED DESCRIPTION, …). Description length is capped
  per document to keep the bundle small; section structure is preserved.
- **Note:** Google Patents rate-limits bursts (HTTP 503); the builder backs off politely and
  skips any document it cannot fetch cleanly, so the corpus size is "as many as fetched"
  (≥ 12 required). `index.json` lists what was built.

## `xml/` — genuine patent XML (WIPO ST.96)

- **Source:** WIPO Standard **ST.96 v9.0**, Annex VII official *example XML instances* package
  (`03-96-vii-example-instances.zip`, wipo.int).
- **`PatentPublication_Example.xml` / `ST96_PatentPublication_Example.xml`:** WIPO's official
  worked example — it is **real US Patent 8,936,998 B2** rendered in ST.96 (contains MathML).
  It declares `st96Version="V8_0"`, which Chapter 02 uses to teach schema **version drift**:
  validating it against the genuine v9.0 XSD produces two real, explainable errors, after
  which it is migrated to a clean pass.
- Also included: `DesignPatentPublication_Example.xml`, `DesignApplication_Example.xml`,
  `GIApplication_Example.xml`, `TrademarkApplication_Example.xml` (official WIPO examples,
  used to show the shared ST.96 Common components across IP types).

## `schema/` — genuine ST.96 v9.0 XML Schemas

- **Source:** WIPO ST.96 v9.0 Annex III, official *flattened* schema package
  (`ST96XMLSchema_V9_0_Flattened.zip`, wipo.int). Flattened variants inline imports so a
  single-file XSD can validate an instance with `lxml`. `ST96_ExampleInstances_V9_0.zip` is the
  example-instances package the XMLs above were extracted from.

## `pdf/` — genuine born-digital patent PDF

- **Source:** the `patentimages.storage.googleapis.com` PDF linked from the Google Patents page
  of one corpus patent (its `citation_pdf_url`). Chapter 03 extracts its text layer + word
  bounding boxes, then rasterizes a page and runs OCR to compare the two. A reportlab-rendered
  PDF is the documented fallback only and was **not** needed (a real PDF was obtained).

## Licensing

US patent documents are works of the U.S. federal government / public record and are not
subject to copyright. WIPO ST.96 standard materials are published by WIPO for public use in
implementing the standard. This repository redistributes only small excerpts sufficient for
education and records provenance for each.
