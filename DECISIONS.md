# DECISIONS.md — research phase (library + data selection)

Date of research: **2026-08-28**. All versions verified against PyPI / official docs and
actually installed + smoke-tested on Python 3.12 (Windows, CPU) before pinning. This file
records *why* each choice was made; the machine-readable pins live in `requirements.txt`
and the in-notebook version is the **Library Decision Table** in Chapter 00.

## Standards research

| Item | Finding (2026-08) | Source |
|------|-------------------|--------|
| WIPO **ST.96** current version | **v9.0**, approved by the CWS XML4IP Task Force on **2025-04-01**. XSD-based, namespaced, componentized (Common + Patent + …). | wipo.int/standards/en/st96/v9-0 |
| WIPO **ST.36** | Legacy DTD-based patent XML exchange standard; still required for older corpora. | wipo.int/standards/en/st36 |
| USPTO grant XML | ICE / "red book" full-text XML, DTD-based (ST.36 lineage), current DTD **v4.7** (2022–present). | data.uspto.gov |

The bundled genuine ST.96 instance (`data/xml/ST96_PatentPublication_Example.xml`) is WIPO's
official Annex VII example — it is real US Patent **8,936,998 B2** rendered in ST.96 and is
labelled `st96Version="V8_0"`, which lets Chapter 02 teach **schema version drift** by
validating it against the genuine v9.0 XSD (two real, explainable errors) and then migrating
it to a clean pass.

## Library Decision Table (rationale)

| Problem | Selected | Alternatives considered | Why selected |
|---|---|---|---|
| XML parse/XPath/validation | **lxml** 6.1 | xml.etree, xmlschema | Full XPath 1.0, DTD + XSD validation, C-speed; the de-facto standard. |
| PDF text/coords | **PyMuPDF** 1.27 (+ **pdfplumber** for word boxes) | pypdf, pdfminer.six, pypdfium2 | PyMuPDF gives words+bbox+blocks fastest; pdfplumber cross-checks word boxes. |
| OCR | **pytesseract** → Tesseract 5 | easyocr, rapidocr, PaddleOCR | Tesseract is the maintained baseline, gives per-word confidence + boxes, light on CPU; installed via winget locally / `apt-get tesseract-ocr` on Colab. |
| PDF synthesis (fallback) | **reportlab** 5.0 | fpdf2, weasyprint | Only used if a genuine patent PDF cannot be fetched; a real one is bundled so this is a documented fallback. |
| Unicode normalization | stdlib **unicodedata** | ftfy, unidecode | NFC/NFD/NFKC/NFKD are stdlib; ftfy/unidecode are *destructive* and wrong for provenance. |
| Embeddings | **sentence-transformers** + `all-MiniLM-L6-v2` (384-d) | bge-small-en-v1.5, e5-small-v2, gte-small | MiniLM is the most reproducible CPU-friendly baseline; bge/e5/gte named as stronger current upgrades (need query/passage prefixes). |
| Sparse / BM25 | **rank-bm25** 0.2.2 (after a from-scratch impl) | bm25s, Pyserini/Lucene, Elasticsearch | rank-bm25 is pure-Python, zero-native, ideal for teaching; bm25s/Lucene named as the production path. |
| ANN / HNSW | **faiss-cpu** 1.15 (`IndexHNSWFlat`, `IndexPQ`, `IndexScalarQuantizer`) | hnswlib, ScaNN, Annoy | FAISS covers HNSW **and** quantization (Parts XI–XII) in one native dep — avoids a second compiled package (hnswlib). |
| Reranker | **CrossEncoder** `ms-marco-MiniLM-L-6-v2` | bge-reranker-base, mxbai-rerank | Classic, tiny, CPU-fast cross-encoder; bge/mxbai named as stronger heavier options. |
| PII | **Microsoft Presidio** (analyzer+anonymizer) + spaCy `en_core_web_sm` | regex-only, GLiNER, transformer NER | Presidio = recognizers + context + reversible anonymization; regex shown as insufficient baseline. |
| Prompt-injection | **transparent heuristic detector** (evaluated), production DeBERTa described | protectai/deberta-v3 prompt-injection, Llama-Prompt-Guard | The classifier is a ~440 MB model needing sentencepiece; for a deterministic, offline, sub-45-min series we implement + evaluate a heuristic and describe the model path (skipped loudly with the exact reason). |
| Content safety | **heuristic lexicon classifier** (evaluated), Detoxify/Llama-Guard described | Detoxify, Llama-Guard-3, OpenAI mod | Same budget rationale as injection; thresholds + false positives are shown on the heuristic. |
| Language ID | **langdetect** 1.0.9 | fasttext lid.176, lingua | Pure-Python, deterministic with a fixed seed; fasttext named for scale. |
| RAG eval | **manual metrics** (context precision/recall, faithfulness, answer relevance) + framework described | Ragas, DeepEval, TruLens | Metrics implemented from first principles for transparency; live Ragas/DeepEval need an LLM judge → shown as an optional, env-gated path (skipped loudly without a key). |
| Agent eval | **deterministic metrics** (tool P/R/F1, arg correctness, trajectory match) | AgentEvals, LangSmith | Deterministic + reproducible; framework concepts described. |
| LLM provider | **`LLMProvider` abstraction**, deterministic **mock** default | OpenAI/DeepSeek/Anthropic | Core notebook must run with **no paid key**; OpenAI-compatible live path is env-gated (`LLM_BASE_URL/LLM_API_KEY/LLM_MODEL`). |

## Data

- **Main corpus (~30 patents):** real, public US patents about the very technologies the
  series teaches (retrieval, ANN, embeddings, OCR, NLP), fetched from **Google Patents**
  (aggregator of public patent record) and bundled as JSON under `data/corpus/`. Fields are
  real: title, abstract, inventors (incl. genuine Unicode names), assignees, CPC, priority/
  filing/publication dates, citations, full independent/dependent claims, and section-split
  description. Built by `scripts/build_corpus.py` (run once); Colab never re-fetches.
- **Genuine XML (Part III):** WIPO ST.96 v9.0 official example instances
  (`PatentPublication_Example.xml` = real US 8,936,998; plus design/GI/trademark examples)
  and the official flattened v9.0 XSD package under `data/schema/`.
- **Genuine PDF (Part IV):** a real born-digital patent PDF downloaded from Google Patents'
  `patentimages` store (its text layer + rasterized-page OCR are compared). reportlab is the
  documented fallback only.
- **Provenance:** `data/PROVENANCE.md`.

## Runtime posture

CPU-only is the contract; GPU is opportunistic and **never required**. Embedding/reranker
inference is forced to CPU for determinism. Fixed seeds everywhere. Models (MiniLM,
ms-marco cross-encoder, spaCy) download on first use (Colab has network); only the *patent
corpus* is bundled, because patent-site downloads are the flaky part.
