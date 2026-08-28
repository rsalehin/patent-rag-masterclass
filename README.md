# Patent RAG Engineering Masterclass

A production-grade, **executable** teaching series that builds a patent-domain
Retrieval-Augmented Generation system from first principles: real patent XML/PDF parsing,
provenance-preserving normalization, structural chunking, sparse + dense + hybrid retrieval,
reranking, evaluation, grounded generation, a controlled agent, five guardrail boundaries, and
end-to-end observability — over a **real, bundled corpus of US patents**.

Fifteen chapter notebooks in [`notebooks/`](notebooks/), each independently runnable on a fresh
Google Colab VM. Shared, tested implementations live in [`patentrag/`](patentrag/); notebooks
*teach* and import them.

| # | Notebook | Topic |
|---|----------|-------|
| 00 | `00_setup_and_architecture` | Architecture, environment, Library Decision Table, corpus tour |
| 01 | `01_patent_document_model` | Canonical Pydantic model (claims, sections, biblio) |
| 02 | `02_xml_st36_st96` | XML/XPath, DTD validation, WIPO ST.36 vs ST.96 (v9.0), XSD validation + version migration |
| 03 | `03_pdf_ocr_layout` | PDF text/words/boxes, OCR vs text layer, layout, bounding-box anchors |
| 04 | `04_normalization_offsets_language` | NFC/NFD/NFKC/NFKD, the reversible **OffsetMap**, language ID, shadow text |
| 05 | `05_chunking_and_enrichment` | Section/claim-aware chunking, contextual enrichment |
| 06 | `06_sparse_retrieval` | Inverted index, TF-IDF, BM25 (from scratch → `rank-bm25`) |
| 07 | `07_dense_retrieval_ann_quantization` | Embeddings, brute force, HNSW, PQ/int8 quantization |
| 08 | `08_late_interaction_hybrid_rerank_filter` | ColBERT MaxSim, RRF, cross-encoder rerank, metadata filtering |
| 09 | `09_retrieval_evaluation` | P@k, R@k, MRR, NDCG (manual) + the retrieval experiment |
| 10 | `10_rag_generation_and_eval` | Context builder, citation-aware prompts, RAG metrics, failure attribution |
| 11 | `11_agent_and_agent_eval` | One controlled agent, tool schemas, trajectory + agent metrics |
| 12 | `12_guardrails` | PII, injection, safety, scope, retrieval authz, tool permissions, output guards |
| 13 | `13_observability_and_e2e` | Request tracing, latency budget, the end-to-end pipeline, **full provenance chain** |
| 14 | `14_ablation_failures_production` | Measured ablation, failure catalogue, checklist, final benchmark |

**Appendices** (zero-to-working guides for individual building blocks):

| # | Appendix | Topic |
|---|----------|-------|
| A1 | `A1_pydantic` | Pydantic from zero — every concept the series' data models rely on |
| A2 | `A2_langchain` | LangChain from zero — messages, LCEL, parsers, tools, retrieval, RAG chains, memory |
| A3 | `A3_langgraph` | LangGraph from zero — state, nodes/edges, cycles, checkpoints, agents, human-in-the-loop |
| A4 | `A4_langsmith` | LangSmith from zero — tracing, datasets, evaluators, the evaluation loop |

The LangChain/LangGraph/LangSmith appendices install `requirements-appendix.txt` themselves and
run **offline with no API key** (deterministic fake models). To use a real model, add a Colab
secret `LLM_API_KEY` (optionally `LLM_BASE_URL`, `LLM_MODEL`) or `OPENAI_API_KEY`; for LangSmith
tracing/eval add `LANGSMITH_API_KEY`.

## Running on Google Colab (primary)

**Launcher:** open [`index.ipynb`](index.ipynb) in Colab — it has one-click "Open in Colab" links
for every chapter, plus an optional cell that executes the whole series in a single runtime.

Each notebook is self-contained and rebuilds any pipeline artifacts it needs from the bundled
`data/` corpus — so you run **one notebook at a time on a fresh VM**, in any order. (Colab runs one
notebook per runtime; to open several, just click several chapter links — each gets its own tab.)

1. **Push this repo to GitHub** (public or private).
2. **Fill in `REPO_URL`** once. Every notebook's first (bootstrap) cell has the same line:
   ```python
   REPO_URL = ""   # e.g. "https://github.com/<you>/patent-rag-masterclass"
   ```
   Set it to your repo URL. One `sed` updates all 15 at once (run locally before pushing, or edit
   in Colab):
   ```bash
   sed -i 's#REPO_URL = ""#REPO_URL = "https://github.com/<you>/patent-rag-masterclass"#' notebooks/*.ipynb
   ```
3. In Colab: **File → Open notebook → GitHub**, pick a chapter, **Runtime → Run all**. The
   bootstrap cell clones the repo, `pip install -r requirements.txt`, installs `tesseract-ocr`
   for the OCR chapter, then rebuilds any needed artifacts from `data/` (≈2–3 min the first time
   a heavy stage like embeddings is built; cached thereafter on that VM).

**Private repo?** `git clone` on Colab needs a token. Create one with **repo** scope at
<https://github.com/settings/tokens>, then in Colab open the **key icon (Secrets)** in the left
sidebar, add a secret named **`GITHUB_TOKEN`**, paste the token, and enable **Notebook access**.
The bootstrap cell reads that secret automatically (the token is never printed). If the repo is
public, no token is needed.

CPU-only works everywhere; a GPU is used opportunistically and never required. No paid API key is
needed — generation defaults to a deterministic mock. To optionally use a live LLM, set
`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (any OpenAI-compatible endpoint, e.g. DeepSeek).

## Running locally

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# OCR (Chapter 03) needs the tesseract binary:
#   Linux:   sudo apt-get install -y tesseract-ocr
#   macOS:   brew install tesseract
#   Windows: winget install UB-Mannheim.TesseractOCR   (auto-detected at its default path)

make test          # run the pytest suite (patentrag/ invariants)
make validate      # execute all notebooks fresh → VALIDATION_REPORT.md
make validate-one CH=07   # execute just one chapter fresh
```

Or execute a single notebook directly:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/07_dense_retrieval_ann_quantization.ipynb
```

## Repository layout

```
patentrag/        shared library (models, parsing, normalize, chunking, sparse, dense,
                  fusion, evaluation, generation, agent, guardrails, tracing, bootstrap)
notebooks/        the 15 chapter notebooks (saved with outputs)
tests/            pytest suite mirroring patentrag/
data/             bundled real corpus: corpus/ (US patents JSON), xml/ (genuine WIPO ST.96),
                  schema/ (ST.96 v9.0 XSDs), pdf/ (genuine patent PDF), PROVENANCE.md
scripts/          build_corpus.py (one-time corpus builder), validate_all.py
DECISIONS.md      library + data research decisions (versions verified 2026-08)
requirements.txt  pinned, Colab-friendly dependencies
```

## Data & provenance

All data is derived from the **public patent record** (US patents; WIPO ST.96 official example +
schemas). Sources and exact retrieval steps are in [`data/PROVENANCE.md`](data/PROVENANCE.md);
the corpus is reproducible via `scripts/build_corpus.py` but **bundled** so Colab never re-fetches.

## Cross-chapter contract

Chapters never share kernel state. Pipeline stages (corpus → docs → normalized → chunks → BM25 →
embeddings → ANN → eval) are materialized under `artifacts/` (gitignored) and rebuilt on demand
by `patentrag.bootstrap.ensure(upto=<stage>)`, so any chapter runs standalone on a blank VM.
