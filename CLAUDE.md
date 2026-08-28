# CLAUDE.md — Patent RAG Engineering Masterclass (multi-chapter notebook series)

You are building a production-grade educational Jupyter notebook **series** on patent-domain
RAG / LLM / agent engineering. The authoritative content specification is `SPEC.md`.
The chapter decomposition is `CHAPTERS.md`. Read both fully before writing any code.

## Deviation from SPEC.md (intentional, requested by the owner)

SPEC.md describes a single notebook (`patent_rag_engineering_masterclass.ipynb`).
**Override:** the deliverable is instead a series of chapter notebooks in `notebooks/`,
exactly as laid out in `CHAPTERS.md`. Everything else in SPEC.md applies unchanged:
pedagogy pattern, execution discipline, no fabricated outputs, no paid-API requirement,
testing, formulas, tables, visualizations, provenance chain, final benchmark.

The single "Execution & Validation Report" required by SPEC.md becomes:
1. a short validation footer cell at the end of **every** chapter notebook, and
2. a global `VALIDATION_REPORT.md` produced by `scripts/validate_all.py`.

## Target runtime: Google Colab (primary) + local Linux (secondary)

The owner will run these notebooks on **Google Colab, one notebook at a time, each on a
fresh ephemeral kernel and fresh ephemeral disk**. This drives hard requirements:

1. **Every chapter notebook must be independently executable from a blank Colab VM.**
   Cell 1 of every notebook is a standard bootstrap cell (identical pattern everywhere):
   - detect Colab: `IN_COLAB = "google.colab" in sys.modules`
   - on Colab: clone/pull this repo (or, if not yet pushed to GitHub, download the repo
     zip; leave a clearly marked `REPO_URL` constant the owner fills in once), then
     `%pip install -q -r requirements.txt`, and `apt-get install -y -q tesseract-ocr`
     only in chapters that need OCR
   - locally: assume the repo checkout is the CWD and deps are already installed
   - append the repo root to `sys.path` so `import patentrag` works
2. **Cross-chapter state travels through the `artifacts/` directory, never through
   kernel state.** Because Colab disks are ephemeral, every chapter that consumes an
   artifact must be able to **rebuild it cheaply**: call
   `patentrag.bootstrap.ensure(upto=<stage>)`, which builds any missing pipeline stage
   (corpus → parsed docs → normalized+offset-mapped text → chunks → BM25 index → embeddings
   → ANN index → eval dataset) from `data/` in under ~2–3 minutes on CPU. Design the
   corpus size and model choices around this budget.
3. CPU-only must work. GPU (if present) may be used opportunistically, never required.
4. No paid API keys required anywhere (SPEC §11). Provide the `LLMProvider` abstraction
   with a deterministic mock default; optionally support an env-var-configured provider.
   The owner has a `DEEPSEEK_API_KEY` in his environments — support
   `OpenAI-compatible` endpoints via env vars (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`)
   as the optional live path, but every cell must produce meaningful output without it.

## Repository layout you must produce

```
CLAUDE.md  SPEC.md  CHAPTERS.md            # inputs (already present)
README.md                                   # how to run: Colab + local
requirements.txt                            # pinned, verified versions
patentrag/                                  # shared library imported by all chapters
    __init__.py
    bootstrap.py        # ensure(upto=...) artifact builder + Colab setup helper
    models.py           # PatentDocument, PatentSection, PatentChunk, SourceAnchor,
                        # BoundingBox, Citation (Pydantic)
    parsing.py          # XML/ST.36/ST.96 + PDF extraction
    normalize.py        # NFC normalization + OffsetMap
    chunking.py         # section/claim-aware chunking + enrichment
    sparse.py           # inverted index, TF-IDF, BM25 (educational + production)
    dense.py            # embeddings, brute-force, HNSW, quantization helpers
    fusion.py           # RRF, reranking, filtering
    evaluation.py       # P@k, R@k, MRR, NDCG, RAG metrics, agent metrics
    generation.py       # context builder, citation-aware prompting, LLMProvider
    agent.py            # single controlled agent + tool schemas + trajectory capture
    guardrails.py       # input/retrieval/tool/output guardrails
    tracing.py          # request trace model
tests/                                      # pytest suite mirroring patentrag modules
notebooks/
    00_...ipynb ... 14_...ipynb             # per CHAPTERS.md
data/                                       # small real patent corpus + provenance notes
artifacts/                                  # generated; gitignored
scripts/validate_all.py                     # executes every notebook on a fresh kernel
Makefile
```

Notebooks **teach**; `patentrag/` holds implementations that are reused across chapters.
The rule for what lives where: the first time a concept appears, implement the
educational version *visibly in the notebook cell* (per SPEC §16), then show that the
same logic lives in `patentrag/` for later chapters to import. Never make a chapter
re-derive a previous chapter's material — import it and reference the chapter number.

## Data

Follow SPEC §5. Practical guidance: use a small set of **real** patents. Good options,
verify availability at build time:
- Google Patents public XML / USPTO bulk-data single-document samples (patftdata /
  bulkdata.uspto.gov red-book samples contain real ST.36-era grant XML),
- WIPO ST.96 official example instances from wipo.int standards pages,
- HuggingFace `big_patent` or `HUPD` — take a ~30–80 document slice, cache to `data/`
  as JSON with provenance recorded in `data/PROVENANCE.md`.
Bundle the corpus in the repo (small!) so Colab runs never depend on flaky downloads.
At least a few documents must exist as genuine XML (for Part III) and at least one as
PDF (for Part IV) — if no public patent PDF is bundleable, render one of the XML patents
to PDF with reportlab and *say so explicitly in the notebook* (the OCR/layout teaching
still works; provenance of the trick must be honest).

## Execution & validation discipline (non-negotiable)

- Work **one chapter at a time**, in `CHAPTERS.md` order. For each chapter:
  1. write/extend the needed `patentrag/` modules + `tests/`,
  2. run `pytest` for the touched modules,
  3. write the notebook,
  4. execute it end-to-end on a fresh kernel:
     `jupyter nbconvert --to notebook --execute --inplace notebooks/<nb>.ipynb --ExecutePreprocessor.timeout=900`
  5. fix all failures, re-execute from clean state,
  6. only then move to the next chapter.
- After the final chapter, run `python scripts/validate_all.py` (fresh execution of every
  notebook in order) and commit the generated `VALIDATION_REPORT.md`.
- Never claim a cell ran unless it did. Never fabricate outputs, metrics, or benchmark
  numbers (SPEC §24). If something must be skipped (e.g. model too heavy), skip it
  loudly with the exact reason in the notebook and in the report.
- Save notebooks **with outputs** — the owner wants to read them rendered.
- Fixed seeds everywhere randomness exists (SPEC §8).
- Before pinning `requirements.txt`, actually check current versions/maintenance status
  of candidate libraries (SPEC §3) — you have network access to PyPI. Record decisions
  in the Library Decision Table in chapter 00 (SPEC §19).
- Keep total fresh-run wall time of the whole series under ~45 min CPU; any single
  notebook under ~8 min. Choose corpus size and models accordingly.

## Style

- Markdown-heavy notebooks: every concept follows the SPEC §6 eleven-step pattern
  (compressed sensibly for minor concepts — don't produce ritual boilerplate).
- LaTeX for all formulas in SPEC §15, explained symbol by symbol.
- Pandas tables for results; matplotlib only (no seaborn styling dependencies).
- One idea per code cell region. No 300-line mega-cells.
- Type hints, small functions, docstrings in `patentrag/`.

## Git

Initialize a git repo if none exists. Commit after each completed+validated chapter with
message `chapter NN: <title> — executed clean`. Add `artifacts/`, `.ipynb_checkpoints/`,
`__pycache__/` to `.gitignore`. Do not commit anything larger than ~10 MB.
