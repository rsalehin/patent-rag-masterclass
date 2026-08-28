# CHAPTERS.md — Notebook series structure

Fifteen chapter notebooks in `notebooks/`. Each maps to specific SPEC.md parts.
Chapters must be built and validated **in this order** because later chapters import
`patentrag/` code and consume `artifacts/` produced (or cheaply rebuilt) by earlier
stages. "Artifacts out" lists what `patentrag.bootstrap.ensure()` must be able to
(re)build for that stage.

| # | Notebook | SPEC parts | Contents | Artifacts out |
|---|----------|-----------|----------|---------------|
| 00 | `00_setup_and_architecture.ipynb` | Part I, §19 | Full system diagrams (base pipeline + guardrailed agent pipeline), what each subsystem owns, environment report, **Library Decision Table** with verified current versions, corpus tour + provenance | `corpus_raw` |
| 01 | `01_patent_document_model.ipynb` | Part II | Patent structure (biblio, abstract, description, claims, citations, classifications); why flat text destroys information; Pydantic canonical model: `PatentDocument`, `PatentSection`, `PatentChunk`, `SourceAnchor`, `BoundingBox`, `Citation`; load corpus into the model | `docs_canonical` |
| 02 | `02_xml_st36_st96.ipynb` | Part III | XML fundamentals + XPath with lxml on real patent XML; DTD validation demo; ST.36 history + parsing; ST.96 (verify current version on wipo.int) + XSD validation if feasible; ST.36 vs ST.96 comparison table | — |
| 03 | `03_pdf_ocr_layout.ipynb` | Part IV | PDF as presentation format; text/blocks/words/coords extraction; rasterize → OCR (tesseract) vs text layer, show real errors; one lightweight layout demo; bounding boxes wired into `SourceAnchor` | `pdf_extractions` |
| 04 | `04_normalization_offsets_language.ipynb` | Parts V–VI | Unicode NFC/NFD/NFKC/NFKD; **the OffsetMap** (normalized→original span recovery, Unicode edge-case tests); boilerplate removal without provenance loss; DOM anchoring (XPath round-trip); language ID; shadow-text architecture | `docs_normalized` |
| 05 | `05_chunking_and_enrichment.ipynb` | Parts VII–VIII | Chunking as citability problem; fixed vs overlap vs paragraph vs section-aware vs claim-aware vs hierarchical; implement section/claim-aware chunking preserving all anchors; contextual enrichment + before/after retrieval comparison; chunk-size distribution plot | `chunks` |
| 06 | `06_sparse_retrieval.ipynb` | Part IX | Manual inverted index; TF-IDF derivation; BM25 formula (k1, b, saturation, length norm) — manual then production impl; ranked runs over patent chunks; why BM25 stays vital for patents | `bm25_index` |
| 07 | `07_dense_retrieval_ann_quantization.ipynb` | Parts X–XII | Embeddings + similarity math; brute-force dense retrieval; lexical vs semantic comparison; exact vs ANN; HNSW theory (M, efConstruction, efSearch) + real index; recall-vs-latency plot; memory math for 100M vectors; PQ/int8 with FAISS, accuracy/memory tradeoff | `embeddings`, `ann_index` |
| 08 | `08_late_interaction_hybrid_rerank_filter.ipynb` | Parts XIII–XVI | Bi-encoder information loss; ColBERT MaxSim (educational impl with token embeddings, production architecture described); queries where BM25 wins / dense wins; **RRF** manual impl + fusion table; cross-encoder rerank top50→top10 with before/after; metadata filtering; pre- vs post-filter experiment | `fused_runs` |
| 09 | `09_retrieval_evaluation.ipynb` | Part XVII | Small labelled patent-query benchmark (query, relevant ids, reference answer+evidence); derive+implement P@k, R@k, MRR, DCG/NDCG manually; the retrieval experiment table (BM25 / dense / hybrid / +reranker × recall, MRR, NDCG, latency) + interpretation | `eval_dataset`, `retrieval_metrics` |
| 10 | `10_rag_generation_and_eval.ipynb` | Parts XVIII–XX | Context builder (dedup, diversity, budget, ordering); citation-aware prompts `[PATENT=…|CLAIM=…|CHUNK=…]`; `LLMProvider` (mock default, env-configurable live); grounded generation; LLM-eval difficulty (judge biases); deterministic evals; semantic evals; LLM-as-judge rubric; RAG metric decomposition (context precision/recall, faithfulness, answer relevance) manual-first then one framework (Ragas or DeepEval — pick per current status); failure attribution examples A–E | `rag_runs` |
| 11 | `11_agent_and_agent_eval.ipynb` | Parts XXI–XXII | Fixed pipeline vs agentic retrieval; one controlled agent with tools (`search_patents`, `search_claims`, `fetch_patent`, …); Pydantic tool schemas; full trajectory capture; agent eval: final answer, tool selection, argument correctness, result usage, trajectory match variants; agent metrics table | `agent_traces` |
| 12 | `12_guardrails.ipynb` | Parts XXIII–XXVIII | Five trust boundaries; PII detection (Presidio or justified alternative) + reversible masking; prompt-injection detection + adversarial eval (P/R/F1); content safety; scope validation (rules vs embeddings vs classifier vs LLM); retrieved-content injection as data-not-instruction; retrieval authorization; tool schema/parameter/permission guards (guest/researcher/admin, deterministic enforcement); output guards: groundedness via claim decomposition, deterministic citation verification, output PII, safety, structured output validation; red-team dataset + guardrail confusion matrix, over-refusal discussion | `guardrail_eval` |
| 13 | `13_observability_and_e2e.ipynb` | Parts XXIX–XXXI | Full request trace model (all stages, latencies, scores, decisions); OpenTelemetry concepts; PII-safe telemetry; measured latency budget table; scaling discussion 1K→100M; **the end-to-end `patent_rag.query()` pipeline** with every intermediate stage displayed; full provenance chain resolution (answer claim → chunk → normalized offsets → original offsets → section → XML node/PDF page → bbox) per SPEC §20 | `e2e_traces` |
| 14 | `14_ablation_failures_production.ipynb` | Parts XXXII–XXXV, §21–22 | Ablation study with charts (BM25 → +dense → +RRF → +reranker → +structural chunking → +enrichment), measured only; "What Still Goes Wrong" failure catalogue (symptom/root cause/detecting metric/fix); production checklist; final mental model; **final benchmark table** with measured values; global execution summary | `final_benchmark` |

## Cross-chapter rules

- Every notebook begins with the standard bootstrap cell (see CLAUDE.md) and a one-cell
  "Where we are in the system" mini-diagram highlighting this chapter's stage.
- Every notebook ends with: (a) a `pytest`-style assertion cell for the chapter's key
  invariants, (b) a validation footer printing python version, package versions used in
  this chapter, runtime, and `CHAPTER NN VALIDATION: PASS`.
- Chapters reference each other by number ("as built in Chapter 05") — no re-derivation.
- SPEC §18 concept-distinction call-outs go where they naturally arise: retrieval vs
  answer relevance etc. in 09–10; authn vs authz etc. in 12; eval-type distinctions in
  09/10/11/12 respectively.
