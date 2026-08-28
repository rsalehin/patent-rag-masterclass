"""Environment setup + the cross-chapter artifact contract.

Because the owner runs each notebook on a *fresh, ephemeral* Colab kernel and disk, chapters
cannot pass state through kernel memory. Instead every pipeline stage is materialised under
``artifacts/`` and can be **rebuilt cheaply from ``data/``** by :func:`ensure`. A chapter that
consumes stage *X* simply calls ``patentrag.bootstrap.ensure(upto="X")`` and then does its
teaching on top of the returned handle.

Stage dependency chain (CHAPTERS.md):

    corpus_raw → docs_canonical → pdf_extractions
                              ↘ docs_normalized → chunks → bm25_index
                                                        ↘ embeddings → ann_index
                                                                    ↘ eval_dataset

Design goals: deterministic, CPU-only, total cold rebuild ≲ 2–3 min. Heavy artifacts
(embeddings) are cached as ``.npz``; structured artifacts as JSON/pickle.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """Locate the repository root (the directory containing ``patentrag/`` and ``data/``)."""
    here = Path(__file__).resolve().parent.parent
    return here


REPO = repo_root()
DATA = REPO / "data"
ARTIFACTS = REPO / "artifacts"


# ---------------------------------------------------------------------------
# Colab / environment setup
# ---------------------------------------------------------------------------

IN_COLAB = "google.colab" in sys.modules


def setup_environment(repo_url: str | None = None, need_ocr: bool = False, quiet: bool = True) -> Path:
    """Standard bootstrap used by cell 1 of every notebook.

    Locally: assume the repo checkout is CWD and dependencies are installed; just put the
    repo root on ``sys.path``. On Colab: clone/download the repo, ``pip install -r
    requirements.txt``, optionally ``apt-get install tesseract-ocr``. Returns the repo root.
    """
    global REPO, DATA, ARTIFACTS
    if IN_COLAB:  # pragma: no cover - exercised only on Colab
        target = Path("/content/patent-rag-masterclass")
        if not target.exists():
            if repo_url and repo_url.startswith("http") and "github.com" in repo_url:
                subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target)], check=True)
            elif repo_url:
                _download_zip(repo_url, target)
            else:
                raise RuntimeError(
                    "On Colab you must set REPO_URL to this repo's GitHub URL (or a zip URL). "
                    "See README.md."
                )
        os.chdir(target)
        pip = [sys.executable, "-m", "pip", "install"] + (["-q"] if quiet else []) + ["-r", "requirements.txt"]
        subprocess.run(pip, check=True)
        if need_ocr:
            subprocess.run(["apt-get", "install", "-y", "-q", "tesseract-ocr"], check=False)
        REPO = target
    else:
        REPO = repo_root()
    DATA, ARTIFACTS = REPO / "data", REPO / "artifacts"
    ARTIFACTS.mkdir(exist_ok=True)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    return REPO


def _download_zip(url: str, target: Path) -> None:  # pragma: no cover
    import io
    import urllib.request
    import zipfile

    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    tmp = target.parent / "_zip"
    z.extractall(tmp)
    roots = [p for p in tmp.iterdir() if p.is_dir()]
    (roots[0] if roots else tmp).rename(target)


def find_tesseract() -> Optional[str]:
    """Return a usable tesseract command, or None. Handles the Windows default install path."""
    import shutil

    cmd = shutil.which("tesseract")
    if cmd:
        return cmd
    for cand in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if Path(cand).exists():
            return cand
    return None


def environment_report() -> dict[str, Any]:
    """Structured environment facts for the per-chapter validation footers (SPEC §2)."""
    import platform

    rep: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "in_colab": IN_COLAB,
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        rep["torch"] = None
        rep["cuda_available"] = False
    rep["tesseract"] = find_tesseract() is not None
    return rep


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 20240817) -> None:
    """Fix every RNG we touch, so runs are reproducible (SPEC §8)."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # keep CPU kernels available
    except Exception:  # noqa: BLE001
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus() -> list:
    """Load the bundled real patent corpus into :class:`~patentrag.models.PatentDocument`s."""
    from .models import PatentDocument

    corpus_dir = DATA / "corpus"
    docs = []
    for path in sorted(corpus_dir.glob("US*.json")):
        docs.append(PatentDocument.from_corpus_json(json.loads(path.read_text(encoding="utf-8"))))
    if not docs:
        raise FileNotFoundError(f"No corpus documents under {corpus_dir}. Run scripts/build_corpus.py.")
    return docs


# ---------------------------------------------------------------------------
# Artifact stage framework
# ---------------------------------------------------------------------------

# Registered builders: stage -> (dependencies, builder_fn). Builders are registered by the
# modules that own them (sparse.py registers 'bm25_index', etc.), keeping bootstrap decoupled.
_BUILDERS: dict[str, tuple[tuple[str, ...], Callable[[], Any]]] = {}
_STAGE_ORDER: list[str] = [
    "corpus_raw",
    "docs_canonical",
    "pdf_extractions",
    "docs_normalized",
    "chunks",
    "bm25_index",
    "embeddings",
    "ann_index",
    "eval_dataset",
]


def register_stage(name: str, deps: tuple[str, ...], builder: Callable[[], Any]) -> None:
    _BUILDERS[name] = (deps, builder)


def _artifact_path(stage: str, ext: str = "pkl") -> Path:
    ARTIFACTS.mkdir(exist_ok=True)
    return ARTIFACTS / f"{stage}.{ext}"


def save_artifact(stage: str, obj: Any) -> Path:
    p = _artifact_path(stage)
    with open(p, "wb") as fh:
        pickle.dump(obj, fh)
    return p


def load_artifact(stage: str) -> Any:
    p = _artifact_path(stage)
    with open(p, "rb") as fh:
        return pickle.load(fh)


def has_artifact(stage: str) -> bool:
    return _artifact_path(stage).exists()


# -- builders for the first two stages live here; later stages self-register --

def _build_corpus_raw() -> dict:
    docs = load_corpus()
    manifest = {
        "n_docs": len(docs),
        "doc_ids": [d.doc_id for d in docs],
        "n_claims": sum(len(d.claims) for d in docs),
        "n_sections": sum(len(d.sections) for d in docs),
    }
    return manifest


def _build_docs_canonical() -> list:
    return load_corpus()


register_stage("corpus_raw", (), _build_corpus_raw)
register_stage("docs_canonical", ("corpus_raw",), _build_docs_canonical)


def ensure(upto: str, rebuild: bool = False, verbose: bool = False) -> Any:
    """Build (or load from cache) every stage up to and including ``upto``.

    Returns the artifact for ``upto``. Missing dependencies are built first. This is the single
    entry point chapters use to obtain their inputs on a fresh kernel.
    """
    # Trigger the modules that register later stages (lazy import to avoid heavy deps upfront).
    _lazy_register(upto)

    if upto not in _BUILDERS:
        raise KeyError(f"Unknown stage {upto!r}. Known: {sorted(_BUILDERS)}")

    order = _resolve_order(upto)
    result = None
    for stage in order:
        if has_artifact(stage) and not rebuild:
            if verbose:
                print(f"[ensure] {stage}: cached")
            result = load_artifact(stage)
            continue
        deps, builder = _BUILDERS[stage]
        t0 = time.time()
        obj = builder()
        save_artifact(stage, obj)
        result = obj
        if verbose:
            print(f"[ensure] {stage}: built in {time.time() - t0:.1f}s")
    return result


def _resolve_order(upto: str) -> list[str]:
    seen: list[str] = []

    def visit(s: str) -> None:
        if s in seen:
            return
        for dep in _BUILDERS[s][0]:
            visit(dep)
        seen.append(s)

    visit(upto)
    return seen


def _lazy_register(upto: str) -> None:
    """Import the modules that own later stages so their register_stage() calls run."""
    need = {
        "docs_normalized": "patentrag.normalize",
        "chunks": "patentrag.chunking",
        "bm25_index": "patentrag.sparse",
        "embeddings": "patentrag.dense",
        "ann_index": "patentrag.dense",
        "eval_dataset": "patentrag.evaluation",
        "pdf_extractions": "patentrag.parsing",
    }
    import importlib

    # Import everything up to `upto` in the canonical order that is registered-or-needed.
    for stage, module in need.items():
        if stage not in _BUILDERS:
            try:
                importlib.import_module(module)
            except Exception:  # noqa: BLE001 - stage may not be needed yet
                pass
