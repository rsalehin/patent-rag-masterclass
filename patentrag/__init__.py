"""patentrag — shared library for the Patent RAG Engineering Masterclass notebook series.

Notebooks *teach*; this package holds the reusable, tested implementations that later
chapters import instead of re-deriving. The first time a concept appears it is implemented
visibly in the notebook cell, then shown to live here.

Import surface is kept lazy: importing :mod:`patentrag` does not pull in heavy optional
dependencies (torch, faiss, presidio). Import the submodule you need.
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import models  # lightweight (pydantic only) — safe to import eagerly

__all__ = ["models", "__version__"]
