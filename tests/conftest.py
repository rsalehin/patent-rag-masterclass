"""Shared pytest fixtures: put the repo on sys.path, fix seeds, expose corpus/chunks."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from patentrag import bootstrap as bs  # noqa: E402

bs.set_seeds()


@pytest.fixture(scope="session")
def docs():
    return bs.ensure("docs_canonical")


@pytest.fixture(scope="session")
def chunks():
    return bs.ensure("chunks")


@pytest.fixture(scope="session")
def chunks_by_id(chunks):
    return {c.chunk_id: c for c in chunks}
