"""Context construction + citation-aware grounded generation (SPEC Part XVIII).

Retrieval ≠ context construction: a good context is deduplicated, source-diverse, ordered and
within budget. Generation is citation-aware — every context passage carries a stable tag and
the model must answer *with* those tags, which makes the output mechanically verifiable
downstream (Chapter 12/13).

No paid API is required: the default :class:`MockLLMProvider` is deterministic and extractive,
so the whole pipeline runs offline; an OpenAI-compatible endpoint is an optional env-gated path.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .models import Citation, PatentChunk


# ---------------------------------------------------------------------------
# LLM provider abstraction
# ---------------------------------------------------------------------------

class LLMProvider:
    """Minimal text-completion interface. Implementations must be swappable."""

    name = "abstract"

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:  # noqa: D401
        raise NotImplementedError


_LABEL_RE = re.compile(r"\[PATENT=[^\]]+\]")
_PASSAGE_RE = re.compile(r"(\[PATENT=[^\]]+\])\n(.*?)(?=\n\[PATENT=|\nQuestion:|\Z)", re.DOTALL)


class MockLLMProvider(LLMProvider):
    """Deterministic, grounded, citation-preserving stand-in for a real LLM.

    It reads the labelled passages out of the prompt and composes an extractive answer from the
    first one or two, echoing their citation tags. Because it only ever uses supplied context,
    its answers are grounded by construction — a useful property for teaching the *rest* of the
    pipeline (groundedness/citation checks) without a live model.
    """

    name = "mock"

    def __init__(self, max_passages: int = 2) -> None:
        self.max_passages = max_passages

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        passages = _PASSAGE_RE.findall(prompt)
        if not passages:
            return "I could not find supporting passages in the provided context."
        parts = []
        for label, text in passages[: self.max_passages]:
            first = re.split(r"(?<=[.!?])\s+", text.strip())[0]
            first = first if len(first) < 320 else first[:317] + "..."
            parts.append(f"{first} {label}")
        return " ".join(parts)


class OpenAICompatibleProvider(LLMProvider):
    """Optional live path for any OpenAI-compatible endpoint (e.g. DeepSeek), via env vars:

    ``LLM_BASE_URL``, ``LLM_API_KEY``, ``LLM_MODEL``. Never required; used only if configured.
    """

    name = "openai-compatible"

    def __init__(self) -> None:
        self.base_url = os.environ.get("LLM_BASE_URL")
        self.api_key = os.environ.get("LLM_API_KEY")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_API_KEY"))

    def complete(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        import json
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]


def default_provider() -> LLMProvider:
    """Live provider if configured, otherwise the deterministic mock (never requires a key)."""
    return OpenAICompatibleProvider() if OpenAICompatibleProvider.available() else MockLLMProvider()


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

@dataclass
class ContextPassage:
    citation: Citation
    text: str
    tokens: int


class ContextBuilder:
    """Turn a ranked chunk list into a deduplicated, source-diverse, budgeted context."""

    def __init__(self, token_budget: int = 1200, max_per_doc: int = 2) -> None:
        self.token_budget = token_budget
        self.max_per_doc = max_per_doc

    def build(self, ranked_chunks: list[PatentChunk]) -> list[ContextPassage]:
        used_tokens = 0
        per_doc: dict[str, int] = {}
        seen_text: set[str] = set()
        out: list[ContextPassage] = []
        for c in ranked_chunks:
            key = c.text[:120]
            if key in seen_text:  # dedup near-identical passages
                continue
            if per_doc.get(c.document_id, 0) >= self.max_per_doc:  # source diversity
                continue
            if used_tokens + c.token_estimate > self.token_budget and out:  # budget
                break
            cit = Citation(
                document_id=c.document_id, publication_number=c.publication_number,
                section=c.section, claim_number=c.claim_number, chunk_id=c.chunk_id,
                anchor=c.anchor,
            )
            out.append(ContextPassage(cit, c.text, c.token_estimate))
            seen_text.add(key)
            per_doc[c.document_id] = per_doc.get(c.document_id, 0) + 1
            used_tokens += c.token_estimate
        return out


SYSTEM_PROMPT = (
    "You are a patent analysis assistant. Answer the question using ONLY the context passages. "
    "Every statement must cite the passage it comes from using that passage's bracket tag "
    "verbatim, e.g. [PATENT=US1234567B2 | CLAIM=1 | CHUNK=abc123]. If the context does not "
    "contain the answer, say so."
)


def build_prompt(query: str, passages: list[ContextPassage]) -> str:
    blocks = [f"{p.citation.label()}\n{p.text}" for p in passages]
    return "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {query}\nAnswer:"


# ---------------------------------------------------------------------------
# Grounded answer
# ---------------------------------------------------------------------------

class PatentAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0

    @property
    def answer_without_tags(self) -> str:
        return _LABEL_RE.sub("", self.answer).replace("  ", " ").strip()


def extract_citations(answer_text: str, passages: list[ContextPassage]) -> list[Citation]:
    """Map the bracket tags the model emitted back to the actual context citations."""
    by_chunk = {p.citation.chunk_id: p.citation for p in passages}
    cites: list[Citation] = []
    for tag in _LABEL_RE.findall(answer_text):
        m = re.search(r"CHUNK=([A-Za-z0-9]+)", tag)
        if m and m.group(1) in by_chunk and by_chunk[m.group(1)] not in cites:
            cites.append(by_chunk[m.group(1)])
    return cites


def generate_answer(query: str, ranked_chunks: list[PatentChunk], provider: LLMProvider | None = None,
                    token_budget: int = 1200) -> PatentAnswer:
    """Full grounded generation: build context → prompt → provider → parse+validate citations."""
    provider = provider or default_provider()
    passages = ContextBuilder(token_budget=token_budget).build(ranked_chunks)
    prompt = build_prompt(query, passages)
    text = provider.complete(prompt, system=SYSTEM_PROMPT)
    cites = extract_citations(text, passages)
    confidence = min(1.0, 0.4 + 0.2 * len(cites)) if cites else 0.1
    return PatentAnswer(answer=text, citations=cites, confidence=confidence)
