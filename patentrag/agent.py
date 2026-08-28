"""A single controlled patent agent + tool schemas + trajectory capture (SPEC Part XXI).

A fixed RAG pipeline always runs the same steps; an agent *chooses* tools based on the query.
We keep it deliberately small and deterministic (one controlled agent, rule-based planner) so
the trajectory is inspectable and reproducible — the point is to teach agent structure and
evaluation, not to build a sprawling multi-agent system.

Tool arguments are validated by Pydantic schemas (strict structured tool calls). Every tool is
read-only and carries a required-permission tag consumed by the Chapter-12 tool guardrails.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Tool argument schemas (Pydantic / JSON-Schema)
# ---------------------------------------------------------------------------

class SearchPatentsArgs(BaseModel):
    query: str = Field(min_length=2, max_length=400)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchClaimsArgs(BaseModel):
    query: str = Field(min_length=2, max_length=400)
    top_k: int = Field(default=5, ge=1, le=50)


class FetchPatentArgs(BaseModel):
    publication_number: str

    @field_validator("publication_number")
    @classmethod
    def _valid(cls, v: str) -> str:
        if not re.match(r"^[A-Z]{2}\d{5,}[A-Z]\d?$", v.strip()):
            raise ValueError(f"invalid patent publication number: {v!r}")
        return v.strip()


class LookupClassificationArgs(BaseModel):
    publication_number: str


class RetrievePassagesArgs(BaseModel):
    query: str = Field(min_length=2, max_length=400)
    top_k: int = Field(default=8, ge=1, le=50)


# ---------------------------------------------------------------------------
# Tool + registry
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    schema: type[BaseModel]
    fn: Callable[..., Any]
    required_permission: str = "read"  # consumed by the tool-permission guardrail
    mutating: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def call(self, name: str, args: dict) -> Any:
        tool = self._tools[name]
        validated = tool.schema(**args)  # raises on malformed args
        return tool.fn(**validated.model_dump())


# ---------------------------------------------------------------------------
# Backend: bind tools to the retrieval + corpus stack
# ---------------------------------------------------------------------------

def build_backend():
    """Assemble retrievers + corpus indices and a populated :class:`ToolRegistry`."""
    from . import bootstrap as bs
    from .dense import DenseRetriever
    from .fusion import reciprocal_rank_fusion

    docs = {d.doc_id: d for d in bs.ensure("docs_canonical")}
    chunks = bs.ensure("chunks")
    by_id = {c.chunk_id: c for c in chunks}
    bm25 = bs.ensure("bm25_index")
    emb = bs.ensure("embeddings")
    dense = DenseRetriever(emb["chunk_ids"], emb["matrix"])

    def hybrid(query: str, top_k: int):
        b = [cid for cid, _ in bm25.search(query, 40)]
        d = [cid for cid, _ in dense.search(query, 40)]
        fused = reciprocal_rank_fusion([b, d])
        return [by_id[cid] for cid, _ in fused[:top_k]]

    reg = ToolRegistry()

    def search_patents(query: str, top_k: int = 5):
        seen, out = set(), []
        for c in hybrid(query, top_k * 4):
            if c.document_id in seen:
                continue
            seen.add(c.document_id)
            doc = docs[c.document_id]
            out.append({"publication_number": doc.publication_number, "title": doc.title})
            if len(out) >= top_k:
                break
        return out

    def search_claims(query: str, top_k: int = 5):
        claim_chunks = [c for c in hybrid(query, top_k * 6) if c.claim_number is not None]
        return [{"publication_number": c.publication_number, "claim_number": c.claim_number,
                 "chunk_id": c.chunk_id, "text": c.text[:200]} for c in claim_chunks[:top_k]]

    def fetch_patent(publication_number: str):
        doc = next((d for d in docs.values() if d.publication_number == publication_number), None)
        if not doc:
            return {"error": "not found", "publication_number": publication_number}
        return {"publication_number": doc.publication_number, "title": doc.title,
                "abstract": doc.abstract, "n_claims": len(doc.claims), "cpc": doc.cpc[:5],
                "publication_date": str(doc.publication_date)}

    def lookup_classification(publication_number: str):
        doc = next((d for d in docs.values() if d.publication_number == publication_number), None)
        return {"publication_number": publication_number, "cpc": doc.cpc if doc else []}

    def retrieve_passages(query: str, top_k: int = 8):
        return [{"chunk_id": c.chunk_id, "publication_number": c.publication_number,
                 "section": c.section, "text": c.text} for c in hybrid(query, top_k)]

    reg.register(Tool("search_patents", SearchPatentsArgs, search_patents))
    reg.register(Tool("search_claims", SearchClaimsArgs, search_claims))
    reg.register(Tool("fetch_patent", FetchPatentArgs, fetch_patent))
    reg.register(Tool("lookup_classification", LookupClassificationArgs, lookup_classification, required_permission="researcher"))
    reg.register(Tool("retrieve_passages", RetrievePassagesArgs, retrieve_passages))
    return reg, docs, by_id, hybrid


# ---------------------------------------------------------------------------
# Agent + trajectory
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    tool: str
    args: dict
    result: Any = None
    error: Optional[str] = None


@dataclass
class AgentTrajectory:
    query: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: Optional[Any] = None

    @property
    def tool_sequence(self) -> list[str]:
        return [s.tool for s in self.steps]


_PUBNUM = re.compile(r"\b([A-Z]{2}\d{5,}[A-Z]\d?)\b")


class PatentAgent:
    """Rule-based planner over the tool registry. Deterministic and fully traced."""

    def __init__(self, registry: ToolRegistry, generate_fn: Callable | None = None) -> None:
        self.registry = registry
        self.generate_fn = generate_fn

    def plan(self, query: str) -> list[tuple[str, dict]]:
        """Choose an ordered tool plan from surface features of the query."""
        m = _PUBNUM.search(query)
        plan: list[tuple[str, dict]] = []
        if m:  # a specific patent is named → fetch it, then its classification
            plan.append(("fetch_patent", {"publication_number": m.group(1)}))
            if "classif" in query.lower() or "cpc" in query.lower():
                plan.append(("lookup_classification", {"publication_number": m.group(1)}))
        if re.search(r"\bclaim", query, re.I):
            plan.append(("search_claims", {"query": query, "top_k": 5}))
        # always ground the final answer with passages
        plan.append(("retrieve_passages", {"query": query, "top_k": 8}))
        return plan

    def run(self, query: str, permission: str = "researcher") -> AgentTrajectory:
        traj = AgentTrajectory(query=query)
        passages = []
        for tool_name, args in self.plan(query):
            step = AgentStep(tool=tool_name, args=args)
            try:
                step.result = self.registry.call(tool_name, args)
                if tool_name == "retrieve_passages":
                    passages = step.result
            except Exception as e:  # noqa: BLE001 - captured into the trajectory
                step.error = f"{type(e).__name__}: {e}"
            traj.steps.append(step)
        if self.generate_fn and passages:
            traj.final_answer = self.generate_fn(query, passages)
        return traj
