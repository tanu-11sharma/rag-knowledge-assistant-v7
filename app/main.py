"""
FastAPI app exposing the recency-weighted RAG assistant.

Run with: uvicorn app.main:app --reload
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from app.qa import RecencyRagAssistant

app = FastAPI(
    title="Recency-Weighted RAG Assistant",
    description=(
        "A RAG Q&A agent over a bundled synthetic product release-notes "
        "knowledge base. Retrieval blends text relevance with a recency "
        "decay so answers prefer the most current documentation, and "
        "flags answers that a newer document might supersede."
    ),
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_assistant() -> RecencyRagAssistant:
    return RecencyRagAssistant.from_data_file()


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    recency_weight: float = 0.4


class CitationResponse(BaseModel):
    doc_id: str
    title: str
    date: str
    relevance: float
    recency: float
    combined_score: float


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationResponse]
    possibly_outdated: bool
    newer_alternative: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    assistant = get_assistant()
    result = assistant.ask(request.question, top_k=request.top_k, recency_weight=request.recency_weight)
    return AskResponse(
        query=result.query,
        answer=result.answer,
        citations=[
            CitationResponse(
                doc_id=c.doc_id, title=c.title, date=c.date,
                relevance=c.relevance, recency=c.recency, combined_score=c.combined_score,
            )
            for c in result.citations
        ],
        possibly_outdated=result.possibly_outdated,
        newer_alternative=result.newer_alternative,
    )
