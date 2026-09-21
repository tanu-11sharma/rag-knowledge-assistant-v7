"""
Answer synthesis over recency-weighted retrieval results.

The answer is built extractively from the top-ranked (relevance + recency)
document. If a strictly newer document among the retrieved candidates
covers the same topic but scored lower on text relevance, the answer is
flagged as "a newer document may supersede this" so the caller knows to
double check — this is the core "stale answer" problem RAG systems hit
over changelog/policy-style knowledge bases, made explicit here instead of
silently returning whichever chunk happened to score highest on text
similarity alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.retriever import Document, RecencyWeightedRetriever, build_retriever

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "release_notes.json"


@dataclass
class Citation:
    doc_id: str
    title: str
    date: str
    relevance: float
    recency: float
    combined_score: float


@dataclass
class Answer:
    query: str
    answer: str
    citations: List[Citation]
    possibly_outdated: bool
    newer_alternative: Optional[str] = None


def synthesize(query: str, results: list) -> Answer:
    if not results:
        return Answer(
            query=query,
            answer="I couldn't find anything in the knowledge base relevant to that question.",
            citations=[],
            possibly_outdated=False,
        )

    citations = [
        Citation(
            doc_id=doc.id,
            title=doc.title,
            date=doc.date.isoformat(),
            relevance=round(relevance, 4),
            recency=round(recency, 4),
            combined_score=round(combined, 4),
        )
        for combined, relevance, recency, doc in results
    ]

    top_combined, top_relevance, top_recency, top_doc = results[0]
    answer_text = f"As of {top_doc.title} ({top_doc.date.isoformat()}): {top_doc.text}"

    # Staleness check: is there a strictly newer document among the other
    # retrieved candidates that was topically relevant but ranked lower
    # purely because of recency weighting favoring an even-newer top hit,
    # or conversely a newer doc that scored lower on relevance alone?
    newer_candidates = [
        (combined, relevance, recency, doc)
        for combined, relevance, recency, doc in results[1:]
        if doc.date > top_doc.date and relevance >= top_relevance * 0.6
    ]
    possibly_outdated = False
    newer_alternative = None
    if newer_candidates:
        possibly_outdated = True
        newest = max(newer_candidates, key=lambda r: r[3].date)
        newer_alternative = f"{newest[3].title} ({newest[3].date.isoformat()})"
        answer_text += (
            f" Note: {newer_alternative} also discusses this topic and is more recent — "
            "double-check it in case it supersedes this answer."
        )

    return Answer(
        query=query,
        answer=answer_text,
        citations=citations,
        possibly_outdated=possibly_outdated,
        newer_alternative=newer_alternative,
    )


class RecencyRagAssistant:
    def __init__(self, retriever: RecencyWeightedRetriever):
        self.retriever = retriever

    @classmethod
    def from_data_file(cls, data_path: Path = DATA_PATH) -> "RecencyRagAssistant":
        return cls(build_retriever(data_path))

    def ask(self, query: str, top_k: int = 5, recency_weight: float = 0.4) -> Answer:
        results = self.retriever.search(query, top_k=top_k, recency_weight=recency_weight)
        return synthesize(query, results)
