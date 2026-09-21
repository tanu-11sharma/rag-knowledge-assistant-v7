"""
Recency-weighted TF-IDF retriever.

Standard RAG retrieval ranks purely by text relevance, which is a problem
for knowledge bases where facts change over time (release notes, policy
updates, pricing pages): an old but well-worded document can outscore the
current, correct one. This retriever blends a stdlib TF-IDF/cosine
relevance score with an exponential recency decay, so that when several
documents are topically similar, the newer one is preferred — while still
surfacing older documents (flagged as such) when nothing recent matches.

No external API key or embeddings service is required; everything here
is standard-library Python.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "and", "or", "but", "if", "with",
    "as", "by", "at", "from", "that", "this", "it", "its", "into",
    "now", "per", "per.",
}


def tokenize(text: str) -> List[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Document:
    id: str
    title: str
    date: date
    text: str


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


class RecencyWeightedRetriever:
    """TF-IDF cosine relevance combined with an exponential recency decay."""

    def __init__(self, documents: List[Document], half_life_days: int = 365, as_of: date | None = None):
        self.documents = documents
        self.half_life_days = half_life_days
        self.as_of = as_of or max(d.date for d in documents)

        self._doc_tokens = [tokenize(d.text) for d in documents]
        df = Counter()
        for tokens in self._doc_tokens:
            for term in set(tokens):
                df[term] += 1
        n_docs = len(documents)
        self._idf = {term: math.log((1 + n_docs) / (1 + count)) + 1.0 for term, count in df.items()}
        self._doc_vectors = [self._vectorize(tokens) for tokens in self._doc_tokens]
        self._doc_norms = [self._norm(v) for v in self._doc_vectors]

    def _vectorize(self, tokens: List[str]) -> Counter:
        tf = Counter(tokens)
        return Counter({t: c * self._idf.get(t, 0.0) for t, c in tf.items()})

    @staticmethod
    def _norm(vec: Counter) -> float:
        return math.sqrt(sum(v * v for v in vec.values())) or 1e-9

    def _cosine(self, a: Counter, a_norm: float, b: Counter, b_norm: float) -> float:
        if len(a) > len(b):
            a, b = b, a
        dot = sum(v * b.get(t, 0.0) for t, v in a.items())
        return dot / (a_norm * b_norm)

    def _recency_weight(self, doc_date: date) -> float:
        age_days = max((self.as_of - doc_date).days, 0)
        # Exponential decay: weight halves every `half_life_days`.
        return 0.5 ** (age_days / self.half_life_days)

    def search(self, query: str, top_k: int = 5, recency_weight: float = 0.4) -> List[Tuple[float, float, float, Document]]:
        """Returns (combined_score, relevance_score, recency_score, document) tuples."""
        q_tokens = tokenize(query)
        q_vec = self._vectorize(q_tokens)
        q_norm = self._norm(q_vec)

        results = []
        for doc, doc_vec, doc_norm in zip(self.documents, self._doc_vectors, self._doc_norms):
            if doc.date > self.as_of:
                # Documents "published" after the as-of date did not exist
                # yet from the perspective of this query and are excluded,
                # rather than being treated as maximally recent.
                continue
            relevance = self._cosine(q_vec, q_norm, doc_vec, doc_norm)
            if relevance <= 0:
                continue
            recency = self._recency_weight(doc.date)
            combined = (1 - recency_weight) * relevance + recency_weight * relevance * recency
            results.append((combined, relevance, recency, doc))

        results.sort(key=lambda r: r[0], reverse=True)
        return results[:top_k]


def load_documents(path: Path) -> List[Document]:
    raw = json.loads(path.read_text())
    return [Document(id=d["id"], title=d["title"], date=_parse_date(d["date"]), text=d["text"]) for d in raw]


def build_retriever(path: Path, as_of: date | None = None) -> RecencyWeightedRetriever:
    return RecencyWeightedRetriever(load_documents(path), as_of=as_of)
