# RAG Knowledge Assistant v7 — Recency-Weighted Retrieval

A retrieval-augmented Q&A agent over a synthetic product release-notes
knowledge base, with one specific twist: retrieval blends **text relevance**
with an **exponential recency decay**, so that when several documents are
topically similar (e.g. five release notes all mentioning "rate limit"),
the most current one is preferred — and if an older document is returned
alongside a newer, related one, the answer is explicitly flagged as
**possibly outdated**.

## Why this project

Most RAG demos rank purely by text similarity, which silently breaks on
knowledge bases where facts change over time — changelogs, pricing pages,
policy documents. An older document can be worded closer to the query and
outrank the current, correct answer. This project makes that failure mode
visible and fixes it: each retrieved candidate carries both a `relevance`
score and a `recency` score, the two are blended into a `combined_score`,
and the QA layer checks whether a newer, topically related document exists
before finalizing an answer — surfacing a `possibly_outdated` flag and a
pointer to the newer alternative when it does.

No external LLM or embeddings API key is required — retrieval (TF-IDF +
cosine similarity) and the recency decay are both implemented from the
Python standard library, so the whole thing is deterministic and runs
offline.

## What's inside

- `app/retriever.py` — TF-IDF/cosine relevance scoring blended with exponential recency decay (stdlib only)
- `app/qa.py` — extractive answer synthesis, citations, and the "possibly outdated" staleness check
- `app/main.py` — FastAPI app exposing `/ask` and `/health`
- `app/cli.py` — command-line interface for quick local queries
- `data/release_notes.json` — seven synthetic release notes for a fictional product ("Cascade"), spanning 2023–2026, with several overlapping/evolving topics (rate limits, dark mode, session timeouts)
- `tests/` — pytest suite covering recency weighting, staleness flagging, and the API

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

As an HTTP API:

```bash
uvicorn app.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the ingest API rate limit?"}'
```

Or as a CLI, no server needed:

```bash
python3 -m app.cli "What is the ingest API rate limit?"
```

Example output — five release notes mention rate limits, but the most
recent one wins:

```
A: As of Cascade v5.2 Release Notes (2026-05-20): The default per-workspace
ingest rate limit is raised to 3000 requests per minute...

Retrieved candidates (combined = relevance + recency):
  - [rn-2026-05] Cascade v5.2 Release Notes (2026-05-20) combined=0.38 relevance=0.38 recency=1.0
  - [rn-2025-02] Cascade v4.5 Release Notes (2025-02-18) combined=0.27 relevance=0.35 recency=0.42
  - [rn-2024-06] Cascade v4.0 Release Notes (2024-06-03) combined=0.22 relevance=0.32 recency=0.26
  ...
```

## Test

```bash
pytest -v
```

10 tests cover: recency weighting overriding pure-relevance ranking,
`recency_weight=0` degrading gracefully to plain TF-IDF ranking, the
staleness flag firing (and not firing) correctly at different points in
time, and the HTTP API contract.

## Notes / scope

- All release notes in `data/release_notes.json` are synthetic, written
  for this demo — "Cascade" is not a real product.
- No external API keys are required to run anything in this repo.
- This is the 7th build in an ongoing series of small RAG demos exploring
  different retrieval strategies (see the `rag-knowledge-assistant` family
  of repos); this one's specific contribution is time-aware retrieval and
  explicit staleness detection, not covered by the earlier versions.
- This is a demo/reference implementation, not a production system: the
  in-memory index rebuilds on every process start.
