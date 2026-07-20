# Local RAG Knowledge Base

A fully local, framework-free RAG (Retrieval-Augmented Generation) system for building a personal/organizational knowledge base — drop in documents, query them conversationally, with persistent memory across sessions. No LangChain, no LlamaIndex — every component (chunking, retrieval, reranking, memory) is hand-built for full control over behavior and cost.

## Why

Framework-based RAG stacks are fast to prototype but hard to debug or optimize precisely — you're constrained by the abstraction. This system is built component-by-component so every retrieval, reranking, and memory decision is transparent and tunable. Every feature was added to fix a specific, demonstrated failure mode, and every feature's token cost was measured before being accepted into the pipeline.

## Stack

Python · ChromaDB · sentence-transformers (`all-MiniLM-L6-v2`) · Ollama (`llama3.2`) · CrossEncoder reranker (`ms-marco-MiniLM-L-6-v2`)

Fully local — no API keys, no cloud cost, runs on CPU.

## Architecture

```
INGEST                              QUERY
articles/*.txt                      User question
  ↓                                   ↓
200-word chunks, 50-word overlap    Query Rewriter (resolves pronouns/refs
  ↓                                    using last 2 exchanges, if needed)
sentence-transformers → vectors       ↓
  ↓                                 Embed rewritten question
ChromaDB (vector + text)              ↓
                                     ChromaDB → top 6 candidates
                                       ↓
                                     CrossEncoder rerank → top 2
                                       ↓
                                     Relevance gate (score < -5.0 → refuse,
                                       zero-cost, no generation call)
                                       ↓
                                     Ollama llama3.2 → streamed answer
                                       ↓
                                     Answer + Sources + Excerpts + Token cost
                                       ↓
                                     Memory layers updated (see below)
```

## Memory architecture (three layers)

Modeled independently on working / episodic / semantic memory from cognitive science:

- **Layer 1 — Working memory**: last 2 exchanges, JSON, survives process restarts. Zero token cost (pure disk I/O).
- **Layer 2 — Episodic memory**: append-only JSONL log of every exchange, rolling 100-exchange cap (LIFO). Also embedded into a separate ChromaDB collection, queryable via `recall:` prefix or automatic detection ("what did I conclude about X?").
- **Layer 3 — Semantic memory**: distilled personality/preference profile, updated every 25 exchanges, soft-capped at ~700 tokens. Selectively injected only when the question's topic matches the profile.

## Key engineering decisions

- **Query rewriting** resolves conversational references (pronouns, "these three," "going back to X") before retrieval, fixing context-blindness that caused HyDE to hallucinate unrelated content.
- **Relevance gate**: reuses the CrossEncoder score already computed during reranking to refuse out-of-scope questions with zero additional generation cost.
- **TokenLedger**: every component's cost is tracked per-exchange and per-session, so every claim about cost/benefit in this project is backed by measured numbers, not estimates.
- **Deterministic routing over LLM judgment calls** wherever possible (e.g. `recall:` keyword routing instead of an LLM classifying intent) — same reliability at a fraction of the cost.

## Known limitations (documented, not hidden)

- Reranker has a hard 6-chunk recall ceiling and scores chunks in isolation (no joint-relevance reasoning). Not an issue at current knowledge base size; flagged for revisit at scale.
- Local 3B-class model (`llama3.2`) reliably handles literal pattern-matching (query rewriting, gate refusal) but is unreliable at compound judgment calls — evidenced independently in four places: query rewriting edge cases, semantic profile distillation quality, refusal-calibration variance, and rerank-score variance from non-deterministic rewrite phrasing. This is treated as a documented model-capability ceiling, not chased further.
- Layer 3 (semantic memory) captures topic frequency well but struggles with single-pass interpretive synthesis (e.g. explicitly stated preferences aren't reliably extracted into the profile).

## Design philosophy

Every feature follows a build → measure → compare cost vs. impact → decide cycle. Several capabilities (BM25 hybrid search, multi-query retrieval, LLM-based routing) were deliberately deferred after evaluation — they solve problems that only bite at larger scale, and adding them now would add cost and complexity without a corresponding gain at this knowledge base size.

## Setup

```bash
pip install chromadb sentence-transformers ollama
ollama pull llama3.2
python ingest.py       # ingest articles/*.txt into ChromaDB
python query.py        # start interactive query session
```

## Status

Core pipeline, query rewriting, all three memory layers, queryable episodic memory, and the relevance gate are built and verified against real (not simulated) test runs. Closed for now — remaining known gaps are documented above rather than actively worked.
