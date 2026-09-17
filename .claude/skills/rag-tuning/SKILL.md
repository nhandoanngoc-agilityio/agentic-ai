---
name: rag-tuning
description: Use when changing chunking, embedding, retrieval, or reranking behaviour, or when research findings look irrelevant. Lists the knobs, the code, the reseed step, and the tests.
---

# Tuning the RAG pipeline

Knobs (all in `src/market_research_team/config.py`, overridable via `.env`):
- `section_chunk_size` (2000), `leaf_chunk_size` (800), `leaf_chunk_overlap` (120)
- `embedding_model_name` (`sentence-transformers/all-MiniLM-L6-v2`)
- `reranker_model_name` (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

Code path: `ingestion/loaders.py` → `ingestion/chunking.py` (`chunk_documents`, hierarchical parent/leaf) → `ingestion/index_build.py` → at query time `retrieval/query_rewriter.py` → `retrieval/retriever.py` (`retrieve_for_queries`) → `retrieval/reranker.py` (`rerank`).

Procedure:
1. Change the knob or code.
2. If chunking or embedding changed, reseed: `python scripts/seed_vectorstore.py` (rebuilds `data/vectorstore/` and `data/processed/`).
3. Run hermetic tests: `pytest tests/ingestion/test_chunking.py tests/retrieval/`.
4. To judge quality, run the `run-evals` skill's `full_pipeline` category (costs API calls; ask first).
