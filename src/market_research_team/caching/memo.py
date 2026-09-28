"""In-process memoization of the embedding and cross-encoder model objects.

Distinct from the SQLite result caches in `retrieval_cache.py`/`rerank_cache.py`:
this just avoids reconstructing expensive model objects on every pipeline run
within a process. `_loader` is injectable so tests can substitute a fake
without patching import sites or constructing a real model.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Any


def _default_embeddings_loader(model_name: str) -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)


def _default_cross_encoder_loader(model_name: str) -> Any:
    from market_research_team.retrieval.reranker import load_cross_encoder

    return load_cross_encoder(model_name)


@lru_cache(maxsize=4)
def cached_embeddings(model_name: str, *, _loader: Callable[[str], Any] | None = None) -> Any:
    loader = _loader or _default_embeddings_loader
    return loader(model_name)


@lru_cache(maxsize=4)
def cached_cross_encoder(model_name: str, *, _loader: Callable[[str], Any] | None = None) -> Any:
    loader = _loader or _default_cross_encoder_loader
    return loader(model_name)
