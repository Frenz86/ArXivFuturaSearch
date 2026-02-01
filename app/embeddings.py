"""Compatibility shim for embeddings imports.

This file maintains backward compatibility by re-exporting from the new embeddings package.
Deprecated: Import from app.embeddings.{embeddings,reranker,mmr} instead.
"""

# Re-export everything from the embeddings package
from app.embeddings import (
    E5EmbeddingsWrapper,
    Embedder,
    get_embedder,
    Reranker,
    get_reranker,
    maximal_marginal_relevance,
)

__all__ = [
    "E5EmbeddingsWrapper",
    "Embedder",
    "get_embedder",
    "Reranker",
    "get_reranker",
    "maximal_marginal_relevance",
]
