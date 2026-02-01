"""Embeddings package for text embedding and reranking."""

from app.embeddings.embeddings import (
    E5EmbeddingsWrapper,
    Embedder,
    get_embedder,
)
from app.embeddings.reranker import (
    Reranker,
    get_reranker,
)
from app.embeddings.mmr import (
    maximal_marginal_relevance,
)

__all__ = [
    # Embeddings
    "E5EmbeddingsWrapper",
    "Embedder",
    "get_embedder",
    # Reranker
    "Reranker",
    "get_reranker",
    # MMR
    "maximal_marginal_relevance",
]
