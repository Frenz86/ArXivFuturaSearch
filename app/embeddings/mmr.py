"""Maximal Marginal Relevance (MMR) algorithm for diverse document selection."""

from typing import List

import numpy as np

from app.logging_config import get_logger

logger = get_logger(__name__)


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    documents: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.5,
) -> list[dict]:
    """
    Apply Maximal Marginal Relevance (MMR) for diverse document selection.

    MMR balances relevance to the query with diversity among selected documents.
    Formula: MMR = argmax[λ * Sim(D, Q) - (1-λ) * max Sim(D, D_i)]
    where D_i are already selected documents.

    Args:
        query_embedding: Query embedding vector (1D array)
        document_embeddings: Document embedding vectors (2D array, n_docs x dim)
        documents: List of document dictionaries
        top_k: Number of documents to select
        lambda_param: Balance between relevance (λ=1) and diversity (λ=0)

    Returns:
        List of selected documents with MMR scores
    """
    if len(documents) == 0:
        return []

    if len(documents) <= top_k:
        return documents[:top_k]

    # Normalize embeddings for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    docs_norm = document_embeddings / np.linalg.norm(
        document_embeddings, axis=1, keepdims=True
    )

    # Compute relevance scores (similarity to query)
    relevance_scores = np.dot(docs_norm, query_norm)

    # Initialize selected and remaining indices
    selected_indices = []
    remaining_indices = list(range(len(documents)))

    for _ in range(min(top_k, len(documents))):
        mmr_scores = []

        for idx in remaining_indices:
            # Relevance component
            relevance = relevance_scores[idx]

            # Diversity component (max similarity to already selected docs)
            if selected_indices:
                similarities_to_selected = np.dot(
                    docs_norm[idx], docs_norm[selected_indices].T
                )
                max_similarity = np.max(similarities_to_selected)
            else:
                max_similarity = 0

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            mmr_scores.append((idx, mmr_score))

        # Select document with highest MMR score
        best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    # Build result with MMR scores
    result = []
    for idx in selected_indices:
        doc = documents[idx].copy()
        doc["mmr_score"] = float(relevance_scores[idx])
        result.append(doc)

    logger.debug(
        "MMR reranking complete",
        original=len(documents),
        selected=len(result),
        lambda_param=lambda_param,
    )

    return result
