"""Cross-encoder based re-ranking for improved retrieval precision.


# Copyright 2025 ArXivFuturaSearch Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Uses cross-encoder models to re-rank search results based on
query-document relevance scores.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from functools import lru_cache

import numpy as np

from app.embeddings import Reranker
from app.logging_config import get_logger
from app.config import settings

logger = get_logger(__name__)


# =============================================================================
# CROSS-ENCODER RERANKER
# =============================================================================

class CrossEncoderReranker:
    """
    Re-rank search results using cross-encoder models.

    Cross-encoders provide more accurate relevance scores than
    bi-encoders by processing query-document pairs together.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        top_k: Optional[int] = None,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
            batch_size: Batch size for scoring
            top_k: Number of top results to return (None = all)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.top_k = top_k
        self._reranker: Optional[Reranker] = None

    def _get_reranker(self) -> Reranker:
        """Lazy-load the reranker model."""
        if self._reranker is None:
            logger.info("Loading cross-encoder reranker", model=self.model_name)
            self._reranker = Reranker(self.model_name)
        return self._reranker

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using cross-encoder scores.

        Args:
            query: Search query
            documents: List of document dicts with 'text' or 'content' field
            top_k: Number of top results to return

        Returns:
            Re-ranked list of documents with relevance scores
        """
        if not documents:
            return []

        top_k = top_k or self.top_k or len(documents)

        # Extract document texts
        doc_texts = []
        for doc in documents:
            text = doc.get("text") or doc.get("content") or doc.get("page_content", "")
            doc_texts.append(text)

        try:
            # Get cross-encoder scores
            reranker = self._get_reranker()
            scores = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: reranker.rerank(query, doc_texts),
            )

            # Add scores to documents
            scored_docs = []
            for doc, score in zip(documents, scores):
                scored_doc = doc.copy()
                scored_doc["ce_score"] = float(score)
                scored_docs.append(scored_doc)

            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x["ce_score"], reverse=True)

            logger.info(
                "Cross-encoder reranking complete",
                query=query[:50],
                documents=len(documents),
                top_k=top_k,
                avg_score=np.mean(scores),
                max_score=np.max(scores),
            )

            return scored_docs[:top_k]

        except Exception as e:
            logger.error("Cross-encoder reranking failed", error=str(e))
            # Return original documents on error
            return documents[:top_k]

    async def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Re-rank multiple query-document pairs.

        Args:
            queries: List of search queries
            documents_list: List of document lists for each query
            top_k: Number of top results per query

        Returns:
            List of re-ranked document lists
        """
        tasks = [
            self.rerank(query, docs, top_k)
            for query, docs in zip(queries, documents_list)
        ]

        return await asyncio.gather(*tasks)


# =============================================================================
# RERANKING STRATEGIES
# =============================================================================

class RerankingStrategy:
    """Base class for reranking strategies."""

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Re-rank documents."""
        raise NotImplementedError


class ScoreNormalizationStrategy(RerankingStrategy):
    """
    Normalize scores from different sources and combine them.

    Useful when you have both vector similarity scores and BM25 scores.
    """

    def __init__(
        self,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        """
        Initialize normalization strategy.

        Args:
            vector_weight: Weight for vector similarity scores
            keyword_weight: Weight for keyword/BM25 scores
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Combine and normalize vector and keyword scores.

        Args:
            query: Search query (unused but kept for interface consistency)
            documents: Documents with 'vector_score' and 'keyword_score' fields

        Returns:
            Re-ranked documents with combined scores
        """
        if not documents:
            return []

        # Extract scores
        vector_scores = []
        keyword_scores = []

        for doc in documents:
            vector_scores.append(doc.get("vector_score", 0.0))
            keyword_scores.append(doc.get("keyword_score", 0.0))

        # Normalize scores to [0, 1]
        if vector_scores:
            v_max, v_min = max(vector_scores), min(vector_scores)
            v_range = v_max - v_min if v_max != v_min else 1.0
            vector_scores = [(s - v_min) / v_range for s in vector_scores]

        if keyword_scores:
            k_max, k_min = max(keyword_scores), min(keyword_scores)
            k_range = k_max - k_min if k_max != k_min else 1.0
            keyword_scores = [(s - k_min) / k_range for s in keyword_scores]

        # Combine scores
        for doc, v_score, k_score in zip(documents, vector_scores, keyword_scores):
            combined = (
                self.vector_weight * v_score +
                self.keyword_weight * k_score
            )
            doc["combined_score"] = combined

        # Sort by combined score
        documents.sort(key=lambda x: x["combined_score"], reverse=True)

        return documents


class MMRDiversificationStrategy(RerankingStrategy):
    """
    Maximal Marginal Relevance (MMR) for diverse result sets.

    Balances relevance with diversity to avoid redundant results.
    """

    def __init__(
        self,
        lambda_param: float = 0.5,
        embedder: Optional[Any] = None,
    ):
        """
        Initialize MMR strategy.

        Args:
            lambda_param: Balance between relevance (1) and diversity (0)
            embedder: Embedder instance for computing similarity
        """
        self.lambda_param = lambda_param
        self.embedder = embedder

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Re-rank using MMR for diversity.

        Args:
            query: Search query
            documents: Documents with relevance scores

        Returns:
            Diversified result set
        """
        if not documents:
            return []

        # Get query embedding if embedder available
        query_emb = None
        if self.embedder:
            try:
                query_emb = self.embedder.embed([query], is_query=True)[0]
            except Exception as e:
                logger.warning("Query embedding failed for MMR", error=str(e))

        # Extract document embeddings
        doc_embeddings = []
        relevance_scores = []

        for doc in documents:
            # Get relevance score
            score = doc.get("score", doc.get("vector_score", 0.0))
            relevance_scores.append(score)

            # Get document embedding (if available)
            if "embedding" in doc:
                doc_embeddings.append(doc["embedding"])
            elif query_emb is not None and self.embedder:
                text = doc.get("text", doc.get("content", ""))
                try:
                    emb = self.embedder.embed([text], is_query=False)[0]
                    doc_embeddings.append(emb)
                except Exception:
                    doc_embeddings.append(np.zeros_like(query_emb))
            else:
                doc_embeddings.append(None)

        # If no embeddings, return original order
        if not any(doc_embeddings):
            return documents

        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(documents)))

        while remaining_indices and len(selected_indices) < len(documents):
            best_score = -float('inf')
            best_idx = None

            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]

                # Diversity component (max similarity to selected docs)
                if selected_indices and doc_embeddings[idx] is not None:
                    max_sim = 0.0
                    for sel_idx in selected_indices:
                        if doc_embeddings[sel_idx] is not None:
                            # Cosine similarity
                            sim = np.dot(
                                doc_embeddings[idx],
                                doc_embeddings[sel_idx]
                            ) / (
                                np.linalg.norm(doc_embeddings[idx]) *
                                np.linalg.norm(doc_embeddings[sel_idx]) + 1e-8
                            )
                            max_sim = max(max_sim, sim)
                    diversity = max_sim
                else:
                    diversity = 0.0

                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break

        # Return re-ordered documents
        reranked = [documents[i].copy() for i in selected_indices]
        for i, doc in enumerate(reranked):
            doc["mmr_rank"] = i
            doc["mmr_score"] = documents[selected_indices[i]].get("score", 0.0)

        return reranked


# =============================================================================
# RERANKING PIPELINE
# =============================================================================

class RerankingPipeline:
    """
    Pipeline for chaining multiple reranking strategies.
    """

    def __init__(
        self,
        strategies: List[RerankingStrategy],
        cross_encoder: Optional[CrossEncoderReranker] = None,
    ):
        """
        Initialize reranking pipeline.

        Args:
            strategies: List of reranking strategies to apply in order
            cross_encoder: Optional cross-encoder for final reranking
        """
        self.strategies = strategies
        self.cross_encoder = cross_encoder

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Apply all reranking strategies in sequence.

        Args:
            query: Search query
            documents: Initial document list
            top_k: Final number of results

        Returns:
            Re-ranked and filtered documents
        """
        if not documents:
            return []

        current_docs = documents.copy()

        # Apply each strategy
        for strategy in self.strategies:
            current_docs = await strategy.rerank(query, current_docs)
            logger.debug(
                "Strategy applied",
                strategy=strategy.__class__.__name__,
                results=len(current_docs),
            )

        # Apply cross-encoder if available
        if self.cross_encoder:
            current_docs = await self.cross_encoder.rerank(
                query,
                current_docs,
                top_k=top_k,
            )
        else:
            current_docs = current_docs[:top_k]

        return current_docs


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

_cross_encoder_reranker: Optional[CrossEncoderReranker] = None


def get_cross_encoder_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> CrossEncoderReranker:
    """
    Get or create global cross-encoder reranker.

    Args:
        model_name: Model name for cross-encoder

    Returns:
        CrossEncoderReranker instance
    """
    global _cross_encoder_reranker
    if _cross_encoder_reranker is None or _cross_encoder_reranker.model_name != model_name:
        _cross_encoder_reranker = CrossEncoderReranker(model_name=model_name)
    return _cross_encoder_reranker


def create_reranking_pipeline(
    use_cross_encoder: bool = True,
    use_mmr: bool = False,
    lambda_param: float = 0.5,
    embedder: Optional[Any] = None,
) -> RerankingPipeline:
    """
    Create a reranking pipeline with common configurations.

    Args:
        use_cross_encoder: Include cross-encoder reranking
        use_mmr: Use MMR for diversification
        lambda_param: MMR lambda parameter
        embedder: Embedder for MMR similarity computation

    Returns:
        Configured RerankingPipeline
    """
    strategies = []

    if use_mmr:
        strategies.append(MMRDiversificationStrategy(
            lambda_param=lambda_param,
            embedder=embedder,
        ))

    cross_encoder = get_cross_encoder_reranker() if use_cross_encoder else None

    return RerankingPipeline(
        strategies=strategies,
        cross_encoder=cross_encoder,
    )
