"""Hybrid search combining semantic vector search with BM25 keyword search.


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

Provides improved retrieval by combining dense and sparse representations.
"""

import math
import re
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Any
from functools import lru_cache

import numpy as np

from app.logging_config import get_logger
from app.config import settings

logger = get_logger(__name__)


# =============================================================================
# BM25 IMPLEMENTATION
# =============================================================================

class BM25Index:
    """
    BM25 (Best Matching 25) sparse retrieval index.

    BM25 is a ranking function used by search engines to estimate
    the relevance of documents to a given search query.
    """

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25 index.

        Args:
            k1: Controls term frequency saturation (1.0-2.0 typical)
            b: Controls document length normalization (0.0-1.0)
            epsilon: IDF floor for rare terms
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Index data
        self.corpus_size: int = 0
        self.avg_doc_length: float = 0.0
        self.doc_lengths: List[int] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_texts: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Extract alphanumeric tokens
        tokens = re.findall(r'\b\w+\b', text)

        return tokens

    def index_documents(self, documents: List[str]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of document texts
        """
        logger.info("Building BM25 index", documents=len(documents))

        self.doc_texts = documents
        self.corpus_size = len(documents)
        self.doc_lengths = []
        self.doc_freqs = []

        # Calculate document frequencies and lengths
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            self.doc_freqs.append(Counter(tokens))

        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0

        # Calculate IDF scores
        self._build_idf()

        logger.info(
            "BM25 index built",
            corpus_size=self.corpus_size,
            avg_doc_length=f"{self.avg_doc_length:.1f}",
            vocab_size=len(self.idf),
        )

    def _build_idf(self) -> None:
        """Calculate IDF (Inverse Document Frequency) scores."""
        # Count document frequency for each term
        df = defaultdict(int)
        for freq_dict in self.doc_freqs:
            for term in freq_dict:
                df[term] += 1

        # Calculate IDF with floor
        idf_sum = 0
        negative_idfs = []

        for term, freq in df.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[term] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(term)

        # Apply epsilon floor to negative IDFs
        self.epsilon_idf = idf_sum / len(self.idf) if self.idf else 0
        for term in negative_idfs:
            self.idf[term] = max(self.idf[term], self.epsilon * self.epsilon_idf)

    def _get_score(self, query: str, doc_idx: int) -> float:
        """
        Calculate BM25 score for a query-document pair.

        Args:
            query: Query text
            doc_idx: Document index

        Returns:
            BM25 relevance score
        """
        query_tokens = self._tokenize(query)
        doc_freqs = self.doc_freqs[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        score = 0.0
        for token in query_tokens:
            if token not in doc_freqs:
                continue

            # Get IDF
            idf = self.idf.get(token, 0)

            # Calculate term frequency component
            tf = doc_freqs[token]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search using BM25 ranking.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_idx, score) tuples
        """
        if self.corpus_size == 0:
            return []

        # Calculate scores for all documents
        scores = [
            (idx, self._get_score(query, idx))
            for idx in range(self.corpus_size)
        ]

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = []
        for idx, score in scores[:top_k]:
            if score > 0:  # Only return relevant results
                results.append({
                    "doc_idx": idx,
                    "score": score,
                    "text": self.doc_texts[idx][:500],  # Preview
                })

        logger.debug(
            "BM25 search complete",
            query=query[:50],
            results=len(results),
            top_score=results[0]["score"] if results else 0,
        )

        return results


# =============================================================================
# HYBRID SEARCH (BM25 + VECTOR)
# =============================================================================

class HybridSearchEngine:
    """
    Hybrid search combining BM25 and vector search with score fusion.

    Uses reciprocal rank fusion (RRF) to combine results from both methods.
    """

    def __init__(
        self,
        vector_store: Any,
        alpha: float = 0.5,
        k: int = 60,
    ):
        """
        Initialize hybrid search engine.

        Args:
            vector_store: Vector store for semantic search
            alpha: Weight for vector search (0-1, 0.5 = equal weight)
            k: RRF constant (higher = more emphasis on top ranks)
        """
        self.vector_store = vector_store
        self.alpha = alpha
        self.k = k
        self.bm25_index = BM25Index()
        self._indexed = False

    async def index_documents(self, texts: List[str]) -> None:
        """
        Build BM25 index from document texts.

        Args:
            texts: List of document texts
        """
        logger.info("Building hybrid search index", documents=len(texts))
        self.bm25_index.index_documents(texts)
        self._indexed = True

    async def search(
        self,
        query: str,
        top_k: int = 10,
        search_kwargs: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and vector search.

        Args:
            query: Search query
            top_k: Number of results to return
            search_kwargs: Additional kwargs for vector search

        Returns:
            List of search results with combined scores
        """
        if not self._indexed:
            logger.warning("BM25 index not built, using vector search only")
            return await self._vector_search_only(query, top_k, search_kwargs)

        # Run both searches in parallel (conceptually)
        # For simplicity, we run sequentially here

        # 1. Vector search results
        vector_results = await self._vector_search(query, top_k * 2, search_kwargs)

        # 2. BM25 search results
        bm25_results = self.bm25_index.search(query, top_k * 2)

        # 3. Combine using Reciprocal Rank Fusion (RRF)
        combined_results = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            top_k,
        )

        logger.info(
            "Hybrid search complete",
            query=query[:50],
            vector_results=len(vector_results),
            bm25_results=len(bm25_results),
            combined=len(combined_results),
        )

        return combined_results

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        search_kwargs: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            query: Search query
            top_k: Number of results
            search_kwargs: Additional kwargs

        Returns:
            List of results with doc_idx and score
        """
        kwargs = {"k": top_k}
        if search_kwargs:
            kwargs.update(search_kwargs)

        try:
            # Use vector store similarity search
            results = await self.vector_store.asimilarity_search(query, **kwargs)

            # Convert to standard format
            formatted = []
            for idx, doc in enumerate(results):
                formatted.append({
                    "doc_idx": idx,  # Placeholder - actual mapping depends on store
                    "score": 1.0 - (idx * 0.01),  # Placeholder score
                    "document": doc,
                })

            return formatted

        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            return []

    async def _vector_search_only(
        self,
        query: str,
        top_k: int,
        search_kwargs: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Fallback to vector search only."""
        return await self._vector_search(query, top_k, search_kwargs)

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF formula: score = 1 / (k + rank)

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            top_k: Number of results to return

        Returns:
            Combined and ranked results
        """
        # Score maps for RRF
        rrf_scores = defaultdict(float)
        result_map = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result.get("doc_idx", rank)
            rrf_scores[doc_id] += self.alpha / (self.k + rank)
            result_map[doc_id] = result

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result.get("doc_idx", rank)
            rrf_scores[doc_id] += (1 - self.alpha) / (self.k + rank)
            if doc_id not in result_map:
                result_map[doc_id] = result

        # Sort by combined RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        # Format final results
        combined = []
        for doc_id, rrf_score in sorted_results:
            result = result_map[doc_id].copy()
            result["hybrid_score"] = rrf_score
            combined.append(result)

        return combined


# =============================================================================
# SCORE NORMALIZATION HELPERS
# =============================================================================

def min_max_normalize(scores: List[float]) -> List[float]:
    """
    Normalize scores to [0, 1] range using min-max normalization.

    Args:
        scores: List of scores

    Returns:
        Normalized scores
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


def z_score_normalize(scores: List[float]) -> List[float]:
    """
    Normalize scores using z-score standardization.

    Args:
        scores: List of scores

    Returns:
        Normalized scores
    """
    if len(scores) < 2:
        return [1.0] if scores else []

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(variance)

    if std == 0:
        return [0.0] * len(scores)

    return [(s - mean) / std for s in scores]


# =============================================================================
# GLOBAL HYBRID SEARCH ENGINE
# =============================================================================

_hybrid_engine: Optional[HybridSearchEngine] = None


def get_hybrid_search_engine(vector_store: Any) -> HybridSearchEngine:
    """
    Get or create global hybrid search engine.

    Args:
        vector_store: Vector store instance

    Returns:
        HybridSearchEngine instance
    """
    global _hybrid_engine
    if _hybrid_engine is None:
        _hybrid_engine = HybridSearchEngine(
            vector_store=vector_store,
            alpha=0.5,
            k=60,
        )
    return _hybrid_engine
