"""
Native BM25 implementation for lexical search.

This module provides a fast, memory-efficient BM25 implementation
using rank-bm25.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
from math import log
import pickle
import re

import numpy as np

from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# TOKENIZATION
# =============================================================================

class Tokenizer:
    """Text tokenizer for BM25."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        min_length: int = 2,
        stopwords: Optional[set] = None,
    ):
        """
        Initialize tokenizer.

        Args:
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            min_length: Minimum token length
            stopwords: Set of stopwords to filter
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_length = min_length
        self.stopwords = stopwords or set()

        # Compile regex for punctuation
        self.punct_pattern = re.compile(r'[^\w\s]', re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = self.punct_pattern.sub(' ', text)

        # Split and filter
        tokens = text.split()

        # Filter by length and stopwords
        tokens = [
            t for t in tokens
            if len(t) >= self.min_length and t not in self.stopwords
        ]

        return tokens


# =============================================================================
# BM25 MODEL
# =============================================================================

class BM25:
    """
    Native BM25 implementation.

    Supports multiple BM25 variants:
    - BM25Okapi: Standard BM25
    - BM25L: BM25 with length normalization
    - BM25Plus: BM25++ variant
    """

    def __init__(
        self,
        corpus: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        variant: str = "okapi",
        tokenizer: Optional[Tokenizer] = None,
    ):
        """
        Initialize BM25.

        Args:
            corpus: List of tokenized documents
            k1: Term saturation parameter
            b: Length normalization parameter
            epsilon: Floor value for idf (for BM25Plus)
            variant: BM25 variant ("okapi", "l", "plus")
            tokenizer: Optional tokenizer instance
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.variant = variant
        self.tokenizer = tokenizer

        # Compute corpus statistics
        self.n_docs = len(corpus)
        self.avg_doc_length = sum(len(doc) for doc in corpus) / max(self.n_docs, 1)

        # Document frequencies
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

        # Build index
        self._build_index()

        logger.info(
            "BM25 initialized",
            variant=variant,
            n_docs=self.n_docs,
            avg_doc_length=f"{self.avg_doc_length:.1f}",
            k1=k1,
            b=b,
        )

    def _build_index(self) -> None:
        """Build the inverted index."""
        # Count term frequencies per document
        for doc in self.corpus:
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)

        # Compute IDF for each term
        self.idf = self._compute_idf()

    def _compute_idf(self) -> Dict[str, float]:
        """Compute IDF scores."""
        # Count document frequency for each term
        df = Counter()
        for freqs in self.doc_freqs:
            for term in freqs.keys():
                df[term] += 1

        # Compute IDF based on variant
        idf = {}
        for term, freq in df.items():
            if self.variant == "okapi":
                # Standard BM25 IDF
                idf[term] = log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)
            elif self.variant == "l":
                # BM25L IDF
                idf[term] = log((self.n_docs + 1) / (freq + 0.5))
            elif self.variant == "plus":
                # BM25++ IDF
                idf[term] = log((self.n_docs + 1) / freq) + self.epsilon

        return idf

    def get_scores(self, query: List[str]) -> np.ndarray:
        """
        Compute BM25 scores for a query.

        Args:
            query: Tokenized query

        Returns:
            Array of scores for each document
        """
        scores = np.zeros(self.n_docs)

        for term in query:
            if term not in self.idf:
                continue

            idf = self.idf[term]

            for i, doc_freqs in enumerate(self.doc_freqs):
                if term not in doc_freqs:
                    continue

                # Term frequency in document
                tf = doc_freqs[term]

                # Document length
                doc_length = len(self.corpus[i])

                # Length normalization
                length_norm = 1 - self.b + self.b * (doc_length / self.avg_doc_length)

                # BM25 score component
                if self.variant == "plus":
                    # BM25++ formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + length_norm * self.k1
                    score = idf * (numerator / denominator)
                else:
                    # Standard BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * length_norm
                    score = idf * (numerator / denominator)

                scores[i] += score

        return scores

    def get_top_n(
        self,
        query: List[str],
        n: int = 10,
        indices: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Get top N documents for a query.

        Args:
            query: Tokenized query
            n: Number of results
            indices: Optional list of document indices to search

        Returns:
            List of (index, score) tuples
        """
        scores = self.get_scores(query)

        # Filter to specific indices if provided
        if indices is not None:
            mask = np.zeros(self.n_docs, dtype=bool)
            mask[indices] = True
            scores = scores * mask

        # Get top N
        top_indices = np.argsort(scores)[::-1][:n]
        results = [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

        return results


# =============================================================================
# DOCUMENT RETRIEVER
# =============================================================================

class BM25Retriever:
    """
    BM25-based document retriever.

    Provides an interface similar to vector store retrievers
    for easy integration into RAG pipelines.
    """

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        variant: str = "okapi",
        tokenizer: Optional[Tokenizer] = None,
        text_field: str = "content",
    ):
        """
        Initialize retriever.

        Args:
            documents: List of document dictionaries
            k1: BM25 k1 parameter
            b: BM25 b parameter
            variant: BM25 variant
            tokenizer: Optional tokenizer
            text_field: Field name for document text
        """
        self.documents = documents
        self.text_field = text_field
        self.tokenizer = tokenizer or Tokenizer()

        # Tokenize corpus
        tokenized_corpus = []
        for doc in documents:
            text = doc.get(text_field, "")
            tokens = self.tokenizer.tokenize(text)
            tokenized_corpus.append(tokens)

        # Initialize BM25
        self.bm25 = BM25(
            corpus=tokenized_corpus,
            k1=k1,
            b=b,
            variant=variant,
        )

        logger.info(
            "BM25Retriever initialized",
            n_docs=len(documents),
            text_field=text_field,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query.

        Args:
            query: Query text
            top_k: Number of results
            score_threshold: Minimum score threshold

        Returns:
            List of documents with scores
        """
        # Tokenize query
        query_tokens = self.tokenizer.tokenize(query)

        # Get scores
        results = self.bm25.get_top_n(query_tokens, n=top_k)

        # Build output
        docs = []
        for idx, score in results:
            if score_threshold and score < score_threshold:
                continue

            doc = self.documents[idx].copy()
            doc["score"] = score
            doc["retrieval_type"] = "bm25"
            docs.append(doc)

        return docs

    async def aretrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Async version of retrieve."""
        import asyncio
        return await asyncio.to_thread(
            self.retrieve,
            query,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the index.

        Note: This rebuilds the entire index.

        Args:
            documents: New documents to add
        """
        self.documents.extend(documents)

        # Rebuild index
        tokenized_corpus = []
        for doc in self.documents:
            text = doc.get(self.text_field, "")
            tokens = self.tokenizer.tokenize(text)
            tokenized_corpus.append(tokens)

        self.bm25 = BM25(
            corpus=tokenized_corpus,
            k1=self.bm25.k1,
            b=self.bm25.b,
            variant=self.bm25.variant,
        )

        logger.info("Documents added", count=len(documents), total=len(self.documents))

    def save(self, path: str) -> None:
        """
        Save retriever to disk.

        Args:
            path: File path
        """
        data = {
            "documents": self.documents,
            "text_field": self.text_field,
            "k1": self.bm25.k1,
            "b": self.bm25.b,
            "variant": self.bm25.variant,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info("BM25Retriever saved", path=path)

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        """
        Load retriever from disk.

        Args:
            path: File path

        Returns:
            BM25Retriever instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        return cls(
            documents=data["documents"],
            k1=data["k1"],
            b=data["b"],
            variant=data["variant"],
            text_field=data["text_field"],
        )


# =============================================================================
# HYBRID RETRIEVER
# =============================================================================

class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search.

    Reciprocal Rank Fusion (RRF) is used to combine results.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: Any,  # Vector store retriever
        rrf_k: int = 60,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_retriever: BM25 retriever
            vector_retriever: Vector store retriever
            rrf_k: RRF constant
            bm25_weight: Weight for BM25 results
            vector_weight: Weight for vector results
        """
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        logger.info(
            "HybridRetriever initialized",
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )

    def _reciprocal_rank_fusion(
        self,
        results_list: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion.

        Args:
            results_list: List of result lists from different retrievers

        Returns:
            Fused and ranked results
        """
        # Score accumulation
        scores: Dict[str, float] = {}
        documents: Dict[str, Dict[str, Any]] = {}

        for results, weight in zip(
            results_list,
            [self.bm25_weight, self.vector_weight],
        ):
            for rank, doc in enumerate(results, 1):
                # Use document ID or title as key
                key = doc.get("id") or doc.get("title") or str(rank)

                if key not in documents:
                    documents[key] = doc

                # RRF score
                scores[key] = scores.get(key, 0) + weight / (self.rrf_k + rank)

        # Sort by fused score
        ranked = sorted(
            documents.items(),
            key=lambda x: scores[x[0]],
            reverse=True,
        )

        # Add scores
        results = []
        for key, doc in ranked:
            doc_copy = doc.copy()
            doc_copy["rrf_score"] = scores[key]
            doc_copy["retrieval_type"] = "hybrid"
            results.append(doc_copy)

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using both BM25 and vector search.

        Args:
            query: Query text
            top_k: Number of results
            score_threshold: Minimum score threshold

        Returns:
            Fused results
        """
        # Retrieve from both
        bm25_results = self.bm25.retrieve(query, top_k=top_k * 2)
        vector_results = self.vector.similarity_search(query, k=top_k * 2)

        # Combine with RRF
        fused = self._reciprocal_rank_fusion([bm25_results, vector_results])

        # Apply threshold and limit
        if score_threshold:
            fused = [r for r in fused if r.get("rrf_score", 0) >= score_threshold]

        return fused[:top_k]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_bm25_retriever(
    documents: List[Dict[str, Any]],
    text_field: str = "content",
    **kwargs,
) -> BM25Retriever:
    """
    Create a BM25 retriever.

    Args:
        documents: List of documents
        text_field: Field containing text
        **kwargs: Additional BM25 parameters

    Returns:
        BM25Retriever instance
    """
    return BM25Retriever(
        documents=documents,
        text_field=text_field,
        **kwargs,
    )
