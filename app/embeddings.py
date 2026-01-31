"""Embedding generation using LangChain with HuggingFace models."""

from functools import lru_cache
from typing import Optional, List

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from sentence_transformers import CrossEncoder

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class E5EmbeddingsWrapper(Embeddings):
    """
    Wrapper for LangChain embeddings that adds E5 model prefixes.

    E5 models require:
    - "query: " prefix for questions
    - "passage: " prefix for documents
    """

    def __init__(self, base_embeddings: HuggingFaceEmbeddings):
        """
        Initialize the wrapper.

        Args:
            base_embeddings: The underlying LangChain embeddings
        """
        self.base_embeddings = base_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents with "passage: " prefix for E5 models.

        Args:
            texts: List of document texts

        Returns:
            List of embedding vectors
        """
        # Add passage prefix for E5 models
        prefixed = ["passage: " + t for t in texts]
        return self.base_embeddings.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed query with "query: " prefix for E5 models.

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        # Add query prefix for E5 models
        prefixed = "query: " + text
        return self.base_embeddings.embed_query(prefixed)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        # Sample embedding to get dimension
        return len(self.embed_query("test"))


class Embedder:
    """Wrapper for LangChain HuggingFaceEmbeddings with backward compatibility."""

    _instances: dict[str, "Embedder"] = {}

    def __new__(cls, model_name: str) -> "Embedder":
        """Singleton pattern - reuse model instances."""
        if model_name not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[model_name] = instance
        return cls._instances[model_name]

    def __init__(self, model_name: str):
        """
        Initialize the embedder using LangChain (only once per model).

        Args:
            model_name: HuggingFace model name
        """
        if self._initialized:
            return

        logger.info("Loading embedding model via LangChain", model=model_name)
        self.model_name = model_name

        # Check if this is an E5 model (requires query/passage prefixes)
        self._is_e5_model = "e5" in model_name.lower()

        # Initialize LangChain HuggingFaceEmbeddings
        self.langchain_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Get dimension from first embedding
        test_query = "query: test" if self._is_e5_model else "test"
        self._dimension = len(self.langchain_embeddings.embed_query(test_query))

        self._initialized = True
        logger.info("Embedding model loaded via LangChain", model=model_name, dim=self._dimension, is_e5=self._is_e5_model)

    @property
    def is_e5_model(self) -> bool:
        """Check if this is an E5 model that requires prefixes."""
        return self._is_e5_model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def embed(self, texts: list[str], batch_size: int = 32, show_progress: bool = True, is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed
            batch_size: Batch size for encoding (not used with LangChain but kept for compatibility)
            show_progress: Show progress bar (not used with LangChain but kept for compatibility)
            is_query: If True, add "query: " prefix for E5 models; if False, add "passage: " prefix

        Returns:
            NumPy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([], dtype="float32").reshape(0, self._dimension)

        # Add E5 prefixes if needed
        if self._is_e5_model:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]

        # Use LangChain's embed_documents method
        vectors = self.langchain_embeddings.embed_documents(texts)
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query (optimized for single text).

        Args:
            query: Query string

        Returns:
            NumPy array of shape (embedding_dim,)
        """
        # Add E5 prefix if needed
        if self._is_e5_model:
            query = "query: " + query

        # Use LangChain's embed_query method
        vector = self.langchain_embeddings.embed_query(query)
        return np.asarray(vector, dtype="float32")

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings | Embeddings:
        """
        Get the LangChain embeddings object (wrapped for E5 models).

        Returns:
            LangChain embeddings instance (possibly wrapped for E5)
        """
        if self._is_e5_model:
            # Return wrapper that adds E5 prefixes
            return E5EmbeddingsWrapper(self.langchain_embeddings)
        return self.langchain_embeddings


class Reranker:
    """Cross-encoder reranker with singleton support."""

    _instances: dict[str, "Reranker"] = {}

    def __new__(cls, model_name: str) -> "Reranker":
        """Singleton pattern - reuse model instances."""
        if model_name not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[model_name] = instance
        return cls._instances[model_name]

    def __init__(self, model_name: str):
        """
        Initialize the reranker (only once per model).

        Args:
            model_name: HuggingFace cross-encoder model name
        """
        if self._initialized:
            return

        logger.info("Loading reranker model", model=model_name)
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        self._initialized = True
        logger.info("Reranker model loaded", model=model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
        text_key: str = "text",
    ) -> list[dict]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Query string
            documents: List of document dicts (must contain text_key)
            top_k: Number of top documents to return
            text_key: Key for text content in document dicts

        Returns:
            Reranked list of documents with 'rerank_score' added
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [(query, doc[text_key]) for doc in documents]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Add scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sort by rerank score descending
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Get the default embedder singleton."""
    return Embedder(settings.EMBED_MODEL)


@lru_cache(maxsize=1)
def get_reranker() -> Optional[Reranker]:
    """Get the default reranker singleton (if enabled)."""
    if not settings.RERANK_ENABLED:
        return None
    return Reranker(settings.RERANK_MODEL)


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
        # Not enough documents to apply MMR
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
                # Compute similarity to all selected documents
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
