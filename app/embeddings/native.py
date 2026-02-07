"""
Native embeddings implementation using sentence-transformers directly.

Provides high-quality text embeddings with minimal dependencies.
"""

from typing import List, Optional, Union, Dict, Any
import numpy as np
from threading import Lock
import time

from sentence_transformers import SentenceTransformer
from app.logging_config import get_logger
from app.config import settings

logger = get_logger(__name__)


class EmbeddingResult:
    """Result from embedding generation."""

    def __init__(
        self,
        embedding: List[float],
        index: int,
        model: str,
        tokens_used: Optional[int] = None,
    ):
        self.embedding = embedding
        self.index = index
        self.model = model
        self.tokens_used = tokens_used

    def __repr__(self):
        return f"<EmbeddingResult(index={self.index}, model={self.model}, dim={len(self.embedding)})>"


class NativeEmbeddings:
    """
    Native sentence-transformers embeddings.

    Provides efficient text embedding generation with support for
    multiple models and devices (CPU/GPU).
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        normalize_embeddings: bool = True,
        query_prefix: str = "query:",
        document_prefix: str = "passage:",
    ):
        """
        Initialize native embeddings.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
            cache_folder: Cache folder for models
            normalize_embeddings: Whether to normalize embeddings
            query_prefix: Prefix for query embeddings (E5 models)
            document_prefix: Prefix for document embeddings (E5 models)
        """
        self.model_name = model_name
        self.device = device
        self.cache_folder = cache_folder
        self.normalize_embeddings = normalize_embeddings
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

        self._model: Optional[SentenceTransformer] = None
        self._load_lock = Lock()
        self._embedding_dim: Optional[int] = None

        logger.info(
            "NativeEmbeddings initialized",
            model=model_name,
            device=device,
            normalize=normalize_embeddings,
        )

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model (thread-safe)."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    start_time = time.time()
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        cache_folder=self.cache_folder,
                    )
                    self._embedding_dim = self._model.get_sentence_embedding_dimension()
                    load_time = time.time() - start_time
                    logger.info(
                        "Model loaded",
                        model=self.model_name,
                        dim=self._embedding_dim,
                        load_time=f"{load_time:.2f}s",
                    )
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            # Force model load
            _ = self.model
        return self._embedding_dim or 1024  # Default for E5-large

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of document texts

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Add prefix for E5 models
        prefixed_texts = [
            f"{self.document_prefix} {text}" if self.document_prefix else text
            for text in texts
        ]

        start_time = time.time()
        embeddings = self.model.encode(
            prefixed_texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        elapsed = time.time() - start_time

        logger.debug(
            "Documents embedded",
            count=len(texts),
            time=f"{elapsed:.3f}s",
            throughput=f"{len(texts)/elapsed:.1f} docs/s",
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query string.

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        # Add prefix for E5 models
        prefixed_text = f"{self.query_prefix} {text}" if self.query_prefix else text

        start_time = time.time()
        embedding = self.model.encode(
            prefixed_text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        elapsed = time.time() - start_time

        logger.debug(
            "Query embedded",
            text_length=len(text),
            time=f"{elapsed:.3f}s",
        )

        return embedding.tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Async embed a list of documents.

        Note: sentence-transformers is synchronous, so this runs in a thread pool.
        For true async, use the BatchEmbeddingProcessor instead.
        """
        # Run in thread pool to avoid blocking
        import asyncio
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a query string."""
        import asyncio
        return await asyncio.to_thread(self.embed_query, text)

    def embed_texts_with_metadata(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[EmbeddingResult]:
        """
        Embed texts with additional metadata tracking.

        Args:
            texts: List of texts to embed
            metadata: Optional metadata for each text

        Returns:
            List of EmbeddingResult objects
        """
        embeddings = self.embed_documents(texts)

        results = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            result = EmbeddingResult(
                embedding=embedding,
                index=i,
                model=self.model_name,
                tokens_used=len(text.split()),  # Rough estimate
            )
            results.append(result)

        return results


class MultiQueryEmbeddings:
    """
    Generate multiple query embeddings for better retrieval.

    Creates variations of the query to improve recall.
    """

    def __init__(
        self,
        base_embeddings: NativeEmbeddings,
        query_variations: int = 3,
    ):
        """
        Initialize multi-query embeddings.

        Args:
            base_embeddings: Base embeddings instance
            query_variations: Number of query variations to generate
        """
        self.base_embeddings = base_embeddings
        self.query_variations = query_variations

        # Simple query templates for variations
        self.variation_templates = [
            "",  # Original query
            "What is {query}?",
            "Explain {query} in detail.",
            "Discuss the key aspects of {query}.",
            "How does {query} work?",
        ]

    def embed_query_multi(self, query: str) -> List[List[float]]:
        """
        Generate multiple embeddings for a query.

        Args:
            query: Original query text

        Returns:
            List of embedding vectors (original + variations)
        """
        variations = []

        for i in range(min(self.query_variations, len(self.variation_templates))):
            template = self.variation_templates[i]
            if template:
                varied_query = template.format(query=query)
            else:
                varied_query = query

            embedding = self.base_embeddings.embed_query(varied_query)
            variations.append(embedding)

        return variations

    async def aembed_query_multi(self, query: str) -> List[List[float]]:
        """Async version of embed_query_multi."""
        import asyncio
        return await asyncio.to_thread(self.embed_query_multi, query)


class EmbeddingModelFactory:
    """Factory for creating embedding models."""

    _models: Dict[str, NativeEmbeddings] = {}
    _lock = Lock()

    @classmethod
    def get_model(
        cls,
        model_name: str = "intfloat/e5-large-v2",
        **kwargs,
    ) -> NativeEmbeddings:
        """
        Get or create a model instance (cached).

        Args:
            model_name: Model name
            **kwargs: Additional arguments for NativeEmbeddings

        Returns:
            NativeEmbeddings instance
        """
        if model_name not in cls._models:
            with cls._lock:
                if model_name not in cls._models:
                    cls._models[model_name] = NativeEmbeddings(
                        model_name=model_name,
                        **kwargs,
                    )
        return cls._models[model_name]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the model cache."""
        with cls._lock:
            cls._models.clear()
        logger.info("Model cache cleared")


# Pre-configured models for common use cases
class PretrainedModels:
    """Pre-configured embedding models."""

    # E5 models (excellent for retrieval)
    E5_LARGE = "intfloat/e5-large-v2"
    E5_BASE = "intfloat/e5-base-v2"
    E5_SMALL = "intfloat/e5-small-v2"

    # BGE models (better for Chinese/mixed)
    BGE_LARGE = "BAAI/bge-large-en-v1.5"
    BGE_BASE = "BAAI/bge-base-en-v1.5"

    # MPNet (good general purpose)
    MPNET = "sentence-transformers/all-mpnet-base-v2"

    # MiniLM (faster, slightly lower quality)
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings(
    model_name: Optional[str] = None,
    **kwargs,
) -> NativeEmbeddings:
    """
    Convenience function to get embeddings.

    Args:
        model_name: Model name (defaults to E5_LARGE)
        **kwargs: Additional arguments

    Returns:
        NativeEmbeddings instance
    """
    if model_name is None:
        model_name = getattr(settings, "EMBEDDING_MODEL", PretrainedModels.E5_LARGE)

    return EmbeddingModelFactory.get_model(model_name, **kwargs)


# Compatibility layer removed - no longer needed
