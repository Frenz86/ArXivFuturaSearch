"""Vector store factory and base abstraction for multi-backend support.

This module provides a unified interface for different vector store backends:
- ChromaDB: Local embedded vector store (development)
- Pgvector: PostgreSQL with pgvector extension (production)
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class VectorStoreInterface(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    def add(
        self,
        vectors: np.ndarray,
        chunk_ids: list[str],
        texts: list[str],
        metas: list[dict],
    ) -> None:
        """Add documents to the store."""
        pass

    @abstractmethod
    def search(
        self,
        query_vec: np.ndarray,
        query_text: str = "",
        top_k: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Hybrid search combining semantic and lexical search."""
        pass

    @abstractmethod
    def search_ensemble(
        self,
        query_text: str,
        top_k: int = 5,
        query_expansion: bool = True,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Advanced ensemble search with query expansion and RRF."""
        pass

    @abstractmethod
    def get_langchain_retriever(
        self,
        search_type: str = "mmr",
        search_kwargs: Optional[dict] = None,
    ) -> Any:
        """Get a LangChain retriever for use in chains."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total number of documents in the collection."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Delete all documents from the collection."""
        pass


def get_vectorstore(collection_name: str = "arxiv_papers") -> VectorStoreInterface:
    """
    Factory function to get the appropriate vector store based on configuration.

    Args:
        collection_name: Name of the collection/table

    Returns:
        VectorStoreInterface instance (ChromaDB or Pgvector)
    """
    mode = settings.VECTORSTORE_MODE.lower()

    logger.info(
        "Initializing vector store",
        mode=mode,
        collection=collection_name,
    )

    if mode == "pgvector":
        from app.vectorstore_pgvector import PgvectorStore
        logger.info("Using Pgvector (PostgreSQL) backend")
        return PgvectorStore(collection_name=collection_name)
    elif mode == "chroma":
        from app.vectorstore_chroma import ChromaHybridStore
        logger.info("Using ChromaDB backend")
        return ChromaHybridStore(collection_name=collection_name)
    else:
        raise ValueError(
            f"Unknown VECTORSTORE_MODE: {mode}. "
            "Supported modes: 'chroma', 'pgvector'"
        )


# Re-export for backward compatibility
def load_store() -> VectorStoreInterface:
    """Load the vector store (backward compatibility wrapper)."""
    return get_vectorstore(collection_name="arxiv_papers")
