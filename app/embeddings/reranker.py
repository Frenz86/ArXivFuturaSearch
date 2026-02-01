"""Cross-encoder reranker for document re-ranking."""

import threading
from functools import lru_cache
from typing import Optional

from sentence_transformers import CrossEncoder

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Thread-safe lock for singleton initialization
_reranker_lock = threading.Lock()


class Reranker:
    """Thread-safe cross-encoder reranker with singleton support."""

    _instances: dict[str, "Reranker"] = {}

    def __new__(cls, model_name: str) -> "Reranker":
        """Thread-safe singleton pattern - reuse model instances."""
        if model_name not in cls._instances:
            with _reranker_lock:
                if model_name not in cls._instances:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instances[model_name] = instance
        return cls._instances[model_name]

    def __init__(self, model_name: str):
        """Initialize the reranker (only once per model)."""
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
def get_reranker() -> Optional[Reranker]:
    """Get the default reranker singleton (if enabled)."""
    if not settings.RERANK_ENABLED:
        return None
    return Reranker(settings.RERANK_MODEL)
