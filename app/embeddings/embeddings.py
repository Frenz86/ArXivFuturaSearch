"""Embedding generation using LangChain with HuggingFace models."""

import threading
from functools import lru_cache
from typing import List

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Thread-safe lock for singleton initialization
_embedder_lock = threading.Lock()


class E5EmbeddingsWrapper(Embeddings):
    """
    Wrapper for LangChain embeddings that adds E5 model prefixes.

    E5 models require:
    - "query: " prefix for questions
    - "passage: " prefix for documents
    """

    def __init__(self, base_embeddings: HuggingFaceEmbeddings):
        """Initialize the wrapper."""
        self.base_embeddings = base_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with "passage: " prefix for E5 models."""
        prefixed = ["passage: " + t for t in texts]
        return self.base_embeddings.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        """Embed query with "query: " prefix for E5 models."""
        prefixed = "query: " + text
        return self.base_embeddings.embed_query(prefixed)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.embed_query("test"))


class Embedder:
    """Thread-safe wrapper for LangChain HuggingFaceEmbeddings."""

    _instances: dict[str, "Embedder"] = {}

    def __new__(cls, model_name: str) -> "Embedder":
        """Thread-safe singleton pattern - reuse model instances."""
        if model_name not in cls._instances:
            with _embedder_lock:
                if model_name not in cls._instances:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instances[model_name] = instance
        return cls._instances[model_name]

    def __init__(self, model_name: str):
        """Initialize the embedder using LangChain (only once per model)."""
        if self._initialized:
            return

        logger.info("Loading embedding model via LangChain", model=model_name)
        self.model_name = model_name

        # Check if this is an E5 model
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
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([], dtype="float32").reshape(0, self._dimension)

        # Add E5 prefixes if needed
        if self._is_e5_model:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]

        vectors = self.langchain_embeddings.embed_documents(texts)
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query (optimized for single text)."""
        if self._is_e5_model:
            query = "query: " + query

        vector = self.langchain_embeddings.embed_query(query)
        return np.asarray(vector, dtype="float32")

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings | Embeddings:
        """Get the LangChain embeddings object (wrapped for E5 models)."""
        if self._is_e5_model:
            return E5EmbeddingsWrapper(self.langchain_embeddings)
        return self.langchain_embeddings


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Get the default embedder singleton."""
    return Embedder(settings.EMBED_MODEL)
