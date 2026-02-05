"""
Semantic-aware caching with similarity-based retrieval and intelligent invalidation.

Provides cache functionality that can return results for semantically similar queries,
not just exact matches. Includes cache warming strategies and pre-computed embeddings.
"""

import hashlib
import json
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum

import numpy as np

from app.cache import CacheClient
from app.embeddings.embeddings import get_embedder
from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class CacheStrategy(Enum):
    """Cache retrieval strategies."""
    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    query: str
    query_embedding: np.ndarray
    results: List[Dict[str, Any]]
    timestamp: float
    hit_count: int = 0
    ttl: int = 3600

    @property
    def age(self) -> float:
        """Age of the cache entry in seconds."""
        return time.time() - self.timestamp

    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        return self.age > self.ttl


class SemanticCache:
    """
    Semantic-aware cache for query results.

    Features:
    - Exact query matching
    - Semantic similarity-based retrieval
    - Pre-computed query embeddings
    - LRU eviction policy
    - Cache warming support
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_entries: int = 1000,
        strategy: CacheStrategy = CacheStrategy.HYBRID,
        default_ttl: int = 3600,
        enable_precomputed_embeddings: bool = True,
    ):
        """
        Initialize the semantic cache.

        Args:
            similarity_threshold: Minimum similarity for semantic hits (0-1)
            max_entries: Maximum number of cache entries
            strategy: Cache retrieval strategy
            default_ttl: Default time-to-live for cache entries
            enable_precomputed_embeddings: Cache pre-computed embeddings
        """
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.enable_precomputed_embeddings = enable_precomputed_embeddings

        # Get embedder
        self.embedder = get_embedder()

        # Get base cache client
        self.base_cache: Optional[CacheClient] = None
        try:
            self.base_cache = CacheClient()
        except Exception as e:
            logger.warning("Failed to initialize base cache", error=str(e))

        # In-memory cache for fast access
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Pre-computed embeddings cache
        self._embedding_cache: Dict[str, np.ndarray] = {}

        logger.info(
            "SemanticCache initialized",
            strategy=strategy.value,
            similarity_threshold=similarity_threshold,
            max_entries=max_entries,
        )

    def _make_key(self, query: str) -> str:
        """Create a cache key for a query."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"arxiv_rag:semantic_cache:{query_hash}"

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity
        return np.dot(embedding1, embedding2) / (norm1 * norm2)

    def _find_semantic_match(
        self,
        query_embedding: np.ndarray,
    ) -> Optional[Tuple[str, CacheEntry, float]]:
        """Find a semantically similar cached query."""
        best_match = None
        best_similarity = 0.0

        for key, entry in self._memory_cache.items():
            # Skip expired entries
            if entry.is_expired:
                continue

            similarity = self._compute_similarity(query_embedding, entry.query_embedding)

            if similarity >= self.similarity_threshold and similarity > best_similarity:
                best_match = (key, entry, similarity)
                best_similarity = similarity

        return best_match

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        while len(self._memory_cache) >= self.max_entries:
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]

    async def get(
        self,
        query: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached results for a query.

        Tries exact match first, then semantic similarity based on strategy.

        Args:
            query: Search query

        Returns:
            Cached results or None
        """
        # Compute query embedding
        query_embedding = self.embedder.embed_query(query)

        # Try exact match first
        exact_key = self._make_key(query)

        if exact_key in self._memory_cache:
            entry = self._memory_cache[exact_key]

            if not entry.is_expired:
                # Hit!
                entry.hit_count += 1
                # Move to end (most recently used)
                self._memory_cache.move_to_end(exact_key)

                logger.debug(
                    "Exact cache hit",
                    query=query[:50],
                    hit_count=entry.hit_count,
                )

                return entry.results

        # Try semantic match if enabled
        if self.strategy in (CacheStrategy.SEMANTIC_SIMILARITY, CacheStrategy.HYBRID):
            match = self._find_semantic_match(query_embedding)

            if match:
                key, entry, similarity = match

                # Move to end (most recently used)
                self._memory_cache.move_to_end(key)
                entry.hit_count += 1

                logger.debug(
                    "Semantic cache hit",
                    query=query[:50],
                    cached_query=entry.query[:50],
                    similarity=f"{similarity:.3f}",
                )

                return entry.results

        # Cache miss
        logger.debug("Cache miss", query=query[:50])
        return None

    async def set(
        self,
        query: str,
        results: List[Dict[str, Any]],
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store query results in the cache.

        Args:
            query: Search query
            results: Query results to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        # Compute query embedding
        query_embedding = self.embedder.embed_query(query)

        # Create cache entry
        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            results=results,
            timestamp=time.time(),
            ttl=ttl or self.default_ttl,
        )

        # Make key
        key = self._make_key(query)

        # Evict if needed
        self._evict_if_needed()

        # Store in memory cache
        self._memory_cache[key] = entry

        logger.debug(
            "Cached query results",
            query=query[:50],
            ttl=entry.ttl,
            total_entries=len(self._memory_cache),
        )

    async def get_precomputed_embedding(
        self,
        text: str,
    ) -> Optional[np.ndarray]:
        """
        Get pre-computed embedding for common terms.

        Args:
            text: Text to get embedding for

        Returns:
            Pre-computed embedding or None
        """
        if not self.enable_precomputed_embeddings:
            return None

        key = f"arxiv_rag:embed_cache:{hashlib.md5(text.encode()).hexdigest()}"

        # Check memory cache first
        if key in self._embedding_cache:
            return self._embedding_cache[key]

        # Check Redis
        if self.base_cache and self.base_cache.enabled:
            try:
                cached = self.base_cache.get(key)
                if cached:
                    embedding = np.array(cached, dtype=np.float32)
                    # Store in memory cache
                    self._embedding_cache[key] = embedding
                    return embedding
            except Exception as e:
                logger.warning("Failed to get precomputed embedding from Redis", error=str(e))

        return None

    async def set_precomputed_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        ttl: int = 86400,  # 24 hours
    ) -> None:
        """
        Store pre-computed embedding for common terms.

        Args:
            text: Text to store embedding for
            embedding: Pre-computed embedding
            ttl: Time-to-live in seconds
        """
        if not self.enable_precomputed_embeddings:
            return

        key = f"arxiv_rag:embed_cache:{hashlib.md5(text.encode()).hexdigest()}"

        # Store in memory cache
        self._embedding_cache[key] = embedding

        # Store in Redis
        if self.base_cache and self.base_cache.enabled:
            try:
                self.base_cache.set(key, embedding.tolist(), ttl=ttl)
            except Exception as e:
                logger.warning("Failed to store precomputed embedding in Redis", error=str(e))

    async def warm_up(
        self,
        common_queries: List[str],
    ) -> Dict[str, Any]:
        """
        Warm up the cache with common queries.

        Args:
            common_queries: List of common queries to pre-cache

        Returns:
            Statistics about the warm-up process
        """
        logger.info("Starting cache warm-up", queries=len(common_queries))

        start_time = time.time()
        precomputed_count = 0

        for query in common_queries:
            # Pre-compute embeddings
            embedding = self.embedder.embed_query(query)
            await self.set_precomputed_embedding(query, embedding)
            precomputed_count += 1

        elapsed = time.time() - start_time

        result = {
            "total_queries": len(common_queries),
            "precomputed_embeddings": precomputed_count,
            "elapsed_time": elapsed,
            "avg_time_per_query": elapsed / len(common_queries) if common_queries else 0,
        }

        logger.info("Cache warm-up complete", **result)
        return result

    async def invalidate_expired(self) -> int:
        """Invalidate all expired entries."""
        before_count = len(self._memory_cache)

        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if entry.is_expired
        ]

        for key in expired_keys:
            del self._memory_cache[key]

        invalidated = before_count - len(self._memory_cache)
        logger.info("Invalidated expired entries", count=invalidated)
        return invalidated

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_queries = sum(entry.hit_count for entry in self._memory_cache.values())
        exact_matches = sum(1 for entry in self._memory_cache.values() if entry.hit_count > 0)

        return {
            "total_entries": len(self._memory_cache),
            "total_queries": total_queries,
            "exact_hits": exact_matches,
            "similarity_threshold": self.similarity_threshold,
            "strategy": self.strategy.value,
        }

    async def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()
        self._embedding_cache.clear()
        logger.info("Semantic cache cleared")


# Global semantic cache instance
_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get the global semantic cache instance."""
    global _semantic_cache

    if _semantic_cache is None:
        _semantic_cache = SemanticCache(
            similarity_threshold=getattr(settings, 'SEMANTIC_CACHE_SIMILARITY_THRESHOLD', 0.85),
            max_entries=getattr(settings, 'SEMANTIC_CACHE_MAX_ENTRIES', 1000),
            strategy=CacheStrategy.HYBRID,
            default_ttl=getattr(settings, 'CACHE_TTL', 3600),
        )

    return _semantic_cache
