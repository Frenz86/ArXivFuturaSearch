"""
Cache warming strategies for pre-loading common queries.

Provides utilities for warming up the semantic cache with common queries
and frequently accessed embeddings.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.cache.semantic import get_semantic_cache
from app.embeddings.embeddings import get_embedder
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class WarmupStats:
    """Statistics for cache warm-up."""
    total_queries: int
    successful_warmups: int
    failed_warmups: int
    elapsed_time: float

    @property
    def success_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.successful_warmups / self.total_queries


class CacheWarmer:
    """
    Cache warming utility for pre-loading common queries.

    Strategies:
    1. Static list of common queries
    2. Historical query analysis
    3. Domain-specific term pre-computation
    """

    # Common ML/AI search queries
    COMMON_QUERIES = [
        "machine learning",
        "deep learning",
        "neural networks",
        "transformer architecture",
        "attention mechanism",
        "large language models",
        "reinforcement learning",
        "computer vision",
        "natural language processing",
        "graph neural networks",
        "generative adversarial networks",
        "transfer learning",
        "federated learning",
        "contrastive learning",
        "self-supervised learning",
        "prompt engineering",
        "retrieval augmented generation",
        "vector databases",
        "embedding models",
        "fine-tuning",
    ]

    def __init__(
        self,
        semantic_cache=None,
        enable_static_warmup: bool = True,
        enable_historical_warmup: bool = False,
        warmup_concurrency: int = 5,
    ):
        """
        Initialize the cache warmer.

        Args:
            semantic_cache: Semantic cache instance (uses default if None)
            enable_static_warmup: Enable static query warm-up
            enable_historical_warmup: Enable historical query warm-up
            warmup_concurrency: Number of concurrent warm-up tasks
        """
        self.semantic_cache = semantic_cache
        self.enable_static_warmup = enable_static_warmup
        self.enable_historical_warmup = enable_historical_warmup
        self.warmup_concurrency = warmup_concurrency

        if self.semantic_cache is None:
            self.semantic_cache = get_semantic_cache()

        self.embedder = get_embedder()

        logger.info(
            "CacheWarmer initialized",
            static_warmup=enable_static_warmup,
            historical_warmup=enable_historical_warmup,
        )

    async def warm_up_static_queries(self) -> WarmupStats:
        """
        Warm up cache with static common queries.

        Returns:
            Warmup statistics
        """
        if not self.enable_static_warmup:
            logger.info("Static warm-up disabled")
            return WarmupStats(0, 0, 0, 0.0)

        logger.info("Starting static cache warm-up", queries=len(self.COMMON_QUERIES))

        import time
        start_time = time.time()

        successful = 0
        failed = 0

        # Process queries concurrently
        semaphore = asyncio.Semaphore(self.warmup_concurrency)

        async def warm_query(query: str) -> bool:
            async with semaphore:
                try:
                    # Pre-compute embedding
                    embedding = self.embedder.embed_query(query)
                    await self.semantic_cache.set_precomputed_embedding(query, embedding)
                    return True
                except Exception as e:
                    logger.warning("Failed to warm up query", query=query, error=str(e))
                    return False

        tasks = [warm_query(q) for q in self.COMMON_QUERIES]
        results = await asyncio.gather(*tasks)

        successful = sum(results)
        failed = len(results) - successful
        elapsed = time.time() - start_time

        stats = WarmupStats(
            total_queries=len(self.COMMON_QUERIES),
            successful_warmups=successful,
            failed_warmups=failed,
            elapsed_time=elapsed,
        )

        logger.info(
            "Static cache warm-up complete",
            **{
                "total": stats.total_queries,
                "successful": successful,
                "failed": failed,
                "elapsed": f"{elapsed:.2f}s",
            }
        )

        return stats

    async def warm_up_common_terms(
        self,
        terms: Optional[List[str]] = None,
    ) -> WarmupStats:
        """
        Warm up cache with common domain terms.

        Args:
            terms: List of terms to pre-compute (uses defaults if None)

        Returns:
            Warmup statistics
        """
        if terms is None:
            terms = [
                "attention",
                "transformer",
                "embedding",
                "vector",
                "gradient",
                "backpropagation",
                "optimization",
                "regularization",
                "normalization",
                "activation",
                "convolution",
                "recurrent",
                "attention mechanism",
                "loss function",
                "hyperparameters",
                "batch normalization",
                "dropout",
                "fine-tuning",
                "tokenization",
                "softmax",
            ]

        logger.info("Starting common terms warm-up", terms=len(terms))

        import time
        start_time = time.time()

        successful = 0
        failed = 0

        for term in terms:
            try:
                embedding = self.embedder.embed_query(term)
                await self.semantic_cache.set_precomputed_embedding(term, embedding)
                successful += 1
            except Exception as e:
                logger.warning("Failed to warm up term", term=term, error=str(e))
                failed += 1

        elapsed = time.time() - start_time

        stats = WarmupStats(
            total_queries=len(terms),
            successful_warmups=successful,
            failed_warmups=failed,
            elapsed_time=elapsed,
        )

        logger.info(
            "Common terms warm-up complete",
            **{
                "total": stats.total_queries,
                "successful": successful,
                "failed": failed,
                "elapsed": f"{elapsed:.2f}s",
            }
        )

        return stats

    async def warm_up_all(self) -> Dict[str, WarmupStats]:
        """
        Run all warm-up strategies.

        Returns:
            Dictionary with statistics for each strategy
        """
        logger.info("Starting comprehensive cache warm-up")

        results = {}

        # Static queries
        results["static"] = await self.warm_up_static_queries()

        # Common terms
        results["terms"] = await self.warm_up_common_terms()

        # Summary
        total_queries = sum(s.total_queries for s in results.values())
        total_successful = sum(s.successful_warmups for s in results.values())
        total_elapsed = sum(s.elapsed_time for s in results.values())

        logger.info(
            "Comprehensive cache warm-up complete",
            total_queries=total_queries,
            total_successful=total_successful,
            total_elapsed=f"{total_elapsed:.2f}s",
        )

        return results


# Global cache warmer instance
_cache_warmer: Optional[CacheWarmer] = None


def get_cache_warmer() -> CacheWarmer:
    """Get the global cache warmer instance."""
    global _cache_warmer

    if _cache_warmer is None:
        _cache_warmer = CacheWarmer()

    return _cache_warmer


async def warm_up_cache_on_startup() -> None:
    """
    Warm up cache when application starts.

    Should be called during application initialization.
    """
    logger.info("Starting cache warm-up on startup")

    warmer = get_cache_warmer()
    results = await warmer.warm_up_all()

    logger.info("Cache warm-up on startup complete", results=results)
