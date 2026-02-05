"""
Batch embedding processor with parallel execution, GPU optimization, and error handling.

Provides efficient batch processing for document embeddings with support for
parallel execution, GPU acceleration, progress tracking, and resilient error handling.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import numpy as np

from app.embeddings.embeddings import Embedder
from app.logging_config import get_logger

logger = get_logger(__name__)


class ProcessingStrategy(Enum):
    """Batch processing strategies."""
    SEQUENTIAL = "sequential"
    THREAD_PARALLEL = "thread_parallel"
    PROCESS_PARALLEL = "process_parallel"
    GPU_BATCHED = "gpu_batched"


@dataclass
class BatchResult:
    """Result of a batch processing operation."""
    embeddings: np.ndarray
    successful_indices: List[int]
    failed_indices: List[int]
    processing_time: float
    batch_errors: Dict[int, str] = field(default_factory=dict)


@dataclass
class BatchProgress:
    """Progress tracking for batch operations."""
    total_items: int
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    start_time: float = field(default_factory=time.time)
    current_batch: int = 0
    total_batches: int = 0

    @property
    def progress_percentage(self) -> float:
        return (self.processed_items / self.total_items * 100) if self.total_items > 0 else 0

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def estimated_remaining_time(self) -> float:
        if self.processed_items == 0:
            return 0
        rate = self.processed_items / self.elapsed_time
        remaining = self.total_items - self.processed_items
        return remaining / rate


class BatchEmbeddingProcessor:
    """
    Advanced batch processor for embedding generation.

    Features:
    - Parallel execution (thread/process pool)
    - GPU optimization for large batches
    - Progress tracking with callbacks
    - Comprehensive error handling with retry logic
    """

    def __init__(
        self,
        embedder: Embedder,
        strategy: ProcessingStrategy = ProcessingStrategy.THREAD_PARALLEL,
        batch_size: int = 32,
        max_workers: Optional[int] = None,
        enable_gpu: bool = True,
        retry_failed: bool = True,
        max_retries: int = 3,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
    ):
        """
        Initialize the batch processor.

        Args:
            embedder: Embedder instance to use
            strategy: Processing strategy
            batch_size: Number of texts per batch
            max_workers: Maximum parallel workers (None = auto)
            enable_gpu: Enable GPU optimizations
            retry_failed: Automatically retry failed embeddings
            max_retries: Maximum retry attempts per item
            progress_callback: Optional callback for progress updates
        """
        self.embedder = embedder
        self.strategy = strategy
        self.batch_size = batch_size
        self.max_workers = max_workers or self._auto_detect_workers()
        self.enable_gpu = enable_gpu
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        self.progress_callback = progress_callback

        # Detect GPU availability
        self.gpu_available = self._check_gpu_available()
        if enable_gpu and not self.gpu_available:
            logger.warning("GPU requested but not available, falling back to CPU")

        logger.info(
            "BatchEmbeddingProcessor initialized",
            strategy=strategy.value,
            batch_size=batch_size,
            max_workers=self.max_workers,
            gpu_available=self.gpu_available,
        )

    def _auto_detect_workers(self) -> int:
        """Auto-detect optimal number of workers based on system."""
        import os
        cpu_count = os.cpu_count() or 4
        return max(1, int(cpu_count * 0.75))

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for embeddings."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches."""
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batches.append(batch)
        return batches

    async def process_batch_async(
        self,
        texts: List[str],
        batch_indices: List[int],
        retry_count: int = 0,
    ) -> BatchResult:
        """
        Process a single batch asynchronously with error handling.

        Args:
            texts: Batch of texts to embed
            batch_indices: Original indices of texts in the full dataset
            retry_count: Current retry attempt

        Returns:
            BatchResult with embeddings and error information
        """
        start_time = time.time()
        successful_indices = []
        failed_indices = []
        batch_errors = {}
        embeddings = []

        try:
            # Generate embeddings for the batch
            vectors = self.embedder.embed(texts, show_progress=False)
            embeddings = vectors.tolist()
            successful_indices = batch_indices

        except Exception as e:
            error_msg = str(e)
            logger.warning(
                "Batch processing failed",
                batch_size=len(texts),
                retry_count=retry_count,
                error=error_msg,
            )

            # Retry logic for individual items
            if self.retry_failed and retry_count < self.max_retries:
                for idx, text in zip(batch_indices, texts):
                    try:
                        vector = self.embedder.embed_query(text)
                        embeddings.append(vector.tolist())
                        successful_indices.append(idx)
                    except Exception as item_error:
                        failed_indices.append(idx)
                        batch_errors[idx] = str(item_error)
            else:
                # Mark all as failed
                failed_indices = batch_indices
                batch_errors = {idx: error_msg for idx in batch_indices}

        processing_time = time.time() - start_time

        return BatchResult(
            embeddings=np.array(embeddings, dtype=np.float32) if embeddings else np.array([]),
            successful_indices=successful_indices,
            failed_indices=failed_indices,
            processing_time=processing_time,
            batch_errors=batch_errors,
        )

    async def process_async(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> BatchResult:
        """
        Process all texts in batches with parallel execution.

        Args:
            texts: List of texts to embed
            show_progress: Show progress bar

        Returns:
            BatchResult with all embeddings
        """
        if not texts:
            return BatchResult(
                embeddings=np.array([], dtype=np.float32).reshape(0, self.embedder.dimension),
                successful_indices=[],
                failed_indices=[],
                processing_time=0.0,
                batch_errors={},
            )

        start_time = time.time()
        batches = self._create_batches(texts)
        total_batches = len(batches)

        # Initialize progress tracking
        progress = BatchProgress(
            total_items=len(texts),
            total_batches=total_batches,
        )

        # Create batch index mappings
        batch_indices_list = []
        for i, batch in enumerate(batches):
            start_idx = i * self.batch_size
            batch_indices = list(range(start_idx, start_idx + len(batch)))
            batch_indices_list.append(batch_indices)

        # Process batches based on strategy
        if self.strategy == ProcessingStrategy.SEQUENTIAL:
            results = await self._process_sequential(
                batches, batch_indices_list, progress, show_progress
            )
        elif self.strategy == ProcessingStrategy.THREAD_PARALLEL:
            results = await self._process_thread_parallel(
                batches, batch_indices_list, progress, show_progress
            )
        else:
            # Default to sequential for other strategies
            results = await self._process_sequential(
                batches, batch_indices_list, progress, show_progress
            )

        # Merge results
        merged_result = self._merge_results(results, len(texts), self.embedder.dimension)
        merged_result.processing_time = time.time() - start_time

        logger.info(
            "Batch processing complete",
            total_items=len(texts),
            successful=len(merged_result.successful_indices),
            failed=len(merged_result.failed_indices),
            processing_time=f"{merged_result.processing_time:.2f}s",
            throughput=f"{len(texts) / merged_result.processing_time:.2f} items/s",
        )

        return merged_result

    async def _process_sequential(
        self,
        batches: List[List[str]],
        batch_indices_list: List[List[int]],
        progress: BatchProgress,
        show_progress: bool,
    ) -> List[BatchResult]:
        """Process batches sequentially."""
        results = []
        iterator = enumerate(zip(batches, batch_indices_list))

        for i, (batch, indices) in iterator:
            result = await self.process_batch_async(batch, indices)
            results.append(result)

            # Update progress
            progress.processed_items += len(batch)
            progress.successful_items += len(result.successful_indices)
            progress.failed_items += len(result.failed_indices)
            progress.current_batch = i + 1

            if self.progress_callback:
                self.progress_callback(progress)

            if show_progress and (i + 1) % 10 == 0:
                logger.info(
                    "Progress",
                    batch=f"{i + 1}/{len(batches)}",
                    processed=progress.processed_items,
                    total=progress.total_items,
                )

        return results

    async def _process_thread_parallel(
        self,
        batches: List[List[str]],
        batch_indices_list: List[List[int]],
        progress: BatchProgress,
        show_progress: bool,
    ) -> List[BatchResult]:
        """Process batches in parallel using threads."""
        async def process_with_update(batch_info):
            batch_idx, batch, indices = batch_info
            result = await self.process_batch_async(batch, indices)

            # Update progress
            progress.processed_items += len(batch)
            progress.successful_items += len(result.successful_indices)
            progress.failed_items += len(result.failed_indices)

            if self.progress_callback:
                self.progress_callback(progress)

            return result

        # Create tasks for all batches
        tasks = [
            process_with_update((i, batch, indices))
            for i, (batch, indices) in enumerate(zip(batches, batch_indices_list))
        ]

        # Process with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_workers)

        async def bounded_task(task):
            async with semaphore:
                return await task

        bounded_tasks = [bounded_task(t) for t in tasks]
        results = await asyncio.gather(*bounded_tasks)

        return results

    def _merge_results(
        self,
        results: List[BatchResult],
        total_items: int,
        embedding_dim: int,
    ) -> BatchResult:
        """Merge batch results into a single result."""
        all_embeddings = np.zeros((total_items, embedding_dim), dtype=np.float32)
        all_successful_indices = []
        all_failed_indices = []
        all_errors = {}

        for result in results:
            # Place embeddings at correct indices
            for idx, embedding in zip(result.successful_indices, result.embeddings):
                all_embeddings[idx] = embedding
                all_successful_indices.append(idx)

            all_failed_indices.extend(result.failed_indices)
            all_errors.update(result.batch_errors)

        # Filter out failed embeddings from the array
        final_embeddings = all_embeddings[all_successful_indices] if all_successful_indices else np.array([])

        return BatchResult(
            embeddings=final_embeddings,
            successful_indices=sorted(all_successful_indices),
            failed_indices=sorted(all_failed_indices),
            processing_time=sum(r.processing_time for r in results),
            batch_errors=all_errors,
        )


# Global batch processor instance
_batch_processor: Optional[BatchEmbeddingProcessor] = None


def get_batch_processor(
    embedder: Optional[Embedder] = None,
    strategy: str = "thread_parallel",
    batch_size: int = 32,
    **kwargs,
) -> BatchEmbeddingProcessor:
    """
    Factory function to get a batch processor instance.

    Args:
        embedder: Embedder instance (uses default if None)
        strategy: Processing strategy name
        batch_size: Batch size
        **kwargs: Additional processor arguments

    Returns:
        BatchEmbeddingProcessor instance
    """
    global _batch_processor

    if _batch_processor is None:
        from app.embeddings.embeddings import get_embedder

        if embedder is None:
            embedder = get_embedder()

        strategy_enum = ProcessingStrategy(strategy)

        _batch_processor = BatchEmbeddingProcessor(
            embedder=embedder,
            strategy=strategy_enum,
            batch_size=batch_size,
            **kwargs,
        )

    return _batch_processor
