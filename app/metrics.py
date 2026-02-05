"""Prometheus metrics for monitoring RAG system performance."""


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

import time
from functools import wraps
from typing import Callable, Any, Optional

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    REGISTRY,
)

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# METRICS DEFINITIONS
# =============================================================================

# Request metrics
requests_total = Counter(
    "arxiv_rag_requests_total",
    "Total number of requests",
    ["endpoint", "method"],
)

requests_errors = Counter(
    "arxiv_rag_errors_total",
    "Total number of errors",
    ["endpoint", "error_type"],
)

# Latency metrics
request_latency = Histogram(
    "arxiv_rag_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

retrieval_latency = Histogram(
    "arxiv_rag_retrieval_latency_seconds",
    "Retrieval latency in seconds",
    ["method"],  # "semantic", "bm25", "hybrid", "rerank"
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

llm_latency = Histogram(
    "arxiv_rag_llm_latency_seconds",
    "LLM generation latency in seconds",
    ["provider", "model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

# Cache metrics
cache_hits = Counter(
    "arxiv_rag_cache_hits_total",
    "Total number of cache hits",
    ["cache_type"],
)

cache_misses = Counter(
    "arxiv_rag_cache_misses_total",
    "Total number of cache misses",
    ["cache_type"],
)

# Retrieval metrics
documents_retrieved = Histogram(
    "arxiv_rag_documents_retrieved",
    "Number of documents retrieved",
    ["method"],
    buckets=[1, 5, 10, 20, 50, 100],
)

retrieval_scores = Histogram(
    "arxiv_rag_retrieval_scores",
    "Retrieval scores distribution",
    ["method"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Query expansion metrics
query_expansion_hits = Counter(
    "arxiv_rag_query_expansion_hits_total",
    "Total number of queries with expansion applied",
    ["method"],  # "acronym", "related", "both"
)

query_expansion_terms = Histogram(
    "arxiv_rag_query_expansion_terms",
    "Number of terms added via query expansion",
    ["method"],
    buckets=[1, 2, 3, 5, 10, 15, 20],
)

# Reranking metrics
rerank_latency = Histogram(
    "arxiv_rag_rerank_latency_seconds",
    "Reranking latency in seconds",
    ["method"],  # "mmr", "cross_encoder"
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

rerank_candidates = Histogram(
    "arxiv_rag_rerank_candidates",
    "Number of candidates before reranking",
    ["method"],
    buckets=[5, 10, 20, 30, 50, 100],
)

rerank_diversity_score = Gauge(
    "arxiv_rag_rerank_diversity_score",
    "Average diversity score of reranked results (for MMR)",
    ["lambda_value"],
)

# LLM metrics
llm_tokens_total = Counter(
    "arxiv_rag_llm_tokens_total",
    "Total number of LLM tokens used",
    ["provider", "model", "type"],  # type: "prompt" or "completion"
)

llm_requests_total = Counter(
    "arxiv_rag_llm_requests_total",
    "Total number of LLM requests",
    ["provider", "model", "status"],  # status: "success" or "error"
)

# System metrics
index_documents = Gauge(
    "arxiv_rag_index_documents",
    "Number of documents in the index",
)

index_chunks = Gauge(
    "arxiv_rag_index_chunks",
    "Number of chunks in the index",
)

# System info
system_info = Info(
    "arxiv_rag_system",
    "System information",
)

# Set system info
system_info.info({
    "version": settings.VERSION,
    "vectorstore": "chromadb",
    "embed_model": settings.EMBED_MODEL,
    "llm_mode": settings.LLM_MODE,
    "cache_enabled": str(settings.CACHE_ENABLED),
})


# =============================================================================
# METRIC DECORATORS
# =============================================================================

def track_latency(metric: Histogram, labels: dict = None):
    """
    Decorator to track function latency.

    Args:
        metric: Histogram metric to record latency
        labels: Optional labels for the metric

    Example:
        @track_latency(retrieval_latency, {"method": "hybrid"})
        def search(query):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                latency = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(latency)
                else:
                    metric.observe(latency)

        return wrapper
    return decorator


def track_latency_async(metric: Histogram, labels: dict = None):
    """Async version of track_latency decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                latency = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(latency)
                else:
                    metric.observe(latency)

        return wrapper
    return decorator


def track_errors(counter: Counter, labels: dict = None):
    """
    Decorator to track errors.

    Args:
        counter: Counter metric to increment on errors
        labels: Optional labels for the metric

    Example:
        @track_errors(requests_errors, {"endpoint": "/ask"})
        def process_request():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_type = type(e).__name__
                error_labels = {**(labels or {}), "error_type": error_type}
                counter.labels(**error_labels).inc()
                raise

        return wrapper
    return decorator


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def record_retrieval(method: str, count: int, scores: list[float]) -> None:
    """
    Record retrieval metrics.

    Args:
        method: Retrieval method ("semantic", "bm25", "hybrid", etc.)
        count: Number of documents retrieved
        scores: List of retrieval scores
    """
    documents_retrieved.labels(method=method).observe(count)

    for score in scores:
        retrieval_scores.labels(method=method).observe(score)


def record_cache_access(cache_type: str, hit: bool) -> None:
    """
    Record cache access.

    Args:
        cache_type: Type of cache ("query", "embed", etc.)
        hit: Whether it was a cache hit
    """
    if hit:
        cache_hits.labels(cache_type=cache_type).inc()
    else:
        cache_misses.labels(cache_type=cache_type).inc()


def record_llm_request(
    provider: str,
    model: str,
    success: bool,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """
    Record LLM request metrics.

    Args:
        provider: LLM provider ("openrouter", "ollama", etc.)
        model: Model name
        success: Whether request succeeded
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
    """
    status = "success" if success else "error"
    llm_requests_total.labels(provider=provider, model=model, status=status).inc()

    if prompt_tokens > 0:
        llm_tokens_total.labels(provider=provider, model=model, type="prompt").inc(
            prompt_tokens
        )

    if completion_tokens > 0:
        llm_tokens_total.labels(
            provider=provider, model=model, type="completion"
        ).inc(completion_tokens)


def update_index_stats(documents: int, chunks: int) -> None:
    """
    Update index statistics.

    Args:
        documents: Number of documents in index
        chunks: Number of chunks in index
    """
    index_documents.set(documents)
    index_chunks.set(chunks)


def record_query_expansion(method: str, terms_added: int) -> None:
    """
    Record query expansion metrics.

    Args:
        method: Expansion method used ("acronym", "related", "both")
        terms_added: Number of terms added
    """
    query_expansion_hits.labels(method=method).inc()
    query_expansion_terms.labels(method=method).observe(terms_added)


def record_reranking(
    method: str,
    candidates: int,
    latency_seconds: float,
    diversity_score: Optional[float] = None,
) -> None:
    """
    Record reranking metrics.

    Args:
        method: Reranking method ("mmr", "cross_encoder")
        candidates: Number of candidates before reranking
        latency_seconds: Time taken for reranking
        diversity_score: Average diversity score (for MMR)
    """
    rerank_latency.labels(method=method).observe(latency_seconds)
    rerank_candidates.labels(method=method).observe(candidates)

    if diversity_score is not None and method == "mmr":
        # Store diversity score with lambda as label (rounded to 1 decimal)
        lambda_label = str(int(diversity_score * 10) / 10)
        rerank_diversity_score.labels(lambda_value=lambda_label).set(diversity_score)


def get_metrics() -> bytes:
    """
    Get Prometheus metrics in text format.

    Returns:
        Metrics in Prometheus text format
    """
    return generate_latest(REGISTRY)


def get_metrics_summary() -> dict:
    """
    Get a human-readable summary of key metrics.

    Returns:
        Dictionary with metric summaries
    """
    return {
        "requests": {
            "total": sum(
                sample.value
                for family in REGISTRY.collect()
                if family.name == "arxiv_rag_requests_total"
                for sample in family.samples
            ),
            "errors": sum(
                sample.value
                for family in REGISTRY.collect()
                if family.name == "arxiv_rag_errors_total"
                for sample in family.samples
            ),
        },
        "cache": {
            "hits": sum(
                sample.value
                for family in REGISTRY.collect()
                if family.name == "arxiv_rag_cache_hits_total"
                for sample in family.samples
            ),
            "misses": sum(
                sample.value
                for family in REGISTRY.collect()
                if family.name == "arxiv_rag_cache_misses_total"
                for sample in family.samples
            ),
        },
        "index": {
            "documents": index_documents._value.get(),
            "chunks": index_chunks._value.get(),
        },
        "llm": {
            "requests": sum(
                sample.value
                for family in REGISTRY.collect()
                if family.name == "arxiv_rag_llm_requests_total"
                for sample in family.samples
            ),
            "tokens": sum(
                sample.value
                for family in REGISTRY.collect()
                if family.name == "arxiv_rag_llm_tokens_total"
                for sample in family.samples
            ),
        },
    }


# =============================================================================
# INITIALIZATION
# =============================================================================

def init_metrics():
    """Initialize metrics (called on startup)."""
    logger.info(
        "Metrics initialized",
        enabled=settings.METRICS_ENABLED,
        port=settings.METRICS_PORT,
    )


# Initialize on module load
if settings.METRICS_ENABLED:
    init_metrics()
