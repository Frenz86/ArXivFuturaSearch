"""Comprehensive Pydantic schemas for request/response validation.


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

Provides granular validation for all API endpoints with detailed field validators,
custom error messages, and type safety.
"""

import re
from datetime import datetime, UTC
from typing import Optional, List, Dict, Any, Literal, Union
from enum import Enum

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    StringConstraints,
)
from pydantic.types import PositiveInt

from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# COMMON SCHEMAS
# =============================================================================

class ArxivCategory(str):
    """Valid ArXiv category."""

    @classmethod
    def __get_priors__(cls):
        # Common CS categories
        return {
            "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "stat.ML",
            "cs.CR", "cs.DC", "cs.DS", "cs.DB", "cs.GL", "cs.GR",
            "cs.AR", "cs.CC", "cs.CE", "cs.IT", "cs.LO", "cs.MA",
            "cs.MM", "cs.MS", "cs.NA", "cs.NI", "cs.OH", "cs.OS",
            "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE",
        }


class SortOrder(str, Enum):
    """Sort order for search results."""
    ASC = "ascending"
    DESC = "descending"
    RELEVANCE = "relevance"


class RerankMethod(str, Enum):
    """Reranking method options."""
    NONE = "none"
    MMR = "mmr"
    CROSS_ENCODER = "cross_encoder"
    HYBRID = "hybrid"


class SearchMethod(str, Enum):
    """Search method options."""
    VECTOR = "vector"
    HYBRID = "hybrid"
    BM25 = "bm25"
    MULTI_QUERY = "multi_query"


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class BaseSearchRequest(BaseModel):
    """Base schema for all search requests."""

    top_k: PositiveInt = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return",
        json_schema_extra={"examples": [5, 10, 20]},
    )

    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters for search",
        json_schema_extra={
            "examples": [
                {"published_after": "2023-01-01"},
                {"categories": ["cs.AI", "cs.LG"]},
            ]
        },
    )

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate filter keys and values."""
        if v is None:
            return v

        for key, value in v.items():
            # Validate filter key
            if not isinstance(key, str):
                raise ValueError(f"Filter key must be string, got {type(key).__name__}")

            if len(key) > 100:
                raise ValueError(f"Filter key too long: {key[:50]}...")

            # Validate filter value
            if isinstance(value, str) and len(value) > 500:
                raise ValueError(f"Filter value too long for key: {key}")

            # Check for dangerous patterns
            if isinstance(value, str):
                dangerous_patterns = ["<script", "javascript:", "onerror=", "onload="]
                if any(pattern in value.lower() for pattern in dangerous_patterns):
                    raise ValueError(f"Dangerous content in filter value for key: {key}")

        return v


class AskRequest(BaseSearchRequest):
    """Request schema for /ask endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Question to ask the RAG system",
        json_schema_extra={"examples": ["What is attention mechanism in transformers?"]},
    )

    stream: bool = Field(
        default=False,
        description="Enable streaming response",
    )

    use_cot: bool = Field(
        default=True,
        description="Use chain-of-thought prompting",
    )

    rerank_method: Optional[RerankMethod] = Field(
        default=None,
        description="Reranking method to apply",
    )

    query_expansion: bool = Field(
        default=True,
        description="Enable automatic query expansion",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate question content."""
        # Remove excessive whitespace
        v = " ".join(v.split())

        # Check for empty or whitespace-only questions
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace only")

        # Check for injection attempts
        dangerous_keywords = ["drop table", "truncate", "delete from", "union select"]
        if any(kw in v.lower() for kw in dangerous_keywords):
            logger.warning("Potential SQL injection attempt detected", question=v[:100])

        return v


class BuildIndexRequest(BaseModel):
    """Request schema for /build endpoint."""

    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="ArXiv search query",
        json_schema_extra={
            "examples": [
                "cat:cs.LG AND (machine learning OR deep learning)",
                "all:transformer architecture",
            ]
        },
    )

    max_results: PositiveInt = Field(
        default=30,
        ge=1,
        le=500,
        description="Maximum number of papers to fetch",
    )

    chunk_size: PositiveInt = Field(
        default=900,
        ge=100,
        le=2000,
        description="Chunk size for text splitting",
    )

    chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=500,
        description="Overlap between chunks",
    )

    semantic_chunking: bool = Field(
        default=False,
        description="Use semantic chunking",
    )

    @field_validator("query")
    @classmethod
    def validate_arxiv_query(cls, v: str) -> str:
        """Validate ArXiv query syntax."""
        v = v.strip()

        # Basic ArXiv query syntax validation
        # Check for balanced parentheses
        if v.count("(") != v.count(")"):
            raise ValueError("Unbalanced parentheses in query")

        # Check for common invalid patterns
        invalid_patterns = ["&&", "||", ";", "--"]
        for pattern in invalid_patterns:
            if pattern in v:
                raise ValueError(f"Invalid pattern in query: {pattern}")

        return v

    @model_validator(mode="after")
    def validate_chunk_params(self) -> "BuildIndexRequest":
        """Validate chunk parameters are consistent."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class HybridSearchRequest(BaseSearchRequest):
    """Request schema for /search/hybrid endpoint."""

    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Search query",
    )

    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector search (0-1)",
    )

    rrf_k: int = Field(
        default=60,
        ge=1,
        le=100,
        description="RRF constant for score fusion",
    )


class MultiQuerySearchRequest(BaseSearchRequest):
    """Request schema for /search/multi-query endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Original search query",
    )

    num_queries: int = Field(
        default=4,
        ge=2,
        le=10,
        description="Number of query variants to generate",
    )

    merge_strategy: Literal["rrf", "weighted", "union"] = Field(
        default="rrf",
        description="Strategy for merging query results",
    )


class RerankedSearchRequest(BaseSearchRequest):
    """Request schema for /search/rerank endpoint."""

    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Search query",
    )

    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model to use",
    )

    candidates_multiplier: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Multiplier for initial candidate retrieval",
    )


class AutocompleteRequest(BaseModel):
    """Request schema for /suggest endpoint."""

    q: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Query prefix for autocomplete",
    )

    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of suggestions",
    )

    include_trending: bool = Field(
        default=True,
        description="Include trending queries for empty prefix",
    )

    suggestion_type: Literal["prefix", "semantic", "fuzzy", "all"] = Field(
        default="all",
        description="Type of suggestions to return",
    )


class EvaluateRequest(BaseModel):
    """Request schema for /evaluate endpoint."""

    test_dataset_path: Optional[str] = Field(
        default=None,
        description="Path to custom test dataset (JSON/CSV)",
    )

    rerank_method: RerankMethod = Field(
        default=RerankMethod.MMR,
        description="Reranking method to evaluate",
    )

    use_cot_prompting: bool = Field(
        default=True,
        description="Use chain-of-thought prompting",
    )

    num_samples: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of samples to evaluate (None = all)",
    )

    split_ratio: float = Field(
        default=0.8,
        ge=0.5,
        le=0.95,
        description="Train/test split ratio",
    )


class CacheClearRequest(BaseModel):
    """Request schema for /cache/clear endpoint."""

    pattern: str = Field(
        default="arxiv_rag:*",
        min_length=1,
        max_length=200,
        description="Cache key pattern to clear",
    )

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str) -> str:
        """Validate cache key pattern."""
        # Only allow alphanumeric, underscore, colon, asterisk, dash
        if not re.match(r'^[a-zA-Z0-9_:*\-.]+$', v):
            raise ValueError("Invalid pattern format")
        return v


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class SourceMetadata(BaseModel):
    """Metadata for a source document."""

    title: str = Field(..., description="Paper title")
    authors: str = Field(default="", description="Authors")
    published: str = Field(..., description="Publication date")
    link: str = Field(..., description="ArXiv link")
    tags: str = Field(default="", description="Comma-separated tags")


class SourceResult(BaseModel):
    """Single search result."""

    rank: int = Field(..., ge=1, description="Result rank")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    title: Optional[str] = Field(None, description="Paper title")
    link: Optional[str] = Field(None, description="ArXiv link")
    published: Optional[str] = Field(None, description="Publication date")
    authors: Optional[str] = Field(None, description="Authors")
    excerpt: Optional[str] = Field(None, description="Text excerpt")


class RetrievalInfo(BaseModel):
    """Information about the retrieval process."""

    candidates_retrieved: int = Field(..., ge=0)
    rerank_method: Optional[str] = None
    final_count: int = Field(..., ge=0)
    filters_applied: bool
    retrieval_time_ms: int = Field(..., ge=0)


class AskResponse(BaseModel):
    """Response schema for /ask endpoint."""

    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    retrieval_info: RetrievalInfo = Field(..., description="Retrieval metadata")


class BuildResponse(BaseModel):
    """Response schema for /build endpoint."""

    status: str = Field(..., description="Build status")
    stats: Dict[str, Any] = Field(..., description="Build statistics")


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""

    status: str = Field(..., description="Service status")
    llm_mode: str = Field(..., description="LLM provider")
    llm_health: Dict[str, Any] = Field(..., description="LLM health check")
    index_loaded: bool = Field(..., description="Whether index is loaded")
    index_documents: int = Field(..., ge=0, description="Number of indexed documents")
    embedder_loaded: bool = Field(..., description="Whether embedder is loaded")
    embedder_model: Optional[str] = None
    reranker_enabled: bool = Field(..., description="Whether reranking is enabled")
    query_expansion_enabled: bool = Field(..., description="Whether query expansion is enabled")


class HybridSearchResponse(BaseModel):
    """Response schema for /search/hybrid endpoint."""

    query: str = Field(..., description="Search query")
    method: str = Field(..., description="Search method used")
    results: List[Dict[str, Any]] = Field(..., description="Search results")


class MultiQueryResponse(BaseModel):
    """Response schema for /search/multi-query endpoint."""

    original_query: str = Field(..., description="Original query")
    expanded_queries: List[str] = Field(..., description="Generated query variants")
    method: str = Field(..., description="Search method")
    results: List[Dict[str, Any]] = Field(..., description="Merged results")


class RerankedSearchResponse(BaseModel):
    """Response schema for /search/rerank endpoint."""

    query: str = Field(..., description="Search query")
    method: str = Field(..., description="Reranking method")
    candidates_retrieved: int = Field(..., description="Number of candidates")
    results: List[Dict[str, Any]] = Field(..., description="Reranked results")


class AutocompleteSuggestion(BaseModel):
    """Single autocomplete suggestion."""

    text: str = Field(..., description="Suggested text")
    type: str = Field(..., description="Suggestion type (prefix_match, fuzzy_match, etc.)")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class AutocompleteResponse(BaseModel):
    """Response schema for /suggest endpoint."""

    prefix: str = Field(..., description="Query prefix")
    suggestions: List[AutocompleteSuggestion] = Field(..., description="Suggestions")


class MetricsSummary(BaseModel):
    """Summary of Prometheus metrics."""

    total_requests: int = Field(..., ge=0)
    avg_latency_ms: float = Field(..., ge=0.0)
    p95_latency_ms: float = Field(..., ge=0.0)
    error_rate: float = Field(..., ge=0.0, le=1.0)


class CacheStatsResponse(BaseModel):
    """Response schema for /cache/stats endpoint."""

    enabled: bool = Field(..., description="Whether cache is enabled")
    type: str = Field(..., description="Cache type (memory/redis)")
    stats: Dict[str, Any] = Field(..., description="Cache statistics")


class EvaluationResult(BaseModel):
    """Single evaluation result."""

    ragas_scores: Dict[str, float] = Field(..., description="RAGAS metric scores")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    num_samples: int = Field(..., ge=0, description="Number of samples evaluated")


class ComparisonResult(BaseModel):
    """Comparative evaluation result."""

    configurations: List[str] = Field(..., description="Compared configurations")
    results: Dict[str, EvaluationResult] = Field(..., description="Results per config")
    winner: str = Field(..., description="Best performing configuration")


# =============================================================================
# ERROR SCHEMAS
# =============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed errors")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Error timestamp")


# =============================================================================
# PAGINATION SCHEMAS
# =============================================================================

class PaginatedRequest(BaseModel):
    """Base schema for paginated requests."""

    page: int = Field(
        default=1,
        ge=1,
        description="Page number (1-indexed)",
    )

    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Items per page",
    )

    sort_by: Optional[str] = Field(
        default=None,
        description="Field to sort by",
    )

    sort_order: SortOrder = Field(
        default=SortOrder.DESC,
        description="Sort direction",
    )


class PaginatedResponse(BaseModel):
    """Base schema for paginated responses."""

    items: List[Any] = Field(..., description="Page items")
    total: int = Field(..., ge=0, description="Total items")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total pages")
    has_next: bool = Field(..., description="Whether next page exists")
    has_prev: bool = Field(..., description="Whether previous page exists")

    @classmethod
    def create(
        cls,
        items: List[Any],
        total: int,
        page: int,
        page_size: int,
    ) -> "PaginatedResponse":
        """Create paginated response."""
        total_pages = (total + page_size - 1) // page_size
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )


# =============================================================================
# WEBHOOK SCHEMAS
# =============================================================================

class WebhookEventType(str, Enum):
    """Webhook event types."""
    INDEX_BUILT = "index.built"
    INDEX_UPDATED = "index.updated"
    ERROR = "error"
    HEALTH_CHECK = "health.check"


class WebhookPayload(BaseModel):
    """Webhook payload."""

    event_type: WebhookEventType = Field(..., description="Type of event")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: Dict[str, Any] = Field(..., description="Event data")
    signature: Optional[str] = Field(None, description="HMAC signature for verification")


# =============================================================================
# BATCH OPERATION SCHEMAS
# =============================================================================

class BatchSearchRequest(BaseModel):
    """Request for batch search operations."""

    queries: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of queries to search",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Results per query",
    )

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, v: List[str]) -> List[str]:
        """Validate all queries in batch."""
        for q in v:
            if not q.strip():
                raise ValueError("Queries cannot be empty")
            if len(q) > 1000:
                raise ValueError(f"Query too long: {q[:50]}...")
        return v


class BatchSearchResponse(BaseModel):
    """Response for batch search operations."""

    results: List[List[Dict[str, Any]]] = Field(..., description="Results for each query")
    total_queries: int = Field(..., ge=1)
    total_results: int = Field(..., ge=0)
    processing_time_ms: int = Field(..., ge=0)
