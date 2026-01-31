"""Type-safe data models for internal RAG structures.


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

Provides Pydantic models for retrieval results, chunks, and other
internal data structures to improve type safety and reduce bugs.
"""

from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# CHUNK MODELS
# =============================================================================

class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""

    title: str
    link: str
    published: str
    authors: str = ""
    tags: str = ""
    chunk_id: str = ""
    paper_id: str = ""
    chunk_index: int = 0
    total_chunks: int = 1

    @field_validator("published")
    @classmethod
    def validate_published(cls, v: str) -> str:
        """Ensure published date is a valid string."""
        if not v:
            return "Unknown"
        return v


class Chunk(BaseModel):
    """A document chunk with text and metadata."""

    text: str
    meta: ChunkMetadata
    chunk_id: str = ""
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary format for vector store."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "meta": self.meta.model_dump(),
        }


# =============================================================================
# RETRIEVAL RESULT MODELS
# =============================================================================

class RetrievalResult(BaseModel):
    """A single retrieval result with scoring information."""

    text: str
    meta: ChunkMetadata
    score: float = Field(ge=0.0, le=1.0, description="Hybrid search score")
    rerank_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Reranking score (if reranking was applied)"
    )
    mmr_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="MMR score (if MMR was applied)"
    )

    def get_final_score(self) -> float:
        """Get the final score (prioritize rerank > mmr > hybrid)."""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.mmr_score is not None:
            return self.mmr_score
        return self.score

    def to_source_dict(self, rank: int) -> dict:
        """Convert to source dictionary for API response."""
        return {
            "rank": rank + 1,
            "score": self.get_final_score(),
            "hybrid_score": self.score,
            "title": self.meta.title,
            "link": self.meta.link,
            "published": self.meta.published,
            "authors": self.meta.authors,
        }


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class RetrievalInfo(BaseModel):
    """Information about the retrieval process."""

    candidates_retrieved: int = Field(ge=0, description="Number of candidates retrieved")
    rerank_method: Optional[str] = Field(
        default=None,
        description="Reranking method used (MMR, Cross-Encoder, None)"
    )
    final_count: int = Field(ge=0, description="Final number of results")
    filters_applied: bool = Field(default=False, description="Whether filters were applied")
    retrieval_time_ms: int = Field(ge=0, description="Retrieval time in milliseconds")
    query_expansion_used: bool = Field(default=False, description="Whether query expansion was used")
    expansion_terms: list[str] = Field(default_factory=list, description="Terms added via expansion")


class SourceInfo(BaseModel):
    """Information about a source document."""

    rank: int = Field(ge=1, description="Source rank (1-based)")
    score: float = Field(ge=0.0, le=1.0, description="Final retrieval score")
    hybrid_score: float = Field(ge=0.0, le=1.0, description="Original hybrid search score")
    title: str
    link: str
    published: str
    authors: str


# =============================================================================
# LLM MODELS
# =============================================================================

class LLMRequest(BaseModel):
    """LLM request with prompt and parameters."""

    prompt: str
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    streaming: bool = Field(default=False)


class LLMResponse(BaseModel):
    """LLM response with metadata."""

    text: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    model: str = ""
    provider: str = ""


# =============================================================================
# EVALUATION MODELS
# =============================================================================

class EvaluationSample(BaseModel):
    """A single evaluation sample."""

    question: str
    ground_truth: Optional[str] = None
    context: Optional[str] = None
    answer: Optional[str] = None
    retrieval_scores: list[float] = Field(default_factory=list)


class EvaluationMetrics(BaseModel):
    """Evaluation metrics from RAGAS."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    answer_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    answer_correctness: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    def get_average(self) -> float:
        """Get average of all defined metrics."""
        values = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
        ]
        if self.answer_similarity is not None:
            values.append(self.answer_similarity)
        if self.answer_correctness is not None:
            values.append(self.answer_correctness)
        return sum(values) / len(values)


class EvaluationResult(BaseModel):
    """Complete evaluation result."""

    config: dict
    ragas_scores: EvaluationMetrics
    num_samples: int = Field(ge=0)
    latency_stats: dict = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# INDEX MODELS
# =============================================================================

class IndexStats(BaseModel):
    """Statistics about the search index."""

    papers: int = Field(ge=0, description="Number of unique papers")
    chunks: int = Field(ge=0, description="Number of chunks")
    dimension: int = Field(ge=0, description="Embedding dimension")
    vectorstore_mode: str = Field(description="Vector store backend (chroma/pgvector)")
    semantic_chunking: bool = Field(default=False, description="Whether semantic chunking was used")


class PaperInfo(BaseModel):
    """Information about a single indexed paper."""

    index: int = Field(ge=1, description="Paper index in the list")
    title: str
    authors: str
    link: str
    published: str
    tags: str
    preview: str = Field(description="Preview of the paper content")
    chunks: int = Field(ge=1, description="Number of chunks for this paper")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def dict_to_retrieval_result(data: dict) -> RetrievalResult:
    """
    Convert a dictionary to RetrievalResult.

    Handles both old dict format and new RetrievalResult format.
    """
    if isinstance(data, RetrievalResult):
        return data

    # Extract meta
    meta_data = data.get("meta", {})
    if not isinstance(meta_data, dict):
        meta_data = {"title": str(meta_data)}

    # Ensure meta has required fields
    if "title" not in meta_data:
        meta_data["title"] = data.get("title", "Unknown")
    if "link" not in meta_data:
        meta_data["link"] = data.get("link", "")
    if "published" not in meta_data:
        meta_data["published"] = data.get("published", "")
    if "authors" not in meta_data:
        meta_data["authors"] = data.get("authors", "")

    meta = ChunkMetadata(**meta_data)

    return RetrievalResult(
        text=data.get("text", ""),
        meta=meta,
        score=data.get("score", 0.0),
        rerank_score=data.get("rerank_score"),
        mmr_score=data.get("mmr_score"),
    )
