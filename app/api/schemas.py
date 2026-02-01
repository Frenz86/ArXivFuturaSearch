"""Pydantic schemas for API request/response models."""


from typing import Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for asking a question about indexed papers."""

    question: str
    top_k: int = Field(default=30, ge=1, le=20)
    filters: Optional[dict] = Field(default=None, description="Metadata filters")
    stream: bool = Field(default=False, description="Enable streaming response")


class BuildRequest(BaseModel):
    """Request model for building the search index."""

    query: str
    max_results: int = Field(default=30, ge=1, le=200)


class AskResponse(BaseModel):
    """Response model for ask endpoint."""

    answer: str
    sources: list[dict]
    retrieval_info: dict


class BuildResponse(BaseModel):
    """Response model for build endpoint."""

    status: str
    stats: dict


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str
    llm_mode: str
    llm_health: dict
    index_loaded: bool
    index_documents: int
    embedder_loaded: bool
    embedder_model: str | None = None
    reranker_enabled: bool
    query_expansion_enabled: bool = True


class EvaluateRequest(BaseModel):
    """Request model for evaluation endpoint."""

    test_dataset_path: Optional[str] = Field(
        default=None,
        description="Optional path to custom test dataset (JSON/CSV)",
    )
    rerank_method: str = Field(
        default="mmr",
        description="Reranking method to use",
    )
    use_cot_prompting: bool = Field(
        default=True,
        description="Use chain-of-thought prompting",
    )


class CompareRequest(BaseModel):
    """Request model for comparative evaluation."""

    test_dataset_path: Optional[str] = Field(default=None)
    compare_rerank_methods: bool = Field(
        default=True,
        description="Compare MMR vs Cross-Encoder vs None",
    )
