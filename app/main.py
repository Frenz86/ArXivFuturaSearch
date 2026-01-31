"""FastAPI server for ArXiv Futura Search v0.3.0 with LangChain, ChromaDB, caching, and metrics."""


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

import os
import json
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional, Union, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.arxiv_loader import fetch_arxiv_async, save_raw
from app.chunking import build_chunks
from app.embeddings import get_embedder, get_reranker, maximal_marginal_relevance
from app.rag import (
    build_prompt,
    llm_generate_async,
    llm_generate_stream,
    check_llm_health,
    create_rag_chain,
)
from app.cache import get_cache
from app.metrics import (
    get_metrics,
    get_metrics_summary,
    requests_total,
    request_latency,
    retrieval_latency,
    llm_latency,
    record_retrieval,
    record_llm_request,
    update_index_stats,
    record_query_expansion,
    record_reranking,
)
from app.logging_config import get_logger

# Import vector store factory (supports ChromaDB and Pgvector)
from app.vectorstore import get_vectorstore, VectorStoreInterface
from app import dependencies as deps
from app.error_handling import LLMError, classify_error, log_error_context, ErrorSeverity
from app.middleware import setup_middleware, setup_cors_middleware

logger = get_logger(__name__)

# Global store instance (loaded on startup)
_store: Optional[VectorStoreInterface] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models, cache, and index on startup."""
    global _store

    logger.info(
        "Starting ArXiv Futura Search",
        version=settings.VERSION,
        mode="LangChain + ChromaDB"
    )

    # Download NLTK data asynchronously
    from app.chunking import ensure_nltk_data_async
    await ensure_nltk_data_async()

    # Initialize cache
    cache = get_cache()
    if cache.enabled:
        logger.info("Cache initialized", stats=cache.get_stats())
    else:
        logger.warning("Cache disabled or unavailable")

    # Pre-load embedding model (singleton)
    logger.info("Pre-loading embedding model...", model=settings.EMBED_MODEL)
    get_embedder()

    # Pre-load reranker if enabled
    if settings.RERANK_ENABLED:
        logger.info("Pre-loading reranker model...", model=settings.RERANK_MODEL)
        get_reranker()

    # Initialize OpenTelemetry if enabled
    if settings.ENVIRONMENT != "test":
        try:
            from app.tracing import init_telemetry, OpenTelemetryConfig

            otel_config = OpenTelemetryConfig(
                service_name="arxiv_futura_search",
                service_version=settings.VERSION,
                enabled=True,
                trace_exporter="console",  # Use console for dev, "otlp" for production
                metrics_exporter="console",
            )

            await init_telemetry(app, otel_config)
            logger.info("OpenTelemetry initialized", export="console")
        except ImportError:
            logger.info("OpenTelemetry packages not installed")
        except Exception as e:
            logger.warning("OpenTelemetry initialization failed", error=str(e))

    # Initialize vector store (ChromaDB or Pgvector based on config)
    logger.info("Initializing vector store...", mode=settings.VECTORSTORE_MODE)
    try:
        _store = get_vectorstore(collection_name="arxiv_papers")
        doc_count = _store.count()
        logger.info("Vector store loaded", documents=doc_count, mode=settings.VECTORSTORE_MODE)
        update_index_stats(documents=doc_count, chunks=doc_count)

        # Set store instance for dependency injection
        deps.set_store_instance(_store)
    except Exception as e:
        logger.warning("Vector store initialization failed", error=str(e))

    yield

    logger.info("Shutting down ArXiv Futura Search")

    # Shutdown OpenTelemetry
    try:
        from app.tracing import shutdown_telemetry
        shutdown_telemetry()
        logger.info("OpenTelemetry shutdown complete")
    except Exception:
        pass


app = FastAPI(
    title="ArXiv Futura Search",
    description="Hybrid RAG pipeline for ML research papers with OpenRouter integration",
    version=settings.VERSION,
    lifespan=lifespan,
)

# Setup middleware (CORS, rate limiting, correlation ID, security)
setup_cors_middleware(app)
setup_middleware(app)

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")


# Request/Response Models
class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=settings.TOP_K, ge=1, le=20)
    filters: Optional[dict] = Field(default=None, description="Metadata filters")
    stream: bool = Field(default=False, description="Enable streaming response")


class BuildRequest(BaseModel):
    query: str
    max_results: int = Field(default=30, ge=1, le=200)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    retrieval_info: dict


class BuildResponse(BaseModel):
    status: str
    stats: dict


class HealthResponse(BaseModel):
    status: str
    llm_mode: str
    llm_health: dict
    index_loaded: bool
    index_documents: int
    embedder_loaded: bool
    embedder_model: str | None = None
    reranker_enabled: bool
    query_expansion_enabled: bool = True


# Utility functions
def ensure_dirs() -> None:
    """Create necessary directories if they don't exist."""
    os.makedirs(settings.RAW_DIR, exist_ok=True)
    os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
    # Only create ChromaDB directory if using ChromaDB
    if settings.VECTORSTORE_MODE == "chroma":
        os.makedirs(settings.CHROMA_DIR, exist_ok=True)


async def build_index_async(query: str, max_results: int = 30) -> dict:
    """
    Build the hybrid index from arXiv papers.

    Args:
        query: arXiv search query
        max_results: Maximum number of papers to fetch

    Returns:
        Statistics about the build process
    """
    global _store
    ensure_dirs()

    # Fetch papers from arXiv
    logger.info("Fetching papers from arXiv", query=query, max_results=max_results)
    papers = await fetch_arxiv_async(query, max_results=max_results)
    save_raw(papers, os.path.join(settings.RAW_DIR, "arxiv_papers.json"))

    # Get embedder for chunking and embedding
    embedder = get_embedder()

    # Chunk the papers (with optional semantic chunking)
    use_semantic = settings.USE_SEMANTIC_CHUNKING
    logger.info(
        "Chunking papers",
        semantic=use_semantic,
        chunk_size=settings.CHUNK_SIZE,
    )
    chunks = build_chunks(
        papers,
        settings.CHUNK_SIZE,
        settings.CHUNK_OVERLAP,
        sentence_aware=True,
        semantic_chunking=use_semantic,
        embedder=embedder if use_semantic else None,
    )
    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    # Convert ChunkMetadata Pydantic models to dicts for vector store
    metas = [c.meta.model_dump() if hasattr(c.meta, 'model_dump') else c.meta for c in chunks]

    # Generate embeddings
    logger.info("Generating embeddings", count=len(texts))
    vectors = embedder.embed(texts, show_progress=True)

    # Build vector store index (ChromaDB or Pgvector)
    if _store is None:
        _store = get_vectorstore(collection_name="arxiv_papers")
    else:
        # Reset existing collection
        _store.reset()

    _store.add(vectors, chunk_ids, texts, metas)

    # Update metrics
    update_index_stats(documents=len(papers), chunks=len(chunks))

    # Clear cache after rebuild
    cache = get_cache()
    if cache.enabled:
        cleared = cache.clear_pattern("arxiv_rag:*")
        logger.info("Cleared cache after index rebuild", keys_deleted=cleared)

    logger.info("Index built successfully", papers=len(papers), chunks=len(chunks))

    return {
        "papers": len(papers),
        "chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "vectorstore_mode": settings.VECTORSTORE_MODE,
        "semantic_chunking": use_semantic,
    }


def get_store() -> VectorStoreInterface:
    """Get the loaded store or raise error."""
    return deps.get_store()


# API Endpoints
def count_unique_documents() -> int:
    """Count unique documents by title from the vector store."""
    if _store is None:
        return 0
    try:
        collection = _store.client.get_collection(_store.collection_name)
        results = collection.get(include=["metadatas"])
        titles = set()
        for meta in results.get("metadatas", []):
            title = meta.get("title")
            if title:
                titles.add(title)
        return len(titles)
    except Exception:
        return 0


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    embedder = get_embedder()
    llm_health = await check_llm_health()

    return HealthResponse(
        status="ok" if llm_health.get("healthy", False) else "degraded",
        llm_mode=settings.LLM_MODE,
        llm_health=llm_health,
        index_loaded=_store is not None,
        index_documents=count_unique_documents(),
        embedder_loaded=embedder is not None,
        embedder_model=settings.EMBED_MODEL if embedder else None,
        reranker_enabled=settings.RERANK_ENABLED,
        query_expansion_enabled=settings.QUERY_EXPANSION_ENABLED,
    )


@app.get("/", response_class=HTMLResponse)
async def root_web(request: Request):
    """Serve the web interface."""
    health_data = await health_check()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "health": health_data,
        },
    )


@app.get("/api", response_class=HTMLResponse)
async def api_info():
    """Root API endpoint with basic info."""
    return {
        "name": "ArXiv Futura Search",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# Web Interface Routes
@app.get("/web/search", response_class=HTMLResponse)
async def web_search(request: Request):
    """Search page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/web/build", response_class=HTMLResponse)
async def web_build(request: Request):
    """Build index page."""
    health_data = await health_check()
    return templates.TemplateResponse(
        "build.html",
        {
            "request": request,
            "health": health_data,
        },
    )


@app.get("/web/index", response_class=HTMLResponse)
async def web_index_view(request: Request):
    """View indexed papers page."""
    return templates.TemplateResponse("index_view.html", {"request": request})


@app.get("/web/config", response_class=HTMLResponse)
async def web_config_view(request: Request):
    """View configuration page."""
    return templates.TemplateResponse("config_view.html", {"request": request})


@app.post("/build", response_model=BuildResponse)
async def build(req: BuildRequest):
    """
    Build or rebuild the search index from arXiv papers.

    Example query: 'cat:cs.LG AND (rag OR retrieval OR agentic)'
    """
    try:
        logger.info("Building index", query=req.query, max_results=req.max_results)
        stats = await build_index_async(req.query, req.max_results)
        return BuildResponse(status="ok", stats=stats)
    except Exception as e:
        logger.error("Build failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(req: AskRequest):
    """
    Ask a question about the indexed papers.

    Supports both regular and streaming responses with caching and metrics.
    """
    import time

    start_time = time.time()
    requests_total.labels(endpoint="/ask", method="POST").inc()

    try:
        store = get_store()
        embedder = get_embedder()
        reranker = get_reranker() if settings.RERANK_ENABLED else None
        cache = get_cache()

        # Try cache first (for non-streaming requests)
        cache_key = None
        if cache.enabled and not req.stream:
            # Use json.dumps with sort_keys=True for deterministic cache keys
            filters_hash = json.dumps(req.filters, sort_keys=True) if req.filters else "no_filters"
            cache_key = cache._make_key(
                "query",
                req.question,
                req.top_k,
                filters_hash,
            )
            cached_response = cache.get(cache_key)
            if cached_response:
                logger.info("Cache hit", question=req.question[:50])
                return AskResponse(**cached_response)

        # Embed the question
        retrieval_start = time.time()
        q_vec = embedder.embed_query(req.question)

        # Choose search method: ensemble (with query expansion) or standard hybrid
        retrieval_k = settings.RETRIEVAL_K if (reranker or settings.RERANK_USE_MMR) else req.top_k

        if settings.QUERY_EXPANSION_ENABLED:
            # Use ensemble search with query expansion and RRF
            logger.debug("Using ensemble search with query expansion")
            original_len = len(req.question)
            retrieved = store.search_ensemble(
                query_text=req.question,
                top_k=retrieval_k,
                query_expansion=True,
                filters=req.filters,
            )
            # Record query expansion metrics
            # Estimate terms added by checking if expansion happened
            terms_added = max(0, len(retrieved) - req.top_k) if retrieved else 0
            record_query_expansion(settings.QUERY_EXPANSION_METHOD, terms_added)
        else:
            # Standard hybrid search
            retrieved = store.search(
                q_vec,
                query_text=req.question,
                top_k=retrieval_k,
                semantic_weight=settings.SEMANTIC_WEIGHT,
                bm25_weight=settings.BM25_WEIGHT,
                filters=req.filters,
            )

        retrieval_time = time.time() - retrieval_start
        retrieval_latency.labels(method="hybrid").observe(retrieval_time)

        # Record retrieval metrics and check quality
        scores = [r["score"] for r in retrieved]
        record_retrieval("hybrid", len(retrieved), scores)

        # Warn if retrieval scores are low (poor matches)
        if scores and scores[0] < 0.3:
            logger.warning(
                "Low retrieval quality - index may not contain relevant papers",
                top_score=f"{scores[0]:.3f}",
                query=req.question[:50]
            )

        # Rerank if enabled
        rerank_method = None
        if settings.RERANK_USE_MMR and len(retrieved) > req.top_k:
            # MMR reranking for diversity
            logger.info("Applying MMR reranking", candidates=len(retrieved))
            rerank_start = time.time()

            # Get embeddings for retrieved docs
            doc_texts = [r["text"] for r in retrieved]
            doc_embeddings = embedder.embed(doc_texts, show_progress=False)

            retrieved = maximal_marginal_relevance(
                q_vec,
                doc_embeddings,
                retrieved,
                top_k=req.top_k,
                lambda_param=settings.MMR_LAMBDA,
            )

            rerank_time = time.time() - rerank_start
            retrieval_latency.labels(method="mmr").observe(rerank_time)

            # Record reranking metrics
            record_reranking(
                method="mmr",
                candidates=len(retrieved) + req.top_k,  # Approximate original candidates
                latency_seconds=rerank_time,
                diversity_score=settings.MMR_LAMBDA,  # Lambda as proxy for diversity
            )

            rerank_method = "MMR"

        elif reranker and retrieved:
            # Cross-encoder reranking
            logger.info("Applying cross-encoder reranking", candidates=len(retrieved))
            rerank_start = time.time()

            retrieved = reranker.rerank(req.question, retrieved, top_k=req.top_k)

            rerank_time = time.time() - rerank_start
            retrieval_latency.labels(method="cross_encoder").observe(rerank_time)

            # Record reranking metrics
            record_reranking(
                method="cross_encoder",
                candidates=len(retrieved),
                latency_seconds=rerank_time,
            )

            rerank_method = "Cross-Encoder"

        # Build prompt with CoT
        prompt = build_prompt(req.question, retrieved, use_cot=True)

        # Format sources for response
        sources = [
            {
                "rank": i + 1,
                "score": r.get("rerank_score", r.get("mmr_score", r["score"])),
                "hybrid_score": r["score"],
                "title": r["meta"].get("title"),
                "link": r["meta"].get("link"),
                "published": r["meta"].get("published"),
                "authors": r["meta"].get("authors", ""),
            }
            for i, r in enumerate(retrieved)
        ]

        retrieval_info = {
            "candidates_retrieved": retrieval_k,
            "rerank_method": rerank_method,
            "final_count": len(retrieved),
            "filters_applied": req.filters is not None,
            "retrieval_time_ms": int(retrieval_time * 1000),
        }

        # Streaming response
        if req.stream:
            async def generate():
                import json
                logger.info("Starting stream response", sources_count=len(sources))

                # Send sources
                payload = {"event": "sources", "data": json.dumps(sources)}
                yield {"data": json.dumps(payload)}

                payload = {"event": "retrieval_info", "data": json.dumps(retrieval_info)}
                yield {"data": json.dumps(payload)}

                # Stream answer tokens
                llm_start = time.time()
                token_count = 0

                try:
                    async for chunk in llm_generate_stream(prompt):
                        if chunk:
                            token_count += 1
                            payload = {"event": "token", "data": chunk}
                            yield {"data": json.dumps(payload)}

                    llm_time = time.time() - llm_start
                    llm_latency.labels(
                        provider=settings.LLM_MODE,
                        model=settings.OPENROUTER_MODEL,
                    ).observe(llm_time)

                    record_llm_request(
                        provider=settings.LLM_MODE,
                        model=settings.OPENROUTER_MODEL,
                        success=True,
                        completion_tokens=token_count,
                    )

                    logger.info("Stream completed", total_tokens=token_count)
                except Exception as e:
                    record_llm_request(
                        provider=settings.LLM_MODE,
                        model=settings.OPENROUTER_MODEL,
                        success=False,
                    )
                    raise

                payload = {"event": "done", "data": ""}
                yield {"data": json.dumps(payload)}

            return EventSourceResponse(generate())

        # Regular response
        llm_start = time.time()
        answer = await llm_generate_async(prompt)
        llm_time = time.time() - llm_start

        llm_latency.labels(
            provider=settings.LLM_MODE,
            model=settings.OPENROUTER_MODEL,
        ).observe(llm_time)

        record_llm_request(
            provider=settings.LLM_MODE,
            model=settings.OPENROUTER_MODEL,
            success=True,
        )

        response_data = {
            "answer": answer,
            "sources": sources,
            "retrieval_info": retrieval_info,
        }

        # Cache the response
        if cache.enabled and cache_key:
            cache.set(cache_key, response_data, ttl=settings.CACHE_TTL)

        # Record total latency
        total_time = time.time() - start_time
        request_latency.labels(endpoint="/ask").observe(total_time)

        return AskResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        record_llm_request(
            provider=settings.LLM_MODE,
            model=settings.OPENROUTER_MODEL,
            success=False,
        )

        # Classify error for better handling
        severity, error_type = classify_error(e)
        log_error_context(
            e,
            context={
                "question": req.question[:100],
                "endpoint": "/ask",
                "llm_mode": settings.LLM_MODE,
            },
            severity=severity,
        )

        # Provide more helpful error messages
        detail = str(e)
        if error_type == "authentication":
            detail = "LLM authentication failed. Please check your API key configuration."
        elif error_type == "rate_limit":
            detail = "LLM rate limit exceeded. Please try again later."
        elif error_type == "connection":
            detail = "LLM connection failed. Please check your network or API status."

        raise HTTPException(status_code=500, detail=detail)


@app.get("/config")
def get_config():
    """Get current configuration (without sensitive data)."""
    embedder = get_embedder()
    return {
        "version": settings.VERSION,
        "vectorstore_mode": settings.VECTORSTORE_MODE,
        "environment": settings.ENVIRONMENT,
        "llm_mode": settings.LLM_MODE,
        "openrouter_model": settings.OPENROUTER_MODEL,
        "embed_model": settings.EMBED_MODEL,
        "embed_dimension": embedder.dimension if embedder else None,
        "is_e5_model": embedder.is_e5_model if embedder else False,
        "top_k": settings.TOP_K,
        "retrieval_k": settings.RETRIEVAL_K,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "semantic_chunking": settings.USE_SEMANTIC_CHUNKING,
        "semantic_weight": settings.SEMANTIC_WEIGHT,
        "bm25_weight": settings.BM25_WEIGHT,
        "rerank_enabled": settings.RERANK_ENABLED,
        "rerank_model": settings.RERANK_MODEL if settings.RERANK_ENABLED else None,
        "rerank_use_mmr": settings.RERANK_USE_MMR,
        "mmr_lambda": settings.MMR_LAMBDA,
        "query_expansion_enabled": settings.QUERY_EXPANSION_ENABLED,
        "query_expansion_method": settings.QUERY_EXPANSION_METHOD,
        "cache_enabled": settings.CACHE_ENABLED,
        "metrics_enabled": settings.METRICS_ENABLED,
        "has_openrouter_key": bool(settings.OPENROUTER_API_KEY),
    }


@app.get("/metrics")
def metrics_endpoint():
    """Prometheus metrics endpoint."""
    if not settings.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics disabled")

    metrics_data = get_metrics()
    return Response(content=metrics_data, media_type="text/plain")


@app.get("/metrics/summary")
def metrics_summary():
    """Get human-readable metrics summary."""
    if not settings.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics disabled")

    return get_metrics_summary()


@app.get("/cache/stats")
def cache_stats():
    """Get cache statistics."""
    cache = get_cache()
    return cache.get_stats()


@app.post("/cache/clear")
def cache_clear(pattern: str = "arxiv_rag:*"):
    """Clear cache entries matching pattern."""
    cache = get_cache()
    if not cache.enabled:
        raise HTTPException(status_code=503, detail="Cache not available")

    deleted = cache.clear_pattern(pattern)
    return {"status": "ok", "deleted": deleted, "pattern": pattern}


@app.get("/search")
async def search_only(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20),
    published_after: Optional[str] = Query(default=None, description="Filter: YYYY-MM-DD"),
    published_before: Optional[str] = Query(default=None, description="Filter: YYYY-MM-DD"),
):
    """
    Search without LLM generation (useful for browsing).
    """
    store = get_store()
    embedder = get_embedder()

    q_vec = embedder.embed_query(q)

    filters = {}
    if published_after:
        filters["published_after"] = published_after
    if published_before:
        filters["published_before"] = published_before

    retrieved = store.search(
        q_vec,
        query_text=q,
        top_k=top_k,
        semantic_weight=settings.SEMANTIC_WEIGHT,
        bm25_weight=settings.BM25_WEIGHT,
        filters=filters if filters else None,
    )

    return {
        "query": q,
        "results": [
            {
                "rank": i + 1,
                "score": r["score"],
                "title": r["meta"].get("title"),
                "link": r["meta"].get("link"),
                "published": r["meta"].get("published"),
                "authors": r["meta"].get("authors", ""),
                "excerpt": r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"],
            }
            for i, r in enumerate(retrieved)
        ],
    }


# =============================================================================
# ADVANCED SEARCH ENDPOINTS
# =============================================================================

@app.post("/search/hybrid")
async def hybrid_search(req: AskRequest):
    """
    Advanced hybrid search combining BM25 and vector search with RRF.

    Provides improved retrieval by combining dense and sparse representations.
    """
    try:
        from app.hybrid_search import get_hybrid_search_engine

        store = get_store()
        embedder = get_embedder()

        # Get or create hybrid search engine
        engine = get_hybrid_search_engine(store.vectorstore)

        # Index documents if not already indexed
        if not engine._indexed:
            try:
                # Get all documents from store
                collection = store.client.get_collection(store.collection_name)
                results = collection.get(include=["documents"])
                texts = results.get("documents", [])
                await engine.index_documents(texts[:1000])  # Limit for initial indexing
            except Exception as e:
                logger.warning("BM25 indexing failed, using vector search only", error=str(e))

        # Perform hybrid search
        results = await engine.search(
            query=req.question,
            top_k=req.top_k,
            search_kwargs={"k": req.top_k * 2} if req.filters else None,
        )

        return {
            "query": req.question,
            "method": "hybrid_bm25_vector",
            "results": results[:req.top_k],
        }

    except ImportError:
        raise HTTPException(status_code=501, detail="Hybrid search not available")
    except Exception as e:
        logger.error("Hybrid search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/multi-query")
async def multi_query_search(req: AskRequest):
    """
    Multi-query retrieval with LLM query expansion.

    Generates multiple query variants using LLM and merges results using RRF.
    """
    try:
        from app.multi_query import get_multi_query_retriever
        from app.rag import get_chat_model

        store = get_store()
        embedder = get_embedder()
        llm = get_chat_model()

        # Get multi-query retriever
        retriever = get_multi_query_retriever(llm)

        # Generate query variants
        queries = await retriever.generate_queries(req.question, include_original=True)

        # Define retrieval function
        async def retrieve_func(query: str, k: int):
            q_vec = embedder.embed_query(query)
            return store.search(
                q_vec,
                query_text=query,
                top_k=k,
                semantic_weight=settings.SEMANTIC_WEIGHT,
                bm25_weight=settings.BM25_WEIGHT,
            )

        # Retrieve and merge results
        results = await retriever.retrieve(
            queries=queries,
            retriever_func=retrieve_func,
            top_k=req.top_k,
        )

        return {
            "original_query": req.question,
            "expanded_queries": queries,
            "method": "multi_query_rrf",
            "results": results,
        }

    except ImportError:
        raise HTTPException(status_code=501, detail="Multi-query search not available")
    except Exception as e:
        logger.error("Multi-query search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/rerank")
async def reranked_search(req: AskRequest):
    """
    Search with cross-encoder re-ranking for improved precision.

    Uses a cross-encoder model to re-rank search results based on
    query-document relevance.
    """
    try:
        from app.reranking import get_cross_encoder_reranker

        store = get_store()
        embedder = get_embedder()

        # Initial retrieval
        q_vec = embedder.embed_query(req.question)
        candidates = store.search(
            q_vec,
            query_text=req.question,
            top_k=req.top_k * 3,  # Get more candidates for re-ranking
            semantic_weight=settings.SEMANTIC_WEIGHT,
            bm25_weight=settings.BM25_WEIGHT,
        )

        # Convert to reranker format
        documents = [
            {
                "text": r["text"],
                "title": r["meta"].get("title", ""),
                "link": r["meta"].get("link", ""),
            }
            for r in candidates
        ]

        # Re-rank with cross-encoder
        reranker = get_cross_encoder_reranker()
        reranked = await reranker.rerank(req.question, documents, top_k=req.top_k)

        return {
            "query": req.question,
            "method": "cross_encoder_rerank",
            "candidates_retrieved": len(candidates),
            "results": reranked,
        }

    except ImportError:
        raise HTTPException(status_code=501, detail="Cross-encoder reranking not available")
    except Exception as e:
        logger.error("Reranked search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suggest")
async def autocomplete_suggestions(
    q: str = Query(..., description="Query prefix"),
    limit: int = Query(default=10, ge=1, le=20),
):
    """
    Get autocomplete suggestions for a query prefix.

    Returns relevant query suggestions based on indexed content and query history.
    """
    try:
        from app.autocomplete import get_autocompleter, get_trending_queries

        autocompleter = get_autocompleter()

        # Index documents if not already indexed
        if not autocompleter._vocabulary:
            try:
                store = get_store()
                collection = store.client.get_collection(store.collection_name)
                results = collection.get(include=["documents"])
                documents = [{"text": doc} for doc in results.get("documents", [])[:1000]]
                autocompleter.index_documents(documents)
            except Exception as e:
                logger.warning("Autocomplete indexing failed", error=str(e))

        # Get suggestions
        suggestions = autocompleter.get_suggestions(q, max_results=limit)

        # Add trending queries if prefix is empty
        if not q.strip() and limit > len(suggestions):
            trending = get_trending_queries()
            trending_suggestions = trending.get_trending(limit=limit - len(suggestions))
            suggestions.extend(trending_suggestions)

        return {
            "prefix": q,
            "suggestions": suggestions,
        }

    except ImportError:
        raise HTTPException(status_code=501, detail="Autocomplete not available")
    except Exception as e:
        logger.error("Autocomplete failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CACHE AND TELEMETRY ENDPOINTS
# =============================================================================

@app.get("/cache/redis/status")
async def redis_cache_status():
    """Get Redis cache status and statistics."""
    try:
        from app.redis_cache import get_redis_cache

        cache = get_redis_cache()
        info = await cache.get_info()

        return {
            "enabled": cache._client is not None,
            "prefix": cache.prefix,
            "serializer": "pickle" if isinstance(cache.serializer, type(cache).PickleSerializer) else "json",
            "info": info,
        }

    except ImportError:
        raise HTTPException(status_code=501, detail="Redis cache not available")
    except Exception as e:
        logger.error("Redis status check failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/background-tasks/status")
async def background_tasks_status():
    """Get status of background tasks (ArXiv feed updates)."""
    try:
        from app.background_tasks import get_task_manager

        manager = get_task_manager()
        status = manager.get_status()

        return {
            "tasks": status,
            "running": manager._running,
        }

    except ImportError:
        raise HTTPException(status_code=501, detail="Background tasks not available")
    except Exception as e:
        logger.error("Background tasks status failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/background-tasks/start")
async def start_background_tasks():
    """Start all background tasks."""
    try:
        from app.background_tasks import get_task_manager

        manager = get_task_manager()
        await manager.start_all()

        return {"status": "ok", "message": "Background tasks started"}

    except ImportError:
        raise HTTPException(status_code=501, detail="Background tasks not available")
    except Exception as e:
        logger.error("Failed to start background tasks", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/background-tasks/stop")
async def stop_background_tasks():
    """Stop all background tasks."""
    try:
        from app.background_tasks import get_task_manager

        manager = get_task_manager()
        await manager.stop_all()

        return {"status": "ok", "message": "Background tasks stopped"}

    except ImportError:
        raise HTTPException(status_code=501, detail="Background tasks not available")
    except Exception as e:
        logger.error("Failed to stop background tasks", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tracing/status")
async def tracing_status():
    """Get OpenTelemetry tracing status."""
    try:
        from app.tracing import _tracer_provider, _meter_provider

        return {
            "tracing_enabled": _tracer_provider is not None,
            "metrics_enabled": _meter_provider is not None,
            "otel_exporter": settings.OTEL_EXPORTER_OTLP_ENDPOINT,
        }

    except ImportError:
        raise HTTPException(status_code=501, detail="OpenTelemetry not available")
    except Exception as e:
        logger.error("Tracing status failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# INDEX INSPECTION ENDPOINTS
# =============================================================================

@app.get("/index/papers")
def list_index_papers(limit: int = Query(default=50, ge=1, le=500)):
    """
    List all unique papers currently in the index (grouped by title).

    Useful for debugging and verifying what papers were indexed.
    """
    store = get_store()
    try:
        collection = store.client.get_collection(store.collection_name)
        results = collection.get(
            include=["documents", "metadatas"]
        )

        # Group chunks by paper title to show unique papers
        papers_by_title = {}
        for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            title = meta.get("title", "Unknown")
            if title not in papers_by_title:
                papers_by_title[title] = {
                    "title": title,
                    "authors": meta.get("authors", ""),
                    "link": meta.get("link", ""),
                    "published": meta.get("published", ""),
                    "tags": meta.get("tags", ""),
                    "preview": doc[:300] + "..." if len(doc) > 300 else doc,
                    "chunks": 0,
                }
            papers_by_title[title]["chunks"] += 1

        # Convert to list and limit
        papers_list = list(papers_by_title.values())[:limit]

        # Add index
        for i, paper in enumerate(papers_list):
            paper["index"] = i + 1

        return {
            "total": len(papers_by_title),  # Unique papers
            "chunks": store.count(),  # Total chunks
            "shown": len(papers_list),
            "papers": papers_list,
        }
    except Exception as e:
        logger.error("Failed to list papers", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# EVALUATION ENDPOINTS
# =============================================================================

class EvaluateRequest(BaseModel):
    """Request model for evaluation endpoint."""
    test_dataset_path: Optional[str] = Field(
        default=None,
        description="Optional path to custom test dataset (JSON/CSV)",
    )
    rerank_method: Literal["mmr", "cross_encoder", "none"] = Field(
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


@app.post("/evaluate")
async def run_evaluation_endpoint(req: EvaluateRequest):
    """
    Run RAGAS evaluation on the RAG system.

    This endpoint evaluates the system using RAGAS metrics:
    - Faithfulness: Answer consistency with sources
    - Answer Relevancy: How well the answer addresses the question
    - Context Precision: Retrieval quality
    - Context Recall: Coverage of relevant information
    - Answer Similarity: Semantic similarity to ground truth
    - Answer Correctness: Factual correctness

    Returns detailed results including performance metrics.
    """
    try:
        from app.evals import run_evaluation_async, EvalConfig

        config = EvalConfig(
            rerank_method=req.rerank_method,
            use_semantic_chunking=settings.USE_SEMANTIC_CHUNKING,
            use_cot_prompting=req.use_cot_prompting,
            top_k=settings.TOP_K,
            retrieval_k=settings.RETRIEVAL_K,
            mmr_lambda=settings.MMR_LAMBDA,
        )

        logger.info("Starting evaluation via API", config=req.dict())
        results = await run_evaluation_async(config, req.test_dataset_path)

        return {
            "status": "success",
            "results": results,
            "message": "Evaluation completed successfully",
        }

    except Exception as e:
        logger.error("Evaluation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/compare")
async def run_comparative_evaluation_endpoint(req: CompareRequest):
    """
    Run comparative A/B evaluation with different configurations.

    Compares different reranking methods (MMR, Cross-Encoder, None)
    to help you choose the best configuration for your use case.

    Returns a detailed comparison table with RAGAS and performance metrics.
    """
    try:
        from app.evals import run_comparative_evaluation, EvalConfig

        if req.compare_rerank_methods:
            # Compare reranking methods
            configs = [
                EvalConfig(rerank_method="mmr", use_cot_prompting=True),
                EvalConfig(rerank_method="cross_encoder", use_cot_prompting=True),
                EvalConfig(rerank_method="none", use_cot_prompting=True),
            ]
        else:
            # Default: just current config
            configs = [
                EvalConfig(
                    rerank_method="mmr" if settings.RERANK_USE_MMR else "cross_encoder",
                    use_semantic_chunking=settings.USE_SEMANTIC_CHUNKING,
                    use_cot_prompting=True,
                )
            ]

        logger.info("Starting comparative evaluation via API", num_configs=len(configs))
        results = await run_comparative_evaluation(configs, req.test_dataset_path)

        return {
            "status": "success",
            "comparison": results,
            "message": f"Comparative evaluation completed ({len(configs)} configurations)",
        }

    except Exception as e:
        logger.error("Comparative evaluation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluate/results")
def get_latest_evaluation_results():
    """Get the most recent evaluation results."""
    try:
        import glob

        # Find latest summary file
        pattern = os.path.join(settings.PROCESSED_DIR, "eval_summary_*.json")
        files = glob.glob(pattern)

        if not files:
            raise HTTPException(
                status_code=404,
                detail="No evaluation results found. Run /evaluate first.",
            )

        latest_file = max(files, key=os.path.getctime)

        with open(latest_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        return {
            "status": "success",
            "results": results,
            "file": latest_file,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to load evaluation results", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluate/history")
def get_evaluation_history(limit: int = Query(default=10, ge=1, le=50)):
    """Get history of evaluation runs."""
    try:
        import glob

        # Find all summary files
        pattern = os.path.join(settings.PROCESSED_DIR, "eval_summary_*.json")
        files = sorted(glob.glob(pattern), key=os.path.getctime, reverse=True)[:limit]

        history = []
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                history.append({
                    "timestamp": data.get("timestamp"),
                    "config": data.get("config"),
                    "ragas_scores": data.get("ragas_scores"),
                    "num_samples": data.get("num_samples"),
                    "file": file_path,
                })

        return {
            "status": "success",
            "count": len(history),
            "history": history,
        }

    except Exception as e:
        logger.error("Failed to load evaluation history", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

