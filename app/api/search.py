"""Core search and RAG endpoints."""

import json
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.api.schemas import AskRequest, AskResponse
from app import dependencies as deps
from app.embeddings import get_embedder, get_reranker, maximal_marginal_relevance
from app.rag import build_prompt, llm_generate_async, llm_generate_stream
from app.cache import get_cache
from app.metrics import (
    requests_total,
    request_latency,
    retrieval_latency,
    llm_latency,
    record_retrieval,
    record_llm_request,
    record_query_expansion,
    record_reranking,
)
from app.logging_config import get_logger
from app.error_handling import classify_error, log_error_context, ErrorSeverity

router = APIRouter()
logger = get_logger(__name__)


def get_store():
    """Get the loaded store or raise error."""
    return deps.get_store()


@router.post("/ask")
async def ask(req: AskRequest):
    """
    Ask a question about the indexed papers.

    Supports both regular and streaming responses with caching and metrics.
    """
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

        # Choose search method
        retrieval_k = settings.RETRIEVAL_K if (reranker or settings.RERANK_USE_MMR) else req.top_k

        if settings.QUERY_EXPANSION_ENABLED:
            logger.debug("Using ensemble search with query expansion")
            retrieved = store.search_ensemble(
                query_text=req.question,
                top_k=retrieval_k,
                query_expansion=True,
                filters=req.filters,
            )
            terms_added = max(0, len(retrieved) - req.top_k) if retrieved else 0
            record_query_expansion(settings.QUERY_EXPANSION_METHOD, terms_added)
        else:
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

        # Record retrieval metrics
        scores = [r["score"] for r in retrieved]
        record_retrieval("hybrid", len(retrieved), scores)

        # Warn if retrieval scores are low
        if scores and scores[0] < 0.3:
            logger.warning(
                "Low retrieval quality - index may not contain relevant papers",
                top_score=f"{scores[0]:.3f}",
                query=req.question[:50]
            )

        # Rerank if enabled
        rerank_method = None
        if settings.RERANK_USE_MMR and len(retrieved) > req.top_k:
            logger.info("Applying MMR reranking", candidates=len(retrieved))
            rerank_start = time.time()

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

            record_reranking(
                method="mmr",
                candidates=len(retrieved) + req.top_k,
                latency_seconds=rerank_time,
                diversity_score=settings.MMR_LAMBDA,
            )

            rerank_method = "MMR"

        elif reranker and retrieved:
            logger.info("Applying cross-encoder reranking", candidates=len(retrieved))
            rerank_start = time.time()

            retrieved = reranker.rerank(req.question, retrieved, top_k=req.top_k)

            rerank_time = time.time() - rerank_start
            retrieval_latency.labels(method="cross_encoder").observe(rerank_time)

            record_reranking(
                method="cross_encoder",
                candidates=len(retrieved),
                latency_seconds=rerank_time,
            )

            rerank_method = "Cross-Encoder"

        # Build prompt with CoT
        prompt = build_prompt(req.question, retrieved, use_cot=True)

        # Format sources
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
                logger.info("Starting stream response", sources_count=len(sources))

                payload = {"event": "sources", "data": json.dumps(sources)}
                yield {"data": json.dumps(payload)}

                payload = {"event": "retrieval_info", "data": json.dumps(retrieval_info)}
                yield {"data": json.dumps(payload)}

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
                except Exception:
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

        detail = str(e)
        if error_type == "authentication":
            detail = "LLM authentication failed. Please check your API key configuration."
        elif error_type == "rate_limit":
            detail = "LLM rate limit exceeded. Please try again later."
        elif error_type == "connection":
            detail = "LLM connection failed. Please check your network or API status."

        raise HTTPException(status_code=500, detail=detail)


@router.get("/search")
async def search_only(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20),
    published_after: Optional[str] = Query(default=None, description="Filter: YYYY-MM-DD"),
    published_before: Optional[str] = Query(default=None, description="Filter: YYYY-MM-DD"),
):
    """Search without LLM generation (useful for browsing)."""
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
