"""Monitoring endpoints for metrics, cache, tracing, and background tasks."""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse

from app.config import settings
from app.api.schemas import BuildRequest, BuildResponse
from app import dependencies as deps
from app.services import build_index_async
from app.embeddings import get_embedder
from app.rag import check_llm_health
from app.metrics import get_metrics, get_metrics_summary
from app.cache import get_cache
from app.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


def get_store():
    """Get the loaded store or raise error."""
    return deps.get_store()


@router.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    from app.api.schemas import HealthResponse
    from app.services import count_unique_documents

    embedder = get_embedder()
    llm_health = await check_llm_health()

    return HealthResponse(
        status="ok" if llm_health.get("healthy", False) else "degraded",
        llm_mode=settings.LLM_MODE,
        llm_health=llm_health,
        index_loaded=deps.get_store() is not None,
        index_documents=count_unique_documents(deps.get_store()),
        embedder_loaded=embedder is not None,
        embedder_model=settings.EMBED_MODEL if embedder else None,
        reranker_enabled=settings.RERANK_ENABLED,
        query_expansion_enabled=settings.QUERY_EXPANSION_ENABLED,
    )


@router.post("/build")
async def build(req: BuildRequest) -> BuildResponse:
    """
    Build or rebuild the search index from arXiv papers.

    Example query: 'cat:cs.LG AND (rag OR retrieval OR agentic)'
    """
    try:
        logger.info("Building index", query=req.query, max_results=req.max_results)
        stats = await build_index_async(req.query, req.max_results, get_store())
        return BuildResponse(status="ok", stats=stats)
    except Exception as e:
        logger.error("Build failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
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


@router.get("/metrics")
def metrics_endpoint():
    """Prometheus metrics endpoint."""
    if not settings.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics disabled")

    metrics_data = get_metrics()
    return Response(content=metrics_data, media_type="text/plain")


@router.get("/metrics/summary")
def metrics_summary():
    """Get human-readable metrics summary."""
    if not settings.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics disabled")

    return get_metrics_summary()


@router.get("/cache/stats")
def cache_stats():
    """Get cache statistics."""
    cache = get_cache()
    return cache.get_stats()


@router.post("/cache/clear")
def cache_clear(pattern: str = "arxiv_rag:*"):
    """Clear cache entries matching pattern."""
    cache = get_cache()
    if not cache.enabled:
        raise HTTPException(status_code=503, detail="Cache not available")

    deleted = cache.clear_pattern(pattern)
    return {"status": "ok", "deleted": deleted, "pattern": pattern}


@router.get("/cache/redis/status")
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


@router.get("/background-tasks/status")
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


@router.post("/background-tasks/start")
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


@router.post("/background-tasks/stop")
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


@router.get("/tracing/status")
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


@router.get("/index/papers")
def list_index_papers(limit: int = 50):
    """
    List all unique papers currently in the index (grouped by title).

    Useful for debugging and verifying what papers were indexed.
    """
    store = get_store()
    try:
        collection = store.client.get_collection(store.collection_name)
        results = collection.get(include=["documents", "metadatas"])

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

        papers_list = list(papers_by_title.values())[:limit]

        for i, paper in enumerate(papers_list):
            paper["index"] = i + 1

        return {
            "total": len(papers_by_title),
            "chunks": store.count(),
            "shown": len(papers_list),
            "papers": papers_list,
        }
    except Exception as e:
        logger.error("Failed to list papers", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
