"""FastAPI server for ArXiv Futura Search v0.4.0 with LangChain, ChromaDB, caching, and metrics."""
## https://github.com/Frenz86/ArXivFuturaSearch

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

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.chunking import ensure_nltk_data_async
from app.embeddings import get_embedder, get_reranker
from app.vectorstore import get_vectorstore, VectorStoreInterface
from app import dependencies as deps
from app.middleware import setup_middleware, setup_cors_middleware
from app.logging_config import get_logger

# Import route modules
from app.api import web, search, advanced_search, monitoring, evaluation

# NEW: Import additional route modules
from app.api import auth, audit, conversations, export, alerts, collections

# NEW: Import middleware
from app.auth.middleware import AuthMiddleware
from app.audit.middleware import AuditMiddleware

# NEW: Import error handlers
from app.errors.handlers import setup_error_handlers

# NEW: Import DI container
from app.container import (
    init_containers,
    close_containers,
    database_container,
)

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
        mode="LangChain + ChromaDB + Native Components"
    )

    # NEW: Initialize dependency injection containers
    try:
        await init_containers()
        logger.info("Dependency injection containers initialized")
    except Exception as e:
        logger.warning("Container initialization failed (continuing without database)", error=str(e))

    # Download NLTK data asynchronously
    await ensure_nltk_data_async()

    # Initialize cache
    from app.cache import get_cache
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

    # NEW: Warm up semantic cache if enabled
    if settings.SEMANTIC_CACHE_ENABLED and settings.CACHE_WARMING_ENABLED:
        try:
            from app.cache.warming import warm_up_cache_on_startup
            await warm_up_cache_on_startup()
            logger.info("Cache warming completed")
        except Exception as e:
            logger.warning("Cache warming failed", error=str(e))

    # Initialize OpenTelemetry if enabled
    if settings.ENVIRONMENT != "test":
        try:
            from app.tracing import init_telemetry, OpenTelemetryConfig

            otel_config = OpenTelemetryConfig(
                service_name="arxiv_futura_search",
                service_version=settings.VERSION,
                enabled=True,
                trace_exporter=settings.OTEL_TRACE_EXPORTER,
                metrics_exporter=settings.OTEL_METRICS_EXPORTER,
            )

            await init_telemetry(app, otel_config)
            logger.info("OpenTelemetry initialized",
                       trace_exporter=settings.OTEL_TRACE_EXPORTER,
                       metrics_exporter=settings.OTEL_METRICS_EXPORTER)
        except ImportError:
            logger.info("OpenTelemetry packages not installed")
        except Exception as e:
            logger.warning("OpenTelemetry initialization failed", error=str(e))

    # Initialize vector store
    logger.info("Initializing vector store...", mode=settings.VECTORSTORE_MODE)
    try:
        _store = get_vectorstore(collection_name="arxiv_papers")
        doc_count = _store.count()
        logger.info("Vector store loaded", documents=doc_count, mode=settings.VECTORSTORE_MODE)

        from app.metrics import update_index_stats
        update_index_stats(documents=doc_count, chunks=doc_count)

        deps.set_store_instance(_store)
    except Exception as e:
        logger.warning("Vector store initialization failed", error=str(e))

    yield

    logger.info("Shutting down ArXiv Futura Search")

    # NEW: Close dependency injection containers
    try:
        await close_containers()
        logger.info("Dependency injection containers closed")
    except Exception as e:
        logger.warning("Container cleanup failed", error=str(e))

    # Shutdown OpenTelemetry
    try:
        from app.tracing import shutdown_telemetry
        shutdown_telemetry()
        logger.info("OpenTelemetry shutdown complete")
    except Exception:
        pass


# Create FastAPI app
app = FastAPI(
    title="ArXiv Futura Search",
    description="""Hybrid RAG pipeline for ML research papers with OpenRouter integration.

## Features
- Semantic search with hybrid BM25 + vector retrieval
- Multi-turn conversations with context management
- Export results (PDF, Markdown, BibTeX, JSON, CSV)
- Alert system for ArXiv feed monitoring
- Collaborative collections with annotations
- OAuth2 authentication (Google, GitHub)
- Comprehensive audit logging

## Authentication
Most endpoints are publicly accessible. Some features require authentication:
- Create/manage alerts
- Create/manage collections
- Save conversations
- Access audit logs (admin only)
""",
    version=settings.VERSION,
    lifespan=lifespan,
    # NEW: OpenAPI configuration for auth
    openapi_tags=[
        {"name": "Web Interface", "description": "Web UI endpoints"},
        {"name": "Search", "description": "Search and query papers"},
        {"name": "Advanced Search", "description": "Advanced search features"},
        {"name": "Authentication", "description": "User authentication and OAuth2"},
        {"name": "Conversations", "description": "Multi-turn chat management"},
        {"name": "Export", "description": "Export search results in various formats"},
        {"name": "Alerts", "description": "ArXiv paper monitoring alerts"},
        {"name": "Collections", "description": "Shared paper collections"},
        {"name": "Audit", "description": "Security audit logs (admin)"},
        {"name": "Monitoring", "description": "System health and metrics"},
        {"name": "Evaluation", "description": "RAG evaluation tools"},
    ],
)

# NEW: Setup error handlers
setup_error_handlers(app)


# =============================================================================
# HEALTH CHECK (Enhanced)
# =============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint.

    Returns the status of all system components.
    """
    from app.cache import get_cache

    health_status = {
        "status": "healthy",
        "version": settings.VERSION,
        "components": {
            "vector_store": {
                "status": "ok" if _store else "not_initialized",
                "mode": settings.VECTORSTORE_MODE,
                "documents": _store.count() if _store else 0,
            },
            "cache": {
                "status": "enabled" if get_cache().enabled else "disabled",
                "type": "redis" if get_cache()._client else "memory",
            },
            "llm": {
                "status": "configured",
                "mode": settings.LLM_MODE,
                "model": settings.OPENROUTER_MODEL if settings.LLM_MODE == "openrouter" else settings.OLLAMA_MODEL,
            },
            "database": {
                "status": "configured" if database_container._engine else "not_configured",
            },
        },
    }

    return health_status


# =============================================================================
# FAVICON
# =============================================================================

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Redirect to SVG favicon."""
    return RedirectResponse(url="/static/favicon.svg")


# =============================================================================
# MIDDLEWARE
# =============================================================================

# Setup CORS and standard middleware
setup_cors_middleware(app)
setup_middleware(app)

# NEW: Add authentication middleware (optional - doesn't require auth by default)
app.add_middleware(
    AuthMiddleware,
    require_auth=False,  # Set to True to require auth for all endpoints
    excluded_paths=["/health", "/api/docs", "/api/openapi.json", "/static/", "/favicon.ico"],
)

# NEW: Add audit logging middleware
app.add_middleware(
    AuditMiddleware,
    excluded_paths=["/health", "/metrics", "/static/", "/favicon.ico"],
)


# =============================================================================
# TEMPLATES & STATIC FILES
# =============================================================================

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Mount static files if directory exists
import os
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# =============================================================================
# ROUTERS
# =============================================================================

# Core routers
app.include_router(web.router, tags=["Web Interface"])
app.include_router(search.router, tags=["Search"])
app.include_router(advanced_search.router, tags=["Advanced Search"])
app.include_router(monitoring.router, tags=["Monitoring"])
app.include_router(evaluation.router, tags=["Evaluation"])

# NEW: Additional feature routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["Conversations"])
app.include_router(export.router, prefix="/api/export", tags=["Export"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(collections.router, prefix="/api/collections", tags=["Collections"])
app.include_router(audit.router, prefix="/api/audit", tags=["Audit"])
