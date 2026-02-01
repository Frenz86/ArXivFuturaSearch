"""FastAPI server for ArXiv Futura Search v0.4.0 with LangChain, ChromaDB, caching, and metrics."""


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
from fastapi.responses import HTMLResponse
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

    # Initialize OpenTelemetry if enabled
    if settings.ENVIRONMENT != "test":
        try:
            from app.tracing import init_telemetry, OpenTelemetryConfig

            otel_config = OpenTelemetryConfig(
                service_name="arxiv_futura_search",
                service_version=settings.VERSION,
                enabled=True,
                trace_exporter="console",
                metrics_exporter="console",
            )

            await init_telemetry(app, otel_config)
            logger.info("OpenTelemetry initialized", export="console")
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
            description="Hybrid RAG pipeline for ML research papers with OpenRouter integration",
            version=settings.VERSION,
            lifespan=lifespan,
            )

# Setup middleware
setup_cors_middleware(app)
setup_middleware(app)

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Mount static files if directory exists
import os
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(web.router, tags=["Web Interface"])
app.include_router(search.router, tags=["Search"])
app.include_router(advanced_search.router, tags=["Advanced Search"])
app.include_router(monitoring.router, tags=["Monitoring"])
app.include_router(evaluation.router, tags=["Evaluation"])
