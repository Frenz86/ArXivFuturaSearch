"""Dependency injection for FastAPI endpoints.


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

Provides FastAPI Depends() dependencies for services and resources.
"""

from typing import Optional
from functools import lru_cache

from fastapi import Depends

from app.config import settings, Settings
from app.vectorstore import VectorStoreInterface, get_vectorstore
from app.embeddings import Embedder, get_embedder, Reranker, get_reranker
from app.cache import CacheClient, get_cache
from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# SETTINGS DEPENDENCY
# =============================================================================

@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return settings


# =============================================================================
# VECTOR STORE DEPENDENCY
# =============================================================================

# Global store instance (managed by main.py's lifespan)
_store_instance: Optional[VectorStoreInterface] = None


def set_store_instance(store: VectorStoreInterface) -> None:
    """Set the global store instance (called from lifespan)."""
    global _store_instance
    _store_instance = store


def get_store() -> VectorStoreInterface:
    """Get the vector store instance."""
    if _store_instance is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Please build the index first.",
        )
    return _store_instance


# Optional store dependency (returns None if not initialized)
async def get_optional_store() -> Optional[VectorStoreInterface]:
    """Get the vector store instance or None if not initialized."""
    return _store_instance


# =============================================================================
# EMBEDDER DEPENDENCY
# =============================================================================

async def get_embedder_dependency() -> Embedder:
    """Get embedder instance as a FastAPI dependency."""
    embedder = get_embedder()
    if embedder is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Embedder not initialized. Check configuration.",
        )
    return embedder


# =============================================================================
# RERANKER DEPENDENCY
# =============================================================================

async def get_reranker_dependency() -> Optional[Reranker]:
    """Get reranker instance as a FastAPI dependency (returns None if disabled)."""
    if not settings.RERANK_ENABLED:
        return None
    return get_reranker()


# =============================================================================
# CACHE DEPENDENCY
# =============================================================================

async def get_cache_dependency() -> CacheClient:
    """Get cache client as a FastAPI dependency."""
    return get_cache()


# =============================================================================
# HEALTH CHECK DEPENDENCIES
# =============================================================================

async def check_embedder_health() -> dict:
    """Check embedder health."""
    try:
        embedder = get_embedder()
        if embedder is None:
            return {"healthy": False, "details": "Embedder not initialized"}

        # Test embedding
        test_vec = embedder.embed_query("test")
        if test_vec is None or len(test_vec) == 0:
            return {"healthy": False, "details": "Embedder returned empty vector"}

        return {
            "healthy": True,
            "details": f"Embedder working, dimension: {len(test_vec)}",
            "model": settings.EMBED_MODEL,
            "dimension": len(test_vec),
        }
    except Exception as e:
        logger.error("Embedder health check failed", error=str(e))
        return {"healthy": False, "details": str(e)}


async def check_store_health() -> dict:
    """Check vector store health."""
    try:
        store = get_store()
        if store is None:
            return {"healthy": False, "details": "Vector store not initialized"}

        count = store.count()
        return {
            "healthy": True,
            "details": f"Vector store connected, documents: {count}",
            "documents": count,
        }
    except Exception as e:
        logger.error("Vector store health check failed", error=str(e))
        return {"healthy": False, "details": str(e)}


async def check_cache_health() -> dict:
    """Check cache health."""
    try:
        cache = get_cache()
        if not cache.enabled:
            return {"healthy": True, "details": "Cache disabled"}

        stats = cache.get_stats()
        if stats.get("status") == "connected":
            return {
                "healthy": True,
                "details": "Cache connected",
                "keys": stats.get("keys", 0),
            }
        else:
            return {"healthy": False, "details": stats.get("status", "Unknown")}
    except Exception as e:
        logger.error("Cache health check failed", error=str(e))
        return {"healthy": False, "details": str(e)}
