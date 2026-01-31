"""Redis-based caching layer for retrieval results and embeddings."""

import json
import hashlib
from typing import Optional, Any
from functools import wraps

import redis
from redis.exceptions import RedisError

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class CacheClient:
    """Redis cache client with automatic fallback."""

    def __init__(self):
        """Initialize Redis client."""
        self._client: Optional[redis.Redis] = None
        self._enabled = settings.CACHE_ENABLED

        if self._enabled:
            try:
                self._client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                # Test connection
                self._client.ping()
                logger.info(
                    "Redis cache connected",
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                )
            except RedisError as e:
                logger.warning(
                    "Redis connection failed, caching disabled",
                    error=str(e),
                )
                self._client = None
                self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if cache is enabled and working."""
        return self._enabled and self._client is not None

    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Create a cache key from prefix and arguments.

        Args:
            prefix: Key prefix (e.g., "query", "embed")
            *args: Positional arguments to hash
            **kwargs: Keyword arguments to hash

        Returns:
            Cache key string
        """
        # Combine all arguments into a string
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = ":".join(key_parts)

        # Hash for consistent key length
        key_hash = hashlib.md5(key_str.encode()).hexdigest()

        return f"arxiv_rag:{prefix}:{key_hash}"

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.enabled:
            return None

        try:
            value = self._client.get(key)
            if value is not None:
                logger.debug("Cache hit", key=key)
                return json.loads(value)
            logger.debug("Cache miss", key=key)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (default: settings.CACHE_TTL)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            ttl = ttl or settings.CACHE_TTL
            serialized = json.dumps(value)
            self._client.setex(key, ttl, serialized)
            logger.debug("Cache set", key=key, ttl=ttl)
            return True
        except (RedisError, TypeError, ValueError) as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False otherwise
        """
        if not self.enabled:
            return False

        try:
            result = self._client.delete(key)
            logger.debug("Cache delete", key=key, deleted=bool(result))
            return bool(result)
        except RedisError as e:
            logger.warning("Cache delete failed", key=key, error=str(e))
            return False

    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.

        Args:
            pattern: Redis pattern (e.g., "arxiv_rag:query:*")

        Returns:
            Number of keys deleted
        """
        if not self.enabled:
            return 0

        try:
            keys = self._client.keys(pattern)
            if keys:
                deleted = self._client.delete(*keys)
                logger.info("Cache cleared", pattern=pattern, deleted=deleted)
                return deleted
            return 0
        except RedisError as e:
            logger.warning("Cache clear failed", pattern=pattern, error=str(e))
            return 0

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self.enabled:
            return {"enabled": False, "status": "disabled"}

        try:
            info = self._client.info("stats")
            return {
                "enabled": True,
                "status": "connected",
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "keys": self._client.dbsize(),
            }
        except RedisError as e:
            logger.warning("Failed to get cache stats", error=str(e))
            return {"enabled": True, "status": "error", "error": str(e)}


# Global cache instance
_cache_client: Optional[CacheClient] = None


def get_cache() -> CacheClient:
    """Get the global cache client instance."""
    global _cache_client
    if _cache_client is None:
        _cache_client = CacheClient()
    return _cache_client


def cached(prefix: str, ttl: Optional[int] = None):
    """
    Decorator to cache function results.

    Args:
        prefix: Cache key prefix
        ttl: Time-to-live in seconds (default: settings.CACHE_TTL)

    Example:
        @cached("query", ttl=3600)
        def search(query: str, top_k: int = 5):
            # expensive operation
            return results
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()

            if not cache.enabled:
                # Cache disabled, call function directly
                return func(*args, **kwargs)

            # Generate cache key from function arguments
            key = cache._make_key(prefix, *args, **kwargs)

            # Try to get from cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            # Cache miss - call function
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


# Async version for async functions
def cached_async(prefix: str, ttl: Optional[int] = None):
    """
    Decorator to cache async function results.

    Args:
        prefix: Cache key prefix
        ttl: Time-to-live in seconds

    Example:
        @cached_async("embed", ttl=7200)
        async def embed_text(text: str):
            # expensive async operation
            return embedding
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = get_cache()

            if not cache.enabled:
                return await func(*args, **kwargs)

            key = cache._make_key(prefix, *args, **kwargs)

            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            result = await func(*args, **kwargs)
            cache.set(key, result, ttl=ttl)

            return result

        return wrapper
    return decorator
