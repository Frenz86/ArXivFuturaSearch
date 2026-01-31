"""Distributed Redis caching for multi-instance deployments.


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

Provides scalable caching with Redis backend for production environments.
"""

import json
import pickle
import asyncio
from typing import Any, Optional, Dict, List, Union
from datetime import timedelta

import redis.asyncio as aioredis

from app.logging_config import get_logger
from app.config import settings

logger = get_logger(__name__)


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================

class Serializer:
    """Abstract base class for cache serialization."""

    @staticmethod
    def serialize(value: Any) -> bytes:
        """Serialize value to bytes."""
        raise NotImplementedError

    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize bytes to value."""
        raise NotImplementedError


class JSONSerializer(Serializer):
    """JSON serializer for simple data types."""

    @staticmethod
    def serialize(value: Any) -> bytes:
        return json.dumps(value).encode('utf-8')

    @staticmethod
    def deserialize(data: bytes) -> Any:
        return json.loads(data.decode('utf-8'))


class PickleSerializer(Serializer):
    """Pickle serializer for complex Python objects."""

    @staticmethod
    def serialize(value: Any) -> bytes:
        return pickle.dumps(value)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        return pickle.loads(data)


# =============================================================================
# REDIS CACHE CLIENT
# =============================================================================

class RedisCache:
    """
    Async Redis cache client with connection pooling.

    Features:
    - Connection pooling for performance
    - Multiple serialization formats
    - TTL support
    - Prefix-based namespacing
    - Automatic retry on connection failures
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        password: str = None,
        prefix: str = "arxiv_rag:",
        serializer: str = "json",
        max_connections: int = 50,
    ):
        """
        Initialize Redis cache client.

        Args:
            host: Redis host (default from settings)
            port: Redis port (default from settings)
            db: Redis database number (default from settings)
            password: Redis password (default from settings)
            prefix: Key prefix for namespacing
            serializer: Serialization format ('json' or 'pickle')
            max_connections: Maximum pool connections
        """
        self.host = host or settings.REDIS_HOST
        self.port = port or settings.REDIS_PORT
        self.db = db or settings.REDIS_DB
        self.password = password or settings.REDIS_PASSWORD
        self.prefix = prefix
        self.max_connections = max_connections

        # Select serializer
        if serializer == "pickle":
            self.serializer = PickleSerializer()
        else:
            self.serializer = JSONSerializer()

        self._pool: Optional[aioredis.ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """Establish Redis connection pool."""
        if self._client is not None:
            return

        logger.info(
            "Connecting to Redis",
            host=self.host,
            port=self.port,
            db=self.db,
        )

        self._pool = aioredis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            max_connections=self.max_connections,
            decode_responses=False,  # We handle encoding in serializer
            socket_connect_timeout=5.0,
            socket_timeout=5.0,
            retry_on_timeout=True,
        )

        self._client = aioredis.Redis(connection_pool=self._pool)

        # Test connection
        try:
            await self._client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error("Redis connection failed", error=str(e))
            await self.close()
            raise

    async def close(self) -> None:
        """Close Redis connections."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        logger.info("Redis connection closed")

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"

    async def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key (without prefix)
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        if self._client is None:
            await self.connect()

        try:
            full_key = self._make_key(key)
            data = await self._client.get(full_key)

            if data is None:
                return default

            value = self.serializer.deserialize(data)
            logger.debug("Cache hit", key=key)
            return value

        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key (without prefix)
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        if self._client is None:
            await self.connect()

        try:
            full_key = self._make_key(key)
            data = self.serializer.serialize(value)

            if ttl:
                await self._client.setex(full_key, ttl, data)
            else:
                await self._client.set(full_key, data)

            logger.debug("Cache set", key=key, ttl=ttl)
            return True

        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key (without prefix)

        Returns:
            True if key was deleted
        """
        if self._client is None:
            await self.connect()

        try:
            full_key = self._make_key(key)
            result = await self._client.delete(full_key)
            logger.debug("Cache delete", key=key, deleted=result > 0)
            return result > 0

        except Exception as e:
            logger.warning("Cache delete failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key (without prefix)

        Returns:
            True if key exists
        """
        if self._client is None:
            await self.connect()

        try:
            full_key = self._make_key(key)
            result = await self._client.exists(full_key)
            return result > 0

        except Exception as e:
            logger.warning("Cache exists check failed", key=key, error=str(e))
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set TTL on existing key.

        Args:
            key: Cache key (without prefix)
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        if self._client is None:
            await self.connect()

        try:
            full_key = self._make_key(key)
            result = await self._client.expire(full_key, ttl)
            return result > 0

        except Exception as e:
            logger.warning("Cache expire failed", key=key, error=str(e))
            return False

    async def ttl(self, key: str) -> int:
        """
        Get remaining TTL for key.

        Args:
            key: Cache key (without prefix)

        Returns:
            TTL in seconds (-1 if no expiry, -2 if key doesn't exist)
        """
        if self._client is None:
            await self.connect()

        try:
            full_key = self._make_key(key)
            return await self._client.ttl(full_key)

        except Exception as e:
            logger.warning("Cache TTL check failed", key=key, error=str(e))
            return -2

    async def increment(
        self,
        key: str,
        amount: int = 1,
        ttl: Optional[int] = None,
    ) -> int:
        """
        Increment counter value.

        Args:
            key: Cache key (without prefix)
            amount: Amount to increment by
            ttl: Set TTL on new keys

        Returns:
            New counter value
        """
        if self._client is None:
            await self.connect()

        try:
            full_key = self._make_key(key)
            value = await self._client.incrby(full_key, amount)

            # Set TTL if this is a new key
            if ttl and value == amount:
                await self._client.expire(full_key, ttl)

            return value

        except Exception as e:
            logger.warning("Cache increment failed", key=key, error=str(e))
            return 0

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys (without prefix)

        Returns:
            Dict of key-value pairs
        """
        if not keys:
            return {}

        if self._client is None:
            await self.connect()

        try:
            full_keys = [self._make_key(k) for k in keys]
            values = await self._client.mget(full_keys)

            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = self.serializer.deserialize(value)
                    except Exception:
                        pass

            return result

        except Exception as e:
            logger.warning("Cache get_many failed", error=str(e))
            return {}

    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> int:
        """
        Set multiple values in cache.

        Args:
            mapping: Dict of key-value pairs
            ttl: Time-to-live in seconds

        Returns:
            Number of keys set
        """
        if not mapping:
            return 0

        if self._client is None:
            await self.connect()

        try:
            # Use pipeline for batch operations
            pipe = self._client.pipeline()

            for key, value in mapping.items():
                full_key = self._make_key(key)
                data = self.serializer.serialize(value)
                if ttl:
                    pipe.setex(full_key, ttl, data)
                else:
                    pipe.set(full_key, data)

            await pipe.execute()
            return len(mapping)

        except Exception as e:
            logger.warning("Cache set_many failed", error=str(e))
            return 0

    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.

        Args:
            pattern: Key pattern (e.g., "user:*")

        Returns:
            Number of keys deleted
        """
        if self._client is None:
            await self.connect()

        try:
            full_pattern = self._make_key(pattern)
            keys = []
            async for key in self._client.scan_iter(match=full_pattern, count=100):
                keys.append(key)

            if keys:
                return await self._client.delete(*keys)
            return 0

        except Exception as e:
            logger.warning("Cache clear_pattern failed", pattern=pattern, error=str(e))
            return 0

    async def get_info(self) -> Dict[str, Any]:
        """
        Get Redis server information.

        Returns:
            Dict with server info
        """
        if self._client is None:
            await self.connect()

        try:
            info = await self._client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "total_commands": info.get("total_commands_processed", 0),
                "keyspace": info.get("db" + str(self.db), {}),
            }
        except Exception as e:
            logger.warning("Failed to get Redis info", error=str(e))
            return {}


# =============================================================================
# CACHED FUNCTION DECORATOR
# =============================================================================

def cached(
    cache: RedisCache,
    ttl: int = 3600,
    key_prefix: str = "",
    key_builder: Optional[callable] = None,
):
    """
    Decorator to cache function results in Redis.

    Args:
        cache: RedisCache instance
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys
        key_builder: Optional custom key builder function

    Example:
        @cached(redis_cache, ttl=600, key_prefix="search:")
        async def search(query: str):
            return await vector_store.search(query)
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key builder
                key_parts = [key_prefix, func.__name__]
                if args:
                    key_parts.extend(str(a) for a in args)
                if kwargs:
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await cache.set(cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


# =============================================================================
# GLOBAL REDIS CACHE INSTANCE
# =============================================================================

_redis_cache: Optional[RedisCache] = None


def get_redis_cache() -> RedisCache:
    """
    Get or create global Redis cache instance.

    Returns:
        RedisCache instance
    """
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache


async def init_redis_cache() -> None:
    """Initialize Redis cache connection on startup."""
    cache = get_redis_cache()
    await cache.connect()


async def close_redis_cache() -> None:
    """Close Redis cache connection on shutdown."""
    global _redis_cache
    if _redis_cache:
        await _redis_cache.close()
        _redis_cache = None
