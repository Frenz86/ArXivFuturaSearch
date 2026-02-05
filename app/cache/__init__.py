"""Cache package for advanced caching features.

This package provides semantic caching, cache warming, and other advanced features.
The basic Redis cache implementation is available via re-exports from simple_cache.
"""

# Re-export basic cache functionality for backward compatibility
from app.simple_cache import (
    CacheClient,
    get_cache,
    cached,
    cached_async,
)

__all__ = [
    "CacheClient",
    "get_cache",
    "cached",
    "cached_async",
]
