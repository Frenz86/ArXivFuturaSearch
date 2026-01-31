"""Connection pooling configuration for databases and external services.

Provides optimized connection pooling settings for PostgreSQL, Redis,
and HTTP clients to improve performance under load.
"""

from typing import Optional

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# POSTGRESQL CONNECTION POOL SETTINGS
# =============================================================================

def get_postgres_pool_config(
    min_size: int = 2,
    max_size: Optional[int] = None,
    max_overflow: Optional[int] = None,
    pool_timeout: float = 30.0,
    pool_recycle: int = 3600,
    pool_pre_ping: bool = True,
) -> dict:
    """
    Get optimized connection pool configuration for PostgreSQL.

    Args:
        min_size: Minimum pool size
        max_size: Maximum pool size (default from settings)
        max_overflow: Max overflow connections (default from settings)
        pool_timeout: Seconds to wait before giving up on getting a connection
        pool_recycle: Seconds before recycling a connection (prevent stale connections)
        pool_pre_ping: Test connections before using them

    Returns:
        Dictionary with pool configuration for SQLAlchemy
    """
    if max_size is None:
        max_size = settings.POSTGRES_POOL_SIZE

    if max_overflow is None:
        max_overflow = settings.POSTGRES_MAX_OVERFLOW

    config = {
        "pool_size": max(1, min_size),
        "max_overflow": max(0, max_overflow),
        "pool_timeout": pool_timeout,
        "pool_recycle": pool_recycle,
        "pool_pre_ping": pool_pre_ping,
        "echo": settings.ENVIRONMENT == "development",  # Log SQL in dev
    }

    logger.info(
        "PostgreSQL pool config",
        pool_size=config["pool_size"],
        max_overflow=config["max_overflow"],
        max_connections=config["pool_size"] + config["max_overflow"],
    )

    return config


# =============================================================================
# REDIS CONNECTION POOL SETTINGS
# =============================================================================

def get_redis_pool_config(
    max_connections: int = 50,
    socket_timeout: float = 5.0,
    socket_connect_timeout: float = 5.0,
    retry_on_timeout: bool = True,
    health_check_interval: int = 30,
) -> dict:
    """
    Get optimized connection pool configuration for Redis.

    Args:
        max_connections: Maximum connections in pool
        socket_timeout: Socket timeout in seconds
        socket_connect_timeout: Connection timeout in seconds
        retry_on_timeout: Retry commands on timeout
        health_check_interval: Seconds between health checks

    Returns:
        Dictionary with pool configuration for redis-py
    """
    config = {
        "max_connections": max_connections,
        "socket_timeout": socket_timeout,
        "socket_connect_timeout": socket_connect_timeout,
        "retry_on_timeout": retry_on_timeout,
        "health_check_interval": health_check_interval,
        "decode_responses": True,  # Always decode to strings
    }

    logger.info(
        "Redis pool config",
        max_connections=config["max_connections"],
        socket_timeout=config["socket_timeout"],
    )

    return config


# =============================================================================
# HTTP CLIENT POOL SETTINGS
# =============================================================================

def get_http_client_config(
    max_connections: int = 100,
    max_keepalive_connections: int = 20,
    keepalive_expiry: float = 5.0,
    timeout: float = 30.0,
    connect_timeout: float = 10.0,
) -> dict:
    """
    Get optimized HTTP client pool configuration.

    Args:
        max_connections: Maximum concurrent connections
        max_keepalive_connections: Maximum keep-alive connections
        keepalive_expiry: Expiry time for keep-alive connections
        timeout: Request timeout in seconds
        connect_timeout: Connection timeout in seconds

    Returns:
        Dictionary with limits configuration for httpx
    """
    config = {
        "max_connections": max_connections,
        "max_keepalive_connections": max_keepalive_connections,
        "keepalive_expiry": keepalive_expiry,
        "timeout": timeout,
        "connect_timeout": connect_timeout,
    }

    logger.info(
        "HTTP client pool config",
        max_connections=config["max_connections"],
        max_keepalive=config["max_keepalive_connections"],
    )

    return config


# =============================================================================
# POOL MONITORING
# =============================================================================

class PoolMetrics:
    """Metrics collector for connection pools."""

    def __init__(self):
        self.postgres_metrics = {
            "size": 0,
            "checked_in": 0,
            "checked_out": 0,
            "overflow": 0,
        }
        self.redis_metrics = {
            "connections": 0,
            "available": 0,
        }

    def update_postgres_metrics(self, pool):
        """Update PostgreSQL pool metrics."""
        if pool is None:
            return

        self.postgres_metrics = {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
        }

    def update_redis_metrics(self, pool):
        """Update Redis pool metrics."""
        if pool is None:
            return

        connection_pool = pool.connection_pool
        if connection_pool:
            self.redis_metrics = {
                "connections": connection_pool.num_connections,
                "available": connection_pool.num_available_connections,
            }

    def get_all_metrics(self) -> dict:
        """Get all pool metrics."""
        return {
            "postgres": self.postgres_metrics,
            "redis": self.redis_metrics,
        }


# Global metrics instance
_pool_metrics = PoolMetrics()


def get_pool_metrics() -> PoolMetrics:
    """Get the global pool metrics instance."""
    return _pool_metrics


# =============================================================================
# CONNECTION POOL FACTORY
# =============================================================================

async def create_postgres_engine():
    """
    Create SQLAlchemy engine with optimized connection pooling.

    Returns:
        SQLAlchemy async engine
    """
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
    from sqlalchemy import pool

    postgres_url = (
        f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )

    pool_config = get_postgres_pool_config()

    engine: AsyncEngine = create_async_engine(
        postgres_url,
        poolclass=pool.QueuePool,
        **pool_config,
    )

    logger.info("PostgreSQL engine created with connection pooling")
    return engine


async def create_redis_pool():
    """
    Create Redis connection pool with optimized settings.

    Returns:
        Redis connection pool
    """
    import redis.asyncio as redis

    pool_config = get_redis_pool_config()

    redis_pool = redis.ConnectionPool(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
        **pool_config,
    )

    logger.info("Redis pool created with connection pooling")
    return redis_pool


def create_http_client():
    """
    Create HTTP client with connection pooling.

    Returns:
        httpx.AsyncClient with connection pooling
    """
    import httpx

    client_config = get_http_client_config()

    # Create async client with connection pooling
    client = httpx.AsyncClient(
        limits=httpx.Limits(**client_config),
        http2=True,  # Enable HTTP/2 if supported
    )

    logger.info("HTTP client created with connection pooling")
    return client
