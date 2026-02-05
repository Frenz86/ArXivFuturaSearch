"""
Adaptive PostgreSQL connection pool with dynamic sizing and health monitoring.

Provides intelligent connection pool management with automatic scaling,
health checks, and exhaustion handling.
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class PoolState(Enum):
    """Connection pool states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    EXHAUSTED = "exhausted"
    RECOVERING = "recovering"


@dataclass
class PoolHealthMetrics:
    """Health metrics for the connection pool."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    average_connection_age: float = 0.0
    connection_failure_rate: float = 0.0
    last_health_check: float = 0.0
    health_check_failures: int = 0

    @property
    def utilization_percentage(self) -> float:
        if self.total_connections == 0:
            return 0.0
        return (self.active_connections / self.total_connections) * 100


@dataclass
class PoolConfig:
    """Configuration for adaptive connection pool."""
    min_size: int = 2
    max_size: int = 20
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    pool_pre_ping: bool = True

    # Adaptive settings
    enable_adaptive_sizing: bool = True
    scale_up_threshold: float = 0.75  # Scale up when 75% utilized
    scale_down_threshold: float = 0.25  # Scale down when 25% utilized
    scale_up_step: int = 2
    scale_down_step: int = 1
    adjustment_interval: int = 60  # Seconds between adjustments

    # Health check settings
    health_check_interval: int = 30  # Seconds
    health_check_timeout: float = 5.0
    max_health_check_failures: int = 3


class AdaptiveConnectionPool:
    """
    Adaptive connection pool for PostgreSQL.

    Features:
    - Dynamic pool sizing based on workload
    - Health monitoring with automatic recovery
    - Connection exhaustion handling
    - Prometheus metrics integration
    """

    def __init__(
        self,
        connection_string: str,
        config: Optional[PoolConfig] = None,
    ):
        """
        Initialize the adaptive connection pool.

        Args:
            connection_string: PostgreSQL connection string
            config: Pool configuration (uses defaults if None)
        """
        self.connection_string = connection_string
        self.config = config or PoolConfig(
            min_size=getattr(settings, 'POSTGRES_POOL_SIZE', 10),
            max_size=getattr(settings, 'POSTGRES_POOL_SIZE', 10) + getattr(settings, 'POSTGRES_MAX_OVERFLOW', 20),
            max_overflow=getattr(settings, 'POSTGRES_MAX_OVERFLOW', 20),
        )

        self.engine: Optional[AsyncEngine] = None
        self.state = PoolState.HEALTHY
        self.metrics = PoolHealthMetrics()

        # Health check tracking
        self._health_check_lock = threading.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._consecutive_failures = 0

        # Adaptive sizing tracking
        self._adjustment_lock = threading.Lock()
        self._adjustment_task: Optional[asyncio.Task] = None
        self._utilization_history: deque = deque(maxlen=10)

        # Exhaustion handling
        self._exhaustion_count = 0
        self._exhaustion_callbacks: List[Callable] = []

        logger.info(
            "AdaptiveConnectionPool initialized",
            config={
                "min_size": self.config.min_size,
                "max_size": self.config.max_size,
                "max_overflow": self.config.max_overflow,
                "adaptive_sizing": self.config.enable_adaptive_sizing,
            }
        )

    async def initialize(self) -> AsyncEngine:
        """
        Initialize the connection pool and start background tasks.

        Returns:
            SQLAlchemy AsyncEngine
        """
        logger.info("Initializing connection pool")

        # Create engine with initial pool settings
        self.engine = create_async_engine(
            self.connection_string,
            poolclass=pool.QueuePool,
            pool_size=self.config.min_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
        )

        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        # Start adaptive sizing task if enabled
        if self.config.enable_adaptive_sizing:
            self._adjustment_task = asyncio.create_task(self._adaptive_sizing_loop())

        # Verify connection
        await self._verify_initial_connection()

        logger.info("Connection pool initialized successfully")
        return self.engine

    async def _verify_initial_connection(self) -> None:
        """Verify that the pool can establish connections."""
        try:
            async with self.engine.connect() as conn:
                await conn.execute("SELECT 1")
            logger.info("Initial connection verified")
        except Exception as e:
            logger.error("Initial connection verification failed", error=str(e))
            raise

    async def _health_check_loop(self) -> None:
        """Background task to perform periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception as e:
                logger.error("Health check loop error", error=str(e))

    async def _perform_health_check(self) -> None:
        """Perform a health check on the connection pool."""
        with self._health_check_lock:
            start_time = time.time()

            try:
                # Get pool statistics
                pool = self.engine.pool
                self.metrics.total_connections = pool.size()
                self.metrics.active_connections = pool.checkedout()
                self.metrics.idle_connections = pool.checkedin()

                # Test a connection
                async with self.engine.connect() as conn:
                    await conn.execute("SELECT 1")

                health_check_time = time.time() - start_time
                self.metrics.last_health_check = time.time()
                self._consecutive_failures = 0

                # Determine pool state
                if self.metrics.utilization_percentage > 90:
                    self.state = PoolState.EXHAUSTED
                    self._handle_exhaustion()
                elif self.metrics.utilization_percentage > 75:
                    self.state = PoolState.DEGRADED
                else:
                    self.state = PoolState.HEALTHY

                logger.debug(
                    "Health check complete",
                    state=self.state.value,
                    utilization=f"{self.metrics.utilization_percentage:.1f}%",
                    connections=self.metrics.total_connections,
                    health_check_time=f"{health_check_time:.3f}s",
                )

            except Exception as e:
                self._consecutive_failures += 1
                self.metrics.health_check_failures += 1

                logger.warning(
                    "Health check failed",
                    failure_count=self._consecutive_failures,
                    error=str(e),
                )

                if self._consecutive_failures >= self.config.max_health_check_failures:
                    self.state = PoolState.RECOVERING

    async def _adaptive_sizing_loop(self) -> None:
        """Background task to adjust pool size based on utilization."""
        while True:
            try:
                await asyncio.sleep(self.config.adjustment_interval)
                await self._adjust_pool_size()
            except asyncio.CancelledError:
                logger.info("Adaptive sizing loop cancelled")
                break
            except Exception as e:
                logger.error("Adaptive sizing loop error", error=str(e))

    async def _adjust_pool_size(self) -> None:
        """Adjust pool size based on current utilization."""
        with self._adjustment_lock:
            if not self.config.enable_adaptive_sizing:
                return

            # Record current utilization
            self._utilization_history.append(self.metrics.utilization_percentage)

            # Calculate average utilization over history
            if len(self._utilization_history) > 0:
                avg_utilization = sum(self._utilization_history) / len(self._utilization_history)
            else:
                avg_utilization = self.metrics.utilization_percentage

            current_size = self.metrics.total_connections

            # Scale up if utilization is high
            if avg_utilization > self.config.scale_up_threshold * 100:
                if current_size < self.config.max_size:
                    new_size = min(
                        current_size + self.config.scale_up_step,
                        self.config.max_size
                    )
                    logger.info(
                        "Scaling up pool",
                        current_size=current_size,
                        new_size=new_size,
                        utilization=f"{avg_utilization:.1f}%",
                    )
                    await self._apply_pool_size(new_size)

            # Scale down if utilization is low
            elif avg_utilization < self.config.scale_down_threshold * 100:
                if current_size > self.config.min_size:
                    new_size = max(
                        current_size - self.config.scale_down_step,
                        self.config.min_size
                    )
                    logger.info(
                        "Scaling down pool",
                        current_size=current_size,
                        new_size=new_size,
                        utilization=f"{avg_utilization:.1f}%",
                    )
                    await self._apply_pool_size(new_size)

    async def _apply_pool_size(self, new_size: int) -> None:
        """Apply new pool size by recreating the engine."""
        logger.info("Applying new pool size", new_size=new_size)

        # Close existing connections
        await self.engine.dispose()

        # Recreate engine with new size
        overflow = max(0, new_size - self.config.min_size)
        self.engine = create_async_engine(
            self.connection_string,
            poolclass=pool.QueuePool,
            pool_size=self.config.min_size,
            max_overflow=overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
        )

        logger.info("Pool size applied successfully")

    def _handle_exhaustion(self) -> None:
        """Handle pool exhaustion event."""
        self._exhaustion_count += 1

        logger.warning(
            "Pool exhaustion detected",
            exhaustion_count=self._exhaustion_count,
            utilization=f"{self.metrics.utilization_percentage:.1f}%",
        )

        # Trigger callbacks
        for callback in self._exhaustion_callbacks:
            try:
                callback(self.metrics)
            except Exception as e:
                logger.error("Exhaustion callback error", error=str(e))

    def register_exhaustion_callback(self, callback: Callable[[PoolHealthMetrics], None]) -> None:
        """Register a callback to be called on pool exhaustion."""
        self._exhaustion_callbacks.append(callback)

    async def close(self) -> None:
        """Close the connection pool and cancel background tasks."""
        logger.info("Closing connection pool")

        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._adjustment_task:
            self._adjustment_task.cancel()
            try:
                await self._adjustment_task
            except asyncio.CancelledError:
                pass

        # Close engine
        if self.engine:
            await self.engine.dispose()

        logger.info("Connection pool closed")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current pool metrics."""
        return {
            "state": self.state.value,
            "total_connections": self.metrics.total_connections,
            "active_connections": self.metrics.active_connections,
            "idle_connections": self.metrics.idle_connections,
            "utilization_percentage": self.metrics.utilization_percentage,
            "failed_connections": self.metrics.failed_connections,
            "last_health_check": self.metrics.last_health_check,
            "health_check_failures": self.metrics.health_check_failures,
            "exhaustion_count": self._exhaustion_count,
        }


# Global pool instance
_adaptive_pool: Optional[AdaptiveConnectionPool] = None


async def get_adaptive_pool() -> AdaptiveConnectionPool:
    """Get the global adaptive connection pool instance."""
    global _adaptive_pool

    if _adaptive_pool is None:
        connection_string = (
            f"postgresql+asyncpg://{getattr(settings, 'POSTGRES_USER', 'postgres')}:"
            f"{getattr(settings, 'POSTGRES_PASSWORD', 'postgres')}"
            f"@{getattr(settings, 'POSTGRES_HOST', 'localhost')}:"
            f"{getattr(settings, 'POSTGRES_PORT', 5432)}"
            f"/{getattr(settings, 'POSTGRES_DB', 'arxiv_rag')}"
        )

        _adaptive_pool = AdaptiveConnectionPool(connection_string)
        await _adaptive_pool.initialize()

    return _adaptive_pool
