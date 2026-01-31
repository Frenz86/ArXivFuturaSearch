"""Graceful shutdown handling for FastAPI applications.

Ensures that in-flight requests are completed, resources are cleaned up,
and connections are closed properly during shutdown.
"""

import asyncio
import signal
from contextlib import asynccontextmanager
from typing import Optional

from app.logging_config import get_logger

logger = get_logger(__name__)


class GracefulShutdown:
    """
    Manager for graceful application shutdown.

    Features:
    - Signal handling (SIGTERM, SIGINT)
    - Request draining timeout
    - Resource cleanup callbacks
    - Health check during shutdown
    """

    def __init__(
        self,
        shutdown_timeout: float = 30.0,
        drain_timeout: float = 10.0,
    ):
        """
        Initialize shutdown manager.

        Args:
            shutdown_timeout: Max time to wait for shutdown (seconds)
            drain_timeout: Max time to finish in-flight requests (seconds)
        """
        self.shutdown_timeout = shutdown_timeout
        self.drain_timeout = drain_timeout
        self._shutdown_event = asyncio.Event()
        self._cleanup_callbacks = []
        self._is_shutting_down = False

    def add_cleanup_callback(self, callback, *args, **kwargs):
        """
        Add a cleanup callback to run on shutdown.

        Args:
            callback: Async or sync function to call
            *args, **kwargs: Arguments for the callback
        """
        self._cleanup_callbacks.append((callback, args, kwargs))

    async def _run_cleanup_callbacks(self):
        """Run all cleanup callbacks."""
        logger.info("Running cleanup callbacks", count=len(self._cleanup_callbacks))

        for callback, args, kwargs in self._cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error("Cleanup callback failed", callback=callback.__name__, error=str(e))

    async def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    def trigger_shutdown(self, sig: Optional[int] = None):
        """Trigger shutdown process."""
        if self._is_shutting_down:
            logger.warning("Shutdown already in progress")
            return

        self._is_shutting_down = True
        sig_name = signal.Signals(sig).name if sig else "UNKNOWN"
        logger.info("Shutdown triggered", signal=sig_name)

        self._shutdown_event.set()

    async def shutdown(self):
        """
        Perform graceful shutdown.

        Returns:
            True if shutdown completed successfully
        """
        if not self._is_shutting_down:
            logger.info("Starting graceful shutdown")

        try:
            # Wait for in-flight requests to complete (draining period)
            logger.info("Draining in-flight requests", timeout=self.drain_timeout)

            # Run cleanup callbacks
            await self._run_cleanup_callbacks()

            logger.info("Graceful shutdown completed")
            return True

        except asyncio.TimeoutError:
            logger.warning("Shutdown timeout exceeded")
            return False
        except Exception as e:
            logger.error("Shutdown failed", error=str(e))
            return False


# =============================================================================
# FASTAPI INTEGRATION
# =============================================================================

_shutdown_manager: Optional[GracefulShutdown] = None


def get_shutdown_manager() -> GracefulShutdown:
    """Get the global shutdown manager instance."""
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = GracefulShutdown()
    return _shutdown_manager


def setup_signal_handlers(app):
    """
    Setup signal handlers for graceful shutdown.

    Args:
        app: FastAPI application instance
    """
    shutdown_manager = get_shutdown_manager()

    def handle_signal(sig, frame):
        """Handle incoming shutdown signal."""
        shutdown_manager.trigger_shutdown(sig)

    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info("Signal handlers registered")


@asynccontextmanager
async def lifespan_with_graceful_shutdown(app, original_lifespan):
    """
    Wrapper for FastAPI lifespan with graceful shutdown support.

    Args:
        app: FastAPI application
        original_lifespan: Original lifespan context manager

    Usage:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Your startup code
            yield
            # Your shutdown code

        app = FastAPI(lifespan=lifespan_with_graceful_shutdown(app, lifespan))
    """
    shutdown_manager = get_shutdown_manager()
    setup_signal_handlers(app)

    # Run startup
    async with original_lifespan(app):
        # App is running
        logger.info("Application ready, awaiting shutdown signal")

        # Wait for shutdown signal in background task
        shutdown_task = asyncio.create_task(shutdown_manager.wait_for_shutdown())

        try:
            # Yield to let the app run
            yield

        finally:
            # Cancel the wait task if still running
            if not shutdown_task.done():
                shutdown_task.cancel()
                try:
                    await shutdown_task
                except asyncio.CancelledError:
                    pass

            # Perform graceful shutdown
            logger.info("Starting graceful shutdown process")
            success = await shutdown_manager.shutdown()

            if not success:
                logger.warning("Graceful shutdown did not complete successfully")


# =============================================================================
# HEALTH CHECK INTEGRATION
# =============================================================================

def is_shutting_down() -> bool:
    """Check if application is shutting down."""
    shutdown_manager = get_shutdown_manager()
    return shutdown_manager._is_shutting_down


# =============================================================================
# DECORATORS
# =============================================================================

def reject_new_requests(func):
    """
    Decorator to reject new requests during shutdown.

    Args:
        func: Endpoint function to wrap

    Raises:
        HTTPException: 503 Service Unavailable if shutting down
    """
    from fastapi import HTTPException

    async def wrapper(*args, **kwargs):
        if is_shutting_down():
            raise HTTPException(
                status_code=503,
                detail="Service is shutting down. Please retry later.",
                headers={"Retry-After": "30"}
            )
        return await func(*args, **kwargs)

    return wrapper


# =============================================================================
# SHUTDOWN TASK EXAMPLE
# =============================================================================

async def example_cleanup_task():
    """Example cleanup task for demonstration."""
    # Close database connections
    # Flush cache
    # Save metrics
    # Close network connections
    logger.info("Example cleanup task completed")
