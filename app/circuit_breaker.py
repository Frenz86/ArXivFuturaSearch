"""Circuit breaker pattern for resilient LLM and external API calls.

Prevents cascading failures by automatically failing fast when a service
is experiencing problems, and allowing it to recover when healthy.
"""

import asyncio
import time
from enum import Enum
from typing import Callable, Optional, Any, TypeVar
from functools import wraps
from collections import deque

from app.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Circuit is open, requests fail immediately
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is OPEN."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for resilient service calls.

    States:
    - CLOSED: Normal operation, requests pass through. Failure count increases on errors.
    - OPEN: Circuit is tripped, requests fail immediately. No requests go through.
    - HALF_OPEN: Testing recovery. Allows one request through to test service.

    Transitions:
    - CLOSED -> OPEN: When failure threshold is reached
    - OPEN -> HALF_OPEN: After timeout period has elapsed
    - HALF_OPEN -> CLOSED: If test request succeeds
    - HALF_OPEN -> OPEN: If test request fails
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_attempts: int = 1,
        success_threshold: int = 2,
        rolling_window_size: int = 100,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_attempts: Number of attempts allowed in HALF_OPEN state
            success_threshold: Consecutive successes needed to close circuit
            rolling_window_size: Size of rolling window for metrics
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts
        self.success_threshold = success_threshold

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._opened_at: Optional[float] = None
        self._half_open_attempts_used = 0

        # Metrics (rolling window)
        self._rolling_window = deque(maxlen=rolling_window_size)
        self._total_requests = 0
        self._total_failures = 0
        self._total_successes = 0

        # Lock for thread safety
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate from rolling window."""
        if not self._rolling_window:
            return 0.0
        return sum(1 for x in self._rolling_window if not x) / len(self._rolling_window)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._opened_at is None:
            return True
        return time.time() - self._opened_at >= self.recovery_timeout

    async def _record_success(self):
        """Record a successful call."""
        self._success_count += 1
        self._failure_count = 0
        self._total_successes += 1
        self._rolling_window.append(True)

        if self._state == CircuitState.HALF_OPEN:
            if self._success_count >= self.success_threshold:
                async with self._lock:
                    self._state = CircuitState.CLOSED
                    self._half_open_attempts_used = 0
                    logger.info(
                        "Circuit breaker CLOSED after recovery",
                        success_count=self._success_count,
                    )

    async def _record_failure(self):
        """Record a failed call."""
        self._failure_count += 1
        self._success_count = 0
        self._total_failures += 1
        self._rolling_window.append(False)
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Failed during recovery, reopen circuit
            async with self._lock:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                self._half_open_attempts_used = 0
                logger.warning("Circuit breaker RE-OPENED after half-open failure")

        elif self._failure_count >= self.failure_threshold:
            # Too many failures, open circuit
            async with self._lock:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                logger.warning(
                    "Circuit breaker OPENED",
                    failure_count=self._failure_count,
                    threshold=self.failure_threshold,
                )

    async def call(self, func: Callable[..., T], *args: **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to call (can be sync or async)
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is OPEN
            Exception: If function call fails
        """
        self._total_requests += 1

        # Check if circuit should transition from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN and self._should_attempt_reset():
            async with self._lock:
                if self._state == CircuitState.OPEN:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_attempts_used = 0
                    logger.info("Circuit breaker HALF_OPEN (attempting recovery)")

        # Fail fast if circuit is OPEN
        if self._state == CircuitState.OPEN:
            logger.debug("Circuit breaker OPEN - failing fast")
            raise CircuitBreakerError(
                f"Circuit breaker is OPEN. Try again after {self.recovery_timeout} seconds."
            )

        # Check half-open attempts limit
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_attempts_used >= self.half_open_attempts:
                logger.debug("Half-open attempts exhausted")
                raise CircuitBreakerError("Half-open attempts exhausted, circuit remains OPEN.")

        # Execute the function
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            await self._record_success()
            return result

        except Exception as e:
            # Record failure
            await self._record_failure()
            raise

    def get_metrics(self) -> dict:
        """Get circuit breaker metrics."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_requests": self._total_requests,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "failure_rate": f"{self.failure_rate:.2%}",
            "opened_at": self._opened_at,
            "last_failure_time": self._last_failure_time,
        }


# =============================================================================
# GLOBAL CIRCUIT BREAKER INSTANCES
# =============================================================================

_llm_circuit_breaker: Optional[CircuitBreaker] = None
_api_circuit_breaker: Optional[CircuitBreaker] = None


def get_llm_circuit_breaker() -> CircuitBreaker:
    """Get or create LLM circuit breaker."""
    global _llm_circuit_breaker
    if _llm_circuit_breaker is None:
        _llm_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            half_open_attempts=2,
            success_threshold=2,
        )
    return _llm_circuit_breaker


def get_api_circuit_breaker() -> CircuitBreaker:
    """Get or create external API circuit breaker."""
    global _api_circuit_breaker
    if _api_circuit_breaker is None:
        _api_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            half_open_attempts=1,
            success_threshold=1,
        )
    return _api_circuit_breaker


# =============================================================================
# DECORATORS
# =============================================================================

def with_circuit_breaker(
    circuit_breaker: Optional[CircuitBreaker] = None,
    fallback: Optional[Any] = None,
):
    """
    Decorator to apply circuit breaker to a function.

    Args:
        circuit_breaker: Circuit breaker instance (default: LLM circuit breaker)
        fallback: Value to return when circuit is OPEN

    Example:
        @with_circuit_breaker()
        async def call_llm(prompt: str) -> str:
            return await llm.ainvoke(prompt)
    """
    if circuit_breaker is None:
        circuit_breaker = get_llm_circuit_breaker()

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await circuit_breaker.call(func, *args, **kwargs)
            except CircuitBreakerError:
                if fallback is not None:
                    logger.warning("Circuit open, using fallback", function=func.__name__)
                    return fallback
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run the async wrapper in event loop
            return asyncio.run(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# HEALTH CHECK
# =============================================================================

def get_circuit_breaker_status() -> dict:
    """Get status of all circuit breakers."""
    return {
        "llm": get_llm_circuit_breaker().get_metrics(),
        "api": get_api_circuit_breaker().get_metrics(),
    }
