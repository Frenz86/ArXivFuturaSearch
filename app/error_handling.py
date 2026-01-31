"""Error handling utilities for LLM calls and external requests.

Provides fallback strategies and error recovery mechanisms.
"""

from typing import Optional, Any, Callable
from functools import wraps
from enum import Enum

from app.logging_config import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        model: str = "",
        recoverable: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.recoverable = recoverable
        super().__init__(message)


class LLMTimeoutError(LLMError):
    """LLM request timeout."""

    def __init__(self, message: str, provider: str = "", model: str = ""):
        super().__init__(message, provider, model, recoverable=True)


class LLMConnectionError(LLMError):
    """LLM connection failure."""

    def __init__(self, message: str, provider: str = ""):
        super().__init__(message, provider, recoverable=True)


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""

    def __init__(self, message: str, provider: str = ""):
        super().__init__(message, provider, recoverable=True)


class LLMInvalidResponseError(LLMError):
    """Invalid LLM response."""

    def __init__(self, message: str, provider: str = "", response: str = ""):
        self.response = response
        super().__init__(message, provider, recoverable=False)


# =============================================================================
# FALLBACK STRATEGIES
# =============================================================================

class FallbackResult:
    """Result from a fallback strategy."""

    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: Optional[Exception] = None,
        fallback_used: str = "",
    ):
        self.success = success
        self.data = data
        self.error = error
        self.fallback_used = fallback_used

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": str(self.error) if self.error else None,
            "fallback_used": self.fallback_used,
        }


def with_fallback(
    primary: Callable,
    fallbacks: list[Callable],
    fallback_names: Optional[list[str]] = None,
) -> Callable:
    """
    Wrap a function with fallback strategies.

    Args:
        primary: Primary function to call
        fallbacks: List of fallback functions to try if primary fails
        fallback_names: Optional names for fallback functions (for logging)

    Returns:
        Wrapped function that tries primary, then fallbacks in order

    Example:
        def primary_llm(prompt: str) -> str:
            return openrouter_client.generate(prompt)

        def fallback_ollama(prompt: str) -> str:
            return ollama_client.generate(prompt)

        def fallback_mock(prompt: str) -> str:
            return "Mock response (LLM unavailable)"

        llm_with_fallback = with_fallback(
            primary_llm,
            [fallback_ollama, fallback_mock],
            ["ollama", "mock"]
        )
    """
    if fallback_names is None:
        fallback_names = [f"fallback_{i}" for i in range(len(fallbacks))]

    @wraps(primary)
    async def async_wrapper(*args, **kwargs) -> FallbackResult:
        last_error = None

        # Try primary
        try:
            result = await primary(*args, **kwargs)
            return FallbackResult(success=True, data=result, fallback_used="primary")
        except Exception as e:
            last_error = e
            logger.warning(
                "Primary function failed",
                function=primary.__name__,
                error=str(e),
            )

        # Try fallbacks
        for fallback, name in zip(fallbacks, fallback_names):
            try:
                result = await fallback(*args, **kwargs)
                logger.info(
                    "Fallback succeeded",
                    fallback=name,
                    function=primary.__name__,
                )
                return FallbackResult(success=True, data=result, fallback_used=name)
            except Exception as e:
                last_error = e
                logger.warning(
                    "Fallback failed",
                    fallback=name,
                    error=str(e),
                )

        # All failed
        logger.error(
            "All strategies failed",
            function=primary.__name__,
            error=str(last_error),
        )
        return FallbackResult(
            success=False,
            error=last_error,
            fallback_used="none",
        )

    @wraps(primary)
    def sync_wrapper(*args, **kwargs) -> FallbackResult:
        import asyncio
        return asyncio.run(async_wrapper(*args, **kwargs))

    # Return appropriate wrapper based on whether primary is async
    import asyncio
    if asyncio.iscoroutinefunction(primary):
        return async_wrapper
    return sync_wrapper


def safe_llm_call(
    prompt: str,
    generate_func: Callable,
    fallback_message: str = "I apologize, but I'm having trouble generating a response right now. Please try again later.",
) -> tuple[bool, str, Optional[Exception]]:
    """
    Safely call an LLM generation function with fallback.

    Args:
        prompt: The prompt to send
        generate_func: Function that takes prompt and returns response
        fallback_message: Message to return on error

    Returns:
        Tuple of (success: bool, response: str, error: Optional[Exception])
    """
    try:
        response = generate_func(prompt)
        return True, response, None
    except Exception as e:
        logger.error(
            "LLM call failed, using fallback",
            error=str(e),
            error_type=type(e).__name__,
        )
        return False, fallback_message, e


def handle_streaming_error(
    generator,
    fallback_message: str = "\n\n[Connection lost. Please try again.]",
):
    """
    Wrap a streaming generator to handle errors gracefully.

    Args:
        generator: The streaming generator
        fallback_message: Message to yield on error

    Yields:
        Chunks from the generator, or fallback message on error
    """
    try:
        for chunk in generator:
            yield chunk
    except GeneratorExit:
        raise
    except Exception as e:
        logger.error("Streaming generator failed", error=str(e))
        yield fallback_message


# =============================================================================
# ERROR MAPPING
# =============================================================================

def classify_error(error: Exception) -> tuple[ErrorSeverity, str]:
    """
    Classify an error by severity and type.

    Args:
        error: The exception to classify

    Returns:
        Tuple of (severity, error_type)
    """
    error_type = type(error).__name__
    error_msg = str(error).lower()

    # Timeouts and connection errors - usually recoverable
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return ErrorSeverity.WARNING, "connection"

    # Rate limits - recoverable with backoff
    if "rate limit" in error_msg or "429" in error_msg:
        return ErrorSeverity.WARNING, "rate_limit"

    # Authentication errors - not recoverable without config change
    if "auth" in error_msg or "401" in error_msg or "api key" in error_msg:
        return ErrorSeverity.ERROR, "authentication"

    # Invalid requests - not recoverable
    if "invalid" in error_msg or "400" in error_msg:
        return ErrorSeverity.ERROR, "invalid_request"

    # Server errors - may be recoverable
    if "503" in error_msg or "502" in error_msg or "unavailable" in error_msg:
        return ErrorSeverity.WARNING, "server_error"

    # Default
    return ErrorSeverity.ERROR, "unknown"


def log_error_context(
    error: Exception,
    context: dict,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
) -> None:
    """
    Log an error with structured context.

    Args:
        error: The exception to log
        context: Additional context information
        severity: Severity level for logging
    """
    severity_func = {
        ErrorSeverity.WARNING: logger.warning,
        ErrorSeverity.ERROR: logger.error,
        ErrorSeverity.CRITICAL: logger.critical,
    }.get(severity, logger.error)

    severity_func(
        "Error occurred",
        error_type=type(error).__name__,
        error_message=str(error),
        **context,
    )
