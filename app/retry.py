"""Retry logic with tenacity for LLM calls and external API requests."""


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

import asyncio
from typing import Callable, TypeVar, Any
from functools import wraps

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# LLM RETRY CONFIGURATION
# =============================================================================

def should_retry_llm(exception: Exception) -> bool:
    """
    Determine if an LLM exception should trigger a retry.

    Args:
        exception: The exception that was raised

    Returns:
        True if the exception is retryable
    """
    # Retry on timeout, connection errors, and rate limits
    retryable_errors = (
        TimeoutError,
        ConnectionError,
        OSError,
    )

    # Check exception type
    if isinstance(exception, retryable_errors):
        return True

    # Check for specific error messages
    error_msg = str(exception).lower()
    retryable_patterns = (
        "timeout",
        "connection",
        "rate limit",
        "429",  # HTTP 429 Too Many Requests
        "503",  # HTTP 503 Service Unavailable
        "502",  # HTTP 502 Bad Gateway
        "temporary",
        "unavailable",
    )

    return any(pattern in error_msg for pattern in retryable_patterns)


# =============================================================================
# RETRY DECORATORS
# =============================================================================

def llm_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: float = 2.0,
) -> Callable:
    """
    Retry decorator for LLM calls with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
        exponential_base: Base for exponential backoff

    Returns:
        Decorated function with retry logic

    Example:
        @llm_retry(max_attempts=3)
        async def generate(prompt: str) -> str:
            return await llm.ainvoke(prompt)
    """
    def decorator(func: Callable) -> Callable:
        # Async wrapper
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt = 0
                last_exception = None

                while attempt < max_attempts:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        attempt += 1

                        if attempt >= max_attempts:
                            logger.error(
                                "LLM call failed after all retries",
                                function=func.__name__,
                                attempts=attempt,
                                error=str(e),
                            )
                            raise

                        # Calculate wait time with exponential backoff
                        wait_time = min(
                            min_wait * (exponential_base ** (attempt - 1)),
                            max_wait
                        )

                        logger.warning(
                            "LLM call failed, retrying",
                            function=func.__name__,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            wait_time=f"{wait_time:.2f}s",
                            error=str(e),
                        )

                        await asyncio.sleep(wait_time)

                # Should not reach here, but just in case
                raise last_exception

            return async_wrapper

        # Sync wrapper
        else:
            import time

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt = 0
                last_exception = None

                while attempt < max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        attempt += 1

                        if attempt >= max_attempts:
                            logger.error(
                                "LLM call failed after all retries",
                                function=func.__name__,
                                attempts=attempt,
                                error=str(e),
                            )
                            raise

                        # Calculate wait time with exponential backoff
                        wait_time = min(
                            min_wait * (exponential_base ** (attempt - 1)),
                            max_wait
                        )

                        logger.warning(
                            "LLM call failed, retrying",
                            function=func.__name__,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            wait_time=f"{wait_time:.2f}s",
                            error=str(e),
                        )

                        time.sleep(wait_time)

                # Should not reach here, but just in case
                raise last_exception

            return sync_wrapper

    return decorator


def api_retry(
    max_attempts: int = 2,
    min_wait: float = 0.5,
    max_wait: float = 5.0,
) -> Callable:
    """
    Retry decorator for external API calls (lighter than LLM retry).

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt = 0
                last_exception = None

                while attempt < max_attempts:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        attempt += 1

                        if attempt >= max_attempts or not should_retry_llm(e):
                            raise

                        wait_time = min(min_wait * (2 ** (attempt - 1)), max_wait)
                        logger.debug(
                            "API call retrying",
                            function=func.__name__,
                            attempt=attempt,
                            error=str(e),
                        )
                        await asyncio.sleep(wait_time)

                raise last_exception

            return async_wrapper
        else:
            import time

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt = 0
                last_exception = None

                while attempt < max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        attempt += 1

                        if attempt >= max_attempts or not should_retry_llm(e):
                            raise

                        wait_time = min(min_wait * (2 ** (attempt - 1)), max_wait)
                        logger.debug(
                            "API call retrying",
                            function=func.__name__,
                            attempt=attempt,
                            error=str(e),
                        )
                        time.sleep(wait_time)

                raise last_exception

            return sync_wrapper

    return decorator
