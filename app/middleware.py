"""Middleware for security, observability, and cross-cutting concerns.


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

Includes rate limiting, CORS, correlation ID, and security validation.
"""

import uuid
import time
import os
from typing import Callable, Optional
from functools import wraps

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.logging_config import get_logger
from app.config import settings

logger = get_logger(__name__)


# =============================================================================
# CORRELATION ID MIDDLEWARE
# =============================================================================

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Add correlation ID to all requests for distributed tracing.

    Generates a unique ID for each request and adds it to:
    - Response headers (X-Request-ID)
    - Logging context (correlation_id)
    """

    async def dispatch(self, request: Request, call_next):
        # Get existing correlation ID or generate new one
        correlation_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store in state for access in endpoints
        request.state.correlation_id = correlation_id

        # Process request
        start_time = time.time()
        response = await call_next(request)

        # Add to response headers
        response.headers["X-Request-ID"] = correlation_id

        # Log request duration
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=f"{duration_ms:.2f}",
            correlation_id=correlation_id,
        )

        return response


# =============================================================================
# SECURITY VALIDATION MIDDLEWARE
# =============================================================================

class SecurityValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate security settings on startup.

    Warns about insecure defaults and configuration issues.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._validated = False
        self._validate_on_startup()

    def _validate_on_startup(self):
        """Validate security settings on initialization."""
        if self._validated:
            return

        warnings = []

        # Check for default passwords
        default_passwords = [
            ("changeme", "POSTGRES_PASSWORD"),
            ("postgres", "POSTGRES_PASSWORD"),
            ("redis", "REDIS_PASSWORD"),
        ]

        for pwd, env_var in default_passwords:
            env_value = os.getenv(env_var, "")
            if env_value.lower() in [pwd, f"${{{env_var}:-{pwd}}}".lower()]:
                warnings.append(f"Using default {env_var}='{pwd}' - CHANGE IN PRODUCTION!")

        # Check if running in development mode
        if settings.ENVIRONMENT == "development":
            warnings.append("Running in DEVELOPMENT mode - not suitable for production")

        # Check for missing API keys in required modes
        if settings.LLM_MODE == "openrouter" and not settings.OPENROUTER_API_KEY:
            warnings.append("OPENROUTER_API_KEY not set but LLM_MODE=openrouter")

        # Log warnings
        for warning in warnings:
            logger.warning("Security validation", warning=warning)

        self._validated = True

    async def dispatch(self, request: Request, call_next):
        # No runtime checks, validation done on startup
        return await call_next(request)


# =============================================================================
# RATE LIMITING MIDDLEWARE (In-Memory)
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute per client
        """
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check if request is allowed.

        Args:
            client_id: Unique identifier for the client (IP address)

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        now = time.time()
        window_start = now - 60  # 1 minute window

        # Get or init client request history
        if client_id not in self._requests:
            self._requests[client_id] = []

        # Clean old requests outside the window
        self._requests[client_id] = [
            req_time for req_time in self._requests[client_id]
            if req_time > window_start
        ]

        # Check limit
        request_count = len(self._requests[client_id])
        if request_count >= self.requests_per_minute:
            # Calculate when the oldest request will expire
            oldest_request = min(self._requests[client_id])
            retry_after = int(oldest_request - window_start) + 1
            return False, retry_after

        # Add current request
        self._requests[client_id].append(now)
        return True, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to prevent abuse.

    Limits requests per client IP address.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        excluded_paths: Optional[set[str]] = None,
    ):
        """
        Initialize rate limiting middleware.

        Args:
            app: ASGI application
            requests_per_minute: Maximum requests per minute
            excluded_paths: Paths to exclude from rate limiting
        """
        super().__init__(app)
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.excluded_paths = excluded_paths or {"/health", "/metrics", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        # Skip excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Get client IP
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        # Check rate limit
        allowed, retry_after = self.rate_limiter.is_allowed(client_ip)

        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                path=request.url.path,
                retry_after=retry_after,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)


# =============================================================================
# REQUEST LOGGING MIDDLEWARE
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Detailed request/response logging middleware.

    Logs incoming requests with contextual information for debugging.
    """

    async def dispatch(self, request: Request, call_next):
        # Get correlation ID
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")

        # Log incoming request
        logger.info(
            "Incoming request",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query),
            client_ip=client_ip,
            user_agent=user_agent,
            correlation_id=correlation_id,
        )

        # Process request
        try:
            response = await call_next(request)
            logger.info(
                "Request successful",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                correlation_id=correlation_id,
            )
            return response
        except Exception as e:
            logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id,
            )
            raise


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def setup_cors_middleware(app) -> None:
    """
    Configure CORS middleware for the application.

    Args:
        app: FastAPI application instance
    """
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

    if not allowed_origins or allowed_origins == [""]:
        # Default to localhost for development
        allowed_origins = [
            "http://localhost:8000",
            "http://localhost:3000",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:3000",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    logger.info("CORS middleware configured", allowed_origins=allowed_origins)


def setup_middleware(app) -> None:
    """
    Set up all middleware for the application.

    Args:
        app: FastAPI application instance
    """
    # Order matters! Middleware is applied in reverse order of addition.

    # 1. Security validation (no-op after startup)
    app.add_middleware(SecurityValidationMiddleware)

    # 2. Request validation with Pydantic schemas
    try:
        from app.validation_middleware import ValidationMiddleware
        enable_validation = settings.ENVIRONMENT != "test"
        app.add_middleware(ValidationMiddleware, enable_validation=enable_validation)
        logger.info("Pydantic validation middleware enabled")
    except ImportError:
        logger.warning("Validation middleware not available")

    # 3. Rate limiting (configurable via env)
    rate_limit_rpm = int(os.getenv("RATE_LIMIT_RPM", "60"))
    if rate_limit_rpm > 0:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit_rpm)
        logger.info("Rate limiting enabled", requests_per_minute=rate_limit_rpm)
    else:
        logger.info("Rate limiting disabled")

    # 4. Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # 5. Correlation ID (always last to wrap everything)
    app.add_middleware(CorrelationIDMiddleware)

    # CORS is added separately via setup_cors_middleware
    logger.info("Middleware stack configured")
