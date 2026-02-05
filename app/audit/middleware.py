"""
Audit middleware for FastAPI.

Logs all requests to the audit log for security and compliance.
"""

from typing import Optional, Set
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time

from app.logging_config import get_logger

logger = get_logger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Audit middleware for logging all HTTP requests.

    Logs request details including:
    - Method and path
    - Response status
    - Request duration
    - IP address and user agent
    - User ID (if authenticated)

    Example:
        app.add_middleware(
            AuditMiddleware,
            excluded_paths={"/health", "/metrics", "/static/"}
        )
    """

    def __init__(
        self,
        app: ASGIApp,
        excluded_paths: Optional[Set[str]] = None,
        excluded_prefixes: Optional[Set[str]] = None,
    ):
        """
        Initialize audit middleware.

        Args:
            app: ASGI application
            excluded_paths: Set of exact paths to exclude from audit logging
            excluded_prefixes: Set of path prefixes to exclude from audit logging
        """
        super().__init__(app)

        # Default excluded paths (high-volume endpoints)
        self.excluded_paths = excluded_paths or {
            "/health",
            "/healthz",
            "/readiness",
            "/metrics",
            "/favicon.ico",
        }

        # Excluded path prefixes
        self.excluded_prefixes = excluded_prefixes or {
            "/static/",
        }

        logger.info(
            "AuditMiddleware initialized",
            excluded_paths=len(self.excluded_paths),
            excluded_prefixes=len(self.excluded_prefixes),
        )

    def _is_excluded_path(self, path: str) -> bool:
        """
        Check if path is excluded from audit logging.

        Args:
            path: Request path

        Returns:
            True if path is excluded from audit logging
        """
        # Check exact matches
        if path in self.excluded_paths:
            return True

        # Check prefix matches
        for prefix in self.excluded_prefixes:
            if path.startswith(prefix):
                return True

        return False

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request through audit middleware.

        Logs request and response details to the audit log.
        """
        path = request.url.path
        method = request.method

        # Skip excluded paths
        if self._is_excluded_path(path):
            return await call_next(request)

        # Get client info
        client_host = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent", "")
        user_id = getattr(request.state, "user_id", None)

        # Start timer
        start_time = time.time()

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            error = None
        except Exception as e:
            status_code = 500
            error = str(e)
            # Create a minimal error response
            from fastapi.responses import JSONResponse
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Determine status
        status_str = "success" if 200 <= status_code < 400 else "failure"

        # Log audit event
        logger.info(
            "HTTP request",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=round(duration_ms, 2),
            ip_address=client_host,
            user_id=user_id,
            user_agent=user_agent[:100] if user_agent else None,
            status=status_str,
        )

        # Add audit headers to response
        response.headers["X-Audit-Duration-Ms"] = str(round(duration_ms, 2))

        return response
