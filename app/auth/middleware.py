"""
Authentication middleware for FastAPI.

Provides optional or required authentication middleware for route protection.
"""

from typing import Optional, Set
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.auth.security import SecurityManager
from app.logging_config import get_logger

logger = get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for route protection.

    Can be configured to require authentication or allow optional authentication.
    Excludes certain paths from authentication requirements (health, docs, etc.).

    Example:
        app.add_middleware(
            AuthMiddleware,
            require_auth=False,  # Optional auth by default
            excluded_paths={"/health", "/docs", "/openapi.json", "/api/auth/login"}
        )
    """

    def __init__(
        self,
        app: ASGIApp,
        require_auth: bool = False,
        excluded_paths: Optional[Set[str]] = None,
        excluded_prefixes: Optional[Set[str]] = None,
    ):
        """
        Initialize authentication middleware.

        Args:
            app: ASGI application
            require_auth: If True, all routes require auth unless excluded
            excluded_paths: Set of exact paths to exclude from auth
            excluded_prefixes: Set of path prefixes to exclude from auth
        """
        super().__init__(app)
        self.require_auth = require_auth

        # Default excluded paths
        self.excluded_paths = excluded_paths or {
            "/health",
            "/healthz",
            "/readiness",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/auth/login",
            "/api/auth/register",
            "/api/auth/oauth",
        }

        # Excluded path prefixes (for dynamic routes)
        self.excluded_prefixes = excluded_prefixes or {
            "/api/auth/oauth/",
            "/static/",
            "/favicon",
        }

        logger.info(
            "AuthMiddleware initialized",
            require_auth=require_auth,
            excluded_paths=len(self.excluded_paths),
            excluded_prefixes=len(self.excluded_prefixes),
        )

    def _is_excluded_path(self, path: str) -> bool:
        """
        Check if path is excluded from authentication.

        Args:
            path: Request path

        Returns:
            True if path is excluded
        """
        # Check exact matches
        if path in self.excluded_paths:
            return True

        # Check prefix matches
        for prefix in self.excluded_prefixes:
            if path.startswith(prefix):
                return True

        return False

    async def dispatch(self, request: Request, call_next):
        """
        Process request through authentication middleware.

        Validates JWT token and stores user info in request state.
        """
        path = request.url.path

        # Skip excluded paths
        if self._is_excluded_path(path):
            return await call_next(request)

        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")

        token = None
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

        # Decode token and validate
        user_id = None
        user_roles = []

        if token:
            payload = SecurityManager.decode_token(token)

            if payload is not None:
                token_type = payload.get("type")

                if token_type != "access":
                    logger.warning(
                        "Invalid token type",
                        path=path,
                        token_type=token_type,
                    )
                else:
                    user_id = payload.get("sub")
                    user_roles = payload.get("roles", [])

                    # Store in request state for access in endpoints
                    request.state.user_id = user_id
                    request.state.user_roles = user_roles
                    request.state.authenticated = True

                    logger.debug(
                        "User authenticated via middleware",
                        user_id=user_id,
                        path=path,
                    )

                    return await call_next(request)

        # No valid token found
        if self.require_auth:
            logger.warning(
                "Authentication required but not provided",
                path=path,
                ip_address=request.client.host if request.client else None,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Optional auth - continue without user
        request.state.authenticated = False
        return await call_next(request)
