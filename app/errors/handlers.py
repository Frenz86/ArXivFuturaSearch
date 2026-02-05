"""
Centralized error handling for the application.

Provides custom exceptions and error handlers for consistent error responses.
"""

from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class AppException(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class AuthenticationException(AppException):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_FAILED",
            details=details,
        )


class AuthorizationException(AppException):
    """User lacks permission for this action."""

    def __init__(self, message: str = "Permission denied", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_FAILED",
            details=details,
        )


class ValidationException(AppException):
    """Request validation failed."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=422,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class NotFoundException(AppException):
    """Resource not found."""

    def __init__(self, resource: str = "Resource", identifier: Optional[str] = None):
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"
        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND",
        )


class ConflictException(AppException):
    """Resource conflict (e.g., duplicate entry)."""

    def __init__(self, message: str = "Resource conflict", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=409,
            error_code="CONFLICT",
            details=details,
        )


class RateLimitException(AppException):
    """Rate limit exceeded."""

    def __init__(self, retry_after: Optional[int] = None):
        super().__init__(
            message="Rate limit exceeded",
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details={"retry_after": retry_after} if retry_after else {},
        )


class ExternalServiceException(AppException):
    """External service (LLM, vector store, etc.) error."""

    def __init__(
        self,
        service: str,
        message: str = "External service error",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"{service}: {message}",
            status_code=503,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service": service, **(details or {})},
        )


# =============================================================================
# ERROR FORMATTERS
# =============================================================================

def format_error_response(
    status_code: int,
    message: str,
    error_code: str = "ERROR",
    details: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Format standard error response."""
    response = {
        "error": {
            "message": message,
            "code": error_code,
            "status": status_code,
        }
    }

    if details:
        response["error"]["details"] = details

    if path:
        response["error"]["path"] = path

    return response


# =============================================================================
# ERROR HANDLERS
# =============================================================================

async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle custom application exceptions."""
    logger.warning(
        "Application error",
        error_code=exc.error_code,
        status=exc.status_code,
        message=exc.message,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=format_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            details=exc.details if exc.details else None,
            path=request.url.path,
        ),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.warning(
        "HTTP exception",
        status=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
    )

    # Map status codes to error codes
    error_codes = {
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "UNPROCESSABLE_ENTITY",
        429: "RATE_LIMIT_EXCEEDED",
        500: "INTERNAL_SERVER_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }

    error_code = error_codes.get(exc.status_code, "HTTP_ERROR")

    return JSONResponse(
        status_code=exc.status_code,
        content=format_error_response(
            status_code=exc.status_code,
            message=str(exc.detail),
            error_code=error_code,
            path=request.url.path,
        ),
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger.warning(
        "Validation error",
        errors=exc.errors(),
        path=request.url.path,
    )

    # Format validation errors
    formatted_errors = []
    for error in exc.errors():
        formatted_errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })

    return JSONResponse(
        status_code=422,
        content=format_error_response(
            status_code=422,
            message="Request validation failed",
            error_code="VALIDATION_ERROR",
            details={"errors": formatted_errors},
            path=request.url.path,
        ),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled exceptions."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        type=type(exc).__name__,
        path=request.url.path,
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content=format_error_response(
            status_code=500,
            message="An unexpected error occurred",
            error_code="INTERNAL_ERROR",
            path=request.url.path,
        ),
    )


# =============================================================================
# REGISTRATION
# =============================================================================

def setup_error_handlers(app: FastAPI) -> None:
    """Register all error handlers with the FastAPI app."""
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Error handlers registered")
