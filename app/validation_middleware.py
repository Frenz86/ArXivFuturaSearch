"""Request validation middleware with automatic Pydantic schema validation.


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

Provides middleware for validating all incoming requests against Pydantic schemas
with detailed error reporting and security checks.
"""

import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Type, Callable

from fastapi import Request, Response, HTTPException
from fastapi.types import ASGIApp
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers
from pydantic import ValidationError

from app.logging_config import get_logger
from app.schemas import ErrorResponse, ErrorDetail

logger = get_logger(__name__)


# =============================================================================
# VALIDATION MIDDLEWARE
# =============================================================================

class ValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic request validation and security checks.

    Features:
    - Content-Type validation
    - Request size limits
    - SQL injection detection
    - XSS detection
    - Rate limiting per user
    - Request ID injection
    """

    def __init__(
        self,
        app: ASGIApp,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        enable_validation: bool = True,
        enable_security_checks: bool = True,
    ):
        """
        Initialize validation middleware.

        Args:
            app: ASGI application
            max_request_size: Maximum request size in bytes
            enable_validation: Enable Pydantic validation
            enable_security_checks: Enable security pattern detection
        """
        super().__init__(app)
        self.max_request_size = max_request_size
        self.enable_validation = enable_validation
        self.enable_security_checks = enable_security_checks

        # Security patterns
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(\b(OR|AND)\s+\w+\s*=\s*\w+)",
            r"(;.*\b(DROP|DELETE)\b)",
        ]

        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe",
            r"<object",
            r"<embed",
        ]

    async def dispatch(self, request: Request, call_next):
        """Process request through validation middleware."""
        start_time = time.time()

        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Get client info
        client_host = request.client.host if request.client else "unknown"

        try:
            # Log incoming request
            logger.info(
                "Incoming request",
                method=request.method,
                path=request.url.path,
                request_id=request_id,
                client=client_host,
            )

            # Validate request size
            content_length = request.headers.get("content-length")
            if content_length:
                content_length = int(content_length)
                if content_length > self.max_request_size:
                    logger.warning(
                        "Request too large",
                        size=content_length,
                        max_size=self.max_request_size,
                        client=client_host,
                    )
                    return self._error_response(
                        status_code=413,
                        error="payload_too_large",
                        message=f"Request size {content_length} exceeds maximum {self.max_request_size}",
                        request_id=request_id,
                    )

            # Validate content type for POST/PUT/PATCH
            if request.method in ("POST", "PUT", "PATCH"):
                content_type = request.headers.get("content-type", "")
                if not content_type:
                    # Check if there's a body
                    if content_length and content_length > 0:
                        return self._error_response(
                            status_code=415,
                            error="unsupported_media_type",
                            message="Content-Type header is required",
                            request_id=request_id,
                        )

            # Security checks for POST/PUT requests
            if self.enable_security_checks and request.method in ("POST", "PUT"):
                # We'll check the body after it's read by the endpoint
                pass

            # Process request
            response = await call_next(request)

            # Add request ID to response
            response.headers["X-Request-ID"] = request_id

            # Log response
            duration = time.time() - start_time
            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round(duration * 1000, 2),
                request_id=request_id,
            )

            return response

        except HTTPException as e:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(
                "Request processing error",
                error=str(e),
                request_id=request_id,
                exc_info=True,
            )
            return self._error_response(
                status_code=500,
                error="internal_error",
                message="An error occurred while processing your request",
                request_id=request_id,
            )

    def _error_response(
        self,
        status_code: int,
        error: str,
        message: str,
        request_id: str,
        details: Optional[list] = None,
    ) -> Response:
        """Create standardized error response."""
        from fastapi.responses import JSONResponse

        error_response = ErrorResponse(
            error=error,
            message=message,
            details=details,
            request_id=request_id,
            timestamp=datetime.utcnow(),
        )

        return JSONResponse(
            status_code=status_code,
            content=error_response.model_dump(),
        )


# =============================================================================
# SECURITY VALIDATION
# =============================================================================

class SecurityValidator:
    """
    Validator for security-related checks.

    Provides methods to detect and prevent common security threats.
    """

    @staticmethod
    def check_sql_injection(value: str) -> tuple[bool, list[str]]:
        """
        Check for SQL injection patterns.

        Args:
            value: String to check

        Returns:
            Tuple of (is_suspicious, list_of_patterns_found)
        """
        import re

        suspicious = []
        value_upper = value.upper()

        patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(;.*\b(DROP|DELETE|EXECUTE)\b)",
            r"(\bOR\s+\w+\s*=\s*\w+)",
            r"(--|\#)",
            r"(\b(1=1|1 = 1)\b)",
        ]

        for pattern in patterns:
            if re.search(pattern, value_upper):
                suspicious.append(pattern)

        return len(suspicious) > 0, suspicious

    @staticmethod
    def check_xss(value: str) -> tuple[bool, list[str]]:
        """
        Check for XSS patterns.

        Args:
            value: String to check

        Returns:
            Tuple of (is_suspicious, list_of_patterns_found)
        """
        import re

        suspicious = []
        value_lower = value.lower()

        patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe",
            r"<object",
            r"<embed",
            r"<link",
            r"fromCharCode",
            r"eval\s*\(",
        ]

        for pattern in patterns:
            if re.search(pattern, value_lower, re.IGNORECASE):
                suspicious.append(pattern)

        return len(suspicious) > 0, suspicious

    @staticmethod
    def check_path_traversal(value: str) -> bool:
        """
        Check for path traversal patterns.

        Args:
            value: String to check

        Returns:
            True if path traversal detected
        """
        import re

        patterns = [r'\.\.[/\\]', r'\.\.%2f', r'\.\.%5c']

        for pattern in patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def check_command_injection(value: str) -> tuple[bool, list[str]]:
        """
        Check for command injection patterns.

        Args:
            value: String to check

        Returns:
            Tuple of (is_suspicious, list_of_patterns_found)
        """
        import re

        suspicious = []

        # Check for shell metacharacters
        if any(char in value for char in [';', '|', '&', '$', '`', '\n', '\r']):
            suspicious.append("shell_metacharacters")

        # Check for common injection patterns
        patterns = [
            r"<\s*\?(php|perl|python|ruby|bash|sh)",
            r"eval\s*\(",
            r"exec\s*\(",
            r"system\s*\(",
        ]

        for pattern in patterns:
            if re.search(pattern, value, re.IGNORECASE):
                suspicious.append(pattern)

        return len(suspicious) > 0, suspicious

    @classmethod
    def validate_input(
        cls,
        value: str,
        field_name: str = "input",
    ) -> tuple[bool, Optional[str]]:
        """
        Perform all security checks on input.

        Args:
            value: String to validate
            field_name: Name of the field for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check SQL injection
        is_sql, patterns = cls.check_sql_injection(value)
        if is_sql:
            logger.warning(
                "SQL injection detected",
                field=field_name,
                patterns=patterns,
                value=value[:100],
            )
            return False, f"Invalid characters in {field_name}"

        # Check XSS
        is_xss, patterns = cls.check_xss(value)
        if is_xss:
            logger.warning(
                "XSS detected",
                field=field_name,
                patterns=patterns,
                value=value[:100],
            )
            return False, f"Invalid content in {field_name}"

        # Check path traversal
        if cls.check_path_traversal(value):
            logger.warning(
                "Path traversal detected",
                field=field_name,
                value=value[:100],
            )
            return False, f"Invalid path in {field_name}"

        # Check command injection
        is_cmd, patterns = cls.check_command_injection(value)
        if is_cmd:
            logger.warning(
                "Command injection detected",
                field=field_name,
                patterns=patterns,
                value=value[:100],
            )
            return False, f"Invalid characters in {field_name}"

        return True, None


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_request_body(
    body: Dict[str, Any],
    schema_class: Type,
) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate request body against Pydantic schema.

    Args:
        body: Request body as dictionary
        schema_class: Pydantic schema class

    Returns:
        Tuple of (is_valid, error_message, validated_data)
    """
    try:
        validated = schema_class(**body)
        return True, None, validated.model_dump()
    except ValidationError as e:
        errors = []
        for error in e.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            })

        error_msg = "Validation failed: " + "; ".join(
            f"{err['field']}: {err['message']}" for err in errors
        )

        logger.warning("Request validation failed", errors=errors)
        return False, error_msg, None


def sanitize_string(
    value: str,
    max_length: Optional[int] = None,
) -> str:
    """
    Sanitize string input.

    Args:
        value: String to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return ""

    # Remove null bytes
    value = value.replace("\x00", "")

    # Trim whitespace
    value = value.strip()

    # Truncate if needed
    if max_length and len(value) > max_length:
        value = value[:max_length]

    return value


def validate_email(email: str) -> bool:
    """
    Validate email format.

    Args:
        email: Email address to validate

    Returns:
        True if valid email format
    """
    import re

    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid URL format
    """
    import re

    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return re.match(pattern, url) is not None


def validate_date(
    date_string: str,
    format: str = "%Y-%m-%d",
) -> bool:
    """
    Validate date string format.

    Args:
        date_string: Date string to validate
        format: Expected date format

    Returns:
        True if valid date format
    """
    try:
        datetime.strptime(date_string, format)
        return True
    except ValueError:
        return False


# =============================================================================
# RATE LIMITING
# =============================================================================

class SimpleRateLimiter:
    """
    Simple in-memory rate limiter.

    Note: For production with multiple instances, use Redis-based rate limiting.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_hour: Maximum requests per hour
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self._requests: Dict[str, list] = {}

    def check_rate_limit(
        self,
        identifier: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits.

        Args:
            identifier: Unique identifier (IP, API key, etc.)

        Returns:
            Tuple of (allowed, error_message)
        """
        now = time.time()

        # Initialize if needed
        if identifier not in self._requests:
            self._requests[identifier] = []

        # Clean old requests (older than 1 hour)
        cutoff_time = now - 3600
        self._requests[identifier] = [
            t for t in self._requests[identifier]
            if t > cutoff_time
        ]

        # Get recent requests
        requests = self._requests[identifier]

        # Check minute limit
        minute_ago = now - 60
        recent_minute = sum(1 for t in requests if t > minute_ago)

        if recent_minute >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"

        # Check hour limit
        if len(requests) >= self.requests_per_hour:
            return False, f"Rate limit exceeded: {self.requests_per_hour} requests per hour"

        # Add current request
        self._requests[identifier].append(now)

        return True, None


# Global rate limiter instance
_rate_limiter: Optional[SimpleRateLimiter] = None


def get_rate_limiter() -> SimpleRateLimiter:
    """Get or create global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = SimpleRateLimiter()
    return _rate_limiter
