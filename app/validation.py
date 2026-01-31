"""Input validation and sanitization for security.


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

Provides utilities to validate and sanitize user input to prevent
injection attacks, DoS, and other security issues.
"""

import re
import html
from typing import Optional, Union
from pydantic import BaseModel, field_validator, Field

from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Maximum lengths for various inputs
MAX_QUESTION_LENGTH = 2000
MAX_QUERY_LENGTH = 500
MAX_FILTER_KEY_LENGTH = 100
MAX_FILTER_VALUE_LENGTH = 500

# Regex patterns for dangerous content
SQL_INJECTION_PATTERN = re.compile(
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)",
    re.IGNORECASE
)

XSS_PATTERN = re.compile(
    r'<script[^>]*>.*?</script>|javascript:|on\w+\s*=',
    re.IGNORECASE
)

PATH_TRAVERSAL_PATTERN = re.compile(r'\.\.[/\\]')

COMMAND_INJECTION_PATTERN = re.compile(
    r'[;&|`$()]|<\s*\?(?:php|perl|python|ruby|bash|sh)',
    re.IGNORECASE
)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def sanitize_string(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize a string input by removing dangerous content.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized string
    """
    if not isinstance(text, str):
        return ""

    # Trim whitespace
    text = text.strip()

    # Truncate to max length
    if max_length and len(text) > max_length:
        logger.warning("Input truncated", original_length=len(text), max_length=max_length)
        text = text[:max_length]

    # Remove null bytes
    text = text.replace('\x00', '')

    # Escape HTML entities
    text = html.escape(text)

    return text


def validate_question(question: str) -> tuple[bool, str, Optional[str]]:
    """
    Validate a user question for RAG queries.

    Args:
        question: User's question

    Returns:
        Tuple of (is_valid, error_message, sanitized_question)
    """
    if not question:
        return False, "Question cannot be empty", None

    if not isinstance(question, str):
        return False, "Question must be a string", None

    # Check length
    if len(question) > MAX_QUESTION_LENGTH:
        return False, f"Question too long (max {MAX_QUESTION_LENGTH} characters)", None

    if len(question.strip()) < 3:
        return False, "Question too short (min 3 characters)", None

    # Check for dangerous patterns
    if COMMAND_INJECTION_PATTERN.search(question):
        logger.warning("Command injection attempt detected", question=question[:100])
        return False, "Question contains invalid characters", None

    # Sanitize
    sanitized = sanitize_string(question, MAX_QUESTION_LENGTH)

    return True, None, sanitized


def validate_filters(filters: dict) -> tuple[bool, str, Optional[dict]]:
    """
    Validate metadata filters.

    Args:
        filters: Dictionary of filters

    Returns:
        Tuple of (is_valid, error_message, sanitized_filters)
    """
    if not filters:
        return True, None, {}

    if not isinstance(filters, dict):
        return False, "Filters must be a dictionary", None

    sanitized = {}
    for key, value in filters.items():
        # Validate key
        if not isinstance(key, str):
            return False, f"Filter key must be string, got {type(key).__name__}", None

        key = sanitize_string(key, MAX_FILTER_KEY_LENGTH)
        if not key:
            continue

        # Validate value
        if isinstance(value, str):
            value = sanitize_string(value, MAX_FILTER_VALUE_LENGTH)
        elif isinstance(value, (int, float, bool)):
            pass  # Keep as-is
        elif value is None:
            continue
        else:
            return False, f"Filter value must be string, number, or boolean, got {type(value).__name__}", None

        sanitized[key] = value

    return True, None, sanitized


def validate_top_k(top_k: int, min_k: int = 1, max_k: int = 20) -> tuple[bool, str, int]:
    """
    Validate top_k parameter.

    Args:
        top_k: Number of results requested
        min_k: Minimum allowed value
        max_k: Maximum allowed value

    Returns:
        Tuple of (is_valid, error_message, sanitized_value)
    """
    if not isinstance(top_k, int):
        try:
            top_k = int(top_k)
        except (ValueError, TypeError):
            return False, "top_k must be an integer", min_k

    if top_k < min_k:
        return False, f"top_k must be at least {min_k}", min_k

    if top_k > max_k:
        logger.warning("top_k exceeds maximum", requested=top_k, maximum=max_k)
        return False, f"top_k cannot exceed {max_k}", max_k

    return True, None, top_k


def detect_suspicious_patterns(text: str) -> list[str]:
    """
    Detect suspicious patterns in input text.

    Args:
        text: Text to analyze

    Returns:
        List of detected threat types (empty if safe)
    """
    threats = []

    if SQL_INJECTION_PATTERN.search(text):
        threats.append("sql_injection")

    if XSS_PATTERN.search(text):
        threats.append("xss")

    if PATH_TRAVERSAL_PATTERN.search(text):
        threats.append("path_traversal")

    if COMMAND_INJECTION_PATTERN.search(text):
        threats.append("command_injection")

    return threats


# =============================================================================
# PYDANTIC MODELS FOR VALIDATION
# =============================================================================

class ValidatedAskRequest(BaseModel):
    """Validated request model for /ask endpoint."""

    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[dict] = None
    stream: bool = False

    @field_validator("question")
    @classmethod
    def validate_question_field(cls, v: str) -> str:
        """Validate question field."""
        if not v or not v.strip():
            raise ValueError("question cannot be empty")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"question too long (max {MAX_QUESTION_LENGTH} characters)")
        if len(v.strip()) < 3:
            raise ValueError("question too short (min 3 characters)")

        # Check for threats
        threats = detect_suspicious_patterns(v)
        if threats:
            logger.warning("Suspicious patterns detected", threats=threats, question=v[:100])
            raise ValueError(f"Question contains suspicious content: {', '.join(threats)}")

        return v

    @field_validator("filters")
    @classmethod
    def validate_filters_field(cls, v: Optional[dict]) -> Optional[dict]:
        """Validate filters field."""
        if v is None:
            return None

        if not isinstance(v, dict):
            raise ValueError("filters must be a dictionary")

        # Validate each filter
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError("filter keys must be strings")
            if len(key) > MAX_FILTER_KEY_LENGTH:
                raise ValueError(f"filter key too long (max {MAX_FILTER_KEY_LENGTH})")
            if isinstance(value, str) and len(value) > MAX_FILTER_VALUE_LENGTH:
                raise ValueError(f"filter value too long (max {MAX_FILTER_VALUE_LENGTH})")

        return v


class ValidatedBuildRequest(BaseModel):
    """Validated request model for /build endpoint."""

    query: str
    max_results: int = Field(default=30, ge=1, le=200)

    @field_validator("query")
    @classmethod
    def validate_query_field(cls, v: str) -> str:
        """Validate arXiv query field."""
        if not v or not v.strip():
            raise ValueError("query cannot be empty")
        if len(v) > 500:
            raise ValueError("query too long (max 500 characters)")
        return v.strip()


# =============================================================================
# SANITIZATION MIDDLEWARE HELPERS
# =============================================================================

def sanitize_request_data(data: dict) -> dict:
    """
    Recursively sanitize all string values in request data.

    Args:
        data: Request data dictionary

    Returns:
        Sanitized data dictionary
    """
    if not isinstance(data, dict):
        return data

    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str):
            sanitized[key] = sanitize_string(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_request_data(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_string(item) if isinstance(item, str) else item
                for item in value
            ]
        else:
            sanitized[key] = value

    return sanitized


# =============================================================================
# LOGGING HELPERS
# =============================================================================

def log_validation_result(
    endpoint: str,
    is_valid: bool,
    error: Optional[str] = None,
    **context
):
    """Log validation result for monitoring."""
    if is_valid:
        logger.debug("Input validation passed", endpoint=endpoint, **context)
    else:
        logger.warning(
            "Input validation failed",
            endpoint=endpoint,
            error=error,
            **context
        )
