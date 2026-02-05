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


# =============================================================================
# PROMPT INJECTION DETECTION
# =============================================================================

# Prompt injection patterns for LLM security
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all)\s+instructions?",
    r"disregard\s+(previous|all)\s+instructions?",
    r"forget\s+(previous|all)\s+instructions?",
    r"(new|updated)\s+instructions?:?",
    r"override\s+instructions?",
    r"system\s*:\s*you\s+are",
    r"<\|(.*?)\|>",  # Special instruction delimiters
    r"<<.*?>>",  # Angle bracket delimiters
    r"\[SYSTEM\]:?",
    r"\[INSTRUCTIONS?\]:?",
    r"act\s+as\s+(a|an)?\s*.*?",
    r"pretend\s+(to\s+be|you\s+are)",
    r"roleplay\s+as",
    r"assume\s+the\s+role",
    r"you\s+are\s+now\s+(a|an)",
    r"from\s+now\s+on\s+you\s+are",
    r"i\s+want\s+you\s+to\s+act\s+as",
    r"you\s+must\s+(ignore|forget|disregard)",
    r"instead\s+of\s+.*\s+,",
    r"no\s+matter\s+what",
    r"above\s+instructions?\s+(are|were)",
    r"previous\s+(instructions?|text|prompts?)",
    r"<\|.*?\|>",  # Custom delimiters
]

# Compile all patterns
PROMPT_INJECTION_REGEX = re.compile(
    "|".join(f"(?:{pattern})" for pattern in PROMPT_INJECTION_PATTERNS),
    re.IGNORECASE | re.MULTILINE
)


def validate_prompt_injection(text: str) -> tuple[bool, list[str]]:
    """
    Detect prompt injection attempts in text.

    Args:
        text: Input text to check

    Returns:
        Tuple of (is_injection, list_of_patterns_found)
    """
    if not isinstance(text, str):
        return False, []

    threats = []
    matches = PROMPT_INJECTION_REGEX.finditer(text)

    for match in matches:
        matched_text = match.group(0)
        if matched_text not in threats:
            threats.append(matched_text)

    if threats:
        logger.warning(
            "Prompt injection detected",
            patterns_found=len(threats),
            text_preview=text[:100],
            threats=threats[:5],  # Log first 5
        )

    return len(threats) > 0, threats


def sanitize_search_query(
    query: str,
    max_length: int = 2000,
    check_prompt_injection: bool = True,
) -> str:
    """
    Sanitize search query with enhanced security checks.

    Args:
        query: Search query to sanitize
        max_length: Maximum allowed length
        check_prompt_injection: Whether to check for prompt injection

    Returns:
        Sanitized query

    Raises:
        ValueError: If query contains prompt injection
    """
    if not isinstance(query, str):
        return ""

    # Check for prompt injection
    if check_prompt_injection:
        is_injection, patterns = validate_prompt_injection(query)
        if is_injection:
            raise ValueError("Query contains suspicious content that may be an injection attempt")

    # Remove null bytes
    query = query.replace("\x00", "")

    # Normalize whitespace
    query = " ".join(query.split())

    # Truncate if too long
    if len(query) > max_length:
        logger.warning(
            "Query truncated",
            original_length=len(query),
            max_length=max_length,
        )
        query = query[:max_length]

    return query


# =============================================================================
# FILE UPLOAD VALIDATION
# =============================================================================

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".json", ".csv", ".md", ".markdown"}

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Suspicious filename patterns
SUSPICIOUS_FILENAME_PATTERNS = [
    re.compile(r"\.\."),  # Path traversal
    re.compile(r"\.php$", re.I),  # Web shells
    re.compile(r"\.jsp$", re.I),
    re.compile(r"\.asp$", re.I),
    re.compile(r"\.exe$", re.I),  # Executables
    re.compile(r"\.sh$", re.I),  # Shell scripts
    re.compile(r"\.bat$", re.I),
    re.compile(r"\.cmd$", re.I),
    re.compile(r"\.htaccess$", re.I),  # Apache config
    re.compile(r"\.htpasswd$", re.I),
]


def validate_file_upload(
    filename: str,
    file_size: int,
    content_type: str,
) -> tuple[bool, Optional[str]]:
    """
    Validate file upload for security.

    Args:
        filename: Name of the uploaded file
        file_size: Size in bytes
        content_type: MIME type

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    from pathlib import Path

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(
            "Invalid file extension",
            filename=filename,
            extension=ext,
        )
        return False, f"File type {ext} is not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"

    # Check file size
    if file_size > MAX_FILE_SIZE:
        logger.warning(
            "File too large",
            filename=filename,
            size=file_size,
            max_size=MAX_FILE_SIZE,
        )
        return False, f"File size exceeds maximum of {MAX_FILE_SIZE // (1024*1024)}MB"

    # Check for suspicious filenames
    filename_lower = filename.lower()
    for pattern in SUSPICIOUS_FILENAME_PATTERNS:
        if pattern.search(filename):
            logger.warning(
                "Suspicious filename detected",
                filename=filename,
                pattern=pattern.pattern,
            )
            return False, "Filename contains suspicious patterns"

    return True, None


# =============================================================================
# ARXIV QUERY VALIDATION
# =============================================================================

def validate_arxiv_query(query: str) -> tuple[bool, Optional[str]]:
    """
    Validate ArXiv search query with enhanced checks.

    Args:
        query: ArXiv query string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"

    if not isinstance(query, str):
        return False, "Query must be a string"

    # Check length
    if len(query) > 500:
        return False, "Query too long (max 500 characters)"

    # Check for balanced parentheses
    if query.count("(") != query.count(")"):
        return False, "Unbalanced parentheses in query"

    if query.count("[") != query.count("]"):
        return False, "Unbalanced brackets in query"

    # Check for balanced quotes
    if query.count('"') % 2 != 0:
        return False, "Unbalanced quotes in query"

    if query.count("'") % 2 != 0:
        return False, "Unbalanced single quotes in query"

    # Check for dangerous operators
    dangerous_patterns = [
        ("&&", "shell operators"),
        ("||", "shell operators"),
        (";", "command separator"),
        ("--", "SQL comment"),
        ("/*", "C-style comment"),
    ]

    for pattern, name in dangerous_patterns:
        if pattern in query:
            logger.warning(
                "Dangerous pattern in ArXiv query",
                pattern=pattern,
                description=name,
            )
            return False, f"Query contains invalid pattern: {name}"

    # Check for prompt injection
    is_injection, _ = validate_prompt_injection(query)
    if is_injection:
        return False, "Query contains suspicious patterns"

    return True, None


# =============================================================================
# VALIDATION PYDANTIC MODELS - ENHANCED
# =============================================================================

class EnhancedValidatedAskRequest(ValidatedAskRequest):
    """Enhanced validated request model for /ask endpoint with prompt injection detection."""

    @field_validator("question")
    @classmethod
    def validate_question_enhanced(cls, v: str) -> str:
        """Validate question with enhanced security checks."""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")

        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"Question too long (max {MAX_QUESTION_LENGTH} characters)")

        if len(v.strip()) < 3:
            raise ValueError("Question too short (min 3 characters)")

        # Check for prompt injection
        is_injection, patterns = validate_prompt_injection(v)
        if is_injection:
            logger.warning(
                "Prompt injection detected in question",
                patterns=patterns,
                question_preview=v[:100],
            )
            raise ValueError("Question contains invalid content that may be an injection attempt")

        # Check for other threats
        threats = detect_suspicious_patterns(v)
        if threats:
            logger.warning("Suspicious patterns detected", threats=threats, question=v[:100])
            raise ValueError(f"Question contains suspicious content: {', '.join(threats)}")

        return sanitize_string(v, MAX_QUESTION_LENGTH)


# =============================================================================
# VALIDATION MIDDLEWARE
# =============================================================================

class ValidationMiddleware:
    """
    Middleware for automatic request validation.

    Can be used with FastAPI to validate all incoming requests.
    """

    def __init__(
        self,
        enable_prompt_injection_check: bool = True,
        enable_file_validation: bool = True,
        max_body_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        """
        Initialize validation middleware.

        Args:
            enable_prompt_injection_check: Enable prompt injection detection
            enable_file_validation: Enable file upload validation
            max_body_size: Maximum request body size
        """
        self.enable_prompt_injection_check = enable_prompt_injection_check
        self.enable_file_validation = enable_file_validation
        self.max_body_size = max_body_size

        logger.info(
            "ValidationMiddleware initialized",
            prompt_injection_check=enable_prompt_injection_check,
            file_validation=enable_file_validation,
            max_body_size=max_body_size,
        )

    async def validate_request_body(
        self,
        body: dict,
        endpoint: str,
    ) -> tuple[bool, Optional[str], Optional[dict]]:
        """
        Validate request body.

        Args:
            body: Request body dictionary
            endpoint: Endpoint path for logging

        Returns:
            Tuple of (is_valid, error_message, sanitized_body)
        """
        if not isinstance(body, dict):
            return False, "Request body must be a JSON object", None

        # Check for prompt injection in string values
        if self.enable_prompt_injection_check:
            for key, value in body.items():
                if isinstance(value, str) and len(value) < 10000:  # Only check reasonable lengths
                    is_injection, _ = validate_prompt_injection(value)
                    if is_injection:
                        return False, f"Field '{key}' contains suspicious content", None

        # Sanitize body
        try:
            sanitized = sanitize_request_data(body)
            return True, None, sanitized
        except Exception as e:
            logger.error("Sanitization error", endpoint=endpoint, error=str(e))
            return False, f"Sanitization error: {str(e)}", None
