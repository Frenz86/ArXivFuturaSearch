"""
Security utilities for authentication and authorization.

Includes JWT token management, password hashing, and security helpers.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import secrets
import hashlib

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS = settings.REFRESH_TOKEN_EXPIRE_DAYS


class SecurityManager:
    """Centralized security utilities for JWT and password management."""

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            plain_password: Plain text password to verify
            hashed_password: Hashed password to compare against

        Returns:
            True if password matches
        """
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token.

        Args:
            data: Payload data to encode (typically user_id, roles, etc.)
            expires_delta: Custom expiration time

        Returns:
            Encoded JWT access token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({
            "exp": expire,
            "type": "access"
        })

        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def create_refresh_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT refresh token.

        Args:
            data: Payload data to encode
            expires_delta: Custom expiration time

        Returns:
            Encoded JWT refresh token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        to_encode.update({
            "exp": expire,
            "type": "refresh",
            "jti": secrets.token_urlsafe(32)  # Unique token ID for revocation
        })

        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def decode_token(token: str) -> Optional[Dict[str, Any]]:
        """
        Decode and validate JWT token.

        Args:
            token: JWT token to decode

        Returns:
            Decoded payload or None if invalid
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.warning("Token decode failed", error=str(e))
            return None

    @staticmethod
    def hash_token(token: str) -> str:
        """
        Hash a token for secure database storage.

        Args:
            token: Token to hash

        Returns:
            SHA256 hash of the token
        """
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def generate_reset_token() -> str:
        """
        Generate a secure password reset token.

        Returns:
            URL-safe random token
        """
        return secrets.token_urlsafe(32)

    @staticmethod
    def verify_reset_token(token: str, max_age_seconds: int = 3600) -> bool:
        """
        Verify a reset token's age (implementation depends on storage).

        Args:
            token: Reset token to verify
            max_age_seconds: Maximum age of token in seconds

        Returns:
            True if token is valid
        """
        # Implementation depends on how tokens are stored (DB, Redis, etc.)
        # This is a placeholder for the verification logic
        return True

    @staticmethod
    def generate_api_key() -> str:
        """
        Generate a secure API key.

        Returns:
            URL-safe random API key
        """
        return f"arx_{secrets.token_urlsafe(32)}"


def get_password_requirements() -> Dict[str, Any]:
    """
    Get current password requirements from settings.

    Returns:
        Dictionary with password requirements
    """
    return {
        "min_length": getattr(settings, 'PASSWORD_MIN_LENGTH', 8),
        "require_uppercase": getattr(settings, 'PASSWORD_REQUIRE_UPPERCASE', True),
        "require_lowercase": getattr(settings, 'PASSWORD_REQUIRE_LOWERCASE', True),
        "require_digit": getattr(settings, 'PASSWORD_REQUIRE_DIGIT', True),
        "require_special": getattr(settings, 'PASSWORD_REQUIRE_SPECIAL', True),
    }


def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """
    Validate password against security requirements.

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    reqs = get_password_requirements()
    errors = []

    if len(password) < reqs["min_length"]:
        errors.append(f"Password must be at least {reqs['min_length']} characters")

    if reqs["require_uppercase"] and not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")

    if reqs["require_lowercase"] and not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")

    if reqs["require_digit"] and not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit")

    if reqs["require_special"]:
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            errors.append("Password must contain at least one special character")

    return len(errors) == 0, errors
