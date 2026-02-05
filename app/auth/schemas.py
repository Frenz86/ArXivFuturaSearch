"""
Pydantic schemas for authentication and authorization.

Request and response models for auth endpoints.
"""

from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class AuthProvider(str, Enum):
    """Authentication providers."""
    LOCAL = "local"
    GOOGLE = "google"
    GITHUB = "github"


# ==================== Request Schemas ====================

class LoginRequest(BaseModel):
    """Request schema for user login."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    remember_me: bool = False


class RegisterRequest(BaseModel):
    """Request schema for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    username: str = Field(..., min_length=3, max_length=50, pattern=r'^[a-zA-Z0-9_-]+$')
    full_name: Optional[str] = Field(None, max_length=255)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        from app.auth.security import validate_password_strength
        is_valid, errors = validate_password_strength(v)
        if not is_valid:
            raise ValueError("; ".join(errors))
        return v


class OAuthCallbackRequest(BaseModel):
    """Request schema for OAuth callback."""
    code: str
    state: Optional[str] = None
    provider: AuthProvider


class RefreshTokenRequest(BaseModel):
    """Request schema for token refresh."""
    refresh_token: str = Field(..., min_length=1)


class ChangePasswordRequest(BaseModel):
    """Request schema for password change."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Validate new password strength."""
        from app.auth.security import validate_password_strength
        is_valid, errors = validate_password_strength(v)
        if not is_valid:
            raise ValueError("; ".join(errors))
        return v


class ResetPasswordRequest(BaseModel):
    """Request schema for password reset."""
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)

    @field_validator("new_password")
    @classmethod
    def validate_reset_password(cls, v: str) -> str:
        """Validate reset password strength."""
        from app.auth.security import validate_password_strength
        is_valid, errors = validate_password_strength(v)
        if not is_valid:
            raise ValueError("; ".join(errors))
        return v


# ==================== Response Schemas ====================

class TokenResponse(BaseModel):
    """Response schema for authentication tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Seconds until expiration
    user: "UserResponse"


class UserResponse(BaseModel):
    """Response schema for user information."""
    id: str
    email: str
    username: str
    full_name: Optional[str]
    avatar_url: Optional[str]
    roles: List[str]
    provider: AuthProvider
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True


class PermissionResponse(BaseModel):
    """Response schema for permission information."""
    id: str
    name: str
    resource: str
    action: str
    description: Optional[str]

    class Config:
        from_attributes = True


class RoleResponse(BaseModel):
    """Response schema for role information."""
    id: str
    name: UserRole
    description: Optional[str]
    permissions: List[PermissionResponse]

    class Config:
        from_attributes = True


class SessionResponse(BaseModel):
    """Response schema for user session."""
    id: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str]
    user_agent: Optional[str]
    is_active: bool

    class Config:
        from_attributes = True


# ==================== Other Schemas ====================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    success: bool = True
