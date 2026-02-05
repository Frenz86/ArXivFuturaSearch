"""
Authentication API endpoints.

Provides endpoints for login, registration, OAuth, and session management.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.schemas import (
    LoginRequest,
    RegisterRequest,
    OAuthCallbackRequest,
    TokenResponse,
    UserResponse,
    MessageResponse,
)
from app.auth.dependencies import (
    get_current_user,
    require_authenticated_user,
    require_admin,
)
from app.auth.service import AuthService
from app.database.base import User
from app.database.session import get_db
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse)
async def register(
    request_data: RegisterRequest,
    req: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user.

    Creates a new user account and returns authentication tokens.
    Password must meet security requirements.
    """
    service = AuthService(db)

    # Check if user exists
    existing = await service.get_user_by_email(request_data.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Hash password
    from app.auth.security import SecurityManager
    hashed_password = SecurityManager.hash_password(request_data.password)

    # Create user
    try:
        user = await service.create_user(
            email=request_data.email,
            password=hashed_password,
            username=request_data.username,
            full_name=request_data.full_name,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Log registration
    await service.log_audit_event(
        action="user.register",
        resource="user",
        resource_id=user.id,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("User-Agent"),
        user_id=user.id,
        success=True,
    )

    # Create tokens
    tokens = await service.create_user_session(
        user=user,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("User-Agent"),
    )

    return tokens


@router.post("/login", response_model=TokenResponse)
async def login(
    request_data: LoginRequest,
    req: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate user with email and password.

    Returns JWT tokens for authentication.
    """
    service = AuthService(db)

    user = await service.authenticate_user(
        email=request_data.email,
        password=request_data.password,
    )

    if not user:
        # Log failed attempt
        await service.log_audit_event(
            action="auth.login_failed",
            resource="auth",
            ip_address=req.client.host if req.client else None,
            user_agent=req.headers.get("User-Agent"),
            success=False,
            error_message="Invalid credentials",
            details={"email": request_data.email},
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    # Log successful login
    await service.log_audit_event(
        action="auth.login",
        resource="auth",
        user_id=user.id,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("User-Agent"),
        success=True,
    )

    # Create tokens
    tokens = await service.create_user_session(
        user=user,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("User-Agent"),
        remember_me=request_data.remember_me,
    )

    return tokens


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_token: str,
    req: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Refresh access token using refresh token.

    Use this to get a new access token before the current one expires.
    """
    service = AuthService(db)

    tokens = await service.refresh_session(refresh_token)

    if not tokens:
        # Log failed refresh
        await service.log_audit_event(
            action="auth.refresh_failed",
            resource="auth",
            ip_address=req.client.host if req.client else None,
            user_agent=req.headers.get("User-Agent"),
            success=False,
            error_message="Invalid refresh token",
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    # Log successful refresh
    await service.log_audit_event(
        action="auth.refresh",
        resource="auth",
        user_id=tokens.user.id,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("User-Agent"),
        success=True,
    )

    return tokens


@router.post("/logout", response_model=MessageResponse)
async def logout(
    req: Request,
    current_user: User = Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Logout current user and invalidate tokens.

    Revokes the current session token.
    """
    service = AuthService(db)

    # Get token from header
    auth_header = req.headers.get("Authorization")
    token = auth_header.split(" ")[1] if auth_header and auth_header.startswith("Bearer ") else None

    if token:
        await service.revoke_session(token)

    # Log logout
    await service.log_audit_event(
        action="auth.logout",
        resource="auth",
        user_id=current_user.id,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("User-Agent"),
        success=True,
    )

    return MessageResponse(message="Successfully logged out")


@router.get("/oauth/{provider}/authorize")
async def oauth_authorize(
    provider: str,
    req: Request,
):
    """
    Get OAuth authorization URL.

    Initiates OAuth flow by returning the authorization URL
    for the specified provider (google, github).
    """
    import secrets

    from app.auth.oauth import get_oauth_provider

    oauth_provider = get_oauth_provider(provider)
    state = secrets.token_urlsafe(32)

    # Store state in session/cache for verification (simplified here)
    # In production, store in Redis/database with expiration

    auth_url = await oauth_provider.get_auth_url(state)

    return {
        "authorization_url": auth_url,
        "state": state,
        "provider": provider,
    }


@router.post("/oauth/{provider}/callback", response_model=TokenResponse)
async def oauth_callback(
    provider: str,
    request_data: OAuthCallbackRequest,
    req: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Handle OAuth callback.

    Exchanges authorization code for tokens and creates/links user account.
    """
    from app.auth.oauth import get_oauth_provider

    service = AuthService(db)
    oauth_provider = get_oauth_provider(provider)

    # Exchange code for token
    token_data = await oauth_provider.exchange_code_for_token(request_data.code)

    # Get user info
    user_info = await oauth_provider.get_user_info(token_data["access_token"])

    # Find or create user
    try:
        user = await service.get_or_create_oauth_user(
            provider=provider,
            provider_id=user_info["id"],
            email=user_info["email"],
            username=user_info.get("login") or user_info["email"].split("@")[0],
            full_name=user_info.get("name"),
            avatar_url=user_info.get("picture") or user_info.get("avatar_url"),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Log OAuth login
    await service.log_audit_event(
        action=f"auth.oauth.{provider}",
        resource="auth",
        user_id=user.id,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("User-Agent"),
        success=True,
        details={"provider": provider},
    )

    # Create tokens
    tokens = await service.create_user_session(
        user=user,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("User-Agent"),
    )

    return tokens


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(require_authenticated_user),
):
    """
    Get current user information.

    Returns the authenticated user's profile.
    """
    return UserResponse.model_validate(current_user)


@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    current_password: str,
    new_password: str,
    req: Request,
    current_user: User = Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Change user password.

    Requires current password for verification.
    Only for local auth users (not OAuth).
    """
    from app.auth.security import SecurityManager, validate_password_strength

    # Check if user has local password
    if current_user.provider != "local" or current_user.hashed_password is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change password for OAuth users",
        )

    # Verify current password
    if not SecurityManager.verify_password(current_password, current_user.hashed_password):
        await service.log_audit_event(
            action="auth.password_change_failed",
            resource="user",
            resource_id=current_user.id,
            user_id=current_user.id,
            ip_address=req.client.host if req.client else None,
            user_agent=req.headers.get("User-Agent"),
            success=False,
            error_message="Invalid current password",
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid current password",
        )

    # Validate new password
    is_valid, errors = validate_password_strength(new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="; ".join(errors),
        )

    # Update password
    service = AuthService(db)
    current_user.hashed_password = SecurityManager.hash_password(new_password)
    await db.commit()

    # Log password change
    await service.log_audit_event(
        action="auth.password_changed",
        resource="user",
        resource_id=current_user.id,
        user_id=current_user.id,
        ip_address=req.client.host if req.client else None,
        user_agent=req.headers.get("User-Agent"),
        success=True,
    )

    return MessageResponse(message="Password changed successfully")
