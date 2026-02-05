"""
Authentication service with business logic for user management.

Handles user registration, login, OAuth, sessions, and audit logging.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import secrets

from app.auth.schemas import (
    RegisterRequest, TokenResponse, UserResponse,
)
from app.auth.security import SecurityManager
from app.database.base import User, Role, UserSession, AuditLog
from app.logging_config import get_logger

logger = get_logger(__name__)


class AuthService:
    """
    Service for authentication operations.

    Handles user registration, authentication, session management,
    and audit logging.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize auth service.

        Args:
            db: Database session
        """
        self.db = db

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User object or None
        """
        result = await self.db.execute(
            select(User)
            .options(selectinload(User.roles).selectinload(Role.permissions))
            .where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email.

        Args:
            email: User email

        Returns:
            User object or None
        """
        result = await self.db.execute(
            select(User)
            .options(selectinload(User.roles).selectinload(Role.permissions))
            .where(User.email == email.lower())
        )
        return result.scalar_one_or_none()

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.

        Args:
            username: Username

        Returns:
            User object or None
        """
        result = await self.db.execute(
            select(User)
            .options(selectinload(User.roles).selectinload(Role.permissions))
            .where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def create_user(
        self,
        email: str,
        password: Optional[str],
        username: str,
        full_name: Optional[str] = None,
        provider: str = "local",
        provider_id: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> User:
        """
        Create a new user.

        Args:
            email: User email
            password: Hashed password (None for OAuth users)
            username: Username
            full_name: Full name
            provider: Auth provider (local, google, github)
            provider_id: Provider's user ID
            avatar_url: Profile picture URL

        Returns:
            Created User object
        """
        # Check if user already exists
        existing = await self.get_user_by_email(email)
        if existing:
            raise ValueError(f"User with email {email} already exists")

        existing_username = await self.get_user_by_username(username)
        if existing_username:
            raise ValueError(f"Username {username} already taken")

        # Create user
        user = User(
            email=email.lower(),
            username=username,
            hashed_password=password,
            full_name=full_name,
            avatar_url=avatar_url,
            provider=provider,
            provider_id=provider_id,
            is_active=True,
            is_verified=provider != "local",  # OAuth users are pre-verified
        )

        # Assign default role
        guest_role = await self._get_role_by_name("user")
        if guest_role:
            user.roles.append(guest_role)
        else:
            logger.warning("Default 'user' role not found, creating it")
            user.roles.append(await self._create_default_role("user"))

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        logger.info(
            "User created",
            user_id=user.id,
            email=email,
            username=username,
            provider=provider,
        )

        return user

    async def authenticate_user(
        self,
        email: str,
        password: str,
    ) -> Optional[User]:
        """
        Authenticate user with email and password.

        Args:
            email: User email
            password: Plain text password

        Returns:
            User object if authentication successful, None otherwise
        """
        user = await self.get_user_by_email(email)

        if user is None:
            return None

        if not user.is_active:
            logger.warning("Authentication attempt for inactive user", email=email)
            return None

        # Local users must have a password
        if user.provider != "local" or user.hashed_password is None:
            logger.warning("Password authentication attempted for OAuth user", email=email)
            return None

        if not SecurityManager.verify_password(password, user.hashed_password):
            return None

        # Update last login
        user.last_login = datetime.utcnow()
        await self.db.commit()

        return user

    async def create_user_session(
        self,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        remember_me: bool = False,
    ) -> TokenResponse:
        """
        Create user session with JWT tokens.

        Args:
            user: User object
            ip_address: Client IP address
            user_agent: Client user agent
            remember_me: Whether to extend token lifetime

        Returns:
            TokenResponse with access and refresh tokens
        """
        # Get user roles
        roles = [role.name for role in user.roles]

        # Create access token
        access_expire = timedelta(hours=24 if remember_me else 1)
        access_token = SecurityManager.create_access_token(
            data={
                "sub": user.id,
                "email": user.email,
                "roles": roles,
            },
            expires_delta=access_expire,
        )

        # Create refresh token
        refresh_expire = timedelta(days=30 if remember_me else 7)
        refresh_token = SecurityManager.create_refresh_token(
            data={"sub": user.id},
            expires_delta=refresh_expire,
        )

        # Store session in database
        session = UserSession(
            user_id=user.id,
            token_hash=SecurityManager.hash_token(access_token),
            refresh_token_hash=SecurityManager.hash_token(refresh_token),
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.utcnow() + refresh_expire,
        )

        self.db.add(session)
        await self.db.commit()

        logger.info(
            "User session created",
            user_id=user.id,
            ip_address=ip_address,
            remember_me=remember_me,
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=int(access_expire.total_seconds()),
            user=UserResponse.model_validate(user),
        )

    async def refresh_session(
        self,
        refresh_token: str,
    ) -> Optional[TokenResponse]:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            New TokenResponse or None if refresh token invalid
        """
        # Decode refresh token
        payload = SecurityManager.decode_token(refresh_token)
        if payload is None or payload.get("type") != "refresh":
            return None

        user_id = payload.get("sub")
        if user_id is None:
            return None

        # Check refresh token in database
        token_hash = SecurityManager.hash_token(refresh_token)
        result = await self.db.execute(
            select(UserSession).where(
                UserSession.refresh_token_hash == token_hash,
                UserSession.revoked == False,
            )
        )
        session = result.scalar_one_or_none()

        if session is None:
            return None

        # Check expiration
        if session.expires_at < datetime.utcnow():
            return None

        # Get user
        user = await self.get_user_by_id(user_id)
        if user is None or not user.is_active:
            return None

        # Create new tokens (revoke old session)
        session.revoked = True
        await self.db.commit()

        # Create new session
        return await self.create_user_session(
            user=user,
            ip_address=session.ip_address,
            user_agent=session.user_agent,
        )

    async def revoke_session(
        self,
        access_token: str,
    ) -> bool:
        """
        Revoke user session.

        Args:
            access_token: Access token to revoke

        Returns:
            True if session was revoked
        """
        token_hash = SecurityManager.hash_token(access_token)
        result = await self.db.execute(
            select(UserSession).where(
                UserSession.token_hash == token_hash,
            )
        )
        session = result.scalar_one_or_none()

        if session is None:
            return False

        session.revoked = True
        await self.db.commit()

        logger.info("Session revoked", session_id=session.id)
        return True

    async def get_or_create_oauth_user(
        self,
        provider: str,
        provider_id: str,
        email: str,
        username: str,
        full_name: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> User:
        """
        Get or create OAuth user.

        Args:
            provider: Provider name (google, github)
            provider_id: Provider's user ID
            email: User email
            username: Username
            full_name: Full name
            avatar_url: Profile picture URL

        Returns:
            User object
        """
        # Check for existing user by provider ID
        result = await self.db.execute(
            select(User).where(
                User.provider == provider,
                User.provider_id == provider_id,
            )
        )
        user = result.scalar_one_or_none()

        if user:
            # Update user info
            if avatar_url:
                user.avatar_url = avatar_url
            if full_name:
                user.full_name = full_name
            await self.db.commit()
            return user

        # Check if email is already used by another account
        existing = await self.get_user_by_email(email)
        if existing:
            # Link OAuth to existing account
            existing.provider = provider
            existing.provider_id = provider_id
            existing.is_verified = True
            if avatar_url:
                existing.avatar_url = avatar_url
            await self.db.commit()
            logger.info(
                "OAuth linked to existing account",
                user_id=existing.id,
                provider=provider,
            )
            return existing

        # Create new user
        return await self.create_user(
            email=email,
            password=None,  # OAuth users don't have passwords
            username=username,
            full_name=full_name,
            provider=provider,
            provider_id=provider_id,
            avatar_url=avatar_url,
        )

    async def log_audit_event(
        self,
        action: str,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        user_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditLog:
        """
        Record an audit event.

        Args:
            action: Action performed (auth.login, search.query, etc.)
            resource: Resource type (auth, search, index, etc.)
            resource_id: Resource ID
            details: Additional details as JSON
            ip_address: Client IP address
            user_agent: Client user agent
            user_id: User ID (None for guest)
            success: Whether the action succeeded
            error_message: Error message if failed
            correlation_id: Request correlation ID

        Returns:
            Created AuditLog
        """
        log_entry = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            correlation_id=correlation_id,
        )

        self.db.add(log_entry)
        await self.db.commit()

        logger.debug(
            "Audit log recorded",
            action=action,
            user_id=user_id,
            resource=resource,
            success=success,
        )

        return log_entry

    # ==================== Helper Methods ====================

    async def _get_role_by_name(self, name: str) -> Optional[Role]:
        """Get role by name."""
        result = await self.db.execute(
            select(Role).where(Role.name == name)
        )
        return result.scalar_one_or_none()

    async def _create_default_role(self, name: str) -> Role:
        """Create a default role with basic permissions."""
        role = Role(
            name=name,
            description=f"Default {name} role",
        )

        # Add basic permissions based on role
        if name == "admin":
            # Admin gets all permissions (to be implemented)
            pass
        elif name == "user":
            # Regular user can search and read
            pass
        elif name == "guest":
            # Guest has limited permissions
            pass

        self.db.add(role)
        await self.db.commit()
        return role
