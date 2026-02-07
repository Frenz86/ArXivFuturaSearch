"""
FastAPI dependencies for authentication and authorization.

Provides dependency injection functions for protecting routes
and accessing current user information.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.auth.security import SecurityManager
from app.database.session import get_db
from app.database.base import User, UserSession
from app.logging_config import get_logger

logger = get_logger(__name__)

# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Get current authenticated user from JWT token.

    Returns None if not authenticated (optional auth).
    Use require_authenticated_user for mandatory authentication.

    Args:
        credentials: HTTP Bearer credentials from Authorization header
        db: Database session

    Returns:
        User object or None if not authenticated

    Example:
        @app.get("/profile")
        async def get_profile(user: Optional[User] = Depends(get_current_user)):
            if user:
                return {"user": user.email}
            return {"user": None}
    """
    if credentials is None:
        return None

    token = credentials.credentials
    payload = SecurityManager.decode_token(token)

    if payload is None or payload.get("type") != "access":
        return None

    user_id = payload.get("sub")
    if user_id is None:
        return None

    # Check if session is valid (not revoked, not expired)
    token_hash = SecurityManager.hash_token(token)
    result = await db.execute(
        select(UserSession).where(
            UserSession.token_hash == token_hash,
            UserSession.revoked == False,
        )
    )
    session = result.scalar_one_or_none()

    if session is None:
        logger.warning("Invalid or revoked session token", user_id=user_id)
        return None

    # Check if session expired
    from datetime import datetime, UTC
    if session.expires_at < datetime.now(UTC):
        logger.warning("Expired session token", user_id=user_id)
        return None

    # Get user
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        return None

    return user


async def require_authenticated_user(
    current_user: Optional[User] = Depends(get_current_user),
) -> User:
    """
    Require authenticated user (raises 401 if not authenticated).

    Use this dependency for routes that require authentication.

    Args:
        current_user: Current user from get_current_user

    Returns:
        Authenticated User object

    Raises:
        HTTPException: 401 if not authenticated

    Example:
        @app.get("/protected")
        async def protected_route(user: User = Depends(require_authenticated_user)):
            return {"message": f"Hello {user.username}"}
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


class RoleChecker:
    """
    Dependency for checking user roles.

    Checks if the current user has one of the required roles.

    Example:
        @app.get("/admin")
        async def admin_route(user: User = Depends(RoleChecker(["admin"]))):
            return {"message": "Welcome admin"}
    """

    def __init__(self, allowed_roles: list[str]):
        """
        Initialize role checker.

        Args:
            allowed_roles: List of role names that are allowed
        """
        self.allowed_roles = allowed_roles

    def __call__(
        self,
        current_user: User = Depends(require_authenticated_user),
    ) -> User:
        """
        Check if user has required role.

        Args:
            current_user: Current authenticated user

        Returns:
            User if authorized

        Raises:
            HTTPException: 403 if user lacks required role
        """
        user_roles = [role.name for role in current_user.roles]

        if not any(role in self.allowed_roles for role in user_roles):
            logger.warning(
                "Access denied: insufficient role permissions",
                user_id=current_user.id,
                user_roles=user_roles,
                required_roles=self.allowed_roles,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {', '.join(self.allowed_roles)}",
            )

        return current_user


class PermissionChecker:
    """
    Dependency for checking user permissions.

    Checks if the current user has the required permission
    through their roles.

    Example:
        @app.post("/index/build")
        async def build_index(user: User = Depends(PermissionChecker("index:write"))):
            ...
    """

    def __init__(self, required_permission: str):
        """
        Initialize permission checker.

        Args:
            required_permission: Permission name required (e.g., "search:read")
        """
        self.required_permission = required_permission

    def __call__(
        self,
        current_user: User = Depends(require_authenticated_user),
    ) -> User:
        """
        Check if user has required permission.

        Args:
            current_user: Current authenticated user

        Returns:
            User if authorized

        Raises:
            HTTPException: 403 if user lacks required permission
        """
        # Collect all permissions from user's roles
        permissions = set()
        for role in current_user.roles:
            for perm in role.permissions:
                permissions.add(perm.name)

        if self.required_permission not in permissions:
            logger.warning(
                "Access denied: missing permission",
                user_id=current_user.id,
                required_permission=self.required_permission,
                user_permissions=list(permissions),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {self.required_permission}",
            )

        return current_user


# ==================== Common Role Checkers ====================

# Admin only routes
require_admin = RoleChecker(["admin"])

# User or admin routes
require_user = RoleChecker(["admin", "user"])

# Any authenticated user (guest, user, admin)
require_guest = RoleChecker(["admin", "user", "guest"])

# ==================== Common Permission Checkers ====================

# Search permissions
require_search_read = PermissionChecker("search:read")

# Index permissions
require_index_read = PermissionChecker("index:read")
require_index_write = PermissionChecker("index:write")
require_index_delete = PermissionChecker("index:delete")

# Admin permissions
require_admin_access = PermissionChecker("admin:all")

# User management
require_user_read = PermissionChecker("user:read")
require_user_write = PermissionChecker("user:write")
require_user_delete = PermissionChecker("user:delete")

# Audit log access
require_audit_read = PermissionChecker("audit:read")


# ==================== Optional Auth ====================

async def get_optional_user(
    current_user: Optional[User] = Depends(get_current_user),
) -> Optional[User]:
    """
    Alias for get_current_user for clarity.

    Returns None for unauthenticated users (guest access).
    Useful for routes that work for both authenticated and guest users.

    Example:
        @app.get("/search")
        async def search(user: Optional[User] = Depends(get_optional_user)):
            user_id = user.id if user else "guest"
            ...
    """
    return current_user
