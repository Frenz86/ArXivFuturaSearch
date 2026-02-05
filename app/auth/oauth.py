"""
OAuth2 authentication providers for Google and GitHub.

Handles OAuth2 flow for external authentication providers.
"""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import httpx
from fastapi import HTTPException

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class OAuthProvider(ABC):
    """
    Base OAuth provider class.

    Abstract base class for OAuth2 authentication providers.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: list[str],
    ):
        """
        Initialize OAuth provider.

        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret
            redirect_uri: OAuth redirect URI
            scopes: List of OAuth scopes to request
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scopes = scopes
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    @abstractmethod
    async def get_auth_url(self, state: Optional[str] = None) -> str:
        """Get the authorization URL for OAuth flow."""
        pass

    @abstractmethod
    async def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        pass

    @abstractmethod
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from provider."""
        pass

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class GoogleOAuthProvider(OAuthProvider):
    """
    Google OAuth2 provider.

    Implements OAuth2 flow for Google authentication.
    """

    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    USER_INFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

    async def get_auth_url(self, state: Optional[str] = None) -> str:
        """
        Get Google OAuth authorization URL.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Authorization URL
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
            "response_type": "code",
            "access_type": "offline",
            "state": state or "",
        }

        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.AUTH_URL}?{param_str}"

    async def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access token.

        Args:
            code: Authorization code from OAuth callback

        Returns:
            Token response dict with access_token, refresh_token, etc.
        """
        client = await self._get_client()

        response = await client.post(
            self.TOKEN_URL,
            data={
                "code": code,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code",
            },
        )

        if response.status_code != 200:
            logger.error(
                "Google token exchange failed",
                status=response.status_code,
                text=response.text,
            )
            raise HTTPException(status_code=400, detail="Failed to exchange authorization code")

        return response.json()

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        Get user info from Google.

        Args:
            access_token: OAuth access token

        Returns:
            User info dict with id, email, name, picture, etc.
        """
        client = await self._get_client()

        response = await client.get(
            self.USER_INFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if response.status_code != 200:
            logger.error(
                "Google user info failed",
                status=response.status_code,
                text=response.text,
            )
            raise HTTPException(status_code=400, detail="Failed to get user information")

        data = response.json()

        # Normalize user info
        return {
            "id": data.get("id"),
            "email": data.get("email"),
            "name": data.get("name"),
            "given_name": data.get("given_name"),
            "family_name": data.get("family_name"),
            "picture": data.get("picture"),
            "verified_email": data.get("verified_email", False),
        }


class GitHubOAuthProvider(OAuthProvider):
    """
    GitHub OAuth2 provider.

    Implements OAuth2 flow for GitHub authentication.
    """

    AUTH_URL = "https://github.com/login/oauth/authorize"
    TOKEN_URL = "https://github.com/login/oauth/access_token"
    USER_INFO_URL = "https://api.github.com/user"
    USER_EMAIL_URL = "https://api.github.com/user/emails"

    async def get_auth_url(self, state: Optional[str] = None) -> str:
        """
        Get GitHub OAuth authorization URL.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Authorization URL
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": ",".join(self.scopes),
            "response_type": "code",
            "state": state or "",
        }

        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.AUTH_URL}?{param_str}"

    async def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access token.

        Args:
            code: Authorization code from OAuth callback

        Returns:
            Token response dict with access_token, etc.
        """
        client = await self._get_client()

        response = await client.post(
            self.TOKEN_URL,
            data={
                "code": code,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": self.redirect_uri,
            },
            headers={"Accept": "application/json"},
        )

        if response.status_code != 200:
            logger.error(
                "GitHub token exchange failed",
                status=response.status_code,
                text=response.text,
            )
            raise HTTPException(status_code=400, detail="Failed to exchange authorization code")

        return response.json()

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        Get user info from GitHub.

        Args:
            access_token: OAuth access token

        Returns:
            User info dict with id, email, name, avatar_url, etc.
        """
        client = await self._get_client()

        # Get basic user info
        response = await client.get(
            self.USER_INFO_URL,
            headers={"Authorization": f"token {access_token}"},
        )

        if response.status_code != 200:
            logger.error(
                "GitHub user info failed",
                status=response.status_code,
                text=response.text,
            )
            raise HTTPException(status_code=400, detail="Failed to get user information")

        data = response.json()

        # Get email separately (GitHub requires explicit scope)
        email_response = await client.get(
            self.USER_EMAIL_URL,
            headers={"Authorization": f"token {access_token}"},
        )

        email = None
        verified_email = False
        if email_response.status_code == 200:
            emails = email_response.json()
            # Find primary verified email
            for e in emails:
                if e.get("primary", False) and e.get("verified", False):
                    email = e.get("email")
                    verified_email = True
                    break
            # Fallback to first email
            if not email and emails:
                email = emails[0].get("email")

        # Normalize user info
        return {
            "id": str(data.get("id")),
            "email": email or data.get("email"),  # GitHub may not provide email without scope
            "login": data.get("login"),
            "name": data.get("name") or data.get("login"),
            "avatar_url": data.get("avatar_url"),
            "verified_email": verified_email,
        }


# Provider factory
def get_oauth_provider(provider: str) -> OAuthProvider:
    """
    Get OAuth provider instance by name.

    Args:
        provider: Provider name ('google' or 'github')

    Returns:
        OAuthProvider instance

    Raises:
        HTTPException: If provider is not supported
    """
    providers = {
        "google": GoogleOAuthProvider(
            client_id=getattr(settings, 'GOOGLE_CLIENT_ID', ''),
            client_secret=getattr(settings, 'GOOGLE_CLIENT_SECRET', ''),
            redirect_uri=getattr(settings, 'GOOGLE_REDIRECT_URI', 'http://localhost:8000/api/auth/oauth/google/callback'),
            scopes=["openid", "email", "profile"],
        ),
        "github": GitHubOAuthProvider(
            client_id=getattr(settings, 'GITHUB_CLIENT_ID', ''),
            client_secret=getattr(settings, 'GITHUB_CLIENT_SECRET', ''),
            redirect_uri=getattr(settings, 'GITHUB_REDIRECT_URI', 'http://localhost:8000/api/auth/oauth/github/callback'),
            scopes=["user:email"],
        ),
    }

    if provider not in providers:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported OAuth provider: {provider}",
        )

    return providers[provider]
