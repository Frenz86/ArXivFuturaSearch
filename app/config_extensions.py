"""
Configuration extensions for new features.

Add these settings to your app/config.py or import them separately.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecuritySettings(BaseSettings):
    """Security-related settings."""
    model_config = SettingsConfigDict(env_prefix="", env_file=".env")

    # JWT Settings
    SECRET_KEY: str = Field(default="changeme-secret-key")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Password Requirements
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True

    # OAuth Settings
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/auth/oauth/google/callback"

    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""
    GITHUB_REDIRECT_URI: str = "http://localhost:8000/api/auth/oauth/github/callback"

    # Audit Settings
    AUDIT_LOG_RETENTION_DAYS: int = 90


class PerformanceSettings(BaseSettings):
    """Performance-related settings."""
    model_config = SettingsConfigDict(env_prefix="", env_file=".env")

    # Batch Processing
    BATCH_PROCESSING_STRATEGY: str = "thread_parallel"  # sequential, thread_parallel, process_parallel, gpu_batched
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_MAX_WORKERS: int = 0  # 0 = auto-detect
    EMBEDDING_ENABLE_GPU: bool = True
    EMBEDDING_RETRY_FAILED: bool = True
    EMBEDDING_MAX_RETRIES: int = 3

    # Connection Pooling
    POSTGRES_ENABLE_ADAPTIVE_POOL: bool = True
    POSTGRES_POOL_ADAPTIVE_SIZING: bool = True
    POSTGRES_POOL_SCALE_UP_THRESHOLD: float = 0.75
    POSTGRES_POOL_SCALE_DOWN_THRESHOLD: float = 0.25
    POSTGRES_POOL_ADJUSTMENT_INTERVAL: int = 60
    POSTGRES_POOL_HEALTH_CHECK_INTERVAL: int = 30

    # Semantic Caching
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.85
    SEMANTIC_CACHE_MAX_ENTRIES: int = 1000
    SEMANTIC_CACHE_STRATEGY: str = "hybrid"  # exact_match, semantic_similarity, hybrid
    SEMANTIC_CACHE_PRECOMPUTED_EMBEDDINGS: bool = True

    # Cache Warming
    CACHE_WARMING_ENABLED: bool = True
    CACHE_WARMING_STATIC: bool = True
    CACHE_WARMING_CONCURRENCY: int = 5


class FeatureSettings(BaseSettings):
    """Feature-related settings."""
    model_config = SettingsConfigDict(env_prefix="", env_file=".env")

    # Conversation Memory
    CONVERSATION_ENABLED: bool = True
    MAX_CONTEXT_TOKENS: int = 8000
    CONVERSATION_SUMMARIZATION_THRESHOLD: int = 10

    # Alerts
    ALERTS_ENABLED: bool = True
    ALERT_CHECK_INTERVAL_MINUTES: int = 60

    # Collections
    COLLECTIONS_ENABLED: bool = True
    PUBLIC_COLLECTIONS_ENABLED: bool = True
    SHARE_LINK_EXPIRY_HOURS: int = 168  # 1 week

    # Export
    EXPORT_ENABLED: bool = True
    EXPORT_TEMP_DIR: str = "data/exports"
    EXPORT_MAX_SIZE_MB: int = 50


class DatabaseSettings(BaseSettings):
    """Database settings."""
    model_config = SettingsConfigDict(env_prefix="", env_file=".env")

    # PostgreSQL (for auth, conversations, etc.)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/arxiv_rag"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "arxiv_rag"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_POOL_SIZE: int = 10
    POSTGRES_MAX_OVERFLOW: int = 20
    POSTGRES_POOL_TIMEOUT: int = 30
    POSTGRES_POOL_RECYCLE: int = 3600

    # Database Echo (for debugging)
    DATABASE_ECHO: bool = False


# =============================================================================
# EXAMPLE: INTEGRATION WITH EXISTING CONFIG
# =============================================================================
#
# Add these to your existing app/config.py Settings class:
#
# class Settings(BaseSettings):
#     # ... existing settings ...
#
#     # Security
#     SECRET_KEY: str = "changeme-secret-key"
#     ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
#     REFRESH_TOKEN_EXPIRE_DAYS: int = 7
#     GOOGLE_CLIENT_ID: str = ""
#     GOOGLE_CLIENT_SECRET: str = ""
#     GITHUB_CLIENT_ID: str = ""
#     GITHUB_CLIENT_SECRET: str = ""
#
#     # Performance
#     BATCH_PROCESSING_STRATEGY: str = "thread_parallel"
#     EMBEDDING_BATCH_SIZE: int = 32
#     SEMANTIC_CACHE_ENABLED: bool = True
#
#     # Features
#     CONVERSATION_ENABLED: bool = True
#     ALERTS_ENABLED: bool = True
#     COLLECTIONS_ENABLED: bool = True
#     EXPORT_ENABLED: bool = True
#
#     # Database
#     DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/arxiv_rag"
#
# =============================================================================
# ENVIRONMENT VARIABLES (.env)
# =============================================================================
#
# # Security
# SECRET_KEY=your-super-secret-key-here
# GOOGLE_CLIENT_ID=your-google-client-id
# GOOGLE_CLIENT_SECRET=your-google-client-secret
# GITHUB_CLIENT_ID=your-github-client-id
# GITHUB_CLIENT_SECRET=your-github-client-secret
#
# # Database
# DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/arxiv_rag
#
# # Performance
# BATCH_PROCESSING_STRATEGY=thread_parallel
# SEMANTIC_CACHE_ENABLED=true
#
# # Features
# CONVERSATION_ENABLED=true
# ALERTS_ENABLED=true
# COLLECTIONS_ENABLED=true
# EXPORT_ENABLED=true
#
# =============================================================================
