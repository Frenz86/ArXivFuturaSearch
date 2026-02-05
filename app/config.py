
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

from pydantic_settings import BaseSettings, SettingsConfigDict


# Single source of truth for version
__version__ = "0.4.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Version (single source of truth)
    VERSION: str = __version__

    # Environment
    ENVIRONMENT: str = "development"  # "development" or "production"

    # Vector Store Configuration
    # Options: "chroma" (local, dev) or "pgvector" (PostgreSQL, production)
    VECTORSTORE_MODE: str = "chroma"

    # Data
    DATA_DIR: str = "data"
    RAW_DIR: str = "data/raw"
    PROCESSED_DIR: str = "data/processed"
    CHROMA_DIR: str = "data/chroma_db"

    # PostgreSQL Configuration (for pgvector in production)
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "arxiv_rag"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_POOL_SIZE: int = 10
    POSTGRES_MAX_OVERFLOW: int = 20

    # Retrieval
    TOP_K: int = 5
    RETRIEVAL_K: int = 20  # Retrieve more for reranking
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 150

    # Semantic Chunking
    USE_SEMANTIC_CHUNKING: bool = True
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.7

    # Hybrid search weights
    SEMANTIC_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3

    # Reranking
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_USE_MMR: bool = True  # Maximal Marginal Relevance for diversity
    MMR_LAMBDA: float = 0.5  # Balance between relevance and diversity

    # Embeddings - intfloat/multilingual-e5-large
    EMBED_MODEL: str = "intfloat/multilingual-e5-large"
    # Alternative options:
    # - "intfloat/multilingual-e5-base" (faster multilingual, 768 dim)
    # - "intfloat/multilingual-e5-small" (smallest multilingual, 384 dim)
    # - "sentence-transformers/all-MiniLM-L6-v2" (fast English-only, 384 dim)
    # - "BAAI/bge-m3-retromae" (excellent for retrieval, 1024 dim)
    # - "BAAI/bge-large-en-v1.5" (great for English, 1024 dim)
    # - "jinaai/jina-embeddings-v2" (great quality, 768 dim)
    # - "nomic-ai/nomic-embed-text-v1" (great for long context, 768 dim)

    # Query expansion
    QUERY_EXPANSION_ENABLED: bool = True
    QUERY_EXPANSION_METHOD: str = "acronym"  # "acronym", "related", "both", "none"

    # Caching
    CACHE_ENABLED: bool = True
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None
    CACHE_TTL: int = 3600  # 1 hour

    # Monitoring
    METRICS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 8001

    # LLM mode: "openrouter", "ollama", or "mock"
    LLM_MODE: str = "openrouter"

    # OpenRouter (default)
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "anthropic/claude-3.5-sonnet"
    OPENROUTER_TIMEOUT: int = 120
    OPENROUTER_MAX_RETRIES: int = 3

    # Ollama (optional local)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" or "console"

    # OpenTelemetry
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"  # OTLP collector
    OTEL_EXPORTER_JAEGER_HOST: str = "localhost"
    OTEL_EXPORTER_JAEGER_PORT: int = 6831
    OTEL_SERVICE_NAME: str = "arxiv-futura-search"
    OTEL_TRACE_EXPORTER: str = "console"  # "console", "otlp", "jaeger", "none"
    OTEL_METRICS_EXPORTER: str = "console"  # "console", "otlp", "none"

    # =============================================================================
    # SECURITY SETTINGS (from config_extensions.py)
    # =============================================================================
    SECRET_KEY: str = "changeme-secret-key"
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

    # =============================================================================
    # PERFORMANCE SETTINGS (from config_extensions.py)
    # =============================================================================
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

    # =============================================================================
    # FEATURE SETTINGS (from config_extensions.py)
    # =============================================================================
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

    # =============================================================================
    # DATABASE SETTINGS (from config_extensions.py)
    # =============================================================================
    # PostgreSQL (for auth, conversations, etc.)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/arxiv_rag"
    DATABASE_ECHO: bool = False


settings = Settings()
