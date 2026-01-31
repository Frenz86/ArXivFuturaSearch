from pydantic_settings import BaseSettings, SettingsConfigDict


# Single source of truth for version
__version__ = "0.3.0"


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


settings = Settings()
