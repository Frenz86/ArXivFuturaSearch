# ArXiv RAG Copilot — 1Password-managed configuration
# Secrets are resolved at runtime via `op run --env-file=.env.tpl`
# Docs: https://developer.1password.com/docs/cli/secrets-environment-variables/

# === Secrets (from 1Password vault "Team", item "ArXivFuturaSearch") ===
OPENROUTER_API_KEY=op://Team/ArXivFuturaSearch/OPENROUTER_API_KEY
POSTGRES_PASSWORD=op://Team/ArXivFuturaSearch/POSTGRES_PASSWORD
REDIS_PASSWORD=op://Team/ArXivFuturaSearch/REDIS_PASSWORD
SECRET_KEY=op://Team/ArXivFuturaSearch/SECRET_KEY

# === Non-sensitive configuration (plain values) ===

# LLM
LLM_MODE=openrouter
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_TIMEOUT=120
OPENROUTER_MAX_RETRIES=3

# Retrieval
TOP_K=5
RETRIEVAL_K=20
CHUNK_SIZE=900
CHUNK_OVERLAP=150

# Hybrid search
SEMANTIC_WEIGHT=0.7
BM25_WEIGHT=0.3

# Reranking
RERANK_ENABLED=true
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_USE_MMR=true
MMR_LAMBDA=0.5

# Embeddings
EMBED_MODEL=intfloat/multilingual-e5-large

# Semantic chunking
USE_SEMANTIC_CHUNKING=true
SEMANTIC_SIMILARITY_THRESHOLD=0.7

# Query expansion
QUERY_EXPANSION_ENABLED=true
QUERY_EXPANSION_METHOD=acronym

# Ollama (if LLM_MODE=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=console

# Environment
ENVIRONMENT=development
