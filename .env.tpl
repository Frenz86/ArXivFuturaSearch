# ArXiv RAG Copilot Configuration
# Copy this to .env and fill in your values

# LLM Mode: "openrouter" (default), "ollama", or "mock"
LLM_MODE=openrouter

POSTGRES_PASSWORD=op://ArXivFuturaSearch/POSTGRES_PASSWORD/password

# OpenRouter API Key (get yours at https://openrouter.ai/keys)
OPENROUTER_API_KEY=op://ArXivFuturaSearch/OPENROUTER_API_KEY/password

# OpenRouter Model (see https://openrouter.ai/models for options)
# Popular choices:
#   anthropic/claude-3.5-sonnet  - Best quality, ~$3/1M tokens
#   openai/gpt-4o-mini           - Fast & cheap, ~$0.15/1M tokens
#   google/gemini-flash-1.5      - Very fast, ~$0.075/1M tokens
#   meta-llama/llama-3.1-70b-instruct - Open weights
#   mistralai/mixtral-8x7b-instruct   - Good balance

OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# OpenRouter settings
OPENROUTER_TIMEOUT=120
OPENROUTER_MAX_RETRIES=3

# Retrieval settings
TOP_K=5
RETRIEVAL_K=20
CHUNK_SIZE=900
CHUNK_OVERLAP=150

# Hybrid search weights (must sum to 1.0)
SEMANTIC_WEIGHT=0.7
BM25_WEIGHT=0.3

# Reranking (improves precision but adds latency)
RERANK_ENABLED=true
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Embedding model (local, no API needed)
EMBED_MODEL=intfloat/multilingual-e5-large

# Ollama settings (if using LLM_MODE=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=console  # "json" for production, "console" for development
