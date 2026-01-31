# ArXiv RAG Copilot v0.4.0

**Enterprise-grade Retrieval-Augmented Generation (RAG) system** powered by LangChain for querying ArXiv research papers with production-ready features.

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## 🚀 What's New in v0.4.0

### Advanced Search Features
- **🔍 Hybrid Search**: BM25 (keyword) + Vector (semantic) with Reciprocal Rank Fusion
- **🎯 Multi-Query Retrieval**: LLM-generated query variants merged with RRF
- **🔄 Cross-Encoder Reranking**: Precision re-ranking with ms-marco-MiniLM-L-6-v2
- **💡 Query Autocomplete**: Prefix-based, fuzzy matching, and semantic suggestions
- **⚖️ MMR Diversification**: Balanced relevance-diversity result selection

### Infrastructure & Observability
- **📡 OpenTelemetry Tracing**: Distributed tracing with OTLP/Jaeger export
- **📦 Redis Distributed Cache**: Multi-instance caching with connection pooling
- **🔄 Background Tasks**: Automatic ArXiv RSS feed parsing and index updates
- **📊 Enhanced Metrics**: Query expansion, reranking, and circuit breaker monitoring

### Security & Reliability
- **🔒 Security Middleware**: CORS, rate limiting, input validation, request correlation ID
- **⚡ Circuit Breaker**: Automatic failure isolation for LLM and external API calls
- **🔄 Retry Logic**: Tenacity-based exponential backoff for transient failures
- **🛡️ Input Sanitization**: Protection against injection attacks (XSS, SQLi, command injection)
- **🔐 Caddy Reverse Proxy**: Automatic HTTPS with Let's Encrypt, security headers

### Performance & Resilience
- **🧵 Thread-Safe Singletons**: Double-checked locking for concurrent model access
- **📊 Connection Pooling**: Optimized PostgreSQL, Redis, and HTTP client pools
- **⏬ Graceful Shutdown**: Signal handling with request draining and cleanup callbacks
- **🚀 Optimized Semantic Chunking**: Batch processing for faster embeddings
- **📦 Type-Safe Models**: Pydantic validation throughout the stack

### Observability
- **🔍 Distributed Tracing**: Correlation ID propagation across requests
- **📈 Extended Metrics**: Query expansion, reranking, and circuit breaker metrics
- **🧪 RAGAS Evaluation**: Automated quality assessment with comparative testing

---

## ✨ Features Overview

### 🔍 Intelligent Retrieval
- **Hybrid Search**: Semantic (ChromaDB/Pgvector) + Lexical (BM25) with Reciprocal Rank Fusion
- **Query Expansion**: Intelligent acronym expansion and related term suggestion
- **Smart Reranking**: Cross-encoder precision + MMR diversity selection
- **Ensemble Retrieval**: Multiple query variants fused with RRF

### 🤖 LLM Integration
- **Multiple Providers**: OpenRouter, Ollama, or Mock mode
- **Streaming Responses**: Real-time answer generation via SSE
- **Chain-of-Thought**: Structured reasoning with few-shot examples
- **Smart Caching**: Redis-backed with TTL and automatic invalidation

### 🏗️ Production Ready
- **Docker Compose**: Full stack with PostgreSQL, Redis, Caddy
- **Automatic HTTPS**: Let's Encrypt integration via Caddy
- **Health Checks**: Comprehensive system monitoring
- **Graceful Shutdown**: Clean resource cleanup on termination
- **Prometheus Metrics**: Request latency, cache hits, circuit breaker state

---

## 📋 Quick Start

### Prerequisites

- Python 3.13+
- Docker & Docker Compose (for production deployment)
- ArXiv API access (free, no key required)

### 1. Clone and Install

```bash
# Clone repository
git clone https://github.com/yourusername/arxiv-rag-copilot.git
cd arxiv-rag-copilot

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your configuration
nano .env
```

**Minimum required configuration:**

```env
# Embeddings
EMBED_MODEL=intfloat/multilingual-e5-large

# LLM Provider
LLM_MODE=openrouter
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# Vector Store (ChromaDB for development)
VECTORSTORE_MODE=chroma
CHROMA_DIR=data/chroma_db
```

### 3. Start Development Server

```bash
# Start with hot reload
uv run uvicorn app.main:app --reload --port 8000

# Or using python directly
uvicorn app.main:app --reload --port 8000
```

Access the web interface at `http://localhost:8000`

### 4. Build Search Index

```bash
# Via web interface: http://localhost:8000/web/build
# Or via API:
curl -X POST http://localhost:8000/build \
  -H "Content-Type: application/json" \
  -d '{"query": "cat:cs.AI AND (rag OR retrieval)", "max_results": 50}'
```

---

## 🐳 Production Deployment with Docker

### Quick Start

```bash
# Start all services (PostgreSQL, Redis, App, Caddy)
docker compose up -d

# View logs
docker compose logs -f app

# Stop all services
docker compose down
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │       Caddy (Reverse Proxy)          │
        │  HTTPS, Security Headers, Compression │
        │  Rate Limiting, Request Logging        │
        └───────────────────┬───────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │          FastAPI Application          │
        │  ┌─────────────────────────────────┐  │
        │  │ Middleware Stack               │  │
        │  │ • CORS                         │  │
        │  │ • Rate Limiting                │  │
        │  │ • Correlation ID               │  │
        │  │ • Security Validation          │  │
        │  └─────────────────────────────────┘  │
        │  ┌─────────────────────────────────┐  │
        │  │ Resilience Layer               │  │
        │  │ • Circuit Breaker              │  │
        │  │ • Retry with Backoff           │  │
        │  │ • Graceful Shutdown            │  │
        │  └─────────────────────────────────┘  │
        │  ┌─────────────────────────────────┐  │
        │  │ Core Services (Thread-Safe)    │  │
        │  │ • Vector Store                 │  │
        │  │ • Embedder                     │  │
        │  │ • Reranker                     │  │
        │  │ • Cache                        │  │
        │  └─────────────────────────────────┘  │
        └───────────────────┬───────────────────┘
                                │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
       ┌─────────┐    ┌─────────┐    ┌─────────┐
       │PostgreSQL│    │  Redis  │    │  ArXiv  │
       │+pgvector│    │ (pool)  │    │   API   │
       └─────────┘    └─────────┘    └─────────┘
```

### Configuration

Edit `.env` for production:

```env
# Environment
ENVIRONMENT=production

# Vector Store (Pgvector for production)
VECTORSTORE_MODE=pgvector
POSTGRES_HOST=postgres
POSTGRES_USER=arxiv_rag
POSTGRES_PASSWORD=CHANGE_ME_PRODUCTION
POSTGRES_DB=arxiv_rag

# LLM Provider
LLM_MODE=openrouter
OPENROUTER_API_KEY=your-production-key
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# Caddy / Reverse Proxy
ACME_EMAIL=admin@yourdomain.com
CADDY_DOMAIN=your-domain.com
ALLOWED_ORIGINS=https://your-domain.com

# Rate Limiting (requests per minute, 0 = disabled)
RATE_LIMIT_RPM=60

# Cache
CACHE_ENABLED=true
REDIS_HOST=redis
REDIS_PORT=6379
CACHE_TTL=3600

# Monitoring
METRICS_ENABLED=true
```

### HTTPS with Caddy

1. **Set your domain** in `.env`:
   ```env
   CADDY_DOMAIN=your-domain.com
   ACME_EMAIL=admin@your-domain.com
   ```

2. **Update Caddyfile** - replace `your-domain.com` with your actual domain

3. **Start services**:
   ```bash
   docker compose up -d
   ```

Caddy will automatically obtain and renew SSL certificates from Let's Encrypt!

### Production Tips

- **Change default passwords** in `docker-compose.yml`
- **Set `ALLOWED_ORIGINS`** to your actual frontend domain
- **Enable metrics** and connect Prometheus/Grafana
- **Configure log aggregation** (JSON format enabled)
- **Set up monitoring alerts** for circuit breaker state

---

## 🏗️ Project Structure

```
arxiv-rag-copilot/
├── app/
│   ├── main.py                    # FastAPI server & routes
│   ├── config.py                  # Configuration management
│   │
│   # Core RAG Pipeline
│   ├── arxiv_loader.py            # ArXiv API client
│   ├── arxiv_document_loader.py   # LangChain document loader
│   ├── chunking.py                # Text chunking (semantic/sentence-aware)
│   ├── embeddings.py              # Embedding & reranking (thread-safe)
│   ├── vectorstore.py             # Vector store factory
│   ├── vectorstore_chroma.py      # ChromaDB implementation
│   ├── vectorstore_pgvector.py    # Pgvector implementation
│   ├── query_expansion.py         # Query expansion techniques
│   ├── rag.py                     # RAG logic & LLM generation
│   │
│   # Resilience & Reliability
│   ├── circuit_breaker.py         # Circuit breaker pattern
│   ├── retry.py                   # Retry with exponential backoff
│   ├── error_handling.py           # Error types & fallback strategies
│   ├── shutdown.py                # Graceful shutdown handling
│   │
│   # Performance & Pooling
│   ├── pooling.py                 # Connection pool configuration
│   │
│   # Security & Validation
│   ├── middleware.py              # CORS, rate limiting, correlation ID
│   ├── validation.py               # Input sanitization & threat detection
│   │
│   # Cross-cutting Concerns
│   ├── dependencies.py            # Dependency injection
│   ├── models.py                  # Type-safe Pydantic models
│   ├── cache.py                   # Redis caching layer
│   ├── metrics.py                 # Prometheus metrics
│   ├── logging_config.py          # Structured logging
│   │
│   └── evals.py                   # RAGAS evaluation
│
├── templates/                      # Web UI templates
├── Caddyfile                       # Caddy reverse proxy config
├── docker-compose.yml              # Production deployment
├── docker-compose.dev.yml          # Development deployment
├── tests/                          # Test suite
├── data/                           # Data directory
│   ├── raw/                        # Raw ArXiv data
│   ├── processed/                  # Evaluation results
│   └── chroma_db/                  # ChromaDB storage (dev)
└── pyproject.toml                  # Dependencies
```

---

## 🌐 API Documentation

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/build` | Build search index from ArXiv papers |
| `POST` | `/ask` | Ask question (streaming supported) |
| `GET` | `/search` | Search without LLM generation |
| `GET` | `/health` | System health check |
| `GET` | `/config` | Current configuration |

### Advanced Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/search/hybrid` | Hybrid semantic + keyword search |
| `POST` | `/search/multi-query` | Multi-query retrieval with expansion |
| `POST` | `/search/rerank` | Cross-encoder reranked search |
| `GET` | `/suggest` | Query autocomplete suggestions |

### Search & Index

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/index/papers` | List all indexed papers |
| `GET` | `/index/stats` | Index statistics |

### Background Tasks

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/background-tasks/status` | Background task manager status |
| `POST` | `/background-tasks/start` | Start ArXiv feed update task |
| `POST` | `/background-tasks/stop` | Stop background tasks |
| `GET` | `/background-tasks/jobs` | List active background jobs |

### Tracing & Telemetry

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/tracing/status` | OpenTelemetry tracing status |
| `POST` | `/tracing/flush` | Force flush trace data |

### Evaluation (RAGAS)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/evaluate` | Run RAGAS evaluation |
| `POST` | `/evaluate/compare` | Compare configurations |
| `GET` | `/evaluate/results` | Latest evaluation results |
| `GET` | `/evaluate/history` | Evaluation history |

### Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/metrics/summary` | Human-readable metrics |
| `GET` | `/cache/stats` | Cache statistics |
| `POST` | `/cache/clear` | Clear cache entries |
| `GET` | `/circuit-breaker` | Circuit breaker status |

---

## ⚙️ Configuration Reference

### Vector Store Options

```env
# ChromaDB (development, local files)
VECTORSTORE_MODE=chroma
CHROMA_DIR=data/chroma_db

# Pgvector (production, PostgreSQL)
VECTORSTORE_MODE=pgvector
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=arxiv_rag
POSTGRES_USER=arxiv_rag
POSTGRES_PASSWORD=changeme
POSTGRES_POOL_SIZE=10
```

### Embedding Models

```env
# Best quality (1024 dim, multilingual)
EMBED_MODEL=intfloat/multilingual-e5-large

# Alternatives:
# intfloat/multilingual-e5-base        # 768 dim, faster
# intfloat/multilingual-e5-small       # 384 dim, fastest
# sentence-transformers/all-MiniLM-L6-v2  # 384 dim, English only
# BAAI/bge-m3-retromae                 # 1024 dim, excellent retrieval
```

### LLM Configuration

```env
# OpenRouter (multi-provider access)
LLM_MODE=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_TIMEOUT=120

# Ollama (local inference)
LLM_MODE=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Mock (testing)
LLM_MODE=mock
```

### Retrieval Settings

```env
# Results configuration
TOP_K=5                   # Final results to return
RETRIEVAL_K=20            # Candidates before reranking

# Chunking
CHUNK_SIZE=900
CHUNK_OVERLAP=150
USE_SEMANTIC_CHUNKING=true
SEMANTIC_SIMILARITY_THRESHOLD=0.7

# Hybrid search weights
SEMANTIC_WEIGHT=0.7       # Semantic search weight
BM25_WEIGHT=0.3           # Lexical search weight
```

### Reranking

```env
# Enable reranking
RERANK_ENABLED=true
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# MMR (diversity)
RERANK_USE_MMR=true
MMR_LAMBDA=0.5            # 0=max diversity, 1=max relevance
```

### Query Expansion

```env
QUERY_EXPANSION_ENABLED=true
QUERY_EXPANSION_METHOD=acronym  # "acronym", "related", "both", "none"
```

### Caching

```env
CACHE_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL=3600            # 1 hour
```

### Security

```env
# CORS (comma-separated origins)
ALLOWED_ORIGINS=http://localhost:3000,https://example.com

# Rate Limiting
RATE_LIMIT_RPM=60          # Requests per minute (0 = disabled)

# Caddy / Reverse Proxy
ACME_EMAIL=admin@yourdomain.com
CADDY_DOMAIN=your-domain.com
```

---

## 📊 Monitoring & Observability

### Prometheus Metrics

Available at `http://localhost:8001/metrics`:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `arxiv_rag_requests_total` | Counter | endpoint, method | Total requests |
| `arxiv_rag_errors_total` | Counter | endpoint, error_type | Total errors |
| `arxiv_rag_request_latency_seconds` | Histogram | endpoint | Request duration |
| `arxiv_rag_retrieval_latency_seconds` | Histogram | method | Retrieval duration |
| `arxiv_rag_llm_latency_seconds` | Histogram | provider, model | LLM duration |
| `arxiv_rag_cache_hits_total` | Counter | cache_type | Cache hits |
| `arxiv_rag_cache_misses_total` | Counter | cache_type | Cache misses |
| `arxiv_rag_query_expansion_hits_total` | Counter | method | Query expansions |
| `arxiv_rag_rerank_latency_seconds` | Histogram | method | Reranking duration |
| `arxiv_rag_documents_retrieved` | Histogram | method | Docs per retrieval |

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "llm_mode": "openrouter",
  "llm_health": {"healthy": true},
  "index_loaded": true,
  "index_documents": 150,
  "embedder_loaded": true,
  "reranker_enabled": true,
  "query_expansion_enabled": true
}
```

### Distributed Tracing

Each request includes a `X-Request-ID` header for tracing:

```bash
curl -v http://localhost:8000/health | grep X-Request-ID
```

---

## 🧪 Testing & Evaluation

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v
```

### RAGAS Evaluation

```bash
# Command line
python -m app.evals

# With comparison
python -m app.evals --compare

# Via API
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"rerank_method": "mmr", "use_cot_prompting": true}'
```

**RAGAS Metrics:**
- **Faithfulness**: Answer consistency with sources
- **Answer Relevancy**: How well answer addresses question
- **Context Precision**: Retrieval quality
- **Context Recall**: Coverage of relevant information
- **Answer Correctness**: Factual accuracy

---

## 🎯 Performance Optimization

### Caching Strategy

| Scenario | Cache Type | Speedup |
|----------|------------|---------|
| Repeated queries | Redis | 10-100x |
| Embedding computation | In-memory | 5-20x |
| Vector store (ChromaDB) | Internal cache | 2-5x |

### Tuning Parameters

**For precision** (focus on accuracy):
```env
MMR_LAMBDA=0.8-1.0          # Higher relevance weight
SEMANTIC_WEIGHT=0.8         # Stronger semantic focus
RERANK_ENABLED=true
```

**For diversity** (explore different aspects):
```env
MMR_LAMBDA=0.3-0.5          # Higher diversity weight
SEMANTIC_WEIGHT=0.6
QUERY_EXPANSION_METHOD=both   # More related terms
```

**For speed** (minimize latency):
```env
RETRIEVAL_K=10               # Fewer candidates
RERANK_ENABLED=false         # Skip reranking
CHUNK_SIZE=600               # Smaller chunks
EMBED_MODEL=intfloat/multilingual-e5-small
```

---

## 🐛 Troubleshooting

### Docker Issues

**Container won't start:**
```bash
# Check logs
docker compose logs app

# Verify ports are available
netstat -tuln | grep -E '8000|5432|6379'
```

**Database connection errors:**
```bash
# Check PostgreSQL is healthy
docker compose exec postgres pg_isready -U arxiv_rag

# Reset database (WARNING: deletes data)
docker compose down -v
docker compose up -d
```

### Caddy HTTPS Issues

**Certificate not obtained:**
```bash
# Check Caddy logs
docker compose logs caddy

# Verify domain DNS points to server
nslookup your-domain.com

# Check port 80/443 accessibility
curl http://your-domain.com
```

**Use HTTP locally (no HTTPS):**
```bash
# Access via :8080 port
curl http://localhost:8080/health
```

### Application Issues

**No answers from LLM:**
```bash
# Check health endpoint
curl http://localhost:8000/health

# Verify API key
echo $OPENROUTER_API_KEY

# Check circuit breaker state
curl http://localhost:8000/circuit-breaker
```

**Slow queries:**
```bash
# Check cache stats
curl http://localhost:8000/cache/stats

# View metrics summary
curl http://localhost:8000/metrics/summary
```

---

## 📈 Benchmarks

**System**: E5-large, Claude 3.5 Sonnet, 10K documents

| Operation | Latency | Notes |
|-----------|---------|-------|
| Query Embedding | ~50ms | CPU, single thread |
| Hybrid Retrieval | ~30ms | Semantic + BM25 |
| MMR Reranking | ~100ms | 20→5 candidates |
| LLM Generation | ~2-5s | Streaming enabled |
| **Cached Query** | **~50ms** | Redis hit |
| **Uncached Query** | **~3-6s** | Full pipeline |

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright 2025 ArXivFuturaSearch Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

## 🙏 Acknowledgments

- **ChromaDB** - Scalable vector database
- **Pgvector** - Vector similarity in PostgreSQL
- **LangChain** - Framework for LLM applications
- **Sentence Transformers** - E5 multilingual embeddings
- **RAGAS** - RAG evaluation framework
- **FastAPI** - Modern async web framework
- **Caddy** - Automatic HTTPS server
- **ArXiv** - Open access research papers

---

**Built with ❤️ for AI researchers and developers**
