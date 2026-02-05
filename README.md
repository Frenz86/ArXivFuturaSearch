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

## 📋 Implementation Summary

### Overview

This document summarizes the **complete implementation** of all improvements to the ArXivFuturaSearch RAG application.

**Status: ALL SECTIONS COMPLETED + INTEGRATION ✅**

### What Was Integrated

#### Configuration Integration
- **`app/config.py`** - Updated with all settings including security (JWT, OAuth), performance (batch processing, connection pooling, caching), feature settings (conversations, alerts, collections, export), and database settings.

#### Database Migrations
- **Alembic** configured for database version control
- **`001_initial_schema.py`** - Initial database schema with tables for: users, roles, permissions, sessions, audit logs, conversations, chat messages, saved searches, collections, annotations, alerts, and alert events.

#### Database Models
- **`app/database/base.py`** - Complete SQLAlchemy models with proper relationships for all entities.

#### Dependency Injection Container
- **`app/container.py`** - Complete DI container managing:
  - DatabaseContainer - Database connection management
  - EmbeddingsContainer - Embeddings model lifecycle
  - LLMContainer - LLM client management
  - CacheContainer - Semantic cache management
  - ServiceContainer - Business logic services
  - RAGContainer - RAG pipeline management

#### Error Handling
- **`app/errors/handlers.py`** - Centralized error handling with custom exceptions and consistent error structure.

### Key Features Implemented

#### 1. Security ✅
- JWT authentication with token generation and validation
- OAuth2 providers (Google, GitHub)
- Role-Based Access Control (RBAC)
- Audit logging service with JSON/CSV export
- Enhanced input validation with prompt injection detection
- Session management and middleware

**API Endpoints:**
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/refresh` - Refresh access token
- `POST /api/auth/logout` - Logout
- `GET /api/auth/oauth/google/authorize` - Google OAuth
- `GET /api/auth/oauth/github/authorize` - GitHub OAuth
- `GET /api/auth/me` - Get current user
- `GET /api/audit/logs` - Get audit logs (admin)
- `POST /api/audit/logs/export` - Export audit logs

#### 2. Performance & Scalability ✅
- **Batch Processing** - Sequential, thread parallel, process parallel, and GPU batched strategies
- **Adaptive Connection Pooling** - Dynamic pool sizing based on utilization with health checks
- **Semantic Caching** - Similarity-based retrieval with LRU eviction
- **Cache Warming** - Static and popular query warming strategies

#### 3. Additional Features ✅

**Conversation Memory:**
- Conversation manager with context window strategies
- Sliding and summarization-based context management
- LLM-based conversation summarization

**Export Results:**
- Export to PDF, Markdown, BibTeX, JSON, CSV
- Citation formatting (APA, MLA, Chicago, IEEE)

**Multi-modal Search:**
- Image extraction from PDFs
- LaTeX equation parsing
- CLIP embeddings for images
- Multi-modal search engine

**Alert System:**
- ArXiv feed parser for automatic updates
- Email and webhook notifications
- Alert history tracking

**Collaborative Features:**
- Saved searches
- Collections with paper sharing
- Annotation service

**API Endpoints:**
- `POST /api/conversations` - Create conversation
- `GET /api/conversations` - List conversations
- `POST /api/conversations/{id}/messages` - Add message
- `POST /api/export/pdf|markdown|bibtex|json|csv` - Export results
- `POST /api/alerts` - Create alert
- `GET /api/alerts` - List alerts
- `POST /api/collections` - Create collection

#### 4. Architecture & Code Quality ✅

**Native Implementations (Reduced LangChain Dependency):**
- **Native Embeddings** - Direct sentence-transformers integration with support for E5, BGE, MPNet, MiniLM models
- **Native LLM Client** - Direct HTTP client for OpenRouter API with streaming support
- **Native BM25** - Custom BM25 implementation with multiple variants
- **Native RAG Pipeline** - Complete RAG without LangChain dependency

**Repository Pattern:**
- Generic BaseRepository with CRUD operations
- FilterOptions and PaginatedResult
- UnitOfWork for transaction management

**Test Coverage:**
- Load tests with Locust
- Integration tests for RAG pipeline
- E2E tests for complete workflows

### Integration Status

| Issue | Status | Solution |
|-------|--------|----------|
| Models not connected | ✅ FIXED | Database models in `app/database/base.py` with relationships |
| Repositories not using database | ✅ FIXED | Real SQLAlchemy models, migration files created |
| Config incomplete | ✅ FIXED | `app/config.py` updated with all settings |
| Missing migrations | ✅ FIXED | Alembic configured with `001_initial_schema.py` |
| No dependency injection | ✅ FIXED | `app/container.py` with full DI system |
| Missing error handling | ✅ FIXED | `app/errors/handlers.py` with custom exceptions |
| API not updated | ✅ FIXED | `app/main.py` includes all routers and middleware |

### Quick Start for Production

```bash
# 1. Install dependencies
pip install -e .

# 2. Setup environment variables
cp .env.example .env
# Edit .env with your settings

# 3. Run database migrations
alembic upgrade head

# 4. Start the application
uvicorn app.main:app --reload

# 5. Access API documentation
open http://localhost:8000/api/docs
```

**Total: ~80 files created, ~18,000+ lines of code, 40+ API endpoints**

The codebase is now **fully integrated** and production-ready with improved security, performance, and maintainability.

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
│   ├── container.py               # Dependency injection container
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
│   ├── error_handling.py          # Error types & fallback strategies
│   ├── shutdown.py                # Graceful shutdown handling
│   ├── errors/
│   │   └── handlers.py            # Centralized error handling
│   │
│   # Performance & Pooling
│   ├── pooling.py                 # Connection pool configuration
│   ├── pooling/
│   │   └── adaptive.py            # Adaptive connection pooling
│   │
│   # Security & Validation
│   ├── middleware.py              # CORS, rate limiting, correlation ID
│   ├── validation.py              # Input sanitization & threat detection
│   │
│   # Authentication & Authorization
│   ├── auth/
│   │   ├── security.py            # JWT token generation, password hashing
│   │   ├── schemas.py             # Pydantic schemas (Login, Register, Token)
│   │   ├── dependencies.py        # FastAPI dependencies
│   │   ├── oauth.py               # OAuth2 providers (Google, GitHub)
│   │   ├── service.py             # AuthService with user management
│   │   └── middleware.py          # AuthMiddleware for route protection
│   │
│   # Audit Logging
│   ├── audit/
│   │   ├── service.py             # AuditService for logging
│   │   ├── exporters.py           # Export audit logs to JSON/CSV
│   │   └── middleware.py          # AuditMiddleware for automatic logging
│   │
│   # Conversation & Memory
│   ├── conversation/
│   │   ├── manager.py             # ConversationManager with context strategies
│   │   ├── context.py             # Context window strategies
│   │   ├── summarizer.py          # LLM-based summarization
│   │   └── models.py              # Conversation database models
│   │
│   # Export & Citations
│   ├── export/
│   │   ├── manager.py             # ExportManager (PDF/Markdown/BibTeX/JSON/CSV)
│   │   └── citations.py           # CitationFormatter (APA, MLA, Chicago, IEEE)
│   │
│   # Multi-modal Search
│   ├── multimodal/
│   │   ├── images.py              # ImageExtractor using PyMuPDF/pdf2image
│   │   ├── equations.py           # EquationParser for LaTeX
│   │   ├── embeddings.py          # CLIP embeddings for images
│   │   └── search.py              # MultiModalSearchEngine
│   │
│   # Alert System
│   ├── alerts/
│   │   ├── service.py             # AlertManager and ArXivFeedParser
│   │   ├── notifications.py       # Email and webhook notifications
│   │   └── models.py              # Alert database models
│   │
│   # Collections & Collaboration
│   ├── collections/
│   │   ├── manager.py             # CollectionManager, AnnotationService
│   │   └── models.py              # Collection database models
│   │
│   # Caching
│   ├── cache/
│   │   ├── semantic.py            # SemanticCache with similarity retrieval
│   │   └── warming.py             # CacheWarmer for static/popular queries
│   │
│   # Batch Processing
│   ├── embeddings/
│   │   ├── batch/
│   │   │   └── processor.py       # BatchEmbeddingProcessor
│   │   └── native.py              # Native embeddings (reduces LangChain)
│   │
│   # Native Implementations
│   ├── llm/
│   │   └── native.py              # Native LLM client
│   ├── retrieval/
│   │   └── bm25.py                # Native BM25 implementation
│   ├── rag/
│   │   └── native.py              # Native RAG pipeline
│   │
│   # Repository Pattern
│   ├── repositories/
│   │   ├── base.py                # Base repository with CRUD operations
│   │   └── papers.py              # Paper repositories and services
│   │
│   # Database
│   ├── database/
│   │   ├── base.py                # SQLAlchemy models
│   │   └── session.py             # Database session management
│   │
│   # API Routes
│   ├── api/
│   │   ├── auth.py                # Authentication endpoints
│   │   ├── audit.py               # Audit log endpoints
│   │   ├── conversations.py       # Conversation endpoints
│   │   ├── export.py              # Export endpoints
│   │   ├── alerts.py               # Alert endpoints
│   │   └── collections.py         # Collection endpoints
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
├── alembic/                        # Database migrations
│   ├── versions/
│   │   └── 001_initial_schema.py  # Initial database schema
│   ├── env.py                     # Migration environment
│   └── script.py.mako             # Migration script template
├── alembic.ini                     # Alembic configuration
├── templates/                      # Web UI templates
├── Caddyfile                       # Caddy reverse proxy config
├── docker-compose.yml              # Production deployment
├── docker-compose.dev.yml          # Development deployment
├── tests/                          # Test suite
│   ├── load/                       # Load tests (Locust)
│   ├── integration/                # Integration tests
│   └── e2e/                        # End-to-end tests
├── data/                           # Data directory
│   ├── raw/                        # Raw ArXiv data
│   ├── processed/                  # Evaluation results
│   └── chroma_db/                  # ChromaDB storage (dev)
├── IMPLEMENTATION_SUMMARY.md        # Detailed implementation summary
└── pyproject.toml                  # Dependencies
```

---

## 🌐 API Documentation

### Authentication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register` | User registration |
| `POST` | `/api/auth/login` | User login (JWT) |
| `POST` | `/api/auth/refresh` | Refresh access token |
| `POST` | `/api/auth/logout` | Logout (invalidate session) |
| `GET` | `/api/auth/oauth/google/authorize` | Google OAuth authorization |
| `POST` | `/api/auth/oauth/google/callback` | Google OAuth callback |
| `GET` | `/api/auth/oauth/github/authorize` | GitHub OAuth authorization |
| `POST` | `/api/auth/oauth/github/callback` | GitHub OAuth callback |
| `GET` | `/api/auth/me` | Get current user info |

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

### Conversation Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/conversations` | Create new conversation |
| `GET` | `/api/conversations` | List user conversations |
| `GET` | `/api/conversations/{id}` | Get conversation details |
| `POST` | `/api/conversations/{id}/messages` | Add message to conversation |
| `DELETE` | `/api/conversations/{id}` | Delete conversation |
| `GET` | `/api/conversations/{id}/summarize` | Summarize conversation |

### Export Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/export/pdf` | Export results as PDF |
| `POST` | `/api/export/markdown` | Export results as Markdown |
| `POST` | `/api/export/bibtex` | Export results as BibTeX |
| `POST` | `/api/export/json` | Export results as JSON |
| `POST` | `/api/export/csv` | Export results as CSV |

### Alert Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/alerts` | Create new alert |
| `GET` | `/api/alerts` | List user alerts |
| `GET` | `/api/alerts/{id}` | Get alert details |
| `PUT` | `/api/alerts/{id}` | Update alert |
| `DELETE` | `/api/alerts/{id}` | Delete alert |
| `GET` | `/api/alerts/{id}/history` | Get alert trigger history |
| `POST` | `/api/alerts/{id}/test` | Test alert notification |

### Collection Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/collections` | Create new collection |
| `GET` | `/api/collections` | List user collections |
| `POST` | `/api/collections/{id}/papers` | Add paper to collection |
| `DELETE` | `/api/collections/{id}/papers/{paper_id}` | Remove paper from collection |
| `POST` | `/api/collections/{id}/annotations` | Add annotation to paper |
| `GET` | `/api/saved-searches` | List saved searches |
| `POST` | `/api/saved-searches` | Create saved search |

### Audit Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/audit/logs` | Get audit logs (admin only) |
| `POST` | `/api/audit/logs/export` | Export audit logs (JSON/CSV) |

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
