# ArXiv RAG Copilot - Docker Deployment Guide

This guide explains how to deploy ArXiv RAG Copilot using Docker, with support for both development (ChromaDB) and production (PostgreSQL + pgvector) environments.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ArXiv RAG Copilot                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │   FastAPI App    │────────▶│  Embedding Model │            │
│  │   (Python 3.13)  │         │  (HF Transformers)│            │
│  └────────┬─────────┘         └──────────────────┘            │
│           │                                                      │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Vector Store Layer                         │    │
│  ├────────────────────────────────────────────────────────┤    │
│  │                                                          │    │
│  │  ┌─────────────────────┐  ┌──────────────────────┐     │    │
│  │  │ ChromaDB (Dev)      │  │ PostgreSQL+pgvector  │     │    │
│  │  │ - Embedded          │  │ (Production)         │     │    │
│  │  │ - No external DB    │  │ - Scalable           │     │    │
│  │  │ - Fast setup        │  │ - ACID compliant     │     │    │
│  │  └─────────────────────┘  └──────────────────────┘     │    │
│  └────────────────────────────────────────────────────────┘    │
│           │                                                      │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │  Redis Cache     │         │  LLM Provider    │            │
│  │  (Optional)      │         │  (OpenRouter/    │            │
│  │                  │         │   Ollama/Mock)    │            │
│  └──────────────────┘         └──────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker 24.0+
- Docker Compose 2.20+
- 4GB RAM minimum (8GB recommended)
- 10GB disk space

## Quick Start

### Development (ChromaDB)

```bash
# 1. Copy environment configuration
cp .env.dev.example .env

# 2. Edit .env and add your OpenRouter API key
# OPENROUTER_API_KEY=sk-or-...

# 3. Start the services
docker compose -f docker-compose.dev.yml up -d

# 4. View logs
docker compose -f docker-compose.dev.yml logs -f

# 5. Access the application
open http://localhost:8000
```

### Production (PostgreSQL + pgvector)

```bash
# 1. Copy environment configuration
cp .env.prod.example .env

# 2. Edit .env and set strong passwords
# POSTGRES_PASSWORD=changeme
# OPENROUTER_API_KEY=sk-or-...

# 3. Start the services
docker compose up -d

# 4. Wait for PostgreSQL to be ready (check health status)
docker compose ps

# 5. View logs
docker compose logs -f

# 6. Access the application
open http://localhost:8000
```

## Configuration Files

| File | Purpose |
|------|---------|
| [docker-compose.yml](docker-compose.yml) | Production deployment with PostgreSQL |
| [docker-compose.dev.yml](docker-compose.dev.yml) | Development with ChromaDB |
| [Dockerfile](Dockerfile) | Multi-stage container image |
| [.env.dev.example](.env.dev.example) | Development environment template |
| [.env.prod.example](.env.prod.example) | Production environment template |
| [scripts/init-postgres.sql](scripts/init-postgres.sql) | PostgreSQL initialization |

## Vector Store Modes

### ChromaDB (Development)

**Advantages:**
- No external database required
- Fast startup and simple setup
- Ideal for local development and testing
- Embedded storage (file-based)

**When to use:**
- Local development
- Quick prototyping
- Testing and experimentation
- Small datasets (< 10K documents)

**Configuration:**
```bash
VECTORSTORE_MODE=chroma
```

### PostgreSQL + pgvector (Production)

**Advantages:**
- Scalable to millions of documents
- ACID compliance and data integrity
- Concurrent access support
- Backup and replication friendly
- Integration with existing PostgreSQL infrastructure

**When to use:**
- Production deployments
- Large datasets (> 10K documents)
- High concurrency requirements
- Need for proper backups

**Configuration:**
```bash
VECTORSTORE_MODE=pgvector
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=arxiv_rag
POSTGRES_USER=arxiv_rag
POSTGRES_PASSWORD=strong_password_here
```

## Environment Variables

### Required

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | `None` |
| `POSTGRES_PASSWORD` | PostgreSQL password (prod) | `changeme` |

### Optional Configuration

| Category | Variable | Default |
|----------|----------|---------|
| **Ports** | `APP_PORT` | `8000` |
| | `PROMETHEUS_PORT` | `8001` |
| | `REDIS_PORT` | `6379` |
| | `POSTGRES_PORT` | `5432` |
| **Vector Store** | `VECTORSTORE_MODE` | `chroma` (dev) / `pgvector` (prod) |
| **LLM** | `LLM_MODE` | `openrouter` |
| | `OPENROUTER_MODEL` | `anthropic/claude-3.5-sonnet` |
| **Retrieval** | `TOP_K` | `5` |
| | `RETRIEVAL_K` | `20` |
| | `CHUNK_SIZE` | `900` |
| | `CHUNK_OVERLAP` | `150` |
| **Hybrid Search** | `SEMANTIC_WEIGHT` | `0.7` |
| | `BM25_WEIGHT` | `0.3` |
| **Reranking** | `RERANK_ENABLED` | `true` |
| | `RERANK_USE_MMR` | `true` |
| **Cache** | `CACHE_ENABLED` | `true` |
| | `REDIS_MAX_MEMORY` | `128mb` (dev) / `256mb` (prod) |
| **Logging** | `LOG_LEVEL` | `DEBUG` (dev) / `INFO` (prod) |
| | `LOG_FORMAT` | `console` (dev) / `json` (prod) |

## Docker Commands

### Build and Start

```bash
# Development
docker compose -f docker-compose.dev.yml up -d --build

# Production
docker compose up -d --build
```

### View Logs

```bash
# All services
docker compose -f docker-compose.dev.yml logs -f

# Specific service
docker compose -f docker-compose.dev.yml logs -f app
```

### Stop Services

```bash
# Development
docker compose -f docker-compose.dev.yml down

# Production (removes volumes too)
docker compose down -v
```

### Health Check

```bash
# Check service status
docker compose ps

# Application health endpoint
curl http://localhost:8000/health
```

## Storage and Volumes

### Development Volumes

| Volume | Purpose |
|--------|---------|
| `arxiv_dev_data` | ChromaDB data, downloaded papers |
| `redis_dev_data` | Redis persistence |
| `hf_cache_dev` | HuggingFace model cache |

### Production Volumes

| Volume | Purpose |
|--------|---------|
| `postgres_data` | PostgreSQL database with embeddings |
| `arxiv_data` | Downloaded papers, processed results |
| `redis_data` | Redis persistence |
| `hf_cache` | HuggingFace model cache |

## Backup and Restore

### Development (ChromaDB)

```bash
# Backup
docker run --rm -v arxiv_dev_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/chromadb-backup.tar.gz /data

# Restore
docker run --rm -v arxiv_dev_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/chromadb-backup.tar.gz -C /
```

### Production (PostgreSQL)

```bash
# Backup
docker exec arxiv-postgres pg_dump -U arxiv_rag arxiv_rag > backup.sql

# Restore
docker exec -i arxiv-postgres psql -U arxiv_rag arxiv_rag < backup.sql
```

## Troubleshooting

### Issue: Application won't start

```bash
# Check logs
docker compose logs app

# Common causes:
# 1. Missing OPENROUTER_API_KEY
# 2. PostgreSQL not ready yet
# 3. Port already in use
```

### Issue: PostgreSQL connection refused

```bash
# Check PostgreSQL health
docker compose ps postgres

# View PostgreSQL logs
docker compose logs postgres

# Ensure pgvector extension is installed
docker exec arxiv-postgres psql -U arxiv_rag -d arxiv_rag \
  -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Issue: High memory usage

```bash
# Reduce Redis memory limit in .env
REDIS_MAX_MEMORY=128mb

# Reduce embedding model size
EMBED_MODEL=intfloat/multilingual-e5-small
```

## Migration: Dev to Prod

To migrate from ChromaDB (dev) to PostgreSQL (prod):

```bash
# 1. Export data from ChromaDB
# (The application will need to re-index documents)

# 2. Switch to production mode
cp .env.prod.example .env
# Edit .env with production values

# 3. Start production services
docker compose up -d

# 4. Rebuild the index via API
curl -X POST http://localhost:8000/build \
  -H "Content-Type: application/json" \
  -d '{"query":"cat:cs.LG","max_results":100}'
```

## Security Checklist

Before deploying to production:

- [ ] Change all default passwords
- [ ] Set strong `POSTGRES_PASSWORD`
- [ ] Use secrets management for API keys
- [ ] Enable HTTPS/TLS
- [ ] Restrict network access
- [ ] Set up firewall rules
- [ ] Enable rate limiting
- [ ] Configure log aggregation
- [ ] Set up monitoring alerts
- [ ] Regular backup schedule

## Performance Tuning

### PostgreSQL

```bash
# Increase shared buffers in postgres
POSTGRES_SHARED_BUFFERS=256MB

# Increase connection pool
POSTGRES_POOL_SIZE=20
POSTGRES_MAX_OVERFLOW=40
```

### Redis

```bash
# Adjust memory limit based on cache hit rate
REDIS_MAX_MEMORY=512mb
```

### Application

```bash
# Adjust retrieval parameters
TOP_K=10
RETRIEVAL_K=50

# Disable features not needed
RERANK_ENABLED=false
QUERY_EXPANSION_ENABLED=false
```

## Additional Services

### Prometheus (Monitoring)

Uncomment in [docker-compose.yml](docker-compose.yml):

```yaml
prometheus:
  image: prom/prometheus:latest
  # ...
```

### Grafana (Visualization)

Uncomment in [docker-compose.yml](docker-compose.yml):

```yaml
grafana:
  image: grafana/grafana:latest
  # ...
```

## Support

- **Documentation**: See [README.md](README.md)
- **Issues**: Open an issue on GitHub
- **API Docs**: http://localhost:8000/docs (when running)
