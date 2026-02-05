# Quick Start Guide

## Prerequisites

- Python 3.13+
- Docker & Docker Compose (optional, for full stack)
- PostgreSQL (or use Docker)

## 1. Quick Setup (Recommended)

Run the setup script for your platform:

### Windows
```batch
scripts\dev-setup.bat
```

### Linux/Mac
```bash
bash scripts/dev-setup.sh
```

## 2. Manual Setup

### Install dependencies
```bash
pip install -e .
```

### Configure environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### Run database migrations
```bash
alembic upgrade head
```

### Start the application
```bash
# Option A: Local development
uvicorn app.main:app --reload

# Option B: Docker (full stack with all services)
docker-compose -f docker-compose.dev.yml up -d
```

## 3. Available Services

| Service | URL | Description |
|---------|-----|-------------|
| Application | http://localhost:8000 | Main API |
| API Docs | http://localhost:8000/api/docs | Swagger UI |
| Adminer | http://localhost:8080 | Database GUI |
| Redis Commander | http://localhost:8081 | Redis GUI |
| Prometheus | http://localhost:9090 | Metrics |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| Jaeger | http://localhost:16686 | Tracing |

## 4. Generate Secure Secret Key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copy the output to `SECRET_KEY` in `.env`.

## 5. First Time Setup

After starting the application:

1. **Access API Docs**: http://localhost:8000/api/docs

2. **Create admin user** (optional - via `/api/auth/register`)

3. **Check health**: http://localhost:8000/health

4. **Run tests** (optional):
```bash
pytest tests/ -v
```

## Troubleshooting

### Database connection error
```bash
# Make sure PostgreSQL is running
docker-compose -f docker-compose.dev.yml up postgres

# Or check if PostgreSQL is installed locally
psql -U postgres -h localhost
```

### Redis connection error
```bash
# Start Redis with Docker
docker-compose -f docker-compose.dev.yml up redis
```

### Port already in use
```bash
# Check what's using the port
# Windows:
netstat -ano | findstr :8000
# Linux/Mac:
lsof -i :8000
```

## Development Tips

1. **Hot reload**: Changes to code auto-reload with `--reload`

2. **Database GUI**: Use Adminer at http://localhost:8080
   - Server: `postgres`
   - Username: `postgres`
   - Password: `postgres`
   - Database: `arxiv_rag`

3. **View logs**: Check `logs/` directory or use `docker-compose logs -f app`

4. **Run migrations**: `alembic upgrade head`

5. **Create new migration**: `alembic revision --autogenerate -m "description"`
