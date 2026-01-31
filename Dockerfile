# =============================================================================
# ArXiv RAG Copilot - Multi-Stage Dockerfile
# =============================================================================
# Stage 1: Builder - Install dependencies using uv for faster builds
# =============================================================================
FROM python:3.13-slim-bookworm AS builder

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (faster than pip)
# --frozen ensures reproducible builds from lockfile
RUN uv sync --frozen --no-dev

# =============================================================================
# Stage 2: Runtime - Minimal image for production
# =============================================================================
FROM python:3.13-slim-bookworm

# Set labels
LABEL maintainer="Futura AI"
LABEL description="ArXiv RAG Copilot - Advanced AI-powered research paper search"
LABEL version="0.3.0"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy uv for potential future use
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# Copy application and virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY app ./app
COPY templates ./templates
COPY pyproject.toml ./

# Create data directories with proper permissions
RUN mkdir -p data/raw data/processed data/chroma_db && \
    chown -R appuser:appuser /app

# Ensure Python uses the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose ports
# 8000 - FastAPI main application
# 8001 - Prometheus metrics (optional)
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER appuser

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
