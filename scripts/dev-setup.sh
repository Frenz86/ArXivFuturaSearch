#!/bin/bash
# =============================================================================
# Development Setup Script
# =============================================================================

set -e

echo "🚀 ArXivFuturaSearch - Development Setup"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your configuration (especially SECRET_KEY and API keys)"
    echo ""
fi

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -e .

# Create data directories
echo ""
echo "📁 Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/chroma_db
mkdir -p data/exports
mkdir -p logs

# Download NLTK data
echo ""
echo "📥 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Check Docker
echo ""
echo "🐳 Checking Docker..."
if command -v docker &> /dev/null; then
    echo "   Docker is installed"
    if command -v docker-compose &> /dev/null; then
        echo "   Docker Compose is installed"
    else
        echo "   ⚠️  Docker Compose not found. Install it from https://docs.docker.com/compose/install/"
    fi
else
    echo "   ⚠️  Docker not found. Install it from https://docs.docker.com/get-docker/"
fi

# Database migrations
echo ""
echo "🗄️  Running database migrations..."
if command -v alembic &> /dev/null; then
    alembic upgrade head || echo "   ⚠️  Migration failed (database might not be running)"
else
    echo "   ⚠️  Alembic not installed. Run: pip install alembic"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  Local:   uvicorn app.main:app --reload"
echo "  Docker:  docker-compose -f docker-compose.dev.yml up -d"
echo ""
echo "Available services:"
echo "  Application:   http://localhost:8000"
echo "  API Docs:      http://localhost:8000/api/docs"
echo "  Adminer:       http://localhost:8080"
echo "  Redis GUI:     http://localhost:8081"
echo "  Prometheus:    http://localhost:9090"
echo "  Grafana:       http://localhost:3000 (admin/admin)"
echo "  Jaeger:        http://localhost:16686"
