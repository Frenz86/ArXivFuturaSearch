# ArXiv Futura Search — common development commands
# Requires: GNU Make, uv (https://astral.sh/uv)
# On Windows use scripts/dev-setup.bat instead, or install GNU Make via Chocolatey.

.PHONY: help install dev test test-cov docker-dev docker-down clean

help:
	@echo "Available targets:"
	@echo "  make install      Install dependencies (uv sync)"
	@echo "  make dev          Start the app with auto-reload"
	@echo "  make test         Run the test suite"
	@echo "  make test-cov     Run tests and open HTML coverage report"
	@echo "  make docker-dev   Start full dev stack (ChromaDB + Redis + tools)"
	@echo "  make docker-down  Stop Docker containers"
	@echo "  make clean        Remove caches and build artifacts"

install:
	uv sync

dev: install
	uv run uvicorn app.main:app --reload --port 8000

test: install
	uv run pytest

test-cov: install
	uv run pytest --cov=app --cov-report=html:htmlcov
	@echo "Coverage report: htmlcov/index.html"

docker-dev:
	docker compose -f docker-compose.dev.yml up -d --build

docker-down:
	docker compose -f docker-compose.dev.yml down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache htmlcov data/exports
