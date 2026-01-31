"""Tests for FastAPI endpoints.

Run with: pytest tests/test_api.py
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from typing import Generator

# Import the FastAPI app
from app.main import app, _store
from app.config import settings


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def client() -> Generator:
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_store():
    """Mock vector store."""
    store = Mock()
    store.count.return_value = 100
    store.search.return_value = [
        {
            "text": "Test document about RAG systems",
            "meta": {
                "title": "Test Paper",
                "link": "https://arxiv.org/abs/1234.5678",
                "published": "2024-01-01",
                "authors": "Test Author",
            },
            "score": 0.85,
        }
    ]
    store.search_ensemble.return_value = store.search.return_value
    return store


@pytest.fixture
def mock_embedder():
    """Mock embedder."""
    embedder = Mock()
    embedder.embed_query.return_value = [0.1] * 1024
    embedder.embed.return_value = [[0.1] * 1024]
    embedder.dimension = 1024
    embedder.is_e5_model = True
    return embedder


@pytest.fixture
def mock_cache():
    """Mock cache client."""
    cache = Mock()
    cache.enabled = True
    cache.get.return_value = None
    cache.set.return_value = True
    cache.get_stats.return_value = {
        "enabled": True,
        "status": "connected",
        "hits": 10,
        "misses": 5,
        "keys": 15,
    }
    cache.clear_pattern.return_value = 5
    return cache


# =============================================================================
# HEALTH ENDPOINT TESTS
# =============================================================================

class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client, mock_store, mock_embedder):
        """Test basic health check."""
        with patch('app.main.get_store', return_value=mock_store):
            with patch('app.main.get_embedder', return_value=mock_embedder):
                with patch('app.rag.check_llm_health', return_value={"healthy": True}):
                    response = client.get("/health")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "ok"
                    assert data["index_loaded"] is True
                    assert data["embedder_loaded"] is True
                    assert "llm_health" in data

    def test_health_no_index(self, client, mock_embedder):
        """Test health check when index is not loaded."""
        with patch('app.main.get_store', side_effect=Exception("No index")):
            with patch('app.main.get_embedder', return_value=mock_embedder):
                with patch('app.rag.check_llm_health', return_value={"healthy": True}):
                    response = client.get("/health")

                    # Health endpoint should still work even without index
                    assert response.status_code in [200, 503]


# =============================================================================
# CONFIG ENDPOINT TESTS
# =============================================================================

class TestConfigEndpoint:
    """Tests for /config endpoint."""

    def test_get_config(self, client, mock_embedder):
        """Test getting configuration."""
        with patch('app.main.get_embedder', return_value=mock_embedder):
            response = client.get("/config")

            assert response.status_code == 200
            data = response.json()
            assert "version" in data
            assert "vectorstore_mode" in data
            assert "llm_mode" in data
            assert "embed_model" in data
            # Should not expose sensitive data
            assert "OPENROUTER_API_KEY" not in data
            assert "api_key" not in str(data).lower()


# =============================================================================
# SEARCH ENDPOINT TESTS
# =============================================================================

class TestSearchEndpoint:
    """Tests for /search endpoint."""

    def test_search_without_filters(self, client, mock_store, mock_embedder):
        """Test search without date filters."""
        with patch('app.main.get_store', return_value=mock_store):
            with patch('app.main.get_embedder', return_value=mock_embedder):
                response = client.get("/search?q=RAG&top_k=5")

                assert response.status_code == 200
                data = response.json()
                assert "query" in data
                assert "results" in data
                assert len(data["results"]) <= 5

    def test_search_with_filters(self, client, mock_store, mock_embedder):
        """Test search with date filters."""
        with patch('app.main.get_store', return_value=mock_store):
            with patch('app.main.get_embedder', return_value=mock_embedder):
                response = client.get(
                    "/search?q=RAG&top_k=5&published_after=2023-01-01&published_before=2024-12-31"
                )

                assert response.status_code == 200
                data = response.json()
                assert "results" in data


# =============================================================================
# CACHE ENDPOINT TESTS
# =============================================================================

class TestCacheEndpoints:
    """Tests for cache-related endpoints."""

    def test_cache_stats(self, client, mock_cache):
        """Test getting cache statistics."""
        with patch('app.main.get_cache', return_value=mock_cache):
            response = client.get("/cache/stats")

            assert response.status_code == 200
            data = response.json()
            assert "enabled" in data
            assert "status" in data

    def test_cache_clear(self, client, mock_cache):
        """Test clearing cache."""
        with patch('app.main.get_cache', return_value=mock_cache):
            response = client.post("/cache/clear?pattern=test:*")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "deleted" in data


# =============================================================================
# INDEX ENDPOINT TESTS
# =============================================================================

class TestIndexEndpoints:
    """Tests for index-related endpoints."""

    def test_list_papers(self, client, mock_store):
        """Test listing indexed papers."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "documents": ["Test content 1", "Test content 2"],
            "metadatas": [
                {
                    "title": "Paper 1",
                    "authors": "Author 1",
                    "link": "https://arxiv.org/abs/0001",
                    "published": "2024-01-01",
                    "tags": "cs.AI",
                },
                {
                    "title": "Paper 2",
                    "authors": "Author 2",
                    "link": "https://arxiv.org/abs/0002",
                    "published": "2024-01-02",
                    "tags": "cs.LG",
                },
            ],
        }
        mock_store.client.get_collection.return_value = mock_collection
        mock_store.collection_name = "arxiv_papers"

        with patch('app.main.get_store', return_value=mock_store):
            response = client.get("/index/papers?limit=10")

            assert response.status_code == 200
            data = response.json()
            assert "papers" in data
            assert "total" in data
            assert isinstance(data["papers"], list)


# =============================================================================
# ASK ENDPOINT TESTS
# =============================================================================

class TestAskEndpoint:
    """Tests for /ask endpoint."""

    def test_ask_non_streaming(self, client, mock_store, mock_embedder, mock_cache):
        """Test ask endpoint with non-streaming response."""
        # Setup mocks
        mock_store.search.return_value = [
            {
                "text": "RAG is a system that combines retrieval with generation",
                "meta": {
                    "title": "RAG Systems",
                    "link": "https://arxiv.org/abs/1234",
                    "published": "2024-01-01",
                    "authors": "Test Author",
                },
                "score": 0.9,
            }
        ]

        with patch('app.main.get_store', return_value=mock_store):
            with patch('app.main.get_embedder', return_value=mock_embedder):
                with patch('app.main.get_cache', return_value=mock_cache):
                    with patch('app.main.llm_generate_async', return_value="Test answer about RAG"):
                        response = client.post(
                            "/ask",
                            json={
                                "question": "What is RAG?",
                                "top_k": 5,
                                "stream": False,
                            }
                        )

                        assert response.status_code == 200
                        data = response.json()
                        assert "answer" in data
                        assert "sources" in data
                        assert "retrieval_info" in data

    def test_ask_streaming(self, client, mock_store, mock_embedder, mock_cache):
        """Test ask endpoint with streaming response."""
        mock_store.search.return_value = [
            {
                "text": "Test content",
                "meta": {
                    "title": "Test",
                    "link": "https://test.com",
                    "published": "2024-01-01",
                    "authors": "Test",
                },
                "score": 0.9,
            }
        ]

        async def mock_stream(prompt):
            yield "Hello"
            yield " world"

        with patch('app.main.get_store', return_value=mock_store):
            with patch('app.main.get_embedder', return_value=mock_embedder):
                with patch('app.main.get_cache', return_value=mock_cache):
                    with patch('app.main.llm_generate_stream', side_effect=mock_stream):
                        response = client.post(
                            "/ask",
                            json={
                                "question": "Test question",
                                "stream": True,
                            }
                        )

                        # Streaming returns text/event-stream
                        assert response.status_code == 200
                        assert "text/event-stream" in response.headers.get("content-type", "")


# =============================================================================
# METRICS ENDPOINT TESTS
# =============================================================================

class TestMetricsEndpoints:
    """Tests for metrics endpoints."""

    def test_metrics_summary(self, client):
        """Test getting metrics summary."""
        with patch.object(settings, 'METRICS_ENABLED', True):
            response = client.get("/metrics/summary")

            assert response.status_code == 200
            data = response.json()
            assert "requests" in data
            assert "cache" in data

    def test_metrics_disabled(self, client):
        """Test metrics endpoint when disabled."""
        with patch.object(settings, 'METRICS_ENABLED', False):
            response = client.get("/metrics/summary")

            assert response.status_code == 404


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_ask_no_index(self, client, mock_embedder, mock_cache):
        """Test ask when index is not built."""
        with patch('app.main.get_store', side_effect=Exception("No index")):
            with patch('app.main.get_embedder', return_value=mock_embedder):
                response = client.post(
                    "/ask",
                    json={"question": "Test", "stream": False}
                )

                # Should return error
                assert response.status_code in [500, 503]

    def test_invalid_top_k(self, client):
        """Test ask with invalid top_k value."""
        response = client.post(
            "/ask",
            json={"question": "Test", "top_k": 100}
        )

        # Pydantic validation should catch this
        assert response.status_code == 422
