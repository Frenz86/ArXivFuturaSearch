"""Shared pytest fixtures and configuration.

Run with: pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Generator

# Add app directory to path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_settings():
    """Mock application settings."""
    from app.config import settings

    mock_settings = Mock()
    mock_settings.APP_NAME = "arxiv_futura_search"
    mock_settings.VERSION = "0.4.0"
    mock_settings.ENVIRONMENT = "test"
    mock_settings.EMBED_MODEL = "intfloat/multilingual-e5-large"
    mock_settings.LLM_MODE = "mock"
    mock_settings.VECTORSTORE_MODE = "chroma"
    mock_settings.CHROMA_DIR = "/tmp/test_chroma"
    mock_settings.TOP_K = 5
    mock_settings.RETRIEVAL_K = 20
    mock_settings.CHUNK_SIZE = 900
    mock_settings.CHUNK_OVERLAP = 150
    mock_settings.SEMANTIC_WEIGHT = 0.7
    mock_settings.BM25_WEIGHT = 0.3
    mock_settings.RERANK_ENABLED = True
    mock_settings.QUERY_EXPANSION_ENABLED = True
    mock_settings.CACHE_ENABLED = False
    mock_settings.METRICS_ENABLED = True

    return mock_settings


@pytest.fixture
def mock_embedder():
    """Mock embedder instance."""
    embedder = Mock()
    embedder.dimension = 1024
    embedder.is_e5_model = True
    embedder.embed_query = Mock(return_value=[0.1] * 1024)
    embedder.embed = Mock(return_value=[[0.1] * 1024, [0.2] * 1024])
    return embedder


@pytest.fixture
def mock_reranker():
    """Mock reranker instance."""
    reranker = Mock()
    reranker.rerank = Mock(return_value=[0.9, 0.8, 0.7, 0.6, 0.5])
    return reranker


@pytest.fixture
def mock_llm():
    """Mock LLM instance."""
    llm = Mock()
    llm.invoke = Mock(return_value=Mock(
        content="This is a test response from the LLM."
    ))
    llm.stream = Mock(return_value=[
        "This ",
        "is ",
        "a ",
        "test ",
        "response."
    ])
    return llm


@pytest.fixture
def mock_vectorstore():
    """Mock vector store instance."""
    store = Mock()
    store.count = Mock(return_value=100)
    store.search = Mock(return_value=[
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
    ])
    store.add_documents = AsyncMock()
    store.delete_documents = AsyncMock()
    store.vectorstore = Mock()
    return store


@pytest.fixture
def mock_cache():
    """Mock cache instance."""
    cache = Mock()
    cache.enabled = True
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    cache.clear_pattern = AsyncMock(return_value=5)
    cache.get_stats = Mock(return_value={
        "enabled": True,
        "status": "connected",
        "hits": 10,
        "misses": 5,
        "keys": 15,
    })
    return cache


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_papers():
    """Sample ArXiv papers for testing."""
    return [
        {
            "id": "2401.00001",
            "title": "Advances in Retrieval-Augmented Generation",
            "summary": "We present novel techniques for improving RAG systems through better retrieval and generation.",
            "authors": ["Alice Johnson", "Bob Smith", "Carol Williams"],
            "published": "2024-01-15T00:00:00Z",
            "link": "https://arxiv.org/abs/2401.00001",
            "tags": ["cs.CL", "cs.AI"],
        },
        {
            "id": "2401.00002",
            "title": "Transformer Architecture Improvements",
            "summary": "This paper explores improvements to the transformer architecture for better performance.",
            "authors": ["David Lee"],
            "published": "2024-01-16T00:00:00Z",
            "link": "https://arxiv.org/abs/2401.00002",
            "tags": ["cs.LG", "cs.AI"],
        },
        {
            "id": "2401.00003",
            "title": "Efficient Neural Network Training",
            "summary": "We propose new methods for training neural networks more efficiently.",
            "authors": ["Emma Garcia", "Frank Martinez"],
            "published": "2024-01-17T00:00:00Z",
            "link": "https://arxiv.org/abs/2401.00003",
            "tags": ["cs.LG"],
        },
    ]


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    return [
        {
            "text": "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval systems with language models.",
            "meta": {
                "title": "RAG Introduction",
                "paper_id": "2401.00001",
                "chunk_id": "chunk_0",
            },
        },
        {
            "text": "Vector databases enable efficient semantic search through embeddings.",
            "meta": {
                "title": "Vector Databases",
                "paper_id": "2401.00001",
                "chunk_id": "chunk_1",
            },
        },
        {
            "text": "Cross-encoder reranking improves retrieval precision by scoring query-document pairs.",
            "meta": {
                "title": "Reranking Methods",
                "paper_id": "2401.00002",
                "chunk_id": "chunk_0",
            },
        },
    ]


@pytest.fixture
def sample_queries():
    """Sample search queries for testing."""
    return [
        "What is retrieval-augmented generation?",
        "How do transformer models work?",
        "Explain vector database architecture",
        "What is cross-encoder reranking?",
        "Deep learning vs machine learning",
    ]


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture
def temp_chroma_db(tmp_path):
    """Create temporary ChromaDB for testing."""
    import chromadb
    from chromadb.config import Settings

    db_dir = tmp_path / "chroma_test"
    db_dir.mkdir()

    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=settings=Settings(anonymized_telemetry=False)
    )

    yield client

    # Cleanup
    import shutil
    if db_dir.exists():
        shutil.rmtree(db_dir)


@pytest.fixture
def temp_postgres(tmp_path):
    """Mock PostgreSQL connection for testing."""
    # Return mock instead of real connection for tests
    mock_conn = Mock()
    mock_conn.cursor = Mock()
    mock_conn.close = Mock()
    return mock_conn


# =============================================================================
# API CLIENT FIXTURES
# =============================================================================

@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers():
    """Mock authentication headers."""
    return {
        "Authorization": "Bearer test_token",
        "Content-Type": "application/json",
    }


# =============================================================================
# ASYNC FIXTURES
# =============================================================================

@pytest.fixture
async def async_mock_store():
    """Async mock store fixture."""
    store = Mock()
    store.count = Mock(return_value=100)
    store.search = Mock(return_value=[
        {
            "text": "Test document",
            "meta": {"title": "Test"},
            "score": 0.9,
        }
    ])
    store.add_documents = AsyncMock()
    return store


@pytest.fixture
async def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# METRICS FIXTURES
# =============================================================================

@pytest.fixture
def reset_metrics():
    """Reset metrics before/after tests."""
    from app import metrics

    # Store original state
    original_state = {}

    yield

    # Reset metrics if needed
    # This would be implemented based on actual metrics module


# =============================================================================
# COVERAGE CONFIGURATION
# =============================================================================

@pytest.fixture
def coverage_config():
    """Coverage configuration for tests."""
    return {
        "omit": [
            "*/tests/*",
            "*/test_*.py",
            "*/__pycache__/*",
            "*/site-packages/*",
        ],
        "branch": True,
        "source": ["app"],
    }


# =============================================================================
# INTEGRATION TEST MARKERS
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """Modify test collection for integration tests."""
    for item in items:
        # Add asyncio marker to async tests
        if asyncio.iscoroutinefunction(item.obj):
            item.add_marker(pytest.mark.asyncio)


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_state():
    """Clean up test state before and after each test."""
    # Setup
    yield

    # Teardown - clean up any global state
    # Reset singletons if needed
    import sys
    modules_to_reload = [
        'app.dependencies',
        'app.metrics',
    ]
    for mod in modules_to_reload:
        if mod in sys.modules:
            # Force reload on next import
            del sys.modules[mod]
