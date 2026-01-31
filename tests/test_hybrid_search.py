"""Tests for hybrid search module.

Run with: pytest tests/test_hybrid_search.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "meta": {"title": "ML Intro", "category": "cs.AI"}
        },
        {
            "text": "Deep learning uses neural networks for feature extraction.",
            "meta": {"title": "Deep Learning", "category": "cs.LG"}
        },
        {
            "text": "Natural language processing deals with text understanding.",
            "meta": {"title": "NLP", "category": "cs.CL"}
        },
        {
            "text": "Computer vision enables machines to interpret visual information.",
            "meta": {"title": "CV", "category": "cs.CV"}
        },
        {
            "text": "Reinforcement learning learns through trial and error.",
            "meta": {"title": "RL", "category": "cs.AI"}
        },
    ]


@pytest.fixture
def mock_vectorstore():
    """Mock vector store for testing."""
    store = Mock()
    store.similarity_search_with_score.return_value = [
        (Mock(page_content="Machine learning is a subset of AI", metadata={"title": "ML"}), 0.95),
        (Mock(page_content="Deep learning uses neural networks", metadata={"title": "DL"}), 0.85),
    ]
    return store


# =============================================================================
# BM25 INDEX TESTS
# =============================================================================

class TestBM25Index:
    """Tests for BM25Index class."""

    @pytest.mark.asyncio
    async def test_bm25_index_documents(self, sample_documents):
        """Test indexing documents for BM25."""
        from app.hybrid_search import BM25Index

        bm25 = BM25Index()
        await bm25.index_documents(sample_documents)

        assert bm25._indexed is True
        assert len(bm25._documents) == len(sample_documents)
        assert bm25._corpus is not None

    @pytest.mark.asyncio
    async def test_bm25_search(self, sample_documents):
        """Test BM25 search functionality."""
        from app.hybrid_search import BM25Index

        bm25 = BM25Index()
        await bm25.index_documents(sample_documents)

        results = bm25.search("machine learning neural networks", top_k=3)

        assert len(results) <= 3
        assert all("score" in r for r in results)
        assert all("text" in r for r in results)
        # Scores should be descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_bm25_empty_search(self):
        """Test BM25 search with no documents indexed."""
        from app.hybrid_search import BM25Index

        bm25 = BM25Index()
        results = bm25.search("test query", top_k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_bm25_index_empty_list(self):
        """Test BM25 with empty document list."""
        from app.hybrid_search import BM25Index

        bm25 = BM25Index()
        await bm25.index_documents([])

        assert bm25._indexed is False


# =============================================================================
# HYBRID SEARCH ENGINE TESTS
# =============================================================================

class TestHybridSearchEngine:
    """Tests for HybridSearchEngine class."""

    @pytest.mark.asyncio
    async def test_hybrid_search_initialization(self, mock_vectorstore):
        """Test hybrid search engine initialization."""
        from app.hybrid_search import HybridSearchEngine, get_hybrid_search_engine

        engine = HybridSearchEngine(mock_vectorstore)
        assert engine._vectorstore == mock_vectorstore
        assert engine._indexed is False

    @pytest.mark.asyncio
    async def test_hybrid_search_with_rrf(self, sample_documents, mock_vectorstore):
        """Test hybrid search with Reciprocal Rank Fusion."""
        from app.hybrid_search import HybridSearchEngine

        # Setup vector store mock
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(page_content=d["text"], metadata=d["meta"]), 0.9 - i * 0.1)
            for i, d in enumerate(sample_documents[:3])
        ]

        engine = HybridSearchEngine(mock_vectorstore)
        await engine.index_documents(sample_documents)

        results = await engine.search(
            query="machine learning and neural networks",
            top_k=3,
            alpha=0.5  # Equal weight for semantic and BM25
        )

        assert len(results) <= 3
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)
        # Check that scores are normalized and descending
        scores = [r["score"] for r in results]
        assert all(0 <= s <= 1 for s in scores)
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_hybrid_search_alpha_weights(self, sample_documents, mock_vectorstore):
        """Test hybrid search with different alpha weights."""
        from app.hybrid_search import HybridSearchEngine

        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(page_content=d["text"], metadata=d["meta"]), 0.9 - i * 0.1)
            for i, d in enumerate(sample_documents[:3])
        ]

        engine = HybridSearchEngine(mock_vectorstore)
        await engine.index_documents(sample_documents)

        # Test with alpha=1.0 (pure semantic)
        results_semantic = await engine.search(
            query="test",
            top_k=3,
            alpha=1.0
        )

        # Test with alpha=0.0 (pure BM25)
        results_bm25 = await engine.search(
            query="test",
            top_k=3,
            alpha=0.0
        )

        assert len(results_semantic) > 0
        assert len(results_bm25) > 0

    @pytest.mark.asyncio
    async def test_hybrid_search_rrf_fusion(self, sample_documents, mock_vectorstore):
        """Test RRF fusion method."""
        from app.hybrid_search import HybridSearchEngine

        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(page_content=d["text"], metadata=d["meta"]), 0.9 - i * 0.1)
            for i, d in enumerate(sample_documents[:3])
        ]

        engine = HybridSearchEngine(mock_vectorstore)
        await engine.index_documents(sample_documents)

        # Test with different RRF k values
        results_k60 = await engine.search("test", top_k=3, rrf_k=60)
        results_k10 = await engine.search("test", top_k=3, rrf_k=10)

        assert len(results_k60) > 0
        assert len(results_k10) > 0

    @pytest.mark.asyncio
    async def test_hybrid_search_empty_query(self, sample_documents, mock_vectorstore):
        """Test hybrid search with empty query."""
        from app.hybrid_search import HybridSearchEngine

        engine = HybridSearchEngine(mock_vectorstore)
        await engine.index_documents(sample_documents)

        results = await engine.search("", top_k=3)

        # Should return results even with empty query (BM25 only)
        assert isinstance(results, list)


# =============================================================================
# GLOBAL INSTANCE TESTS
# =============================================================================

class TestGlobalHybridSearch:
    """Tests for global hybrid search instance."""

    def test_get_hybrid_search_engine_singleton(self, mock_vectorstore):
        """Test that get_hybrid_search_engine returns singleton."""
        from app.hybrid_search import get_hybrid_search_engine

        engine1 = get_hybrid_search_engine(mock_vectorstore)
        engine2 = get_hybrid_search_engine(mock_vectorstore)

        # Same vectorstore should return same engine instance
        assert engine1 is engine2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestHybridSearchIntegration:
    """Integration tests for hybrid search."""

    @pytest.mark.asyncio
    async def test_full_hybrid_search_workflow(self):
        """Test complete hybrid search workflow."""
        from app.hybrid_search import HybridSearchEngine

        # Create mock vector store
        mock_store = Mock()
        mock_store.similarity_search_with_score.return_value = [
            (Mock(page_content="Machine learning algorithms", metadata={"title": "ML"}), 0.95),
            (Mock(page_content="Neural network architectures", metadata={"title": "NN"}), 0.85),
            (Mock(page_content="Statistical methods", metadata={"title": "Stats"}), 0.75),
        ]

        docs = [
            {"text": "Machine learning algorithms are powerful", "meta": {"title": "ML"}},
            {"text": "Neural network architectures learn patterns", "meta": {"title": "NN"}},
            {"text": "Statistical methods provide foundations", "meta": {"title": "Stats"}},
        ]

        engine = HybridSearchEngine(mock_store)
        await engine.index_documents(docs)

        results = await engine.search(
            query="machine learning neural networks",
            top_k=2,
            alpha=0.6,
            rrf_k=60
        )

        assert len(results) == 2
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)
        assert all("meta" in r for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self):
        """Test hybrid search with metadata filters."""
        from app.hybrid_search import HybridSearchEngine

        mock_store = Mock()
        mock_store.similarity_search_with_score.return_value = [
            (Mock(page_content="ML content", metadata={"category": "cs.AI"}), 0.9),
            (Mock(page_content="NLP content", metadata={"category": "cs.CL"}), 0.8),
        ]

        engine = HybridSearchEngine(mock_store)

        results = await engine.search(
            query="test",
            top_k=5,
            search_kwargs={"filter": {"category": "cs.AI"}}
        )

        assert isinstance(results, list)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestHybridSearchEdgeCases:
    """Edge case tests for hybrid search."""

    @pytest.mark.asyncio
    async def test_single_document(self):
        """Test with only one document."""
        from app.hybrid_search import HybridSearchEngine

        mock_store = Mock()
        mock_store.similarity_search_with_score.return_value = [
            (Mock(page_content="Single document", metadata={}), 1.0),
        ]

        engine = HybridSearchEngine(mock_store)
        await engine.index_documents([{"text": "Single", "meta": {}}])

        results = await engine.search("test", top_k=5)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_large_document_set(self):
        """Test with large number of documents."""
        from app.hybrid_search import HybridSearchEngine

        mock_store = Mock()
        docs = [{"text": f"Document {i}", "meta": {"id": i}} for i in range(100)]

        mock_store.similarity_search_with_score.return_value = [
            (Mock(page_content=d["text"], metadata=d["meta"]), 0.9)
            for d in docs[:10]
        ]

        engine = HybridSearchEngine(mock_store)
        await engine.index_documents(docs)

        results = await engine.search("test", top_k=10)

        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self):
        """Test query with special characters."""
        from app.hybrid_search import HybridSearchEngine

        mock_store = Mock()
        mock_store.similarity_search_with_score.return_value = []

        engine = HybridSearchEngine(mock_store)
        await engine.index_documents([{"text": "Test", "meta": {}}])

        # Should not crash with special characters
        results = await engine.search("test?!@#$%^&*()", top_k=5)

        assert isinstance(results, list)
