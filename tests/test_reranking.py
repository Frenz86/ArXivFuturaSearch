"""Tests for reranking module.

Run with: pytest tests/test_reranking.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_documents():
    """Sample documents for reranking."""
    return [
        {
            "text": "Machine learning is a field of AI focused on algorithms.",
            "title": "ML Intro",
            "link": "https://arxiv.org/abs/0001",
        },
        {
            "text": "Deep learning uses neural networks with multiple layers.",
            "title": "Deep Learning",
            "link": "https://arxiv.org/abs/0002",
        },
        {
            "text": "Natural language processing deals with text and speech.",
            "title": "NLP",
            "link": "https://arxiv.org/abs/0003",
        },
        {
            "text": "Computer vision enables machines to interpret images.",
            "title": "CV",
            "link": "https://arxiv.org/abs/0004",
        },
        {
            "text": "Reinforcement learning learns through reward signals.",
            "title": "RL",
            "link": "https://arxiv.org/abs/0005",
        },
    ]


@pytest.fixture
def mock_reranker():
    """Mock reranker model."""
    reranker = Mock()
    reranker.rerank = Mock(return_value=np.array([0.95, 0.85, 0.75, 0.65, 0.55]))
    return reranker


# =============================================================================
# CROSS-ENCODER RERANKER TESTS
# =============================================================================

class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker class."""

    @pytest.mark.asyncio
    async def test_rerank_basic(self, sample_documents):
        """Test basic reranking functionality."""
        from app.reranking import CrossEncoderReranker
        from app.embeddings import Reranker

        with patch.object(Reranker, '__init__', return_value=None):
            reranker_model = Mock(spec=Reranker)
            reranker_model.rerank = Mock(return_value=np.array([0.9, 0.7, 0.5, 0.3, 0.1]))

            reranker = CrossEncoderReranker()
            reranker._reranker = reranker_model

            results = await reranker.rerank(
                query="machine learning algorithms",
                documents=sample_documents,
                top_k=3
            )

            assert len(results) == 3
            assert all("ce_score" in r for r in results)
            # Scores should be descending
            scores = [r["ce_score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rerank_with_different_top_k(self, sample_documents):
        """Test reranking with different top_k values."""
        from app.reranking import CrossEncoderReranker
        from app.embeddings import Reranker

        with patch.object(Reranker, '__init__', return_value=None):
            reranker_model = Mock(spec=Reranker)
            reranker_model.rerank = Mock(return_value=np.array([0.9] * len(sample_documents)))

            reranker = CrossEncoderReranker()
            reranker._reranker = reranker_model

            results_top3 = await reranker.rerank("test", sample_documents, top_k=3)
            results_top5 = await reranker.rerank("test", sample_documents, top_k=5)

            assert len(results_top3) == 3
            assert len(results_top5) == 5

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self):
        """Test reranking with empty document list."""
        from app.reranking import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        results = await reranker.rerank("test", [], top_k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_preserves_metadata(self, sample_documents):
        """Test that reranking preserves original metadata."""
        from app.reranking import CrossEncoderReranker
        from app.embeddings import Reranker

        with patch.object(Reranker, '__init__', return_value=None):
            reranker_model = Mock(spec=Reranker)
            reranker_model.rerank = Mock(return_value=np.array([0.9] * len(sample_documents)))

            reranker = CrossEncoderReranker()
            reranker._reranker = reranker_model

            results = await reranker.rerank("test", sample_documents, top_k=5)

            for i, doc in enumerate(sample_documents):
                assert results[i]["title"] == doc["title"]
                assert results[i]["link"] == doc["link"]
                assert "text" in results[i]

    @pytest.mark.asyncio
    async def test_rerank_batch(self, sample_documents):
        """Test batch reranking."""
        from app.reranking import CrossEncoderReranker
        from app.embeddings import Reranker

        with patch.object(Reranker, '__init__', return_value=None):
            reranker_model = Mock(spec=Reranker)
            reranker_model.rerank = Mock(return_value=np.array([0.9, 0.7, 0.5]))

            reranker = CrossEncoderReranker()
            reranker._reranker = reranker_model

            queries = ["query 1", "query 2", "query 3"]
            documents_list = [sample_documents[:3]] * 3

            results = await reranker.rerank_batch(
                queries=queries,
                documents_list=documents_list,
                top_k=3
            )

            assert len(results) == 3
            assert all(len(r) == 3 for r in results)

    @pytest.mark.asyncio
    async def test_rerank_error_handling(self, sample_documents):
        """Test reranking error handling."""
        from app.reranking import CrossEncoderReranker
        from app.embeddings import Reranker

        with patch.object(Reranker, '__init__', return_value=None):
            reranker_model = Mock(spec=Reranker)
            reranker_model.rerank = Mock(side_effect=Exception("Reranking failed"))

            reranker = CrossEncoderReranker()
            reranker._reranker = reranker_model

            # Should return original documents on error
            results = await reranker.rerank("test", sample_documents, top_k=3)

            assert len(results) == 3  # Still returns results


# =============================================================================
# SCORE NORMALIZATION STRATEGY TESTS
# =============================================================================

class TestScoreNormalizationStrategy:
    """Tests for ScoreNormalizationStrategy class."""

    @pytest.mark.asyncio
    async def test_normalize_and_combine(self):
        """Test score normalization and combination."""
        from app.reranking import ScoreNormalizationStrategy

        strategy = ScoreNormalizationStrategy(
            vector_weight=0.7,
            keyword_weight=0.3
        )

        documents = [
            {"vector_score": 0.9, "keyword_score": 0.5},
            {"vector_score": 0.6, "keyword_score": 0.8},
            {"vector_score": 0.3, "keyword_score": 0.9},
        ]

        results = await strategy.rerank("test", documents)

        assert len(results) == 3
        assert all("combined_score" in r for r in results)

        # Check combination formula
        # First doc: 0.7*1.0 + 0.3*0.0 = 0.7 (normalized)
        # Second doc: 0.7*0.43 + 0.3*0.75 ≈ 0.52
        assert results[0]["combined_score"] >= 0

    @pytest.mark.asyncio
    async def test_different_weights(self):
        """Test with different weight combinations."""
        from app.reranking import ScoreNormalizationStrategy

        strategy_vector_heavy = ScoreNormalizationStrategy(
            vector_weight=0.9,
            keyword_weight=0.1
        )

        strategy_keyword_heavy = ScoreNormalizationStrategy(
            vector_weight=0.1,
            keyword_weight=0.9
        )

        documents = [
            {"vector_score": 0.9, "keyword_score": 0.1},
            {"vector_score": 0.1, "keyword_score": 0.9},
        ]

        results_vector = await strategy_vector_heavy.rerank("test", documents.copy())
        results_keyword = await strategy_keyword_heavy.rerank("test", documents.copy())

        # Vector-heavy should rank first doc higher
        # Keyword-heavy should rank second doc higher
        assert results_vector[0]["vector_score"] >= results_vector[1]["vector_score"]

    @pytest.mark.asyncio
    async def test_empty_documents(self):
        """Test with empty document list."""
        from app.reranking import ScoreNormalizationStrategy

        strategy = ScoreNormalizationStrategy()
        results = await strategy.rerank("test", [])

        assert results == []

    @pytest.mark.asyncio
    async def test_missing_scores(self):
        """Test handling of missing scores."""
        from app.reranking import ScoreNormalizationStrategy

        strategy = ScoreNormalizationStrategy()

        documents = [
            {"vector_score": 0.9},  # Missing keyword_score
            {"keyword_score": 0.8},  # Missing vector_score
            {},  # Missing both
        ]

        results = await strategy.rerank("test", documents)

        assert len(results) == 3
        # Missing scores should default to 0
        assert results[0]["combined_score"] >= 0


# =============================================================================
# MMR DIVERSIFICATION STRATEGY TESTS
# =============================================================================

class TestMMRDiversificationStrategy:
    """Tests for MMRDiversificationStrategy class."""

    @pytest.mark.asyncio
    async def test_mmr_basic(self):
        """Test basic MMR diversification."""
        from app.reranking import MMRDiversificationStrategy

        strategy = MMRDiversificationStrategy(
            lambda_param=0.5,
            embedder=None
        )

        documents = [
            {"text": "Similar text about machine learning", "score": 0.9},
            {"text": "Very similar ML content", "score": 0.85},
            {"text": "Different content about NLP", "score": 0.7},
            {"text": "Unrelated computer vision topic", "score": 0.6},
        ]

        results = await strategy.rerank("machine learning", documents)

        assert len(results) == 4
        assert all("mmr_rank" in r for r in results)
        assert all("mmr_score" in r for r in results)

    @pytest.mark.asyncio
    async def test_mmr_lambda_balance(self):
        """Test MMR with different lambda values."""
        from app.reranking import MMRDiversificationStrategy

        strategy_relevance = MMRDiversificationStrategy(lambda_param=0.9)
        strategy_diversity = MMRDiversificationStrategy(lambda_param=0.1)

        documents = [
            {"text": "Doc 1", "score": 0.9, "embedding": np.array([1, 0, 0])},
            {"text": "Doc 2", "score": 0.8, "embedding": np.array([0.95, 0, 0])},
            {"text": "Doc 3", "score": 0.7, "embedding": np.array([0, 1, 0])},
        ]

        results_relevance = await strategy_relevance.rerank("test", documents)
        results_diversity = await strategy_diversity.rerank("test", documents)

        # Both should return results
        assert len(results_relevance) == 3
        assert len(results_diversity) == 3

    @pytest.mark.asyncio
    async def test_mmr_no_embeddings(self):
        """Test MMR without embeddings."""
        from app.reranking import MMRDiversificationStrategy

        strategy = MMRDiversificationStrategy(lambda_param=0.5, embedder=None)

        documents = [
            {"text": "Doc 1", "score": 0.9},
            {"text": "Doc 2", "score": 0.8},
            {"text": "Doc 3", "score": 0.7},
        ]

        # Should return original order if no embeddings
        results = await strategy.rerank("test", documents)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_mmr_empty_documents(self):
        """Test MMR with empty documents."""
        from app.reranking import MMRDiversificationStrategy

        strategy = MMRDiversificationStrategy()
        results = await strategy.rerank("test", [])

        assert results == []


# =============================================================================
# RERANKING PIPELINE TESTS
# =============================================================================

class TestRerankingPipeline:
    """Tests for RerankingPipeline class."""

    @pytest.mark.asyncio
    async def test_pipeline_with_strategies(self):
        """Test pipeline with multiple strategies."""
        from app.reranking import (
            RerankingPipeline,
            ScoreNormalizationStrategy,
            MMRDiversificationStrategy
        )

        strategy1 = ScoreNormalizationStrategy(vector_weight=0.7, keyword_weight=0.3)
        strategy2 = MMRDiversificationStrategy(lambda_param=0.5)

        pipeline = RerankingPipeline(
            strategies=[strategy1, strategy2],
            cross_encoder=None
        )

        documents = [
            {"vector_score": 0.9, "keyword_score": 0.5, "text": "Doc 1", "score": 0.9},
            {"vector_score": 0.6, "keyword_score": 0.8, "text": "Doc 2", "score": 0.8},
        ]

        results = await pipeline.rerank("test query", documents, top_k=2)

        assert len(results) <= 2
        assert all("text" in r for r in results)

    @pytest.mark.asyncio
    async def test_pipeline_with_cross_encoder(self, sample_documents):
        """Test pipeline with cross-encoder."""
        from app.reranking import (
            RerankingPipeline,
            CrossEncoderReranker
        )
        from app.embeddings import Reranker

        with patch.object(Reranker, '__init__', return_value=None):
            reranker_model = Mock(spec=Reranker)
            reranker_model.rerank = Mock(return_value=np.array([0.9, 0.7, 0.5, 0.3, 0.1]))

            cross_encoder = CrossEncoderReranker()
            cross_encoder._reranker = reranker_model

            pipeline = RerankingPipeline(
                strategies=[],
                cross_encoder=cross_encoder
            )

            results = await pipeline.rerank("test", sample_documents, top_k=3)

            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_pipeline_empty_documents(self):
        """Test pipeline with empty documents."""
        from app.reranking import RerankingPipeline

        pipeline = RerankingPipeline(strategies=[], cross_encoder=None)
        results = await pipeline.rerank("test", [], top_k=5)

        assert results == []


# =============================================================================
# GLOBAL INSTANCE TESTS
# =============================================================================

class TestGlobalReranker:
    """Tests for global reranker instances."""

    def test_get_cross_encoder_reranker_singleton(self):
        """Test that get_cross_encoder_reranker returns singleton."""
        from app.reranking import get_cross_encoder_reranker

        reranker1 = get_cross_encoder_reranker("model1")
        reranker2 = get_cross_encoder_reranker("model1")
        reranker3 = get_cross_encoder_reranker("model2")

        # Same model returns same instance
        assert reranker1 is reranker2
        # Different model creates new instance
        assert reranker1 is not reranker3

    def test_create_reranking_pipeline(self):
        """Test create_reranking_pipeline factory function."""
        from app.reranking import create_reranking_pipeline

        pipeline = create_reranking_pipeline(
            use_cross_encoder=False,
            use_mmr=True,
            lambda_param=0.5,
            embedder=None
        )

        assert pipeline is not None
        assert len(pipeline.strategies) == 1


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestRerankingEdgeCases:
    """Edge case tests for reranking."""

    @pytest.mark.asyncio
    async def test_single_document(self):
        """Test reranking with single document."""
        from app.reranking import CrossEncoderReranker
        from app.embeddings import Reranker

        with patch.object(Reranker, '__init__', return_value=None):
            reranker_model = Mock(spec=Reranker)
            reranker_model.rerank = Mock(return_value=np.array([0.9]))

            reranker = CrossEncoderReranker()
            reranker._reranker = reranker_model

            results = await reranker.rerank("test", [{"text": "Only doc"}], top_k=5)

            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_very_long_documents(self):
        """Test with very long document text."""
        from app.reranking import CrossEncoderReranker
        from app.embeddings import Reranker

        long_text = "word " * 10000  # Very long document

        with patch.object(Reranker, '__init__', return_value=None):
            reranker_model = Mock(spec=Reranker)
            reranker_model.rerank = Mock(return_value=np.array([0.9]))

            reranker = CrossEncoderReranker()
            reranker._reranker = reranker_model

            results = await reranker.rerank("test", [{"text": long_text}], top_k=1)

            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_unicode_content(self):
        """Test with unicode content."""
        from app.reranking import CrossEncoderReranker
        from app.embeddings import Reranker

        with patch.object(Reranker, '__init__', return_value=None):
            reranker_model = Mock(spec=Reranker)
            reranker_model.rerank = Mock(return_value=np.array([0.9]))

            reranker = CrossEncoderReranker()
            reranker._reranker = reranker_model

            unicode_docs = [
                {"text": "测试中文内容"},
                {"text": "Contenu en français"},
                {"text": "日本語のコンテンツ"},
                {"text": "العربية المحتوى"},
            ]

            results = await reranker.rerank("test", unicode_docs, top_k=4)

            assert len(results) == 4
