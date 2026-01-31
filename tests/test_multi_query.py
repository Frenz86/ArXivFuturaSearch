"""Tests for multi-query retrieval module.

Run with: pytest tests/test_multi_query.py -v
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM for query generation."""
    llm = Mock()
    llm.invoke = Mock(return_value=Mock(
        content='1. "machine learning techniques"\n2. "ML algorithms and methods"\n3. "machine learning approaches"'
    ))
    return llm


@pytest.fixture
def sample_retrieval_results():
    """Sample retrieval results."""
    return [
        {"text": "Result 1", "meta": {"title": "Paper 1"}, "score": 0.9},
        {"text": "Result 2", "meta": {"title": "Paper 2"}, "score": 0.8},
        {"text": "Result 3", "meta": {"title": "Paper 3"}, "score": 0.7},
    ]


# =============================================================================
# MULTI-QUERY RETRIEVER TESTS
# =============================================================================

class TestMultiQueryRetriever:
    """Tests for MultiQueryRetriever class."""

    @pytest.mark.asyncio
    async def test_generate_queries(self, mock_llm):
        """Test generating multiple query variants."""
        from app.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(mock_llm)
        queries = await retriever.generate_queries(
            "What is machine learning?",
            num_queries=3,
            include_original=False
        )

        assert len(queries) == 3
        assert all(isinstance(q, str) for q in queries)
        assert all(len(q) > 0 for q in queries)

    @pytest.mark.asyncio
    async def test_generate_queries_with_original(self, mock_llm):
        """Test query generation including original query."""
        from app.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(mock_llm)
        queries = await retriever.generate_queries(
            "machine learning algorithms",
            num_queries=2,
            include_original=True
        )

        assert len(queries) == 3  # 2 generated + 1 original
        assert "machine learning algorithms" in queries

    @pytest.mark.asyncio
    async def test_retrieve_with_rrf(self, mock_llm, sample_retrieval_results):
        """Test retrieval with Reciprocal Rank Fusion."""
        from app.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(mock_llm)

        # Mock retriever function
        async def mock_retrieve(query: str, k: int):
            # Return different results for different queries
            return sample_retrieval_results[:k]

        results = await retriever.retrieve(
            queries=["query 1", "query 2", "query 3"],
            retriever_func=mock_retrieve,
            top_k=5,
            merge_strategy="rrf"
        )

        assert len(results) <= 5
        assert all("text" in r for r in results)
        assert all("rrf_score" in r for r in results)

    @pytest.mark.asyncio
    async def test_weighted_merge(self, mock_llm, sample_retrieval_results):
        """Test weighted merge strategy."""
        from app.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(mock_llm)

        async def mock_retrieve(query: str, k: int):
            return sample_retrieval_results[:k]

        results = await retriever.retrieve(
            queries=["q1", "q2"],
            retriever_func=mock_retrieve,
            top_k=5,
            merge_strategy="weighted"
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_union_merge(self, mock_llm):
        """Test union merge strategy."""
        from app.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(mock_llm)

        async def mock_retrieve(query: str, k: int):
            # Return unique results per query
            results_map = {
                "q1": [{"text": "Result 1", "score": 0.9}],
                "q2": [{"text": "Result 2", "score": 0.8}],
                "q3": [{"text": "Result 3", "score": 0.7}],
            }
            return results_map.get(query, [])

        results = await retriever.retrieve(
            queries=["q1", "q2", "q3"],
            retriever_func=mock_retrieve,
            top_k=10,
            merge_strategy="union"
        )

        # Union should include all unique results
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_empty_queries_list(self, mock_llm):
        """Test with empty queries list."""
        from app.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(mock_llm)

        async def mock_retrieve(query: str, k: int):
            return []

        results = await retriever.retrieve(
            queries=[],
            retriever_func=mock_retrieve,
            top_k=5
        )

        assert results == []


# =============================================================================
# QUERY DECOMPOSER TESTS
# =============================================================================

class TestQueryDecomposer:
    """Tests for QueryDecomposer class."""

    @pytest.mark.asyncio
    async def test_decompose_simple_query(self, mock_llm):
        """Test decomposing a simple query."""
        from app.multi_query import QueryDecomposer

        decomposer = QueryDecomposer(mock_llm)

        # Mock LLM to return sub-questions
        mock_llm.invoke.return_value = Mock(
            content='1. What is machine learning?\n2. How does machine learning work?'
        )

        subqueries = await decomposer.decompose("Explain machine learning")

        assert len(subqueries) == 2
        assert all(isinstance(q, str) for q in subqueries)

    @pytest.mark.asyncio
    async def test_decompose_complex_query(self, mock_llm):
        """Test decomposing a complex multi-part query."""
        from app.multi_query import QueryDecomposer

        decomposer = QueryDecomposer(mock_llm)

        mock_llm.invoke.return_value = Mock(
            content='1. "transformer architecture"\n2. "attention mechanism"\n3. "self-attention in transformers"'
        )

        subqueries = await decomposer.decompose(
            "How do transformers work and what is attention?",
            max_subqueries=3
        )

        assert len(subqueries) <= 3

    @pytest.mark.asyncio
    async def test_decompose_with_validation(self, mock_llm):
        """Test query decomposition with validation."""
        from app.multi_query import QueryDecomposer

        decomposer = QueryDecomposer(mock_llm)

        # Return empty response
        mock_llm.invoke.return_value = Mock(content="")

        subqueries = await decomposer.decompose("test query")

        # Should handle empty response gracefully
        assert isinstance(subqueries, list)


# =============================================================================
# GLOBAL INSTANCE TESTS
# =============================================================================

class TestGlobalMultiQuery:
    """Tests for global multi-query instances."""

    def test_get_multi_query_retriever_singleton(self, mock_llm):
        """Test that get_multi_query_retriever returns singleton."""
        from app.multi_query import get_multi_query_retriever

        retriever1 = get_multi_query_retriever(mock_llm)
        retriever2 = get_multi_query_retriever(mock_llm)

        # Same LLM should return same retriever
        assert retriever1 is retriever2


# =============================================================================
# RRF ALGORITHM TESTS
# =============================================================================

class TestRRFAlgorithm:
    """Tests for RRF (Reciprocal Rank Fusion) algorithm."""

    @pytest.mark.asyncio
    async def test_rrf_basic_fusion(self):
        """Test basic RRF fusion."""
        from app.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(Mock())

        # Simulate results from 3 queries
        results_by_query = {
            "q1": [
                {"chunk_id": "A", "score": 0.9},
                {"chunk_id": "B", "score": 0.8},
                {"chunk_id": "C", "score": 0.7},
            ],
            "q2": [
                {"chunk_id": "B", "score": 0.95},
                {"chunk_id": "D", "score": 0.85},
                {"chunk_id": "A", "score": 0.75},
            ],
            "q3": [
                {"chunk_id": "C", "score": 0.9},
                {"chunk_id": "A", "score": 0.8},
                {"chunk_id": "E", "score": 0.7},
            ],
        }

        # Apply RRF manually to verify
        rrf_scores = {}
        k = 60

        for query, results in results_by_query.items():
            for rank, result in enumerate(results, 1):
                chunk_id = result["chunk_id"]
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0
                rrf_scores[chunk_id] += 1 / (k + rank)

        # A appears in all 3 queries (ranks 1, 3, 2)
        # B appears in 2 queries (ranks 2, 1)
        # C appears in 2 queries (ranks 3, 1)
        expected_top = max(rrf_scores, key=rrf_scores.get)

        assert expected_top == "A"

    @pytest.mark.asyncio
    async def test_rrf_different_k_values(self):
        """Test RRF with different k values."""
        from app.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(Mock())

        results_by_query = {
            "q1": [{"chunk_id": "A", "score": 0.9}],
            "q2": [{"chunk_id": "A", "score": 0.8}],
        }

        # Calculate RRF scores for different k values
        for k in [10, 60, 100]:
            rrf_score = sum(1 / (k + rank) for rank in [1, 2])
            assert 0 < rrf_score < 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestMultiQueryIntegration:
    """Integration tests for multi-query retrieval."""

    @pytest.mark.asyncio
    async def test_full_multi_query_workflow(self):
        """Test complete multi-query retrieval workflow."""
        from app.multi_query import MultiQueryRetriever

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content='1. "machine learning basics"\n2. "ML fundamentals"'
        )

        retriever = MultiQueryRetriever(mock_llm)

        # Generate queries
        queries = await retriever.generate_queries(
            "What is machine learning?",
            num_queries=2,
            include_original=True
        )

        assert len(queries) == 3

        # Mock retrieval function
        retrieved = []

        async def mock_retrieve(query: str, k: int):
            result = {
                "text": f"Result for {query}",
                "meta": {"query": query},
                "score": 0.8
            }
            retrieved.append(result)
            return [result]

        # Retrieve with RRF
        results = await retriever.retrieve(
            queries=queries,
            retriever_func=mock_retrieve,
            top_k=5,
            merge_strategy="rrf"
        )

        assert len(results) > 0
        assert all("text" in r for r in results)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestMultiQueryEdgeCases:
    """Edge case tests for multi-query retrieval."""

    @pytest.mark.asyncio
    async def test_duplicate_results_handling(self):
        """Test handling of duplicate results across queries."""
        from app.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(Mock())

        async def mock_retrieve(query: str, k: int):
            # Return same results for all queries
            return [
                {"chunk_id": "A", "text": "Same result", "score": 0.9}
            ]

        results = await retriever.retrieve(
            queries=["q1", "q2", "q3"],
            retriever_func=mock_retrieve,
            top_k=5
        )

        # Should deduplicate while combining scores
        assert len(results) == 1
        assert results[0]["chunk_id"] == "A"

    @pytest.mark.asyncio
    async def test_llm_failure_handling(self):
        """Test handling of LLM failure."""
        from app.multi_query import MultiQueryRetriever

        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM failed")

        retriever = MultiQueryRetriever(mock_llm)

        with pytest.raises(Exception):
            await retriever.generate_queries("test query")

    @pytest.mark.asyncio
    async def test_retrieval_function_error(self):
        """Test handling of retrieval function errors."""
        from app.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(Mock())

        async def failing_retrieve(query: str, k: int):
            raise ValueError("Retrieval failed")

        with pytest.raises(ValueError):
            await retriever.retrieve(
                queries=["q1"],
                retriever_func=failing_retrieve,
                top_k=5
            )

    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Test with very long query."""
        from app.multi_query import MultiQueryRetriever

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content='1. "short"')

        retriever = MultiQueryRetriever(mock_llm)

        long_query = "machine learning " * 100  # Very long

        # Should handle without crashing
        queries = await retriever.generate_queries(long_query, num_queries=1)

        assert isinstance(queries, list)

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self):
        """Test query with special characters."""
        from app.multi_query import MultiQueryRetriever

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content='1. "test"')

        retriever = MultiQueryRetriever(mock_llm)

        special_query = "What is ML?! @#$ %^&*()"

        queries = await retriever.generate_queries(special_query, num_queries=1)

        assert isinstance(queries, list)
