"""Tests for autocomplete module.

Run with: pytest tests/test_autocomplete.py -v
"""

import pytest
from unittest.mock import Mock, patch


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_documents():
    """Sample documents for autocomplete indexing."""
    return [
        {"text": "Machine learning algorithms and neural networks"},
        {"text": "Deep learning with transformer models"},
        {"text": "Natural language processing techniques"},
        {"text": "Computer vision and image recognition"},
        {"text": "Reinforcement learning for games"},
    ]


@pytest.fixture
def sample_queries():
    """Sample query history."""
    return [
        "machine learning basics",
        "deep learning transformers",
        "neural network architecture",
        "natural language understanding",
        "computer vision applications",
    ]


# =============================================================================
# QUERY AUTOCOMPLETER TESTS
# =============================================================================

class TestQueryAutocompleter:
    """Tests for QueryAutocompleter class."""

    def test_initialization(self):
        """Test autocompleter initialization."""
        from app.autocomplete import QueryAutocompleter

        autocompleter = QueryAutocompleter()

        assert autocompleter._vocabulary == []
        assert autocompleter._trie is not None

    def test_index_documents(self, sample_documents):
        """Test document indexing."""
        from app.autocomplete import QueryAutocompleter

        autocompleter = QueryAutocompleter()
        autocompleter.index_documents(sample_documents)

        assert autocompleter._indexed is True
        assert len(autocompleter._vocabulary) > 0

    def test_get_suggestions_prefix(self, sample_documents):
        """Test prefix-based suggestions."""
        from app.autocomplete import QueryAutocompleter

        autocompleter = QueryAutocompleter()
        autocompleter.index_documents(sample_documents)

        suggestions = autocompleter.get_suggestions("mach", max_results=5)

        assert len(suggestions) > 0
        assert all(s["text"].startswith("mach") for s in suggestions)

    def test_get_suggestions_empty_prefix(self, sample_documents):
        """Test suggestions with empty prefix."""
        from app.autocomplete import QueryAutocompleter

        autocompleter = QueryAutocompleter()
        autocompleter.index_documents(sample_documents)

        suggestions = autocompleter.get_suggestions("", max_results=3)

        # Should return popular/short queries
        assert isinstance(suggestions, list)

    def test_get_suggestions_limit(self, sample_documents):
        """Test suggestion limit."""
        from app.autocomplete import QueryAutocompleter

        autocompleter = QueryAutocompleter()
        autocompleter.index_documents(sample_documents)

        suggestions = autocompleter.get_suggestions("m", max_results=2)

        assert len(suggestions) <= 2

    def test_get_suggestions_no_match(self, sample_documents):
        """Test with no matching prefix."""
        from app.autocomplete import QueryAutocompleter

        autocompleter = QueryAutocompleter()
        autocompleter.index_documents(sample_documents)

        suggestions = autocompleter.get_suggestions("xyzabc123", max_results=5)

        assert len(suggestions) == 0

    def test_add_query(self):
        """Test adding query to history."""
        from app.autocomplete import QueryAutocompleter

        autocompleter = QueryAutocompleter()
        autocompleter.add_query("machine learning tutorial")

        assert "machine learning tutorial" in autocompleter._query_history

    def test_get_fuzzy_suggestions(self, sample_documents):
        """Test fuzzy matching suggestions."""
        from app.autocomplete import QueryAutocompleter

        autocompleter = QueryAutocompleter()
        autocompleter.index_documents(sample_documents)

        # Typo in query
        suggestions = autocompleter.get_suggestions("machne lerning", max_results=5)

        # Should return some suggestions
        assert isinstance(suggestions, list)


# =============================================================================
# TRENDING QUERIES TESTS
# =============================================================================

class TestTrendingQueries:
    """Tests for TrendingQueries class."""

    def test_initialization(self):
        """Test trending queries initialization."""
        from app.autocomplete import TrendingQueries

        trending = TrendingQueries(max_size=100)

        assert trending._max_size == 100
        assert trending._queries == {}

    def test_add_query(self):
        """Test adding query to trending."""
        from app.autocomplete import TrendingQueries

        trending = TrendingQueries()
        trending.add("machine learning")

        assert "machine learning" in trending._queries

    def test_add_query_multiple_times(self):
        """Test incrementing query count."""
        from app.autocomplete import TrendingQueries

        trending = TrendingQueries()

        trending.add("test query")
        trending.add("test query")
        trending.add("test query")

        assert trending._queries["test query"] == 3

    def test_get_trending(self):
        """Test getting trending queries."""
        from app.autocomplete import TrendingQueries

        trending = TrendingQueries()

        # Add queries with different frequencies
        for _ in range(10):
            trending.add("popular query")
        for _ in range(5):
            trending.add("somewhat popular")
        for _ in range(1):
            trending.add("unpopular")

        results = trending.get_trending(limit=2)

        assert len(results) == 2
        assert results[0]["query"] == "popular query"
        assert results[0]["count"] == 10

    def test_get_trending_limit(self):
        """Test trending with limit."""
        from app.autocomplete import TrendingQueries

        trending = TrendingQueries()

        for i in range(10):
            trending.add(f"query {i}")

        results = trending.get_trending(limit=5)

        assert len(results) == 5

    def test_max_size_enforcement(self):
        """Test max size enforcement."""
        from app.autocomplete import TrendingQueries

        trending = TrendingQueries(max_size=3)

        # Add more than max_size
        for i in range(10):
            trending.add(f"query {i}")

        # Should only keep top 3
        assert len(trending._queries) <= 3

    def test_decay_old_queries(self):
        """Test decaying old query counts."""
        from app.autocomplete import TrendingQueries

        trending = TrendingQueries()

        trending.add("old query")
        trending.add("new query")

        # Decay should reduce counts
        trending.decay(factor=0.5)

        assert 0 < trending._queries["old query"] < 1
        assert 0 < trending._queries["new query"] < 1


# =============================================================================
# SEMANTIC AUTOCOMPLETER TESTS
# =============================================================================

class TestSemanticAutocompleter:
    """Tests for SemanticAutocompleter class."""

    def test_initialization(self):
        """Test semantic autocompleter initialization."""
        from app.autocomplete import SemanticAutocompleter

        autocompleter = SemanticAutocompleter(embedder=None)

        assert autocompleter._embedder is None
        assert autocompleter._indexed is False

    def test_index_documents_with_embedder(self, sample_documents):
        """Test indexing with embedder."""
        from app.autocomplete import SemanticAutocompleter

        mock_embedder = Mock()
        mock_embedder.embed = Mock(return_value=[[0.1, 0.2, 0.3] for _ in sample_documents])

        autocompleter = SemanticAutocompleter(embedder=mock_embedder)
        autocompleter.index_documents(sample_documents)

        assert autocompleter._indexed is True

    def test_get_semantic_suggestions(self, sample_documents):
        """Test semantic suggestions."""
        from app.autocomplete import SemanticAutocompleter

        mock_embedder = Mock()
        mock_embedder.embed_query = Mock(return_value=[0.1, 0.2, 0.3])

        autocompleter = SemanticAutocompleter(embedder=mock_embedder)

        # Mock indexed documents
        autocompleter._documents = sample_documents
        autocompleter._embeddings = [[0.1, 0.2, 0.3] for _ in sample_documents]
        autocompleter._indexed = True

        suggestions = autocompleter.get_suggestions("machine learning", max_results=3)

        assert len(suggestions) <= 3
        assert all("text" in s for s in suggestions)
        assert all("score" in s for s in suggestions)


# =============================================================================
# GLOBAL INSTANCE TESTS
# =============================================================================

class TestGlobalAutocompleter:
    """Tests for global autocompleter instances."""

    def test_get_autocompleter_singleton(self):
        """Test that get_autocompleter returns singleton."""
        from app.autocomplete import get_autocompleter

        autocompleter1 = get_autocompleter()
        autocompleter2 = get_autocompleter()

        assert autocompleter1 is autocompleter2

    def test_get_trending_queries_singleton(self):
        """Test that get_trending_queries returns singleton."""
        from app.autocomplete import get_trending_queries

        trending1 = get_trending_queries()
        trending2 = get_trending_queries()

        assert trending1 is trending2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestAutocompleteIntegration:
    """Integration tests for autocomplete."""

    def test_full_autocomplete_workflow(self, sample_documents):
        """Test complete autocomplete workflow."""
        from app.autocomplete import QueryAutocompleter, TrendingQueries

        # Create autocompleter
        autocompleter = QueryAutocompleter()
        autocompleter.index_documents(sample_documents)

        # Create trending
        trending = TrendingQueries()
        for query in ["machine learning", "deep learning", "machine learning"]:
            trending.add(query)

        # Get suggestions
        suggestions = autocompleter.get_suggestions("mach", max_results=5)

        assert len(suggestions) > 0

        # Get trending
        top_trending = trending.get_trending(limit=3)

        assert len(top_trending) >= 1


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestAutocompleteEdgeCases:
    """Edge case tests for autocomplete."""

    def test_empty_documents(self):
        """Test with empty document list."""
        from app.autocomplete import QueryAutocompleter

        autocompleter = QueryAutocompleter()
        autocompleter.index_documents([])

        suggestions = autocompleter.get_suggestions("test", max_results=5)

        assert suggestions == []

    def test_special_characters(self):
        """Test with special characters."""
        from app.autocomplete import QueryAutocompleter

        autocompleter = QueryAutocompleter()
        autocompleter.index_documents([
            {"text": "C++ programming"},
            {"text": "C# development"},
            {"text": "Python & NumPy"},
        ])

        suggestions = autocompleter.get_suggestions("c", max_results=5)

        assert isinstance(suggestions, list)

    def test_unicode_content(self):
        """Test with unicode content."""
        from app.autocomplete import QueryAutocompleter

        autocompleter = QueryAutocompleter()
        autocompleter.index_documents([
            {"text": "机器学习算法"},
            {"text": "日本語のテスト"},
            {"text": "Тест на русском"},
        ])

        suggestions = autocompleter.get_suggestions("機", max_results=5)

        assert isinstance(suggestions, list)

    def test_very_long_query(self):
        """Test with very long query."""
        from app.autocomplete import QueryAutocompleter

        autocompleter = QueryAutocompleter()
        autocompleter.add_query("word " * 1000)

        # Should handle without crashing
        suggestions = autocompleter.get_suggestions("word", max_results=5)

        assert isinstance(suggestions, list)

    def test_case_sensitivity(self):
        """Test case sensitivity."""
        from app.autocomplete import QueryAutocompleter

        autocompleter = QueryAutocompleter()
        autocompleter.index_documents([
            {"text": "Machine Learning"},
            {"text": "machine learning"},
        ])

        suggestions_lower = autocompleter.get_suggestions("machine", max_results=5)
        suggestions_upper = autocompleter.get_suggestions("Machine", max_results=5)

        assert isinstance(suggestions_lower, list)
        assert isinstance(suggestions_upper, list)
