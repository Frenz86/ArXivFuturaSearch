"""Auto-completion and query suggestion for improved search experience.


# Copyright 2025 ArXivFuturaSearch Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Provides intelligent suggestions as users type their queries.
"""

import re
from typing import List, Dict, Any, Optional, Set
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# QUERY AUTOCOMPLETION
# =============================================================================

class QueryAutocompleter:
    """
    Provides query suggestions based on indexed content and query history.

    Features:
    - Prefix-based autocomplete
    - Semantic similarity suggestions
    - Popular query suggestions
    - Spelling correction suggestions
    """

    def __init__(
        self,
        min_prefix_length: int = 2,
        max_suggestions: int = 10,
        use_frequency: bool = True,
    ):
        """
        Initialize query autocompleter.

        Args:
            min_prefix_length: Minimum prefix length to trigger suggestions
            max_suggestions: Maximum number of suggestions to return
            use_frequency: Whether to use query frequency for ranking
        """
        self.min_prefix_length = min_prefix_length
        self.max_suggestions = max_suggestions
        self.use_frequency = use_frequency

        # Data structures for suggestions
        self._vocabulary: Set[str] = set()
        self._ngrams: Dict[int, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._query_frequency: Counter = Counter()
        self._document_terms: Dict[str, int] = Counter()

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build vocabulary from indexed documents.

        Args:
            documents: List of documents with 'text' or 'content' field
        """
        logger.info("Building autocomplete vocabulary", documents=len(documents))

        for doc in documents:
            text = doc.get("text") or doc.get("content") or doc.get("page_content", "")
            terms = self._extract_terms(text)

            for term in terms:
                self._vocabulary.add(term)
                self._document_terms[term] += 1

                # Add n-grams
                for n in range(2, min(4, len(term) + 1)):
                    prefix = term[:n]
                    self._ngrams[n][prefix].add(term)

        logger.info(
            "Autocomplete vocabulary built",
            vocabulary_size=len(self._vocabulary),
            unique_terms=len(self._document_terms),
        )

    def _extract_terms(self, text: str) -> List[str]:
        """
        Extract searchable terms from text.

        Args:
            text: Input text

        Returns:
            List of terms
        """
        # Convert to lowercase
        text = text.lower()

        # Extract alphanumeric phrases
        terms = re.findall(r'\b[a-z]{3,}\b', text)

        # Also extract multi-word phrases (bigrams, trigrams)
        words = text.split()
        phrases = []
        for n in [2, 3]:
            if len(words) >= n:
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i + n])
                    if len(phrase) >= 5:
                        phrases.append(phrase)

        return list(set(terms + phrases))

    def add_query(self, query: str) -> None:
        """
        Add a query to history (for frequency-based ranking).

        Args:
            query: User query
        """
        self._query_frequency[query.lower()] += 1

    def get_suggestions(
        self,
        prefix: str,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get autocomplete suggestions for a prefix.

        Args:
            prefix: Query prefix
            max_results: Maximum number of suggestions

        Returns:
            List of suggestion dicts with text and score
        """
        prefix = prefix.lower().strip()
        max_results = max_results or self.max_suggestions

        if len(prefix) < self.min_prefix_length:
            return []

        suggestions = []

        # 1. Exact prefix matches from n-grams
        n = min(len(prefix), 3)
        if n in self._ngrams and prefix in self._ngrams[n]:
            for term in self._ngrams[n][prefix]:
                if term.startswith(prefix):
                    suggestions.append({
                        "text": term,
                        "type": "prefix_match",
                        "score": self._score_suggestion(term),
                    })

        # 2. Fuzzy matches (edit distance of 1)
        if len(suggestions) < max_results:
            fuzzy_matches = self._get_fuzzy_matches(prefix, max_results - len(suggestions))
            suggestions.extend(fuzzy_matches)

        # 3. Semantic/related terms (from document co-occurrence)
        if len(suggestions) < max_results:
            related = self._get_related_terms(prefix, max_results - len(suggestions))
            suggestions.extend(related)

        # Sort by score and deduplicate
        seen = set()
        unique_suggestions = []
        for suggestion in sorted(suggestions, key=lambda x: x["score"], reverse=True):
            text = suggestion["text"]
            if text not in seen and text.lower() != prefix:
                seen.add(text)
                unique_suggestions.append(suggestion)
                if len(unique_suggestions) >= max_results:
                    break

        logger.debug(
            "Autocomplete suggestions",
            prefix=prefix,
            suggestions=len(unique_suggestions),
        )

        return unique_suggestions

    def _score_suggestion(self, term: str) -> float:
        """
        Score a suggestion based on various factors.

        Args:
            term: Suggested term

        Returns:
            Relevance score
        """
        score = 0.0

        # Document frequency (more common = higher score)
        doc_freq = self._document_terms.get(term, 0)
        score += min(doc_freq / 100.0, 1.0) * 0.5

        # Query frequency (previously searched = higher score)
        query_freq = self._query_frequency.get(term, 0)
        score += min(query_freq / 10.0, 1.0) * 0.3

        # Length preference (moderate length is better)
        length = len(term)
        if 3 <= length <= 20:
            score += 0.1
        elif 20 < length <= 40:
            score += 0.05

        return score

    def _get_fuzzy_matches(
        self,
        prefix: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """
        Get fuzzy matches within edit distance 1.

        Args:
            prefix: Query prefix
            max_results: Maximum results

        Returns:
            List of fuzzy match suggestions
        """
        matches = []

        for term in self._vocabulary:
            if self._edit_distance(prefix, term) <= 2:
                matches.append({
                    "text": term,
                    "type": "fuzzy_match",
                    "score": self._score_suggestion(term) * 0.8,  # Penalize fuzzy matches
                })
                if len(matches) >= max_results:
                    break

        return matches

    def _get_related_terms(
        self,
        prefix: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """
        Get related terms based on document co-occurrence.

        Args:
            prefix: Query prefix
            max_results: Maximum results

        Returns:
            List of related term suggestions
        """
        # Simple implementation: return terms that share words with prefix
        related = []
        prefix_words = set(prefix.split())

        for term in self._vocabulary:
            term_words = set(term.split())
            # Check for word overlap
            if prefix_words & term_words:
                related.append({
                    "text": term,
                    "type": "related_term",
                    "score": self._score_suggestion(term) * 0.6,
                })
                if len(related) >= max_results:
                    break

        return related

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return QueryAutocompleter._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


# =============================================================================
# TRENDING QUERIES
# =============================================================================

class TrendingQueries:
    """
    Track and suggest trending/popular search queries.

    Useful for:
    - Showing what others are searching for
    - Discovering popular topics
    - Providing inspiration for searches
    """

    def __init__(
        self,
        window_size: int = 1000,
        decay_factor: float = 0.95,
    ):
        """
        Initialize trending queries tracker.

        Args:
            window_size: Number of recent queries to consider
            decay_factor: Exponential decay for older queries
        """
        self.window_size = window_size
        self.decay_factor = decay_factor

        self._query_history: List[str] = []
        self._trending_scores: Counter = Counter()

    def add_query(self, query: str) -> None:
        """
        Add a query to history.

        Args:
            query: User query
        """
        self._query_history.append(query.lower())

        # Maintain window size
        if len(self._query_history) > self.window_size:
            # Decay old scores
            self._trending_scores = Counter({
                q: score * self.decay_factor
                for q, score in self._trending_scores.items()
            })
            # Remove oldest queries
            removed = self._query_history[:-self.window_size]
            for q in removed:
                self._trending_scores[q] -= 1
                if self._trending_scores[q] <= 0:
                    del self._trending_scores[q]
            self._query_history = self._query_history[-self.window_size:]

        # Update trending score
        self._trending_scores[query.lower()] += 1

    def get_trending(
        self,
        limit: int = 10,
        min_queries: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Get current trending queries.

        Args:
            limit: Maximum number of results
            min_queries: Minimum query frequency to be included

        Returns:
            List of trending query dicts
        """
        trending = []

        for query, score in self._trending_scores.most_common(limit * 2):
            if score >= min_queries:
                trending.append({
                    "query": query,
                    "frequency": score,
                    "type": "trending",
                })

        return trending[:limit]


# =============================================================================
# SEMANTIC AUTOCOMPLETE
# =============================================================================

class SemanticAutocompleter:
    """
    Semantic autocomplete using vector similarity.

    Suggests queries based on semantic similarity to the partial query.
    """

    def __init__(
        self,
        embedder: Any,
        vector_store: Any,
        max_suggestions: int = 10,
    ):
        """
        Initialize semantic autocompleter.

        Args:
            embedder: Embedder instance
            vector_store: Vector store of past queries/documents
            max_suggestions: Maximum suggestions to return
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.max_suggestions = max_suggestions

    async def get_suggestions(
        self,
        prefix: str,
        min_similarity: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Get semantic suggestions for a partial query.

        Args:
            prefix: Query prefix
            min_similarity: Minimum similarity threshold

        Returns:
            List of semantic suggestions
        """
        if len(prefix.strip()) < 3:
            return []

        try:
            # Embed the prefix
            prefix_embedding = self.embedder.embed([prefix], is_query=True)[0]

            # Search for similar queries/documents in vector store
            results = await self.vector_store.asimilarity_search_with_score(
                prefix,
                k=self.max_suggestions * 2,
            )

            suggestions = []
            for doc, score in results:
                # Convert score to similarity (if it's distance)
                similarity = 1.0 - score if score <= 1.0 else score

                if similarity >= min_similarity:
                    text = doc.get("text", doc.get("content", doc.page_content))
                    suggestions.append({
                        "text": text[:100],  # Limit length
                        "type": "semantic",
                        "score": similarity,
                    })

            return sorted(suggestions, key=lambda x: x["score"], reverse=True)[:self.max_suggestions]

        except Exception as e:
            logger.warning("Semantic autocomplete failed", error=str(e))
            return []


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

_autocompleter: Optional[QueryAutocompleter] = None
_trending_queries: Optional[TrendingQueries] = None
_semantic_autocompleter: Optional[SemanticAutocompleter] = None


def get_autocompleter() -> QueryAutocompleter:
    """Get or create global query autocompleter."""
    global _autocompleter
    if _autocompleter is None:
        _autocompleter = QueryAutocompleter()
    return _autocompleter


def get_trending_queries() -> TrendingQueries:
    """Get or create global trending queries tracker."""
    global _trending_queries
    if _trending_queries is None:
        _trending_queries = TrendingQueries()
    return _trending_queries


def get_semantic_autocompleter(
    embedder: Any,
    vector_store: Any,
) -> SemanticAutocompleter:
    """Get or create semantic autocompleter."""
    global _semantic_autocompleter
    if _semantic_autocompleter is None:
        _semantic_autocompleter = SemanticAutocompleter(embedder, vector_store)
    return _semantic_autocompleter
