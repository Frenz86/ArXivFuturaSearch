"""Query expansion techniques for better retrieval."""


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

import re
from typing import List, Optional
from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# QUERY EXPANSION SETTINGS
# =============================================================================

# Minimum and maximum query lengths for expansion
MIN_QUERY_LENGTH_FOR_EXPANSION = 10  # Too short queries don't need expansion
MAX_QUERY_LENGTH_FOR_EXPANSION = 200  # Too long queries are already specific

# Short queries benefit more from expansion (below threshold)
SHORT_QUERY_THRESHOLD = 30
LONG_QUERY_THRESHOLD = 100


def should_expand_query(
    query: str,
    method: str = "acronym",
    force_expansion: bool = False
) -> bool:
    """
    Determine if a query should be expanded based on its characteristics.

    Args:
        query: The query to evaluate
        method: The expansion method being considered
        force_expansion: Force expansion regardless of query characteristics

    Returns:
        True if the query should be expanded
    """
    if force_expansion:
        return True

    if method == "none":
        return False

    query_len = len(query.strip())

    # Too short - don't expand (likely a simple keyword)
    if query_len < MIN_QUERY_LENGTH_FOR_EXPANSION:
        logger.debug("Query too short for expansion", length=query_len, query=query[:50])
        return False

    # Too long - don't expand (already specific)
    if query_len > MAX_QUERY_LENGTH_FOR_EXPANSION:
        logger.debug("Query too long for expansion", length=query_len)
        return False

    # Medium length queries benefit from expansion
    return True


def get_expansion_intensity(query: str) -> str:
    """
    Determine the intensity of expansion based on query length.

    Args:
        query: The query to evaluate

    Returns:
        "full", "moderate", or "minimal"
    """
    query_len = len(query.strip())

    if query_len < SHORT_QUERY_THRESHOLD:
        return "full"  # Short queries need more help
    elif query_len < LONG_QUERY_THRESHOLD:
        return "moderate"  # Medium queries need some help
    else:
        return "minimal"  # Long queries are already specific


# Common acronyms and their expansions in ML/AI
ACRONYM_EXPANSIONS = {
    "RAG": "retrieval augmented generation",
    "LLM": "large language model",
    "ML": "machine learning",
    "DL": "deep learning",
    "NLP": "natural language processing",
    "CV": "computer vision",
    "API": "application programming interface",
    "GPU": "graphics processing unit",
    "TPU": "tensor processing unit",
    "SOTA": "state of the art",
    "GPT": "generative pre-trained transformer",
    "BERT": "bidirectional encoder representations from transformers",
    "TF": "transformer",
}

# Related terms for common concepts
RELATED_TERMS = {
    "retrieval": ["search", "fetching", "query", "indexing"],
    "embedding": ["vector", "representation", "encoding"],
    "chunk": ["segment", "fragment", "split"],
    "semantic": ["meaning", "context", "understanding"],
    "transformer": ["attention", "self-attention", "GPT", "BERT"],
    "attention": ["focus", "mechanism", "self-attention", "cross-attention"],
    "rag": ["retrieval", "augmented", "generation", "knowledge", "external"],
    "query": ["question", "search", "inquiry", "request"],
    "model": ["neural network", "architecture", "algorithm"],
}

# Multilingual synonyms for common ML terms (for E5 multilingual)
MULTILINGUAL_TERMS = {
    "neural network": ["rete neurale", "red neuronal", "réseau neuronal"],
    "model": ["modello", "modelo", "modèle"],
    "training": ["addestramento", "entrenamiento", "entraînement"],
    "learning": ["apprendimento", "aprendizaje", "apprentissage"],
    "data": ["dati", "datos", "données"],
}


def expand_query(
    query: str,
    method: str = "acronym",
    force_expansion: bool = False
) -> str:
    """
    Expand query with additional related terms.

    Args:
        query: Original query
        method: Expansion method - "acronym", "related", "both", or "none"
        force_expansion: Force expansion regardless of query characteristics

    Returns:
        Expanded query string
    """
    if not should_expand_query(query, method, force_expansion):
        return query

    query_lower = query.lower()
    expanded_parts = [query]

    # Determine expansion intensity based on query length
    intensity = get_expansion_intensity(query)

    # Add acronym expansions
    if method in ["acronym", "both"]:
        for acronym, expansion in ACRONYM_EXPANSIONS.items():
            if acronym.lower() in query_lower and expansion not in query_lower:
                # For minimal intensity, only add 1-2 most important expansions
                if intensity == "minimal" and len(expanded_parts) > 2:
                    break
                expanded_parts.append(expansion)

    # Add related terms based on intensity
    if method in ["related", "both"]:
        words = re.findall(r'\b\w+\b', query_lower)
        terms_added = 0

        for word in words:
            if word in RELATED_TERMS:
                for related in RELATED_TERMS[word]:
                    if related not in query_lower:
                        # Limit based on intensity
                        if intensity == "minimal" and terms_added >= 1:
                            break
                        if intensity == "moderate" and terms_added >= 3:
                            break
                        # Full intensity can add all related terms
                        expanded_parts.append(related)
                        terms_added += 1

                if intensity == "minimal" and terms_added >= 1:
                    break

    # Combine expanded terms
    expanded_query = " ".join(expanded_parts)

    if expanded_query != query:
        logger.info(
            "Query expanded",
            original=query,
            expanded=expanded_query,
            method=method,
            intensity=intensity,
            terms_added=len(expanded_parts) - 1,
        )

    return expanded_query


def generate_query_variants(query: str) -> List[str]:
    """
    Generate multiple query variants for ensemble retrieval.

    Args:
        query: Original query

    Returns:
        List of query variants
    """
    variants = [query]

    # Add expanded version
    expanded = expand_query(query, method="acronym")
    if expanded != query:
        variants.append(expanded)

    # Create a more specific variant (first sentence mainly)
    if "." in query:
        specific = query.split(".")[0].strip()
        if len(specific) > 10:  # Only if substantial
            variants.append(specific)

    # Create a simpler variant (remove parenthetical explanations)
    simple = re.sub(r'\([^)]*\)', '', query).strip()
    simple = re.sub(r'\s+', ' ', simple)
    if simple != query and len(simple) > 5:
        variants.append(simple)

    return list(set(variants))  # Remove duplicates


def reciprocal_rank_fusion(
    results_list: List[List[dict]],
    k: int = 60,
    top_k: int = 5
) -> List[dict]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = sum(1 / (k + rank(d))) for each result list

    Args:
        results_list: List of result lists from different queries/retrievers
        k: RRF constant (default 60)
        top_k: Number of top results to return

    Returns:
        Fused and reranked results
    """
    if not results_list:
        return []

    # Calculate RRF scores
    rrf_scores = {}
    doc_data = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            # Use content as identifier
            doc_id = doc["text"][:100]

            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
                doc_data[doc_id] = doc

            rrf_scores[doc_id] += 1 / (k + rank + 1)

    # Sort by RRF score
    sorted_docs = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Update documents with RRF score
    fused_results = []
    for doc_id, score in sorted_docs[:top_k]:
        doc = doc_data[doc_id].copy()
        doc["rrf_score"] = score
        doc["score"] = score  # Also set main score
        fused_results.append(doc)

    return fused_results
