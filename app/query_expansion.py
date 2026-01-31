"""Query expansion techniques for better retrieval."""

import re
from typing import List
from app.logging_config import get_logger

logger = get_logger(__name__)


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


def expand_query(query: str, method: str = "acronym") -> str:
    """
    Expand query with additional related terms.

    Args:
        query: Original query
        method: Expansion method - "acronym", "related", "both", or "none"

    Returns:
        Expanded query string
    """
    if method == "none":
        return query

    query_lower = query.lower()
    expanded_parts = [query]

    # Add acronym expansions
    if method in ["acronym", "both"]:
        for acronym, expansion in ACRONYM_EXPANSIONS.items():
            if acronym.lower() in query_lower and expansion not in query_lower:
                expanded_parts.append(expansion)

    # Add related terms
    if method in ["related", "both"]:
        words = re.findall(r'\b\w+\b', query_lower)
        for word in words:
            if word in RELATED_TERMS:
                for related in RELATED_TERMS[word]:
                    if related not in query_lower:
                        expanded_parts.append(related)

    # Combine expanded terms
    expanded_query = " ".join(expanded_parts)

    if expanded_query != query:
        logger.debug("Query expanded", original=query, expanded=expanded_query, method=method)

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
