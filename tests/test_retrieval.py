"""Tests for hybrid vector store retrieval."""

import numpy as np
from app.vectorstore import HybridStore


def test_hybrid_store_search():
    """Test basic hybrid search functionality."""
    store = HybridStore(dim=3)

    # Add some normalized vectors
    vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
    store.add(
        vecs,
        ["a", "b", "c"],
        ["apple banana", "banana cherry", "cherry date"],
        [{"x": 1}, {"x": 2}, {"x": 3}],
    )

    # Search for vector most similar to [1, 0, 0]
    q = np.array([1, 0, 0], dtype="float32")
    res = store.search(q, query_text="apple", top_k=1)

    assert len(res) == 1
    assert res[0]["chunk_id"] == "a"
    assert res[0]["text"] == "apple banana"
    assert res[0]["meta"]["x"] == 1


def test_hybrid_store_top_k():
    """Test that top_k returns correct number of results."""
    store = HybridStore(dim=2)

    vecs = np.array(
        [[1, 0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0, 1]], dtype="float32"
    )
    # Normalize
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    store.add(
        vecs,
        [f"id_{i}" for i in range(5)],
        [f"text_{i}" for i in range(5)],
        [{"i": i} for i in range(5)],
    )

    q = np.array([1, 0], dtype="float32")
    res = store.search(q, top_k=3)

    assert len(res) == 3
    # First result should be closest to [1, 0]
    assert res[0]["chunk_id"] == "id_0"


def test_hybrid_store_empty():
    """Test searching empty store."""
    store = HybridStore(dim=3)
    q = np.array([1, 0, 0], dtype="float32")
    res = store.search(q, top_k=5)
    assert res == []


def test_hybrid_store_scores_descending():
    """Test that results are sorted by score descending."""
    store = HybridStore(dim=3)

    vecs = np.array(
        [[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0]], dtype="float32"
    )
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    store.add(vecs, ["a", "b", "c"], ["A", "B", "C"], [{}, {}, {}])

    q = np.array([1, 0, 0], dtype="float32")
    res = store.search(q, top_k=3)

    # Scores should be descending
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_store_bm25_search():
    """Test that BM25 lexical search works."""
    store = HybridStore(dim=3)

    vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
    store.add(
        vecs,
        ["doc1", "doc2", "doc3"],
        ["machine learning algorithms", "deep neural networks", "natural language processing"],
        [{}, {}, {}],
    )

    # Search with semantic vector pointing to doc1, but BM25 query for "neural"
    q = np.array([1, 0, 0], dtype="float32")
    res = store.search(q, query_text="neural networks", top_k=3, semantic_weight=0.3, bm25_weight=0.7)

    # With high BM25 weight, "neural networks" should boost doc2
    assert len(res) == 3


def test_hybrid_store_metadata_filtering():
    """Test metadata filtering in search."""
    store = HybridStore(dim=3)

    vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
    store.add(
        vecs,
        ["doc1", "doc2", "doc3"],
        ["text1", "text2", "text3"],
        [
            {"published": "2024-01-01", "authors": ["Alice"]},
            {"published": "2024-06-01", "authors": ["Bob"]},
            {"published": "2023-01-01", "authors": ["Charlie"]},
        ],
    )

    q = np.array([1, 0, 0], dtype="float32")

    # Filter by published_after
    res = store.search(q, top_k=3, filters={"published_after": "2024-01-01"})
    assert len(res) == 2  # Only doc1 and doc2

    # Filter by author
    res = store.search(q, top_k=3, filters={"authors": ["Alice"]})
    assert len(res) == 1
    assert res[0]["chunk_id"] == "doc1"
