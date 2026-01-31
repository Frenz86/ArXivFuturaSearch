"""Tests for chunking module."""

from app.chunking import chunk_text, chunk_text_simple, chunk_text_sentences, build_chunks


def test_chunking_produces_output():
    """Test that chunking produces multiple chunks for long text."""
    chunks = chunk_text("hello " * 1000, chunk_size=100, overlap=20)
    assert len(chunks) > 1


def test_empty_text():
    """Test that empty text returns empty list."""
    assert chunk_text("") == []


def test_short_text():
    """Test that short text returns single chunk."""
    chunks = chunk_text("short text", chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == "short text"


def test_simple_chunking_max_size():
    """Test that simple chunking respects max size."""
    chunks = chunk_text_simple("hello " * 1000, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)


def test_sentence_aware_chunking():
    """Test that sentence-aware chunking works."""
    text = "This is sentence one. This is sentence two. This is sentence three. This is a longer sentence that contains more words."
    chunks = chunk_text_sentences(text, chunk_size=100, overlap_sentences=1)
    assert len(chunks) >= 1
    # Each chunk should end with a sentence (period)
    for chunk in chunks[:-1]:  # Except maybe the last one
        assert "." in chunk


def test_build_chunks_from_papers():
    """Test building chunks from paper data."""
    papers = [
        {
            "id": "test/123",
            "title": "Test Paper",
            "summary": "This is a test abstract. It has multiple sentences. This allows testing sentence-aware chunking.",
            "authors": ["Author A"],
            "published": "2024-01-01",
            "link": "http://example.com",
            "tags": ["cs.LG"],
        }
    ]

    chunks = build_chunks(papers, chunk_size=500, overlap=50, sentence_aware=True)

    assert len(chunks) >= 1
    assert chunks[0].doc_id == "test/123"
    assert "Test Paper" in chunks[0].text
    assert chunks[0].meta["title"] == "Test Paper"


def test_build_chunks_preserves_metadata():
    """Test that chunking preserves all metadata."""
    papers = [
        {
            "id": "arxiv/2024.12345",
            "title": "Advanced RAG Techniques",
            "summary": "We present novel techniques for RAG systems.",
            "authors": ["Alice", "Bob", "Charlie"],
            "published": "2024-06-15T10:00:00Z",
            "link": "https://arxiv.org/abs/2024.12345",
            "tags": ["cs.CL", "cs.LG"],
        }
    ]

    chunks = build_chunks(papers, chunk_size=1000, overlap=100)

    assert len(chunks) >= 1
    meta = chunks[0].meta
    assert meta["title"] == "Advanced RAG Techniques"
    assert meta["authors"] == ["Alice", "Bob", "Charlie"]
    assert meta["published"] == "2024-06-15T10:00:00Z"
    assert meta["link"] == "https://arxiv.org/abs/2024.12345"
    assert "cs.CL" in meta["tags"]
