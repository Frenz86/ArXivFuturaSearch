"""Text chunking utilities for RAG pipeline with sentence-aware and semantic splitting."""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _ensure_nltk_data() -> None:
    """Download NLTK data if not present (cached)."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt_tab", quiet=True)


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    doc_id: str
    chunk_id: str
    text: str
    meta: dict


def chunk_text_simple(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """
    Simple character-based chunking (fallback).

    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    text = " ".join(text.split())  # Normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def chunk_text_sentences(
    text: str,
    chunk_size: int = 900,
    overlap_sentences: int = 2,
) -> list[str]:
    """
    Sentence-aware chunking that respects sentence boundaries.

    Args:
        text: Input text to chunk
        chunk_size: Target maximum characters per chunk
        overlap_sentences: Number of sentences to overlap between chunks

    Returns:
        List of text chunks
    """
    _ensure_nltk_data()

    text = " ".join(text.split())  # Normalize whitespace
    if not text:
        return []

    # Split into sentences
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.warning("Sentence tokenization failed, falling back to simple chunking", error=str(e))
        return chunk_text_simple(text, chunk_size, overlap=150)

    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If single sentence exceeds chunk_size, split it
        if sentence_length > chunk_size:
            # Flush current chunk first
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                # Keep overlap sentences
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences:] if overlap_sentences else []
                current_length = sum(len(s) + 1 for s in current_chunk_sentences)

            # Split long sentence by punctuation or characters
            sub_chunks = _split_long_sentence(sentence, chunk_size)
            chunks.extend(sub_chunks[:-1])

            # Start new chunk with remainder
            current_chunk_sentences = [sub_chunks[-1]] if sub_chunks else []
            current_length = len(current_chunk_sentences[0]) if current_chunk_sentences else 0
            continue

        # Check if adding this sentence exceeds limit
        if current_length + sentence_length + 1 > chunk_size and current_chunk_sentences:
            # Flush current chunk
            chunks.append(" ".join(current_chunk_sentences))

            # Keep overlap sentences for context
            current_chunk_sentences = current_chunk_sentences[-overlap_sentences:] if overlap_sentences else []
            current_length = sum(len(s) + 1 for s in current_chunk_sentences)

        current_chunk_sentences.append(sentence)
        current_length += sentence_length + 1

    # Don't forget the last chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


def _split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Split a sentence that exceeds chunk_size."""
    # Try splitting by semicolons, commas, or other punctuation
    parts = re.split(r'(?<=[;,:])\s+', sentence)

    if len(parts) > 1:
        # Recombine parts into chunks
        chunks = []
        current = ""
        for part in parts:
            if len(current) + len(part) + 1 <= chunk_size:
                current = f"{current} {part}".strip() if current else part
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        return chunks

    # Fallback: hard split by characters
    return chunk_text_simple(sentence, chunk_size, overlap=50)


def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150,
    sentence_aware: bool = True,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters/sentences to overlap
        sentence_aware: Use sentence-boundary-aware chunking

    Returns:
        List of text chunks
    """
    if sentence_aware:
        # Convert character overlap to approximate sentence overlap
        overlap_sentences = max(1, overlap // 100)  # ~100 chars per sentence
        return chunk_text_sentences(text, chunk_size, overlap_sentences)
    else:
        return chunk_text_simple(text, chunk_size, overlap)


def chunk_text_semantic(
    text: str,
    embedder,
    chunk_size: int = 900,
    similarity_threshold: float = 0.7,
) -> list[str]:
    """
    Semantic chunking that groups sentences by semantic similarity.

    This creates more coherent chunks by analyzing the semantic similarity
    between consecutive sentences and creating boundaries when similarity drops.

    Args:
        text: Input text to chunk
        embedder: Embedder instance for computing sentence embeddings
        chunk_size: Target maximum characters per chunk
        similarity_threshold: Minimum similarity to keep sentences together (0-1)

    Returns:
        List of semantically coherent text chunks
    """
    _ensure_nltk_data()

    text = " ".join(text.split())  # Normalize whitespace
    if not text:
        return []

    # Split into sentences
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.warning("Sentence tokenization failed in semantic chunking", error=str(e))
        return chunk_text_simple(text, chunk_size, overlap=150)

    if len(sentences) <= 1:
        return sentences

    # Compute embeddings for all sentences
    sentence_embeddings = embedder.embed(sentences, show_progress=False)

    # Compute cosine similarities between consecutive sentences
    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        sim = np.dot(sentence_embeddings[i], sentence_embeddings[i + 1])
        similarities.append(sim)

    # Find split points where similarity drops below threshold
    chunks = []
    current_chunk = [sentences[0]]
    current_length = len(sentences[0])

    for i, sentence in enumerate(sentences[1:], start=1):
        sentence_len = len(sentence)

        # Check if we should split based on similarity or size
        should_split = False

        # Split if similarity is low (semantic boundary)
        if i - 1 < len(similarities) and similarities[i - 1] < similarity_threshold:
            should_split = True

        # Split if chunk size would exceed limit
        if current_length + sentence_len + 1 > chunk_size and len(current_chunk) > 0:
            should_split = True

        if should_split:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            # Continue building current chunk
            current_chunk.append(sentence)
            current_length += sentence_len + 1

    # Don't forget last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.debug(
        "Semantic chunking complete",
        sentences=len(sentences),
        chunks=len(chunks),
        avg_similarity=np.mean(similarities) if similarities else 0,
    )

    return chunks


def build_chunks(
    papers: list[dict],
    chunk_size: int,
    overlap: int,
    sentence_aware: bool = True,
    semantic_chunking: bool = False,
    embedder = None,
) -> list[Chunk]:
    """
    Build chunks from a list of papers.

    Args:
        papers: List of paper dictionaries with 'id', 'title', 'summary'
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap
        sentence_aware: Use sentence-boundary-aware chunking
        semantic_chunking: Use semantic similarity-based chunking (requires embedder)
        embedder: Embedder instance (required if semantic_chunking=True)

    Returns:
        List of Chunk objects
    """
    if semantic_chunking and embedder is None:
        logger.warning("Semantic chunking requested but no embedder provided, falling back to sentence-aware")
        semantic_chunking = False

    out = []

    for p in papers:
        doc_id = p["id"]
        # Combine title + abstract as the document content
        full = f"Title: {p['title']}\n\nAbstract: {p['summary']}"

        # Choose chunking strategy
        if semantic_chunking and embedder is not None:
            parts = chunk_text_semantic(
                full,
                embedder,
                chunk_size=chunk_size,
                similarity_threshold=settings.SEMANTIC_SIMILARITY_THRESHOLD,
            )
        elif sentence_aware:
            overlap_sentences = max(1, overlap // 100)
            parts = chunk_text_sentences(full, chunk_size, overlap_sentences)
        else:
            parts = chunk_text_simple(full, chunk_size, overlap)

        for i, part in enumerate(parts):
            out.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::chunk_{i}",
                    text=part,
                    meta={
                        "title": p["title"],
                        "authors": ", ".join(p["authors"]) if p.get("authors") else "",
                        "published": p["published"],
                        "link": p["link"],
                        "tags": ", ".join(p.get("tags", [])) if p.get("tags") else "",
                    },
                )
            )

    logger.info("Built chunks from papers", papers=len(papers), chunks=len(out))
    return out

