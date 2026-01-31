"""Text chunking utilities for RAG pipeline with sentence-aware and semantic splitting."""

import asyncio
import re
from functools import lru_cache
from typing import Optional

import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Flag to track if NLTK data has been downloaded
_nltk_downloaded = False
_nltk_lock = asyncio.Lock()


async def ensure_nltk_data_async() -> None:
    """
    Download NLTK data asynchronously if not present.

    This is the preferred method to call during startup.
    Uses asyncio.Lock to prevent concurrent downloads.
    """
    global _nltk_downloaded

    if _nltk_downloaded:
        return

    async with _nltk_lock:
        # Double-check after acquiring lock
        if _nltk_downloaded:
            return

        try:
            nltk.data.find("tokenizers/punkt_tab")
            _nltk_downloaded = True
            return
        except LookupError:
            pass

        # Download in thread pool to avoid blocking
        logger.info("Downloading NLTK punkt tokenizer asynchronously...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, nltk.download, "punkt_tab")
        _nltk_downloaded = True
        logger.info("NLTK punkt tokenizer downloaded successfully")


@lru_cache(maxsize=1)
def _ensure_nltk_data() -> None:
    """
    Download NLTK data if not present (cached, synchronous fallback).

    This synchronous version is kept for backward compatibility
    but ensure_nltk_data_async() should be preferred during startup.
    """
    global _nltk_downloaded

    if _nltk_downloaded:
        return

    try:
        nltk.data.find("tokenizers/punkt_tab")
        _nltk_downloaded = True
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer (synchronous)...")
        nltk.download("punkt_tab", quiet=True)
        _nltk_downloaded = True


class ChunkMetadata(BaseModel):
    """Type-safe metadata for a document chunk."""

    title: str = Field(description="Paper title")
    authors: str = Field(default="", description="Comma-separated authors")
    published: str = Field(description="Publication date")
    link: str = Field(description="arXiv link")
    tags: str = Field(default="", description="Comma-separated tags")

    @field_validator("title", "link", "published")
    @classmethod
    def validate_not_empty(cls, v: str, info) -> str:
        """Ensure required fields are not empty."""
        if not v or not v.strip():
            field_name = info.field_name
            if field_name in ["title", "link"]:
                raise ValueError(f"{field_name} cannot be empty")
        return v


class Chunk(BaseModel):
    """A chunk of text with validated metadata."""

    doc_id: str = Field(description="Document/paper ID")
    chunk_id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Chunk text content")
    meta: ChunkMetadata = Field(description="Chunk metadata")

    def to_dict(self) -> dict:
        """Convert to dictionary format for backward compatibility."""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "meta": self.meta.model_dump(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Create Chunk from dictionary (backward compatibility)."""
        # Handle old format with dict meta
        meta_data = data.get("meta", {})
        if isinstance(meta_data, dict):
            meta = ChunkMetadata(**meta_data)
        else:
            meta = meta_data

        return cls(
            doc_id=data["doc_id"],
            chunk_id=data["chunk_id"],
            text=data["text"],
            meta=meta,
        )


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

    OPTIMIZED: Uses batch processing for embeddings instead of per-sentence calls.

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

    # Compute embeddings for all sentences in a single batch call
    # This is MUCH faster than embedding each sentence individually
    try:
        sentence_embeddings = embedder.embed(sentences, show_progress=False, is_query=False)
    except Exception as e:
        logger.warning("Embedding computation failed, falling back to sentence-aware chunking", error=str(e))
        # Fallback to sentence-aware chunking without semantic analysis
        overlap_sentences = max(1, chunk_size // 500)
        return chunk_text_sentences(text, chunk_size, overlap_sentences)

    # Compute cosine similarities between consecutive sentences
    # Using vectorized operations for better performance
    if len(sentence_embeddings) > 1:
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        normalized_embeddings = sentence_embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero

        # Compute similarities between consecutive sentences using dot product
        similarities = np.dot(normalized_embeddings[:-1], normalized_embeddings[1:].T).diagonal()
    else:
        similarities = np.array([])

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
        avg_similarity=float(np.mean(similarities)) if len(similarities) > 0 else 0,
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
            # Create validated metadata
            chunk_meta = ChunkMetadata(
                title=p["title"],
                authors=", ".join(p["authors"]) if p.get("authors") else "",
                published=p["published"],
                link=p["link"],
                tags=", ".join(p.get("tags", [])) if p.get("tags") else "",
            )
            out.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::chunk_{i}",
                    text=part,
                    meta=chunk_meta,
                )
            )

    logger.info("Built chunks from papers", papers=len(papers), chunks=len(out))
    return out

