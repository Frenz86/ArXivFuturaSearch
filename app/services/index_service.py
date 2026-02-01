"""Service layer for index building and management."""

import os
from typing import Optional

from app.config import settings
from app.arxiv_loader import fetch_arxiv_async, save_raw
from app.chunking import build_chunks
from app.embeddings import get_embedder
from app.vectorstore import get_vectorstore, VectorStoreInterface
from app.metrics import update_index_stats
from app.logging_config import get_logger

logger = get_logger(__name__)


def ensure_dirs() -> None:
    """Create necessary directories if they don't exist."""
    os.makedirs(settings.RAW_DIR, exist_ok=True)
    os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
    if settings.VECTORSTORE_MODE == "chroma":
        os.makedirs(settings.CHROMA_DIR, exist_ok=True)


async def build_index_async(
    query: str,
    max_results: int = 30,
    store: Optional[VectorStoreInterface] = None,
) -> dict:
    """
    Build the hybrid index from arXiv papers.

    Args:
        query: arXiv search query
        max_results: Maximum number of papers to fetch
        store: Optional existing vector store instance

    Returns:
        Statistics about the build process
    """
    ensure_dirs()

    # Fetch papers from arXiv
    logger.info("Fetching papers from arXiv", query=query, max_results=max_results)
    papers = await fetch_arxiv_async(query, max_results=max_results)
    save_raw(papers, os.path.join(settings.RAW_DIR, "arxiv_papers.json"))

    # Get embedder for chunking and embedding
    embedder = get_embedder()

    # Chunk the papers (with optional semantic chunking)
    use_semantic = settings.USE_SEMANTIC_CHUNKING
    logger.info(
        "Chunking papers",
        semantic=use_semantic,
        chunk_size=settings.CHUNK_SIZE,
    )
    chunks = build_chunks(
        papers,
        settings.CHUNK_SIZE,
        settings.CHUNK_OVERLAP,
        sentence_aware=True,
        semantic_chunking=use_semantic,
        embedder=embedder if use_semantic else None,
    )
    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    metas = [c.meta.model_dump() if hasattr(c.meta, 'model_dump') else c.meta for c in chunks]

    # Generate embeddings
    logger.info("Generating embeddings", count=len(texts))
    vectors = embedder.embed(texts, show_progress=True)

    # Build vector store index
    if store is None:
        store = get_vectorstore(collection_name="arxiv_papers")
    else:
        store.reset()

    store.add(vectors, chunk_ids, texts, metas)

    # Update metrics
    update_index_stats(documents=len(papers), chunks=len(chunks))

    # Clear cache after rebuild
    from app.cache import get_cache
    cache = get_cache()
    if cache.enabled:
        cleared = cache.clear_pattern("arxiv_rag:*")
        logger.info("Cleared cache after index rebuild", keys_deleted=cleared)

    logger.info("Index built successfully", papers=len(papers), chunks=len(chunks))

    return {
        "papers": len(papers),
        "chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "vectorstore_mode": settings.VECTORSTORE_MODE,
        "semantic_chunking": use_semantic,
    }


def count_unique_documents(store: Optional[VectorStoreInterface]) -> int:
    """Count unique documents by title from the vector store."""
    if store is None:
        return 0
    try:
        collection = store.client.get_collection(store.collection_name)
        results = collection.get(include=["metadatas"])
        titles = set()
        for meta in results.get("metadatas", []):
            title = meta.get("title")
            if title:
                titles.add(title)
        return len(titles)
    except Exception:
        return 0
