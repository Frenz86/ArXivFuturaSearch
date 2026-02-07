"""
Paper repository implementations.

Provides data access for papers and chunks using the repository pattern.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, UTC
from uuid import uuid4

from sqlalchemy import select, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base import BaseRepository, FilterOptions, PaginatedResult
from app.database.base import Base
from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# DOMAIN ENTITIES
# =============================================================================

class PaperEntity:
    """
    Paper domain entity.

    Represents a paper in the domain model, independent of database structure.
    """

    def __init__(
        self,
        id: str,
        title: str,
        authors: List[str],
        abstract: str,
        published_date: Optional[datetime] = None,
        categories: Optional[List[str]] = None,
        arxiv_id: Optional[str] = None,
        pdf_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.published_date = published_date
        self.categories = categories or []
        self.arxiv_id = arxiv_id
        self.pdf_url = pdf_url
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "categories": self.categories,
            "arxiv_id": self.arxiv_id,
            "pdf_url": self.pdf_url,
            "metadata": self.metadata,
        }


class ChunkEntity:
    """Chunk domain entity."""

    def __init__(
        self,
        id: str,
        paper_id: str,
        content: str,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.paper_id = paper_id
        self.content = content
        self.chunk_index = chunk_index
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "paper_id": self.paper_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


# =============================================================================
# PAPER REPOSITORY
# =============================================================================

class PaperRepository(BaseRepository):
    """
    Repository for paper operations.

    Note: This is a template. In production, this would work with
    actual SQLAlchemy models or the vector store.
    """

    def __init__(self, session: AsyncSession):
        """
        Initialize paper repository.

        Args:
            session: Database session
        """
        # For now, use a simple dict-based model
        # In production, this would use actual SQLAlchemy models
        self.session = session
        self._papers: Dict[str, PaperEntity] = {}

    async def get_by_id(self, id: str) -> Optional[PaperEntity]:
        """Get paper by ID."""
        return self._papers.get(id)

    async def get_by_arxiv_id(self, arxiv_id: str) -> Optional[PaperEntity]:
        """Get paper by ArXiv ID."""
        for paper in self._papers.values():
            if paper.arxiv_id == arxiv_id:
                return paper
        return None

    async def search(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        authors: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[PaperEntity]:
        """
        Search papers with filters.

        Args:
            query: Search query
            categories: Filter by categories
            authors: Filter by authors
            date_from: Start date
            date_to: End date
            limit: Max results

        Returns:
            List of papers
        """
        results = list(self._papers.values())

        # Filter by categories
        if categories:
            results = [
                p for p in results
                if any(cat in p.categories for cat in categories)
            ]

        # Filter by authors
        if authors:
            results = [
                p for p in results
                if any(author in p.authors for author in authors)
            ]

        # Filter by date range
        if date_from:
            results = [
                p for p in results
                if p.published_date and p.published_date >= date_from
            ]

        if date_to:
            results = [
                p for p in results
                if p.published_date and p.published_date <= date_to
            ]

        # Filter by text query
        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if query_lower in p.title.lower()
                or query_lower in p.abstract.lower()
            ]

        # Sort by date (newest first)
        results.sort(
            key=lambda p: p.published_date or datetime.min,
            reverse=True,
        )

        return results[:limit]

    async def create(self, paper: PaperEntity) -> PaperEntity:
        """Create paper."""
        if not paper.id:
            paper.id = str(uuid4())

        self._papers[paper.id] = paper
        logger.info("Paper created", id=paper.id, title=paper.title[:50])
        return paper

    async def create_many(self, papers: List[PaperEntity]) -> List[PaperEntity]:
        """Create multiple papers."""
        for paper in papers:
            if not paper.id:
                paper.id = str(uuid4())
            self._papers[paper.id] = paper

        logger.info("Papers created", count=len(papers))
        return papers

    async def update(self, paper: PaperEntity) -> PaperEntity:
        """Update paper."""
        if paper.id in self._papers:
            self._papers[paper.id] = paper
            logger.info("Paper updated", id=paper.id)
        return paper

    async def delete(self, paper: PaperEntity) -> None:
        """Delete paper."""
        if paper.id in self._papers:
            del self._papers[paper.id]
            logger.info("Paper deleted", id=paper.id)

    async def get_recent(
        self,
        days: int = 7,
        limit: int = 10,
    ) -> List[PaperEntity]:
        """
        Get recent papers.

        Args:
            days: Number of days to look back
            limit: Max results

        Returns:
            List of recent papers
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)

        results = [
            p for p in self._papers.values()
            if p.published_date and p.published_date >= cutoff
        ]

        results.sort(
            key=lambda p: p.published_date or datetime.min,
            reverse=True,
        )

        return results[:limit]

    async def get_by_category(
        self,
        category: str,
        limit: int = 10,
    ) -> List[PaperEntity]:
        """
        Get papers by category.

        Args:
            category: Category name
            limit: Max results

        Returns:
            List of papers
        """
        results = [
            p for p in self._papers.values()
            if category in p.categories
        ]

        results.sort(
            key=lambda p: p.published_date or datetime.min,
            reverse=True,
        )

        return results[:limit]

    async def count_by_category(
        self,
        category: str,
    ) -> int:
        """Count papers by category."""
        return sum(
            1 for p in self._papers.values()
            if category in p.categories
        )


# =============================================================================
# CHUNK REPOSITORY
# =============================================================================

class ChunkRepository(BaseRepository):
    """Repository for chunk operations."""

    def __init__(self, session: AsyncSession):
        """Initialize chunk repository."""
        self.session = session
        self._chunks: Dict[str, ChunkEntity] = {}

    async def get_by_id(self, id: str) -> Optional[ChunkEntity]:
        """Get chunk by ID."""
        return self._chunks.get(id)

    async def get_by_paper_id(
        self,
        paper_id: str,
    ) -> List[ChunkEntity]:
        """Get all chunks for a paper."""
        return [
            c for c in self._chunks.values()
            if c.paper_id == paper_id
        ]

    async def create(self, chunk: ChunkEntity) -> ChunkEntity:
        """Create chunk."""
        if not chunk.id:
            chunk.id = str(uuid4())

        self._chunks[chunk.id] = chunk
        return chunk

    async def create_many(self, chunks: List[ChunkEntity]) -> List[ChunkEntity]:
        """Create multiple chunks."""
        for chunk in chunks:
            if not chunk.id:
                chunk.id = str(uuid4())
            self._chunks[chunk.id] = chunk

        logger.info("Chunks created", count=len(chunks))
        return chunks

    async def delete_by_paper_id(self, paper_id: str) -> int:
        """Delete all chunks for a paper."""
        to_delete = [
            chunk_id for chunk_id, chunk in self._chunks.items()
            if chunk.paper_id == paper_id
        ]

        for chunk_id in to_delete:
            del self._chunks[chunk_id]

        logger.info("Chunks deleted by paper", paper_id=paper_id, count=len(to_delete))
        return len(to_delete)


# =============================================================================
# PAPER SERVICE
# =============================================================================

class PaperService:
    """
    High-level paper service.

    Coordinates between repositories and business logic.
    """

    def __init__(
        self,
        paper_repo: PaperRepository,
        chunk_repo: ChunkRepository,
    ):
        """
        Initialize paper service.

        Args:
            paper_repo: Paper repository
            chunk_repo: Chunk repository
        """
        self.paper_repo = paper_repo
        self.chunk_repo = chunk_repo

    async def ingest_paper(
        self,
        title: str,
        authors: List[str],
        abstract: str,
        categories: Optional[List[str]] = None,
        arxiv_id: Optional[str] = None,
        chunks: Optional[List[str]] = None,
    ) -> PaperEntity:
        """
        Ingest a new paper.

        Args:
            title: Paper title
            authors: List of authors
            abstract: Paper abstract
            categories: ArXiv categories
            arxiv_id: ArXiv ID
            chunks: Text chunks for vector search

        Returns:
            Created paper
        """
        # Create paper
        paper = PaperEntity(
            id=str(uuid4()),
            title=title,
            authors=authors,
            abstract=abstract,
            categories=categories or [],
            arxiv_id=arxiv_id,
            published_date=datetime.now(UTC),
        )

        paper = await self.paper_repo.create(paper)

        # Create chunks if provided
        if chunks:
            chunk_entities = []
            for i, chunk_text in enumerate(chunks):
                chunk = ChunkEntity(
                    id=str(uuid4()),
                    paper_id=paper.id,
                    content=chunk_text,
                    chunk_index=i,
                )
                chunk_entities.append(chunk)

            await self.chunk_repo.create_many(chunk_entities)

        logger.info(
            "Paper ingested",
            paper_id=paper.id,
            arxiv_id=arxiv_id,
            n_chunks=len(chunks) if chunks else 0,
        )

        return paper

    async def get_paper_with_chunks(
        self,
        paper_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get paper with its chunks.

        Args:
            paper_id: Paper ID

        Returns:
            Paper with chunks or None
        """
        paper = await self.paper_repo.get_by_id(paper_id)
        if not paper:
            return None

        chunks = await self.chunk_repo.get_by_paper_id(paper_id)

        return {
            "paper": paper.to_dict(),
            "chunks": [c.to_dict() for c in chunks],
        }

    async def delete_paper(self, paper_id: str) -> bool:
        """
        Delete a paper and its chunks.

        Args:
            paper_id: Paper ID

        Returns:
            True if deleted
        """
        paper = await self.paper_repo.get_by_id(paper_id)
        if not paper:
            return False

        # Delete chunks first
        await self.chunk_repo.delete_by_paper_id(paper_id)

        # Delete paper
        await self.paper_repo.delete(paper)

        logger.info("Paper deleted with chunks", paper_id=paper_id)
        return True


# =============================================================================
# FACTORY
# =============================================================================

def create_repositories(session: AsyncSession) -> Dict[str, Any]:
    """
    Create repository instances.

    Args:
        session: Database session

    Returns:
        Dictionary of repositories
    """
    return {
        "papers": PaperRepository(session),
        "chunks": ChunkRepository(session),
    }


def create_paper_service(session: AsyncSession) -> PaperService:
    """
    Create paper service.

    Args:
        session: Database session

    Returns:
        PaperService instance
    """
    repos = create_repositories(session)
    return PaperService(
        paper_repo=repos["papers"],
        chunk_repo=repos["chunks"],
    )
