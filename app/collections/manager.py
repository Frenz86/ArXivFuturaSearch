"""
Collaborative features: collections, saved searches, and annotations.

Allows users to save searches, create shared collections of papers,
and add annotations for collaborative research.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import secrets

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.base import SavedSearch, Collection, CollectionPaper, Annotation
from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# COLLECTION MANAGER
# =============================================================================

class CollectionManager:
    """
    Manager for paper collections and sharing.

    Handles collection CRUD operations, sharing, and access control.
    """

    def __init__(self, db: AsyncSession):
        """Initialize collection manager."""
        self.db = db

    async def create_collection(
        self,
        user_id: str,
        name: str,
        description: Optional[str] = None,
        is_public: bool = False,
    ) -> Collection:
        """Create a new collection."""
        collection = Collection(
            user_id=user_id,
            name=name,
            description=description,
            is_public=is_public,
            share_token=secrets.token_urlsafe(48),
        )

        self.db.add(collection)
        await self.db.commit()
        await self.db.refresh(collection)

        logger.info(
            "Collection created",
            collection_id=collection.id,
            user_id=user_id,
            name=name,
        )

        return collection

    async def add_paper_to_collection(
        self,
        collection_id: str,
        paper_id: str,
        user_id: str,
        notes: Optional[str] = None,
    ) -> CollectionPaper:
        """Add a paper to a collection."""
        # Verify collection exists and user owns it
        collection = await self._get_collection_for_user(collection_id, user_id)
        if not collection:
            raise ValueError("Collection not found")

        # Check if paper already in collection
        result = await self.db.execute(
            select(CollectionPaper).where(
                CollectionPaper.collection_id == collection_id,
                CollectionPaper.paper_id == paper_id,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update notes
            existing.notes = notes
            await self.db.commit()
            return existing

        # Get next order index
        count_result = await self.db.execute(
            select(CollectionPaper).where(
                CollectionPaper.collection_id == collection_id
            )
        )
        count = len(count_result.scalars().all())

        # Add paper
        collection_paper = CollectionPaper(
            collection_id=collection_id,
            paper_id=paper_id,
            order_index=count,
            notes=notes,
        )

        self.db.add(collection_paper)
        await self.db.commit()

        logger.info(
            "Paper added to collection",
            collection_id=collection_id,
            paper_id=paper_id,
        )

        return collection_paper

    async def get_collection(
        self,
        collection_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Collection]:
        """Get collection with papers (checks access if user_id provided)."""
        result = await self.db.execute(
            select(Collection).where(Collection.id == collection_id)
        )
        collection = result.scalar_one_or_none()

        if not collection:
            return None

        # Check access
        if user_id and collection.user_id != user_id and not collection.is_public:
            return None

        # Load papers
        papers_result = await self.db.execute(
            select(CollectionPaper)
            .where(CollectionPaper.collection_id == collection_id)
            .order_by(CollectionPaper.order_index)
        )
        collection.papers = papers_result.scalars().all()

        return collection

    async def generate_share_link(
        self,
        collection_id: str,
        user_id: str,
        expires_in_hours: int = 168,  # 1 week
    ) -> str:
        """Generate shareable link for collection."""
        collection = await self._get_collection_for_user(collection_id, user_id)
        if not collection:
            raise ValueError("Collection not found")

        # Generate new share token
        collection.share_token = secrets.token_urlsafe(48)
        await self.db.commit()

        # Build share URL
        base_url = ""  # Would get from settings
        share_url = f"{base_url}/collections/shared/{collection.share_token}"

        logger.info(
            "Share link generated",
            collection_id=collection_id,
            expires_in_hours=expires_in_hours,
        )

        return share_url

    async def get_collection_by_token(
        self,
        share_token: str,
    ) -> Optional[Collection]:
        """Access collection via share token."""
        result = await self.db.execute(
            select(Collection)
            .where(Collection.share_token == share_token)
        )
        collection = result.scalar_one_or_none()

        if collection and not collection.is_public:
            return None

        return collection

    async def _get_collection_for_user(
        self,
        collection_id: str,
        user_id: str,
    ) -> Optional[Collection]:
        """Get collection owned by user."""
        result = await self.db.execute(
            select(Collection).where(
                Collection.id == collection_id,
                Collection.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()


# =============================================================================
# ANNOTATION SERVICE
# =============================================================================

class AnnotationService:
    """Service for managing paper annotations."""

    def __init__(self, db: AsyncSession):
        """Initialize annotation service."""
        self.db = db

    async def add_annotation(
        self,
        user_id: str,
        paper_id: str,
        annotation_type: str,
        content: str,
        position: Optional[Dict[str, Any]] = None,
        collection_id: Optional[str] = None,
    ) -> Annotation:
        """Add annotation to paper."""
        annotation = Annotation(
            user_id=user_id,
            paper_id=paper_id,
            annotation_type=annotation_type,
            content=content,
            position=position or {},
            collection_id=collection_id,
        )

        self.db.add(annotation)
        await self.db.commit()
        await self.db.refresh(annotation)

        logger.info(
            "Annotation added",
            annotation_id=annotation.id,
            paper_id=paper_id,
            type=annotation_type,
        )

        return annotation

    async def get_paper_annotations(
        self,
        user_id: str,
        paper_id: str,
    ) -> List[Annotation]:
        """Get all user's annotations for a paper."""
        result = await self.db.execute(
            select(Annotation).where(
                Annotation.user_id == user_id,
                Annotation.paper_id == paper_id,
            )
        )
        return result.scalars().all()

    async def get_collection_annotations(
        self,
        collection_id: str,
        user_id: str,
    ) -> List[Annotation]:
        """Get annotations for all papers in a collection."""
        result = await self.db.execute(
            select(Annotation).where(
                Annotation.collection_id == collection_id,
            )
        )
        return result.scalars().all()


# =============================================================================
# SAVED SEARCH SERVICE
# =============================================================================

class SavedSearchService:
    """Service for managing saved search queries."""

    def __init__(self, db: AsyncSession):
        """Initialize saved search service."""
        self.db = db

    async def save_search(
        self,
        user_id: str,
        name: str,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SavedSearch:
        """Save a search query."""
        saved_search = SavedSearch(
            user_id=user_id,
            name=name,
            query=query,
            filters=filters or {},
        )

        self.db.add(saved_search)
        await self.db.commit()
        await self.db.refresh(saved_search)

        logger.info(
            "Search saved",
            saved_search_id=saved_search.id,
            user_id=user_id,
            name=name,
        )

        return saved_search

    async def list_saved_searches(
        self,
        user_id: str,
    ) -> List[SavedSearch]:
        """List user's saved searches."""
        result = await self.db.execute(
            select(SavedSearch)
            .where(SavedSearch.user_id == user_id)
            .order_by(SavedSearch.created_at.desc())
        )
        return result.scalars().all()

    async def delete_saved_search(
        self,
        search_id: str,
        user_id: str,
    ) -> bool:
        """Delete a saved search."""
        result = await self.db.execute(
            select(SavedSearch).where(
                SavedSearch.id == search_id,
                SavedSearch.user_id == user_id,
            )
        )
        saved_search = result.scalar_one_or_none()

        if not saved_search:
            return False

        await self.db.delete(saved_search)
        await self.db.commit()

        logger.info(
            "Saved search deleted",
            search_id=search_id,
            user_id=user_id,
        )

        return True


# =============================================================================
# SHARING SERVICE
# =============================================================================

class SharingService:
    """Service for sharing collections and results."""

    async def create_export_link(
        self,
        search_results: Dict[str, Any],
        format: str = "json",
        expires_in_hours: int = 24,
    ) -> str:
        """
        Create time-limited export link.

        Args:
            search_results: Search results to share
            format: Export format
            expires_in_hours: Link validity in hours

        Returns:
            Shareable URL
        """
        # Generate unique token
        token = secrets.token_urlsafe(32)

        # Store in cache with TTL
        # Implementation depends on cache setup
        # ...

        base_url = ""  # Would get from settings
        share_url = f"{base_url}/shared/{token}"

        return share_url

    async def access_export_link(
        self,
        token: str,
    ) -> Optional[Dict[str, Any]]:
        """Access exported results via share link."""
        # Retrieve from cache
        # ...
        return None
