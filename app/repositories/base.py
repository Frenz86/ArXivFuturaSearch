"""
Repository pattern implementation for data access layer.

Provides abstraction over database operations, making the codebase
more testable and maintainable.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TypeVar, Generic
from dataclasses import dataclass

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.logging_config import get_logger
from app.database.base import Base

logger = get_logger(__name__)

T = TypeVar("T", bound=Base)


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class PaginatedResult(Generic[T]):
    """Paginated query result."""
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int

    def __repr__(self):
        return f"<PaginatedResult(page={self.page}/{self.total_pages}, total={self.total})>"


@dataclass
class FilterOptions:
    """Generic filter options."""
    filters: Dict[str, Any] = None
    search_text: Optional[str] = None
    search_fields: Optional[List[str]] = None
    sort_by: Optional[str] = None
    sort_order: str = "asc"  # asc or desc
    offset: int = 0
    limit: int = 100


# =============================================================================
# BASE REPOSITORY
# =============================================================================

class BaseRepository(Generic[T], ABC):
    """
    Abstract base repository.

    Provides common CRUD operations for all repositories.
    """

    def __init__(self, session: AsyncSession, model: type[T]):
        """
        Initialize repository.

        Args:
            session: Database session
            model: SQLAlchemy model class
        """
        self.session = session
        self.model = model

    async def get_by_id(self, id: str) -> Optional[T]:
        """
        Get entity by ID.

        Args:
            id: Entity ID

        Returns:
            Entity or None
        """
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        options: Optional[FilterOptions] = None,
    ) -> List[T]:
        """
        Get all entities with optional filtering.

        Args:
            options: Filter and pagination options

        Returns:
            List of entities
        """
        query = select(self.model)

        if options:
            query = self._apply_filters(query, options)
            query = self._apply_sorting(query, options)
            query = self._apply_pagination(query, options)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_paginated(
        self,
        options: FilterOptions,
    ) -> PaginatedResult[T]:
        """
        Get paginated results.

        Args:
            options: Filter and pagination options

        Returns:
            PaginatedResult
        """
        # Count total
        count_query = select(func.count(self.model.id))
        count_query = self._apply_filters(count_query, options)
        total_result = await self.session.execute(count_query)
        total = total_result.scalar()

        # Get items
        query = select(self.model)
        query = self._apply_filters(query, options)
        query = self._apply_sorting(query, options)
        query = self._apply_pagination(query, options)

        result = await self.session.execute(query)
        items = result.scalars().all()

        # Calculate pages
        page_size = options.limit
        page = (options.offset // page_size) + 1
        total_pages = (total + page_size - 1) // page_size

        return PaginatedResult(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    async def create(self, entity: T) -> T:
        """
        Create new entity.

        Args:
            entity: Entity to create

        Returns:
            Created entity
        """
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)

        logger.info(
            "Entity created",
            model=self.model.__name__,
            id=entity.id,
        )

        return entity

    async def create_many(self, entities: List[T]) -> List[T]:
        """
        Create multiple entities.

        Args:
            entities: Entities to create

        Returns:
            Created entities
        """
        for entity in entities:
            self.session.add(entity)

        await self.session.flush()

        for entity in entities:
            await self.session.refresh(entity)

        logger.info(
            "Entities created",
            model=self.model.__name__,
            count=len(entities),
        )

        return entities

    async def update(self, entity: T) -> T:
        """
        Update entity.

        Args:
            entity: Entity to update

        Returns:
            Updated entity
        """
        await self.session.flush()
        await self.session.refresh(entity)

        logger.info(
            "Entity updated",
            model=self.model.__name__,
            id=entity.id,
        )

        return entity

    async def update_by_id(
        self,
        id: str,
        updates: Dict[str, Any],
    ) -> Optional[T]:
        """
        Update entity by ID.

        Args:
            id: Entity ID
            updates: Fields to update

        Returns:
            Updated entity or None
        """
        query = (
            update(self.model)
            .where(self.model.id == id)
            .values(**updates)
            .returning(self.model)
        )

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def delete(self, entity: T) -> None:
        """
        Delete entity.

        Args:
            entity: Entity to delete
        """
        entity_id = entity.id
        await self.session.delete(entity)
        await self.session.flush()

        logger.info(
            "Entity deleted",
            model=self.model.__name__,
            id=entity_id,
        )

    async def delete_by_id(self, id: str) -> bool:
        """
        Delete entity by ID.

        Args:
            id: Entity ID

        Returns:
            True if deleted, False if not found
        """
        query = delete(self.model).where(self.model.id == id)
        result = await self.session.execute(query)
        return result.rowcount > 0

    async def exists(self, id: str) -> bool:
        """
        Check if entity exists.

        Args:
            id: Entity ID

        Returns:
            True if exists
        """
        result = await self.session.execute(
            select(func.count(self.model.id)).where(self.model.id == id)
        )
        return result.scalar() > 0

    async def count(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Count entities.

        Args:
            filters: Optional filters

        Returns:
            Count
        """
        query = select(func.count(self.model.id))

        if filters:
            for key, value in filters.items():
                query = query.where(getattr(self.model, key) == value)

        result = await self.session.execute(query)
        return result.scalar()

    def _apply_filters(
        self,
        query,
        options: FilterOptions,
    ):
        """Apply filters to query."""
        if options.filters:
            for key, value in options.filters.items():
                if hasattr(self.model, key):
                    query = query.where(getattr(self.model, key) == value)

        if options.search_text and options.search_fields:
            conditions = []
            for field in options.search_fields:
                if hasattr(self.model, field):
                    conditions.append(
                        getattr(self.model, field).ilike(f"%{options.search_text}%")
                    )
            if conditions:
                from sqlalchemy import or_
                query = query.where(or_(*conditions))

        return query

    def _apply_sorting(
        self,
        query,
        options: FilterOptions,
    ):
        """Apply sorting to query."""
        if options.sort_by and hasattr(self.model, options.sort_by):
            column = getattr(self.model, options.sort_by)
            if options.sort_order == "desc":
                query = query.order_by(column.desc())
            else:
                query = query.order_by(column.asc())

        return query

    def _apply_pagination(
        self,
        query,
        options: FilterOptions,
    ):
        """Apply pagination to query."""
        return query.offset(options.offset).limit(options.limit)


# =============================================================================
# UNIT OF WORK
# =============================================================================

class UnitOfWork:
    """
    Unit of Work pattern for transaction management.

    Coordinates multiple repositories within a single transaction.
    """

    def __init__(self, session: AsyncSession):
        """
        Initialize Unit of Work.

        Args:
            session: Database session
        """
        self.session = session
        self._repositories: Dict[str, BaseRepository] = {}

    def register_repository(
        self,
        name: str,
        repository: BaseRepository,
    ) -> None:
        """
        Register a repository.

        Args:
            name: Repository name
            repository: Repository instance
        """
        self._repositories[name] = repository
        logger.debug("Repository registered", name=name)

    def get_repository(self, name: str) -> Optional[BaseRepository]:
        """
        Get registered repository.

        Args:
            name: Repository name

        Returns:
            Repository or None
        """
        return self._repositories.get(name)

    async def commit(self) -> None:
        """Commit transaction."""
        await self.session.commit()
        logger.debug("Transaction committed")

    async def rollback(self) -> None:
        """Rollback transaction."""
        await self.session.rollback()
        logger.debug("Transaction rolled back")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type:
            await self.rollback()
        else:
            await self.commit()


# =============================================================================
# REPOSITORY FACTORY
# =============================================================================

class RepositoryFactory:
    """
    Factory for creating repository instances.

    Provides a central point for repository configuration.
    """

    _repositories: Dict[str, type[BaseRepository]] = {}

    @classmethod
    def register(cls, name: str, repository_class: type[BaseRepository]) -> None:
        """
        Register a repository class.

        Args:
            name: Repository name
            repository_class: Repository class
        """
        cls._repositories[name] = repository_class

    @classmethod
    def create(
        cls,
        name: str,
        session: AsyncSession,
    ) -> Optional[BaseRepository]:
        """
        Create repository instance.

        Args:
            name: Repository name
            session: Database session

        Returns:
            Repository instance or None
        """
        repository_class = cls._repositories.get(name)
        if repository_class:
            return repository_class(session)
        return None
