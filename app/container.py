"""
Dependency Injection Container.

Manages the creation and lifecycle of all application services.
Uses a simple factory pattern instead of heavy DI frameworks.
"""

from typing import Optional
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.config import settings
from app.database.base import Base
from app.database.session import get_db
from app.auth.service import AuthService
from app.audit.service import AuditService
from app.conversation.manager import ConversationManager
from app.alerts.service import AlertManager
from app.collections.manager import CollectionManager
from app.embeddings.native import get_embeddings, NativeEmbeddings
from app.llm.native import LLMFactory, NativeLLM
from app.rag.native import create_rag_pipeline, RAGPipeline
# from app.cache.semantic import SemanticCache  # TODO: Fix module naming conflict
from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# DATABASE
# =============================================================================

class DatabaseContainer:
    """Database connection management."""

    def __init__(self):
        self._engine = None
        self._session_factory = None

    async def init(self):
        """Initialize database engine."""
        if self._engine is None:
            from sqlalchemy.ext.asyncio import create_async_engine

            self._engine = create_async_engine(
                settings.DATABASE_URL,
                echo=settings.DATABASE_ECHO,
                pool_size=settings.POSTGRES_POOL_SIZE,
                max_overflow=settings.POSTGRES_MAX_OVERFLOW,
                pool_timeout=settings.POSTGRES_POOL_TIMEOUT,
                pool_recycle=settings.POSTGRES_POOL_RECYCLE,
            )

            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            logger.info("Database engine initialized")

    async def close(self):
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")

    async def get_session(self) -> AsyncSession:
        """Get a database session."""
        if self._session_factory is None:
            await self.init()
        return self._session_factory()

    async def create_tables(self):
        """Create all tables (for development)."""
        if self._engine is None:
            await self.init()
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")


# =============================================================================
# EMBEDDINGS
# =============================================================================

class EmbeddingsContainer:
    """Embeddings model management."""

    def __init__(self):
        self._embeddings: Optional[NativeEmbeddings] = None

    async def get(self) -> NativeEmbeddings:
        """Get or create embeddings instance."""
        if self._embeddings is None:
            self._embeddings = get_embeddings()
            logger.info("Embeddings model loaded")
        return self._embeddings


# =============================================================================
# LLM
# =============================================================================

class LLMContainer:
    """LLM client management."""

    def __init__(self):
        self._llm: Optional[NativeLLM] = None

    async def get(self) -> NativeLLM:
        """Get or create LLM instance."""
        if self._llm is None:
            self._llm = await LLMFactory.get_llm(
                model=settings.OPENROUTER_MODEL,
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
            )
            logger.info("LLM client initialized")
        return self._llm

    async def close(self):
        """Close LLM connections."""
        if self._llm:
            await self._llm.close()


# =============================================================================
# CACHE
# =============================================================================

# TODO: Fix module naming conflict between app/cache.py and app/cache/
# class CacheContainer:
#     """Cache management."""
#
#     def __init__(self):
#         self._semantic_cache: Optional[SemanticCache] = None
#
#     async def get_semantic_cache(self) -> Optional[SemanticCache]:
#         """Get or create semantic cache."""
#         if not settings.SEMANTIC_CACHE_ENABLED:
#             return None
#
#         if self._semantic_cache is None:
#             embeddings = await embeddings_container.get()
#             self._semantic_cache = SemanticCache(
#                 embeddings=embeddings,
#                 similarity_threshold=settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
#                 max_entries=settings.SEMANTIC_CACHE_MAX_ENTRIES,
#             )
#             logger.info("Semantic cache initialized")
#
#         return self._semantic_cache


# =============================================================================
# SERVICES
# =============================================================================

class ServiceContainer:
    """Business logic services."""

    def __init__(self):
        self._auth_service: Optional[AuthService] = None
        self._audit_service: Optional[AuditService] = None
        self._conversation_manager: Optional[ConversationManager] = None
        self._alert_manager: Optional[AlertManager] = None
        self._collection_manager: Optional[CollectionManager] = None

    async def get_auth_service(self, db: AsyncSession) -> AuthService:
        """Get auth service."""
        if self._auth_service is None:
            self._auth_service = AuthService(db)
        return self._auth_service

    async def get_audit_service(self, db: AsyncSession) -> AuditService:
        """Get audit service."""
        if self._audit_service is None:
            self._audit_service = AuditService(db)
        return self._audit_service

    async def get_conversation_manager(
        self,
        db: AsyncSession,
        llm: NativeLLM,
    ) -> ConversationManager:
        """Get conversation manager."""
        if self._conversation_manager is None:
            from app.rag.native import RAGLLM

            self._conversation_manager = ConversationManager(
                db=db,
                llm=RAGLLM(llm),
                max_context_tokens=settings.MAX_CONTEXT_TOKENS,
            )
        return self._conversation_manager

    async def get_alert_manager(self, db: AsyncSession) -> AlertManager:
        """Get alert manager."""
        if self._alert_manager is None:
            self._alert_manager = AlertManager(db)
        return self._alert_manager

    async def get_collection_manager(self, db: AsyncSession) -> CollectionManager:
        """Get collection manager."""
        if self._collection_manager is None:
            self._collection_manager = CollectionManager(db)
        return self._collection_manager


# =============================================================================
# RAG PIPELINE
# =============================================================================

class RAGContainer:
    """RAG pipeline management."""

    def __init__(self):
        self._pipeline: Optional[RAGPipeline] = None

    async def get(self, db: AsyncSession) -> RAGPipeline:
        """Get or create RAG pipeline."""
        if self._pipeline is None:
            # This is a simplified version - in production, integrate with
            # actual vector store and BM25 retriever
            from app.rag.native import Retriever, RetrievalMethod

            # For now, create a basic retriever
            # TODO: Integrate with actual vector store
            embeddings = await embeddings_container.get()

            # Create BM25 retriever if papers exist
            from app.retrieval.bm25 import create_bm25_retriever

            # Mock documents for now - replace with actual paper retrieval
            mock_docs = []
            bm25_retriever = create_bm25_retriever(mock_docs) if mock_docs else None

            retriever = Retriever(
                vector_store=None,  # TODO: Add vector store
                bm25_retriever=bm25_retriever,
                embeddings=embeddings,
                default_method=RetrievalMethod.SEMANTIC,
            )

            llm = await llm_container.get()

            self._pipeline = RAGPipeline(
                retriever=retriever,
                llm=llm,
                max_context_length=settings.MAX_CONTEXT_TOKENS,
            )
            logger.info("RAG pipeline initialized")

        return self._pipeline


# =============================================================================
# GLOBAL CONTAINERS
# =============================================================================

# Global container instances
database_container = DatabaseContainer()
embeddings_container = EmbeddingsContainer()
llm_container = LLMContainer()
# cache_container = CacheContainer()  # TODO: Fix module naming conflict
service_container = ServiceContainer()
rag_container = RAGContainer()


# =============================================================================
# LIFECYCLE FUNCTIONS
# =============================================================================

async def init_containers():
    """Initialize all containers."""
    logger.info("Initializing dependency injection containers")

    await database_container.init()

    # Pre-load embeddings
    if settings.CACHE_WARMING_ENABLED:
        await embeddings_container.get()

    logger.info("All containers initialized")


async def close_containers():
    """Close all container connections."""
    logger.info("Closing dependency injection containers")

    await database_container.close()
    await llm_container.close()

    logger.info("All containers closed")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def get_db_session() -> AsyncSession:
    """Get database session."""
    return await database_container.get_session()


async def get_embeddings_async() -> NativeEmbeddings:
    """Get embeddings (convenience async wrapper)."""
    return await embeddings_container.get()


async def get_llm() -> NativeLLM:
    """Get LLM."""
    return await llm_container.get()


async def get_rag_pipeline(db: AsyncSession) -> RAGPipeline:
    """Get RAG pipeline."""
    return await rag_container.get(db)
