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
from app.rag.native import create_rag_pipeline, RAGPipeline, Retriever, RetrievalMethod, RetrievedDocument, VectorStore
from app.cache.semantic import SemanticCache
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

class CacheContainer:
    """Cache management."""

    def __init__(self):
        self._semantic_cache: Optional[SemanticCache] = None

    async def get_semantic_cache(self) -> Optional[SemanticCache]:
        """Get or create semantic cache."""
        if not settings.SEMANTIC_CACHE_ENABLED:
            return None

        if self._semantic_cache is None:
            self._semantic_cache = SemanticCache(
                similarity_threshold=settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
                max_entries=settings.SEMANTIC_CACHE_MAX_ENTRIES,
            )
            logger.info("Semantic cache initialized")

        return self._semantic_cache


# ---------------------------------------------------------------------------
# Adapter: wraps VectorStoreInterface (used by main.py / deps) into the
# VectorStore protocol expected by rag.native.Retriever
# ---------------------------------------------------------------------------
class VectorStoreAdapter(VectorStore):
    """Adapts app.vectorstore.VectorStoreInterface → rag.native.VectorStore."""

    def __init__(self, store, embeddings: NativeEmbeddings):
        self._store = store
        self._embeddings = embeddings

    async def add_texts(self, texts, metadatas=None, ids=None):
        import numpy as np
        vectors = np.array([self._embeddings.embed_query(t) for t in texts])
        chunk_ids = ids or [str(i) for i in range(len(texts))]
        metas = metadatas or [{} for _ in texts]
        self._store.add(vectors=vectors, chunk_ids=chunk_ids, texts=texts, metas=metas)
        return chunk_ids

    async def similarity_search(self, query, k=10, score_threshold=None, filter_dict=None):
        import numpy as np
        query_vec = np.array(self._embeddings.embed_query(query))
        raw = self._store.search(query_vec=query_vec, query_text=query, top_k=k)
        docs = []
        for r in raw:
            score = r.get("score", 0.0)
            if score_threshold is not None and score < score_threshold:
                continue
            docs.append(RetrievedDocument(
                id=r.get("chunk_id", r.get("id", "")),
                content=r.get("text", r.get("content", "")),
                metadata=r.get("metadata", {}),
                score=score,
            ))
        return docs

    async def delete(self, ids):
        pass  # Not exposed by VectorStoreInterface; no-op


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
            from app import dependencies as deps

            embeddings = await embeddings_container.get()

            # Wire vector store from the global instance set by main.py lifespan
            vector_store = None
            if deps._store_instance is not None:
                vector_store = VectorStoreAdapter(deps._store_instance, embeddings)
            else:
                logger.warning("Vector store not yet initialized — RAG semantic search will be unavailable until index is built")

            retriever = Retriever(
                vector_store=vector_store,
                bm25_retriever=None,
                embeddings=embeddings,
                default_method=RetrievalMethod.HYBRID if vector_store else RetrievalMethod.KEYWORD,
            )

            llm = await llm_container.get()

            self._pipeline = RAGPipeline(
                retriever=retriever,
                llm=llm,
                max_context_length=settings.MAX_CONTEXT_TOKENS,
            )
            logger.info("RAG pipeline initialized", has_vector_store=vector_store is not None)

        return self._pipeline


# =============================================================================
# GLOBAL CONTAINERS
# =============================================================================

# Global container instances
database_container = DatabaseContainer()
embeddings_container = EmbeddingsContainer()
llm_container = LLMContainer()
cache_container = CacheContainer()
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
