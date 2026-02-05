"""
Database session management for SQLAlchemy.

Provides async engine and session factory for PostgreSQL.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Global engine and session factory
_engine = None
_async_session_factory = None


def get_database_url() -> str:
    """
    Get the database URL from settings.

    Returns:
        PostgreSQL async connection string
    """
    # Check for custom database URL
    if hasattr(settings, 'DATABASE_URL'):
        return settings.DATABASE_URL

    # Build from individual settings
    user = getattr(settings, 'POSTGRES_USER', 'postgres')
    password = getattr(settings, 'POSTGRES_PASSWORD', 'postgres')
    host = getattr(settings, 'POSTGRES_HOST', 'localhost')
    port = getattr(settings, 'POSTGRES_PORT', 5432)
    db = getattr(settings, 'POSTGRES_DB', 'arxiv_rag')

    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


def get_engine():
    """
    Get or create the async database engine.

    Returns:
        SQLAlchemy async engine
    """
    global _engine

    if _engine is None:
        database_url = get_database_url()

        # Engine configuration
        engine_kwargs = {
            "echo": getattr(settings, 'DATABASE_ECHO', False),
            "pool_size": getattr(settings, 'POSTGRES_POOL_SIZE', 10),
            "max_overflow": getattr(settings, 'POSTGRES_MAX_OVERFLOW', 20),
            "pool_timeout": getattr(settings, 'POSTGRES_POOL_TIMEOUT', 30),
            "pool_recycle": getattr(settings, 'POSTGRES_POOL_RECYCLE', 3600),
            "pool_pre_ping": True,  # Verify connections before using
        }

        _engine = create_async_engine(database_url, **engine_kwargs)

        logger.info(
            "Database engine created",
            database=getattr(settings, 'POSTGRES_DB', 'arxiv_rag'),
            pool_size=engine_kwargs['pool_size'],
            max_overflow=engine_kwargs['max_overflow'],
        )

    return _engine


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session.

    Yields:
        Async database session

    Example:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    global _async_session_factory

    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async with _async_session_factory() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize the database - create all tables.

    Should be called on application startup.
    """
    from app.database.base import Base

    engine = get_engine()

    async with engine.begin() as conn:
        # Import all models here to ensure they're registered
        from app.database.base import User, Role, Permission, UserSession, AuditLog

        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized successfully")


async def close_db() -> None:
    """
    Close the database connection.

    Should be called on application shutdown.
    """
    global _engine

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        logger.info("Database connection closed")


async def check_db_connection() -> bool:
    """
    Check if database connection is healthy.

    Returns:
        True if connection is working
    """
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error("Database connection check failed", error=str(e))
        return False
