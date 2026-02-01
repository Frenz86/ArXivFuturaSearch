"""Service layer for business logic."""

from app.services.index_service import (
    ensure_dirs,
    build_index_async,
    count_unique_documents,
)

__all__ = [
    "ensure_dirs",
    "build_index_async",
    "count_unique_documents",
]
