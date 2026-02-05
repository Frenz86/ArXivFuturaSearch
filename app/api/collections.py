"""
Collections API endpoints.

Provides endpoints for managing shared paper collections.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.collections.manager import CollectionManager
from app.auth.dependencies import require_authenticated_user, get_optional_user
from app.database.base import User
from app.database.session import get_db
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/collections", tags=["Collections"])


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class CollectionCreate(BaseModel):
    """Schema for creating a collection."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    is_public: bool = False


class CollectionResponse(BaseModel):
    """Schema for collection response."""
    id: str
    user_id: str
    name: str
    description: Optional[str]
    is_public: bool
    share_token: Optional[str]
    created_at: str
    updated_at: str
    paper_count: int = 0


class CollectionUpdate(BaseModel):
    """Schema for updating a collection."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    is_public: Optional[bool] = None


class AddPaperRequest(BaseModel):
    """Schema for adding a paper to a collection."""
    paper_id: str
    notes: Optional[str] = None


class AnnotationCreate(BaseModel):
    """Schema for creating an annotation."""
    paper_id: str
    collection_id: Optional[str] = None
    annotation_type: str = Field(..., pattern="^(note|highlight|question)$")
    content: str = Field(..., min_length=1)
    position: Optional[dict] = None


class AnnotationResponse(BaseModel):
    """Schema for annotation response."""
    id: str
    user_id: str
    paper_id: str
    collection_id: Optional[str]
    annotation_type: str
    content: str
    position: Optional[dict]
    created_at: str
    updated_at: str


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/", response_model=CollectionResponse)
async def create_collection(
    collection_data: CollectionCreate,
    current_user: User = Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new paper collection.

    Requires authentication.
    """
    manager = CollectionManager(db)

    collection = await manager.create_collection(
        user_id=current_user.id,
        name=collection_data.name,
        description=collection_data.description,
        is_public=collection_data.is_public,
    )

    return CollectionResponse(
        id=collection.id,
        user_id=collection.user_id,
        name=collection.name,
        description=collection.description,
        is_public=collection.is_public,
        share_token=collection.share_token,
        created_at=collection.created_at.isoformat(),
        updated_at=collection.updated_at.isoformat(),
    )


@router.get("/", response_model=List[CollectionResponse])
async def get_collections(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user's collections.

    Returns public collections if not authenticated, otherwise
    returns user's private collections plus all public collections.
    """
    manager = CollectionManager(db)

    if current_user:
        collections = await manager.get_user_collections(
            user_id=current_user.id,
            skip=skip,
            limit=limit,
        )
    else:
        collections = await manager.get_public_collections(
            skip=skip,
            limit=limit,
        )

    return [
        CollectionResponse(
            id=c.id,
            user_id=c.user_id,
            name=c.name,
            description=c.description,
            is_public=c.is_public,
            share_token=c.share_token,
            created_at=c.created_at.isoformat(),
            updated_at=c.updated_at.isoformat(),
            paper_count=0,  # Would be populated with actual count
        )
        for c in collections
    ]


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: str,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a collection by ID or share token.

    Public collections can be accessed by anyone.
    Private collections require authentication and ownership.
    """
    manager = CollectionManager(db)

    collection = await manager.get_collection(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    # Check access
    if not collection.is_public:
        if not current_user or current_user.id != collection.user_id:
            raise HTTPException(status_code=403, detail="Access denied")

    return CollectionResponse(
        id=collection.id,
        user_id=collection.user_id,
        name=collection.name,
        description=collection.description,
        is_public=collection.is_public,
        share_token=collection.share_token,
        created_at=collection.created_at.isoformat(),
        updated_at=collection.updated_at.isoformat(),
    )


@router.post("/{collection_id}/papers", response_model=dict)
async def add_paper_to_collection(
    collection_id: str,
    request: AddPaperRequest,
    current_user: User = Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Add a paper to a collection.

    Requires authentication and collection ownership.
    """
    manager = CollectionManager(db)

    await manager.add_paper_to_collection(
        collection_id=collection_id,
        paper_id=request.paper_id,
        user_id=current_user.id,
        notes=request.notes,
    )

    return {"status": "success", "message": "Paper added to collection"}


@router.delete("/{collection_id}/papers/{paper_id}", response_model=dict)
async def remove_paper_from_collection(
    collection_id: str,
    paper_id: str,
    current_user: User = Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Remove a paper from a collection.

    Requires authentication and collection ownership.
    """
    manager = CollectionManager(db)

    await manager.remove_paper_from_collection(
        collection_id=collection_id,
        paper_id=paper_id,
        user_id=current_user.id,
    )

    return {"status": "success", "message": "Paper removed from collection"}


@router.post("/{collection_id}/share", response_model=dict)
async def generate_share_link(
    collection_id: str,
    current_user: User = Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a shareable link for a collection.

    Requires authentication and collection ownership.
    """
    manager = CollectionManager(db)

    share_token = await manager.generate_share_token(
        collection_id=collection_id,
        user_id=current_user.id,
    )

    return {
        "status": "success",
        "share_url": f"/api/collections/shared/{share_token}",
        "share_token": share_token
    }


@router.get("/shared/{share_token}", response_model=CollectionResponse)
async def get_shared_collection(
    share_token: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a collection via its share token.

    Does not require authentication.
    """
    manager = CollectionManager(db)

    collection = await manager.get_collection_by_share_token(share_token)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    return CollectionResponse(
        id=collection.id,
        user_id=collection.user_id,
        name=collection.name,
        description=collection.description,
        is_public=collection.is_public,
        share_token=collection.share_token,
        created_at=collection.created_at.isoformat(),
        updated_at=collection.updated_at.isoformat(),
    )


@router.delete("/{collection_id}", response_model=dict)
async def delete_collection(
    collection_id: str,
    current_user: User = Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a collection.

    Requires authentication and collection ownership.
    """
    manager = CollectionManager(db)

    await manager.delete_collection(
        collection_id=collection_id,
        user_id=current_user.id,
    )

    return {"status": "success", "message": "Collection deleted"}
