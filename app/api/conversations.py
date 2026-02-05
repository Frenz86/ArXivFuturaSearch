"""
Conversation API endpoints.

Provides endpoints for managing chat conversations and history.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversation.manager import ConversationManager
from app.conversation.models import Conversation
from app.auth.dependencies import require_authenticated_user, get_optional_user
from app.database.base import User
from app.database.session import get_db
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])


@router.post("")
async def create_conversation(
    title: Optional[str] = None,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new conversation session."""
    manager = ConversationManager(db)

    conversation = await manager.create_conversation(
        user_id=current_user.id if current_user else None,
        title=title,
    )

    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
    }


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Get conversation details with messages."""
    manager = ConversationManager(db)

    conversation = await manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check ownership
    if conversation.user_id and conversation.user_id != (current_user.id if current_user else None):
        raise HTTPException(status_code=403, detail="Not your conversation")

    context = await manager.get_conversation_context(conversation_id)

    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "messages": context["messages"],
        "summary": context.get("summary"),
    }


@router.get("")
async def list_conversations(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """List user's conversations."""
    if not current_user:
        return {"conversations": []}

    manager = ConversationManager(db)

    conversations = await manager.list_user_conversations(
        user_id=current_user.id,
        limit=limit,
        offset=offset,
    )

    return {
        "conversations": [
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "message_count": len(conv.messages),
            }
            for conv in conversations
        ],
        "count": len(conversations),
    }


@router.post("/{conversation_id}/messages")
async def add_message(
    conversation_id: str,
    role: str,
    content: str,
    sources: Optional[List[dict]] = None,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Add a message to a conversation."""
    manager = ConversationManager(db)

    # Verify conversation exists
    conversation = await manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check ownership
    if conversation.user_id and conversation.user_id != (current_user.id if current_user else None):
        raise HTTPException(status_code=403, detail="Not your conversation")

    # Validate role
    if role not in ["user", "assistant", "system"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    message = await manager.add_message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=sources,
    )

    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "timestamp": message.timestamp.isoformat(),
    }


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation (soft delete by default)."""
    manager = ConversationManager(db)

    success = await manager.delete_conversation(
        conversation_id=conversation_id,
        user_id=current_user.id if current_user else None,
    )

    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"message": "Conversation deleted"}


@router.get("/{conversation_id}/summarize")
async def summarize_conversation(
    conversation_id: str,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate conversation summary."""
    manager = ConversationManager(db)

    # Verify conversation exists and check ownership
    conversation = await manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if conversation.user_id and conversation.user_id != (current_user.id if current_user else None):
        raise HTTPException(status_code=403, detail="Not your conversation")

    summary = await manager.summarize_conversation(conversation_id)

    return {"summary": summary}


@router.get("/search")
async def search_conversations(
    query: str,
    limit: int = Query(10, ge=1, le=50),
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Search user's conversations by content."""
    if not current_user:
        return {"results": []}

    manager = ConversationManager(db)

    results = await manager.search_conversations(
        user_id=current_user.id,
        query=query,
        limit=limit,
    )

    return {"results": results}
