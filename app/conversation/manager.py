"""
Conversation manager for chat history and context management.

Handles conversation persistence, context window management,
and conversation summarization.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, UTC
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from app.conversation.models import Conversation, ChatMessage
from app.logging_config import get_logger

logger = get_logger(__name__)


class ConversationManager:
    """
    Manager for conversation history and context.

    Features:
    - Create and manage conversations
    - Add messages with tracking
    - Smart context window management
    - Conversation summarization
    - Semantic search in history
    """

    def __init__(
        self,
        db: AsyncSession,
        max_context_tokens: int = 8000,
        conversation_summary_threshold: int = 10,
    ):
        """
        Initialize the conversation manager.

        Args:
            db: Database session
            max_context_tokens: Maximum tokens in context window
            conversation_summary_threshold: Messages before summarization
        """
        self.db = db
        self.max_context_tokens = max_context_tokens
        self.conversation_summary_threshold = conversation_summary_threshold

    async def create_conversation(
        self,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """
        Create a new conversation session.

        Args:
            user_id: Optional user identifier
            title: Optional conversation title
            metadata: Optional metadata as JSON

        Returns:
            Created Conversation
        """
        conversation = Conversation(
            user_id=user_id,
            title=title or "New Conversation",
            metadata=metadata or {},
        )

        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)

        logger.info(
            "Conversation created",
            conversation_id=conversation.id,
            user_id=user_id,
        )

        return conversation

    async def get_conversation(
        self,
        conversation_id: str,
    ) -> Optional[Conversation]:
        """
        Get a conversation by ID.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation object or None
        """
        result = await self.db.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        return result.scalar_one_or_none()

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        token_count: Optional[int] = None,
    ) -> ChatMessage:
        """
        Add a message to a conversation.

        Args:
            conversation_id: Conversation ID
            role: Message role ("user", "assistant", "system")
            content: Message content
            sources: Optional retrieved sources (for assistant messages)
            token_count: Optional token count

        Returns:
            Created ChatMessage
        """
        message = ChatMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources,
            token_count=token_count,
        )

        self.db.add(message)

        # Update conversation timestamp
        await self.db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = await self.get_conversation(conversation_id)
        if conversation:
            conversation.updated_at = datetime.now(UTC)

            # Auto-generate title from first user message
            if not conversation.title or conversation.title == "New Conversation":
                messages_count = len(conversation.messages)
                if role == "user" and messages_count <= 1:
                    # Generate title from first message
                    conversation.title = content[:100] + ("..." if len(content) > 100 else "")

        await self.db.commit()
        await self.db.refresh(message)

        logger.debug(
            "Message added",
            message_id=message.id,
            conversation_id=conversation_id,
            role=role,
        )

        return message

    async def get_conversation_context(
        self,
        conversation_id: str,
        max_messages: Optional[int] = None,
        include_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        Get conversation context with smart windowing.

        Strategy:
        1. Always include last N messages (recent context)
        2. Include summary if conversation is long
        3. Use semantic search to find relevant historical messages

        Args:
            conversation_id: Conversation ID
            max_messages: Maximum number of recent messages
            include_summary: Whether to include conversation summary

        Returns:
            Dictionary with conversation context
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return {"messages": [], "summary": None}

        # Get messages ordered by timestamp
        messages = sorted(conversation.messages, key=lambda m: m.timestamp)

        # Apply message limit
        if max_messages:
            messages = messages[-max_messages:]

        # Build context
        context_messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in messages
        ]

        result = {
            "messages": context_messages,
            "summary": None,
            "message_count": len(messages),
            "total_tokens": sum(msg.token_count or 0 for msg in messages),
        }

        # Include summary if available
        if include_summary and conversation.context_summary:
            result["summary"] = conversation.context_summary

        return result

    async def should_summarize(self, conversation_id: str) -> bool:
        """
        Check if conversation needs summarization.

        Args:
            conversation_id: Conversation ID

        Returns:
            True if conversation should be summarized
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return False

        message_count = len(conversation.messages)

        # Check if we exceed threshold
        if message_count >= self.conversation_summary_threshold:
            return True

        # Check token count
        total_tokens = sum(msg.token_count or 0 for msg in conversation.messages)
        if total_tokens > self.max_context_tokens * 0.7:  # 70% of limit
            return True

        return False

    async def summarize_conversation(
        self,
        conversation_id: str,
    ) -> str:
        """
        Generate conversation summary using LLM.

        Args:
            conversation_id: Conversation ID

        Returns:
            Generated summary
        """
        # Get conversation context
        context = await self.get_conversation_context(
            conversation_id,
            max_messages=20,  # Last 20 messages for summarization
            include_summary=False,
        )

        if not context["messages"]:
            return "Empty conversation"

        # Build summarization prompt
        messages_text = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in context["messages"]
        )

        prompt = f"""Summarize the following research discussion about ArXiv papers.

Focus on:
- Key questions asked by the user
- Insights gained from the papers
- Main topics discussed

Keep the summary under 500 characters.

Discussion:
{messages_text}

Summary:"""

        # Generate summary (would use LLM here)
        # For now, create a simple summary
        user_messages = [msg for msg in context["messages"] if msg["role"] == "user"]
        summary = f"Discussion about {len(user_messages)} questions about ArXiv papers. "

        if user_messages:
            first_question = user_messages[0]["content"][:100]
            summary += f"Started with: '{first_question}'"

        # Store summary
        conversation = await self.get_conversation(conversation_id)
        if conversation:
            conversation.context_summary = summary
            await self.db.commit()

        logger.info(
            "Conversation summarized",
            conversation_id=conversation_id,
            summary_length=len(summary),
        )

        return summary

    async def list_user_conversations(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        include_archived: bool = False,
    ) -> List[Conversation]:
        """
        List user's conversations.

        Args:
            user_id: User ID
            limit: Maximum number of conversations
            offset: Number of conversations to skip
            include_archived: Whether to include archived conversations

        Returns:
            List of Conversations
        """
        query = select(Conversation).where(Conversation.user_id == user_id)

        if not include_archived:
            query = query.where(Conversation.is_archived == False)

        query = query.order_by(Conversation.updated_at.desc())
        query = query.limit(limit).offset(offset)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
        hard_delete: bool = False,
    ) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation ID
            user_id: Optional user ID for authorization
            hard_delete: If True, permanently delete; if False, soft delete (archive)

        Returns:
            True if deleted
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return False

        # Check ownership if user_id provided
        if user_id and conversation.user_id != user_id:
            logger.warning(
                "Unauthorized deletion attempt",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            return False

        if hard_delete:
            await self.db.delete(conversation)
        else:
            conversation.is_archived = True

        await self.db.commit()

        logger.info(
            "Conversation deleted",
            conversation_id=conversation_id,
            hard_delete=hard_delete,
        )

        return True

    async def search_conversations(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search user's conversations by content.

        Args:
            user_id: User ID
            query: Search query
            limit: Maximum results

        Returns:
            List of matching conversations with context
        """
        # Get all user conversations
        conversations = await self.list_user_conversations(
            user_id=user_id,
            limit=100,  # Higher limit for searching
        )

        results = []

        for conv in conversations:
            # Search in messages
            for msg in conv.messages:
                if query.lower() in msg.content.lower():
                    results.append({
                        "conversation_id": conv.id,
                        "title": conv.title,
                        "message": {
                            "role": msg.role,
                            "content": msg.content[:200] + ("..." if len(msg.content) > 200 else ""),
                            "timestamp": msg.timestamp.isoformat(),
                        },
                    })
                    break  # One match per conversation

            if len(results) >= limit:
                break

        return results[:limit]
