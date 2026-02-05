"""
Conversation and message database models for chat history.

This module re-exports the models defined in app.database.base
to maintain backwards compatibility and provide a clean import path.
"""

# Import models from the canonical location in app.database.base
from app.database.base import Conversation, ChatMessage

__all__ = ["Conversation", "ChatMessage"]
