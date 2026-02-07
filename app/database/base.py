"""
Database models for ArXivFuturaSearch.

Includes models for:
- Authentication (User, Role, Permission, UserSession)
- Audit logging (AuditLog)
- Conversations (Conversation, ChatMessage)
- Collections (SavedSearch, Collection, CollectionPaper, Annotation)
- Alerts (Alert, AlertEvent)
"""

from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Table, Text, Integer, JSON as JSONType, Float, LargeBinary
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime, UTC
import uuid


def _utcnow():
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


# =============================================================================
# AUTH & RBAC MODELS
# =============================================================================

# Many-to-many relationship for users and roles
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', String(36), ForeignKey('users.id'), primary_key=True),
    Column('role_id', String(36), ForeignKey('roles.id'), primary_key=True),
    Column('assigned_at', DateTime, default=_utcnow),
    Column('assigned_by', String(36)),
)

# Role permissions mapping
role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', String(36), ForeignKey('roles.id'), primary_key=True),
    Column('permission_id', String(36), ForeignKey('permissions.id'), primary_key=True),
)


class User(Base):
    """User model for authentication."""
    __tablename__ = 'users'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=True)  # Null for OAuth-only users
    name = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")
    saved_searches = relationship("SavedSearch", back_populates="user")
    collections = relationship("Collection", back_populates="user")
    alerts = relationship("Alert", back_populates="user")
    annotations = relationship("Annotation", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"


class Role(Base):
    """Role model for RBAC."""
    __tablename__ = 'roles'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(50), unique=True, nullable=False)  # 'admin', 'user', 'guest'
    description = Column(String(255))
    created_at = Column(DateTime, default=_utcnow)

    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")

    def __repr__(self):
        return f"<Role(id={self.id}, name={self.name})>"


class Permission(Base):
    """Permission model for fine-grained access control."""
    __tablename__ = 'permissions'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), unique=True, nullable=False)  # 'search:read', 'index:write'
    resource = Column(String(50), nullable=False)  # 'search', 'index', 'admin'
    action = Column(String(50), nullable=False)    # 'read', 'write', 'delete'
    description = Column(String(255))
    created_at = Column(DateTime, default=_utcnow)

    # Relationships
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")

    def __repr__(self):
        return f"<Permission(id={self.id}, name={self.name})>"


class UserSession(Base):
    """User session model for JWT token tracking."""
    __tablename__ = 'user_sessions'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)
    refresh_token = Column(String(500), unique=True, nullable=True)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=_utcnow)
    revoked_at = Column(DateTime, nullable=True)

    # Relationship
    user = relationship("User", back_populates="sessions")

    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id})>"


class AuditLog(Base):
    """Audit log model for security and compliance."""
    __tablename__ = 'audit_logs'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)  # 'auth.login', 'search.query'
    resource = Column(String(100))  # 'auth', 'search', 'index'
    resource_id = Column(String(100))
    details = Column(JSONType)  # Flexible JSON for additional details
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    status = Column(String(20), nullable=False)  # 'success', 'failure'
    created_at = Column(DateTime, default=_utcnow, index=True)

    # Relationship
    user = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, status={self.status})>"


# =============================================================================
# CONVERSATION MODELS
# =============================================================================

class Conversation(Base):
    """Conversation model for multi-turn chat."""
    __tablename__ = 'conversations'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True)
    title = Column(String(500))
    context_summary = Column(Text)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Conversation(id={self.id}, title={self.title})>"


class ChatMessage(Base):
    """Chat message model."""
    __tablename__ = 'chat_messages'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey('conversations.id'), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    token_count = Column(Integer)
    created_at = Column(DateTime, default=_utcnow)

    # Relationship
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role={self.role})>"


# =============================================================================
# COLLECTION MODELS
# =============================================================================

class SavedSearch(Base):
    """Saved search queries table."""
    __tablename__ = 'saved_searches'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True, index=True)
    name = Column(String(255), nullable=False)
    query = Column(Text)  # Search query
    filters = Column(JSONType)  # JSON encoded filters
    created_at = Column(DateTime, default=_utcnow)
    last_used = Column(DateTime)

    # Relationship
    user = relationship("User", back_populates="saved_searches")

    def __repr__(self):
        return f"<SavedSearch(id={self.id}, name={self.name}, user_id={self.user_id})>"


class Collection(Base):
    """Shared paper collections table."""
    __tablename__ = 'collections'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    is_public = Column(Boolean, default=False)
    share_token = Column(String(64), unique=True)  # For sharing links
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # Relationships
    user = relationship("User", back_populates="collections")
    papers = relationship("CollectionPaper", back_populates="collection", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="collection", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Collection(id={self.id}, name={self.name}, is_public={self.is_public})>"


class CollectionPaper(Base):
    """Papers in collections table."""
    __tablename__ = 'collection_papers'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    collection_id = Column(String(36), ForeignKey('collections.id'), nullable=False, index=True)
    paper_id = Column(String(100))  # ArXiv ID
    added_at = Column(DateTime, default=_utcnow)
    order_index = Column(Integer, default=0)  # For custom ordering
    notes = Column(Text)  # User notes for this paper in collection

    # Relationship
    collection = relationship("Collection", back_populates="papers")

    def __repr__(self):
        return f"<CollectionPaper(id={self.id}, collection_id={self.collection_id}, paper_id={self.paper_id})>"


class Annotation(Base):
    """Paper annotations table."""
    __tablename__ = 'annotations'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)
    paper_id = Column(String(100), index=True)  # ArXiv ID
    collection_id = Column(String(36), ForeignKey('collections.id'), nullable=True)
    annotation_type = Column(String(50))  # "note", "highlight", "question"
    content = Column(Text, nullable=False)
    position = Column(JSONType)  # JSON: {"page": 1, "text": "...", "offset": 100}
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # Relationships
    user = relationship("User", back_populates="annotations")
    collection = relationship("Collection", back_populates="annotations")

    def __repr__(self):
        return f"<Annotation(id={self.id}, paper_id={self.paper_id}, type={self.annotation_type})>"


# =============================================================================
# ALERT MODELS
# =============================================================================

class Alert(Base):
    """Alert model for ArXiv paper monitoring."""
    __tablename__ = 'alerts'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True, index=True)
    name = Column(String(255), nullable=False)
    keywords = Column(JSONType)  # List of keywords
    categories = Column(JSONType)  # List of ArXiv categories
    authors = Column(JSONType)  # List of author names
    notification_method = Column(String(50), nullable=False)  # 'email', 'webhook', 'both'
    notification_config = Column(JSONType)  # Email address, webhook URL, etc.
    is_active = Column(Boolean, default=True, nullable=False)
    last_triggered = Column(DateTime)
    trigger_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # Relationship
    user = relationship("User", back_populates="alerts")
    events = relationship("AlertEvent", back_populates="alert", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Alert(id={self.id}, name={self.name}, is_active={self.is_active})>"


class AlertEvent(Base):
    """Alert trigger event history."""
    __tablename__ = 'alert_events'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    alert_id = Column(String(36), ForeignKey('alerts.id'), nullable=False, index=True)
    triggered_at = Column(DateTime, nullable=False)
    papers = Column(JSONType)  # List of matching papers
    paper_count = Column(Integer, nullable=False)
    notification_sent = Column(Boolean, nullable=False)
    notification_status = Column(String(50))  # 'sent', 'failed', 'pending'
    error_message = Column(Text)

    # Relationship
    alert = relationship("Alert", back_populates="events")

    def __repr__(self):
        return f"<AlertEvent(id={self.id}, alert_id={self.alert_id}, paper_count={self.paper_count})>"
