"""Initial schema

Revision ID: 001_initial
Revises:
Create Date: 2025-02-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('hashed_password', sa.String(255), nullable=True),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('is_verified', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_users_id', 'users', ['id'])
    op.create_index('ix_users_email', 'users', ['email'])

    # Create roles table
    op.create_table(
        'roles',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(50), unique=True, nullable=False),
        sa.Column('description', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    # Create permissions table
    op.create_table(
        'permissions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(100), unique=True, nullable=False),
        sa.Column('resource', sa.String(50), nullable=False),
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('description', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    # Create user_roles junction table
    op.create_table(
        'user_roles',
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), primary_key=True),
        sa.Column('role_id', sa.String(36), sa.ForeignKey('roles.id'), primary_key=True),
        sa.Column('assigned_at', sa.DateTime(), nullable=True),
        sa.Column('assigned_by', sa.String(36), nullable=True),
    )
    op.create_index('ix_user_roles_user_id', 'user_roles', ['user_id'])

    # Create role_permissions junction table
    op.create_table(
        'role_permissions',
        sa.Column('role_id', sa.String(36), sa.ForeignKey('roles.id'), primary_key=True),
        sa.Column('permission_id', sa.String(36), sa.ForeignKey('permissions.id'), primary_key=True),
    )

    # Create user_sessions table
    op.create_table(
        'user_sessions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('refresh_token', sa.String(500), unique=True, nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_user_sessions_user_id', 'user_sessions', ['user_id'])

    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource', sa.String(100), nullable=True),
        sa.Column('resource_id', sa.String(100), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_audit_logs_user_id', 'audit_logs', ['user_id'])
    op.create_index('ix_audit_logs_action', 'audit_logs', ['action'])
    op.create_index('ix_audit_logs_created_at', 'audit_logs', ['created_at'])

    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('title', sa.String(500), nullable=True),
        sa.Column('context_summary', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )

    # Create chat_messages table
    op.create_table(
        'chat_messages',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('conversation_id', sa.String(36), sa.ForeignKey('conversations.id'), nullable=False),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_chat_messages_conversation_id', 'chat_messages', ['conversation_id'])

    # Create saved_searches table
    op.create_table(
        'saved_searches',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('query', sa.Text(), nullable=True),
        sa.Column('filters', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('last_used', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_saved_searches_user_id', 'saved_searches', ['user_id'])

    # Create collections table
    op.create_table(
        'collections',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_public', sa.Boolean(), default=False, nullable=False),
        sa.Column('share_token', sa.String(64), unique=True, nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_collections_user_id', 'collections', ['user_id'])

    # Create collection_papers table
    op.create_table(
        'collection_papers',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('collection_id', sa.String(36), sa.ForeignKey('collections.id'), nullable=False),
        sa.Column('paper_id', sa.String(100), nullable=True),
        sa.Column('added_at', sa.DateTime(), nullable=True),
        sa.Column('order_index', sa.Integer(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
    )
    op.create_index('ix_collection_papers_collection_id', 'collection_papers', ['collection_id'])

    # Create annotations table
    op.create_table(
        'annotations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('paper_id', sa.String(100), nullable=True),
        sa.Column('collection_id', sa.String(36), sa.ForeignKey('collections.id'), nullable=True),
        sa.Column('annotation_type', sa.String(50), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('position', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_annotations_user_id', 'annotations', ['user_id'])
    op.create_index('ix_annotations_paper_id', 'annotations', ['paper_id'])

    # Create alerts table
    op.create_table(
        'alerts',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('keywords', sa.JSON(), nullable=True),
        sa.Column('categories', sa.JSON(), nullable=True),
        sa.Column('authors', sa.JSON(), nullable=True),
        sa.Column('notification_method', sa.String(50), nullable=False),
        sa.Column('notification_config', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('last_triggered', sa.DateTime(), nullable=True),
        sa.Column('trigger_count', sa.Integer(), default=0, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_alerts_user_id', 'alerts', ['user_id'])

    # Create alert_events table
    op.create_table(
        'alert_events',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('alert_id', sa.String(36), sa.ForeignKey('alerts.id'), nullable=False),
        sa.Column('triggered_at', sa.DateTime(), nullable=False),
        sa.Column('papers', sa.JSON(), nullable=True),
        sa.Column('paper_count', sa.Integer(), nullable=False),
        sa.Column('notification_sent', sa.Boolean(), nullable=False),
        sa.Column('notification_status', sa.String(50), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
    )
    op.create_index('ix_alert_events_alert_id', 'alert_events', ['alert_id'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('alert_events')
    op.drop_table('alerts')
    op.drop_table('annotations')
    op.drop_table('collection_papers')
    op.drop_table('collections')
    op.drop_table('saved_searches')
    op.drop_table('chat_messages')
    op.drop_table('conversations')
    op.drop_table('audit_logs')
    op.drop_table('user_sessions')
    op.drop_table('role_permissions')
    op.drop_table('user_roles')
    op.drop_table('permissions')
    op.drop_table('roles')
    op.drop_table('users')
