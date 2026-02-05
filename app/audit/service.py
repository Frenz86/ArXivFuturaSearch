"""
Audit logging service for security and compliance.

Records and retrieves audit logs for authentication attempts,
search queries, configuration changes, and admin operations.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from app.database.base import AuditLog, User
from app.logging_config import get_logger

logger = get_logger(__name__)


class AuditService:
    """
    Service for recording and retrieving audit logs.

    Provides methods for logging security events and querying
    audit history for compliance and monitoring.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize audit service.

        Args:
            db: Database session
        """
        self.db = db

    async def log_event(
        self,
        action: str,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        user_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditLog:
        """
        Record an audit event.

        Args:
            action: Action performed (e.g., 'auth.login', 'search.query')
            resource: Resource type (e.g., 'auth', 'search', 'index')
            resource_id: ID of the affected resource
            details: Additional event details as JSON
            ip_address: Client IP address
            user_agent: Client user agent string
            user_id: User ID (None for guest/anonymous)
            success: Whether the action succeeded
            error_message: Error message if action failed
            correlation_id: Request correlation ID for tracing

        Returns:
            Created AuditLog entry
        """
        log_entry = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            correlation_id=correlation_id,
            created_at=datetime.utcnow(),
        )

        self.db.add(log_entry)
        await self.db.commit()

        logger.info(
            "Audit log recorded",
            action=action,
            user_id=user_id,
            resource=resource,
            success=success,
        )

        return log_entry

    async def get_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        success_only: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLog]:
        """
        Retrieve audit logs with filtering.

        Args:
            user_id: Filter by user ID
            action: Filter by action (e.g., 'auth.login')
            resource: Filter by resource type
            start_date: Filter logs after this date
            end_date: Filter logs before this date
            success_only: Only show successful (True) or failed (False) events
            limit: Maximum number of logs to return
            offset: Number of logs to skip

        Returns:
            List of AuditLog entries
        """
        query = select(AuditLog)

        conditions = []

        if user_id:
            conditions.append(AuditLog.user_id == user_id)

        if action:
            conditions.append(AuditLog.action == action)

        if resource:
            conditions.append(AuditLog.resource == resource)

        if start_date:
            conditions.append(AuditLog.created_at >= start_date)

        if end_date:
            conditions.append(AuditLog.created_at <= end_date)

        if success_only is not None:
            conditions.append(AuditLog.success == success_only)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(AuditLog.created_at.desc())
        query = query.limit(limit).offset(offset)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_failed_login_attempts(
        self,
        ip_address: Optional[str] = None,
        email: Optional[str] = None,
        minutes: int = 15,
    ) -> List[AuditLog]:
        """
        Get recent failed login attempts for rate limiting and security monitoring.

        Args:
            ip_address: Filter by IP address
            email: Filter by email address (searched in details)
            minutes: Look back period in minutes

        Returns:
            List of failed login attempt logs
        """
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)

        query = select(AuditLog).where(
            and_(
                AuditLog.action.in_(["auth.login_failed", "auth.refresh_failed"]),
                AuditLog.created_at >= cutoff,
                AuditLog.success == False,
            )
        )

        if ip_address:
            query = query.where(AuditLog.ip_address == ip_address)

        result = await self.db.execute(query)
        logs = result.scalars().all()

        # Filter by email if provided (stored in details)
        if email:
            logs = [log for log in logs if log.details.get("email") == email]

        return logs

    async def detect_suspicious_activity(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Detect suspicious activity patterns for a user.

        Analyzes recent audit logs to identify potential security issues:
        - Multiple failed login attempts
        - Unusual access patterns (different IPs)
        - Failed search queries (potential injection attempts)

        Args:
            user_id: User ID to analyze

        Returns:
            Dictionary with analysis results
        """
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        # Failed logins in last hour
        failed_logins_result = await self.db.execute(
            select(AuditLog).where(
                and_(
                    AuditLog.user_id == user_id,
                    AuditLog.action.in_(["auth.login_failed", "auth.refresh_failed"]),
                    AuditLog.created_at >= hour_ago,
                )
            )
        )
        failed_logins = len(failed_logins_result.scalars().all())

        # Unique IPs in last day
        ips_result = await self.db.execute(
            select(AuditLog.ip_address).where(
                and_(
                    AuditLog.user_id == user_id,
                    AuditLog.created_at >= day_ago,
                    AuditLog.ip_address.isnot(None),
                )
            ).distinct()
        )
        unique_ips = len([ip for ip in ips_result.scalars().all() if ip])

        # Failed/invalid searches in last hour
        failed_searches_result = await self.db.execute(
            select(AuditLog).where(
                and_(
                    AuditLog.user_id == user_id,
                    AuditLog.action == "search.query",
                    AuditLog.success == False,
                    AuditLog.created_at >= hour_ago,
                )
            )
        )
        failed_searches = len(failed_searches_result.scalars().all())

        # Determine if activity is suspicious
        suspicious = (
            failed_logins > 5
            or unique_ips > 10
            or failed_searches > 20
        )

        return {
            "user_id": user_id,
            "failed_auth_attempts": failed_logins,
            "unique_ips": unique_ips,
            "failed_searches": failed_searches,
            "suspicious": suspicious,
            "analyzed_at": now.isoformat(),
        }

    async def get_user_activity_summary(
        self,
        user_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get activity summary for a user.

        Provides statistics on user activity over a time period.

        Args:
            user_id: User ID
            days: Number of days to analyze

        Returns:
            Dictionary with activity statistics
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Get all user logs in period
        result = await self.db.execute(
            select(AuditLog).where(
                and_(
                    AuditLog.user_id == user_id,
                    AuditLog.created_at >= cutoff,
                )
            )
        )
        logs = result.scalars().all()

        # Analyze by action
        action_counts: Dict[str, int] = {}
        success_count = 0
        failure_count = 0

        for log in logs:
            action_counts[log.action] = action_counts.get(log.action, 0) + 1
            if log.success:
                success_count += 1
            else:
                failure_count += 1

        # Most common actions
        top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "user_id": user_id,
            "period_days": days,
            "total_actions": len(logs),
            "successful_actions": success_count,
            "failed_actions": failure_count,
            "success_rate": success_count / len(logs) if logs else 0,
            "top_actions": dict(top_actions),
            "last_activity": logs[0].created_at.isoformat() if logs else None,
        }

    async def cleanup_old_logs(
        self,
        retention_days: int = 90,
    ) -> int:
        """
        Delete audit logs older than retention period.

        Args:
            retention_days: Number of days to retain logs

        Returns:
            Number of logs deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=retention_days)

        # Get logs to delete
        result = await self.db.execute(
            select(AuditLog).where(AuditLog.created_at < cutoff)
        )
        logs_to_delete = result.scalars().all()
        count = len(logs_to_delete)

        # Delete logs
        for log in logs_to_delete:
            await self.db.delete(log)

        await self.db.commit()

        logger.info(
            "Audit logs cleaned up",
            deleted_count=count,
            retention_days=retention_days,
        )

        return count

    async def get_statistics(
        self,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get overall audit statistics for monitoring.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with statistics
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Get all logs in period
        result = await self.db.execute(
            select(AuditLog).where(AuditLog.created_at >= cutoff)
        )
        logs = result.scalars().all()

        # Analyze
        total = len(logs)
        successful = sum(1 for log in logs if log.success)
        failed = total - successful

        # By action
        by_action: Dict[str, Dict[str, int]] = {}
        for log in logs:
            if log.action not in by_action:
                by_action[log.action] = {"success": 0, "failure": 0}
            if log.success:
                by_action[log.action]["success"] += 1
            else:
                by_action[log.action]["failure"] += 1

        # By resource
        by_resource: Dict[str, int] = {}
        for log in logs:
            resource = log.resource or "unknown"
            by_resource[resource] = by_resource.get(resource, 0) + 1

        return {
            "period_days": days,
            "total_events": total,
            "successful_events": successful,
            "failed_events": failed,
            "success_rate": successful / total if total > 0 else 0,
            "by_action": by_action,
            "by_resource": by_resource,
        }
