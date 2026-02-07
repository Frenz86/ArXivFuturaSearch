"""
Audit API endpoints.

Provides endpoints for viewing and managing audit logs.
Admin-only access for most operations.
"""

from typing import Optional
from datetime import datetime, timedelta, UTC
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.audit.exporters import AuditLogExporter
from app.auth.dependencies import require_admin, require_authenticated_user, require_audit_read
from app.database.base import User
from app.database.session import get_db
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/audit", tags=["Audit"])


@router.get("/logs")
async def get_audit_logs(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    resource: Optional[str] = None,
    success_only: Optional[bool] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(require_audit_read),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve audit logs (admin/audit-read permission required).

    Supports filtering by date range, user, action, resource, and success status.
    Paginated with limit/offset.
    """
    service = AuditService(db)

    logs = await service.get_logs(
        user_id=user_id,
        action=action,
        resource=resource,
        start_date=start_date,
        end_date=end_date,
        success_only=success_only,
        limit=limit,
        offset=offset,
    )

    return {
        "logs": [
            {
                "id": log.id,
                "user_id": log.user_id,
                "action": log.action,
                "resource": log.resource,
                "resource_id": log.resource_id,
                "details": log.details,
                "ip_address": log.ip_address,
                "success": log.success,
                "error_message": log.error_message,
                "correlation_id": log.correlation_id,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ],
        "count": len(logs),
        "limit": limit,
        "offset": offset,
    }


@router.get("/logs/export")
async def export_audit_logs(
    format: str = Query("json", pattern="^(json|csv)$"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Export audit logs in JSON or CSV format (admin only).

    Returns downloadable file with filtered audit logs.
    """
    service = AuditService(db)

    logs = await service.get_logs(
        start_date=start_date,
        end_date=end_date,
        limit=10000,  # Higher limit for exports
    )

    exporter = AuditLogExporter()

    if format == "json":
        content = await exporter.to_json(logs)
        media_type = "application/json"
        filename = f"audit_logs_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    else:  # csv
        content = await exporter.to_csv(logs)
        media_type = "text/csv"
        filename = f"audit_logs_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.get("/stats")
async def get_audit_statistics(
    days: int = Query(7, ge=1, le=365),
    current_user: User = Depends(require_audit_read),
    db: AsyncSession = Depends(get_db),
):
    """
    Get audit log statistics (admin/audit-read permission required).

    Provides overview of system activity over specified time period.
    """
    service = AuditService(db)

    stats = await service.get_statistics(days=days)

    return stats


@router.get("/suspicious/{user_id}")
async def check_suspicious_activity(
    user_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Check for suspicious activity patterns (admin only).

    Analyzes user activity for potential security issues.
    """
    service = AuditService(db)

    activity = await service.detect_suspicious_activity(user_id)

    return activity


@router.get("/activity/{user_id}")
async def get_user_activity(
    user_id: str,
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Get activity summary for a user (admin only).

    Provides statistics on user actions over time period.
    """
    service = AuditService(db)

    summary = await service.get_user_activity_summary(user_id, days=days)

    return summary


@router.delete("/cleanup")
async def cleanup_old_logs(
    retention_days: int = Query(90, ge=30, le=365),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete audit logs older than retention period (admin only).

    Permanently removes old logs from database.
    """
    service = AuditService(db)

    deleted_count = await service.cleanup_old_logs(retention_days=retention_days)

    return {
        "message": f"Deleted {deleted_count} old audit logs",
        "deleted_count": deleted_count,
        "retention_days": retention_days,
    }


@router.get("/failed-logins")
async def get_failed_login_attempts(
    ip_address: Optional[str] = None,
    email: Optional[str] = None,
    minutes: int = Query(15, ge=1, le=1440),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Get recent failed login attempts (admin only).

    Useful for identifying brute force attacks or compromised accounts.
    """
    service = AuditService(db)

    failed_attempts = await service.get_failed_login_attempts(
        ip_address=ip_address,
        email=email,
        minutes=minutes,
    )

    return {
        "failed_attempts": [
            {
                "user_id": log.user_id,
                "ip_address": log.ip_address,
                "user_agent": log.user_agent,
                "error_message": log.error_message,
                "created_at": log.created_at.isoformat(),
            }
            for log in failed_attempts
        ],
        "count": len(failed_attempts),
        "period_minutes": minutes,
    }
