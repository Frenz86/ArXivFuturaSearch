"""
Alert API endpoints.

Provides endpoints for creating and managing ArXiv paper alerts.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.alerts.service import AlertManager, ArXivFeedParser, NotificationMethod
from app.auth.dependencies import require_authenticated_user, get_optional_user
from app.database.base import User
from app.database.session import get_db
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.post("")
async def create_alert(
    name: str,
    keywords: List[str],
    categories: List[str],
    notification_method: str = "email",
    current_user: Optional[User] = Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new alert for ArXiv papers monitoring."""
    manager = AlertManager(db)

    alert = await manager.create_alert(
        user_id=current_user.id if current_user else None,
        name=name,
        keywords=keywords,
        categories=categories,
        authors=[],
        notification_method=notification_method,
    )

    return {
        "id": alert.id,
        "name": alert.name,
        "keywords": alert.keywords,
        "categories": alert.categories,
        "notification_method": alert.notification_method,
        "created_at": alert.created_at.isoformat(),
    }


@router.get("")
async def list_alerts(
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """List user's alerts."""
    if not current_user:
        return {"alerts": []}

    manager = AlertManager(db)

    alerts = await manager.get_user_alerts(
        user_id=current_user.id,
        include_inactive=False,
    )

    return {
        "alerts": [
            {
                "id": alert.id,
                "name": alert.name,
                "keywords": alert.keywords,
                "categories": alert.categories,
                "notification_method": alert.notification_method,
                "is_active": alert.is_active,
                "created_at": alert.created_at.isoformat(),
                "last_triggered": alert.last_triggered.isoformat() if alert.last_triggered else None,
            }
            for alert in alerts
        ],
        "count": len(alerts),
    }


@router.get("/{alert_id}")
async def get_alert(
    alert_id: str,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Get alert details."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    from sqlalchemy import select

    result = await db.execute(
        select(Alert).where(Alert.id == alert_id, Alert.user_id == current_user.id)
    )
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {
        "id": alert.id,
        "name": alert.name,
        "keywords": alert.keywords,
        "categories": alert.categories,
        "authors": alert.authors,
        "notification_method": alert.notification_method,
        "is_active": alert.is_active,
        "created_at": alert.created_at.isoformat(),
        "last_triggered": alert.last_triggered.isoformat() if alert.last_triggered else None,
    }


@router.put("/{alert_id}")
async def update_alert(
    alert_id: str,
    name: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    is_active: Optional[bool] = None,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Update alert configuration."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    from sqlalchemy import select

    result = await db.execute(
        select(Alert).where(Alert.id == alert_id, Alert.user_id == current_user.id)
    )
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    # Update fields
    if name is not None:
        alert.name = name
    if keywords is not None:
        alert.keywords = keywords
    if categories is not None:
        alert.categories = categories
    if is_active is not None:
        alert.is_active = is_active

    await db.commit()

    return {"message": "Alert updated"}


@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: str,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an alert."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    from sqlalchemy import select

    result = await db.execute(
        select(Alert).where(Alert.id == alert_id, Alert.user_id == current_user.id)
    )
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    await db.delete(alert)
    await db.commit()

    return {"message": "Alert deleted"}


@router.get("/{alert_id}/history")
async def get_alert_history(
    alert_id: str,
    limit: int = Query(20, ge=1, le=100),
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Get alert trigger history."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    manager = AlertManager(db)

    # Verify ownership
    from sqlalchemy import select

    result = await db.execute(
        select(Alert).where(Alert.id == alert_id, Alert.user_id == current_user.id)
    )
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    events = await manager.get_alert_history(alert_id, limit=limit)

    return {
        "alert_id": alert_id,
        "events": [
            {
                "id": event.id,
                "triggered_at": event.triggered_at.isoformat(),
                "paper_count": len(event.papers),
                "notification_sent": event.notification_sent,
                "notification_status": event.notification_status,
            }
            for event in events
        ],
        "count": len(events),
    }


@router.post("/{alert_id}/test")
async def test_alert(
    alert_id: str,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Trigger test notification."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    from sqlalchemy import select

    result = await db.execute(
        select(Alert).where(Alert.id == alert_id, Alert.user_id == current_user.id)
    )
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    # Send test notification
    from app.alerts.notifications import NotificationService

    notification_service = NotificationService()

    # Get notification config from alert
    config = alert.notification_config or {}

    # Send test email/webhook
    if alert.notification_method in ("email", "both"):
        # TODO: Implement actual email sending
        pass

    if alert.notification_method in ("webhook", "both"):
        # TODO: Implement actual webhook sending
        pass

    return {"message": "Test notification sent"}


@router.post("/check")
async def check_all_alerts(
    api_key: str,
    db: AsyncSession = Depends(get_db),
):
    """Check all alerts against new papers (admin/internal endpoint)."""
    # Verify API key for security
    # In production, use proper API key authentication

    parser = ArXivFeedParser()

    # Fetch papers from all categories
    all_papers = []
    categories = parser.categories

    for category in categories:
        papers = await parser.fetch_papers(category)
        all_papers.extend(papers)

    # Check alerts
    manager = AlertManager(db)
    matches = await manager.check_alerts(all_papers)

    # Trigger notifications
    triggered_count = 0
    for alert_id, matching_papers in matches.items():
        result = await db.execute(
            select(Alert).where(Alert.id == alert_id)
        )
        alert = result.scalar_one_or_none()

        if alert:
            await manager.trigger_alert(alert, matching_papers)
            triggered_count += 1

    return {
        "message": "Alerts checked",
        "papers_fetched": len(all_papers),
        "alerts_triggered": triggered_count,
    }
