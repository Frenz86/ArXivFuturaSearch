"""
Alert system for monitoring ArXiv and notifying users of new papers.

Monitors ArXiv RSS feeds for new papers matching user keywords and sends notifications.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import httpx
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.base import Alert, AlertEvent
from app.logging_config import get_logger

logger = get_logger(__name__)


class NotificationMethod(str, Enum):
    """Notification delivery methods."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    BOTH = "both"


# =============================================================================
# ALERT MANAGER
# =============================================================================

class ArXivFeedParser:
    """
    Parser for ArXiv RSS feeds.

    Fetches and parses papers from ArXiv RSS feeds.
    """

    CATEGORIES = [
        "cs.AI",  # Artificial Intelligence
        "cs.LG",  # Machine Learning
        "cs.CL",  # Computation and Language
        "cs.CV",  # Computer Vision
        "stat.ML",  # Machine Learning (Statistics)
    ]

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        max_results: int = 100,
        timeout: int = 30,
    ):
        """
        Initialize ArXiv feed parser.

        Args:
            categories: ArXiv categories to fetch
            max_results: Maximum number of papers per category
            timeout: HTTP timeout in seconds
        """
        self.categories = categories or self.CATEGORIES
        self.max_results = max_results
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def fetch_papers(
        self,
        category: str,
    ) -> List[Dict[str, Any]]:
        """
        Fetch papers from ArXiv category via RSS.

        Args:
            category: ArXiv category (e.g., "cs.AI")

        Returns:
            List of paper dictionaries
        """
        import feedparser

        client = await self._get_client()

        # ArXiv RSS URL
        rss_url = f"https://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results={self.max_results}"

        try:
            response = await client.get(rss_url)
            response.raise_for_status()

            # Parse RSS feed
            feed = feedparser.parse(response.content)

            papers = []
            for entry in feed.entries:
                paper = {
                    "id": entry.id.split("/abs/")[-1],
                    "title": entry.title,
                    "summary": entry.get("summary", ""),
                    "authors": [author.name for author in entry.get("authors", [])],
                    "published": entry.get("published", ""),
                    "link": entry.link,
                    "tags": [tag.term for tag in entry.get("tags", [])],
                }
                papers.append(paper)

            logger.info(
                "Fetched papers from ArXiv",
                category=category,
                count=len(papers),
            )

            return papers

        except Exception as e:
            logger.error("Failed to fetch ArXiv feed", category=category, error=str(e))
            return []

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class AlertManager:
    """
    Manager for alert configuration and triggering.

    Handles alert creation, modification, and checking against new papers.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize alert manager.

        Args:
            db: Database session
        """
        self.db = db
        self.parser = ArXivFeedParser()

    async def create_alert(
        self,
        user_id: str,
        name: str,
        keywords: List[str],
        categories: List[str],
        authors: List[str],
        notification_method: str = "email",
        notification_config: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """
        Create a new alert.

        Args:
            user_id: User ID
            name: Alert name
            keywords: List of keywords to match
            categories: ArXiv categories
            authors: Author names
            notification_method: How to send notifications
            notification_config: Notification configuration

        Returns:
            Created Alert
        """
        alert = Alert(
            user_id=user_id,
            name=name,
            keywords=keywords,
            categories=categories,
            authors=authors,
            notification_method=notification_method,
            notification_config=notification_config or {},
        )

        self.db.add(alert)
        await self.db.commit()
        await self.db.refresh(alert)

        logger.info(
            "Alert created",
            alert_id=alert.id,
            user_id=user_id,
            name=name,
        )

        return alert

    async def get_user_alerts(
        self,
        user_id: str,
        include_inactive: bool = False,
    ) -> List[Alert]:
        """
        Get user's alerts.

        Args:
            user_id: User ID
            include_inactive: Include inactive alerts

        Returns:
            List of Alert
        """
        query = select(Alert).where(Alert.user_id == user_id)

        if not include_inactive:
            query = query.where(Alert.is_active == True)

        query = query.order_by(Alert.created_at.desc())

        result = await self.db.execute(query)
        return result.scalars().all()

    async def check_alerts(
        self,
        papers: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Check all active alerts against new papers.

        Args:
            papers: List of papers to check

        Returns:
            Dictionary mapping alert IDs to matching papers
        """
        # Get all active alerts
        result = await self.db.execute(
            select(Alert).where(Alert.is_active == True)
        )
        alerts = result.scalars().all()

        matches = {}

        for alert in alerts:
            matching_papers = []

            for paper in papers:
                if self._paper_matches_alert(paper, alert):
                    matching_papers.append(paper)

            if matching_papers:
                matches[alert.id] = matching_papers

                # Create alert event
                await self._create_alert_event(alert.id, matching_papers)

        logger.info(
            "Alerts checked",
            total_alerts=len(alerts),
            triggered_alerts=len(matches),
            total_papers=len(papers),
        )

        return matches

    def _paper_matches_alert(
        self,
        paper: Dict[str, Any],
        alert: Alert,
    ) -> bool:
        """
        Check if a paper matches alert criteria.

        Args:
            paper: Paper dictionary
            alert: Alert configuration

        Returns:
            True if paper matches
        """
        # Check keywords
        if alert.keywords:
            paper_text = f"{paper['title']} {paper['summary']}".lower()
            if not any(keyword.lower() in paper_text for keyword in alert.keywords):
                return False

        # Check categories
        if alert.categories:
            paper_tags = paper.get("tags", [])
            if not any(tag in alert.categories for tag in paper_tags):
                return False

        # Check authors
        if alert.authors:
            paper_authors = paper.get("authors", [])
            if not any(author in paper_authors for author in alert.authors):
                return False

        return True

    async def _create_alert_event(
        self,
        alert_id: str,
        papers: List[Dict[str, Any]],
    ) -> AlertEvent:
        """Create an alert event record."""
        event = AlertEvent(
            alert_id=alert_id,
            papers=papers,
        )

        self.db.add(event)
        await self.db.commit()

        return event

    async def trigger_alert(
        self,
        alert: Alert,
        papers: List[Dict[str, Any]],
    ) -> bool:
        """
        Trigger alert notifications.

        Args:
            alert: Alert configuration
            papers: Matching papers

        Returns:
            True if notification sent successfully
        """
        from app.alerts.notifications import NotificationService

        notification_service = NotificationService()

        # Get user email/notification config from alert
        config = alert.notification_config or {}

        success = False

        if alert.notification_method in ("email", "both"):
            try:
                await notification_service.send_email(
                    recipient=config.get("email"),
                    subject=f"New papers matching '{alert.name}'",
                    papers=papers,
                    alert_name=alert.name,
                )
                success = True
            except Exception as e:
                logger.error("Failed to send email notification", error=str(e))

        if alert.notification_method in ("webhook", "both"):
            try:
                await notification_service.send_webhook(
                    url=config.get("webhook_url"),
                    papers=papers,
                    alert_name=alert.name,
                )
                success = True
            except Exception as e:
                logger.error("Failed to send webhook notification", error=str(e))

        # Update alert event status
        for event in alert.events:
            if event.papers == papers:
                event.notification_sent = success
                event.notification_status = "sent" if success else "failed"

        await self.db.commit()

        # Update last triggered time
        alert.last_triggered = datetime.utcnow()

        return success

    async def get_alert_history(
        self,
        alert_id: str,
        limit: int = 20,
    ) -> List[AlertEvent]:
        """Get alert trigger history."""
        result = await self.db.execute(
            select(AlertEvent)
            .where(AlertEvent.alert_id == alert_id)
            .order_by(AlertEvent.triggered_at.desc())
            .limit(limit)
        )
        return result.scalars().all()


# =============================================================================
# ALERT MONITORING BACKGROUND TASK
# =============================================================================

class AlertMonitoringTask:
    """
    Background task for monitoring ArXiv and triggering alerts.

    Runs periodically to fetch new papers and check against active alerts.
    """

    def __init__(
        self,
        check_interval_minutes: int = 60,
        categories: Optional[List[str]] = None,
        db_factory = None,
    ):
        """
        Initialize alert monitoring task.

        Args:
            check_interval_minutes: How often to check for new papers
            categories: ArXiv categories to monitor
            db_factory: Database session factory
        """
        self.check_interval = timedelta(minutes=check_interval_minutes)
        self.categories = categories or ArXivFeedParser.CATEGORIES
        self.db_factory = db_factory
        self.parser = ArXivFeedParser(categories=categories)

    async def monitor_and_alert(self):
        """
        Main monitoring loop.

        Fetches new papers and triggers matching alerts.
        """
        logger.info(
            "Starting alert monitoring",
            interval_minutes=self.check_interval.total_seconds() / 60,
            categories=self.categories,
        )

        while True:
            try:
                # Fetch papers from all categories
                all_papers = []

                for category in self.categories:
                    papers = await self.parser.fetch_papers(category)
                    all_papers.extend(papers)

                if all_papers:
                    # Get database session
                    db = await self.db_factory()
                    try:
                        manager = AlertManager(db)

                        # Check alerts against papers
                        matches = await manager.check_alerts(all_papers)

                        # Trigger notifications for matching alerts
                        for alert_id, matching_papers in matches.items():
                            result = await db.execute(
                                select(Alert).where(Alert.id == alert_id)
                            )
                            alert = result.scalar_one_or_none()

                            if alert:
                                await manager.trigger_alert(alert, matching_papers)
                    finally:
                        await db.close()

                # Wait until next check
                await asyncio.sleep(self.check_interval.total_seconds())

            except asyncio.CancelledError:
                logger.info("Alert monitoring cancelled")
                break
            except Exception as e:
                logger.error("Error in alert monitoring loop", error=str(e))
                await asyncio.sleep(self.check_interval.total_seconds())
