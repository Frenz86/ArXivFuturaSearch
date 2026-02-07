"""
Notification service for sending alerts via email and webhook.

Sends notifications when ArXiv papers match user alert criteria.
"""

import asyncio
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Dict, Any
from datetime import datetime, UTC

import httpx

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class NotificationService:
    """
    Service for sending alert notifications.

    Supports email and webhook notifications.
    """

    async def send_email(
        self,
        recipient: str,
        subject: str,
        papers: List[Dict[str, Any]],
        alert_name: str,
    ):
        """
        Send email notification via SMTP.

        Requires SMTP_HOST to be configured in settings.
        Uses STARTTLS and optional auth (SMTP_USER / SMTP_PASSWORD).

        Args:
            recipient: Email address
            subject: Email subject
            papers: List of matching papers
            alert_name: Alert name
        """
        if not settings.SMTP_HOST:
            raise ValueError(
                "SMTP_HOST is not configured. "
                "Set SMTP_HOST (and optionally SMTP_PORT, SMTP_USER, SMTP_PASSWORD) "
                "in your .env to enable email notifications."
            )
        if not recipient:
            raise ValueError("Recipient email address is empty")

        email_body = self._build_email_body(alert_name, papers)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.EMAIL_FROM
        msg["To"] = recipient
        msg.attach(MIMEText(email_body, "html"))

        def _send_sync():
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                server.ehlo()
                server.starttls()
                if settings.SMTP_USER and settings.SMTP_PASSWORD:
                    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.sendmail(settings.EMAIL_FROM, [recipient], msg.as_string())

        await asyncio.to_thread(_send_sync)

        logger.info(
            "Email notification sent",
            recipient=recipient,
            subject=subject,
            paper_count=len(papers),
        )

    async def send_webhook(
        self,
        url: str,
        papers: List[Dict[str, Any]],
        alert_name: str,
    ):
        """
        Send webhook notification.

        Args:
            url: Webhook URL
            papers: List of matching papers
            alert_name: Alert name
        """
        payload = {
            "alert_name": alert_name,
            "triggered_at": datetime.now(UTC).isoformat(),
            "papers": papers,
            "count": len(papers),
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    timeout=10.0,
                )
                response.raise_for_status()

                logger.info(
                    "Webhook notification sent",
                    url=url,
                    alert_name=alert_name,
                    paper_count=len(papers),
                )

            except Exception as e:
                logger.error("Failed to send webhook notification", url=url, error=str(e))
                raise

    def _build_email_body(
        self,
        alert_name: str,
        papers: List[Dict[str, Any]],
    ) -> str:
        """
        Build HTML email body.

        Args:
            alert_name: Alert name
            papers: List of papers

        Returns:
            HTML string
        """
        lines = []

        lines.append("<html>")
        lines.append("<head>")
        lines.append("<style>")
        lines.append("body { font-family: Arial, sans-serif; }")
        lines.append("h1 { color: #333; }")
        lines.append(".paper { margin: 20px 0; padding: 15px; border-left: 3px solid #007bff; }")
        lines.append(".paper-title { font-size: 18px; font-weight: bold; }")
        lines.append(".paper-authors { color: #666; }")
        lines.append(".paper-abstract { color: #444; margin-top: 10px; }")
        lines.append(".paper-link { margin-top: 10px; }")
        lines.append("</style>")
        lines.append("</head>")
        lines.append("<body>")

        lines.append(f"<h1>New Papers Found: {alert_name}</h1>")
        lines.append(f"<p>We found {len(papers)} new paper(s) matching your alert criteria.</p>")

        for i, paper in enumerate(papers, 1):
            lines.append(f"<div class='paper'>")
            lines.append(f"<div class='paper-title'>{i}. {paper['title']}</div>")
            lines.append(f"<div class='paper-authors'>{', '.join(paper['authors'])}</div>")

            abstract = paper.get("summary", "")
            if abstract:
                lines.append(f"<div class='paper-abstract'>{abstract[:300]}...</div>")

            link = paper.get("link", "")
            if link:
                lines.append(f"<div class='paper-link'><a href='{link}'>View on ArXiv</a></div>")

            lines.append("</div>")

        lines.append("</body>")
        lines.append("</html>")

        return "\n".join(lines)
