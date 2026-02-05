"""
Export audit logs in various formats.

Provides JSON and CSV export functionality for audit logs.
"""

import csv
import json
from typing import List
from datetime import datetime
from io import StringIO

from app.database.base import AuditLog
from app.logging_config import get_logger

logger = get_logger(__name__)


class AuditLogExporter:
    """Export audit logs in various formats."""

    @staticmethod
    async def to_json(logs: List[AuditLog]) -> str:
        """
        Export logs as JSON.

        Args:
            logs: List of AuditLog entries

        Returns:
            JSON string with log data
        """
        return json.dumps(
            [
                {
                    "id": log.id,
                    "user_id": log.user_id,
                    "action": log.action,
                    "resource": log.resource,
                    "resource_id": log.resource_id,
                    "details": log.details,
                    "ip_address": log.ip_address,
                    "user_agent": log.user_agent,
                    "success": log.success,
                    "error_message": log.error_message,
                    "correlation_id": log.correlation_id,
                    "created_at": log.created_at.isoformat(),
                }
                for log in logs
            ],
            indent=2,
        )

    @staticmethod
    async def to_csv(logs: List[AuditLog]) -> str:
        """
        Export logs as CSV.

        Args:
            logs: List of AuditLog entries

        Returns:
            CSV string with log data
        """
        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "timestamp",
                "user_id",
                "action",
                "resource",
                "resource_id",
                "ip_address",
                "success",
                "error_message",
                "correlation_id",
            ],
        )
        writer.writeheader()

        for log in logs:
            writer.writerow(
                {
                    "timestamp": log.created_at.isoformat(),
                    "user_id": log.user_id or "",
                    "action": log.action,
                    "resource": log.resource or "",
                    "resource_id": log.resource_id or "",
                    "ip_address": log.ip_address or "",
                    "success": "true" if log.success else "false",
                    "error_message": log.error_message or "",
                    "correlation_id": log.correlation_id or "",
                }
            )

        return output.getvalue()

    @staticmethod
    async def save_to_file(
        logs: List[AuditLog],
        file_path: str,
        format: str = "json",
    ) -> None:
        """
        Save logs to a file.

        Args:
            logs: List of AuditLog entries
            file_path: Path to save file
            format: File format (json or csv)
        """
        from pathlib import Path

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            content = await AuditLogExporter.to_json(logs)
        elif format == "csv":
            content = await AuditLogExporter.to_csv(logs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        with open(path, "w") as f:
            f.write(content)

        logger.info(
            "Audit logs exported",
            path=str(path),
            count=len(logs),
            format=format,
        )
