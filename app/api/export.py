"""
Export API endpoints.

Provides endpoints for exporting search results in various formats.
"""

from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.export.manager import ExportManager, CitationStyle
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/export", tags=["Export"])


class ExportRequest(BaseModel):
    """Request model for export."""
    results: List[dict] = Field(..., min_items=1)
    query: str
    answer: str
    citation_style: str = Field(default=CitationStyle.APA)
    include_abstracts: bool = True


@router.post("/pdf")
async def export_pdf(req: ExportRequest):
    """Export results as PDF (or HTML for now)."""
    manager = ExportManager()

    try:
        content = manager.export_pdf(
            results=req.results,
            query=req.query,
            answer=req.answer,
            citation_style=req.citation_style,
            include_abstracts=req.include_abstracts,
        )

        filename = f"search_results_{req.query[:30]}.html"

        return Response(
            content=content,
            media_type="text/html",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        logger.error("PDF export failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/markdown")
async def export_markdown(req: ExportRequest):
    """Export as Markdown."""
    manager = ExportManager()

    try:
        content = manager.export_markdown(
            results=req.results,
            query=req.query,
            answer=req.answer,
            citation_style=req.citation_style,
        )

        filename = f"search_results_{req.query[:30]}.md"

        return Response(
            content=content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        logger.error("Markdown export failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/bibtex")
async def export_bibtex(req: ExportRequest):
    """Export as BibTeX."""
    manager = ExportManager()

    try:
        content = manager.export_bibtex(req.results)

        filename = "references.bib"

        return Response(
            content=content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        logger.error("BibTeX export failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/json")
async def export_json(req: ExportRequest):
    """Export as JSON."""
    manager = ExportManager()

    try:
        content = manager.export_json(req.results)

        filename = "search_results.json"

        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        logger.error("JSON export failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/csv")
async def export_csv(req: ExportRequest):
    """Export as CSV."""
    manager = ExportManager()

    try:
        content = manager.export_csv(req.results)

        filename = "search_results.csv"

        return Response(
            content=content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        logger.error("CSV export failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
