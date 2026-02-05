"""
Export manager for search results in various formats.

Supports PDF, Markdown, BibTeX, JSON, and CSV exports with proper citations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from io import BytesIO

from app.logging_config import get_logger

logger = get_logger(__name__)


class CitationStyle(str):
    """Citation styles."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    BIBTEX = "bibtex"


class ExportManager:
    """
    Export search results in various formats.

    Features:
    - PDF generation with citations
    - Markdown export with formatting
    - BibTeX citation export
    - JSON/CSV data export
    """

    def export_pdf(
        self,
        results: List[Dict[str, Any]],
        query: str,
        answer: str,
        citation_style: str = CitationStyle.APA,
        include_abstracts: bool = True,
    ) -> bytes:
        """
        Generate PDF with citations.

        Args:
            results: Search results with paper metadata
            query: Original search query
            answer: Generated answer
            citation_style: Citation style
            include_abstracts: Include paper abstracts

        Returns:
            PDF file as bytes
        """
        # Create HTML template for PDF
        html = self._generate_html_export(
            results, query, answer, citation_style, include_abstracts
        )

        # For now, return HTML (would use weasyprint/reportlab in production)
        return html.encode("utf-8")

    def export_markdown(
        self,
        results: List[Dict[str, Any]],
        query: str,
        answer: str,
        citation_style: str = CitationStyle.APA,
    ) -> str:
        """
        Generate Markdown with proper formatting.

        Args:
            results: Search results with paper metadata
            query: Original search query
            answer: Generated answer
            citation_style: Citation style

        Returns:
            Markdown string
        """
        lines = []

        # Title
        lines.append(f"# Search Results: {query}\n")

        # Answer
        lines.append("## Answer\n")
        lines.append(f"{answer}\n")

        # References
        lines.append("## References\n")

        for i, result in enumerate(results, 1):
            paper = result.get("meta", {})
            title = paper.get("title", "Untitled")
            authors = paper.get("authors", "")
            year = paper.get("published", "")[:4]
            link = paper.get("link", "")

            # Format citation
            citation = self._format_citation(paper, citation_style)
            lines.append(f"{i}. {citation}")

            if link:
                lines.append(f"   - **Link**: [{link}]({link})")

            lines.append("")  # Empty line

        return "\n".join(lines)

    def export_bibtex(
        self,
        results: List[Dict[str, Any]],
    ) -> str:
        """
        Generate BibTeX entries.

        Args:
            results: Search results with paper metadata

        Returns:
            BibTeX string
        """
        lines = []

        for result in results:
            paper = result.get("meta", {})
            arxiv_id = paper.get("paper_id", "unknown")
            title = paper.get("title", "Untitled")
            authors = paper.get("authors", "Unknown")
            year = paper.get("published", "")[:4]

            # Create BibTeX key
            first_author_last = authors.split(",")[0].strip() if authors else "unknown"
            key = f"{first_author_last}{year}{arxiv_id}"

            # Build BibTeX entry with proper escaping
            lines.append("@article{" + key + ",")
            lines.append(f"  title={{{{{title}}}}},")
            lines.append(f"  author={{{{{authors}}}}},")
            lines.append("  journal={{arXiv preprint arXiv:" + arxiv_id + "}},")
            lines.append(f"  year={{{{{year}}}}},")
            lines.append("}")
            lines.append("")

        return "\n".join(lines)

    def export_json(
        self,
        results: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Export structured JSON with all metadata.

        Args:
            results: Search results
            metadata: Additional metadata

        Returns:
            JSON string
        """
        import json

        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "count": len(results),
            "results": [],
        }

        for result in results:
            paper = result.get("meta", {})
            data["results"].append({
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "authors": paper.get("authors"),
                "summary": paper.get("summary"),
                "published": paper.get("published"),
                "link": paper.get("link"),
                "tags": paper.get("tags", []),
                "score": result.get("score"),
            })

        return json.dumps(data, indent=2)

    def export_csv(
        self,
        results: List[Dict[str, Any]],
    ) -> str:
        """
        Export CSV with columns: title, authors, year, link, score.

        Args:
            results: Search results

        Returns:
            CSV string
        """
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["title", "authors", "year", "link", "score"],
        )
        writer.writeheader()

        for result in results:
            paper = result.get("meta", {})
            writer.writerow({
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "year": paper.get("published", "")[:4],
                "link": paper.get("link", ""),
                "score": result.get("score", 0),
            })

        return output.getvalue()

    def _format_citation(
        self,
        paper: Dict[str, Any],
        style: str,
    ) -> str:
        """Format a single paper citation."""
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", "Unknown")
        year = paper.get("published", "")[:4]
        link = paper.get("link", "")

        if style == CitationStyle.APA:
            return f"{authors} ({year}). {title}. arXiv."

        elif style == CitationStyle.MLA:
            return f'{authors}. "{title}." {year}, arXiv.'

        elif style == CitationStyle.CHICAGO:
            return f"{authors}. {year}. \"{title}.\" arXiv."

        elif style == CitationStyle.IEEE:
            return f"{authors}, \"{title},\" arXiv, {year}."

        else:  # Default to APA
            return f"{authors} ({year}). {title}."

    def _generate_html_export(
        self,
        results: List[Dict[str, Any]],
        query: str,
        answer: str,
        citation_style: str,
        include_abstracts: bool,
    ) -> str:
        """Generate HTML for PDF export."""
        lines = []

        # HTML header
        lines.append("<!DOCTYPE html>")
        lines.append("<html>")
        lines.append("<head>")
        lines.append("<meta charset='UTF-8'>")
        lines.append("<title>ArXiv Search Results</title>")
        lines.append("<style>")
        lines.append("body { font-family: Arial, sans-serif; margin: 40px; }")
        lines.append("h1 { color: #333; }")
        lines.append("h2 { color: #555; margin-top: 30px; }")
        lines.append(".citation { background: #f5f5f5; padding: 10px; margin: 10px 0; }")
        lines.append("</style>")
        lines.append("</head>")
        lines.append("<body>")

        # Title
        lines.append(f"<h1>Search Results: {query}</h1>")

        # Answer
        lines.append("<h2>Answer</h2>")
        lines.append(f"<p>{answer}</p>")

        # References
        lines.append("<h2>References</h2>")

        for i, result in enumerate(results, 1):
            paper = result.get("meta", {})
            title = paper.get("title", "Untitled")

            lines.append(f"<h3>{i}. {title}</h3>")

            # Citation
            citation = self._format_citation(paper, citation_style)
            lines.append(f"<div class='citation'>{citation}</div>")

            if include_abstracts:
                abstract = paper.get("summary", "")
                if abstract:
                    lines.append(f"<p><strong>Abstract:</strong> {abstract[:500]}...</p>")

            link = paper.get("link", "")
            if link:
                lines.append(f"<p><a href='{link}'>View on ArXiv</a></p>")

            lines.append("<hr>")

        # HTML footer
        lines.append("</body>")
        lines.append("</html>")

        return "\n".join(lines)
