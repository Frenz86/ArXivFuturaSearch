"""
Multi-modal search for images and equations from research papers.

Supports extracting and indexing images from PDFs, parsing LaTeX equations,
and combining visual search with text search.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import hashlib
import re

from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ImageMetadata:
    """Metadata for extracted images."""
    paper_id: str
    image_id: str
    page_number: int
    figure_number: Optional[str] = None
    caption: str = ""
    embedding: Optional[List[float]] = None
    image_path: str = ""
    extracted_at: str = ""


@dataclass
class EquationMetadata:
    """Metadata for LaTeX equations."""
    equation_id: str
    paper_id: str
    latex: str
    mathml: Optional[str] = None
    position: int
    equation_type: str = "display"  # inline, display, numbered
    variables: List[str] = None
    embedding: Optional[List[float]] = None


class MultiModalSearchResult:
    """Combined multi-modal search results."""

    def __init__(self):
        self.text_results: List[Dict[str, Any]] = []
        self.image_results: List[Dict[str, Any]] = []
        self.equation_results: List[Dict[str, Any]] = []

    @property
    def total_count(self) -> int:
        return len(self.text_results) + len(self.image_results) + len(self.equation_results)


class ImageExtractor:
    """
    Extract images from PDF papers.

    Uses pdf2image or PyMuPDF to extract images from PDF pages.
    """

    def __init__(self, output_dir: str = "data/images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def extract_images_from_pdf(
        self,
        pdf_path: str,
        paper_id: str,
    ) -> List[ImageMetadata]:
        """
        Extract images from a PDF file.

        Args:
            pdf_path: Path to PDF file
            paper_id: ArXiv paper ID

        Returns:
            List of ImageMetadata
        """
        images = []

        try:
            # Try using PyMuPDF (fitz)
            import fitz

            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract images
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    xref = img[0]

                    # Extract image
                    base_image = doc.extract_image(xref)

                    # Save image
                    image_filename = f"{paper_id}_page{page_num}_img{img_index}.png"
                    image_path = self.output_dir / image_filename

                    base_image.save(str(image_path))

                    # Create metadata
                    image_id = hashlib.md5(f"{paper_id}_{page_num}_{img_index}".encode()).hexdigest()

                    metadata = ImageMetadata(
                        paper_id=paper_id,
                        image_id=image_id,
                        page_number=page_num,
                        image_path=str(image_path),
                        extracted_at=datetime.utcnow().isoformat(),
                    )

                    images.append(metadata)

            logger.info(
                "Images extracted from PDF",
                paper_id=paper_id,
                count=len(images),
            )

        except ImportError:
            logger.warning("PyMuPDF not available, trying pdf2image")

            # Fallback to pdf2image
            from pdf2image import convert_from_path

            # Convert PDF to images
            pil_images = convert_from_path(pdf_path)

            for page_num, pil_img in enumerate(pil_images):
                image_filename = f"{paper_id}_page{page_num}.png"
                image_path = self.output_dir / image_filename

                pil_img.save(str(image_path))

                image_id = hashlib.md5(f"{paper_id}_{page_num}".encode()).hexdigest()

                metadata = ImageMetadata(
                    paper_id=paper_id,
                    image_id=image_id,
                    page_number=page_num,
                    image_path=str(image_path),
                    extracted_at=datetime.utcnow().isoformat(),
                )

                images.append(metadata)

        except Exception as e:
            logger.error("Failed to extract images from PDF", paper_id=paper_id, error=str(e))

        return images

    async def extract_image_captions(
        self,
        paper_metadata: Dict[str, Any],
    ) -> List[ImageMetadata]:
        """
        Extract image captions from paper metadata.

        Args:
            paper_metadata: Paper metadata with figures info

        Returns:
            Updated ImageMetadata list with captions
        """
        # Implementation would parse paper metadata for figure captions
        # This is a placeholder for the actual implementation
        return []


class EquationParser:
    """
    Parse LaTeX equations from papers.

    Extracts LaTeX equations and converts them to MathML for processing.
    """

    # LaTeX equation patterns
    EQUATION_PATTERNS = [
        r'\$\$([^$]+)\$\$',  # Display math $$...$$
        r'\$([^$]+)\$',      # Inline math $...$
        r'\\begin\{equation\}(.*?)\\end\{equation\}',  # Equation environment
        r'\\begin\{align\}(.*?)\\end\{align\}',            # Align environment
        r'\\begin\{align\*\}(.*?)\\end\{align\*\}',    # Align* environment
    ]

    def __init__(self):
        import re
        self.compiled_patterns = [
            re.compile(pattern, re.DOTALL)
            for pattern in self.EQUATION_PATTERNS
        ]

    async def parse_equations(
        self,
        text: str,
        paper_id: str,
    ) -> List[EquationMetadata]:
        """
        Parse LaTeX equations from text.

        Args:
            text: Paper text containing LaTeX
            paper_id: ArXiv paper ID

        Returns:
            List of EquationMetadata
        """
        equations = []
        equation_count = 0

        for pattern in self.compiled_patterns:
            matches = pattern.finditer(text)

            for match in matches:
                equation_latex = match.group(1).strip()

                # Determine equation type
                equation_type = "display"
                if "$$" not in match.group(0):
                    equation_type = "inline"

                # Extract variables
                variables = self._extract_variables(equation_latex)

                equation_id = hashlib.md5(
                    f"{paper_id}_{equation_count}_{equation_latex[:50]}".encode()
                ).hexdigest()

                metadata = EquationMetadata(
                    equation_id=equation_id,
                    paper_id=paper_id,
                    latex=equation_latex,
                    position=match.start(),
                    equation_type=equation_type,
                    variables=variables,
                )

                equations.append(metadata)
                equation_count += 1

        logger.info(
            "Equations parsed",
            paper_id=paper_id,
            count=len(equations),
        )

        return equations

    def _extract_variables(self, latex: str) -> List[str]:
        """
        Extract variable names from LaTeX equation.

        Args:
            latex: LaTeX equation string

        Returns:
            List of variable names
        """
        # Simple heuristic: find standalone Greek letters and common variable names
        import re

        # Greek letters
        greek_letters = r'\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)'

        # Standalone math symbols (simple heuristic)
        pattern = r'\\?([a-zA-Z])(?:_\d+)?'

        variables = []

        # Extract Greek letters
        greek_matches = re.findall(greek_letters, latex)
        variables.extend(greek_matches)

        # Extract single letters with subscripts/superscripts
        matches = re.findall(pattern, latex)
        variables.extend([m[0] for m in matches if len(m[0]) == 1])

        return list(set(variables))


class MultiModalSearchEngine:
    """
    Multi-modal search combining text, images, and equations.

    Performs combined search across all modalities with configurable weights.
    """

    def __init__(
        self,
        text_weight: float = 0.6,
        image_weight: float = 0.25,
        equation_weight: float = 0.15,
    ):
        """
        Initialize multi-modal search engine.

        Args:
            text_weight: Weight for text search results
            image_weight: Weight for image search results
            equation_weight: Weight for equation search results
        """
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.equation_weight = equation_weight

        logger.info(
            "MultiModalSearchEngine initialized",
            weights={"text": text_weight, "images": image_weight, "equations": equation_weight},
        )

    async def search(
        self,
        query: str,
        query_image: Optional[str] = None,
        top_k: int = 5,
        include_images: bool = True,
        include_equations: bool = True,
    ) -> MultiModalSearchResult:
        """
        Combined search across all modalities.

        Args:
            query: Text query
            query_image: Optional image upload for visual search
            top_k: Number of results per modality
            include_images: Include image search
            include_equations: Include equation search

        Returns:
            MultiModalSearchResult with combined results
        """
        result = MultiModalSearchResult()

        # Text search (would use existing vector store)
        # result.text_results = await self._search_text(query, top_k)

        # Image search (if enabled and query_image provided)
        if include_images and query_image:
            result.image_results = await self._search_images(query_image, top_k)

        # Equation search
        if include_equations:
            result.equation_results = await self._search_equations(query, top_k)

        # Re-rank combined results
        # result = await self._rerank_combined(result)

        return result

    async def _search_images(
        self,
        query_image: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search by image similarity."""
        # Placeholder for CLIP-based image search
        return []

    async def _search_equations(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search by equation similarity."""
        # Placeholder for equation search
        return []

    async def _search_text(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search by text similarity."""
        # Would use existing vector store
        return []

    async def _rerank_combined(
        self,
        result: MultiModalSearchResult,
    ) -> MultiModalSearchResult:
        """Re-rank combined results using RRF."""
        # Implement Reciprocal Rank Fusion for combining results
        return result
