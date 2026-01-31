"""LangChain document loader for ArXiv papers."""


# Copyright 2025 ArXivFuturaSearch Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

from app.arxiv_loader import fetch_arxiv
from app.chunking import chunk_text, chunk_text_semantic
from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class ArXivLoader(BaseLoader):
    """
    LangChain document loader for ArXiv papers.

    Fetches papers from ArXiv API and optionally chunks them into smaller documents.
    """

    def __init__(
        self,
        search_query: str,
        max_results: int = 25,
        start: int = 0,
        load_all_available_meta: bool = True,
        chunk_documents: bool = True,
        use_semantic_chunking: bool = None,
    ):
        """
        Initialize the ArXiv loader.

        Args:
            search_query: ArXiv search query (e.g., "cat:cs.AI", "quantum computing")
            max_results: Maximum number of papers to fetch
            start: Starting index for pagination
            load_all_available_meta: Whether to load all metadata fields
            chunk_documents: Whether to chunk documents into smaller pieces
            use_semantic_chunking: Whether to use semantic chunking (defaults to config setting)
        """
        self.search_query = search_query
        self.max_results = max_results
        self.start = start
        self.load_all_available_meta = load_all_available_meta
        self.chunk_documents = chunk_documents
        self.use_semantic_chunking = (
            use_semantic_chunking
            if use_semantic_chunking is not None
            else settings.USE_SEMANTIC_CHUNKING
        )

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy load documents from ArXiv (yields documents one at a time).

        Yields:
            LangChain Document objects
        """
        # Fetch papers from ArXiv
        papers = fetch_arxiv(
            search_query=self.search_query,
            max_results=self.max_results,
            start=self.start,
        )

        logger.info(
            "Loading ArXiv papers",
            count=len(papers),
            chunk_documents=self.chunk_documents,
        )

        for paper in papers:
            # Combine title and summary for the content
            content = f"Title: {paper['title']}\n\nAbstract: {paper['summary']}"

            # Build metadata
            metadata = {
                "source": "arxiv",
                "arxiv_id": paper.get("id", ""),
                "title": paper.get("title", ""),
                "link": paper.get("link", ""),
                "published": paper.get("published", ""),
                "updated": paper.get("updated", ""),
            }

            if self.load_all_available_meta:
                metadata.update({
                    "authors": paper.get("authors", []),
                    "tags": paper.get("tags", []),
                })

            # Optionally chunk the document
            if self.chunk_documents:
                chunks = self._chunk_content(content, paper)

                for i, chunk_text in enumerate(chunks):
                    # Create a unique chunk ID
                    chunk_id = f"{paper.get('id', 'unknown')}_{i}"

                    # Create chunk metadata
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_id"] = chunk_id
                    chunk_metadata["chunk_index"] = i

                    yield Document(page_content=chunk_text, metadata=chunk_metadata)
            else:
                # Return full document without chunking
                yield Document(page_content=content, metadata=metadata)

    def load(self) -> List[Document]:
        """
        Load all documents from ArXiv at once.

        Returns:
            List of LangChain Document objects
        """
        return list(self.lazy_load())

    def _chunk_content(self, content: str, paper: dict) -> List[str]:
        """
        Chunk document content using configured chunking strategy.

        Args:
            content: Document content to chunk
            paper: Paper metadata dictionary

        Returns:
            List of text chunks
        """
        if self.use_semantic_chunking:
            # Use semantic chunking
            from app.embeddings import get_embedder

            embedder = get_embedder()
            chunks = chunk_text_semantic(
                text=content,
                embedder=embedder,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                similarity_threshold=settings.SEMANTIC_SIMILARITY_THRESHOLD,
            )
            logger.debug(
                "Semantic chunking complete",
                paper_id=paper.get("id", ""),
                chunks=len(chunks),
            )
        else:
            # Use sentence-aware chunking
            chunks = chunk_text(
                text=content,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            logger.debug(
                "Sentence chunking complete",
                paper_id=paper.get("id", ""),
                chunks=len(chunks),
            )

        return chunks


class ArXivPaperLoader(BaseLoader):
    """
    LangChain document loader for loading a specific ArXiv paper by ID.

    This is a convenience loader for loading a single paper when you know its ID.
    """

    def __init__(
        self,
        arxiv_id: str,
        chunk_documents: bool = True,
        use_semantic_chunking: bool = None,
    ):
        """
        Initialize the ArXiv paper loader.

        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.00001" or "cs.AI/0001")
            chunk_documents: Whether to chunk the document
            use_semantic_chunking: Whether to use semantic chunking
        """
        # Convert arxiv_id to search query
        self.search_query = f"id:{arxiv_id}"
        self.loader = ArXivLoader(
            search_query=self.search_query,
            max_results=1,
            chunk_documents=chunk_documents,
            use_semantic_chunking=use_semantic_chunking,
        )

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load the paper."""
        return self.loader.lazy_load()

    def load(self) -> List[Document]:
        """Load the paper."""
        return self.loader.load()
