"""Background tasks for periodic index updates from ArXiv feeds.


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

Automatically fetches new papers from ArXiv RSS feeds and updates
the search index.
"""

import asyncio
import feedparser
import re
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

import httpx

from app.logging_config import get_logger
from app.config import settings

logger = get_logger(__name__)


# =============================================================================
# ARXIV FEED PARSING
# =============================================================================

ARXIV_BASE_URL = "http://export.arxiv.org/api/query?"
ARXIV_RSS_URL = "http://export.arxiv.org/rss/"


@dataclass
class ArXivPaper:
    """Data class for ArXiv paper metadata."""

    id: str
    title: str
    authors: List[str]
    summary: str
    published: str
    link: str
    categories: List[str]
    updated: Optional[str] = None
    primary_category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "summary": self.summary,
            "published": self.published,
            "link": self.link,
            "categories": self.categories,
            "updated": self.updated,
            "primary_category": self.primary_category,
            "tags": self.categories,
        }


class ArXivFeedParser:
    """Parser for ArXiv RSS/API feeds."""

    def __init__(self, timeout: float = 30.0):
        """
        Initialize feed parser.

        Args:
            timeout: HTTP request timeout
        """
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def parse_paper(self, entry: Dict[str, Any]) -> Optional[ArXivPaper]:
        """
        Parse a single ArXiv paper from feed entry.

        Args:
            entry: Feedparser entry dict

        Returns:
            ArXivPaper or None if parsing fails
        """
        try:
            # Extract ID from URL
            link = entry.get("link", "")
            arxiv_id_match = re.search(r'arxiv.org/abs/(\S+)', link)
            if not arxiv_id_match:
                return None
            arxiv_id = arxiv_id_match.group(1)

            # Extract authors
            authors = []
            author_list = entry.get("authors", [])
            if isinstance(author_list, list):
                for author in author_list:
                    if isinstance(author, dict):
                        name = author.get("name", "")
                        if name:
                            authors.append(name)
                    elif isinstance(author, str):
                        authors.append(author)
            elif isinstance(author_list, str):
                authors = [author_list]

            # Extract categories/tags
            tags = entry.get("tags", [])
            categories = []
            for tag in tags:
                if isinstance(tag, dict):
                    term = tag.get("term", "")
                    if term:
                        categories.append(term)
                elif isinstance(tag, str):
                    categories.append(tag)

            # Get primary category
            primary_category = entry.get("arxiv_primary_category", {}).get("term")
            if primary_category:
                categories.insert(0, primary_category)

            # Parse dates
            published = entry.get("published", "")
            updated = entry.get("updated", "")

            # Clean up title (remove newlines)
            title = entry.get("title", "").replace("\n", " ").strip()

            return ArXivPaper(
                id=arxiv_id,
                title=title,
                authors=authors,
                summary=entry.get("summary", "").strip(),
                published=published,
                link=link,
                categories=list(set(categories)),
                updated=updated if updated else None,
                primary_category=primary_category,
            )

        except Exception as e:
            logger.warning("Failed to parse ArXiv paper", error=str(e))
            return None

    async def fetch_by_category(
        self,
        category: str,
        max_results: int = 100,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> List[ArXivPaper]:
        """
        Fetch papers from ArXiv by category.

        Args:
            category: ArXiv category (e.g., "cs.AI", "cs.LG")
            max_results: Maximum number of papers to fetch
            sort_by: Sort field
            sort_order: Sort direction

        Returns:
            List of ArXivPaper objects
        """
        client = await self._get_client()

        # Build query URL
        params = {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        try:
            logger.info(
                "Fetching ArXiv papers",
                category=category,
                max_results=max_results,
            )

            response = await client.get(ARXIV_BASE_URL, params=params)
            response.raise_for_status()

            # Parse XML response
            feed = feedparser.parse(response.content)

            papers = []
            for entry in feed.entries:
                paper = self.parse_paper(entry)
                if paper:
                    papers.append(paper)

            logger.info(
                "ArXiv papers fetched",
                category=category,
                papers=len(papers),
            )

            return papers

        except httpx.HTTPError as e:
            logger.error("HTTP error fetching ArXiv papers", error=str(e))
            return []
        except Exception as e:
            logger.error("Error fetching ArXiv papers", error=str(e))
            return []

    async def fetch_recent(
        self,
        categories: Optional[List[str]] = None,
        days_back: int = 7,
        max_results: int = 500,
    ) -> List[ArXivPaper]:
        """
        Fetch recent papers from multiple categories.

        Args:
            categories: List of ArXiv categories (default: common CS categories)
            days_back: Number of days to look back
            max_results: Maximum results per category

        Returns:
            List of ArXivPaper objects
        """
        # Default CS categories
        if categories is None:
            categories = [
                "cs.AI",  # Artificial Intelligence
                "cs.CL",  # Computation and Language
                "cs.CV",  # Computer Vision
                "cs.LG",  # Machine Learning
                "cs.NE",  # Neural and Evolutionary Computing
                "stat.ML",  # Machine Learning (Statistics)
            ]

        # Calculate date cutoff
        cutoff_date = datetime.now(UTC) - timedelta(days=days_back)

        # Fetch from all categories
        tasks = [
            self.fetch_by_category(cat, max_results=max_results)
            for cat in categories
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter results
        all_papers = []
        seen_ids = set()

        for result in results:
            if isinstance(result, Exception):
                logger.warning("Category fetch failed", error=str(result))
                continue

            for paper in result:
                if paper.id not in seen_ids:
                    # Filter by date if available
                    if paper.published:
                        try:
                            pub_date = datetime.fromisoformat(paper.published.replace("Z", "+00:00"))
                            if pub_date >= cutoff_date:
                                all_papers.append(paper)
                                seen_ids.add(paper.id)
                        except ValueError:
                            # If date parsing fails, include the paper
                            all_papers.append(paper)
                            seen_ids.add(paper.id)
                    else:
                        all_papers.append(paper)
                        seen_ids.add(paper.id)

        logger.info(
            "Recent ArXiv papers fetched",
            total_papers=len(all_papers),
            days_back=days_back,
            categories=categories,
        )

        return all_papers


# =============================================================================
# INDEX UPDATE TASK
# =============================================================================

class IndexUpdateTask:
    """
    Background task for updating the search index with new ArXiv papers.
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        update_interval_hours: int = 24,
        papers_per_update: int = 500,
        index_callback: Optional[Callable] = None,
    ):
        """
        Initialize index update task.

        Args:
            categories: ArXiv categories to monitor
            update_interval_hours: Hours between updates
            papers_per_update: Maximum papers to fetch per update
            index_callback: Async callback to index new papers
        """
        self.categories = categories
        self.update_interval = timedelta(hours=update_interval_hours)
        self.papers_per_update = papers_per_update
        self.index_callback = index_callback

        self.parser = ArXivFeedParser()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_update: Optional[datetime] = None

    async def update_index(self) -> Dict[str, Any]:
        """
        Fetch new papers and update the index.

        Returns:
            Update statistics
        """
        logger.info("Starting index update")

        # Fetch recent papers
        papers = await self.parser.fetch_recent(
            categories=self.categories,
            days_back=int(self.update_interval.total_seconds() / 3600),
            max_results=self.papers_per_update,
        )

        stats = {
            "papers_fetched": len(papers),
            "papers_indexed": 0,
            "errors": 0,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if not papers:
            logger.info("No new papers to index")
            return stats

        # Index papers if callback provided
        if self.index_callback:
            try:
                papers_dicts = [p.to_dict() for p in papers]
                await self.index_callback(papers_dicts)
                stats["papers_indexed"] = len(papers)
                logger.info("Papers indexed successfully", count=len(papers))
            except Exception as e:
                logger.error("Index callback failed", error=str(e))
                stats["errors"] += 1
        else:
            logger.warning("No index callback provided, papers not indexed")

        self._last_update = datetime.now(UTC)
        return stats

    async def start(self) -> None:
        """Start the background update task."""
        if self._running:
            logger.warning("Update task already running")
            return

        self._running = True
        logger.info(
            "Starting index update task",
            interval_hours=self.update_interval.total_seconds() / 3600,
            categories=self.categories,
        )

        # Run initial update
        await self.update_index()

        # Start periodic updates
        async def _periodic_update():
            while self._running:
                await asyncio.sleep(self.update_interval.total_seconds())
                if self._running:
                    await self.update_index()

        self._task = asyncio.create_task(_periodic_update())

    async def stop(self) -> None:
        """Stop the background update task."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        await self.parser.close()
        logger.info("Index update task stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current task status."""
        return {
            "running": self._running,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "update_interval_hours": self.update_interval.total_seconds() / 3600,
            "categories": self.categories,
        }


# =============================================================================
# TASK MANAGER
# =============================================================================

class BackgroundTaskManager:
    """
    Manager for multiple background tasks.
    """

    def __init__(self):
        """Initialize task manager."""
        self._tasks: Dict[str, IndexUpdateTask] = {}
        self._running = False

    def register_task(
        self,
        name: str,
        task: IndexUpdateTask,
    ) -> None:
        """
        Register a background task.

        Args:
            name: Task name
            task: IndexUpdateTask instance
        """
        self._tasks[name] = task
        logger.info("Task registered", name=name)

    async def start_all(self) -> None:
        """Start all registered tasks."""
        if self._running:
            return

        self._running = True
        logger.info("Starting all background tasks", count=len(self._tasks))

        for name, task in self._tasks.items():
            try:
                await task.start()
            except Exception as e:
                logger.error("Task start failed", name=name, error=str(e))

    async def stop_all(self) -> None:
        """Stop all running tasks."""
        self._running = False
        logger.info("Stopping all background tasks")

        tasks = list(self._tasks.values())
        await asyncio.gather(
            *[task.stop() for task in tasks],
            return_exceptions=True,
        )

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tasks."""
        return {
            name: task.get_status()
            for name, task in self._tasks.items()
        }


# =============================================================================
# GLOBAL TASK MANAGER
# =============================================================================

_task_manager: Optional[BackgroundTaskManager] = None


def get_task_manager() -> BackgroundTaskManager:
    """Get or create global task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager


def setup_index_update_task(
    index_callback: Callable,
    categories: Optional[List[str]] = None,
    update_interval_hours: int = 24,
) -> IndexUpdateTask:
    """
    Setup and register an index update task.

    Args:
        index_callback: Async callback to index papers
        categories: ArXiv categories to monitor
        update_interval_hours: Hours between updates

    Returns:
        IndexUpdateTask instance
    """
    task = IndexUpdateTask(
        categories=categories,
        update_interval_hours=update_interval_hours,
        index_callback=index_callback,
    )

    manager = get_task_manager()
    manager.register_task("arxiv_index_update", task)

    return task
