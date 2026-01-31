"""Tests for background tasks module.

Run with: pytest tests/test_background_tasks.py -v
"""

import pytest
import feedparser
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_arxiv_feed():
    """Mock ArXiv RSS feed response."""
    return """
    <?xml version="1.0" encoding="UTF-8"?>
    <rss xmlns:atom="http://www.w3.org/2005/Atom" version="2.0">
        <channel>
            <title>arXiv.org: cs.AI</title>
            <link>https://arxiv.org/list/cs.AI/recent</link>
            <description>cs.AI updates on the arXiv.org e-print archive</description>
            <atom:link href="https://arxiv.org/rss/cs.AI" rel="self" type="application/rss+xml"/>
            <item>
                <title>Test Paper 1: Machine Learning Advances</title>
                <link>https://arxiv.org/abs/2401.00001</link>
                <description>Abstract: This is a test abstract about machine learning.</description>
                <author>John Doe, Jane Smith</author>
                <category>cs.AI</category>
                <pubDate>Mon, 15 Jan 2024 00:00:00 GMT</pubDate>
            </item>
            <item>
                <title>Test Paper 2: Deep Neural Networks</title>
                <link>https://arxiv.org/abs/2401.00002</link>
                <description>Abstract: This paper discusses deep learning architectures.</description>
                <author>Alice Johnson</author>
                <category>cs.LG</category>
                <pubDate>Tue, 16 Jan 2024 00:00:00 GMT</pubDate>
            </item>
        </channel>
    </rss>
    """


@pytest.fixture
def sample_papers():
    """Sample parsed papers."""
    return [
        {
            "id": "2401.00001",
            "title": "Test Paper 1: Machine Learning Advances",
            "summary": "This is a test abstract about machine learning.",
            "authors": ["John Doe", "Jane Smith"],
            "published": "2024-01-15T00:00:00Z",
            "link": "https://arxiv.org/abs/2401.00001",
            "tags": ["cs.AI"],
        },
        {
            "id": "2401.00002",
            "title": "Test Paper 2: Deep Neural Networks",
            "summary": "This paper discusses deep learning architectures.",
            "authors": ["Alice Johnson"],
            "published": "2024-01-16T00:00:00Z",
            "link": "https://arxiv.org/abs/2401.00002",
            "tags": ["cs.LG"],
        },
    ]


# =============================================================================
# ARXIV FEED PARSER TESTS
# =============================================================================

class TestArXivFeedParser:
    """Tests for ArXivFeedParser class."""

    def test_initialization(self):
        """Test feed parser initialization."""
        from app.background_tasks import ArXivFeedParser

        parser = ArXivFeedParser(
            categories=["cs.AI", "cs.LG"],
            max_papers=10
        )

        assert parser.categories == ["cs.AI", "cs.LG"]
        assert parser.max_papers == 10

    def test_parse_feed(self, mock_arxiv_feed):
        """Test parsing ArXiv RSS feed."""
        from app.background_tasks import ArXivFeedParser

        parser = ArXivFeedParser(categories=["cs.AI"])

        with patch('feedparser.parse', return_value=feedparser.parse(mock_arxiv_feed)):
            papers = parser.parse_feed("https://arxiv.org/rss/cs.AI")

            assert len(papers) == 2
            assert papers[0]["title"] == "Test Paper 1: Machine Learning Advances"
            assert papers[0]["id"] == "2401.00001"
            assert "John Doe" in papers[0]["authors"]

    def test_parse_feed_empty(self):
        """Test parsing empty feed."""
        from app.background_tasks import ArXivFeedParser

        parser = ArXivFeedParser(categories=["cs.AI"])

        empty_feed = """
        <?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Empty Feed</title>
            </channel>
        </rss>
        """

        with patch('feedparser.parse', return_value=feedparser.parse(empty_feed)):
            papers = parser.parse_feed("https://arxiv.org/rss/cs.AI")

            assert papers == []

    def test_parse_multiple_categories(self, mock_arxiv_feed):
        """Test parsing multiple categories."""
        from app.background_tasks import ArXivFeedParser

        parser = ArXivFeedParser(
            categories=["cs.AI", "cs.LG", "cs.CV"],
            max_papers=5
        )

        with patch('feedparser.parse', return_value=feedparser.parse(mock_arxiv_feed)):
            all_papers = []
            for category in parser.categories:
                papers = parser.parse_feed(f"https://arxiv.org/rss/{category}")
                all_papers.extend(papers)

            assert len(all_papers) > 0

    def test_extract_arxiv_id(self):
        """Test ArXiv ID extraction from URL."""
        from app.background_tasks import ArXivFeedParser

        parser = ArXivFeedParser()

        # Test various ArXiv URL formats
        urls = [
            "https://arxiv.org/abs/2401.00001",
            "https://arxiv.org/abs/cs.AI/2401.00001v2",
            "https://arxiv.org/pdf/2401.00001.pdf",
        ]

        for url in urls:
            paper_id = parser._extract_id(url)
            assert "2401.00001" in paper_id

    def test_filter_recent_papers(self, sample_papers):
        """Test filtering papers by date."""
        from app.background_tasks import ArXivFeedParser

        parser = ArXivFeedParser()

        # Get papers from last 7 days
        cutoff = datetime.utcnow() - timedelta(days=7)
        recent = parser.filter_recent(sample_papers, days=7)

        # All sample papers are recent (2024-01-15, 2024-01-16)
        assert len(recent) == len(sample_papers)


# =============================================================================
# INDEX UPDATE TASK TESTS
# =============================================================================

class TestIndexUpdateTask:
    """Tests for IndexUpdateTask class."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test task initialization."""
        from app.background_tasks import IndexUpdateTask

        task = IndexUpdateTask(
            name="test_task",
            interval_seconds=3600,
            categories=["cs.AI"]
        )

        assert task.name == "test_task"
        assert task.interval_seconds == 3600
        assert task._running is False

    @pytest.mark.asyncio
    async def test_execute_with_store(self, sample_papers):
        """Test task execution with vector store."""
        from app.background_tasks import IndexUpdateTask

        task = IndexUpdateTask(
            name="test_task",
            categories=["cs.AI"]
        )

        mock_store = Mock()
        mock_store.add_documents = AsyncMock()
        mock_store.count = Mock(return_value=100)

        # Mock feed parsing
        with patch.object(task, '_fetch_new_papers', return_value=sample_papers):
            with patch('app.background_tasks.get_vectorstore', return_value=mock_store):
                result = await task.execute()

                assert result["papers_processed"] == len(sample_papers)

    @pytest.mark.asyncio
    async def test_execute_no_new_papers(self):
        """Test task execution when no new papers."""
        from app.background_tasks import IndexUpdateTask

        task = IndexUpdateTask(name="test_task", categories=["cs.AI"])

        # Mock no new papers
        with patch.object(task, '_fetch_new_papers', return_value=[]):
            result = await task.execute()

            assert result["papers_processed"] == 0
            assert result["status"] == "no_new_papers"

    @pytest.mark.asyncio
    async def test_execute_with_error_handling(self, sample_papers):
        """Test task execution with error handling."""
        from app.background_tasks import IndexUpdateTask

        task = IndexUpdateTask(name="test_task", categories=["cs.AI"])

        # Mock error during paper processing
        with patch.object(task, '_fetch_new_papers', return_value=sample_papers):
            with patch('app.background_tasks.get_vectorstore', side_effect=Exception("Store error")):
                result = await task.execute()

                assert "error" in result

    @pytest.mark.asyncio
    async def test_start_stop_task(self):
        """Test starting and stopping a task."""
        from app.background_tasks import IndexUpdateTask

        task = IndexUpdateTask(
            name="test_task",
            interval_seconds=1,
            categories=["cs.AI"]
        )

        # Mock execute to avoid infinite loop
        with patch.object(task, 'execute', return_value={"status": "ok"}):
            # Start task (in background)
            import asyncio
            start_task = asyncio.create_task(task.start())

            # Give it time to start
            await asyncio.sleep(0.1)

            assert task._running is True

            # Stop task
            await task.stop()

            # Wait for task to complete
            try:
                await asyncio.wait_for(start_task, timeout=2)
            except asyncio.TimeoutError:
                task._task.cancel()

            assert task._running is False

    @pytest.mark.asyncio
    async def test_task_should_run(self):
        """Test task scheduling logic."""
        from app.background_tasks import IndexUpdateTask

        task = IndexUpdateTask(
            name="test_task",
            interval_seconds=60,
            categories=["cs.AI"]
        )

        # Initially should run
        assert task.should_run() is True

        # After running, should not run immediately
        task._last_run = datetime.utcnow()
        assert task.should_run() is False

        # After interval, should run again
        task._last_run = datetime.utcnow() - timedelta(seconds=61)
        assert task.should_run() is True


# =============================================================================
# BACKGROUND TASK MANAGER TESTS
# =============================================================================

class TestBackgroundTaskManager:
    """Tests for BackgroundTaskManager class."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test task manager initialization."""
        from app.background_tasks import BackgroundTaskManager

        manager = BackgroundTaskManager()

        assert manager._running is False
        assert len(manager._tasks) == 0

    @pytest.mark.asyncio
    async def test_register_task(self):
        """Test registering a task."""
        from app.background_tasks import (
            BackgroundTaskManager,
            IndexUpdateTask
        )

        manager = BackgroundTaskManager()
        task = IndexUpdateTask(name="test", categories=["cs.AI"])

        manager.register_task(task)

        assert len(manager._tasks) == 1
        assert "test" in manager._tasks

    @pytest.mark.asyncio
    async def test_start_all(self):
        """Test starting all tasks."""
        from app.background_tasks import (
            BackgroundTaskManager,
            IndexUpdateTask
        )

        manager = BackgroundTaskManager()

        task1 = IndexUpdateTask(name="task1", categories=["cs.AI"])
        task2 = IndexUpdateTask(name="task2", categories=["cs.LG"])

        manager.register_task(task1)
        manager.register_task(task2)

        # Mock execute to avoid infinite loops
        with patch.object(IndexUpdateTask, 'execute', return_value={"status": "ok"}):
            with patch.object(IndexUpdateTask, 'start', return_value=None):
                await manager.start_all()

                assert manager._running is True

    @pytest.mark.asyncio
    async def test_stop_all(self):
        """Test stopping all tasks."""
        from app.background_tasks import (
            BackgroundTaskManager,
            IndexUpdateTask
        )

        manager = BackgroundTaskManager()
        task = IndexUpdateTask(name="test", categories=["cs.AI"])

        manager.register_task(task)

        with patch.object(task, 'stop', return_value=None):
            await manager.stop_all()

            assert manager._running is False

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test getting manager status."""
        from app.background_tasks import (
            BackgroundTaskManager,
            IndexUpdateTask
        )

        manager = BackgroundTaskManager()
        task = IndexUpdateTask(name="test", categories=["cs.AI"])

        manager.register_task(task)

        status = manager.get_status()

        assert isinstance(status, dict)
        assert "test" in status
        assert status["test"]["name"] == "test"
        assert status["test"]["running"] is False


# =============================================================================
# GLOBAL INSTANCE TESTS
# =============================================================================

class TestGlobalBackgroundTasks:
    """Tests for global background task instances."""

    def test_get_task_manager_singleton(self):
        """Test that get_task_manager returns singleton."""
        from app.background_tasks import get_task_manager

        manager1 = get_task_manager()
        manager2 = get_task_manager()

        assert manager1 is manager2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestBackgroundTasksIntegration:
    """Integration tests for background tasks."""

    @pytest.mark.asyncio
    async def test_full_update_workflow(self, sample_papers):
        """Test complete update workflow."""
        from app.background_tasks import (
            BackgroundTaskManager,
            IndexUpdateTask,
            ArXivFeedParser
        )

        manager = BackgroundTaskManager()
        parser = ArXivFeedParser(categories=["cs.AI"])

        task = IndexUpdateTask(
            name="arxiv_update",
            categories=["cs.AI"],
            parser=parser
        )

        manager.register_task(task)

        # Mock feed parsing and vector store
        with patch.object(parser, 'parse_feed', return_value=sample_papers):
            mock_store = Mock()
            mock_store.add_documents = AsyncMock()

            with patch('app.background_tasks.get_vectorstore', return_value=mock_store):
                result = await task.execute()

                assert result["papers_processed"] == len(sample_papers)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestBackgroundTasksEdgeCases:
    """Edge case tests for background tasks."""

    @pytest.mark.asyncio
    async def test_malformed_feed(self):
        """Test handling of malformed RSS feed."""
        from app.background_tasks import ArXivFeedParser

        parser = ArXivFeedParser(categories=["cs.AI"])

        malformed_feed = "This is not valid XML"

        with patch('feedparser.parse', return_value={"entries": []}):
            papers = parser.parse_feed("https://arxiv.org/rss/cs.AI")

            # Should handle gracefully
            assert isinstance(papers, list)

    @pytest.mark.asyncio
    async def test_paper_missing_fields(self):
        """Test paper with missing required fields."""
        from app.background_tasks import ArXivFeedParser

        parser = ArXivFeedParser()

        incomplete_paper = {
            "title": "Test Paper",
            # Missing other fields
        }

        # Should filter out incomplete papers
        valid = parser._validate_paper(incomplete_paper)

        assert valid is False

    @pytest.mark.asyncio
    async def test_task_exception_recovery(self):
        """Test task recovery after exception."""
        from app.background_tasks import IndexUpdateTask

        task = IndexUpdateTask(name="test", categories=["cs.AI"])

        # First call fails
        with patch.object(task, '_fetch_new_papers', side_effect=[Exception("Error"), []]):
            result1 = await task.execute()
            assert "error" in result1

            # Second call succeeds
            result2 = await task.execute()
            assert result2["status"] == "no_new_papers"

    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self):
        """Test concurrent task execution."""
        from app.background_tasks import (
            BackgroundTaskManager,
            IndexUpdateTask
        )

        manager = BackgroundTaskManager()

        task1 = IndexUpdateTask(name="task1", categories=["cs.AI"])
        task2 = IndexUpdateTask(name="task2", categories=["cs.LG"])
        task3 = IndexUpdateTask(name="task3", categories=["cs.CV"])

        manager.register_task(task1)
        manager.register_task(task2)
        manager.register_task(task3)

        # Mock execute to avoid infinite loops
        with patch.object(IndexUpdateTask, 'execute', return_value={"status": "ok"}):
            with patch.object(IndexUpdateTask, 'start', return_value=None):
                with patch.object(IndexUpdateTask, 'stop', return_value=None):
                    # Start all tasks concurrently
                    await manager.start_all()
                    await manager.stop_all()

                    # All tasks should have been registered
                    assert len(manager._tasks) == 3

    @pytest.mark.asyncio
    async def test_very_long_paper_list(self):
        """Test handling of very long paper list."""
        from app.background_tasks import IndexUpdateTask

        task = IndexUpdateTask(name="test", categories=["cs.AI"], max_papers=10000)

        # Mock many papers
        many_papers = [
            {
                "id": f"2401.{i:05d}",
                "title": f"Paper {i}",
                "summary": "Abstract",
                "authors": ["Author"],
                "published": "2024-01-15T00:00:00Z",
                "link": f"https://arxiv.org/abs/2401.{i:05d}",
                "tags": ["cs.AI"],
            }
            for i in range(1000)
        ]

        with patch.object(task, '_fetch_new_papers', return_value=many_papers):
            mock_store = Mock()
            mock_store.add_documents = AsyncMock()

            with patch('app.background_tasks.get_vectorstore', return_value=mock_store):
                result = await task.execute()

                assert result["papers_processed"] == 1000
