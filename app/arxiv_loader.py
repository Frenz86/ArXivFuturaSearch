"""ArXiv paper fetcher using the official API (no key required)."""


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

import asyncio
import os
import json
from urllib.parse import urlencode

import feedparser
import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query?"


def build_query(
    search_query: str,
    start: int = 0,
    max_results: int = 25,
    sortBy: str = "submittedDate",
    sortOrder: str = "descending",
) -> str:
    """Build arXiv API query URL."""
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sortBy,
        "sortOrder": sortOrder,
    }
    return ARXIV_API + urlencode(params)


def _parse_feed(data: bytes) -> list[dict]:
    """Parse arXiv Atom feed into paper dictionaries."""
    feed = feedparser.parse(data)
    papers = []

    for entry in feed.entries:
        papers.append({
            "id": entry.get("id"),
            "title": entry.get("title", "").replace("\n", " ").strip(),
            "summary": entry.get("summary", "").replace("\n", " ").strip(),
            "authors": [a.name for a in entry.get("authors", [])],
            "published": entry.get("published"),
            "updated": entry.get("updated"),
            "link": entry.get("link"),
            "tags": (
                [t["term"] for t in entry.get("tags", [])]
                if entry.get("tags")
                else []
            ),
        })

    return papers


async def fetch_arxiv_async(
    search_query: str,
    max_results: int = 25,
    start: int = 0,
    timeout: int = 30,
) -> list[dict]:
    """
    Fetch papers from arXiv API asynchronously.

    Args:
        search_query: arXiv search query
        max_results: Maximum number of results
        start: Starting index for pagination
        timeout: Request timeout in seconds

    Returns:
        List of paper dictionaries
    """
    url = build_query(search_query, start=start, max_results=max_results)

    logger.info("Fetching from arXiv", query=search_query, max_results=max_results)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()

    papers = _parse_feed(response.content)
    logger.info("Fetched papers from arXiv", count=len(papers))

    return papers


def fetch_arxiv(search_query: str, max_results: int = 25, start: int = 0) -> list[dict]:
    """
    Fetch papers from arXiv API (sync wrapper).

    Args:
        search_query: arXiv search query
        max_results: Maximum number of results
        start: Starting index for pagination

    Returns:
        List of paper dictionaries
    """
    return asyncio.get_event_loop().run_until_complete(
        fetch_arxiv_async(search_query, max_results, start)
    )


def save_raw(papers: list[dict], out_path: str) -> None:
    """Save papers to JSON file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    logger.info("Saved papers to file", path=out_path, count=len(papers))


if __name__ == "__main__":
    import time

    # Example query for ML/AI papers
    query = 'cat:cs.LG AND (rag OR retrieval OR "tool use" OR agentic OR evaluation OR multimodal)'
    papers = fetch_arxiv(query, max_results=30)
    save_raw(papers, "data/raw/arxiv_papers.json")
    print(f"Saved {len(papers)} papers to data/raw/arxiv_papers.json")
    time.sleep(1)  # Be nice to arXiv

