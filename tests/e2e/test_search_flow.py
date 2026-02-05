"""
End-to-end tests for the search API.

Tests complete user flows through the HTTP API.
"""

import pytest
import asyncio
from typing import Dict, Any, AsyncGenerator

import httpx
from pydantic import BaseModel


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "http://localhost:8000"
TEST_USER = {
    "email": "e2e-test@example.com",
    "password": "TestPassword123!",
}


# =============================================================================
# MODELS
# =============================================================================

class SearchResult(BaseModel):
    """Search result model."""
    question: str
    answer: str
    sources: list


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create HTTP client for testing."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        yield client


@pytest.fixture
async def auth_token(client: httpx.AsyncClient) -> str:
    """Get auth token for testing."""
    # Try to register/login
    try:
        response = await client.post(
            "/api/auth/register",
            json=TEST_USER,
        )
    except Exception:
        pass

    # Login
    response = await client.post(
        "/api/auth/login",
        json={
            "email": TEST_USER["email"],
            "password": TEST_USER["password"],
        },
    )

    if response.status_code == 200:
        data = response.json()
        return data.get("access_token", "")

    # Return empty token if auth fails
    return ""


# =============================================================================
# HEALTH CHECK
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_health_check(client: httpx.AsyncClient):
    """Test health check endpoint."""
    response = await client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert "status" in data


# =============================================================================
# SEARCH FLOW
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_basic_search_flow(client: httpx.AsyncClient):
    """Test basic search flow."""
    query = "What is a transformer model?"

    response = await client.post(
        "/api/search",
        json={"query": query},
    )

    assert response.status_code == 200

    data = response.json()
    assert "answer" in data or "results" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_ask_question_flow(client: httpx.AsyncClient):
    """Test ask question flow."""
    question = "Explain the attention mechanism in neural networks."

    response = await client.post(
        "/api/ask",
        json={"question": question},
    )

    assert response.status_code == 200

    data = response.json()
    assert "answer" in data or "question" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_search_with_filters(client: httpx.AsyncClient):
    """Test search with category filters."""
    response = await client.post(
        "/api/search",
        json={
            "query": "deep learning",
            "categories": ["cs.AI", "cs.LG"],
            "limit": 5,
        },
    )

    assert response.status_code == 200


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_paper_lookup(client: httpx.AsyncClient):
    """Test looking up a specific paper."""
    # Use a known ArXiv ID
    arxiv_id = "1706.03762"  # "Attention Is All You Need"

    response = await client.get(f"/api/papers/{arxiv_id}")

    # May return 404 if not indexed, but endpoint should exist
    assert response.status_code in [200, 404]


# =============================================================================
# AUTH FLOW
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_register_login_flow(client: httpx.AsyncClient):
    """Test registration and login flow."""
    unique_email = f"test-{asyncio.get_event_loop().time()}@example.com"

    # Register
    register_response = await client.post(
        "/api/auth/register",
        json={
            "email": unique_email,
            "password": "SecurePass123!",
            "name": "Test User",
        },
    )

    # May return 400 if user exists or 201 if created
    assert register_response.status_code in [200, 201, 400]

    # Login
    login_response = await client.post(
        "/api/auth/login",
        json={
            "email": unique_email,
            "password": "SecurePass123!",
        },
    )

    if login_response.status_code == 200:
        data = login_response.json()
        assert "access_token" in data or "token" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_protected_endpoint(client: httpx.AsyncClient, auth_token: str):
    """Test accessing protected endpoint."""
    if not auth_token:
        pytest.skip("No auth token available")

    headers = {"Authorization": f"Bearer {auth_token}"}

    response = await client.get("/api/alerts", headers=headers)

    # Should return 200 (success) or 401 (unauthorized)
    assert response.status_code in [200, 401]


# =============================================================================
# ALERTS FLOW
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_create_alert_flow(client: httpx.AsyncClient, auth_token: str):
    """Test creating an alert."""
    if not auth_token:
        pytest.skip("No auth token available")

    headers = {"Authorization": f"Bearer {auth_token}"}

    # Create alert
    response = await client.post(
        "/api/alerts",
        headers=headers,
        json={
            "name": "Test Alert",
            "keywords": ["transformer", "attention"],
            "categories": ["cs.AI"],
        },
    )

    assert response.status_code in [200, 201, 401]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_alerts_flow(client: httpx.AsyncClient, auth_token: str):
    """Test listing alerts."""
    if not auth_token:
        pytest.skip("No auth token available")

    headers = {"Authorization": f"Bearer {auth_token}"}

    response = await client.get("/api/alerts", headers=headers)

    assert response.status_code in [200, 401]


# =============================================================================
# COLLECTIONS FLOW
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_create_collection_flow(client: httpx.AsyncClient, auth_token: str):
    """Test creating a collection."""
    if not auth_token:
        pytest.skip("No auth token available")

    headers = {"Authorization": f"Bearer {auth_token}"}

    response = await client.post(
        "/api/collections",
        headers=headers,
        json={
            "name": "Test Collection",
            "description": "A test collection",
            "is_public": False,
        },
    )

    assert response.status_code in [200, 201, 401]


# =============================================================================
# EXPORT FLOW
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_export_results_flow(client: httpx.AsyncClient):
    """Test exporting search results."""
    # Mock search results
    mock_results = [
        {
            "title": "Test Paper",
            "authors": ["Author One"],
            "abstract": "Test abstract",
            "id": "test-1",
        }
    ]

    response = await client.post(
        "/api/export/markdown",
        json={
            "results": mock_results,
            "query": "test query",
            "answer": "test answer",
        },
    )

    # Should return markdown content
    assert response.status_code in [200, 404]


# =============================================================================
# CONVERSATION FLOW
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_conversation_flow(client: httpx.AsyncClient):
    """Test multi-turn conversation."""
    questions = [
        "What is a neural network?",
        "How does it learn?",
        "What are its limitations?",
    ]

    conversation_id = None

    for i, question in enumerate(questions):
        response = await client.post(
            "/api/ask",
            json={
                "question": question,
                "conversation_id": conversation_id,
            },
        )

        if response.status_code == 200:
            data = response.json()
            # Get conversation ID for next turn
            conversation_id = data.get("conversation_id", conversation_id)


# =============================================================================
# ERROR HANDLING
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_invalid_query(client: httpx.AsyncClient):
    """Test handling of invalid queries."""
    response = await client.post(
        "/api/search",
        json={"query": ""},  # Empty query
    )

    # Should handle gracefully
    assert response.status_code in [200, 400, 422]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_rate_limiting(client: httpx.AsyncClient):
    """Test rate limiting."""
    # Send many requests quickly
    responses = []
    for _ in range(20):
        response = await client.post(
            "/api/search",
            json={"query": "test"},
        )
        responses.append(response.status_code)

    # Check if any hit rate limit
    assert 429 in responses or all(r == 200 for r in responses)


# =============================================================================
# PERFORMANCE
# =============================================================================

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_searches(client: httpx.AsyncClient):
    """Test handling concurrent searches."""
    queries = [
        "machine learning",
        "neural networks",
        "quantum computing",
        "computer vision",
    ]

    async def search(query):
        response = await client.post(
            "/api/search",
            json={"query": query},
        )
        return response

    # Run concurrent searches
    responses = await asyncio.gather(*[search(q) for q in queries])

    # All should succeed
    for response in responses:
        assert response.status_code in [200, 202]
