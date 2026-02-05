"""
Integration tests for the RAG pipeline.

Tests the complete flow from document ingestion to query answering.
"""

import pytest
import asyncio
from typing import List, Dict, Any

from app.embeddings.native import NativeEmbeddings
from app.retrieval.bm25 import BM25Retriever, create_bm25_retriever
from app.llm.native import NativeLLM, RAGLLM
from app.rag.native import RAGPipeline, Retriever, RetrievalMethod, RetrievedDocument
from app.repositories.papers import PaperService, PaperEntity, ChunkEntity


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_papers() -> List[Dict[str, Any]]:
    """Sample papers for testing."""
    return [
        {
            "title": "Attention Is All You Need",
            "authors": ["Vaswani et al."],
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            "categories": ["cs.AI", "cs.LG"],
            "arxiv_id": "1706.03762",
            "content": """
            The dominant sequence transduction models are based on complex recurrent or
            convolutional neural networks that include an encoder and a decoder.
            The best performing models also connect the encoder and decoder through
            an attention mechanism. We propose a new simple network architecture,
            the Transformer, based solely on attention mechanisms.
            """,
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Devlin et al."],
            "abstract": "We introduce a new language representation model called BERT...",
            "categories": ["cs.CL"],
            "arxiv_id": "1810.04805",
            "content": """
            We introduce a new language representation model called BERT, which stands
            for Bidirectional Encoder Representations from Transformers. Unlike recent
            language representation models, BERT is designed to pre-train deep bidirectional
            representations from unlabeled text by jointly conditioning on both left and
            right context in all layers.
            """,
        },
        {
            "title": "GPT-4 Technical Report",
            "authors": ["OpenAI"],
            "abstract": "We report the development of GPT-4, a large-scale multimodal model...",
            "categories": ["cs.AI", "cs.CL"],
            "arxiv_id": "2303.08774",
            "content": """
            We report the development of GPT-4, a large-scale, multimodal model which can
            accept image and text inputs and produce text outputs. GPT-4 is a Transformer-based
            model optimized for chat but is also capable of other tasks. GPT-4 generally
            lacks knowledge of events that have occurred after September 2021.
            """,
        },
    ]


@pytest.fixture
def mock_retriever(sample_papers):
    """Create a mock retriever with sample papers."""
    class MockRetriever:
        def __init__(self, papers):
            self.papers = papers

        async def retrieve(self, query: str, method=None, top_k=10):
            # Simple keyword matching for testing
            results = []
            query_lower = query.lower()

            for i, paper in enumerate(self.papers):
                score = 0.0
                content = paper.get("content", "")

                # Count keyword matches
                for word in query_lower.split():
                    if word in content.lower():
                        score += 0.1

                if score > 0:
                    results.append(
                        RetrievedDocument(
                            id=paper["arxiv_id"],
                            content=content,
                            metadata={
                                "title": paper["title"],
                                "authors": paper["authors"],
                            },
                            score=score,
                        )
                    )

            results.sort(key=lambda d: d.score, reverse=True)
            return results[:top_k]

    return MockRetriever(sample_papers)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    class MockLLM:
        async def complete(self, messages, system=None, **kwargs):
            from app.llm.native import CompletionResponse, CompletionResponse

            return CompletionResponse(
                content="This is a mock answer based on the retrieved context.",
                model="mock-model",
                provider="anthropic",
                tokens_used=50,
                latency_ms=100,
            )

        async def chat(self, prompt, system=None, **kwargs):
            from app.llm.native import CompletionResponse, CompletionResponse

            return CompletionResponse(
                content=f"Mock answer to '{prompt}' without context.",
                model="mock-model",
                provider="anthropic",
                tokens_used=30,
                latency_ms=50,
            )

    return MockLLM()


@pytest.fixture
def mock_rag_llm(mock_llm):
    """Create a mock RAG LLM."""
    class MockRAGLLM:
        def __init__(self, llm):
            self.llm = llm

        async def answer_question(self, question, context, sources=None, **kwargs):
            from app.llm.native import CompletionResponse

            return CompletionResponse(
                content=f"Answer to '{question}' based on {len(context)} characters of context.",
                model="mock-model",
                provider="anthropic",
                tokens_used=50 + len(context) // 10,
                latency_ms=100,
            )

    return MockRAGLLM(mock_llm)


# =============================================================================
# EMBEDDING TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_native_embeddings():
    """Test native embedding generation."""
    embeddings = NativeEmbeddings()
    texts = ["Hello world", "Test sentence"]

    result = embeddings.embed_documents(texts)

    assert len(result) == 2
    assert len(result[0]) > 0
    assert len(result[0]) == len(result[1])  # Same dimension


@pytest.mark.asyncio
async def test_async_embeddings():
    """Test async embedding generation."""
    embeddings = NativeEmbeddings()
    texts = ["Async test", "Another async test"]

    result = await embeddings.aembed_documents(texts)

    assert len(result) == 2
    assert len(result[0]) > 0


# =============================================================================
# BM25 TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_bm25_retrieval():
    """Test BM25 document retrieval."""
    documents = [
        {"content": "Machine learning is a subset of AI", "id": "1"},
        {"content": "Deep learning uses neural networks", "id": "2"},
        {"content": "Natural language processing deals with text", "id": "3"},
    ]

    retriever = create_bm25_retriever(documents)
    results = retriever.retrieve("neural networks")

    assert len(results) > 0
    assert results[0]["id"] == "2"


@pytest.mark.asyncio
async def test_bm25_async_retrieval():
    """Test async BM25 retrieval."""
    documents = [
        {"content": "Test document one", "id": "1"},
        {"content": "Test document two", "id": "2"},
    ]

    retriever = create_bm25_retriever(documents)
    results = await retriever.aretrieve("document")

    assert len(results) > 0


# =============================================================================
# RAG PIPELINE TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_rag_pipeline_query(mock_retriever, mock_rag_llm):
    """Test RAG pipeline query."""
    # Mock the retriever interface
    class MockRetrieverAdapter:
        def __init__(self, inner):
            self.inner = inner

        async def retrieve(self, query, method=None, top_k=10):
            return await self.inner.retrieve(query, top_k=top_k)

    retriever = MockRetrieverAdapter(mock_retriever)

    pipeline = RAGPipeline(
        retriever=retriever,
        rag_llm=mock_rag_llm,
    )

    result = await pipeline.query("What is attention?")

    assert result.question == "What is attention?"
    assert result.answer is not None
    assert len(result.answer) > 0
    assert result.latency_ms >= 0  # Allow for zero latency on very fast operations


@pytest.mark.asyncio
async def test_rag_pipeline_no_documents(mock_rag_llm):
    """Test RAG pipeline when no documents are found."""
    class EmptyRetriever:
        async def retrieve(self, query, method=None, top_k=10):
            return []

    pipeline = RAGPipeline(
        retriever=EmptyRetriever(),
        rag_llm=mock_rag_llm,
    )

    result = await pipeline.query("Unknown topic")

    assert result.answer is not None
    assert len(result.documents) == 0


@pytest.mark.asyncio
async def test_retrieval_methods():
    """Test different retrieval methods."""
    # Test that retrieval methods enum works
    assert RetrievalMethod.SEMANTIC == "semantic"
    assert RetrievalMethod.KEYWORD == "keyword"
    assert RetrievalMethod.HYBRID == "hybrid"


# =============================================================================
# REPOSITORY TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_paper_repository():
    """Test paper repository operations."""
    from unittest.mock import AsyncMock

    # Mock session
    session = AsyncMock()

    # Note: This tests the in-memory implementation
    # In production, test with real database
    from app.repositories.papers import PaperRepository

    repo = PaperRepository(session)

    # Create paper
    paper = PaperEntity(
        id="test-1",
        title="Test Paper",
        authors=["Author One"],
        abstract="Test abstract",
    )

    created = await repo.create(paper)
    assert created.id == "test-1"

    # Get paper
    retrieved = await repo.get_by_id("test-1")
    assert retrieved is not None
    assert retrieved.title == "Test Paper"

    # Search papers
    results = await repo.search("Test")
    assert len(results) > 0


@pytest.mark.asyncio
async def test_paper_service():
    """Test paper service operations."""
    from unittest.mock import AsyncMock
    from app.repositories.papers import PaperRepository, ChunkRepository, PaperService

    session = AsyncMock()
    paper_repo = PaperRepository(session)
    chunk_repo = ChunkRepository(session)

    service = PaperService(paper_repo, chunk_repo)

    # Ingest paper
    paper = await service.ingest_paper(
        title="Test Paper",
        authors=["Author One"],
        abstract="Test abstract",
        categories=["cs.AI"],
        chunks=["Chunk one", "Chunk two"],
    )

    assert paper.id is not None
    assert paper.title == "Test Paper"

    # Get paper with chunks
    result = await service.get_paper_with_chunks(paper.id)
    assert result is not None
    assert len(result["chunks"]) == 2


# =============================================================================
# END-TO-END TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_full_ingestion_to_query_flow(sample_papers, mock_retriever, mock_rag_llm):
    """Test complete flow from ingestion to query."""
    # This would normally use real vector store
    # For testing, we use the mock retriever

    retriever = mock_retriever
    pipeline = RAGPipeline(
        retriever=retriever,
        rag_llm=mock_rag_llm,
    )

    # Query the ingested documents
    result = await pipeline.query("What is a Transformer?")

    assert result.question == "What is a Transformer?"
    assert result.answer is not None
    assert result.documents is not None


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.slow
async def test_concurrent_queries(mock_retriever, mock_rag_llm):
    """Test handling multiple concurrent queries."""
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        rag_llm=mock_rag_llm,
    )

    queries = [
        "What is attention?",
        "Explain BERT",
        "What is GPT-4?",
    ]

    # Run queries concurrently
    tasks = [pipeline.query(q) for q in queries]
    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    for result in results:
        assert result.answer is not None


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_query_latency(mock_retriever, mock_rag_llm):
    """Benchmark query latency."""
    import time

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        rag_llm=mock_rag_llm,
    )

    times = []
    for _ in range(10):
        start = time.time()
        await pipeline.query("Test query")
        times.append((time.time() - start) * 1000)

    avg_time = sum(times) / len(times)
    max_time = max(times)

    print(f"\nQuery latency (ms): avg={avg_time:.0f}, max={max_time:.0f}")

    # Mock should be fast
    assert avg_time < 1000
