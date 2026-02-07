"""
Native RAG implementation.

This module provides a complete RAG pipeline using native implementations
for embeddings, retrieval, and LLM interactions.
"""

from typing import List, Dict, Any, Optional, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time

from app.embeddings.native import NativeEmbeddings, get_embeddings
from app.retrieval.bm25 import BM25Retriever, HybridRetriever, create_bm25_retriever
from app.llm.native import NativeLLM, RAGLLM, Message, Role
from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# RESULT TYPES
# =============================================================================

class RetrievalMethod(str, Enum):
    """Retrieval methods."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    retrieval_method: RetrievalMethod = RetrievalMethod.SEMANTIC

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "retrieval_method": self.retrieval_method.value,
        }


@dataclass
class RAGResult:
    """Result from RAG query."""
    question: str
    answer: str
    documents: List[RetrievedDocument]
    sources: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_used: Optional[int] = None
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "documents": [d.to_dict() for d in self.documents],
            "sources": self.sources,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "retrieval_time_ms": self.retrieval_time_ms,
            "generation_time_ms": self.generation_time_ms,
        }


# =============================================================================
# VECTOR STORE INTERFACE
# =============================================================================

class VectorStore:
    """
    Abstract vector store interface.

    Can be implemented for ChromaDB, pgvector, etc.
    """

    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add texts to the store."""
        raise NotImplementedError

    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDocument]:
        """Search by similarity."""
        raise NotImplementedError

    async def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        raise NotImplementedError


# =============================================================================
# RETRIEVER
# =============================================================================

class Retriever:
    """
    Unified retriever combining multiple retrieval methods.

    Supports semantic, keyword, and hybrid retrieval.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        embeddings: Optional[NativeEmbeddings] = None,
        default_method: RetrievalMethod = RetrievalMethod.SEMANTIC,
        top_k: int = 10,
        score_threshold: float = 0.7,
    ):
        """
        Initialize retriever.

        Args:
            vector_store: Vector store for semantic search
            bm25_retriever: BM25 retriever for keyword search
            embeddings: Embeddings for vector search
            default_method: Default retrieval method
            top_k: Number of documents to retrieve
            score_threshold: Minimum score for results
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embeddings = embeddings or get_embeddings()
        self.default_method = default_method
        self.top_k = top_k
        self.score_threshold = score_threshold

        logger.info(
            "Retriever initialized",
            default_method=default_method.value,
            top_k=top_k,
        )

    async def retrieve(
        self,
        query: str,
        method: Optional[RetrievalMethod] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents for a query.

        Args:
            query: Query text
            method: Retrieval method (uses default if None)
            top_k: Number of results (uses default if None)

        Returns:
            List of retrieved documents
        """
        method = method or self.default_method
        k = top_k or self.top_k

        start_time = time.time()

        if method == RetrievalMethod.SEMANTIC:
            docs = await self._semantic_search(query, k)
        elif method == RetrievalMethod.KEYWORD:
            docs = await self._keyword_search(query, k)
        elif method == RetrievalMethod.HYBRID:
            docs = await self._hybrid_search(query, k)
        else:
            docs = []

        elapsed_ms = (time.time() - start_time) * 1000

        logger.debug(
            "Retrieval completed",
            method=method.value,
            n_results=len(docs),
            time=f"{elapsed_ms:.0f}ms",
        )

        return docs

    async def _semantic_search(
        self,
        query: str,
        k: int,
    ) -> List[RetrievedDocument]:
        """Semantic search using vector store."""
        if not self.vector_store:
            logger.warning("Vector store not available")
            return []

        results = await self.vector_store.similarity_search(
            query=query,
            k=k,
            score_threshold=self.score_threshold,
        )

        return results

    async def _keyword_search(
        self,
        query: str,
        k: int,
    ) -> List[RetrievedDocument]:
        """Keyword search using BM25."""
        if not self.bm25_retriever:
            logger.warning("BM25 retriever not available")
            return []

        results = await self.bm25_retriever.aretrieve(
            query=query,
            top_k=k,
            score_threshold=self.score_threshold,
        )

        # Convert to RetrievedDocument
        docs = []
        for r in results:
            doc = RetrievedDocument(
                id=r.get("id", ""),
                content=r.get("content", ""),
                metadata=r.get("metadata", {}),
                score=r.get("score", 0.0),
                retrieval_method=RetrievalMethod.KEYWORD,
            )
            docs.append(doc)

        return docs

    async def _hybrid_search(
        self,
        query: str,
        k: int,
    ) -> List[RetrievedDocument]:
        """Hybrid search combining semantic and keyword."""
        # Parallel retrieval
        semantic_task = self._semantic_search(query, k * 2)
        keyword_task = self._keyword_search(query, k * 2)

        semantic_docs, keyword_docs = await asyncio.gather(
            semantic_task,
            keyword_task,
        )

        # Combine and deduplicate
        seen_ids = set()
        combined = []

        for doc in semantic_docs + keyword_docs:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                doc.retrieval_method = RetrievalMethod.HYBRID
                combined.append(doc)

        # Re-score by average
        for doc in combined:
            semantic_score = next(
                (d.score for d in semantic_docs if d.id == doc.id),
                0.0,
            )
            keyword_score = next(
                (d.score for d in keyword_docs if d.id == doc.id),
                0.0,
            )
            doc.score = (semantic_score + keyword_score) / 2

        # Sort by score and limit
        combined.sort(key=lambda d: d.score, reverse=True)
        return combined[:k]


# =============================================================================
# NATIVE RAG PIPELINE
# =============================================================================

class RAGPipeline:
    """
    Complete RAG pipeline for retrieval-augmented generation.

    Handles retrieval, context building, and answer generation.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: Optional[NativeLLM] = None,
        rag_llm: Optional[RAGLLM] = None,
        max_context_length: int = 8000,
        system_prompt: Optional[str] = None,
        include_sources: bool = True,
    ):
        """
        Initialize RAG pipeline.

        Args:
            retriever: Document retriever
            llm: Base LLM instance
            rag_llm: RAG-optimized LLM (creates from llm if None)
            max_context_length: Max tokens for context
            system_prompt: Optional system prompt
            include_sources: Whether to include sources in output
        """
        self.retriever = retriever
        self.llm = llm or NativeLLM()
        self.rag_llm = rag_llm or RAGLLM(self.llm)
        self.max_context_length = max_context_length
        self.system_prompt = system_prompt
        self.include_sources = include_sources

        logger.info(
            "RAGPipeline initialized",
            max_context_length=max_context_length,
            include_sources=include_sources,
        )

    async def query(
        self,
        question: str,
        retrieval_method: Optional[RetrievalMethod] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> RAGResult:
        """
        Query the RAG pipeline.

        Args:
            question: User question
            retrieval_method: How to retrieve documents
            top_k: Number of documents to retrieve
            **kwargs: Additional LLM parameters

        Returns:
            RAGResult with answer and sources
        """
        start_time = time.time()

        # Retrieve documents
        retrieval_start = time.time()
        docs = await self.retriever.retrieve(
            query=question,
            method=retrieval_method,
            top_k=top_k,
        )
        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        if not docs:
            logger.warning("No documents retrieved", question=question[:100])

            # Generate answer without context
            response = await self.rag_llm.llm.chat(
                prompt=question,
                system=self.system_prompt or "You are a helpful assistant.",
                **kwargs,
            )

            return RAGResult(
                question=question,
                answer=response,
                documents=[],
                latency_ms=(time.time() - start_time) * 1000,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=(time.time() - retrieval_start - retrieval_time_ms / 1000) * 1000,
            )

        # Build context
        context = self._build_context(docs)

        # Build sources list
        sources = [
            {
                "id": doc.id,
                "title": doc.metadata.get("title", ""),
                "authors": doc.metadata.get("authors", []),
                "score": doc.score,
                "url": doc.metadata.get("url", ""),
            }
            for doc in docs
        ]

        # Generate answer
        generation_start = time.time()
        response = await self.rag_llm.answer_question(
            question=question,
            context=context,
            sources=sources if self.include_sources else None,
            **kwargs,
        )
        generation_time_ms = (time.time() - generation_start) * 1000

        result = RAGResult(
            question=question,
            answer=response.content,
            documents=docs,
            sources=sources if self.include_sources else [],
            latency_ms=(time.time() - start_time) * 1000,
            tokens_used=response.tokens_used,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
        )

        logger.info(
            "RAG query completed",
            question_length=len(question),
            n_docs=len(docs),
            tokens=result.tokens_used,
            latency=f"{result.latency_ms:.0f}ms",
        )

        return result

    async def stream_query(
        self,
        question: str,
        retrieval_method: Optional[RetrievalMethod] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream a RAG query response.

        Args:
            question: User question
            retrieval_method: How to retrieve documents
            top_k: Number of documents to retrieve
            **kwargs: Additional LLM parameters

        Yields:
            Response chunks
        """
        # Retrieve documents first
        docs = await self.retriever.retrieve(
            query=question,
            method=retrieval_method,
            top_k=top_k,
        )

        # Build context
        context = self._build_context(docs) if docs else "No relevant documents found."

        # Build prompt
        user_prompt = f"""Context:
{context}

Question: {question}

Provide a comprehensive answer based on the context above."""

        messages = [Message(role=Role.USER, content=user_prompt)]

        # Stream response
        async for chunk in self.llm.stream(messages, system=self.system_prompt, **kwargs):
            yield chunk.delta

    def _build_context(
        self,
        docs: List[RetrievedDocument],
    ) -> str:
        """
        Build context string from retrieved documents.

        Args:
            docs: Retrieved documents

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, doc in enumerate(docs, 1):
            # Get title for header
            title = doc.metadata.get("title", f"Document {doc.id}")

            # Add document
            context_parts.append(f"[{i}] {title}")
            context_parts.append(doc.content[:2000])  # Limit length
            context_parts.append("")  # Blank line

        # Combine and truncate
        full_context = "\n".join(context_parts)

        # Rough token estimation (4 chars per token)
        max_chars = self.max_context_length * 4

        if len(full_context) > max_chars:
            full_context = full_context[:max_chars] + "\n...(truncated)"

        return full_context


# =============================================================================
# STREAMING RAG
# =============================================================================

class StreamingRAGPipeline(RAGPipeline):
    """
    Streaming RAG pipeline for real-time responses.

    Streams the answer while retrieving documents in background.
    """

    async def query_with_thinking(
        self,
        question: str,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Query with streaming thinking process.

        Yields status updates during processing.

        Args:
            question: User question
            **kwargs: Additional parameters

        Yields:
            Status dictionaries with 'type' and 'data' keys
        """
        yield {"type": "start", "data": {"question": question}}

        # Retrieve documents
        yield {"type": "retrieving", "data": {}}
        docs = await self.retriever.retrieve(query=question)

        yield {
            "type": "retrieved",
            "data": {
                "count": len(docs),
                "documents": [
                    {
                        "id": doc.id,
                        "title": doc.metadata.get("title", ""),
                        "score": doc.score,
                    }
                    for doc in docs
                ],
            },
        }

        # Build context
        context = self._build_context(docs)

        yield {"type": "generating", "data": {}}

        # Stream answer
        full_answer = ""
        async for chunk in self.llm.stream(
            messages=[Message(role=Role.USER, content=f"{question}\n\nContext: {context[:1000]}")],
            system=self.system_prompt,
            **kwargs,
        ):
            full_answer += chunk.delta
            yield {"type": "chunk", "data": {"delta": chunk.delta, "full": full_answer}}

        yield {
            "type": "complete",
            "data": {
                "answer": full_answer,
                "documents": [d.to_dict() for d in docs],
            },
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_rag_pipeline(
    vector_store: Optional[VectorStore] = None,
    bm25_retriever: Optional[BM25Retriever] = None,
    **kwargs,
) -> RAGPipeline:
    """
    Create a RAG pipeline.

    Args:
        vector_store: Optional vector store
        bm25_retriever: Optional BM25 retriever
        **kwargs: Additional RAGPipeline parameters

    Returns:
        RAGPipeline instance
    """
    # Create retriever
    retriever = Retriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
    )

    # Create pipeline
    return RAGPipeline(retriever=retriever, **kwargs)


# Compatibility layer removed - no longer needed
