"""Pgvector-based hybrid vector store using LangChain PostgreSQL.


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

This module provides a PostgreSQL + pgvector implementation of the vector store
for production use with better scalability, persistence, and concurrent access.
"""

import os
from typing import Optional, Any
from contextlib import contextmanager

import numpy as np
from sqlalchemy import create_engine, text
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_postgres import PGVector

from app.config import settings
from app.logging_config import get_logger
from app.embeddings import get_embedder
from app.query_expansion import reciprocal_rank_fusion, generate_query_variants, expand_query

logger = get_logger(__name__)


class PgvectorStore:
    """
    Hybrid vector store using PostgreSQL + pgvector (semantic) + BM25 (lexical) search.

    This class provides a production-ready alternative to ChromaDB with:
    - Better scalability and concurrent access
    - ACID guarantees and proper persistence
    - Easier integration with existing PostgreSQL infrastructure
    """

    def __init__(self, collection_name: str = "arxiv_papers"):
        """
        Initialize the Pgvector hybrid store with LangChain.

        Args:
            collection_name: Name of the collection (used as table name)
        """
        self.collection_name = collection_name
        self.table_name = f"langchain_pg_collection_{collection_name}"

        # Get embeddings
        embedder = get_embedder()
        self.langchain_embeddings = embedder.get_langchain_embeddings()

        # Build PostgreSQL connection string
        self.connection_string = self._build_connection_string()

        # Initialize connection pool
        self.engine = create_engine(
            self.connection_string,
            pool_size=settings.POSTGRES_POOL_SIZE,
            max_overflow=settings.POSTGRES_MAX_OVERFLOW,
            pool_pre_ping=True,  # Verify connections before using
        )

        self.SessionLocal = create_engine(
            self.connection_string,
            pool_size=settings.POSTGRES_POOL_SIZE,
            max_overflow=settings.POSTGRES_MAX_OVERFLOW,
            pool_pre_ping=True,
        )

        # Initialize LangChain PGVector store
        self.vectorstore = PGVector(
            embeddings=self.langchain_embeddings,
            collection_name=collection_name,
            connection=self.connection_string,
            use_jsonb=True,
            distance_strategy="cosine",
        )

        # Ensure pgvector extension is installed
        self._ensure_pgvector_extension()

        # BM25 retriever (will be populated when documents are added)
        self.bm25_retriever: Optional[BM25Retriever] = None
        self._documents: list[Document] = []

        # Try to load existing documents for BM25
        self._load_documents_for_bm25()

        logger.info(
            "Initialized PgvectorStore with LangChain",
            collection=collection_name,
            connection=self._safe_connection_string(),
            documents=self.count(),
        )

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from settings."""
        return (
            f"postgresql+psycopg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )

    def _safe_connection_string(self) -> str:
        """Return connection string with password masked for logging."""
        return (
            f"postgresql+psycopg://{settings.POSTGRES_USER}:****@"
            f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )

    def _execute_sql(self, query: str, params: Optional[dict] = None):
        """
        Execute SQL query safely with parameters.

        Args:
            query: SQL query string
            params: Optional parameters dict

        Returns:
            Result object
        """
        with self.SessionLocal.connect() as conn:
            if params:
                return conn.execute(text(query), params)
            return conn.execute(text(query))

    def _ensure_pgvector_extension(self) -> None:
        """Ensure pgvector extension is installed in PostgreSQL."""
        try:
            result = self._execute_sql(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
            exists = result.scalar()

            if not exists:
                logger.warning(
                    "pgvector extension not found. "
                    "Please install it: CREATE EXTENSION vector;"
                )
            else:
                logger.debug("pgvector extension is installed")
        except Exception as e:
            logger.error("Failed to check pgvector extension", error=str(e))

    def _load_documents_for_bm25(self) -> None:
        """Load existing documents from PostgreSQL to initialize BM25."""
        try:
            # Use LangChain's connection to get all documents
            result = self._execute_sql(
                """
                SELECT id, document, metadata
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
                )
                """,
                {"collection_name": self.collection_name}
            )

            self._documents = []
            for row in result:
                self._documents.append(
                    Document(
                        page_content=row[1],  # document
                        metadata=row[2] or {}  # metadata
                    )
                )

            # Initialize BM25 retriever
            if self._documents:
                self.bm25_retriever = BM25Retriever.from_documents(self._documents)
                logger.info("Loaded BM25 retriever", documents=len(self._documents))
        except Exception as e:
            logger.warning("Could not load existing documents for BM25", error=str(e))

    def add(
        self,
        vectors: np.ndarray,
        chunk_ids: list[str],
        texts: list[str],
        metas: list[dict],
    ) -> None:
        """
        Add documents to the store.

        Args:
            vectors: NumPy array of embeddings (n, dim) - not used with LangChain
            chunk_ids: List of unique chunk identifiers
            texts: List of text chunks
            metas: List of metadata dictionaries
        """
        if len(chunk_ids) == 0:
            logger.warning("No documents to add")
            return

        # Create LangChain Document objects
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metas)
        ]

        # Add to PGVector using LangChain (it will compute embeddings automatically)
        self.vectorstore.add_documents(documents=documents, ids=chunk_ids)

        # Update documents list and rebuild BM25
        self._documents.extend(documents)
        self.bm25_retriever = BM25Retriever.from_documents(self._documents)

        logger.info(
            "Added documents to PgvectorStore",
            count=len(chunk_ids),
            total=self.count(),
        )

    def _build_filter_dict(self, filters: Optional[dict]) -> Optional[dict]:
        """
        Convert filter dict to LangChain PGVector filter format.

        Args:
            filters: Filter dictionary

        Returns:
            LangChain compatible filter dict
        """
        if not filters:
            return None

        # LangChain PGVector supports metadata filtering via dict
        # Most filters work as-is, but we may need to transform some
        filter_dict = {}

        for key, value in filters.items():
            # For pgvector, we pass the filters as-is to LangChain
            # LangChain will handle the conversion to SQL WHERE clauses
            filter_dict[key] = value

        return filter_dict

    def search(
        self,
        query_vec: np.ndarray,
        query_text: str = "",
        top_k: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Hybrid search combining semantic and lexical search.

        Args:
            query_vec: Query embedding vector (not used with LangChain)
            query_text: Query text for both semantic and BM25
            top_k: Number of results to return
            semantic_weight: Weight for semantic scores
            bm25_weight: Weight for BM25 scores
            filters: Optional metadata filters

        Returns:
            List of result dictionaries with 'score', 'chunk_id', 'text', 'meta'
        """
        if not query_text:
            logger.warning("Empty query_text provided for hybrid search")
            return []

        retrieval_k = min(top_k * 4, self.count()) if self.count() > 0 else top_k

        # Semantic search using PGVector similarity search
        embedder = get_embedder()
        query_embedding = embedder.embed_query(query_text)

        # Build search kwargs
        search_kwargs = {"k": retrieval_k}
        if filters:
            # Use the same filter format as ChromaDB
            search_kwargs["filter"] = self._build_filter_dict(filters)

        # Get similarity scores (PGVector returns distances, convert to similarities)
        semantic_results_with_scores = self.vectorstore.similarity_search_with_score_by_vector(
            embedding=query_embedding,
            **search_kwargs
        )

        # Process semantic results
        semantic_results = []
        for i, (doc, distance) in enumerate(semantic_results_with_scores):
            # PGVector with cosine distance: lower is better, convert to similarity
            similarity = 1.0 - float(distance)
            semantic_results.append({
                "text": doc.page_content,
                "meta": doc.metadata,
                "semantic_score": similarity,
                "semantic_rank": i,
            })

        # If BM25 is available, combine results
        bm25_lookup = {}
        if self.bm25_retriever:
            self.bm25_retriever.k = retrieval_k
            try:
                bm25_docs = self.bm25_retriever.invoke(query_text)

                # Create lookup for BM25 results
                for i, doc in enumerate(bm25_docs):
                    content_key = doc.page_content[:100]
                    bm25_lookup[content_key] = {
                        "text": doc.page_content,
                        "meta": doc.metadata,
                        "bm25_score": 1.0 - (i / len(bm25_docs)),
                        "bm25_rank": i,
                    }
            except Exception as e:
                logger.warning("BM25 retrieval failed", error=str(e))

        # Combine semantic and BM25 scores
        combined_results = []
        for sem_result in semantic_results:
            content_key = sem_result["text"][:100]

            # Start with semantic score
            combined_score = semantic_weight * sem_result["semantic_score"]

            # Add BM25 score if available
            if content_key in bm25_lookup:
                combined_score += bm25_weight * bm25_lookup[content_key]["bm25_score"]

            combined_results.append({
                "score": combined_score,
                "chunk_id": sem_result["meta"].get("chunk_id", f"doc_{sem_result['semantic_rank']}"),
                "text": sem_result["text"],
                "meta": sem_result["meta"],
                "semantic_score": sem_result["semantic_score"],
            })

        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        results = combined_results[:top_k]

        # Log search quality
        if results:
            avg_score = sum(r["score"] for r in results) / len(results)
            logger.debug("Search completed", avg_score=f"{avg_score:.3f}", top_score=f"{results[0]['score']:.3f}")

        return results

    def search_ensemble(
        self,
        query_text: str,
        top_k: int = 5,
        query_expansion: bool = True,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Advanced ensemble search with query expansion and RRF.

        This method:
        1. Expands the query with related terms (acronyms, synonyms)
        2. Generates multiple query variants
        3. Searches with each variant
        4. Combines results using Reciprocal Rank Fusion (RRF)

        Args:
            query_text: Query text
            top_k: Number of results to return
            query_expansion: Whether to expand the query with related terms
            filters: Optional metadata filters

        Returns:
            List of result dictionaries with 'score', 'chunk_id', 'text', 'meta'
        """
        if not query_text:
            logger.warning("Empty query_text provided for ensemble search")
            return []

        embedder = get_embedder()
        retrieval_k = min(top_k * 3, self.count()) if self.count() > 0 else top_k

        # Generate query variants
        query_variants = generate_query_variants(query_text)

        # Apply query expansion to the first variant
        if query_expansion:
            query_variants = [expand_query(q, method="acronym") for q in query_variants]

        logger.debug("Ensemble search", variants=len(query_variants), queries=query_variants)

        # Search with each query variant
        all_results = []
        for variant_query in query_variants:
            query_embedding = embedder.embed_query(variant_query)

            # Build search kwargs
            search_kwargs = {"k": retrieval_k}
            if filters:
                search_kwargs["filter"] = self._build_filter_dict(filters)

            try:
                semantic_results_with_scores = self.vectorstore.similarity_search_with_score_by_vector(
                    embedding=query_embedding,
                    **search_kwargs
                )

                for doc, distance in semantic_results_with_scores:
                    similarity = 1.0 - float(distance)
                    all_results.append({
                        "text": doc.page_content,
                        "meta": doc.metadata,
                        "score": similarity,  # Will be replaced by RRF score
                        "query_used": variant_query,
                    })
            except Exception as e:
                logger.warning("Search failed for variant", variant=variant_query, error=str(e))

        # If BM25 is available, add BM25 results to ensemble
        if self.bm25_retriever and all_results:
            self.bm25_retriever.k = retrieval_k
            try:
                bm25_docs = self.bm25_retriever.invoke(query_text)
                for i, doc in enumerate(bm25_docs):
                    all_results.append({
                        "text": doc.page_content,
                        "meta": doc.metadata,
                        "score": 1.0 - (i / len(bm25_docs)),
                        "query_used": "bm25",
                    })
            except Exception as e:
                logger.warning("BM25 retrieval failed in ensemble", error=str(e))

        # Apply RRF to combine all results
        if len(all_results) == 0:
            return []

        rrf_results = reciprocal_rank_fusion([all_results], k=60, top_k=top_k)

        # Log ensemble results
        if rrf_results:
            logger.info(
                "Ensemble search completed",
                results=len(rrf_results),
                top_score=f"{rrf_results[0]['score']:.3f}",
                queries_used=len(query_variants)
            )

        return rrf_results

    def get_ensemble_retriever(
        self,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        """
        Get a retriever for hybrid search.

        Note: Returns semantic retriever only. Use search() for true hybrid search.

        Args:
            top_k: Number of results to return
            semantic_weight: Weight for semantic retriever (not used)
            bm25_weight: Weight for BM25 retriever (not used)

        Returns:
            LangChain retriever object (semantic only)
        """
        logger.warning(
            "get_ensemble_retriever() returns semantic-only retriever. "
            "Use search() method for hybrid search (semantic + BM25)."
        )
        return self.vectorstore.as_retriever(search_kwargs={"k": top_k})

    def count(self) -> int:
        """Get total number of documents in the collection."""
        try:
            result = self._execute_sql(
                """
                SELECT COUNT(DISTINCT c.id)
                FROM langchain_pg_embedding c
                WHERE c.collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
                )
                """,
                {"collection_name": self.collection_name}
            )
            return result.scalar() or 0
        except Exception as e:
            logger.error("Failed to count documents", error=str(e))
            return len(self._documents)

    def reset(self) -> None:
        """Delete all documents from the collection."""
        try:
            # Delete all embeddings for this collection
            self._execute_sql(
                """
                DELETE FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
                )
                """,
                {"collection_name": self.collection_name}
            )
            self._documents = []
            self.bm25_retriever = None
            logger.info("Reset PgvectorStore")
        except Exception as e:
            logger.error("Failed to reset PgvectorStore", error=str(e))

    def delete_collection(self) -> None:
        """Permanently delete the collection."""
        try:
            # Delete the collection itself
            self._execute_sql(
                "DELETE FROM langchain_pg_collection WHERE name = :collection_name",
                {"collection_name": self.collection_name}
            )
            self._documents = []
            self.bm25_retriever = None
            logger.info("Deleted collection", name=self.collection_name)
        except Exception as e:
            logger.error("Failed to delete collection", error=str(e))
