"""Pgvector-based hybrid vector store using native PostgreSQL + pgvector.


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

import re
from typing import Optional

import numpy as np
import rank_bm25
from sqlalchemy import create_engine, text

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
        Initialize the Pgvector hybrid store with native SQL.

        Args:
            collection_name: Name of the collection (used as table name)
        """
        # Validate collection_name to prevent SQL injection
        if not re.match(r'^[a-zA-Z0-9_]+$', collection_name):
            raise ValueError(f"Invalid collection_name: {collection_name}")

        self.collection_name = collection_name
        self.table_name = f"pgvector_store_{collection_name}"

        # Get embeddings
        self._embedder = get_embedder()

        # Build PostgreSQL connection string
        self.connection_string = self._build_connection_string()

        # Initialize connection pool
        self.engine = create_engine(
            self.connection_string,
            pool_size=settings.POSTGRES_POOL_SIZE,
            max_overflow=settings.POSTGRES_MAX_OVERFLOW,
            pool_pre_ping=True,
        )

        # Ensure pgvector extension is installed and table exists
        self._ensure_pgvector_extension()
        self._ensure_table_exists()

        # BM25 retriever (will be populated when documents are added)
        self._bm25: Optional[rank_bm25.BM25Okapi] = None
        self._bm25_corpus: list[str] = []

        # Try to load existing documents for BM25
        self._load_documents_for_bm25()

        logger.info(
            "Initialized PgvectorStore (native)",
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
            Result object (must be used within the context)
        """
        with self.engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            # Buffer all results before connection closes
            return list(result)

    def _ensure_pgvector_extension(self) -> None:
        """Ensure pgvector extension is installed in PostgreSQL."""
        try:
            result = self._execute_sql(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
            exists = result[0][0] if result else False

            if not exists:
                logger.warning(
                    "pgvector extension not found. "
                    "Please install it: CREATE EXTENSION vector;"
                )
            else:
                logger.debug("pgvector extension is installed")
        except Exception as e:
            logger.error("Failed to check pgvector extension", error=str(e))

    def _ensure_table_exists(self) -> None:
        """Create the vector store table if it doesn't exist."""
        dim = self._embedder.dimension
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        chunk_id TEXT PRIMARY KEY,
                        embedding vector({dim}),
                        document TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

                # Create index for faster similarity search
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                    ON {self.table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """))

                # Create GIN index for metadata filtering
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx
                    ON {self.table_name}
                    USING GIN (metadata)
                """))
                conn.commit()
                logger.debug(f"Table {self.table_name} is ready")
        except Exception as e:
            logger.error("Failed to create table", error=str(e))

    def _load_documents_for_bm25(self) -> None:
        """Load existing documents from PostgreSQL to initialize BM25."""
        try:
            result = self._execute_sql(
                f"SELECT document FROM {self.table_name} ORDER BY chunk_id"
            )

            self._bm25_corpus = [row[0] for row in result]

            if self._bm25_corpus:
                self._bm25 = rank_bm25.BM25Okapi(self._tokenize(self._bm25_corpus))
                logger.info("Loaded BM25 retriever", documents=len(self._bm25_corpus))
        except Exception as e:
            logger.warning("Could not load existing documents for BM25", error=str(e))

    @staticmethod
    def _tokenize(texts: list[str]) -> list[list[str]]:
        """Simple whitespace tokenizer for BM25."""
        return [text.lower().split() for text in texts]

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
            vectors: NumPy array of embeddings (n, dim)
            chunk_ids: List of unique chunk identifiers
            texts: List of text chunks
            metas: List of metadata dictionaries
        """
        if len(chunk_ids) == 0:
            logger.warning("No documents to add")
            return

        # Insert documents into PostgreSQL
        try:
            with self.engine.connect() as conn:
                for chunk_id, vector, text, meta in zip(chunk_ids, vectors, texts, metas):
                    # Convert numpy array to pgvector string format: "[x,y,z,...]"
                    vector_str = "[" + ",".join(map(str, vector)) + "]"

                    conn.execute(text(f"""
                        INSERT INTO {self.table_name} (chunk_id, embedding, document, metadata)
                        VALUES (:chunk_id, :embedding::vector, :document, :metadata::jsonb)
                        ON CONFLICT (chunk_id) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            document = EXCLUDED.document,
                            metadata = EXCLUDED.metadata
                    """), {
                        "chunk_id": chunk_id,
                        "embedding": vector_str,
                        "document": text,
                        "metadata": meta,
                    })
                conn.commit()

            # Update BM25 corpus and rebuild index
            self._bm25_corpus.extend(texts)
            self._bm25 = rank_bm25.BM25Okapi(self._tokenize(self._bm25_corpus))

            logger.info(
                "Added documents to PgvectorStore",
                count=len(chunk_ids),
                total=self.count(),
            )
        except Exception as e:
            logger.error("Failed to add documents", error=str(e))

    def _build_filter_dict(self, filters: Optional[dict]) -> tuple[str, dict]:
        """
        Convert filter dict to SQL WHERE clause for JSONB metadata.

        Args:
            filters: Filter dictionary

        Returns:
            Tuple of (where_clause, params_dict)
        """
        if not filters:
            return "", {}

        conditions = []
        params = {}

        for i, (key, value) in enumerate(filters.items()):
            param_name = f"filter_{i}"

            # Handle date range filters
            if key == "published_after" and isinstance(value, str):
                conditions.append(f"(metadata->>'published_date')::date >= :{param_name}")
                params[param_name] = value
            elif key == "published_before" and isinstance(value, str):
                conditions.append(f"(metadata->>'published_date')::date <= :{param_name}")
                params[param_name] = value
            # Handle list containment (for authors, tags, etc.)
            elif key in ("authors", "tags", "categories") and isinstance(value, list):
                # Check if any value in the list matches
                conditions.append(f"metadata->>'{key}' ?| :{param_name}")
                params[param_name] = value
            # Handle exact match
            else:
                conditions.append(f"metadata->>'{key}' = :{param_name}")
                params[param_name] = value

        where_clause = " AND " + " AND ".join(conditions) if conditions else ""
        return where_clause, params

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
            query_vec: Query embedding vector (optional, will be computed if not provided)
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

        # Semantic search using pgvector
        if query_vec is None or query_vec.size == 0:
            query_embedding = self._embedder.embed_query(query_text)
        else:
            query_embedding = query_vec

        # Build WHERE clause for filters
        where_clause, filter_params = self._build_filter_dict(filters)

        # Convert query vector to pgvector format
        query_vec_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # Execute similarity search using cosine distance
        semantic_query = f"""
            SELECT
                chunk_id,
                document,
                metadata,
                1 - (embedding <=> :query_embedding::vector) AS similarity
            FROM {self.table_name}
            WHERE 1=1 {where_clause}
            ORDER BY embedding <=> :query_embedding::vector
            LIMIT :limit
        """

        params = {
            "query_embedding": query_vec_str,
            "limit": retrieval_k,
            **filter_params,
        }

        rows = self._execute_sql(semantic_query, params)

        # Process semantic results
        semantic_results = []
        for i, row in enumerate(rows):
            semantic_results.append({
                "text": row[1],  # document
                "meta": row[2],  # metadata
                "semantic_score": float(row[3]),  # similarity
                "semantic_rank": i,
            })

        # If BM25 is available, combine results
        bm25_lookup = {}
        if self._bm25 and query_text:
            try:
                tokenized_query = self._tokenize([query_text])[0]
                scores = self._bm25.get_scores(tokenized_query)
                top_indices = np.argsort(-scores)[:retrieval_k]

                for rank, idx in enumerate(top_indices):
                    if scores[idx] > 0:  # Only include positive scores
                        bm25_lookup[self._bm25_corpus[idx][:100]] = {
                            "text": self._bm25_corpus[idx],
                            "meta": {},
                            "bm25_score": float(scores[idx]),
                            "bm25_rank": rank,
                        }
            except Exception as e:
                logger.warning("BM25 retrieval failed", error=str(e))

        # Combine semantic and BM25 scores
        combined_results = []
        for sem_result in semantic_results:
            content_key = sem_result["text"][:100]

            # Start with semantic score
            combined_score = semantic_weight * sem_result["semantic_score"]

            # Add BM25 score if available (normalized)
            if content_key in bm25_lookup:
                # Normalize BM25 score to 0-1 range
                bm25_max = max(r["bm25_score"] for r in bm25_lookup.values()) if bm25_lookup else 1.0
                normalized_bm25 = bm25_lookup[content_key]["bm25_score"] / max(bm25_max, 0.001)
                combined_score += bm25_weight * normalized_bm25

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

        retrieval_k = min(top_k * 3, self.count()) if self.count() > 0 else top_k

        # Generate query variants
        query_variants = generate_query_variants(query_text)

        # Apply query expansion to the first variant
        if query_expansion:
            query_variants = [expand_query(q, method="acronym") for q in query_variants]

        logger.debug("Ensemble search", variants=len(query_variants), queries=query_variants)

        # Build WHERE clause for filters
        where_clause, filter_params = self._build_filter_dict(filters)

        # Search with each query variant
        all_results = []
        for variant_query in query_variants:
            query_embedding = self._embedder.embed_query(variant_query)
            query_vec_str = "[" + ",".join(map(str, query_embedding)) + "]"

            params = {
                "query_embedding": query_vec_str,
                "limit": retrieval_k,
                **filter_params,
            }

            try:
                rows = self._execute_sql(f"""
                    SELECT
                        chunk_id,
                        document,
                        metadata,
                        1 - (embedding <=> :query_embedding::vector) AS similarity
                    FROM {self.table_name}
                    WHERE 1=1 {where_clause}
                    ORDER BY embedding <=> :query_embedding::vector
                    LIMIT :limit
                """, params)

                for row in rows:
                    all_results.append({
                        "text": row[1],  # document
                        "meta": row[2],  # metadata
                        "score": float(row[3]),  # similarity (will be replaced by RRF)
                        "query_used": variant_query,
                    })
            except Exception as e:
                logger.warning("Search failed for variant", variant=variant_query, error=str(e))

        # If BM25 is available, add BM25 results to ensemble
        if self._bm25 and query_text:
            try:
                tokenized_query = self._tokenize([query_text])[0]
                scores = self._bm25.get_scores(tokenized_query)
                top_indices = np.argsort(-scores)[:retrieval_k]

                for rank, idx in enumerate(top_indices):
                    if scores[idx] > 0:
                        all_results.append({
                            "text": self._bm25_corpus[idx],
                            "meta": {},
                            "score": float(scores[idx]),
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

    def count(self) -> int:
        """Get total number of documents in the collection."""
        try:
            result = self._execute_sql(f"SELECT COUNT(*) FROM {self.table_name}")
            return result[0][0] if result else 0
        except Exception as e:
            logger.error("Failed to count documents", error=str(e))
            return len(self._bm25_corpus)

    def reset(self) -> None:
        """Delete all documents from the collection."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"DELETE FROM {self.table_name}"))
                conn.commit()

            self._bm25_corpus = []
            self._bm25 = None
            logger.info("Reset PgvectorStore")
        except Exception as e:
            logger.error("Failed to reset PgvectorStore", error=str(e))

    def delete_collection(self) -> None:
        """Permanently delete the collection."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {self.table_name}"))
                conn.commit()

            self._bm25_corpus = []
            self._bm25 = None
            logger.info("Deleted collection", name=self.collection_name)
        except Exception as e:
            logger.error("Failed to delete collection", error=str(e))
