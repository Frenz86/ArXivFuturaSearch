"""ChromaDB-based hybrid vector store using LangChain."""


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

import os
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.config import settings
from app.logging_config import get_logger
from app.embeddings import get_embedder
from app.query_expansion import expand_query, reciprocal_rank_fusion, generate_query_variants

logger = get_logger(__name__)


class ChromaHybridStore:
    """
    Hybrid vector store using LangChain's Chroma (semantic) + BM25 (lexical) search.

    This class wraps LangChain's Chroma vector store and combines it with BM25
    retrieval for hybrid search capabilities.
    """

    def __init__(self, collection_name: str = "arxiv_papers"):
        """
        Initialize the ChromaDB hybrid store with LangChain.

        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name

        # Get embeddings
        embedder = get_embedder()
        self.langchain_embeddings = embedder.get_langchain_embeddings()

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_DIR,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Initialize LangChain Chroma vector store
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.langchain_embeddings,
            collection_metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:M": 16,
            },
        )

        # BM25 retriever (will be populated when documents are added)
        self.bm25_retriever: Optional[BM25Retriever] = None
        self._documents: list[Document] = []

        # Try to load existing documents for BM25
        self._load_documents_for_bm25()

        logger.info(
            "Initialized ChromaHybridStore with LangChain",
            collection=collection_name,
            documents=self.count(),
        )

    def _load_documents_for_bm25(self) -> None:
        """Load existing documents from Chroma to initialize BM25."""
        try:
            # Get all documents from the collection
            collection = self.client.get_collection(self.collection_name)
            results = collection.get(include=["documents", "metadatas"])

            if results["documents"]:
                # Convert to LangChain Document objects
                self._documents = [
                    Document(
                        page_content=doc,
                        metadata=meta or {}
                    )
                    for doc, meta in zip(results["documents"], results["metadatas"])
                ]

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
            vectors: NumPy array of embeddings (n, dim) - not used with LangChain (it computes them)
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

        # Add to Chroma using LangChain (it will compute embeddings automatically)
        self.vectorstore.add_documents(documents=documents, ids=chunk_ids)

        # Update documents list and rebuild BM25
        self._documents.extend(documents)
        self.bm25_retriever = BM25Retriever.from_documents(self._documents)

        logger.info(
            "Added documents to ChromaHybridStore",
            count=len(chunk_ids),
            total=self.count(),
        )

    def _semantic_search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[tuple[str, float, str, dict]]:
        """
        Semantic search using LangChain Chroma.

        Note: This method is kept for backward compatibility; prefer search() for hybrid search.

        Returns:
            List of (chunk_id, score, text, metadata) tuples
        """
        # LangChain Chroma doesn't directly accept query vectors, so we use similarity_search_by_vector
        # But first we need the query text for proper usage with LangChain
        logger.warning("_semantic_search called directly - prefer search() for hybrid search")

        # This is a fallback implementation
        results = self.vectorstore._collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(top_k, self.count()),
            where=self._build_where_clause(filters) if filters else None,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][idx]
                distance = results["distances"][0][idx]
                score = 1.0 - distance  # Convert distance to similarity
                text = results["documents"][0][idx]
                meta = results["metadatas"][0][idx]
                output.append((chunk_id, score, text, meta))

        return output

    def _build_where_clause(self, filters: dict) -> dict:
        """
        Convert filter dict to ChromaDB where clause.

        Args:
            filters: Filter dictionary

        Returns:
            ChromaDB where clause
        """
        where_conditions = []

        for key, value in filters.items():
            if key == "published_after":
                where_conditions.append({f"{key}": {"$gte": value}})
            elif key == "published_before":
                where_conditions.append({f"{key}": {"$lte": value}})
            elif key == "authors" or key == "tags":
                if isinstance(value, list):
                    where_conditions.append({f"{key}": {"$in": value}})
                else:
                    where_conditions.append({f"{key}": {"$eq": value}})
            else:
                where_conditions.append({f"{key}": {"$eq": value}})

        if len(where_conditions) == 1:
            return where_conditions[0]
        elif len(where_conditions) > 1:
            return {"$and": where_conditions}

        return {}

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
            query_vec: Query embedding vector (not used with LangChain, kept for compatibility)
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

        # Direct ChromaDB query with actual similarity scores
        collection = self.client.get_collection(self.collection_name)

        # Get embedder for query
        embedder = get_embedder()
        query_embedding = embedder.embed_query(query_text)

        # Query ChromaDB with actual similarity search
        chroma_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieval_k,
            where=self._build_where_clause(filters) if filters else None,
            include=["documents", "metadatas", "distances"]
        )

        if not chroma_results["documents"] or not chroma_results["documents"][0]:
            return []

        # Process semantic results with actual similarity scores
        semantic_results = []
        for i, (doc, meta, distance) in enumerate(zip(
            chroma_results["documents"][0],
            chroma_results["metadatas"][0],
            chroma_results["distances"][0]
        )):
            # Convert distance to similarity (cosine distance to similarity)
            similarity = 1.0 - distance
            semantic_results.append({
                "text": doc,
                "meta": meta,
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
                    # Use content hash as key
                    content_key = doc.page_content[:100]
                    bm25_lookup[content_key] = {
                        "text": doc.page_content,
                        "meta": doc.metadata,
                        "bm25_score": 1.0 - (i / len(bm25_docs)),  # Normalized rank score
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
        collection = self.client.get_collection(self.collection_name)
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

            chroma_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=retrieval_k,
                where=self._build_where_clause(filters) if filters else None,
                include=["documents", "metadatas", "distances"]
            )

            if chroma_results["documents"] and chroma_results["documents"][0]:
                for doc, meta, distance in zip(
                    chroma_results["documents"][0],
                    chroma_results["metadatas"][0],
                    chroma_results["distances"][0]
                ):
                    similarity = 1.0 - distance
                    all_results.append({
                        "text": doc,
                        "meta": meta,
                        "score": similarity,  # Will be replaced by RRF score
                        "query_used": variant_query,
                    })

        # If BM25 is available, add BM25 results to ensemble
        if self.bm25_retriever and all_results:
            self.bm25_retriever.k = retrieval_k
            try:
                # Use the original query for BM25 (works better with exact terms)
                bm25_docs = self.bm25_retriever.invoke(query_text)
                bm25_results = []
                for i, doc in enumerate(bm25_docs):
                    bm25_results.append({
                        "text": doc.page_content,
                        "meta": doc.metadata,
                        "score": 1.0 - (i / len(bm25_docs)),  # Normalized rank score
                        "query_used": "bm25",
                    })
                all_results.extend(bm25_results)
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

        Note: Returns semantic retriever only as EnsembleRetriever is not available.
        Use the search() method for true hybrid search (semantic + BM25).

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
        return self.vectorstore._collection.count()

    def reset(self) -> None:
        """Delete all documents from the collection."""
        # Delete the collection via the client
        try:
            self.client.delete_collection(self.collection_name)
            # Recreate the collection
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.langchain_embeddings,
                collection_metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                },
            )
            self._documents = []
            self.bm25_retriever = None
            logger.info("Reset ChromaHybridStore")
        except Exception as e:
            logger.error("Failed to reset ChromaHybridStore", error=str(e))

    def delete_collection(self) -> None:
        """Permanently delete the collection."""
        self.client.delete_collection(self.collection_name)
        self._documents = []
        self.bm25_retriever = None
        logger.info("Deleted collection", name=self.collection_name)
