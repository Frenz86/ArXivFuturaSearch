"""ChromaDB-based hybrid vector store using native chromadb + BM25."""


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

from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.logging_config import get_logger
from app.embeddings import get_embedder
from app.query_expansion import expand_query, reciprocal_rank_fusion, generate_query_variants

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


class ChromaHybridStore:
    """
    Hybrid vector store using native ChromaDB (semantic) + BM25Okapi (lexical) search.
    """

    def __init__(self, collection_name: str = "arxiv_papers"):
        """
        Initialize the ChromaDB hybrid store.

        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_DIR,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:M": 16,
            },
        )

        # BM25 index (will be populated when documents are loaded)
        self._bm25: Optional[BM25Okapi] = None
        self._doc_texts: list[str] = []
        self._doc_metas: list[dict] = []

        # Try to load existing documents for BM25
        self._load_documents_for_bm25()

        logger.info(
            "Initialized ChromaHybridStore",
            collection=collection_name,
            documents=self.count(),
        )

    def _load_documents_for_bm25(self) -> None:
        """Load existing documents from Chroma to initialize BM25."""
        try:
            results = self.collection.get(include=["documents", "metadatas"])

            if results["documents"]:
                self._doc_texts = results["documents"]
                self._doc_metas = results["metadatas"] or [{}] * len(self._doc_texts)

                # Build BM25 index
                tokenized = [_tokenize(doc) for doc in self._doc_texts]
                if tokenized:
                    self._bm25 = BM25Okapi(tokenized)
                    logger.info("Loaded BM25 index", documents=len(self._doc_texts))
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
            vectors: NumPy array of embeddings (n, dim)
            chunk_ids: List of unique chunk identifiers
            texts: List of text chunks
            metas: List of metadata dictionaries
        """
        if len(chunk_ids) == 0:
            logger.warning("No documents to add")
            return

        # Add to ChromaDB with pre-computed embeddings
        self.collection.add(
            ids=chunk_ids,
            embeddings=vectors.tolist() if isinstance(vectors, np.ndarray) else vectors,
            documents=texts,
            metadatas=metas,
        )

        # Update BM25 index
        self._doc_texts.extend(texts)
        self._doc_metas.extend(metas)
        tokenized = [_tokenize(doc) for doc in self._doc_texts]
        self._bm25 = BM25Okapi(tokenized)

        logger.info(
            "Added documents to ChromaHybridStore",
            count=len(chunk_ids),
            total=self.count(),
        )

    def _build_where_clause(self, filters: dict) -> dict:
        """Convert filter dict to ChromaDB where clause."""
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
            query_vec: Query embedding vector
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

        # Get embedder for query
        embedder = get_embedder()
        query_embedding = embedder.embed_query(query_text)

        # Query ChromaDB directly
        chroma_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
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
            similarity = 1.0 - distance
            semantic_results.append({
                "text": doc,
                "meta": meta,
                "semantic_score": similarity,
                "semantic_rank": i,
            })

        # If BM25 is available, combine results
        bm25_lookup = {}
        if self._bm25 and self._doc_texts:
            try:
                tokenized_query = _tokenize(query_text)
                bm25_scores = self._bm25.get_scores(tokenized_query)

                # Get top BM25 results
                top_indices = np.argsort(bm25_scores)[::-1][:retrieval_k]
                for rank, idx in enumerate(top_indices):
                    if bm25_scores[idx] > 0:
                        content_key = self._doc_texts[idx][:100]
                        bm25_lookup[content_key] = {
                            "bm25_score": 1.0 - (rank / len(top_indices)),
                        }
            except Exception as e:
                logger.warning("BM25 retrieval failed", error=str(e))

        # Combine semantic and BM25 scores
        combined_results = []
        for sem_result in semantic_results:
            content_key = sem_result["text"][:100]
            combined_score = semantic_weight * sem_result["semantic_score"]

            if content_key in bm25_lookup:
                combined_score += bm25_weight * bm25_lookup[content_key]["bm25_score"]

            combined_results.append({
                "score": combined_score,
                "chunk_id": sem_result["meta"].get("chunk_id", f"doc_{sem_result['semantic_rank']}"),
                "text": sem_result["text"],
                "meta": sem_result["meta"],
                "semantic_score": sem_result["semantic_score"],
            })

        combined_results.sort(key=lambda x: x["score"], reverse=True)
        results = combined_results[:top_k]

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

            chroma_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
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
                        "score": similarity,
                        "query_used": variant_query,
                    })

        # If BM25 is available, add BM25 results to ensemble
        if self._bm25 and self._doc_texts and all_results:
            try:
                tokenized_query = _tokenize(query_text)
                bm25_scores = self._bm25.get_scores(tokenized_query)
                top_indices = np.argsort(bm25_scores)[::-1][:retrieval_k]

                for rank, idx in enumerate(top_indices):
                    if bm25_scores[idx] > 0:
                        all_results.append({
                            "text": self._doc_texts[idx],
                            "meta": self._doc_metas[idx] if idx < len(self._doc_metas) else {},
                            "score": 1.0 - (rank / len(top_indices)),
                            "query_used": "bm25",
                        })
            except Exception as e:
                logger.warning("BM25 retrieval failed in ensemble", error=str(e))

        # Apply RRF to combine all results
        if len(all_results) == 0:
            return []

        rrf_results = reciprocal_rank_fusion([all_results], k=60, top_k=top_k)

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
        return self.collection.count()

    def reset(self) -> None:
        """Delete all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                },
            )
            self._doc_texts = []
            self._doc_metas = []
            self._bm25 = None
            logger.info("Reset ChromaHybridStore")
        except Exception as e:
            logger.error("Failed to reset ChromaHybridStore", error=str(e))

    def delete_collection(self) -> None:
        """Permanently delete the collection."""
        self.client.delete_collection(self.collection_name)
        self._doc_texts = []
        self._doc_metas = []
        self._bm25 = None
        logger.info("Deleted collection", name=self.collection_name)
