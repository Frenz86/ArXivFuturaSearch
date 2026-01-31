"""Multi-query retrieval using LLM to generate query variants.


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

Improves retrieval by generating multiple perspectives of the original query
and merging results from all variants.
"""

import asyncio
from typing import List, Optional, Dict, Any, Set
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

from app.logging_config import get_logger
from app.config import settings

logger = get_logger(__name__)


# =============================================================================
# QUERY GENERATION PROMPTS
# =============================================================================

QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at analyzing user questions and generating alternative search queries.
Your task is to generate 3-4 different variations of the user's question that:
1. Rephrase the question using different wording
2. Break down complex questions into simpler sub-questions
3. Include relevant domain-specific terminology
4. Consider different perspectives or interpretations

Output only the alternative queries, one per line, without numbering or bullets."""),
    ("user", "Original question: {question}\n\nGenerate 3-4 alternative search queries:")
])

QUERY_TRANSLATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at translating queries between languages and domains.
Given a user question, generate:
1. An English translation (if not in English)
2. A more technical/academic version
3. A more general/simplified version

Output only the translations, one per line, without numbering or bullets."""),
    ("user", "Question: {question}\n\nGenerate translations:")
])


# =============================================================================
# MULTI-QUERY RETRIEVAL
# =============================================================================

class MultiQueryRetriever:
    """
    Multi-query retriever that expands queries using LLM and merges results.

    Benefits:
    - Captures different aspects of the user's intent
    - Handles ambiguity by exploring multiple interpretations
    - Improves recall by finding documents relevant to query variations
    """

    def __init__(
        self,
        llm: BaseChatModel,
        num_queries: int = 4,
        merge_strategy: str = "rrf",
    ):
        """
        Initialize multi-query retriever.

        Args:
            llm: Language model for query generation
            num_queries: Number of query variants to generate
            merge_strategy: How to merge results ("rrf", "weighted", "union")
        """
        self.llm = llm
        self.num_queries = num_queries
        self.merge_strategy = merge_strategy

        # Create query generation chain
        self.query_chain = QUERY_EXPANSION_PROMPT | llm | StrOutputParser()
        self.translation_chain = QUERY_TRANSLATION_PROMPT | llm | StrOutputParser()

    async def generate_queries(
        self,
        question: str,
        include_original: bool = True,
    ) -> List[str]:
        """
        Generate query variants using LLM.

        Args:
            question: Original user question
            include_original: Whether to include the original query

        Returns:
            List of query variants
        """
        try:
            # Generate query variants
            response = await self.query_chain.ainvoke({"question": question})

            # Parse response into individual queries
            variants = [
                line.strip()
                for line in response.strip().split('\n')
                if line.strip() and len(line.strip()) > 3
            ]

            # Limit to num_queries
            variants = variants[:self.num_queries - 1] if include_original else variants[:self.num_queries]

            # Add original query if requested
            if include_original:
                variants = [question] + variants

            logger.info(
                "Generated query variants",
                original=question[:50],
                variants=len(variants),
            )

            return variants

        except Exception as e:
            logger.warning("Query generation failed, using original", error=str(e))
            return [question]

    async def retrieve(
        self,
        queries: List[str],
        retriever_func,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents for multiple queries and merge results.

        Args:
            queries: List of query strings
            retriever_func: Async function to retrieve for a single query
            top_k: Number of results to return

        Returns:
            Merged and ranked results
        """
        # Retrieve results for all queries concurrently
        tasks = [
            retriever_func(query, k=top_k * 2)
            for query in queries
        ]

        try:
            all_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out failed retrievals
            successful_results = [
                results for results in all_results
                if not isinstance(results, Exception)
            ]

            if not successful_results:
                logger.warning("All query retrievals failed")
                return []

            # Merge results based on strategy
            if self.merge_strategy == "rrf":
                merged = self._reciprocal_rank_fusion(successful_results, top_k)
            elif self.merge_strategy == "weighted":
                merged = self._weighted_merge(successful_results, top_k)
            else:  # union
                merged = self._union_merge(successful_results, top_k)

            logger.info(
                "Multi-query retrieval complete",
                queries=len(queries),
                total_results=sum(len(r) for r in successful_results),
                merged=len(merged),
            )

            return merged

        except Exception as e:
            logger.error("Multi-query retrieval failed", error=str(e))
            # Fallback to first query only
            try:
                return await retriever_func(queries[0], k=top_k)
            except Exception as fallback_error:
                logger.error("Fallback retrieval also failed", error=str(fallback_error))
                return []

    def _reciprocal_rank_fusion(
        self,
        result_sets: List[List[Dict[str, Any]]],
        top_k: int,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        Args:
            result_sets: List of result lists from different queries
            top_k: Number of final results
            k: RRF constant

        Returns:
            Merged and ranked results
        """
        rrf_scores = defaultdict(float)
        result_map = {}

        for results in result_sets:
            for rank, result in enumerate(results, start=1):
                # Create document ID from content hash
                doc_id = self._get_doc_id(result)
                rrf_scores[doc_id] += 1 / (k + rank)

                if doc_id not in result_map:
                    result_map[doc_id] = result.copy()
                    result_map[doc_id]["query_variants"] = []
                result_map[doc_id]["query_variants"].append(rank)

        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        # Format results
        merged = []
        for doc_id, score in sorted_results:
            result = result_map[doc_id]
            result["rrf_score"] = score
            result["rank_positions"] = result.get("query_variants", [])
            del result["query_variants"]
            merged.append(result)

        return merged

    def _weighted_merge(
        self,
        result_sets: List[List[Dict[str, Any]]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Merge results with weighted scoring (original query gets highest weight).

        Args:
            result_sets: List of result lists
            top_k: Number of final results

        Returns:
            Merged results
        """
        # Original query (first set) gets weight 1.0, others get 0.5
        weights = [1.0] + [0.5] * (len(result_sets) - 1)

        combined_scores = defaultdict(float)
        result_map = {}

        for results, weight in zip(result_sets, weights):
            for rank, result in enumerate(results):
                doc_id = self._get_doc_id(result)

                # Get normalized score (assuming results are sorted)
                normalized_score = (1.0 / (rank + 1)) * weight
                combined_scores[doc_id] += normalized_score

                if doc_id not in result_map:
                    result_map[doc_id] = result.copy()

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [
            {**result_map[doc_id], "combined_score": score}
            for doc_id, score in sorted_results
        ]

    def _union_merge(
        self,
        result_sets: List[List[Dict[str, Any]]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Simple union merge - deduplicate but don't re-rank.

        Args:
            result_sets: List of result lists
            top_k: Number of final results

        Returns:
            Deduplicated results
        """
        seen_ids: Set[str] = set()
        unique_results = []

        for results in result_sets:
            for result in results:
                doc_id = self._get_doc_id(result)
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_results.append(result)

                    if len(unique_results) >= top_k:
                        break

            if len(unique_results) >= top_k:
                break

        return unique_results[:top_k]

    def _get_doc_id(self, result: Dict[str, Any]) -> str:
        """
        Generate a stable document ID from a result.

        Args:
            result: Search result

        Returns:
            Document ID string
        """
        # Try to get existing ID
        if "chunk_id" in result:
            return result["chunk_id"]
        if "doc_id" in result:
            return result["doc_id"]
        if "id" in result:
            return str(result["id"])

        # Fallback to hash of text content
        text = result.get("text", result.get("content", ""))
        return str(hash(text))


# =============================================================================
# DECOMPOSITION-BASED RETRIEVAL
# =============================================================================

class QueryDecomposer:
    """
    Break down complex questions into simpler sub-questions.

    Useful for multi-part questions that require different retrieval strategies.
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialize query decomposer.

        Args:
            llm: Language model for decomposition
        """
        self.llm = llm

        self.decompose_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at breaking down complex questions into simpler sub-questions.
Given a user question, decompose it into 2-4 simpler questions that, when answered together,
would provide a complete answer to the original question.

Output only the sub-questions, one per line, without numbering or bullets."""),
            ("user", "Question: {question}\n\nDecompose into sub-questions:")
        ])

        self.decompose_chain = self.decompose_prompt | llm | StrOutputParser()

    async def decompose(self, question: str) -> List[str]:
        """
        Decompose a complex question into sub-questions.

        Args:
            question: Complex user question

        Returns:
            List of simpler sub-questions
        """
        try:
            response = await self.decompose_chain.ainvoke({"question": question})

            sub_questions = [
                line.strip()
                for line in response.strip().split('\n')
                if line.strip() and len(line.strip()) > 5
            ][:4]  # Limit to 4 sub-questions

            logger.info(
                "Decomposed question",
                original=question[:50],
                sub_questions=len(sub_questions),
            )

            return sub_questions

        except Exception as e:
            logger.warning("Query decomposition failed", error=str(e))
            return [question]

    async def retrieve_and_combine(
        self,
        question: str,
        retriever_func,
        answer_func,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Decompose question, retrieve for each sub-question, and combine answers.

        Args:
            question: Complex question
            retriever_func: Async retrieval function
            answer_func: Async answer generation function
            top_k: Results per sub-question

        Returns:
            Combined answer with sources
        """
        # Decompose question
        sub_questions = await self.decompose(question)

        # Retrieve for each sub-question
        retrieval_tasks = [
            retriever_func(q, k=top_k)
            for q in sub_questions
        ]

        results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

        # Generate answers for each sub-question
        answer_tasks = [
            answer_func(q, r if not isinstance(r, Exception) else [])
            for q, r in zip(sub_questions, results)
        ]

        answers = await asyncio.gather(*answer_tasks, return_exceptions=True)

        # Format response
        sub_answers = []
        all_sources = []

        for q, a, r in zip(sub_questions, answers, results):
            sub_answers.append({
                "question": q,
                "answer": a if not isinstance(a, Exception) else "Error generating answer",
                "sources": r if not isinstance(r, Exception) else [],
            })
            if not isinstance(r, Exception):
                all_sources.extend(r)

        return {
            "original_question": question,
            "sub_questions": sub_questions,
            "sub_answers": sub_answers,
            "all_sources": all_sources,
        }


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

_multi_query_retriever: Optional[MultiQueryRetriever] = None
_query_decomposer: Optional[QueryDecomposer] = None


def get_multi_query_retriever(llm: BaseChatModel) -> MultiQueryRetriever:
    """Get or create global multi-query retriever."""
    global _multi_query_retriever
    if _multi_query_retriever is None:
        _multi_query_retriever = MultiQueryRetriever(llm=llm)
    return _multi_query_retriever


def get_query_decomposer(llm: BaseChatModel) -> QueryDecomposer:
    """Get or create global query decomposer."""
    global _query_decomposer
    if _query_decomposer is None:
        _query_decomposer = QueryDecomposer(llm=llm)
    return _query_decomposer
