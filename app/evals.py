"""Advanced evaluation module using RAGAS metrics with comparative analysis."""

import asyncio
import os
import json
import time
from datetime import datetime
from typing import Optional, Literal
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from datasets import Dataset

from app.config import settings
from app.embeddings import get_embedder, get_reranker, maximal_marginal_relevance
from app.rag import build_prompt, llm_generate_async
from app.logging_config import get_logger

# Import ChromaDB vector store
from app.vectorstore_chroma import ChromaHybridStore as VectorStore

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness,
)

logger = get_logger(__name__)

# ChromaDB collection name
CHROMA_COLLECTION = "arxiv_papers"


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    rerank_method: Literal["none", "cross_encoder", "mmr"] = "mmr"
    use_semantic_chunking: bool = True
    use_cot_prompting: bool = True
    top_k: int = 5
    retrieval_k: int = 20
    mmr_lambda: float = 0.5


@dataclass
class EvalResult:
    """Single evaluation result."""
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    retrieval_time_ms: float
    llm_time_ms: float
    total_time_ms: float
    num_retrieved: int
    avg_retrieval_score: float
    config: EvalConfig


def load_store() -> VectorStore:
    """Load the vector store."""
    return VectorStore(collection_name=CHROMA_COLLECTION)


async def rag_answer_async(
    question: str,
    config: EvalConfig,
) -> tuple[str, list[str], dict]:
    """
    Generate a RAG answer with detailed metrics.

    Args:
        question: The question to answer
        config: Evaluation configuration

    Returns:
        Tuple of (answer, contexts, metrics_dict)
    """
    store = load_store()
    embedder = get_embedder()
    reranker = get_reranker() if config.rerank_method == "cross_encoder" else None

    metrics = {}

    # Embed query
    retrieval_start = time.time()
    q_vec = embedder.embed_query(question)

    # Hybrid search
    retrieved = store.search(
        q_vec,
        query_text=question,
        top_k=config.retrieval_k,
        semantic_weight=settings.SEMANTIC_WEIGHT,
        bm25_weight=settings.BM25_WEIGHT,
    )

    metrics["retrieval_time_ms"] = (time.time() - retrieval_start) * 1000
    metrics["num_retrieved"] = len(retrieved)
    metrics["avg_retrieval_score"] = np.mean([r["score"] for r in retrieved]) if retrieved else 0

    # Rerank based on config
    rerank_start = time.time()
    if config.rerank_method == "mmr" and len(retrieved) > config.top_k:
        doc_texts = [r["text"] for r in retrieved]
        doc_embeddings = embedder.embed(doc_texts, show_progress=False)
        retrieved = maximal_marginal_relevance(
            q_vec,
            doc_embeddings,
            retrieved,
            top_k=config.top_k,
            lambda_param=config.mmr_lambda,
        )
        metrics["rerank_method"] = "MMR"
        metrics["rerank_time_ms"] = (time.time() - rerank_start) * 1000

    elif config.rerank_method == "cross_encoder" and reranker and retrieved:
        retrieved = reranker.rerank(question, retrieved, top_k=config.top_k)
        metrics["rerank_method"] = "Cross-Encoder"
        metrics["rerank_time_ms"] = (time.time() - rerank_start) * 1000

    else:
        retrieved = retrieved[:config.top_k]
        metrics["rerank_method"] = "None"
        metrics["rerank_time_ms"] = 0

    # Generate answer
    prompt = build_prompt(question, retrieved, use_cot=config.use_cot_prompting)

    llm_start = time.time()
    answer = await llm_generate_async(prompt)
    metrics["llm_time_ms"] = (time.time() - llm_start) * 1000

    contexts = [r["text"] for r in retrieved]
    metrics["total_time_ms"] = metrics["retrieval_time_ms"] + metrics["rerank_time_ms"] + metrics["llm_time_ms"]

    return answer, contexts, metrics


def load_test_dataset(file_path: Optional[str] = None) -> list[dict]:
    """
    Load evaluation dataset from file or use default.

    Args:
        file_path: Optional path to JSON/CSV file with test questions

    Returns:
        List of test samples with 'question' and 'ground_truth'
    """
    if file_path and os.path.exists(file_path):
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            return df.to_dict("records")

    # Default test set
    return [
        {
            "question": "What are typical failure modes in RAG pipelines?",
            "ground_truth": (
                "Common RAG issues include retrieval mismatch (retrieving irrelevant documents), "
                "missing context (not finding relevant information), hallucinations when evidence is weak, "
                "and context length limitations."
            ),
        },
        {
            "question": "How do researchers evaluate RAG quality?",
            "ground_truth": (
                "RAG evaluation uses multiple dimensions: faithfulness/grounding (answer consistency with sources), "
                "answer relevance (addressing the question), context precision and recall (retrieval quality), "
                "and sometimes human evaluation for nuanced assessment."
            ),
        },
        {
            "question": "What is multimodal retrieval?",
            "ground_truth": (
                "Multimodal retrieval involves searching across different data types like text, images, "
                "audio, and video. It uses techniques like CLIP for cross-modal embeddings to enable "
                "text-to-image search and other cross-modal retrieval tasks."
            ),
        },
        {
            "question": "What are agentic AI systems?",
            "ground_truth": (
                "Agentic AI systems are autonomous agents that can plan, reason, use tools, and take "
                "sequential actions to accomplish complex goals. They often incorporate techniques like "
                "ReAct (reasoning + acting), tool use, and memory mechanisms."
            ),
        },
        {
            "question": "What is chain-of-thought prompting?",
            "ground_truth": (
                "Chain-of-thought (CoT) prompting is a technique where language models are prompted to "
                "show their reasoning steps explicitly before arriving at an answer. This improves "
                "performance on complex reasoning tasks by breaking them into intermediate steps."
            ),
        },
        {
            "question": "How does attention mechanism work in transformers?",
            "ground_truth": (
                "Attention mechanisms in transformers compute weighted relationships between all tokens "
                "in a sequence. They use query, key, and value matrices to determine which tokens should "
                "attend to which others, enabling the model to capture long-range dependencies."
            ),
        },
        {
            "question": "What is semantic chunking?",
            "ground_truth": (
                "Semantic chunking divides text based on semantic boundaries rather than fixed sizes. "
                "It analyzes sentence embeddings and similarity to create coherent chunks that preserve "
                "topical coherence, improving retrieval quality compared to fixed-size chunking."
            ),
        },
        {
            "question": "What are embeddings in machine learning?",
            "ground_truth": (
                "Embeddings are dense vector representations of data (text, images, etc.) in a continuous "
                "space where semantically similar items are positioned close together. They enable machines "
                "to work with high-dimensional data efficiently using techniques like word2vec, BERT, or CLIP."
            ),
        },
    ]


async def run_evaluation_async(
    config: Optional[EvalConfig] = None,
    test_dataset_path: Optional[str] = None,
) -> dict:
    """
    Run RAGAS evaluation with specified configuration.

    Args:
        config: Evaluation configuration (uses defaults if None)
        test_dataset_path: Optional path to test dataset file

    Returns:
        Dictionary with evaluation results
    """
    if config is None:
        config = EvalConfig(
            rerank_method="mmr" if settings.RERANK_USE_MMR else ("cross_encoder" if settings.RERANK_ENABLED else "none"),
            use_semantic_chunking=settings.USE_SEMANTIC_CHUNKING,
            use_cot_prompting=True,
            top_k=settings.TOP_K,
            retrieval_k=settings.RETRIEVAL_K,
            mmr_lambda=settings.MMR_LAMBDA,
        )

    logger.info("Running RAG evaluation", config=asdict(config))
    logger.info("Vector Store", mode=settings.VECTORSTORE_MODE)
    logger.info("LLM Mode", mode=settings.LLM_MODE, model=settings.OPENROUTER_MODEL)

    samples = load_test_dataset(test_dataset_path)
    results: list[EvalResult] = []

    for i, sample in enumerate(samples, 1):
        logger.info(f"Evaluating {i}/{len(samples)}", question=sample["question"][:60])

        try:
            answer, contexts, metrics = await rag_answer_async(sample["question"], config)

            result = EvalResult(
                question=sample["question"],
                answer=answer,
                contexts=contexts,
                ground_truth=sample["ground_truth"],
                retrieval_time_ms=metrics["retrieval_time_ms"],
                llm_time_ms=metrics["llm_time_ms"],
                total_time_ms=metrics["total_time_ms"],
                num_retrieved=metrics["num_retrieved"],
                avg_retrieval_score=metrics["avg_retrieval_score"],
                config=config,
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Evaluation failed for question {i}", error=str(e))
            continue

    if not results:
        raise RuntimeError("No successful evaluations")

    # Prepare dataset for RAGAS
    logger.info("Computing RAGAS metrics...")
    ragas_data = {
        "question": [r.question for r in results],
        "answer": [r.answer for r in results],
        "contexts": [r.contexts for r in results],
        "ground_truth": [r.ground_truth for r in results],
    }
    ds = Dataset.from_dict(ragas_data)

    # Compute RAGAS metrics
    ragas_result = evaluate(
        ds,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_similarity,
            answer_correctness,
        ],
    )

    # Combine with performance metrics
    final_results = {
        "ragas_scores": dict(ragas_result),
        "performance_metrics": {
            "avg_retrieval_time_ms": np.mean([r.retrieval_time_ms for r in results]),
            "avg_llm_time_ms": np.mean([r.llm_time_ms for r in results]),
            "avg_total_time_ms": np.mean([r.total_time_ms for r in results]),
            "avg_num_retrieved": np.mean([r.num_retrieved for r in results]),
            "avg_retrieval_score": np.mean([r.avg_retrieval_score for r in results]),
        },
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(results),
    }

    # Save results
    os.makedirs(settings.PROCESSED_DIR, exist_ok=True)

    # Save detailed results
    results_df = pd.DataFrame([asdict(r) for r in results])
    results_path = os.path.join(
        settings.PROCESSED_DIR,
        f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    results_df.to_csv(results_path, index=False)

    # Save summary
    summary_path = os.path.join(
        settings.PROCESSED_DIR,
        f"eval_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    # Save RAGAS scores separately
    ragas_df = ragas_result.to_pandas()
    ragas_path = os.path.join(settings.PROCESSED_DIR, "ragas_scores_latest.csv")
    ragas_df.to_csv(ragas_path, index=False)

    logger.info("Evaluation complete", results=final_results["ragas_scores"])
    logger.info("Results saved", summary=summary_path, detailed=results_path)

    # Print summary
    print("\n" + "=" * 70)
    print("📈 RAGAS EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n🔧 Configuration:")
    print(f"  - Rerank Method: {config.rerank_method}")
    print(f"  - Semantic Chunking: {config.use_semantic_chunking}")
    print(f"  - CoT Prompting: {config.use_cot_prompting}")
    print(f"  - Top K: {config.top_k}")
    print(f"\n📊 RAGAS Metrics:")
    for metric, score in final_results["ragas_scores"].items():
        print(f"  - {metric}: {score:.4f}")
    print(f"\n⚡ Performance Metrics:")
    for metric, value in final_results["performance_metrics"].items():
        print(f"  - {metric}: {value:.2f}")
    print(f"\n💾 Results saved to:")
    print(f"  - Summary: {summary_path}")
    print(f"  - Detailed: {results_path}")
    print("=" * 70 + "\n")

    return final_results


async def run_comparative_evaluation(
    configs: Optional[list[EvalConfig]] = None,
    test_dataset_path: Optional[str] = None,
) -> dict:
    """
    Run comparative A/B evaluation with multiple configurations.

    Args:
        configs: List of configurations to compare
        test_dataset_path: Optional path to test dataset

    Returns:
        Dictionary with comparative results
    """
    if configs is None:
        # Default comparison: MMR vs Cross-Encoder vs None
        configs = [
            EvalConfig(rerank_method="mmr", use_cot_prompting=True),
            EvalConfig(rerank_method="cross_encoder", use_cot_prompting=True),
            EvalConfig(rerank_method="none", use_cot_prompting=True),
        ]

    logger.info("Running comparative evaluation", num_configs=len(configs))

    all_results = []
    for i, config in enumerate(configs, 1):
        logger.info(f"Evaluating configuration {i}/{len(configs)}")
        result = await run_evaluation_async(config, test_dataset_path)
        result["config_id"] = i
        all_results.append(result)

    # Create comparison table
    comparison = {
        "configurations": [r["config"] for r in all_results],
        "ragas_comparison": {},
        "performance_comparison": {},
    }

    # Compare RAGAS metrics
    for metric in all_results[0]["ragas_scores"].keys():
        comparison["ragas_comparison"][metric] = [
            r["ragas_scores"][metric] for r in all_results
        ]

    # Compare performance metrics
    for metric in all_results[0]["performance_metrics"].keys():
        comparison["performance_comparison"][metric] = [
            r["performance_metrics"][metric] for r in all_results
        ]

    # Save comparison
    comparison_path = os.path.join(
        settings.PROCESSED_DIR,
        f"eval_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 90)
    print("🔬 COMPARATIVE EVALUATION RESULTS")
    print("=" * 90)

    print("\n📊 RAGAS Metrics Comparison:")
    print(f"{'Metric':<30} " + " ".join([f"Config {i+1:>10}" for i in range(len(configs))]))
    print("-" * 90)
    for metric, scores in comparison["ragas_comparison"].items():
        score_str = " ".join([f"{s:>11.4f}" for s in scores])
        print(f"{metric:<30} {score_str}")

    print("\n⚡ Performance Comparison:")
    print(f"{'Metric':<30} " + " ".join([f"Config {i+1:>10}" for i in range(len(configs))]))
    print("-" * 90)
    for metric, values in comparison["performance_comparison"].items():
        value_str = " ".join([f"{v:>11.2f}" for v in values])
        print(f"{metric:<30} {value_str}")

    print(f"\n💾 Comparison saved to: {comparison_path}")
    print("=" * 90 + "\n")

    return {
        "comparison": comparison,
        "individual_results": all_results,
        "comparison_file": comparison_path,
    }


def run_evaluation(
    config: Optional[EvalConfig] = None,
    test_dataset_path: Optional[str] = None,
) -> dict:
    """Sync wrapper for run_evaluation_async."""
    return asyncio.get_event_loop().run_until_complete(
        run_evaluation_async(config, test_dataset_path)
    )


def run_comparative(
    configs: Optional[list[EvalConfig]] = None,
    test_dataset_path: Optional[str] = None,
) -> dict:
    """Sync wrapper for run_comparative_evaluation."""
    return asyncio.get_event_loop().run_until_complete(
        run_comparative_evaluation(configs, test_dataset_path)
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Run comparative evaluation
        run_comparative()
    else:
        # Run single evaluation
        run_evaluation()
