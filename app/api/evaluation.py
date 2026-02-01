"""Evaluation endpoints for RAG system testing and comparison."""

import glob
import json
import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.api.schemas import EvaluateRequest, CompareRequest
from app.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/evaluate")
async def run_evaluation_endpoint(req: EvaluateRequest):
    """
    Run RAGAS evaluation on the RAG system.

    This endpoint evaluates the system using RAGAS metrics:
    - Faithfulness: Answer consistency with sources
    - Answer Relevancy: How well the answer addresses the question
    - Context Precision: Retrieval quality
    - Context Recall: Coverage of relevant information
    - Answer Similarity: Semantic similarity to ground truth
    - Answer Correctness: Factual correctness

    Returns detailed results including performance metrics.
    """
    try:
        from app.evals import run_evaluation_async, EvalConfig

        config = EvalConfig(
            rerank_method=req.rerank_method,
            use_semantic_chunking=settings.USE_SEMANTIC_CHUNKING,
            use_cot_prompting=req.use_cot_prompting,
            top_k=settings.TOP_K,
            retrieval_k=settings.RETRIEVAL_K,
            mmr_lambda=settings.MMR_LAMBDA,
        )

        logger.info("Starting evaluation via API", config=req.dict())
        results = await run_evaluation_async(config, req.test_dataset_path)

        return {
            "status": "success",
            "results": results,
            "message": "Evaluation completed successfully",
        }

    except Exception as e:
        logger.error("Evaluation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate/compare")
async def run_comparative_evaluation_endpoint(req: CompareRequest):
    """
    Run comparative A/B evaluation with different configurations.

    Compares different reranking methods (MMR, Cross-Encoder, None)
    to help you choose the best configuration for your use case.

    Returns a detailed comparison table with RAGAS and performance metrics.
    """
    try:
        from app.evals import run_comparative_evaluation, EvalConfig

        if req.compare_rerank_methods:
            configs = [
                EvalConfig(rerank_method="mmr", use_cot_prompting=True),
                EvalConfig(rerank_method="cross_encoder", use_cot_prompting=True),
                EvalConfig(rerank_method="none", use_cot_prompting=True),
            ]
        else:
            configs = [
                EvalConfig(
                    rerank_method="mmr" if settings.RERANK_USE_MMR else "cross_encoder",
                    use_semantic_chunking=settings.USE_SEMANTIC_CHUNKING,
                    use_cot_prompting=True,
                )
            ]

        logger.info("Starting comparative evaluation via API", num_configs=len(configs))
        results = await run_comparative_evaluation(configs, req.test_dataset_path)

        return {
            "status": "success",
            "comparison": results,
            "message": f"Comparative evaluation completed ({len(configs)} configurations)",
        }

    except Exception as e:
        logger.error("Comparative evaluation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluate/results")
def get_latest_evaluation_results():
    """Get the most recent evaluation results."""
    try:
        pattern = os.path.join(settings.PROCESSED_DIR, "eval_summary_*.json")
        files = glob.glob(pattern)

        if not files:
            raise HTTPException(
                status_code=404,
                detail="No evaluation results found. Run /evaluate first.",
            )

        latest_file = max(files, key=os.path.getctime)

        with open(latest_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        return {
            "status": "success",
            "results": results,
            "file": latest_file,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to load evaluation results", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluate/history")
def get_evaluation_history(limit: int = Query(default=10, ge=1, le=50)):
    """Get history of evaluation runs."""
    try:
        pattern = os.path.join(settings.PROCESSED_DIR, "eval_summary_*.json")
        files = sorted(glob.glob(pattern), key=os.path.getctime, reverse=True)[:limit]

        history = []
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                history.append({
                    "timestamp": data.get("timestamp"),
                    "config": data.get("config"),
                    "ragas_scores": data.get("ragas_scores"),
                    "num_samples": data.get("num_samples"),
                    "file": file_path,
                })

        return {
            "status": "success",
            "count": len(history),
            "history": history,
        }

    except Exception as e:
        logger.error("Failed to load evaluation history", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
