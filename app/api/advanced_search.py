"""Advanced search endpoints with hybrid search, multi-query, and reranking."""

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.api.schemas import AskRequest
from app import dependencies as deps
from app.embeddings import get_embedder
from app.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


def get_store():
    """Get the loaded store or raise error."""
    return deps.get_store()


@router.post("/search/hybrid")
async def hybrid_search(req: AskRequest):
    """
    Advanced hybrid search combining BM25 and vector search with RRF.

    Provides improved retrieval by combining dense and sparse representations.
    """
    try:
        from app.hybrid_search import get_hybrid_search_engine

        store = get_store()
        embedder = get_embedder()

        engine = get_hybrid_search_engine(store.vectorstore)

        if not engine._indexed:
            try:
                collection = store.client.get_collection(store.collection_name)
                results = collection.get(include=["documents"])
                texts = results.get("documents", [])
                await engine.index_documents(texts[:1000])
            except Exception as e:
                logger.warning("BM25 indexing failed, using vector search only", error=str(e))

        results = await engine.search(
            query=req.question,
            top_k=req.top_k,
            search_kwargs={"k": req.top_k * 2} if req.filters else None,
        )

        return {
            "query": req.question,
            "method": "hybrid_bm25_vector",
            "results": results[:req.top_k],
        }

    except ImportError:
        raise HTTPException(status_code=501, detail="Hybrid search not available")
    except Exception as e:
        logger.error("Hybrid search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/multi-query")
async def multi_query_search(req: AskRequest):
    """
    Multi-query retrieval with LLM query expansion.

    Generates multiple query variants using LLM and merges results using RRF.
    """
    try:
        from app.multi_query import get_multi_query_retriever
        from app.rag import get_chat_model

        store = get_store()
        embedder = get_embedder()
        llm = get_chat_model()

        retriever = get_multi_query_retriever(llm)

        queries = await retriever.generate_queries(req.question, include_original=True)

        async def retrieve_func(query: str, k: int):
            q_vec = embedder.embed_query(query)
            return store.search(
                q_vec,
                query_text=query,
                top_k=k,
                semantic_weight=settings.SEMANTIC_WEIGHT,
                bm25_weight=settings.BM25_WEIGHT,
            )

        results = await retriever.retrieve(
            queries=queries,
            retriever_func=retrieve_func,
            top_k=req.top_k,
        )

        return {
            "original_query": req.question,
            "expanded_queries": queries,
            "method": "multi_query_rrf",
            "results": results,
        }

    except ImportError:
        raise HTTPException(status_code=501, detail="Multi-query search not available")
    except Exception as e:
        logger.error("Multi-query search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/rerank")
async def reranked_search(req: AskRequest):
    """
    Search with cross-encoder re-ranking for improved precision.

    Uses a cross-encoder model to re-rank search results based on
    query-document relevance.
    """
    try:
        from app.reranking import get_cross_encoder_reranker

        store = get_store()
        embedder = get_embedder()

        q_vec = embedder.embed_query(req.question)
        candidates = store.search(
            q_vec,
            query_text=req.question,
            top_k=req.top_k * 3,
            semantic_weight=settings.SEMANTIC_WEIGHT,
            bm25_weight=settings.BM25_WEIGHT,
        )

        documents = [
            {
                "text": r["text"],
                "title": r["meta"].get("title", ""),
                "link": r["meta"].get("link", ""),
            }
            for r in candidates
        ]

        reranker = get_cross_encoder_reranker()
        reranked = await reranker.rerank(req.question, documents, top_k=req.top_k)

        return {
            "query": req.question,
            "method": "cross_encoder_rerank",
            "candidates_retrieved": len(candidates),
            "results": reranked,
        }

    except ImportError:
        raise HTTPException(status_code=501, detail="Cross-encoder reranking not available")
    except Exception as e:
        logger.error("Reranked search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggest")
async def autocomplete_suggestions(
    q: str,
    limit: int = 10,
):
    """
    Get autocomplete suggestions for a query prefix.

    Args:
        q: Query prefix
        limit: Maximum number of suggestions (default: 10)

    Returns relevant query suggestions based on indexed content and query history.
    """
    """
    Get autocomplete suggestions for a query prefix.

    Returns relevant query suggestions based on indexed content and query history.
    """
    try:
        from app.autocomplete import get_autocompleter, get_trending_queries

        autocompleter = get_autocompleter()

        if not autocompleter._vocabulary:
            try:
                store = get_store()
                collection = store.client.get_collection(store.collection_name)
                results = collection.get(include=["documents"])
                documents = [{"text": doc} for doc in results.get("documents", [])[:1000]]
                autocompleter.index_documents(documents)
            except Exception as e:
                logger.warning("Autocomplete indexing failed", error=str(e))

        suggestions = autocompleter.get_suggestions(q, max_results=limit)

        if not q.strip() and limit > len(suggestions):
            trending = get_trending_queries()
            trending_suggestions = trending.get_trending(limit=limit - len(suggestions))
            suggestions.extend(trending_suggestions)

        return {
            "prefix": q,
            "suggestions": suggestions,
        }

    except ImportError:
        raise HTTPException(status_code=501, detail="Autocomplete not available")
    except Exception as e:
        logger.error("Autocomplete failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
