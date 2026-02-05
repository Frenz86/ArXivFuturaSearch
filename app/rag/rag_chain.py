"""RAG chain creation — native implementation (no LangChain)."""

import httpx

from app.rag.prompts import COT_PROMPT_TEMPLATE, SIMPLE_PROMPT_TEMPLATE
from app.rag.llm import get_llm, llm_generate_async
from app.logging_config import get_logger

logger = get_logger(__name__)


def _format_docs(docs) -> str:
    """Format retrieved documents into a context string."""
    lines = []
    for i, doc in enumerate(docs, start=1):
        if hasattr(doc, "metadata"):
            title = doc.metadata.get("title", "Untitled")
            link = doc.metadata.get("link", "")
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        else:
            title, link, content = "Untitled", "", str(doc)
        lines.append(f"[{i}] {title}\nSOURCE: {link}\nEXCERPT: {content}\n")
    return "\n".join(lines)


def create_rag_chain(retriever, use_cot: bool = True, streaming: bool = False):
    """Create a native RAG chain.

    Returns an async callable that accepts a question string and returns
    the generated answer.  The returned object also exposes an ``ainvoke``
    method for compatibility with code that previously used LangChain chains.

    Args:
        retriever: Object with an ``invoke(question)`` or ``aretrieve(question)``
                   method that returns a list of documents.
        use_cot: Whether to use chain-of-thought prompting.
        streaming: Unused (streaming is handled at the endpoint level).

    Returns:
        A ``_NativeChain`` callable.
    """
    template = COT_PROMPT_TEMPLATE if use_cot else SIMPLE_PROMPT_TEMPLATE

    class _NativeChain:
        async def ainvoke(self, inputs, config=None):
            question = inputs if isinstance(inputs, str) else inputs.get("question", inputs.get("query", ""))
            return await self(question)

        async def __call__(self, question: str) -> str:
            # Retrieve
            if hasattr(retriever, "aretrieve"):
                docs = await retriever.aretrieve(question)
            elif hasattr(retriever, "invoke"):
                docs = retriever.invoke(question)
            else:
                docs = []

            context = _format_docs(docs)
            prompt = template.format(context=context, question=question)
            return await llm_generate_async(prompt)

    return _NativeChain()


async def list_openrouter_models() -> list[dict]:
    """Fetch available models from the OpenRouter API.

    Returns:
        List of ``{"id": ..., "name": ...}`` dicts.
    """
    from app.config import settings

    if settings.LLM_MODE.lower() != "openrouter" or not settings.OPENROUTER_API_KEY:
        return []

    try:
        headers = {"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"}
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{settings.OPENROUTER_BASE_URL}/models",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            return [{"id": m["id"], "name": m.get("name", m["id"])} for m in data.get("data", [])]
    except Exception as e:
        logger.error("Failed to fetch models", error=str(e))
        return []
