"""RAG chain creation using LangChain."""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.rag.prompts import COT_PROMPT_TEMPLATE, SIMPLE_PROMPT_TEMPLATE
from app.rag.llm import get_llm
from app.logging_config import get_logger

logger = get_logger(__name__)


def create_rag_chain(retriever, use_cot: bool = True, streaming: bool = False):
    """
    Create a LangChain RAG chain.

    Args:
        retriever: LangChain retriever object
        use_cot: Whether to use chain-of-thought prompting
        streaming: Whether to enable streaming

    Returns:
        LangChain chain for RAG
    """
    # Select prompt template
    if use_cot:
        prompt = PromptTemplate.from_template(COT_PROMPT_TEMPLATE)
    else:
        prompt = PromptTemplate.from_template(SIMPLE_PROMPT_TEMPLATE)

    # Get LLM
    llm = get_llm(streaming=streaming)

    # Create chain
    def format_docs(docs):
        """Format documents into context string."""
        lines = []
        for i, doc in enumerate(docs, start=1):
            title = doc.metadata.get("title", "Untitled")
            link = doc.metadata.get("link", "")
            lines.append(f"[{i}] {title}\nSOURCE: {link}\nEXCERPT: {doc.page_content}\n")
        return "\n".join(lines)

    chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


async def list_openrouter_models() -> list[dict]:
    """
    Fetch available models from OpenRouter API.

    Returns:
        List of model info dictionaries
    """
    from app.config import settings

    if settings.LLM_MODE.lower() != "openrouter":
        return []

    if not settings.OPENROUTER_API_KEY:
        return []

    try:
        from app.rag.llm import get_llm

        llm = get_llm(streaming=False)
        if hasattr(llm, "client"):
            client = llm.client
            models = await client.models.list()
            return [{"id": m.id, "name": getattr(m, "name", m.id)} for m in models.data]
    except Exception as e:
        logger.error("Failed to fetch models", error=str(e))

    return []
