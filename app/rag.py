"""RAG logic using LangChain: prompts, LLMs, and retrieval chains."""

from typing import AsyncGenerator, Optional, List
import asyncio

import httpx
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


# Define prompt templates
COT_PROMPT_TEMPLATE = """You are an expert ML research assistant with deep knowledge of machine learning, deep learning, and AI research.

Your task is to answer questions based ONLY on the provided research paper excerpts. Follow this structured approach:

## Instructions:
1. First, identify which sources are most relevant to the question
2. Extract key information from each relevant source
3. Synthesize the information to form a coherent answer
4. Cite sources using [1], [2], etc. throughout your answer
5. If information is insufficient or contradictory, acknowledge this explicitly

## Few-Shot Examples:

**Example 1:**
Question: What are the main benefits of attention mechanisms?
Thinking: Sources [1] and [3] discuss attention mechanisms in detail. [1] mentions computational efficiency, while [3] focuses on interpretability.
Answer: According to the research, attention mechanisms provide two key benefits: First, they enable models to focus on relevant parts of the input, improving computational efficiency [1]. Second, they offer interpretability by revealing which parts of the input the model considers important for its predictions [3].

**Example 2:**
Question: How does batch normalization improve training?
Thinking: Source [2] provides evidence about batch normalization's effects on training stability and convergence.
Answer: Batch normalization improves training primarily by reducing internal covariate shift, which allows for higher learning rates and faster convergence [2]. However, the retrieved papers don't provide specific quantitative results about the magnitude of these improvements.

---

Now answer this question:

Question: {question}

Sources:
{context}

Let's think step by step:
1. Relevant sources:
2. Key information:
3. Synthesis:

Answer:"""


SIMPLE_PROMPT_TEMPLATE = """You are an expert ML research assistant.

Rules:
- Answer ONLY using the provided sources.
- If the sources do not contain enough information, say: "I don't have enough evidence in the retrieved papers to answer that."
- Cite sources like [1], [2] in the answer.
- Be concise but insightful.

Question:
{question}

Sources:
{context}

Answer:"""


def format_context(retrieved: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context string.

    Args:
        retrieved: List of retrieval results with 'text' and 'meta'

    Returns:
        Formatted context string with citations
    """
    lines = []
    for i, r in enumerate(retrieved, start=1):
        title = r["meta"].get("title", "Untitled")
        link = r["meta"].get("link", "")
        lines.append(f"[{i}] {title}\nSOURCE: {link}\nEXCERPT: {r['text']}\n")
    return "\n".join(lines)


def build_prompt(question: str, retrieved: list[dict], use_cot: bool = True) -> str:
    """
    Build the RAG prompt with question and retrieved context.
    (Legacy function for backward compatibility)

    Args:
        question: User's question
        retrieved: List of retrieval results
        use_cot: Whether to use chain-of-thought prompting

    Returns:
        Complete prompt string
    """
    context = format_context(retrieved)

    if use_cot:
        template = PromptTemplate.from_template(COT_PROMPT_TEMPLATE)
    else:
        template = PromptTemplate.from_template(SIMPLE_PROMPT_TEMPLATE)

    return template.format(question=question, context=context)


def get_llm(streaming: bool = False):
    """
    Get LangChain LLM based on configured mode.

    Args:
        streaming: Whether to enable streaming

    Returns:
        LangChain LLM instance
    """
    mode = settings.LLM_MODE.lower()

    if mode == "mock":
        # Return a mock LLM for testing
        from langchain_core.language_models import FakeListLLM
        return FakeListLLM(
            responses=[
                "MOCK ANSWER ✅\n\nI can't generate a real response in mock mode, "
                "but your RAG pipeline is working end-to-end. 📚\n\n"
                "To get real answers, set LLM_MODE=openrouter and add your OPENROUTER_API_KEY."
            ]
        )

    if mode == "openrouter":
        if not settings.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is missing but LLM_MODE=openrouter. "
                "Get your key at https://openrouter.ai/keys"
            )

        return ChatOpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.OPENROUTER_MODEL,
            temperature=0.3,
            max_tokens=1024,
            streaming=streaming,
            timeout=settings.OPENROUTER_TIMEOUT,
            max_retries=settings.OPENROUTER_MAX_RETRIES,
        )

    if mode == "ollama":
        return Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.3,
        )

    raise ValueError(f"Unknown LLM_MODE={settings.LLM_MODE}")


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
    # Format: question -> retrieve docs -> format context -> LLM -> parse output
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


async def llm_generate_async(prompt: str) -> str:
    """
    Generate LLM response asynchronously using LangChain.

    Args:
        prompt: The complete prompt to send to the LLM

    Returns:
        Generated response text
    """
    llm = get_llm(streaming=False)

    logger.info("Calling LLM via LangChain", mode=settings.LLM_MODE)

    try:
        # For LangChain LLMs, we invoke directly
        if hasattr(llm, "ainvoke"):
            response = await llm.ainvoke(prompt)
        else:
            # Fallback to sync invoke in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None, llm.invoke, prompt
            )

        return response if isinstance(response, str) else response.content

    except Exception as e:
        logger.error("LLM generation failed", error=str(e))
        raise


async def llm_generate_stream(prompt: str) -> AsyncGenerator[str, None]:
    """
    Generate LLM response with streaming using LangChain.

    Args:
        prompt: The complete prompt to send to the LLM

    Yields:
        Response text chunks
    """
    mode = settings.LLM_MODE.lower()

    if mode == "mock":
        mock_response = (
            "MOCK ANSWER ✅\n\n"
            "I can't generate a real response in mock mode, but your RAG pipeline "
            "is working end-to-end. 📚"
        )
        for word in mock_response.split():
            yield word + " "
            await asyncio.sleep(0.05)
        return

    llm = get_llm(streaming=True)

    logger.info("Streaming from LLM via LangChain", mode=settings.LLM_MODE)

    try:
        # Use LangChain's astream for streaming
        async for chunk in llm.astream(prompt):
            if hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)

    except Exception as e:
        logger.error("LLM streaming failed", error=str(e))
        yield f"\n\n[Error: {str(e)}]"


def llm_generate(prompt: str) -> str:
    """
    Synchronous wrapper for LLM generation.

    Args:
        prompt: The complete prompt to send to the LLM

    Returns:
        Generated response text
    """
    return asyncio.get_event_loop().run_until_complete(llm_generate_async(prompt))


async def check_llm_health() -> dict:
    """
    Check LLM health (replaces check_openrouter_health).

    Returns:
        Dict with 'healthy' bool and 'details'
    """
    mode = settings.LLM_MODE.lower()

    if mode == "mock":
        return {"healthy": True, "details": "Using mock mode"}

    if mode == "ollama":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                response.raise_for_status()
                return {
                    "healthy": True,
                    "details": f"Ollama connected, model: {settings.OLLAMA_MODEL}",
                }
        except Exception as e:
            logger.error("Ollama health check failed", error=str(e))
            return {"healthy": False, "details": str(e)}

    if mode == "openrouter":
        if not settings.OPENROUTER_API_KEY:
            return {"healthy": False, "details": "OPENROUTER_API_KEY not set"}

        try:
            llm = get_llm(streaming=False)
            # Try a simple invocation
            await llm.ainvoke("test")
            return {
                "healthy": True,
                "details": f"OpenRouter connected, model: {settings.OPENROUTER_MODEL}",
            }
        except Exception as e:
            logger.error("OpenRouter health check failed", error=str(e))
            return {"healthy": False, "details": str(e)}

    return {"healthy": False, "details": f"Unknown LLM_MODE: {mode}"}


# Backward compatibility aliases
check_openrouter_health = check_llm_health


async def list_openrouter_models() -> list[dict]:
    """
    Fetch available models from OpenRouter API.
    (Kept for backward compatibility but may not work with all LLM providers)

    Returns:
        List of model info dictionaries
    """
    if settings.LLM_MODE.lower() != "openrouter":
        return []

    if not settings.OPENROUTER_API_KEY:
        return []

    try:
        # Use LangChain's ChatOpenAI client
        llm = get_llm(streaming=False)
        if hasattr(llm, "client"):
            client = llm.client
            models = await client.models.list()
            return [{"id": m.id, "name": getattr(m, "name", m.id)} for m in models.data]
    except Exception as e:
        logger.error("Failed to fetch models", error=str(e))

    return []
