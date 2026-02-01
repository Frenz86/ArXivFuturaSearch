"""LLM integration and generation functions."""

from typing import AsyncGenerator
import asyncio

import httpx
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

from app.config import settings
from app.logging_config import get_logger
from app.retry import llm_retry
from app.circuit_breaker import with_circuit_breaker

logger = get_logger(__name__)


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


@llm_retry(max_attempts=3)
@with_circuit_breaker()
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
    Check LLM health.

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
            await llm.ainvoke("test")
            return {
                "healthy": True,
                "details": f"OpenRouter connected, model: {settings.OPENROUTER_MODEL}",
            }
        except Exception as e:
            logger.error("OpenRouter health check failed", error=str(e))
            return {"healthy": False, "details": str(e)}

    return {"healthy": False, "details": f"Unknown LLM_MODE: {mode}"}


# Backward compatibility alias
check_openrouter_health = check_llm_health
