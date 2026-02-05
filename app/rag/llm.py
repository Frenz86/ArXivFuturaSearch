"""LLM integration and generation functions.

Native httpx-based implementation for OpenRouter and Ollama.
No LangChain dependency required.
"""

import json as _json
from typing import AsyncGenerator
import asyncio

import httpx

from app.config import settings
from app.logging_config import get_logger
from app.retry import llm_retry
from app.circuit_breaker import with_circuit_breaker

logger = get_logger(__name__)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _openrouter_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def _openrouter_payload(prompt: str, streaming: bool = False) -> dict:
    return {
        "model": settings.OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1024,
        "stream": streaming,
    }


# =============================================================================
# PUBLIC API
# =============================================================================


def get_llm(streaming: bool = False):
    """Return a lightweight LLM wrapper with ainvoke / astream interface.

    Duck-type compatible with LangChain's ChatModel so existing code that
    calls ``llm.ainvoke(prompt)`` or ``async for chunk in llm.astream(...)``
    continues to work without importing LangChain.
    """
    return _SimpleLLM()


@llm_retry(max_attempts=3)
@with_circuit_breaker()
async def llm_generate_async(prompt: str) -> str:
    """Generate LLM response asynchronously.

    Args:
        prompt: The complete prompt to send to the LLM

    Returns:
        Generated response text
    """
    mode = settings.LLM_MODE.lower()
    logger.info("Calling LLM", mode=mode)

    if mode == "mock":
        return (
            "MOCK ANSWER\n\nI can't generate a real response in mock mode, "
            "but your RAG pipeline is working end-to-end.\n\n"
            "To get real answers, set LLM_MODE=openrouter and add your OPENROUTER_API_KEY."
        )

    try:
        if mode == "openrouter":
            if not settings.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY is missing but LLM_MODE=openrouter")
            async with httpx.AsyncClient(timeout=settings.OPENROUTER_TIMEOUT) as client:
                resp = await client.post(
                    f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                    headers=_openrouter_headers(),
                    json=_openrouter_payload(prompt, streaming=False),
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        if mode == "ollama":
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"]

        raise ValueError(f"Unknown LLM_MODE={settings.LLM_MODE}")

    except Exception as e:
        logger.error("LLM generation failed", error=str(e))
        raise


async def llm_generate_stream(prompt: str) -> AsyncGenerator[str, None]:
    """Generate LLM response with streaming.

    Args:
        prompt: The complete prompt to send to the LLM

    Yields:
        Response text chunks
    """
    mode = settings.LLM_MODE.lower()

    if mode == "mock":
        mock_response = (
            "MOCK ANSWER\n\n"
            "I can't generate a real response in mock mode, but your RAG pipeline "
            "is working end-to-end."
        )
        for word in mock_response.split():
            yield word + " "
            await asyncio.sleep(0.05)
        return

    logger.info("Streaming from LLM", mode=mode)

    try:
        if mode == "openrouter":
            if not settings.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY is missing but LLM_MODE=openrouter")
            async with httpx.AsyncClient(timeout=settings.OPENROUTER_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                    headers=_openrouter_headers(),
                    json=_openrouter_payload(prompt, streaming=True),
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            data = _json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                yield delta

        elif mode == "ollama":
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{settings.OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True,
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        data = _json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done"):
                            break

        else:
            raise ValueError(f"Unknown LLM_MODE={settings.LLM_MODE}")

    except Exception as e:
        logger.error("LLM streaming failed", error=str(e))
        yield f"\n\n[Error: {str(e)}]"


def llm_generate(prompt: str) -> str:
    """Synchronous wrapper for LLM generation."""
    return asyncio.get_event_loop().run_until_complete(llm_generate_async(prompt))


async def check_llm_health() -> dict:
    """Check LLM connectivity and return status."""
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
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{settings.OPENROUTER_BASE_URL}/models",
                    headers=_openrouter_headers(),
                )
                response.raise_for_status()
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


# =============================================================================
# DUCK-TYPE LLM WRAPPER
# =============================================================================


class _ChunkProxy:
    """Minimal wrapper so ``chunk.content`` works like a LangChain AIMessageChunk."""
    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


class _SimpleLLM:
    """Lightweight LLM with ainvoke / astream — no LangChain needed.

    Returned by ``get_llm()`` so that existing code calling
    ``llm.ainvoke(prompt)`` or ``async for chunk in llm.astream(prompt)``
    keeps working.
    """

    async def ainvoke(self, prompt) -> str:
        text = prompt if isinstance(prompt, str) else str(prompt)
        return await llm_generate_async(text)

    async def astream(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        async for chunk in llm_generate_stream(text):
            yield _ChunkProxy(chunk)

    def invoke(self, prompt) -> str:
        text = prompt if isinstance(prompt, str) else str(prompt)
        return llm_generate(text)
