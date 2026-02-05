"""
Native LLM implementation for OpenRouter API.

This module provides a direct HTTP client for OpenRouter,
replacing LangChain's LLM abstraction with better control and performance.
"""

from typing import List, Dict, Any, Optional, Union, AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import time

import httpx
from pydantic import BaseModel, Field

from app.logging_config import get_logger
from app.config import settings

logger = get_logger(__name__)


# =============================================================================
# MODELS AND ENUMS
# =============================================================================

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    META = "meta"
    MISTRAL = "mistral"
    COHERE = "cohere"


class Role(str, Enum):
    """Message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Chat message."""
    role: Role
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        """Convert to API format."""
        return {"role": self.role.value, "content": self.content}


@dataclass
class CompletionResponse:
    """LLM completion response."""
    content: str
    model: str
    provider: LLMProvider
    finish_reason: Optional[str] = None
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None

    def __repr__(self):
        return f"<CompletionResponse(model={self.model}, tokens={self.tokens_used}, latency={self.latency_ms:.0f}ms)>"


@dataclass
class StreamChunk:
    """Streaming response chunk."""
    delta: str
    finish_reason: Optional[str] = None
    tokens_used: Optional[int] = None


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LLMConfig:
    """LLM configuration."""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "anthropic/claude-3.5-sonnet"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    top_k: int = 40
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0

    # Streaming
    stream: bool = False

    # OpenRouter specific
    provider_preferences: Optional[Dict[str, str]] = None
    include_reasoning: bool = False
    include_usage: bool = True


# =============================================================================
# NATIVE LLM CLIENT
# =============================================================================

class NativeLLM:
    """
    Native LLM client for OpenRouter API.

    Provides direct HTTP access to LLM providers with better performance
    and control than LangChain's abstraction layer.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM client.

        Args:
            config: LLM configuration (defaults to settings)
        """
        if config is None:
            config = LLMConfig(
                api_key=getattr(settings, "OPENROUTER_API_KEY", ""),
                model=getattr(settings, "LLM_MODEL", "anthropic/claude-3.5-sonnet"),
                temperature=getattr(settings, "LLM_TEMPERATURE", 0.7),
                max_tokens=getattr(settings, "LLM_MAX_TOKENS", 4096),
            )

        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

        # Parse provider from model name
        self.provider = self._parse_provider(config.model)

        logger.info(
            "NativeLLM initialized",
            model=config.model,
            provider=self.provider.value,
        )

    @staticmethod
    def _parse_provider(model: str) -> LLMProvider:
        """Parse provider from model name."""
        if model.startswith("anthropic/"):
            return LLMProvider.ANTHROPIC
        elif model.startswith("openai/"):
            return LLMProvider.OPENAI
        elif model.startswith("google/"):
            return LLMProvider.GOOGLE
        elif model.startswith("meta-llama/"):
            return LLMProvider.META
        elif model.startswith("mistral/"):
            return LLMProvider.MISTRAL
        elif model.startswith("cohere/"):
            return LLMProvider.COHERE
        else:
            # Default to anthropic
            return LLMProvider.ANTHROPIC

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "HTTP-Referer": "https://github.com/arxiv-futura-search",
                    "X-Title": "ArXivFuturaSearch",
                },
                timeout=self.config.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_payload(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build API request payload."""
        # Convert messages to API format
        api_messages = []

        # Add system message if provided
        if system:
            api_messages.append({"role": "system", "content": system})

        # Add conversation messages
        for msg in messages:
            api_messages.append(msg.to_dict())

        payload = {
            "model": self.config.model,
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": kwargs.get("stream", self.config.stream),
        }

        # Add optional parameters
        if self.config.include_reasoning:
            payload["include_reasoning"] = True

        if self.config.provider_preferences:
            payload["providers"] = self.config.provider_preferences

        return payload

    async def complete(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Generate a completion.

        Args:
            messages: Conversation messages
            system: Optional system prompt
            **kwargs: Override config values

        Returns:
            CompletionResponse
        """
        payload = self._build_payload(messages, system, **kwargs)
        start_time = time.time()

        # Retry logic
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.post(
                    "/chat/completions",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                latency_ms = (time.time() - start_time) * 1000

                # Parse response
                choice = data["choices"][0]
                message = choice["message"]

                return CompletionResponse(
                    content=message.get("content", ""),
                    model=data.get("model", self.config.model),
                    provider=self.provider,
                    finish_reason=choice.get("finish_reason"),
                    tokens_used=data.get("usage", {}).get("total_tokens"),
                    prompt_tokens=data.get("usage", {}).get("prompt_tokens"),
                    completion_tokens=data.get("usage", {}).get("completion_tokens"),
                    latency_ms=latency_ms,
                    raw_response=data,
                )

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503, 504):
                    # Retry on rate limiting or server errors
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        "Retryable error, waiting",
                        status=e.response.status_code,
                        attempt=attempt + 1,
                        wait=f"{wait_time:.1f}s",
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # Don't retry on client errors
                    break

            except Exception as e:
                last_error = e
                logger.warning("Request failed, retrying", error=str(e), attempt=attempt + 1)
                await asyncio.sleep(self.config.retry_delay)

        # All retries exhausted
        logger.error("Completion failed after retries", error=str(last_error))
        raise last_error or RuntimeError("Completion failed")

    async def stream(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion.

        Args:
            messages: Conversation messages
            system: Optional system prompt
            **kwargs: Override config values

        Yields:
            StreamChunk with incremental content
        """
        payload = self._build_payload(messages, system, stream=True, **kwargs)

        try:
            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")

                        if content:
                            yield StreamChunk(
                                delta=content,
                                finish_reason=data["choices"][0].get("finish_reason"),
                            )

                    except json.JSONDecodeError:
                        logger.warning("Failed to parse stream chunk", data=data_str)

        except Exception as e:
            logger.error("Streaming failed", error=str(e))
            raise

    async def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        history: Optional[List[Message]] = None,
        **kwargs,
    ) -> str:
        """
        Simple chat interface.

        Args:
            prompt: User prompt
            system: Optional system prompt
            history: Optional conversation history
            **kwargs: Additional config

        Returns:
            Response text
        """
        messages = history or []
        messages.append(Message(role=Role.USER, content=prompt))

        response = await self.complete(messages, system=system, **kwargs)
        return response.content


# =============================================================================
# RAG-SPECIFIC LLM
# =============================================================================

class RAGLLM:
    """
    LLM optimized for RAG applications.

    Includes built-in prompt templates for common RAG tasks.
    """

    def __init__(self, llm: Optional[NativeLLM] = None):
        """
        Initialize RAG LLM.

        Args:
            llm: Base LLM instance (creates default if None)
        """
        self.llm = llm or NativeLLM()

    async def answer_question(
        self,
        question: str,
        context: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Answer a question with retrieved context.

        Args:
            question: User question
            context: Retrieved context
            sources: Optional source information
            **kwargs: Additional LLM config

        Returns:
            CompletionResponse
        """
        # Build prompt
        system_prompt = """You are a helpful research assistant. Answer questions based on the provided context from academic papers.

When answering:
- Be accurate and cite your sources
- If the context doesn't contain enough information, say so
- Use clear, concise language
- Format equations and technical terms properly
- Highlight key insights and findings"""

        # Format sources for citation
        source_text = ""
        if sources:
            source_text = "\n\nSources:\n"
            for i, source in enumerate(sources, 1):
                title = source.get("title", "Unknown")
                authors = source.get("authors", [])
                authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
                source_text += f"{i}. {title} - {authors_str}\n"

        user_prompt = f"""Context:
{context}

{source_text}

Question: {question}

Provide a comprehensive answer based on the context above."""

        messages = [Message(role=Role.USER, content=user_prompt)]

        response = await self.llm.complete(messages, system=system_prompt, **kwargs)

        logger.info(
            "RAG question answered",
            question_length=len(question),
            context_length=len(context),
            tokens=response.tokens_used,
            latency=f"{response.latency_ms:.0f}ms",
        )

        return response

    async def summarize_papers(
        self,
        papers: List[Dict[str, Any]],
        max_length: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Summarize a list of papers.

        Args:
            papers: List of paper dictionaries
            max_length: Target summary length
            **kwargs: Additional LLM config

        Returns:
            CompletionResponse
        """
        # Build paper summaries
        paper_texts = []
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Unknown")
            abstract = paper.get("abstract", "No abstract available")
            paper_texts.append(f"{i}. {title}\n{abstract[:500]}...")

        papers_text = "\n\n".join(paper_texts)

        system_prompt = """You are a research assistant. Summarize academic papers concisely and accurately."""

        user_prompt = f"""Summarize the following papers, highlighting:
- Key contributions and findings
- Methodologies used
- Connections between papers
- Important trends or insights

Papers:
{papers_text}"""

        messages = [Message(role=Role.USER, content=user_prompt)]

        return await self.llm.complete(messages, system=system_prompt, **kwargs)

    async def extract_keywords(
        self,
        text: str,
        max_keywords: int = 10,
        **kwargs,
    ) -> List[str]:
        """
        Extract keywords from text.

        Args:
            text: Input text
            max_keywords: Maximum number of keywords
            **kwargs: Additional LLM config

        Returns:
            List of keywords
        """
        system_prompt = "Extract the most relevant keywords or phrases from the given text."

        user_prompt = f"""Extract up to {max_keywords} important keywords from this text. Return only a comma-separated list.

Text: {text}"""

        messages = [Message(role=Role.USER, content=user_prompt)]

        response = await self.llm.complete(
            messages,
            system=system_prompt,
            temperature=0.3,
            **kwargs,
        )

        # Parse keywords
        keywords_text = response.content.strip()
        keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]

        return keywords[:max_keywords]


# =============================================================================
# FACTORY
# =============================================================================

class LLMFactory:
    """Factory for creating LLM instances."""

    _instances: Dict[str, NativeLLM] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def get_llm(
        cls,
        model: Optional[str] = None,
        **kwargs,
    ) -> NativeLLM:
        """
        Get or create an LLM instance.

        Args:
            model: Model name (uses default if None)
            **kwargs: Additional config

        Returns:
            NativeLLM instance
        """
        if model is None:
            model = getattr(settings, "LLM_MODEL", "anthropic/claude-3.5-sonnet")

        if model not in cls._instances:
            async with cls._lock:
                if model not in cls._instances:
                    config = LLMConfig(model=model, **kwargs)
                    cls._instances[model] = NativeLLM(config)

        return cls._instances[model]

    @classmethod
    async def close_all(cls) -> None:
        """Close all LLM instances."""
        for llm in cls._instances.values():
            await llm.close()
        cls._instances.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def complete(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Simple completion function.

    Args:
        prompt: User prompt
        system: Optional system prompt
        model: Model name
        **kwargs: Additional config

    Returns:
        Response text
    """
    llm = await LLMFactory.get_llm(model, **kwargs)
    return await llm.chat(prompt, system=system)


async def stream_complete(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> AsyncIterator[str]:
    """
    Stream completion.

    Args:
        prompt: User prompt
        system: Optional system prompt
        model: Model name
        **kwargs: Additional config

    Yields:
        Response chunks
    """
    llm = await LLMFactory.get_llm(model, **kwargs)
    messages = [Message(role=Role.USER, content=prompt)]

    async for chunk in llm.stream(messages, system=system, **kwargs):
        yield chunk.delta
