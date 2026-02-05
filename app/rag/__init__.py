"""RAG package for prompts, LLM integration, and chain creation."""

from app.rag.prompts import (
    COT_PROMPT_TEMPLATE,
    SIMPLE_PROMPT_TEMPLATE,
    format_context,
    build_prompt,
)
from app.rag.llm import (
    get_llm,
    llm_generate_async,
    llm_generate_stream,
    llm_generate,
    check_llm_health,
    check_openrouter_health,
)

# Alias used by advanced_search.py
get_chat_model = get_llm
from app.rag.rag_chain import (
    create_rag_chain,
    list_openrouter_models,
)

__all__ = [
    # Prompts
    "COT_PROMPT_TEMPLATE",
    "SIMPLE_PROMPT_TEMPLATE",
    "format_context",
    "build_prompt",
    # LLM
    "get_llm",
    "get_chat_model",
    "llm_generate_async",
    "llm_generate_stream",
    "llm_generate",
    "check_llm_health",
    "check_openrouter_health",
    # RAG Chain
    "create_rag_chain",
    "list_openrouter_models",
]
