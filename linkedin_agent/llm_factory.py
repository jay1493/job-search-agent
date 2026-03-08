"""
LLM factory utilities.
Supports cloud (Anthropic) and local (Ollama) models with one interface.
"""

import os
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"


def create_chat_model(
    llm_model: Optional[str] = None,
    temperature: float = 0,
    max_tokens: int = 4096,
):
    """
    Create a chat model based on environment configuration.

    Env vars:
    - LLM_PROVIDER: "anthropic" (default) or "ollama"
    - LLM_MODEL: model name override
    - OLLAMA_BASE_URL: optional, defaults to http://localhost:11434
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    model_name = llm_model or os.getenv("LLM_MODEL")

    if provider == "ollama":
        return ChatOllama(
            model=model_name or DEFAULT_OLLAMA_MODEL,
            temperature=temperature,
            num_predict=max_tokens,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    return ChatAnthropic(
        model=model_name or DEFAULT_ANTHROPIC_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
