"""LLM adapters for Gaokao-Master."""

from gaokao_master.llm.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleLLM,
    OpenAICompatibleOCR,
)

__all__ = ["OpenAICompatibleConfig", "OpenAICompatibleLLM", "OpenAICompatibleOCR"]
