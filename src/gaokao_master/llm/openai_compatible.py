"""OpenAI-compatible chat completion adapter.

This module supports OpenAI's official API and any provider that implements
the OpenAI chat-completions protocol through a custom base URL.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Configuration for an OpenAI-compatible model endpoint."""

    api_key: str | None = None
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "OpenAICompatibleConfig":
        """Load configuration from common OpenAI-compatible environment vars."""

        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        return cls(
            api_key=os.getenv("OPENAI_API_KEY") or None,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
            timeout=float(os.getenv("OPENAI_TIMEOUT", "60")),
        )


class OpenAICompatibleLLM:
    """Small, dependency-isolated wrapper around the OpenAI Python SDK."""

    def __init__(self, config: OpenAICompatibleConfig | None = None) -> None:
        self.config = config or OpenAICompatibleConfig.from_env()

    @property
    def is_configured(self) -> bool:
        """Return whether enough credentials exist to call the endpoint."""

        return bool(self.config.api_key and self.config.model)

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2_000,
    ) -> str:
        """Call an OpenAI-compatible chat-completions endpoint."""

        if not self.is_configured:
            raise RuntimeError("OpenAI-compatible LLM is not configured.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for external model calls."
            ) from exc

        client_kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
        }
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            logger.warning("OpenAI-compatible endpoint returned an empty response.")
            return ""
        return content.strip()


class OpenAICompatibleOCR:
    """OCR helper backed by an OpenAI-compatible multimodal model."""

    def __init__(self, config: OpenAICompatibleConfig | None = None) -> None:
        self.config = config or OpenAICompatibleConfig.from_env()

    @property
    def is_configured(self) -> bool:
        return bool(self.config.api_key and self.config.model)

    def ocr_image(
        self,
        image_bytes: bytes,
        *,
        page_number: int,
        total_pages: int,
        max_tokens: int = 3_000,
        mime_type: str = "image/png",
    ) -> str:
        """OCR one rendered PDF page and return Markdown text."""

        if not self.is_configured:
            raise RuntimeError("OpenAI-compatible OCR model is not configured.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for OCR model calls."
            ) from exc

        client_kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
        }
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "When the page contains a non-text figure such as a "
                        "geometry diagram, function graph, chart, or embedded "
                        "image, output a standalone [IMAGE_HERE] line at the "
                        "same position in the Markdown. Do not replace the "
                        "figure with a prose description. "
                        "你是高考资料 OCR 引擎。请忠实识别页面内容，输出 Markdown。"
                        "保留题号、选项、公式、表格、答案和解析。不要编造缺失内容。"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"请识别第 {page_number}/{total_pages} 页。"
                                "只输出该页可见文字的 Markdown，不要额外解释。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            },
                        },
                    ],
                },
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
