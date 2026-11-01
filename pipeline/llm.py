"""Small OpenRouter client used by the classification pipeline."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class LLMResponse:
    """Text and provider-reported usage for one completion."""

    content: str
    usage: dict[str, Any]


class OpenRouterClient:
    """OpenAI-compatible OpenRouter client with bounded retry/backoff."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL", "")
        if not self.model:
            raise ValueError("LLM_MODEL is required")

        if client is None:
            resolved_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
            if not resolved_key:
                raise ValueError("OPENROUTER_API_KEY is required")
            # Keep the optional dependency import at construction time so unit
            # tests can use a mocked client without an installed SDK.
            from openai import OpenAI

            client = OpenAI(
                api_key=resolved_key,
                base_url=base_url,
                max_retries=0,
            )

        self._client = client
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep

    def complete(self, messages: Sequence[Mapping[str, str]]) -> LLMResponse:
        """Create one completion, retrying only rate limits and server errors."""
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=0,
                    extra_body={
                        "reasoning": {"enabled": False},
                        "usage": {"include": True},
                    },
                )
                content = response.choices[0].message.content or ""
                return LLMResponse(content=content, usage=_usage_to_dict(response.usage))
            except Exception as exc:
                status = _status_code(exc)
                retryable = status == 429 or (status is not None and 500 <= status < 600)
                if not retryable or attempt >= self._max_retries:
                    raise
                self._sleep(self._backoff_seconds * (2**attempt))

        raise RuntimeError("unreachable")


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        if isinstance(dumped, dict):
            return dumped

    fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cost",
    )
    return {name: getattr(usage, name) for name in fields if hasattr(usage, name)}
