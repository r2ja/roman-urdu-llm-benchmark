"""Minimal, dependency-light OpenRouter chat client.

One client serves both the contestant models and the judge. Uses the raw
HTTP API (via ``requests``) so there is no hard dependency on the openai SDK.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class ChatResult:
    text: str
    model: str
    raw: dict
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class OpenRouterClient:
    """Thin wrapper around OpenRouter's chat completions endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        timeout: int = 120,
        max_retries: int = 4,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional attribution headers OpenRouter recommends.
            "HTTP-Referer": os.environ.get("OPENROUTER_APP_URL", ""),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "roman-urdu-llm-benchmark"),
        }

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        **extra,
    ) -> ChatResult:
        """Send a chat completion. Retries with exponential backoff on transient errors."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra,
        }

        backoff = 2.0
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(
                    OPENROUTER_URL,
                    headers=self._headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices")
                    if choices:
                        msg = choices[0].get("message", {}) or {}
                        usage = data.get("usage", {}) or {}
                        return ChatResult(
                            text=msg.get("content") or "",
                            model=model,
                            raw=data,
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                        )
                    # OpenRouter sometimes returns HTTP 200 with an error body
                    # (rate limit, upstream hiccup) and no choices — treat as retryable.
                    last_err = f"200 without choices: {str(data.get('error') or data)[:200]}"
                # 429 / 5xx are retryable; 4xx (except 429) are not.
                if resp.status_code not in (429, 500, 502, 503, 504):
                    return ChatResult(
                        text="", model=model, raw={},
                        error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                    )
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except requests.RequestException as e:  # network hiccup
                last_err = f"request error: {e}"

            if attempt < self.max_retries - 1:
                time.sleep(backoff)
                backoff *= 2

        return ChatResult(text="", model=model, raw={}, error=last_err or "unknown error")

    def complete(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        **kwargs,
    ) -> ChatResult:
        """Convenience: single user prompt (+ optional system) -> ChatResult."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(model, messages, **kwargs)
