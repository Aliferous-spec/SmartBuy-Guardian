"""
Pluggable LLM client — OpenAI-compatible API.

Supports any provider that implements the OpenAI chat-completions protocol:
  - DeepSeek:    https://api.deepseek.com/v1
  - Qwen (通义): https://dashscope.aliyuncs.com/compatible-mode/v1
  - OpenAI:      https://api.openai.com/v1
  - Local (Ollama/vLLM): http://localhost:11434/v1

Configuration (in config.py or env vars):
  LLM_BASE_URL  — API base URL (default: https://api.deepseek.com/v1)
  LLM_API_KEY   — API key
  LLM_MODEL     — Model name  (default: deepseek-chat)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3  # low temperature for analytical, not creative, output


def _load_config() -> dict:
    """Load LLM config from environment, with optional config.py override."""
    config: dict = {
        "base_url": os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        "api_key": os.getenv("LLM_API_KEY", ""),
        "model": os.getenv("LLM_MODEL", DEFAULT_MODEL),
    }

    # Try config.py override
    try:
        import config as cfg_mod

        for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
            val = getattr(cfg_mod, key, None)
            if val:
                config[key.lower().replace("llm_", "")] = val
    except ImportError:
        pass

    return config


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


class LLMClient:
    """Minimal OpenAI-compatible chat client.

    Usage::

        client = LLMClient()
        result = client.chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "分析这段文本..."},
        ])
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        cfg = _load_config()
        self.base_url = base_url or cfg["base_url"]
        self.api_key = api_key or cfg["api_key"]
        self.model = model or cfg["model"]

    @property
    def is_configured(self) -> bool:
        """Return True if the client has an API key set."""
        return bool(self.api_key)

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_format: Optional[dict] = None,
    ) -> Optional[str]:
        """Send a chat-completion request and return the assistant message text.

        Returns None on any failure (network, auth, rate-limit, etc.).
        Callers should always handle the None case gracefully.
        """
        if not self.is_configured:
            logger.warning("LLM API key not configured — skipping AI call")
            return None

        import requests

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            logger.debug("LLM response: %s", content[:200])
            return content
        except requests.exceptions.Timeout:
            logger.error("LLM API 超时（>60s），请检查网络或 API 服务状态")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("LLM API 连接失败，请检查 LLM_BASE_URL 是否正确: %s", url)
            return None
        except requests.exceptions.HTTPError as exc:
            logger.error(
                "LLM API HTTP 错误 %s: %s",
                exc.response.status_code if exc.response else "?",
                exc.response.text[:300] if exc.response else str(exc),
            )
            return None
        except Exception as exc:
            logger.exception("LLM API 未预期错误: %s", exc)
            return None

    def chat_json(
        self,
        messages: list[dict],
        **kwargs,
    ) -> Optional[dict]:
        """Like :meth:`chat`, but parse the response as JSON.

        Returns None if the response is not valid JSON.
        """
        text = self.chat(messages, **kwargs)
        if text is None:
            return None

        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove opening ```json / ``` and closing ```
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("LLM response is not valid JSON: %s", text[:300])
            return None


# Singleton convenience
_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    """Return a shared LLMClient singleton."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
