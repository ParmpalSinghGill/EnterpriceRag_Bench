"""API client for text generation and evaluation supporting OpenCode and NVIDIA NIM.

Automatically routes to:
  - NVIDIA NIM (https://integrate.api.nvidia.com/v1/chat/completions) for nvidia/* models
  - OpenCode Zen (https://opencode.ai/zen/v1/chat/completions) for OpenCode models
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

OPENCODE_URL = "https://opencode.ai/zen/v1/chat/completions"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


def load_env_var(key: str) -> Optional[str]:
    """Retrieve an environment variable from env or local .env file."""
    val = os.environ.get(key)
    if val:
        return val.strip()

    for path in [Path(".env"), Path("../.env"), Path(__file__).resolve().parents[1] / ".env"]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{key}="):
                        v = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if v:
                            return v
    return None


def get_api_key_for_model(model: str) -> str:
    """Get the appropriate API key based on the model identifier."""
    if model.startswith("nvidia/"):
        key = load_env_var("NVIDIA_API_KEY")
        if key:
            return key
        raise ValueError("NVIDIA_API_KEY not found in environment or .env file.")

    key = load_env_var("OPENCODE_API_KEY")
    if key:
        return key
    raise ValueError("OPENCODE_API_KEY not found in environment or .env file.")


def get_api_key() -> str:
    """Backward compatibility helper for default API key retrieval."""
    key = load_env_var("NVIDIA_API_KEY") or load_env_var("OPENCODE_API_KEY")
    if key:
        return key
    raise ValueError("Neither NVIDIA_API_KEY nor OPENCODE_API_KEY found.")


class OpenCodeClient:
    """Unified client for OpenAI-compatible APIs (OpenCode Zen and NVIDIA NIM)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        timeout: int = 180,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or get_api_key_for_model(model)

        if base_url:
            self.base_url = base_url
        elif model.startswith("nvidia/"):
            self.base_url = NVIDIA_URL
        else:
            self.base_url = OPENCODE_URL

        self.timeout = timeout
        self.max_retries = max_retries

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "EnterpriseRAGBench/1.0",
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> str:
        """Generate response for a prompt."""
        target_model = model or self.model
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self.headers,
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if "error" in data:
                        raise RuntimeError(f"API error: {data['error']}")
                    choices = data.get("choices", [])
                    if not choices:
                        return "The requested information was not found in the available documents."
                    choice = choices[0]
                    msg = choice.get("message", {})
                    content = msg.get("content")
                    if content is None or str(content).strip() == "":
                        content = choice.get("text") or msg.get("reasoning_content") or ""
                    return str(content).strip()
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                print(f"[LLMClient] Attempt {attempt} HTTP {e.code}: {err_body[:160]}")
                if attempt == self.max_retries:
                    raise RuntimeError(f"API error (HTTP {e.code}): {err_body}") from e
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"[LLMClient] Attempt {attempt} error: {e}")
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)

        raise RuntimeError("Failed to generate response after retries.")
