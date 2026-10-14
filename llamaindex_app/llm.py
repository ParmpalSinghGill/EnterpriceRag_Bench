"""LlamaIndex Custom LLM Adapter for Nemotron-3.5 and OpenCode API.

Provides seamless integration with LlamaIndex ResponseSynthesizer and ReActAgent.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional, Sequence

from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)

from generation.opencode_client import OpenCodeClient

DEFAULT_MODEL = os.environ.get(
    "ERAG_LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"
)
DEFAULT_TIMEOUT = int(os.environ.get("ERAG_LLM_TIMEOUT", "180"))


class NemotronLLM(CustomLLM):
    """Custom LlamaIndex LLM adapter for Redwood Inference / OpenCode API."""

    context_window: int = 128000
    num_output: int = 3072
    model_name: str = DEFAULT_MODEL
    temperature: float = 0.0

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ):
        model_name = model or DEFAULT_MODEL
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            **kwargs,
        )
        self._client = OpenCodeClient(model=model_name, timeout=timeout)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=True,
        )

    @staticmethod
    def _clean_reasoning(answer: str) -> str:
        """Strip raw thinking traces or chain-of-thought preambles from model output."""
        if "Here's a thinking process" in answer or "Let me analyze" in answer:
            parts = re.split(r"\n\n(?=[A-Z0-9\"'•\-])", answer)
            if len(parts) > 1:
                non_thinking = [
                    p for p in parts
                    if not p.strip().startswith((
                        "Here's a thinking", "Let me analyze", "1. ", "2. ", "3. ", "4. ", "5. ",
                        "**Analyze", "**Identify", "**Scan", "**Synthesize", "**Draft", "**Conclusion"
                    ))
                ]
                if non_thinking:
                    answer = "\n\n".join(non_thinking).strip()
            m_ans = re.findall(
                r"(?:Answer|Conclusion|Summary|The answer is|Based on the documents):\s*(.+)",
                answer,
                re.DOTALL | re.IGNORECASE,
            )
            if m_ans:
                answer = m_ans[-1].strip()
        return answer.strip()

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        system = kwargs.get(
            "system_prompt",
            "You are an accurate, factual AI assistant answering employee questions for Redwood Inference using internal enterprise documents. Answer using ONLY facts mentioned in the provided context. If information is missing, state: 'The requested information was not found in the available documents.'",
        )
        temp = kwargs.get("temperature", self.temperature)
        max_tok = kwargs.get("max_tokens", self.num_output)

        resp = self._client.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=temp,
            max_tokens=max_tok,
        )
        cleaned = self._clean_reasoning(resp)
        return CompletionResponse(text=cleaned)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        resp = self.complete(prompt, **kwargs)
        yield resp

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        system = "You are a helpful assistant answering enterprise questions."
        user_parts = []
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                system = m.content
            else:
                user_parts.append(f"{m.role.value}: {m.content}")

        prompt = "\n".join(user_parts)
        resp = self._client.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=self.temperature,
            max_tokens=self.num_output,
        )
        cleaned = self._clean_reasoning(resp)
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=cleaned.strip(),
            )
        )

