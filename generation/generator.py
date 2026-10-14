"""RAG Answer Generator using retrieved enterprise context.

Formats retrieved document chunks into clean context and prompts the LLM to generate
factual answers with strict abstention on unanswerable questions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from generation.opencode_client import OpenCodeClient

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

SYSTEM_PROMPT = """You are an accurate, factual AI assistant answering employee questions for Redwood Inference using internal enterprise documents.
Follow these strict instructions:
1. Answer the question using ONLY the facts directly mentioned in the provided Context.
2. If the Context does not contain the answer, or if the information is missing, state clearly: "The requested information was not found in the available documents." Do not speculate or make up information.
3. Be concise, direct, and specific (include exact numbers, versions, metric names, dates, or error codes when present in the context).
4. Do not include markdown citations like [Doc 1] inside your answer text.
5. Do NOT output any internal thinking process, analysis, or preamble like "Let me analyze the question...". Start your response directly with the factual answer.
"""


class RAGGenerator:
    """Generates answers to questions using retrieved enterprise context."""

    def __init__(self, client: Optional[OpenCodeClient] = None, model: Optional[str] = None):
        target_model = model or DEFAULT_MODEL
        self.client = client or OpenCodeClient(model=target_model)
        self.model = target_model

    def format_context(self, context_chunks: Sequence[Dict[str, Any]], max_chars: int = 24000) -> str:
        """Format retrieved chunks into a clean context block."""
        parts = []
        total_chars = 0

        for idx, chunk in enumerate(context_chunks):
            header = chunk.get("header") or f"[{chunk.get('source_type', 'doc')}] Document {idx+1}"
            body = chunk.get("text") or chunk.get("body") or ""
            entry = f"--- Document [{idx+1}] {header} ---\n{body.strip()}\n"

            if total_chars + len(entry) > max_chars and parts:
                break
            parts.append(entry)
            total_chars += len(entry)

        return "\n".join(parts) if parts else "No relevant documents found."

    def build_prompt(self, question: str, context_chunks: Sequence[Dict[str, Any]]) -> str:
        """Construct prompt with context and question."""
        context_str = self.format_context(context_chunks)
        prompt = f"""Context:
{context_str}

Question: {question}

Answer:"""
        return prompt

    def generate_answer(
        self,
        question: str,
        context_chunks: Sequence[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 3072,
    ) -> str:
        """Generate answer for a question given its retrieved context chunks."""
        prompt = self.build_prompt(question, context_chunks)
        answer = self.client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # If model outputs reasoning preamble, strip it
        if "Here's a thinking process" in answer or "Let me analyze" in answer:
            parts = re.split(r"\n\n(?=[A-Z0-9\"'•\-])", answer)
            if len(parts) > 1:
                non_thinking = [
                    p for p in parts
                    if not p.strip().startswith(("Here's a thinking", "Let me analyze", "1. ", "2. ", "3. ", "4. ", "5. ", "**Analyze", "**Identify", "**Scan", "**Synthesize", "**Draft", "**Conclusion"))
                ]
                if non_thinking:
                    answer = "\n\n".join(non_thinking).strip()
            # If the model didn't transition to a clean paragraph, search for conclusion/answer marker
            m_ans = re.findall(r"(?:Answer|Conclusion|Summary|The answer is|Based on the documents):\s*(.+)", answer, re.DOTALL | re.IGNORECASE)
            if m_ans:
                answer = m_ans[-1].strip()

        return answer.strip()
