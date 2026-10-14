"""Generation and LLM Evaluation module for EnterpriseRAGBench.

Integrates with OpenCode API for answer generation and automated LLM-as-a-judge evaluation.
"""

from typing import Any

__all__ = [
    "OpenCodeClient",
    "RAGGenerator",
    "LLMJudge",
]


def __getattr__(name: str) -> Any:
    if name == "OpenCodeClient":
        from generation.opencode_client import OpenCodeClient
        return OpenCodeClient
    if name == "RAGGenerator":
        from generation.generator import RAGGenerator
        return RAGGenerator
    if name == "LLMJudge":
        from generation.judge import LLMJudge
        return LLMJudge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
