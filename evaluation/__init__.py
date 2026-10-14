"""Evaluation module for EnterpriseRAGBench.

Implements the evaluation spec in Docs/evaluation_spec.md:
  - Ranked IR metrics (Recall@K, Precision@K, MRR@K, Hit@K)
  - Connector routing metrics (Source Accuracy, Precision, Recall)
  - Question-type and catalog slice breakdowns
"""

from typing import Any

__all__ = [
    "DEFAULT_KS",
    "QuestionEvaluation",
    "compute_question_retrieval_metrics",
    "compute_question_source_metrics",
    "aggregate_metrics",
]


def __getattr__(name: str) -> Any:
    if name in (
        "DEFAULT_KS",
        "QuestionEvaluation",
        "compute_question_retrieval_metrics",
        "compute_question_source_metrics",
        "aggregate_metrics",
    ):
        import evaluation.metrics as m
        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
