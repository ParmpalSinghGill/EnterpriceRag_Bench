"""Information retrieval and connector routing metrics for EnterpriseRAGBench.

Implements Docs/evaluation_spec.md §4.6 and §4.7:
  - Ranked IR metrics: Recall@K, Precision@K, MRR, MRR@K, Hit@K, Document Recall (unbounded)
  - Connector metrics: Source Accuracy, Source Precision, Source Recall (and @K)
  - Default cutoffs: K in {1, 5, 10, 20}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

DEFAULT_KS = (1, 5, 10, 20)


@dataclass
class QuestionEvaluation:
    question_id: str
    question_type: str
    retrieval_scorable: bool
    source_scorable: bool
    n_expected_docs: int
    retrieved_doc_ids: List[str]
    expected_doc_ids: List[str]
    retrieved_sources: List[str]
    expected_sources: List[str]

    # Retrieval metrics
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    mrr_at_k: Dict[int, float] = field(default_factory=dict)
    hit_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    document_recall: float = 0.0
    first_gold_rank: Optional[int] = None

    # Source metrics
    source_accuracy: float = 0.0
    source_precision: float = 0.0
    source_recall: float = 0.0
    source_accuracy_at_k: Dict[int, float] = field(default_factory=dict)
    source_precision_at_k: Dict[int, float] = field(default_factory=dict)
    source_recall_at_k: Dict[int, float] = field(default_factory=dict)


def compute_question_retrieval_metrics(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    ks: Sequence[int] = DEFAULT_KS,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], float, float, Optional[int]]:
    """Compute ranked IR metrics for a single question.

    Returns:
        (recall_at_k, precision_at_k, mrr_at_k, hit_at_k, mrr, doc_recall, first_gold_rank)
    """
    g_set: Set[str] = set(expected_doc_ids)
    if not g_set:
        return {}, {}, {}, {}, 0.0, 0.0, None

    r_list = list(retrieved_doc_ids)
    n_gold = len(g_set)

    # First gold rank (1-based)
    first_gold_rank: Optional[int] = None
    for idx, doc_id in enumerate(r_list):
        if doc_id in g_set:
            first_gold_rank = idx + 1
            break

    mrr = 1.0 / first_gold_rank if first_gold_rank is not None else 0.0
    doc_recall = len(set(r_list) & g_set) / n_gold

    recall_at_k = {}
    precision_at_k = {}
    mrr_at_k = {}
    hit_at_k = {}

    for k in ks:
        top_k_docs = r_list[:k]
        hits_k = len(set(top_k_docs) & g_set)

        recall_at_k[k] = hits_k / n_gold
        precision_at_k[k] = hits_k / k  # Spec §4.6: denominator is always K
        hit_at_k[k] = 1.0 if hits_k >= 1 else 0.0

        if first_gold_rank is not None and first_gold_rank <= k:
            mrr_at_k[k] = 1.0 / first_gold_rank
        else:
            mrr_at_k[k] = 0.0

    return recall_at_k, precision_at_k, mrr_at_k, hit_at_k, mrr, doc_recall, first_gold_rank


def compute_question_source_metrics(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
    ks: Sequence[int] = DEFAULT_KS,
) -> Tuple[float, float, float, Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Compute connector routing metrics for a single question.

    Returns:
        (source_accuracy, source_precision, source_recall, accuracy_at_k, precision_at_k, recall_at_k)
    """
    s_set: Set[str] = set(expected_sources)
    if not s_set:
        return 0.0, 0.0, 0.0, {}, {}, {}

    p_list = list(retrieved_sources)
    p_set = set(p_list)

    # Unbounded list metrics
    source_acc = 1.0 if p_set == s_set else 0.0
    source_prec = (len(p_set & s_set) / len(p_set)) if p_set else 0.0
    source_rec = len(p_set & s_set) / len(s_set)

    acc_at_k = {}
    prec_at_k = {}
    rec_at_k = {}

    for k in ks:
        p_k = set(p_list[:k])
        acc_at_k[k] = 1.0 if p_k == s_set else 0.0
        prec_at_k[k] = (len(p_k & s_set) / len(p_k)) if p_k else 0.0
        rec_at_k[k] = len(p_k & s_set) / len(s_set)

    return source_acc, source_prec, source_rec, acc_at_k, prec_at_k, rec_at_k


def aggregate_metrics(
    evaluations: Sequence[QuestionEvaluation],
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, float]:
    """Aggregate per-question scores into mean metrics."""
    # Filter scorable questions
    retrieval_set = [e for e in evaluations if e.retrieval_scorable]
    source_set = [e for e in evaluations if e.source_scorable]

    n_ret = len(retrieval_set)
    n_src = len(source_set)

    agg: Dict[str, float] = {
        "n_questions_total": len(evaluations),
        "n_retrieval_scorable": n_ret,
        "n_source_scorable": n_src,
    }

    if n_ret > 0:
        agg["document_recall"] = round(sum(e.document_recall for e in retrieval_set) / n_ret, 4)
        agg["mrr"] = round(sum(e.mrr for e in retrieval_set) / n_ret, 4)

        for k in ks:
            agg[f"recall@{k}"] = round(sum(e.recall_at_k[k] for e in retrieval_set) / n_ret, 4)
            agg[f"precision@{k}"] = round(sum(e.precision_at_k[k] for e in retrieval_set) / n_ret, 4)
            agg[f"mrr@{k}"] = round(sum(e.mrr_at_k[k] for e in retrieval_set) / n_ret, 4)
            agg[f"hit@{k}"] = round(sum(e.hit_at_k[k] for e in retrieval_set) / n_ret, 4)

    if n_src > 0:
        agg["source_accuracy"] = round(sum(e.source_accuracy for e in source_set) / n_src, 4)
        agg["source_precision"] = round(sum(e.source_precision for e in source_set) / n_src, 4)
        agg["source_recall"] = round(sum(e.source_recall for e in source_set) / n_src, 4)

        for k in ks:
            agg[f"source_accuracy@{k}"] = round(sum(e.source_accuracy_at_k[k] for e in source_set) / n_src, 4)
            agg[f"source_precision@{k}"] = round(sum(e.source_precision_at_k[k] for e in source_set) / n_src, 4)
            agg[f"source_recall@{k}"] = round(sum(e.source_recall_at_k[k] for e in source_set) / n_src, 4)

    return agg

