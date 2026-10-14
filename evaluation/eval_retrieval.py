"""Evaluation runner for retrieval benchmarks on EnterpriseRAGBench.

Implements Docs/evaluation_spec.md §4, §6:
  - Ranked IR metrics: Recall@K, Precision@K, MRR@K, Hit@K (K in {1, 5, 10, 20})
  - Source connector metrics: Source Accuracy, Source Precision, Source Recall (and @K)
  - Breakdown by question_type (10 types)
  - Breakdown by slices (single vs multi-doc, multi-hop, multi-source)

Usage:
  # 1. Run dense retrieval on all 500 catalog questions and evaluate:
  python -m evaluation.eval_retrieval --run-retrieval --out-dir experiments/01_dense_bge_m3

  # 2. Evaluate existing predictions / answers file:
  python -m evaluation.eval_retrieval --predictions path/to/answers.jsonl

  # 3. Quick smoke test on 10 questions:
  python -m evaluation.eval_retrieval --run-retrieval --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import (
    DEFAULT_KS,
    QuestionEvaluation,
    aggregate_metrics,
    compute_question_retrieval_metrics,
    compute_question_source_metrics,
)


def load_doc_to_source_mapping(
    docid_map_path: Optional[Path] = None,
    dsn: Optional[str] = None,
) -> Dict[str, str]:
    """Load doc_id -> source_type mapping from docid_map.jsonl or Postgres."""
    if docid_map_path is None:
        docid_map_path = ROOT / "indexes" / "chunk_v1" / "docid_map.jsonl"

    if docid_map_path.exists():
        t0 = time.time()
        mapping = {}
        with open(docid_map_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                did = row["doc_id"]
                if did not in mapping:
                    mapping[did] = row["source_type"]
        print(f"[eval_retrieval] Loaded {len(mapping)} doc->source mappings from {docid_map_path} ({time.time()-t0:.2f}s).")
        return mapping

    # Fallback to Postgres
    dsn = dsn or os.environ.get("ERAG_DATABASE_URL", "postgresql://erag:erag@127.0.0.1:5433/erag")
    print(f"[eval_retrieval] Loading doc->source mapping from Postgres: {dsn}...")
    import psycopg

    mapping = {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT doc_id, source_type FROM documents;")
            for did, st in cur.fetchall():
                mapping[did] = st
    print(f"[eval_retrieval] Loaded {len(mapping)} doc->source mappings from database.")
    return mapping


def load_catalog(catalog_path: Path) -> List[Dict]:
    """Load question catalog rows."""
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    questions = []
    with open(catalog_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def load_predictions(predictions_path: Path) -> Dict[str, List[str]]:
    """Load predictions mapping question_id -> ranked list of doc_ids."""
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    preds: Dict[str, List[str]] = {}
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            doc_ids = row.get("document_ids", [])
            if qid:
                preds[qid] = doc_ids
    return preds


def run_retrieval_predictions(
    catalog: List[Dict],
    top_k: int = 20,
    ef_search: int = 64,
    dsn: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Run DenseRetriever on catalog questions and return question_id -> doc_ids."""
    from retrieval.dense import DenseRetriever

    retriever = DenseRetriever(
        dsn=dsn or os.environ.get("ERAG_DATABASE_URL", "postgresql://erag:erag@127.0.0.1:5433/erag"),
        ef_search=ef_search,
    )

    queries = [q["question"] for q in catalog]
    qids = [q["question_id"] for q in catalog]

    results = retriever.batch_retrieve(queries, top_k=top_k)
    retriever.close()

    predictions: Dict[str, List[str]] = {}
    answers_rows = []

    for qid, res in zip(qids, results):
        doc_ids = res.document_ids
        predictions[qid] = doc_ids
        answers_rows.append(
            {
                "question_id": qid,
                "query": res.query,
                "document_ids": doc_ids,
                "elapsed_sec": round(res.elapsed_sec, 4),
            }
        )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for row in answers_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[eval_retrieval] Saved retrieval predictions to {save_path}.")

    return predictions


def evaluate(
    catalog: List[Dict],
    predictions: Dict[str, List[str]],
    doc_to_source: Dict[str, str],
    ks: Sequence[int] = DEFAULT_KS,
) -> Tuple[Dict[str, float], Dict[str, Dict], Dict[str, Dict], List[QuestionEvaluation]]:
    """Evaluate predictions against catalog.

    Returns:
        (aggregate_metrics, by_question_type, by_slice, question_evaluations)
    """
    evaluations: List[QuestionEvaluation] = []

    for q in catalog:
        qid = q["question_id"]
        qtype = q["question_type"]
        expected_docs = q.get("expected_doc_ids", [])
        expected_sources = q.get("source_types", [])
        ret_scorable = q.get("retrieval_scorable", bool(expected_docs))
        src_scorable = q.get("source_scorable", bool(expected_sources))

        retrieved_docs = predictions.get(qid, [])
        retrieved_sources = [doc_to_source.get(did, "unknown") for did in retrieved_docs]

        q_eval = QuestionEvaluation(
            question_id=qid,
            question_type=qtype,
            retrieval_scorable=ret_scorable,
            source_scorable=src_scorable,
            n_expected_docs=len(expected_docs),
            retrieved_doc_ids=retrieved_docs,
            expected_doc_ids=expected_docs,
            retrieved_sources=retrieved_sources,
            expected_sources=expected_sources,
        )

        if ret_scorable:
            rec_k, prec_k, mrr_k, hit_k, mrr, doc_rec, rank1 = compute_question_retrieval_metrics(
                retrieved_docs, expected_docs, ks=ks
            )
            q_eval.recall_at_k = rec_k
            q_eval.precision_at_k = prec_k
            q_eval.mrr_at_k = mrr_k
            q_eval.hit_at_k = hit_k
            q_eval.mrr = mrr
            q_eval.document_recall = doc_rec
            q_eval.first_gold_rank = rank1

        if src_scorable:
            acc, prec, rec, acc_k, prec_k, rec_k = compute_question_source_metrics(
                retrieved_sources, expected_sources, ks=ks
            )
            q_eval.source_accuracy = acc
            q_eval.source_precision = prec
            q_eval.source_recall = rec
            q_eval.source_accuracy_at_k = acc_k
            q_eval.source_precision_at_k = prec_k
            q_eval.source_recall_at_k = rec_k

        evaluations.append(q_eval)

    # 1. Global aggregate
    agg = aggregate_metrics(evaluations, ks=ks)

    # 2. By question type
    by_type_groups = defaultdict(list)
    for q_eval in evaluations:
        by_type_groups[q_eval.question_type].append(q_eval)

    by_type = {}
    for qtype, group in sorted(by_type_groups.items()):
        by_type[qtype] = aggregate_metrics(group, ks=ks)

    # 3. By slices
    catalog_by_id = {q["question_id"]: q for q in catalog}
    slices = {
        "single_doc": [e for e in evaluations if e.n_expected_docs == 1 and e.retrieval_scorable],
        "multi_doc": [e for e in evaluations if e.n_expected_docs > 1 and e.retrieval_scorable],
        "multi_hop_true": [e for e in evaluations if catalog_by_id[e.question_id].get("multi_hop") and e.retrieval_scorable],
        "multi_hop_false": [e for e in evaluations if not catalog_by_id[e.question_id].get("multi_hop") and e.retrieval_scorable],
        "multi_source_true": [e for e in evaluations if catalog_by_id[e.question_id].get("multi_source") and e.source_scorable],
        "multi_source_false": [e for e in evaluations if not catalog_by_id[e.question_id].get("multi_source") and e.source_scorable],
        "temporal_true": [e for e in evaluations if catalog_by_id[e.question_id].get("temporal") and e.retrieval_scorable],
    }

    by_slice = {}
    for slice_name, group in slices.items():
        if group:
            by_slice[slice_name] = aggregate_metrics(group, ks=ks)

    return agg, by_type, by_slice, evaluations


def format_markdown_tables(
    agg: Dict[str, float],
    by_type: Dict[str, Dict],
    by_slice: Dict[str, Dict],
) -> str:
    """Format evaluation results into clean GitHub-flavored markdown tables."""
    lines = []
    lines.append("# Retrieval Evaluation Results\n")

    # Aggregate IR Table
    lines.append("## 1. Ranked Retrieval Metrics (Aggregated)\n")
    lines.append("| Metric | K=1 | K=5 | K=10 | K=20 | Unbounded |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| **Recall@K** | {agg.get('recall@1', 0):.4f} | {agg.get('recall@5', 0):.4f} | "
        f"{agg.get('recall@10', 0):.4f} | {agg.get('recall@20', 0):.4f} | {agg.get('document_recall', 0):.4f} |"
    )
    lines.append(
        f"| **Precision@K** | {agg.get('precision@1', 0):.4f} | {agg.get('precision@5', 0):.4f} | "
        f"{agg.get('precision@10', 0):.4f} | {agg.get('precision@20', 0):.4f} | — |"
    )
    lines.append(
        f"| **Hit@K** | {agg.get('hit@1', 0):.4f} | {agg.get('hit@5', 0):.4f} | "
        f"{agg.get('hit@10', 0):.4f} | {agg.get('hit@20', 0):.4f} | — |"
    )
    lines.append(
        f"| **MRR@K** | {agg.get('mrr@1', 0):.4f} | {agg.get('mrr@5', 0):.4f} | "
        f"{agg.get('mrr@10', 0):.4f} | {agg.get('mrr@20', 0):.4f} | **{agg.get('mrr', 0):.4f}** (MRR) |"
    )

    # Connector / Routing Table
    lines.append("\n## 2. Connector Routing Metrics\n")
    lines.append("| Metric | @1 | @5 | @10 | @20 | Unbounded |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| **Source Accuracy** | {agg.get('source_accuracy@1', 0):.4f} | {agg.get('source_accuracy@5', 0):.4f} | "
        f"{agg.get('source_accuracy@10', 0):.4f} | {agg.get('source_accuracy@20', 0):.4f} | {agg.get('source_accuracy', 0):.4f} |"
    )
    lines.append(
        f"| **Source Precision** | {agg.get('source_precision@1', 0):.4f} | {agg.get('source_precision@5', 0):.4f} | "
        f"{agg.get('source_precision@10', 0):.4f} | {agg.get('source_precision@20', 0):.4f} | {agg.get('source_precision', 0):.4f} |"
    )
    lines.append(
        f"| **Source Recall** | {agg.get('source_recall@1', 0):.4f} | {agg.get('source_recall@5', 0):.4f} | "
        f"{agg.get('source_recall@10', 0):.4f} | {agg.get('source_recall@20', 0):.4f} | {agg.get('source_recall', 0):.4f} |"
    )

    # Breakdown by Question Type
    lines.append("\n## 3. Breakdown by Question Type\n")
    lines.append("| Question Type | n | Recall@10 | Precision@10 | MRR | Hit@10 | Source Rec@10 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for qtype, m in by_type.items():
        n = m.get("n_retrieval_scorable", 0)
        if n > 0:
            lines.append(
                f"| `{qtype}` | {n} | {m.get('recall@10', 0):.4f} | {m.get('precision@10', 0):.4f} | "
                f"{m.get('mrr', 0):.4f} | {m.get('hit@10', 0):.4f} | {m.get('source_recall@10', 0):.4f} |"
            )
        else:
            lines.append(f"| `{qtype}` | {m.get('n_questions_total', 0)} | *(unscorable)* | — | — | — | — |")

    # Breakdown by Slice
    lines.append("\n## 4. Breakdown by Slices\n")
    lines.append("| Slice | n | Recall@10 | MRR | Hit@10 | Source Rec@10 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for sname, m in by_slice.items():
        n = m.get("n_retrieval_scorable", m.get("n_source_scorable", 0))
        lines.append(
            f"| `{sname}` | {n} | {m.get('recall@10', 0):.4f} | {m.get('mrr', 0):.4f} | "
            f"{m.get('hit@10', 0):.4f} | {m.get('source_recall@10', 0):.4f} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval against EnterpriseRAG-Bench catalog")
    parser.add_argument("--catalog", type=str, default="Data/benchmark_catalog.jsonl")
    parser.add_argument("--predictions", type=str, default=None, help="Existing answers.jsonl file with document_ids")
    parser.add_argument("--run-retrieval", action="store_true", help="Run DenseRetriever on catalog questions")
    parser.add_argument("--top-k", type=int, default=20, help="Retrieval depth top_k")
    parser.add_argument("--ef-search", type=int, default=64, help="HNSW ef_search parameter")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for metrics.json & report")
    parser.add_argument("--dsn", type=str, default=None)
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    catalog = load_catalog(catalog_path)
    if args.limit:
        catalog = catalog[: args.limit]
        print(f"[eval_retrieval] Limited to {len(catalog)} questions.")

    doc_to_source = load_doc_to_source_mapping(dsn=args.dsn)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_retrieval:
        save_path = out_dir / "answers.jsonl" if out_dir else None
        predictions = run_retrieval_predictions(
            catalog=catalog,
            top_k=args.top_k,
            ef_search=args.ef_search,
            dsn=args.dsn,
            save_path=save_path,
        )
    elif args.predictions:
        predictions = load_predictions(Path(args.predictions))
    else:
        default_pred = Path("answer_evaluation/answers.jsonl")
        if default_pred.exists():
            print(f"[eval_retrieval] Loading default predictions from {default_pred}...")
            predictions = load_predictions(default_pred)
        else:
            raise SystemExit("Error: Specify either --predictions PATH or --run-retrieval.")

    agg, by_type, by_slice, q_evals = evaluate(
        catalog=catalog,
        predictions=predictions,
        doc_to_source=doc_to_source,
    )

    report_md = format_markdown_tables(agg, by_type, by_slice)
    print("\n" + report_md + "\n")

    if out_dir:
        metrics_payload = {
            "aggregate": agg,
            "by_question_type": by_type,
            "by_slice": by_slice,
            "n_questions": len(catalog),
            "top_k": args.top_k,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2) + "\n")
        (out_dir / "results.md").write_text(report_md + "\n")

        per_q_data = [
            {
                "question_id": e.question_id,
                "question_type": e.question_type,
                "retrieval_scorable": e.retrieval_scorable,
                "first_gold_rank": e.first_gold_rank,
                "mrr": e.mrr,
                "hit@10": e.hit_at_k.get(10, 0),
                "recall@10": e.recall_at_k.get(10, 0),
                "source_accuracy@10": e.source_accuracy_at_k.get(10, 0),
                "expected_docs": e.expected_doc_ids,
                "retrieved_docs": e.retrieved_doc_ids,
            }
            for e in q_evals
        ]
        (out_dir / "per_question.jsonl").write_text(
            "\n".join(json.dumps(r) for r in per_q_data) + "\n"
        )
        print(f"[eval_retrieval] Saved metrics to {out_dir / 'metrics.json'}")
        print(f"[eval_retrieval] Saved report to {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
