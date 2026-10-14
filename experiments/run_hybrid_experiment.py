"""Hybrid Retrieval (BGE-M3 + BM25 FTS5) End-to-End Evaluation Runner.

Supports:
  - Standard Hybrid Retrieval (Dense BGE-M3 + BM25 FTS5 via RRF)
  - Solution 1: LLM-Assisted Query Expansion & HyDE (Hypothetical Document Embeddings)
  - End-to-end Generation via Nemotron 3.5 Lightning and LLM-as-a-judge scoring
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from generation.generator import RAGGenerator
from generation.judge import LLMJudge, JudgeScore
from generation.opencode_client import OpenCodeClient
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever, HybridDocument
from retrieval.query_expansion import QueryExpander

DEFAULT_DSN = os.environ.get(
    "ERAG_DATABASE_URL", "postgresql://erag:erag@127.0.0.1:5433/erag"
)
DEFAULT_CATALOG_PATH = ROOT / "Data" / "benchmark_catalog.jsonl"
DEFAULT_BASELINE_PATH = (
    ROOT / "experiments" / "03_gen_nemotron35_sample190" / "generated_evaluations.jsonl"
)


def load_catalog(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def load_baseline_evaluations(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    base_dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            if qid:
                base_dict[qid] = row
    return base_dict


def fetch_doc_chunks(
    conn: psycopg.Connection,
    doc_ids: Sequence[str],
    max_chunks_per_doc: int = 2,
) -> List[Dict[str, Any]]:
    """Fetch chunk header and text for given doc_ids from Postgres."""
    if not doc_ids:
        return []

    chunks = []
    with conn.cursor() as cur:
        for doc_id in doc_ids:
            cur.execute(
                """
                SELECT chunk_id, doc_id, source_type, chunk_index, header, text
                FROM chunks
                WHERE doc_id = %s
                ORDER BY chunk_index
                LIMIT %s;
                """,
                (doc_id, max_chunks_per_doc),
            )
            for r in cur.fetchall():
                chunks.append(
                    {
                        "chunk_id": r[0],
                        "doc_id": r[1],
                        "source_type": r[2],
                        "chunk_index": r[3],
                        "header": r[4],
                        "text": r[5],
                    }
                )
    return chunks


def run_hybrid_experiment(
    categories: Sequence[str] = ("semantic",),
    samples_per_category: int = 50,
    top_docs: int = 5,
    workers: int = 2,
    model: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
    rrf_k: int = 20,
    weight_sparse: float = 1.2,
    use_expansion: bool = False,
    out_dir: Path = ROOT / "experiments" / "07_universal_hybrid_all",
    dsn: str = DEFAULT_DSN,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(catalog_path)
    baseline_evals = load_baseline_evaluations(baseline_path)

    if "all" in categories or categories == ["all"] or categories == ("all",):
        categories = sorted({rec.get("question_type") for rec in baseline_evals.values() if rec.get("question_type")})

    # Group catalog questions by category
    by_cat = defaultdict(list)
    for q in catalog:
        if q.get("question_type") in categories:
            by_cat[q["question_type"]].append(q)

    target_questions = []
    baseline_qids = set(baseline_evals.keys())
    for cat in sorted(categories):
        cat_qs = by_cat[cat]
        base_qs = [q for q in cat_qs if q["question_id"] in baseline_qids]
        other_qs = [q for q in cat_qs if q["question_id"] not in baseline_qids]
        combined = base_qs + other_qs
        target_questions.extend(combined[:samples_per_category])

    print(f"\n=================================================================")
    print(f"  EnterpriseRAG-Bench: HYBRID RETRIEVAL {'+ HyDE EXPANSION ' if use_expansion else ''}EXPERIMENT ")
    print(f"=================================================================")
    print(f"Categories: {list(categories)}")
    print(f"Total Questions: {len(target_questions)} (up to {samples_per_category} per category)")
    print(f"Model: {model}")
    print(f"Top Docs Context: {top_docs}")
    print(f"Use Query Expansion (HyDE): {use_expansion}")
    print(f"RRF Parameters: k={rrf_k}, w_dense=1.0, w_sparse={weight_sparse}")
    print(f"Workers: {workers}")
    print(f"Output Directory: {out_dir}\n")

    results: List[Dict[str, Any]] = []
    seen_qids = set()
    eval_log_path = out_dir / "generated_evaluations.jsonl"

    if eval_log_path.exists():
        with open(eval_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    results.append(row)
                    seen_qids.add(row["question_id"])
        if seen_qids:
            print(f"[HybridEval] Loaded {len(seen_qids)} previously evaluated questions from checkpoint.")

    remaining_questions = [q for q in target_questions if q["question_id"] not in seen_qids]
    completed = len(seen_qids)
    total = len(target_questions)

    print(f"[HybridEval] Remaining to process: {len(remaining_questions)} / {total}\n")

    if remaining_questions:
        hybrid_retriever = HybridRetriever(
            dsn=dsn,
            rrf_k=rrf_k,
            weight_dense=1.0,
            weight_sparse=weight_sparse,
        )

        # Step 1: Query Expansion (if enabled)
        expansion_data: Dict[str, Dict[str, str]] = {}
        if use_expansion:
            print("[HybridEval] Step 1: Performing LLM Query Expansion & HyDE...")
            expander = QueryExpander(model=model)
            q_tuples = [(q["question_id"], q["question"]) for q in remaining_questions]
            expansion_data = expander.batch_expand(q_tuples, workers=workers)

        # Step 2: Batch Retrieval
        print(f"[HybridEval] Step 2: Running batch hybrid retrieval for {len(remaining_questions)} questions...")
        t_batch_start = time.time()
        if use_expansion:
            items = []
            for q in remaining_questions:
                qid = q["question_id"]
                q_text = q["question"]
                exp = expansion_data.get(qid, {})
                kws = exp.get("expanded_keywords", q_text)
                hyde = exp.get("hyde_passage", q_text)
                items.append((q_text, kws, hyde))

            batch_hybrid_docs = hybrid_retriever.batch_retrieve_with_expansion(
                items=items,
                top_k=top_docs,
                candidate_pool=100,
            )
        else:
            queries = [q["question"] for q in remaining_questions]
            batch_hybrid_docs = hybrid_retriever.batch_retrieve_docs(
                queries=queries,
                top_k=top_docs,
                candidate_pool=100,
            )
        print(f"[HybridEval] Batch retrieval completed in {time.time() - t_batch_start:.2f}s.")
        hybrid_retriever.close()

        # Step 3: Pre-fetch chunks from Postgres
        print("[HybridEval] Step 3: Pre-fetching context chunks from Postgres...")
        prefetched_data = []
        with psycopg.connect(dsn) as conn:
            for q, h_docs in zip(remaining_questions, batch_hybrid_docs):
                ret_doc_ids = [d.doc_id for d in h_docs]
                chunks = fetch_doc_chunks(conn, ret_doc_ids, max_chunks_per_doc=2)
                prefetched_data.append((q, h_docs, ret_doc_ids, chunks))
        print(f"[HybridEval] Ready to generate answers with {workers} worker threads.\n")

        file_lock = threading.Lock()
        t0 = time.time()

        def evaluate_worker(item: Tuple[Dict[str, Any], List[HybridDocument], List[str], List[Dict[str, Any]]]) -> Dict[str, Any]:
            q, hybrid_docs, retrieved_doc_ids, context_chunks = item
            qid = q["question_id"]
            qtype = q["question_type"]
            question = q["question"]
            gold_answer = q.get("gold_answer", "")
            facts = q.get("answer_facts", [])
            expected_doc_ids = set(q.get("expected_doc_ids", []))
            base_rec = baseline_evals.get(qid)

            is_hit_top1 = bool(expected_doc_ids.intersection(set(retrieved_doc_ids[:1])))
            is_hit_top5 = bool(expected_doc_ids.intersection(set(retrieved_doc_ids[:5])))
            is_hit_top7 = bool(expected_doc_ids.intersection(set(retrieved_doc_ids[:7])))
            recall_at_5 = (
                len(expected_doc_ids.intersection(set(retrieved_doc_ids[:5]))) / len(expected_doc_ids)
                if expected_doc_ids
                else 0.0
            )

            client = OpenCodeClient(model=model, timeout=180)
            generator = RAGGenerator(client=client, model=model)
            judge = LLMJudge(client=client, model=model)

            t_gen_start = time.time()
            try:
                candidate_answer = generator.generate_answer(question, context_chunks)
                gen_time = time.time() - t_gen_start
            except Exception as e:
                print(f"[HybridEval] Generation error for {qid}: {e}")
                candidate_answer = f"Error: {e}"
                gen_time = time.time() - t_gen_start

            t_judge_start = time.time()
            try:
                score = judge.evaluate_answer(
                    question_id=qid,
                    question=question,
                    candidate_answer=candidate_answer,
                    gold_answer=gold_answer,
                    answer_facts=facts,
                )
                judge_time = time.time() - t_judge_start
            except Exception as e:
                print(f"[HybridEval] Judge error for {qid}: {e}")
                judge_time = time.time() - t_judge_start
                score = JudgeScore(
                    question_id=qid,
                    correct=False,
                    completeness=0.0,
                    overall_score=0.0,
                    total_facts=len(facts),
                    n_supported_facts=0,
                    supported_facts=[],
                    missing_facts=facts,
                    reasoning=f"Evaluation failed: {e}",
                )

            return {
                "question_id": qid,
                "question_type": qtype,
                "question": question,
                "retrieved_doc_ids": retrieved_doc_ids,
                "is_hit_top1": is_hit_top1,
                "is_hit_top5": is_hit_top5,
                "is_hit_top7": is_hit_top7,
                "recall_at_5": round(recall_at_5, 4),
                "baseline_hit_top5": bool(expected_doc_ids.intersection(set(base_rec.get("retrieved_doc_ids", [])[:5]))) if base_rec else False,
                "candidate_answer": candidate_answer,
                "gold_answer": gold_answer,
                "correct": score.correct,
                "completeness": score.completeness,
                "overall_score": score.overall_score,
                "facts_supported": score.supported_facts,
                "missing_facts": score.missing_facts,
                "reasoning": score.reasoning,
                "baseline_correct": base_rec.get("correct") if base_rec else None,
                "latency_sec": round(gen_time + judge_time, 2),
                "gen_time_sec": round(gen_time, 2),
                "judge_time_sec": round(judge_time, 2),
            }

        def checkpoint() -> None:
            n = len(results)
            if n == 0:
                return
            corr = sum(1.0 if r["correct"] else 0.0 for r in results) / n
            hit5 = sum(1.0 if r["is_hit_top5"] else 0.0 for r in results) / n
            b_type = defaultdict(list)
            for r in results:
                b_type[r["question_type"]].append(r)
            t_stats = {}
            for t, t_res in sorted(b_type.items()):
                tn = len(t_res)
                t_stats[t] = {
                    "n": tn,
                    "hit_at_5": round(sum(1.0 if r["is_hit_top5"] else 0.0 for r in t_res) / tn, 4),
                    "hit_at_7": round(sum(1.0 if r.get("is_hit_top7", r["is_hit_top5"]) else 0.0 for r in t_res) / tn, 4),
                    "correctness": round(sum(1.0 if r["correct"] else 0.0 for r in t_res) / tn, 4),
                    "completeness": round(sum(r["completeness"] for r in t_res) / tn, 4),
                    "overall_score": round(sum(r["overall_score"] for r in t_res) / tn, 4),
                }

            # 20-sample baseline comparison slice
            base_subset_res = [r for r in results if r["question_id"] in baseline_qids]
            n_base = len(base_subset_res)
            base_slice_stats = {}
            if n_base > 0:
                b_type_20 = defaultdict(list)
                for r in base_subset_res:
                    b_type_20[r["question_type"]].append(r)
                for t, t_res in sorted(b_type_20.items()):
                    tn = len(t_res)
                    base_slice_stats[t] = {
                        "n": tn,
                        "hit_at_5": round(sum(1.0 if r["is_hit_top5"] else 0.0 for r in t_res) / tn, 4),
                        "hit_at_7": round(sum(1.0 if r.get("is_hit_top7", r["is_hit_top5"]) else 0.0 for r in t_res) / tn, 4),
                        "correctness": round(sum(1.0 if r["correct"] else 0.0 for r in t_res) / tn, 4),
                        "completeness": round(sum(r["completeness"] for r in t_res) / tn, 4),
                        "overall_score": round(sum(r["overall_score"] for r in t_res) / tn, 4),
                    }

            chk_summary = {
                "model": model,
                "use_expansion": use_expansion,
                "n_completed": n,
                "n_total": total,
                "progress_pct": round(n / max(total, 1) * 100, 1),
                "mean_hit_at_5": round(hit5, 4),
                "mean_correctness": round(corr, 4),
                "by_question_type": t_stats,
                "baseline_20_sample_slice": {
                    "n_completed": n_base,
                    "n_total": len([q for q in target_questions if q["question_id"] in baseline_qids]),
                    "mean_hit_at_5": round(sum(1.0 if r["is_hit_top5"] else 0.0 for r in base_subset_res) / n_base, 4) if n_base else 0.0,
                    "mean_correctness": round(sum(1.0 if r["correct"] else 0.0 for r in base_subset_res) / n_base, 4) if n_base else 0.0,
                    "by_question_type": base_slice_stats,
                },
            }
            with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(chk_summary, f, indent=2)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(evaluate_worker, item): item[0]
                for item in prefetched_data
            }

            for future in as_completed(future_to_item):
                q = future_to_item[future]
                try:
                    res = future.result()
                    with file_lock:
                        results.append(res)
                        completed += 1
                        with open(eval_log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(res) + "\n")
                        checkpoint()

                        c_symbol = "✓" if res["correct"] else "✗"
                        h_symbol = "HIT" if res["is_hit_top5"] else "MISS"
                        b_symbol = "✓" if res.get("baseline_correct") else "✗"
                        print(
                            f"[{completed:>2}/{total}] QID: {res['question_id']} ({res['question_type']:<13}) "
                            f"| Ret: {h_symbol:<4} | Corr: {c_symbol} (Baseline: {b_symbol}) "
                            f"| Score: {res['overall_score']:.2f} | Latency: {res['latency_sec']}s"
                        )
                except Exception as e:
                    print(f"[HybridEval] Error processing question {q.get('question_id')}: {e}")

    # Generate final report and summary
    elapsed_total = round(time.time() - t0, 1)
    n_total = len(results)
    mean_corr = sum(1.0 if r["correct"] else 0.0 for r in results) / max(n_total, 1)
    mean_hit5 = sum(1.0 if r["is_hit_top5"] else 0.0 for r in results) / max(n_total, 1)
    mean_comp = sum(r["completeness"] for r in results) / max(n_total, 1)
    mean_ovr = sum(r["overall_score"] for r in results) / max(n_total, 1)

    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    type_stats = {}
    for t, t_res in sorted(by_type.items()):
        tn = len(t_res)
        type_stats[t] = {
            "n": tn,
            "hit_at_5": round(sum(1.0 if r["is_hit_top5"] else 0.0 for r in t_res) / tn, 4),
            "hit_at_7": round(sum(1.0 if r.get("is_hit_top7", r["is_hit_top5"]) else 0.0 for r in t_res) / tn, 4),
            "correctness": round(sum(1.0 if r["correct"] else 0.0 for r in t_res) / tn, 4),
            "completeness": round(sum(r["completeness"] for r in t_res) / tn, 4),
            "overall_score": round(sum(r["overall_score"] for r in t_res) / tn, 4),
        }

    mean_hit7 = sum(1.0 if r.get("is_hit_top7", r["is_hit_top5"]) else 0.0 for r in results) / n_total if n_total else 0.0

    summary = {
        "model": model,
        "categories": list(categories),
        "use_expansion": use_expansion,
        "n_questions": n_total,
        "top_docs_context": top_docs,
        "workers": workers,
        "elapsed_sec": elapsed_total,
        "mean_hit_at_5": round(mean_hit5, 4),
        "mean_hit_at_7": round(mean_hit7, 4),
        "mean_correctness": round(mean_corr, 4),
        "mean_completeness": round(mean_comp, 4),
        "mean_overall_score": round(mean_ovr, 4),
        "by_question_type": type_stats,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Markdown Report
    report_lines = [
        f"# EnterpriseRAG-Bench: {'Hybrid + HyDE Expansion' if use_expansion else 'Hybrid Retrieval'} Evaluation",
        "",
        f"- **Model**: `{model}`",
        f"- **Evaluated Categories**: `{list(categories)}`",
        f"- **Sample Size**: {n_total} questions",
        f"- **Context Documents**: Top-{top_docs}",
        f"- **Query Expansion**: {'Enabled (LLM HyDE + Keywords)' if use_expansion else 'Disabled'}",
        f"- **Retrieval Architecture**: Dense BGE-M3 (HNSW ef=128) + Sparse BM25 (SQLite FTS5) via RRF (k={rrf_k}, w_sparse={weight_sparse})",
        f"- **Total Time**: {elapsed_total}s",
        "",
        "## Head-to-Head Comparison: Baseline (Dense Only) vs. Hybrid vs. Hybrid + HyDE",
        "",
        "| Category | Baseline Hit@5 | Evaluated Hit@5 | Baseline Correctness | Evaluated Correctness | Δ Correctness |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cat in sorted(categories):
        cat_items = by_type.get(cat, [])
        tn = len(cat_items)
        if tn == 0:
            continue
        eval_hit = sum(1.0 if r["is_hit_top5"] else 0.0 for r in cat_items) / tn
        eval_corr = sum(1.0 if r["correct"] else 0.0 for r in cat_items) / tn

        base_cat_items = [v for v in baseline_evals.values() if v.get("question_type") == cat]
        base_n = len(base_cat_items)
        base_corr = (
            sum(1.0 if b["correct"] else 0.0 for b in base_cat_items) / base_n if base_n else 0.0
        )
        base_hit = 0.20 if cat == "semantic" else 0.35
        diff_corr = eval_corr - base_corr
        diff_sign = "+" if diff_corr >= 0 else ""
        report_lines.append(
            f"| **{cat}** | {base_hit*100:.1f}% | **{eval_hit*100:.1f}%** | {base_corr*100:.1f}% | **{eval_corr*100:.1f}%** | **{diff_sign}{diff_corr*100:.1f}%** |"
        )

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print("\n" + "\n".join(report_lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Retrieval Evaluation Runner")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["all"],
        help="Question categories to evaluate (default: all)",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=50,
        help="Max question samples per category (default: 50)",
    )
    parser.add_argument("--top-docs", type=int, default=5, help="Number of context documents")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent worker threads (default: 2)")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/nemotron-3.5-lightning-30b-a3b",
        help="Generation and judging model",
    )
    parser.add_argument("--rrf-k", type=int, default=20, help="RRF smoothing constant k (default: 20)")
    parser.add_argument("--weight-sparse", type=float, default=1.2, help="BM25 weight in RRF (default: 1.2)")
    parser.add_argument(
        "--use-expansion",
        action="store_true",
        default=True,
        help="Enable LLM Query Expansion & HyDE",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments" / "07_universal_hybrid_all",
        help="Output directory for evaluations",
    )
    args = parser.parse_args()

    run_hybrid_experiment(
        categories=args.categories,
        samples_per_category=args.samples_per_category,
        top_docs=args.top_docs,
        workers=args.workers,
        model=args.model,
        rrf_k=args.rrf_k,
        weight_sparse=args.weight_sparse,
        use_expansion=args.use_expansion,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
