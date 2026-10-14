"""End-to-end RAG Generation and Evaluation Runner.

Takes retrieved enterprise documents, generates answers via LLM,
and evaluates with LLM-as-a-judge for Correctness, Completeness, and Overall Score.
Supports concurrent multi-threaded execution and stratified category sampling.
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
from typing import Any, Dict, List, Optional, Sequence

import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.generator import RAGGenerator
from generation.judge import LLMJudge, JudgeScore
from generation.opencode_client import OpenCodeClient


def load_catalog(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def load_retrieval_predictions(path: Path) -> Dict[str, List[str]]:
    """Load precomputed retrieval answers.jsonl mapping qid -> doc_ids."""
    if not path.exists():
        return {}
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            doc_ids = row.get("document_ids", [])
            if qid:
                preds[qid] = doc_ids
    return preds


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


def select_stratified_sample(
    catalog: List[Dict[str, Any]],
    samples_per_category: int = 20,
) -> List[Dict[str, Any]]:
    """Select up to `samples_per_category` questions for every question category."""
    by_type = defaultdict(list)
    for q in catalog:
        by_type[q["question_type"]].append(q)

    sample = []
    for cat in sorted(by_type.keys()):
        sample.extend(by_type[cat][:samples_per_category])
    return sample


def evaluate_single_question(
    q: Dict[str, Any],
    retrieval_preds: Dict[str, List[str]],
    dsn: str,
    model: str,
    top_docs: int,
) -> Dict[str, Any]:
    """Execute retrieval, generation, and LLM evaluation for a single question."""
    qid = q["question_id"]
    qtype = q["question_type"]
    question = q["question"]
    gold_answer = q.get("gold_answer", "")
    facts = q.get("answer_facts", [])

    client = OpenCodeClient(model=model, timeout=120)
    generator = RAGGenerator(client=client, model=model)
    judge = LLMJudge(client=client, model=model)

    doc_ids = retrieval_preds.get(qid, [])[:top_docs]

    t_start = time.time()
    try:
        with psycopg.connect(dsn) as conn:
            context_chunks = fetch_doc_chunks(conn, doc_ids)
    except Exception as e:
        context_chunks = []

    t_gen_start = time.time()
    try:
        candidate_answer = generator.generate_answer(question, context_chunks)
        gen_time = time.time() - t_gen_start

        t_judge_start = time.time()
        score = judge.evaluate_answer(
            question_id=qid,
            question=question,
            candidate_answer=candidate_answer,
            gold_answer=gold_answer,
            answer_facts=facts,
        )
        judge_time = time.time() - t_judge_start
    except Exception as e:
        print(f"[GenerationEval] Error processing {qid}: {e}")
        candidate_answer = f"Error: {e}"
        gen_time = time.time() - t_gen_start
        judge_time = 0.0
        score = JudgeScore(
            question_id=qid,
            correct=False,
            completeness=0.0,
            overall_score=0.0,
            total_facts=len(facts),
            n_supported_facts=0,
            supported_facts=[],
            missing_facts=facts,
            reasoning=f"Evaluation failed with exception: {e}",
        )

    latency = round(time.time() - t_start, 2)
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": question,
        "retrieved_doc_ids": doc_ids,
        "candidate_answer": candidate_answer,
        "gold_answer": gold_answer,
        "correct": score.correct,
        "completeness": score.completeness,
        "overall_score": score.overall_score,
        "facts_supported": score.supported_facts,
        "missing_facts": score.missing_facts,
        "reasoning": score.reasoning,
        "latency_sec": latency,
    }


def run_eval(
    catalog: List[Dict[str, Any]],
    retrieval_preds: Dict[str, List[str]],
    dsn: str,
    model: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
    top_docs: int = 5,
    workers: int = 4,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run generation and judge evaluation across questions with multi-threaded concurrency."""
    print(f"\n[GenerationEval] Running RAG generation & evaluation on {len(catalog)} questions...")
    print(f"[GenerationEval] Model: {model} | Top docs per question: {top_docs} | Workers: {workers}\n")

    results: List[Dict[str, Any]] = []
    seen_qids = set()
    eval_log_path = None

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_log_path = out_dir / "generated_evaluations.jsonl"
        if eval_log_path.exists():
            with open(eval_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        results.append(row)
                        seen_qids.add(row["question_id"])
            if seen_qids:
                print(f"[GenerationEval] Loaded {len(seen_qids)} previously evaluated questions from {eval_log_path}")

    # Filter catalog for remaining questions
    remaining_catalog = [q for q in catalog if q["question_id"] not in seen_qids]
    completed = len(seen_qids)
    total = len(catalog)

    print(f"\n[GenerationEval] Progress: {completed}/{total} ({completed/max(total,1)*100:.1f}%) | Remaining: {len(remaining_catalog)}")
    print(f"[GenerationEval] Model: {model} | Top docs: {top_docs} | Workers: {workers}\n")

    file_lock = threading.Lock()
    t0 = time.time()

    def save_checkpoint() -> None:
        n = len(results)
        if n == 0 or not out_dir:
            return
        m_corr = sum(1.0 if r["correct"] else 0.0 for r in results) / n
        m_comp = sum(r["completeness"] for r in results) / n
        m_ovr = sum(r["overall_score"] for r in results) / n

        b_type = defaultdict(list)
        for r in results:
            b_type[r["question_type"]].append(r)

        t_stats = {}
        for t, t_res in sorted(b_type.items()):
            tn = len(t_res)
            t_stats[t] = {
                "n": tn,
                "correctness": round(sum(1.0 if r["correct"] else 0.0 for r in t_res) / tn, 4),
                "completeness": round(sum(r["completeness"] for r in t_res) / tn, 4),
                "overall_score": round(sum(r["overall_score"] for r in t_res) / tn, 4),
            }

        chk_summary = {
            "model": model,
            "n_completed": n,
            "n_total": total,
            "progress_pct": round(n / max(total, 1) * 100, 1),
            "mean_correctness": round(m_corr, 4),
            "mean_completeness": round(m_comp, 4),
            "mean_overall_score": round(m_ovr, 4),
            "by_question_type": t_stats,
        }
        (out_dir / "summary.json").write_text(json.dumps(chk_summary, indent=2) + "\n")

    # If all already done, save checkpoint and return
    if not remaining_catalog:
        save_checkpoint()
        print(f"[GenerationEval] All {total} questions already completed!")
        return json.loads((out_dir / "summary.json").read_text())

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_q = {
            executor.submit(
                evaluate_single_question,
                q=q,
                retrieval_preds=retrieval_preds,
                dsn=dsn,
                model=model,
                top_docs=top_docs,
            ): q
            for q in remaining_catalog
        }

        for future in as_completed(future_to_q):
            res = future.result()
            with file_lock:
                results.append(res)
                completed += 1
                icon = "✅" if res["correct"] else "❌"
                pct = completed / max(total, 1) * 100
                print(
                    f"[{completed:3d}/{total} - {pct:4.1f}%] {res['question_id']} ({res['question_type']}) {icon} "
                    f"Correct: {res['correct']} | Complete: {res['completeness']:.1%} | "
                    f"Score: {res['overall_score']:.2f} ({res['latency_sec']:.1f}s)"
                )
                if eval_log_path:
                    with open(eval_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(res) + "\n")
                save_checkpoint()

    elapsed = time.time() - t0
    n = len(results)
    mean_correct = sum(1.0 if r["correct"] else 0.0 for r in results) / max(n, 1)
    mean_complete = sum(r["completeness"] for r in results) / max(n, 1)
    mean_overall = sum(r["overall_score"] for r in results) / max(n, 1)

    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    type_stats = {}
    for t, t_results in sorted(by_type.items()):
        tn = len(t_results)
        type_stats[t] = {
            "n": tn,
            "correctness": round(sum(1.0 if r["correct"] else 0.0 for r in t_results) / tn, 4),
            "completeness": round(sum(r["completeness"] for r in t_results) / tn, 4),
            "overall_score": round(sum(r["overall_score"] for r in t_results) / tn, 4),
        }

    summary = {
        "model": model,
        "n_questions": n,
        "top_docs_context": top_docs,
        "workers": workers,
        "elapsed_sec": round(elapsed, 1),
        "mean_correctness": round(mean_correct, 4),
        "mean_completeness": round(mean_complete, 4),
        "mean_overall_score": round(mean_overall, 4),
        "by_question_type": type_stats,
    }

    md = []
    md.append("\n# End-to-End RAG Generation Benchmark Results\n")
    md.append(f"**Model**: `{model}` | **Evaluated Questions**: {n} | **Context**: Top-{top_docs} Retrieved Docs\n")
    md.append("## Aggregate Leaderboard Metrics\n")
    md.append("| Metric | Score (%) | Meaning |")
    md.append("|---|---:|---|")
    md.append(f"| **Overall Score** | **{mean_overall*100:.1f}%** | Primary Leaderboard Rank: `mean(Correctness × Completeness)` |")
    md.append(f"| **Answer Correctness** | **{mean_correct*100:.1f}%** | Factual consistency with gold answer (no contradictions) |")
    md.append(f"| **Answer Completeness** | **{mean_complete*100:.1f}%** | Percentage of required gold facts captured |")

    md.append("\n## Breakdown by Question Type\n")
    md.append("| Question Type | n | Correctness | Completeness | Overall Score |")
    md.append("|---|---:|---:|---:|---:|")
    for t, stats in type_stats.items():
        md.append(
            f"| `{t}` | {stats['n']} | {stats['correctness']*100:.1f}% | "
            f"{stats['completeness']*100:.1f}% | **{stats['overall_score']*100:.1f}%** |"
        )

    report_str = "\n".join(md)
    print(report_str)

    if out_dir:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (out_dir / "report.md").write_text(report_str + "\n")
        print(f"\n[GenerationEval] Saved detailed results and summary to {out_dir}/")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end RAG generation and judge evaluation")
    parser.add_argument("--catalog", type=str, default="Data/benchmark_catalog.jsonl")
    parser.add_argument("--answers", type=str, default="experiments/01_dense_bge_m3/answers.jsonl")
    parser.add_argument("--samples-per-category", type=int, default=20, help="Samples per question type (default: 20)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent worker threads (default: 4)")
    parser.add_argument("--model", type=str, default="nvidia/nemotron-3.5-lightning-30b-a3b")
    parser.add_argument("--top-docs", type=int, default=5, help="Number of retrieved docs to provide as context")
    parser.add_argument("--out-dir", type=str, default="experiments/03_gen_nemotron35_sample190")
    parser.add_argument(
        "--dsn",
        type=str,
        default=os.environ.get("ERAG_DATABASE_URL", "postgresql://erag:erag@127.0.0.1:5433/erag"),
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    catalog = load_catalog(catalog_path)
    sample = select_stratified_sample(catalog, samples_per_category=args.samples_per_category)

    retrieval_preds = load_retrieval_predictions(Path(args.answers))
    out_dir = Path(args.out_dir) if args.out_dir else None

    run_eval(
        catalog=sample,
        retrieval_preds=retrieval_preds,
        dsn=args.dsn,
        model=args.model,
        top_docs=args.top_docs,
        workers=args.workers,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
