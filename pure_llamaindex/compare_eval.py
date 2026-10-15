"""Head-to-head comparison between Hybrid LlamaIndex and Pure Solo LlamaIndex.

Evaluates 2 questions per category across all 10 benchmark categories (20 questions total).
Compares:
  1. Hybrid LlamaIndex QueryEngine (Dense BGE-M3 + BM25 FTS5 + RRF + HyDE)
  2. Pure Solo LlamaIndex QueryEngine (PGVectorStore HNSW)
  3. Pure Solo LlamaIndex ReActAgent (PGVectorStore + ReAct Workflow)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.judge import LLMJudge
from llamaindex_app.agent import get_query_engine as get_hybrid_qe
from pure_llamaindex.agent import get_query_engine as get_pure_qe, get_react_agent as get_pure_agent

CATALOG_PATH = ROOT / "Data" / "benchmark_catalog.jsonl"
OUTPUT_DIR = ROOT / "eval_results"


def clean_reasoning_trace(text: str) -> str:
    """Strip raw thinking traces or chain-of-thought preambles if present."""
    if not text:
        return ""
    if "Here's a thinking process" in text or "Let me analyze" in text:
        parts = re.split(r"\n\n(?=[A-Z0-9\"'•\-])", text)
        if len(parts) > 1:
            non_thinking = [
                p for p in parts
                if not p.strip().startswith((
                    "Here's a thinking", "Let me analyze", "1. ", "2. ", "3. ", "4. ", "5. ",
                    "**Analyze", "**Identify", "**Scan", "**Synthesize", "**Draft", "**Conclusion"
                ))
            ]
            if non_thinking:
                text = "\n\n".join(non_thinking).strip()
        m_ans = re.findall(
            r"(?:Answer|Conclusion|Summary|The answer is|Based on the documents):\s*(.+)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m_ans:
            text = m_ans[-1].strip()
    return text.strip()


def select_benchmark_questions(catalog_path: Path, samples_per_cat: int = 2) -> List[Dict[str, Any]]:
    """Select exactly N questions per category from the benchmark catalog."""
    by_cat = defaultdict(list)
    with open(catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                by_cat[item["question_type"]].append(item)

    selected = []
    for cat in sorted(by_cat.keys()):
        items = by_cat[cat][:samples_per_cat]
        selected.extend(items)
    return selected


async def evaluate_pipeline(
    name: str,
    run_fn,
    question: str,
    qid: str,
    gold_answer: str,
    facts: List[str],
    judge: LLMJudge,
) -> Dict[str, Any]:
    """Execute pipeline query and score with LLMJudge."""
    t0 = time.time()
    try:
        if asyncio.iscoroutinefunction(run_fn):
            raw_res = await run_fn(question)
        else:
            raw_res = run_fn(question)
        ans = clean_reasoning_trace(str(raw_res))
    except Exception as e:
        ans = f"Error: {e}"
    lat = round(time.time() - t0, 2)

    score = judge.evaluate_answer(
        question_id=qid,
        question=question,
        candidate_answer=ans,
        gold_answer=gold_answer,
        answer_facts=facts,
    )

    return {
        "pipeline": name,
        "answer": ans,
        "correct": score.correct,
        "completeness": score.completeness,
        "overall_score": score.overall_score,
        "reasoning": score.reasoning,
        "latency_sec": lat,
    }


async def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid vs Pure LlamaIndex")
    parser.add_argument("--samples-per-cat", type=int, default=2)
    parser.add_argument("--skip-agent", action="store_true", help="Evaluate only QueryEngines to run faster")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = select_benchmark_questions(CATALOG_PATH, samples_per_cat=args.samples_per_cat)

    print("=" * 80)
    print("  EnterpriseRAG-Bench: Hybrid LlamaIndex vs. Pure Solo LlamaIndex")
    print(f"  Total Questions: {len(questions)} ({args.samples_per_cat} per category across 10 categories)")
    print(f"  Includes ReAct Agent: {not args.skip_agent}")
    print("=" * 80)

    # Initialize engines
    print("[1/3] Initializing Hybrid LlamaIndex QueryEngine...")
    hybrid_qe = get_hybrid_qe(top_k=7, use_expansion=True)

    print("[2/3] Initializing Pure Solo LlamaIndex QueryEngine...")
    pure_qe = get_pure_qe(similarity_top_k=7)

    if not args.skip_agent:
        print("[3/3] Initializing Pure Solo LlamaIndex ReActAgent...")
        pure_agent = get_pure_agent(similarity_top_k=7)
    else:
        pure_agent = None

    judge = LLMJudge()

    records = []
    summary_counts = {
        "hybrid_qe": {"correct": 0, "total": 0, "total_lat": 0.0},
        "pure_qe": {"correct": 0, "total": 0, "total_lat": 0.0},
    }
    if not args.skip_agent:
        summary_counts["pure_agent"] = {"correct": 0, "total": 0, "total_lat": 0.0}

    cat_counts = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for idx, q_item in enumerate(questions, start=1):
        qid = q_item["question_id"]
        cat = q_item["question_type"]
        q_text = q_item["question"]
        gold = q_item["gold_answer"]
        facts = q_item.get("answer_facts", [])

        # 1. Hybrid LlamaIndex QE
        def run_hybrid_qe(q):
            res = hybrid_qe.query(q)
            return res.response

        eval_h = await evaluate_pipeline("Hybrid QE", run_hybrid_qe, q_text, qid, gold, facts, judge)
        summary_counts["hybrid_qe"]["total"] += 1
        summary_counts["hybrid_qe"]["total_lat"] += eval_h["latency_sec"]
        if eval_h["correct"]:
            summary_counts["hybrid_qe"]["correct"] += 1
            cat_counts[cat]["hybrid_qe"]["correct"] += 1
        cat_counts[cat]["hybrid_qe"]["total"] += 1

        # 2. Pure Solo LlamaIndex QE
        def run_pure_qe(q):
            res = pure_qe.query(q)
            return res.response

        eval_p = await evaluate_pipeline("Pure QE", run_pure_qe, q_text, qid, gold, facts, judge)
        summary_counts["pure_qe"]["total"] += 1
        summary_counts["pure_qe"]["total_lat"] += eval_p["latency_sec"]
        if eval_p["correct"]:
            summary_counts["pure_qe"]["correct"] += 1
            cat_counts[cat]["pure_qe"]["correct"] += 1
        cat_counts[cat]["pure_qe"]["total"] += 1

        # 3. Pure Solo LlamaIndex Agent (optional)
        eval_ag = None
        if pure_agent is not None:
            async def run_pure_ag(q):
                res = await pure_agent.run(user_msg=q)
                return str(res)

            eval_ag = await evaluate_pipeline("Pure Agent", run_pure_ag, q_text, qid, gold, facts, judge)
            summary_counts["pure_agent"]["total"] += 1
            summary_counts["pure_agent"]["total_lat"] += eval_ag["latency_sec"]
            if eval_ag["correct"]:
                summary_counts["pure_agent"]["correct"] += 1
                cat_counts[cat]["pure_agent"]["correct"] += 1
            cat_counts[cat]["pure_agent"]["total"] += 1

        # Real-time console status
        h_sym = "✓" if eval_h["correct"] else "✗"
        p_sym = "✓" if eval_p["correct"] else "✗"
        status_line = (
            f"[{idx:2d}/{len(questions)}] {qid} ({cat:<24}) | "
            f"Hybrid QE: {h_sym} ({eval_h['latency_sec']:4.1f}s) | "
            f"Pure QE: {p_sym} ({eval_p['latency_sec']:4.1f}s)"
        )
        if eval_ag:
            a_sym = "✓" if eval_ag["correct"] else "✗"
            status_line += f" | Pure Agent: {a_sym} ({eval_ag['latency_sec']:4.1f}s)"
        print(status_line)

        record = {
            "question_id": qid,
            "category": cat,
            "question": q_text,
            "gold_answer": gold,
            "hybrid_qe": eval_h,
            "pure_qe": eval_p,
        }
        if eval_ag:
            record["pure_agent"] = eval_ag
        records.append(record)

    # Calculate overall metrics
    n_tot = len(questions)
    h_acc = (summary_counts["hybrid_qe"]["correct"] / n_tot) * 100
    p_acc = (summary_counts["pure_qe"]["correct"] / n_tot) * 100
    h_avg_lat = summary_counts["hybrid_qe"]["total_lat"] / n_tot
    p_avg_lat = summary_counts["pure_qe"]["total_lat"] / n_tot

    print("\n" + "=" * 80)
    print("  FINAL ACCURACY & LATENCY SUMMARY (20 Questions across 10 Categories)")
    print("=" * 80)
    print(f"  1. Hybrid LlamaIndex QE:    {summary_counts['hybrid_qe']['correct']}/{n_tot} ({h_acc:5.1f}%) | Avg Latency: {h_avg_lat:4.1f}s")
    print(f"  2. Pure Solo LlamaIndex QE: {summary_counts['pure_qe']['correct']}/{n_tot} ({p_acc:5.1f}%) | Avg Latency: {p_avg_lat:4.1f}s")
    if not args.skip_agent:
        ag_acc = (summary_counts["pure_agent"]["correct"] / n_tot) * 100
        ag_avg_lat = summary_counts["pure_agent"]["total_lat"] / n_tot
        print(f"  3. Pure Solo ReAct Agent:   {summary_counts['pure_agent']['correct']}/{n_tot} ({ag_acc:5.1f}%) | Avg Latency: {ag_avg_lat:4.1f}s")
    print("=" * 80)

    # Save JSON results
    json_path = OUTPUT_DIR / "comparison_20_questions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total_questions": n_tot,
                    "hybrid_qe_acc": h_acc,
                    "pure_qe_acc": p_acc,
                    "hybrid_qe_avg_latency": h_avg_lat,
                    "pure_qe_avg_latency": p_avg_lat,
                },
                "by_category": dict(cat_counts),
                "records": records,
            },
            f,
            indent=2,
        )

    # Generate Markdown Report
    md_path = OUTPUT_DIR / "comparison_20_questions.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 20-Question Head-to-Head: Hybrid LlamaIndex vs. Pure Solo LlamaIndex\n\n")
        f.write(f"- **Evaluated Questions**: {n_tot} (2 per category across 10 categories)\n")
        f.write(f"- **Model**: `nvidia/nemotron-3.5-lightning-30b-a3b`\n")
        f.write(f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Overall Accuracy & Latency Summary\n\n")
        f.write("| Pipeline | Correct / Total | Accuracy (%) | Avg Latency (s) | Delta vs Hybrid |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Hybrid LlamaIndex QueryEngine** | {summary_counts['hybrid_qe']['correct']}/{n_tot} | **{h_acc:.1f}%** | {h_avg_lat:.1f}s | Baseline |\n")
        f.write(f"| **Pure Solo LlamaIndex QueryEngine** | {summary_counts['pure_qe']['correct']}/{n_tot} | **{p_acc:.1f}%** | {p_avg_lat:.1f}s | **{p_acc - h_acc:+.1f}%** |\n")
        if not args.skip_agent:
            ag_acc = (summary_counts["pure_agent"]["correct"] / n_tot) * 100
            ag_avg_lat = summary_counts["pure_agent"]["total_lat"] / n_tot
            f.write(f"| **Pure Solo LlamaIndex ReAct Agent** | {summary_counts['pure_agent']['correct']}/{n_tot} | **{ag_acc:.1f}%** | {ag_avg_lat:.1f}s | **{ag_acc - h_acc:+.1f}%** |\n")

        f.write("\n## Category-by-Category Breakdown (2 Questions Each)\n\n")
        cols = "| Category | Hybrid QE Acc (%) | Pure QE Acc (%) |"
        sep = "| :--- | :---: | :---: |"
        if not args.skip_agent:
            cols += " Pure Agent Acc (%) |"
            sep += " :---: |"
        cols += " Notes |\n"
        sep += " :--- |\n"
        f.write(cols + sep)

        for cat in sorted(cat_counts.keys()):
            h_c = cat_counts[cat]["hybrid_qe"]["correct"]
            h_t = cat_counts[cat]["hybrid_qe"]["total"]
            p_c = cat_counts[cat]["pure_qe"]["correct"]
            p_t = cat_counts[cat]["pure_qe"]["total"]
            h_pct = (h_c / h_t * 100) if h_t else 0
            p_pct = (p_c / p_t * 100) if p_t else 0

            row = f"| **{cat}** | {h_pct:.0f}% ({h_c}/{h_t}) | {p_pct:.0f}% ({p_c}/{p_t}) |"
            if not args.skip_agent:
                a_c = cat_counts[cat]["pure_agent"]["correct"]
                a_t = cat_counts[cat]["pure_agent"]["total"]
                a_pct = (a_c / a_t * 100) if a_t else 0
                row += f" {a_pct:.0f}% ({a_c}/{a_t}) |"

            diff = p_pct - h_pct
            if diff > 0:
                note = f"Pure +{diff:.0f}%"
            elif diff < 0:
                note = f"Pure {diff:.0f}%"
            else:
                note = "Parity"
            row += f" {note} |\n"
            f.write(row)

    print(f"\nSaved JSON results to: {json_path}")
    print(f"Saved Markdown report to: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
