"""Head-to-Head Comparative Benchmark:
Native Pipeline vs. LlamaIndex QueryEngine vs. LlamaIndex ReActAgent.

Evaluates the exact same benchmark questions previously checked in native experiments,
and compares factual accuracy, completeness, and latency scored by LLMJudge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.judge import LLMJudge
from generation.opencode_client import OpenCodeClient
from llamaindex_app.agent import get_agent, get_query_engine
from llamaindex_app.llm import NemotronLLM

DEFAULT_NATIVE_RUN = ROOT / "experiments" / "08_optimized_hybrid_top7" / "generated_evaluations.jsonl"
DEFAULT_CATALOG_PATH = ROOT / "Data" / "benchmark_catalog.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "eval_results"


def load_previous_evaluations(eval_path: Path) -> Dict[str, Dict[str, Any]]:
    """Index previously evaluated questions by question_id."""
    data = {}
    if not eval_path.exists():
        print(f"[Warning] Previous eval file not found at {eval_path}")
        return data

    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data[item["question_id"]] = item
    return data


def load_catalog(catalog_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load benchmark catalog metadata."""
    catalog = {}
    with open(catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                catalog[item["question_id"]] = item
    return catalog


async def run_comparison(
    samples_per_category: int = 2,
    categories: Optional[List[str]] = None,
    top_k: int = 7,
    use_expansion: bool = True,
    model: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
    eval_source: Path = DEFAULT_NATIVE_RUN,
    catalog_source: Path = DEFAULT_CATALOG_PATH,
) -> Dict[str, Any]:
    print("=" * 80)
    print("  EnterpriseRAG-Bench: Comparative Accuracy & Performance Benchmark")
    print("  Comparing:")
    print("    1. Native Hybrid Pipeline (Baseline)")
    print("    2. LlamaIndex RetrieverQueryEngine")
    print("    3. LlamaIndex ReActAgent Workflow")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prev_evals = load_previous_evaluations(eval_source)
    catalog = load_catalog(catalog_source)

    # Group by category
    by_cat = defaultdict(list)
    for qid, item in prev_evals.items():
        by_cat[item["question_type"]].append(item)

    target_cats = categories or sorted(by_cat.keys())
    selected_items: List[Dict[str, Any]] = []

    for cat in target_cats:
        items = by_cat.get(cat, [])
        selected_items.extend(items[:samples_per_category])

    print(f"\nTotal selected questions: {len(selected_items)} across {len(target_cats)} categories.")
    print(f"Categories: {', '.join(target_cats)}")
    print(f"Top-K context: {top_k} | Model: {model}\n")

    client = OpenCodeClient(model=model, timeout=180)
    judge = LLMJudge(client=client, model=model)
    query_engine = get_query_engine(top_k=top_k, use_expansion=use_expansion, model=model)

    records = []
    stats = {
        "native": {"correct": 0, "total": 0, "total_latency": 0.0},
        "llamaindex_qe": {"correct": 0, "total": 0, "total_latency": 0.0},
        "llamaindex_agent": {"correct": 0, "total": 0, "total_latency": 0.0},
    }
    cat_stats = defaultdict(lambda: {
        "count": 0,
        "native_correct": 0,
        "qe_correct": 0,
        "agent_correct": 0,
    })

    for idx, item in enumerate(selected_items, 1):
        qid = item["question_id"]
        qtype = item["question_type"]
        question = item["question"]
        gold_answer = item.get("gold_answer", "")
        facts = item.get("answer_facts", [])

        # 1. Native Evaluation Result
        native_ans = item.get("candidate_answer", "")
        native_correct = item.get("correct", False)
        native_lat = item.get("gen_time_sec", 0.0)

        # 2. LlamaIndex QueryEngine
        t_qe0 = time.time()
        try:
            res_qe = query_engine.query(question)
            qe_ans = str(res_qe.response).strip()
        except Exception as e:
            qe_ans = f"Error: {e}"
        t_qe = round(time.time() - t_qe0, 2)

        score_qe = judge.evaluate_answer(
            question_id=qid,
            question=question,
            candidate_answer=qe_ans,
            gold_answer=gold_answer,
            answer_facts=facts,
        )

        # 3. LlamaIndex ReAct Agent
        # Create fresh agent instance for independent query execution
        agent = get_agent(top_k=top_k, use_expansion=use_expansion, model=model)
        t_ag0 = time.time()
        try:
            res_ag = await agent.run(user_msg=question)
            raw_reply = getattr(res_ag, "response", str(res_ag))
            if hasattr(raw_reply, "content"):
                ag_ans = str(raw_reply.content).strip()
            else:
                ag_ans = str(raw_reply).strip()
            ag_ans = NemotronLLM._clean_reasoning(ag_ans)
        except Exception as e:
            ag_ans = f"Error: {e}"
        t_ag = round(time.time() - t_ag0, 2)

        score_ag = judge.evaluate_answer(
            question_id=qid,
            question=question,
            candidate_answer=ag_ans,
            gold_answer=gold_answer,
            answer_facts=facts,
        )

        # Update stats
        stats["native"]["total"] += 1
        if native_correct:
            stats["native"]["correct"] += 1
        stats["native"]["total_latency"] += native_lat

        stats["llamaindex_qe"]["total"] += 1
        if score_qe.correct:
            stats["llamaindex_qe"]["correct"] += 1
        stats["llamaindex_qe"]["total_latency"] += t_qe

        stats["llamaindex_agent"]["total"] += 1
        if score_ag.correct:
            stats["llamaindex_agent"]["correct"] += 1
        stats["llamaindex_agent"]["total_latency"] += t_ag

        cs = cat_stats[qtype]
        cs["count"] += 1
        if native_correct:
            cs["native_correct"] += 1
        if score_qe.correct:
            cs["qe_correct"] += 1
        if score_ag.correct:
            cs["agent_correct"] += 1

        mark_nat = "✓" if native_correct else "✗"
        mark_qe = "✓" if score_qe.correct else "✗"
        mark_ag = "✓" if score_ag.correct else "✗"

        print(
            f"[{idx:2d}/{len(selected_items):2d}] {qid} ({qtype:20s}) | "
            f"Native: {mark_nat} | "
            f"LlamaIndex QE: {mark_qe} ({t_qe:5.1f}s) | "
            f"LlamaIndex Agent: {mark_ag} ({t_ag:5.1f}s)",
            flush=True,
        )

        records.append({
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "gold_answer": gold_answer,
            "native": {
                "answer": native_ans,
                "correct": native_correct,
                "latency_sec": native_lat,
            },
            "llamaindex_qe": {
                "answer": qe_ans,
                "correct": score_qe.correct,
                "completeness": score_qe.completeness,
                "reasoning": score_qe.reasoning,
                "latency_sec": t_qe,
            },
            "llamaindex_agent": {
                "answer": ag_ans,
                "correct": score_ag.correct,
                "completeness": score_ag.completeness,
                "reasoning": score_ag.reasoning,
                "latency_sec": t_ag,
            },
        })

    # Summary calculations
    n = len(selected_items)
    acc_nat = stats["native"]["correct"] / n if n else 0.0
    acc_qe = stats["llamaindex_qe"]["correct"] / n if n else 0.0
    acc_ag = stats["llamaindex_agent"]["correct"] / n if n else 0.0

    avg_lat_nat = stats["native"]["total_latency"] / n if n else 0.0
    avg_lat_qe = stats["llamaindex_qe"]["total_latency"] / n if n else 0.0
    avg_lat_ag = stats["llamaindex_agent"]["total_latency"] / n if n else 0.0

    print("\n" + "=" * 80)
    print("  FINAL ACCURACY & LATENCY SUMMARY")
    print("=" * 80)
    print(f"  1. Native Pipeline:         {stats['native']['correct']}/{n} ({acc_nat*100:5.1f}%) | Avg Latency: {avg_lat_nat:5.1f}s")
    print(f"  2. LlamaIndex QueryEngine:  {stats['llamaindex_qe']['correct']}/{n} ({acc_qe*100:5.1f}%) | Avg Latency: {avg_lat_qe:5.1f}s")
    print(f"  3. LlamaIndex ReAct Agent:  {stats['llamaindex_agent']['correct']}/{n} ({acc_ag*100:5.1f}%) | Avg Latency: {avg_lat_ag:5.1f}s")
    print("=" * 80)

    # Markdown Report Generation
    report_lines = [
        "# Head-to-Head Benchmark: Native vs. LlamaIndex QueryEngine vs. LlamaIndex ReActAgent",
        "",
        f"- **Model**: `{model}`",
        f"- **Total Evaluated Questions**: {n}",
        f"- **Categories Evaluated**: {len(target_cats)}",
        f"- **Top-K Context**: {top_k}",
        f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overall Benchmark Comparison",
        "",
        "| Pipeline | Correct / Total | Accuracy (%) | Avg Latency (s) | Relative Delta vs Native |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Native Hybrid Pipeline** | {stats['native']['correct']}/{n} | **{acc_nat*100:.1f}%** | {avg_lat_nat:.1f}s | Baseline |",
        f"| **LlamaIndex QueryEngine** | {stats['llamaindex_qe']['correct']}/{n} | **{acc_qe*100:.1f}%** | {avg_lat_qe:.1f}s | **{(acc_qe - acc_nat)*100:+.1f}%** |",
        f"| **LlamaIndex ReAct Agent** | {stats['llamaindex_agent']['correct']}/{n} | **{acc_ag*100:.1f}%** | {avg_lat_ag:.1f}s | **{(acc_ag - acc_nat)*100:+.1f}%** |",
        "",
        "## Category-Level Breakdown",
        "",
        "| Category | N | Native Acc (%) | LlamaIndex QE (%) | LlamaIndex Agent (%) | Drop / Gain |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cat in sorted(cat_stats.keys()):
        cs = cat_stats[cat]
        c_n = cs["count"]
        nat_pct = (cs["native_correct"] / c_n * 100) if c_n else 0
        qe_pct = (cs["qe_correct"] / c_n * 100) if c_n else 0
        ag_pct = (cs["agent_correct"] / c_n * 100) if c_n else 0
        delta = qe_pct - nat_pct
        report_lines.append(
            f"| **{cat}** | {c_n} | {nat_pct:.1f}% | {qe_pct:.1f}% | {ag_pct:.1f}% | {delta:+.1f}% |"
        )

    report_lines.extend([
        "",
        "## Key Observations",
        "",
        f"1. **Accuracy Parity**: LlamaIndex QueryEngine achieved **{acc_qe*100:.1f}%** compared to Native's **{acc_nat*100:.1f}%**.",
        f"2. **Agent Reasoning**: LlamaIndex ReAct Agent achieved **{acc_ag*100:.1f}%**, executing autonomous multi-step retrieval workflows.",
        f"3. **Latency Profile**: QueryEngine averaged **{avg_lat_qe:.1f}s** per question, while ReAct Agent averaged **{avg_lat_ag:.1f}s** due to multi-turn tool interaction.",
        "",
    ])

    report_content = "\n".join(report_lines)
    report_file = OUTPUT_DIR / "comparison_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    results_data = {
        "summary": {
            "n_questions": n,
            "accuracy_native": acc_nat,
            "accuracy_qe": acc_qe,
            "accuracy_agent": acc_ag,
            "latency_native": avg_lat_nat,
            "latency_qe": avg_lat_qe,
            "latency_agent": avg_lat_ag,
        },
        "by_category": dict(cat_stats),
        "records": records,
    }
    json_file = OUTPUT_DIR / "comparison_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nSaved report to: {report_file}")
    print(f"Saved raw results to: {json_file}")
    return results_data


def main():
    parser = argparse.ArgumentParser(description="LlamaIndex Comparative Evaluation")
    parser.add_argument("--samples-per-cat", type=int, default=2, help="Number of questions per category")
    parser.add_argument("--categories", nargs="+", default=None, help="Categories to evaluate")
    parser.add_argument("--top-k", type=int, default=7, help="Context documents")
    parser.add_argument("--model", type=str, default="nvidia/nemotron-3.5-lightning-30b-a3b")
    args = parser.parse_args()

    asyncio.run(
        run_comparison(
            samples_per_category=args.samples_per_cat,
            categories=args.categories,
            top_k=args.top_k,
            model=args.model,
        )
    )


if __name__ == "__main__":
    main()
