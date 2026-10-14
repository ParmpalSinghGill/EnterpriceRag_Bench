"""Benchmark Evaluation Runner for LlamaIndex Pipeline.

Evaluates questions from Data/benchmark_catalog.jsonl using the LlamaIndex QueryEngine
and scores answers using LLMJudge.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.judge import LLMJudge
from generation.opencode_client import OpenCodeClient
from llamaindex_app.agent import get_query_engine

DEFAULT_CATALOG_PATH = ROOT / "Data" / "benchmark_catalog.jsonl"


def run_llamaindex_eval(
    samples_per_category: int = 5,
    categories: List[str] = None,
    top_k: int = 7,
    use_expansion: bool = True,
    model: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
    catalog_path: Path = DEFAULT_CATALOG_PATH,
) -> Dict[str, Any]:
    """Run evaluation using LlamaIndex QueryEngine."""
    print("=" * 60)
    print("  EnterpriseRAG-Bench: LlamaIndex Pipeline Evaluation")
    print("=" * 60)
    print(f"Samples per category: {samples_per_category}")
    print(f"Top-K context: {top_k}")
    print(f"Model: {model}\n")

    catalog = []
    with open(catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                catalog.append(json.loads(line))

    by_cat = defaultdict(list)
    for q in catalog:
        by_cat[q["question_type"]].append(q)

    target_cats = categories or sorted(by_cat.keys())
    target_questions = []
    for cat in target_cats:
        target_questions.extend(by_cat[cat][:samples_per_category])

    print(f"Loaded {len(target_questions)} evaluation questions across {len(target_cats)} categories.\n")

    query_engine = get_query_engine(top_k=top_k, use_expansion=use_expansion, model=model)
    client = OpenCodeClient(model=model, timeout=180)
    judge = LLMJudge(client=client, model=model)

    correct_count = 0
    results = []

    for idx, q in enumerate(target_questions, 1):
        qid = q["question_id"]
        qtype = q["question_type"]
        question = q["question"]
        gold_answer = q.get("gold_answer", "")
        facts = q.get("answer_facts", [])

        t0 = time.time()
        try:
            res = query_engine.query(question)
            candidate_answer = str(res.response).strip()
        except Exception as e:
            candidate_answer = f"Error: {e}"

        elapsed = round(time.time() - t0, 2)

        score = judge.evaluate_answer(
            question_id=qid,
            question=question,
            candidate_answer=candidate_answer,
            gold_answer=gold_answer,
            answer_facts=facts,
        )

        is_corr = score.correct
        if is_corr:
            correct_count += 1

        mark = "✓" if is_corr else "✗"
        print(f"[{idx:2d}/{len(target_questions):2d}] {qid} ({qtype:20s}) | {mark} | {elapsed}s")
        results.append({
            "qid": qid,
            "type": qtype,
            "correct": is_corr,
            "answer": candidate_answer,
            "gold": gold_answer,
            "reasoning": score.reasoning,
        })

    acc = correct_count / len(target_questions) if target_questions else 0.0
    print("\n" + "=" * 60)
    print(f"LlamaIndex Pipeline Accuracy: {correct_count}/{len(target_questions)} = {acc*100:.1f}%")
    print("=" * 60)
    return {"accuracy": acc, "total": len(target_questions), "correct": correct_count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2, help="Samples per category")
    parser.add_argument("--top-k", type=int, default=7, help="Context documents")
    parser.add_argument("--categories", nargs="+", default=None, help="Categories to test")
    args = parser.parse_args()

    run_llamaindex_eval(samples_per_category=args.samples, top_k=args.top_k, categories=args.categories)

