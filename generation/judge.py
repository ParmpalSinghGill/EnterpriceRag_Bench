"""Automated LLM-as-a-judge for Answer Correctness and Completeness.

Implements Docs/evaluation_spec.md §4.1, §4.2, and §4.5:
  - Correctness: Binary judge (is candidate aligned with gold_answer without contradiction?)
  - Completeness: Fraction of gold answer_facts supported in candidate answer
  - Overall Score: Correctness * Completeness
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from generation.opencode_client import OpenCodeClient

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

JUDGE_SYSTEM_PROMPT = """You are an impartial, expert evaluation judge for an Enterprise RAG benchmark.
Your job is to objectively score a candidate answer against a reference Gold Answer and a list of Gold Facts.

Scoring Rules:
1. Correctness (boolean):
   - Output true if the candidate answer's core claims are factually consistent with the Gold Answer and do NOT contradict it.
   - Output false if the candidate makes claims that directly conflict with the Gold Answer, or if numeric quantities disagree.
   - Extra helpful context or minor wording differences should NOT be penalized.
   - If the gold answer states that the information is not found/unavailable, the candidate is correct if and only if it also refuses or states the info is not found.

2. Completeness (facts_supported):
   - Review each fact in the provided Gold Facts list.
   - Include a fact in "facts_supported" if and only if the candidate answer directly states or clearly implies that fact.
   - Facts must be judged strictly against what the candidate answer actually says.

CRITICAL FORMATTING REQUIREMENT:
You MUST respond with ONLY the JSON object. Do NOT include "Here's a thinking process" or any preamble or explanation outside the JSON.
Start your response immediately with the opening brace '{'.

Output Format:
{
  "correct": true or false,
  "facts_supported": ["exact fact string 1", "exact fact string 2"],
  "reasoning": "brief explanation"
}
"""


@dataclass
class JudgeScore:
    question_id: str
    correct: bool
    completeness: float
    overall_score: float
    total_facts: int
    n_supported_facts: int
    supported_facts: List[str] = field(default_factory=list)
    missing_facts: List[str] = field(default_factory=list)
    reasoning: str = ""


class LLMJudge:
    """Evaluates candidate answers against gold answers and facts."""

    def __init__(self, client: Optional[OpenCodeClient] = None, model: Optional[str] = None):
        target_model = model or DEFAULT_MODEL
        self.client = client or OpenCodeClient(model=target_model)
        self.model = target_model

    def evaluate_answer(
        self,
        question_id: str,
        question: str,
        candidate_answer: str,
        gold_answer: str,
        answer_facts: Optional[Sequence[str]] = None,
    ) -> JudgeScore:
        """Score a single candidate answer."""
        facts_list = list(answer_facts or [])
        facts_str = "\n".join(f"- {f}" for f in facts_list) if facts_list else "(None specified)"

        prompt = f"""Question: {question}

Gold Answer: {gold_answer}

Gold Facts:
{facts_str}

Candidate Answer:
{candidate_answer}

Evaluate and return ONLY the JSON object:"""

        response = self.client.generate(
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=3072,
        )

        correct = False
        facts_supported: List[str] = []
        reasoning = ""

        def extract_judge_json(text: str) -> Optional[Dict[str, Any]]:
            cleaned = text.strip()
            try:
                return json.loads(cleaned)
            except Exception:
                pass
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
            m = re.search(r'(\{\s*"correct"\s*:.*?\"reasoning\"\s*:.*?\})', cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
            m = re.search(r'(\{\s*"correct"\s*:.*?\})', cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
            pos = cleaned.find('"correct"')
            if pos != -1:
                start = cleaned.rfind('{', 0, pos)
                if start != -1:
                    depth = 0
                    for i in range(start, len(cleaned)):
                        if cleaned[i] == '{':
                            depth += 1
                        elif cleaned[i] == '}':
                            depth -= 1
                            if depth == 0:
                                try:
                                    return json.loads(cleaned[start:i+1])
                                except Exception:
                                    break
            return None

        data = extract_judge_json(response)
        if data is not None:
            correct = bool(data.get("correct", False))
            facts_supported = list(data.get("facts_supported", []))
            reasoning = data.get("reasoning", "")
        else:
            print(f"[LLMJudge] Fallback parsing for {question_id}\nResponse: {response[:150]}")
            m_corr = re.search(r"\"correct\"\s*:\s*(true|false)", response, re.I)
            if m_corr:
                correct = m_corr.group(1).lower() == "true"
            elif "true" in response.lower() and "false" not in response.lower()[:30]:
                correct = True

            m_facts = re.search(r"\"facts_supported\"\s*:\s*\[(.*?)(?:\]|$)", response, re.DOTALL)
            if m_facts:
                raw_facts = m_facts.group(1)
                facts_supported = re.findall(r"\"([^\"]+)\"", raw_facts)
            else:
                for fact in facts_list:
                    # Check token overlap
                    f_words = set(re.findall(r"\w+", fact.lower()))
                    c_words = set(re.findall(r"\w+", candidate_answer.lower()))
                    if f_words and len(f_words.intersection(c_words)) / len(f_words) >= 0.7:
                        facts_supported.append(fact)

        n_total = len(facts_list)
        if n_total > 0:
            matched_facts = [
                f for f in facts_list
                if any(f.lower() in s.lower() or s.lower() in f.lower() for s in facts_supported)
            ]
            n_supported = len(matched_facts)
            completeness = n_supported / n_total
            missing_facts = [f for f in facts_list if f not in matched_facts]
        else:
            completeness = 1.0 if correct else 0.0
            n_supported = 0
            missing_facts = []

        overall_score = (1.0 if correct else 0.0) * completeness

        return JudgeScore(
            question_id=question_id,
            correct=correct,
            completeness=round(completeness, 4),
            overall_score=round(overall_score, 4),
            total_facts=n_total,
            n_supported_facts=n_supported,
            supported_facts=matched_facts if n_total > 0 else facts_supported,
            missing_facts=missing_facts,
            reasoning=reasoning,
        )
