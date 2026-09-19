# Evaluation Spec — EnterpriseRAGBench

This document defines **what we evaluate**, **against which data**, and **how scores are computed**. It is the evaluation contract for this repo.

Upstream references:

- Dataset: [`onyx-dot-app/EnterpriseRAG-Bench`](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)
- Methodology: [EnterpriseRAG-Bench `methodology.md`](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/main/methodology.md)
- Paper: [arXiv:2605.05253](https://arxiv.org/abs/2605.05253)

---

## 1. Goal

Evaluate a RAG system on **company-internal knowledge** for a synthetic enterprise (Redwood Inference): multi-source corpus, noisy/overlapping docs, and questions that require more than single-doc keyword lookup.

Primary ranking objective:

```text
Overall Score = mean over questions of (answer_correct × completeness)
```

- `answer_correct` ∈ {0, 1}
- `completeness` ∈ [0, 1] (fraction of gold `answer_facts` supported)
- Incorrect answers contribute **0** to Overall Score even if partially complete

Tie-break order (leaderboard-compatible): Overall Score → Answer Correctness → Answer Completeness → Document Recall.

---

## 2. Dataset under test

| Split | Hugging Face config | Rows | Role |
|---|---|---|---|
| Documents | `documents` / `test` | 511,962 | Retrieval corpus |
| Questions | `questions` / `test` | 500 | Eval set |

Local catalog (one JSONL row per question, with retrieval/source-scorable flags and applicable metrics):

```text
Data/benchmark_catalog.jsonl
```

Each catalog row includes:

| Field | Meaning |
|---|---|
| `question_id` | `qst_…` |
| `question_type` | One of the 10 benchmark types |
| `number_of_reference_documents` | `\|expected_doc_ids\|` (alias: `n_expected_docs`) |
| `source_types` | Gold connector list |
| `answerability` | `answerable` \| `unanswerable` (`info_not_found` → unanswerable) |
| `multi_hop` / `multi_hop_kind` | Needs combining evidence: `none` \| `intra_document` \| `multi_document` |
| `temporal` | Date/recency/superseding cues (heuristic; `conflicting_info` always true) |
| `multi_document` / `multi_source` | `\|expected_doc_ids\| > 1` / `\|source_types\| > 1` |
| `tags` | Flat filter labels derived from the above |
| plus | `question`, `expected_doc_ids`, `gold_answer`, `answer_facts`, scorable flags, `metrics` |

### 2.1 Document schema

| Field | Type | Notes |
|---|---|---|
| `doc_id` | string | Prefixed `dsid_…` |
| `source_type` | string | One of 9 connectors below |
| `title` | string | Channel name, ticket title, company name, etc. |
| `content` | string | Flattened text body |

### 2.2 Document source types (what lives where)

| `source_type` | Count | What it actually contains |
|---|---:|---|
| `slack` | 285,605 | Multi-speaker channel threads (`title` = channel) |
| `gmail` | 121,390 | Email threads (headers + body) |
| `linear` | 35,308 | Internal eng/product work items (goals, AC, rollout) |
| `google_drive` | 25,108 | Working docs/memos/notes/draft runbooks |
| `hubspot` | 15,017 | CRM account notes (use case, POC, next steps) |
| `fireflies` | 10,173 | Meeting summary + transcript |
| `github` | 8,052 | PR / release-note **text** (not source code) |
| `jira` | 6,120 | Customer/ops tickets (impact, repro, resolution) |
| `confluence` | 5,189 | Formal knowledge: runbooks, playbooks, architecture |

Use this map when interpreting retrieval failures (e.g. “GitHub miss” usually means a PR description miss, not a code miss).

### 2.3 Question schema

| Field | Type | Used for |
|---|---|---|
| `question_id` | string (`qst_…`) | Join answers → questions |
| `question_type` | string | Per-type score breakdown |
| `source_types` | list[string] | Analysis only (not required by scorer) |
| `question` | string | Query to the system |
| `expected_doc_ids` | list[string] | Gold retrieval set (`dsid_…`) |
| `gold_answer` | string | Correctness judge target |
| `answer_facts` | list[string] | Completeness judge targets |

### 2.4 Question-type mix (n = 500)

| `question_type` | Count | What it stress-tests |
|---|---:|---|
| `basic` | 175 | Single-doc factual lookup |
| `semantic` | 125 | Loose paraphrase / weak keyword overlap |
| `intra_document_reasoning` | 40 | Facts that require connecting distant parts of one doc |
| `project_related` | 40 | Multi-doc synthesis across a project cluster |
| `constrained` | 30 | Qualifiers that filter many near-matches |
| `conflicting_info` | 20 | Superseding / contradictory docs; prefer updated truth |
| `completeness` | 20 | Exhaustive multi-doc recall (partial retrieval fails) |
| `miscellaneous` | 20 | Peripheral / misc-path docs easy to miss |
| `info_not_found` | 20 | Must refuse / say not answerable (no hallucination) |
| `high_level` | 10 | Org-level synthesis; **no** gold `expected_doc_ids` |

Notes from the released set:

- Most questions cite **1 source type** and **1 gold doc**
- Multi-source / multi-doc questions are concentrated in `project_related`, `completeness`, `conflicting_info`, `constrained`
- `high_level` and `info_not_found` have **empty** `expected_doc_ids` / `source_types` → **Document Recall**, **Invalid Extra Docs**, all **@K / MRR** retrieval metrics, and **Source Accuracy / Precision / Recall** are not computed for them

---

## 3. System under evaluation — required outputs

For each of the 500 questions, the system must produce a JSONL row:

```json
{
  "question_id": "qst_0001",
  "answer": "…natural language answer…",
  "document_ids": ["dsid_…", "dsid_…"]
}
```

Rules:

- At least one of `answer` or `document_ids` is required; **both** are required for a full eval
- `document_ids` = documents the system retrieved / cited for that question, in **rank order** (best first). Order is required for MRR and for @K cuts
- Do not rely on citation markers inside `answer` for scoring (upstream eval strips citations before judging)

Recommended artifact path for this repo:

```text
answer_evaluation/answers.jsonl
```

---

## 4. Metrics (required)

Align with upstream EnterpriseRAG-Bench metrics-based evaluation.

### 4.1 Answer Correctness (binary, per question)

LLM judge: is the candidate answer aligned with `gold_answer`?

- Lenient on style / extra helpful context
- Strict that core claims match and **do not conflict** with gold
- Numeric quantities that appear in both must match
- Reported as % of questions marked correct

### 4.2 Answer Completeness (0–100%, per question)

For each string in `answer_facts`, an LLM judge checks whether the candidate answer contains or implies that fact.

```text
completeness_q = (# facts supported) / (# answer_facts)
```

Facts are judged **independently** (no leakage from the correctness judge).

### 4.3 Document Recall (%, only if `expected_doc_ids` non-empty)

Upstream “Document Recall” is recall over the full retrieved set returned in `document_ids` (no fixed K):

```text
recall_q = |retrieved ∩ expected| / |expected|
```

Average over questions that have gold docs. For fixed-cutoff retrieval quality, also report the ranked metrics in **§4.6**.

### 4.4 Invalid Extra Documents (count, only if gold docs exist)

Count of retrieved docs that are neither:

1. in the (possibly corrected) gold/required set, nor
2. classified as **valid** (relevant but not required)

Lower is better. Measures retrieval precision / noise under a relevance-aware definition (not hard precision against gold alone).

### 4.5 Overall Score (primary)

```text
score_q = answer_correct_q * completeness_q
Overall Score = mean(score_q) across all 500 questions
```

### 4.6 Ranked retrieval metrics (required)

These are classical IR metrics over the **ranked** `document_ids` list. They are required for this repo’s retrieval evaluation in addition to upstream Document Recall / Invalid Extra Docs.

**Eligibility:** only questions with non-empty `expected_doc_ids` (exclude `high_level` and `info_not_found`). Use the (possibly corrected) gold/required set as `G_q`.

**Notation (per question `q`):**

- `G_q` = gold/required doc set
- `R_q = [d_1, d_2, …]` = ranked retrieval list (best first)
- `R_q@K = {d_1, …, d_min(K, |R_q|)}`
- `hits_q@K = |R_q@K ∩ G_q|`
- `rank_q*` = 1-based rank of the **first** gold doc in `R_q` (undefined if none)

**Default cutoffs:** report all of the following at **K ∈ {1, 5, 10, 20}**. If the system returns fewer than K docs, evaluate on the returned prefix (do not pad).

#### Recall@K

Fraction of gold docs recovered in the top-K:

```text
Recall@K_q = hits_q@K / |G_q|
Recall@K   = mean_q(Recall@K_q)
```

- Range: [0, 1]
- Higher is better
- At K large enough to cover the full returned list, Recall@K ≈ upstream Document Recall for that question

#### Precision@K

Fraction of the top-K that are gold:

```text
Precision@K_q = hits_q@K / K
Precision@K   = mean_q(Precision@K_q)
```

- Range: [0, 1]
- Higher is better
- Denominator is always K (standard IR definition), even if fewer than K docs were returned — unless `|R_q| = 0`, then Precision@K_q = 0
- Optional companion (report if useful): `Precision@K_returned_q = hits_q@K / |R_q@K|` when `|R_q| < K`

#### MRR (Mean Reciprocal Rank)

How highly the first relevant doc is ranked:

```text
RR_q = 1 / rank_q*     if any gold doc appears in R_q
RR_q = 0               otherwise

MRR = mean_q(RR_q)
```

Also report **MRR@K**: same as RR_q but only counting a hit if `rank_q* ≤ K` (else 0). Required cutoffs: MRR@1, MRR@5, MRR@10, MRR@20, plus unbounded MRR.

- Range: [0, 1]
- Higher is better
- Especially informative for single-gold-doc questions (majority of the set)

#### Hit@K

Binary success: did we retrieve **at least one** gold doc in the top-K?

```text
Hit@K_q = 1  if hits_q@K ≥ 1 else 0
Hit@K   = mean_q(Hit@K_q)     # = % of questions with ≥1 gold in top-K
```

- Range: [0, 1] (report as fraction or %)
- Higher is better
- Alias sometimes called Success@K; use **Hit@K** in reports

#### How these relate

| Metric | Asks |
|---|---|
| Recall@K | Did we get **all / most** gold docs into top-K? |
| Precision@K | Of the top-K, how many were gold (noise rate)? |
| MRR / MRR@K | How soon is the **first** gold doc? |
| Hit@K | Did we get **any** gold doc into top-K? |

Upstream **Invalid Extra Docs** remains the LLM-aware noise metric (valid-but-not-required docs are not penalized). Precision@K is the hard gold-only counterpart and will usually look stricter.

### 4.7 Source Accuracy / Precision / Recall (required)

These score whether retrieval lands in the **correct connector(s)** (`source_type`), independent of exact `doc_id` match. Useful for diagnosing connector routing / hybrid-index failures.

**Eligibility:** only questions with non-empty gold `source_types` (exclude `high_level` and `info_not_found`). Flagged as `source_scorable: true` in `Data/benchmark_catalog.jsonl`.

**Notation (per question `q`):**

- `S_q` = gold source-type set from the question (`source_types`)
- `R_q = [d_1, d_2, …]` = ranked retrieved doc ids
- `source(d)` = `source_type` of document `d` from the corpus
- `P_q = { source(d) | d ∈ R_q }` — predicted source set from the full retrieved list
- `P_q@K = { source(d) | d ∈ R_q@K }`
- A retrieved doc is a **source-hit** if `source(d) ∈ S_q`

Resolve `source(d)` via the documents table (`doc_id` → `source_type`). Docs missing from the corpus map count as source-misses.

#### Source Precision

Of the sources present in the retrieved set, how many are gold?

```text
SourcePrecision_q = |P_q ∩ S_q| / |P_q|     if |P_q| > 0 else 0
SourcePrecision   = mean_q(SourcePrecision_q)
```

Also report **Source Precision@K** using `P_q@K` instead of `P_q`, for K ∈ {1, 5, 10, 20}.

Optional doc-level companion (report if useful):

```text
SourceDocPrecision@K_q = (# of d in R_q@K with source(d) ∈ S_q) / |R_q@K|
```

#### Source Recall

Of the gold sources, how many appear at least once in the retrieved set?

```text
SourceRecall_q = |P_q ∩ S_q| / |S_q|
SourceRecall   = mean_q(SourceRecall_q)
```

Also report **Source Recall@K** using `P_q@K`, for K ∈ {1, 5, 10, 20}.

#### Source Accuracy

Exact source-set match (strict):

```text
SourceAccuracy_q = 1  if P_q == S_q else 0
SourceAccuracy   = mean_q(SourceAccuracy_q)
```

Also report **Source Accuracy@K** where `P_q@K == S_q`.

For the common case `|S_q| = 1`, Source Accuracy@K equals Source Recall@K (both ask: “did the gold source appear?”), while Source Precision@K additionally penalizes retrieving other connectors.

#### How these relate to doc metrics

| Metric | Granularity | Asks |
|---|---|---|
| Source Precision / Recall / Accuracy | connector set | Did we search / retrieve from the right system(s)? |
| Precision@K / Recall@K / Hit@K / MRR | document ids | Did we get the right document(s)? |

A system can have high Source Recall but low Document Recall (right connector, wrong docs) — that pattern points to ranking/chunking inside a source, not connector selection.

---

## 5. Gold-label correction (required behavior when using official scorer)

Official eval may revise gold labels when candidate docs disagree with `expected_doc_ids`:

1. Gold + candidate docs are classified by 3 LLM judges as `required` / `valid` / `invalid`
2. Majority vote can update the required set
3. If the required set changes, `gold_answer` + `answer_facts` may be regenerated
4. Scoring then uses the **updated** ground truth; corrected questions are flagged in results

Implication for this project: treat gold as **revisable under evidence**, not immutable.

---

## 6. Reporting requirements

Every evaluation run should produce:

### 6.1 Aggregate table

| Metric | Value |
|---|---|
| Overall Score | |
| Answer Correctness (%) | |
| Answer Completeness (%) | |
| Document Recall (%) | |
| Invalid Extra Docs (avg) | |
| Recall@{1,5,10,20} | |
| Precision@{1,5,10,20} | |
| MRR and MRR@{1,5,10,20} | |
| Hit@{1,5,10,20} | |
| Source Accuracy / Accuracy@{1,5,10,20} | |
| Source Precision / Precision@{1,5,10,20} | |
| Source Recall / Recall@{1,5,10,20} | |

### 6.2 Breakdown by `question_type`

Report for each of the 10 types (where defined):

- Answer Correctness
- Document Recall
- Recall@10, Precision@10, MRR, Hit@10
- Source Accuracy, Source Precision, Source Recall (and @10)

This is mandatory — aggregate scores hide semantic / completeness / ranking / connector failures.

### 6.3 Optional but recommended breakdowns

| Slice | Why |
|---|---|
| By `source_types` on the question | Connector-specific retrieval gaps |
| Per predicted `source(d)` confusion | Which wrong connectors are retrieved most |
| Single-doc vs multi-doc questions | Chunking / multi-hop failures; Recall@K vs Hit@K diverge on multi-doc |
| `info_not_found` abstention rate | Hallucination control |
| Latency / cost per question | Ops readiness (not a leaderboard metric) |

### 6.4 Result artifacts

```text
answer_evaluation/results.json              # metrics-based run
answer_evaluation/results-comparative.json  # optional A/B
answer_evaluation/questions_updated.*       # corrected gold, if any
```

---

## 7. Comparative evaluation (optional)

When comparing two systems on the same 500 questions:

1. Strip citations; union retrieved docs → same correction flow
2. Three-judge head-to-head preference (+ equivalence flag)
3. Still report per-system completeness, recall, invalid extras, ranked metrics (Recall@K, Precision@K, MRR, Hit@K), and source metrics (Accuracy / Precision / Recall)

Use for ablations (chunking, hybrid retrieval, rerank, agent search), not as the sole public number.

---

## 8. What this repo should / should not put in scope

### In scope

- Full 500-question metrics-based eval against HF `questions`
- Ranked retrieval metrics: Recall@K, Precision@K, MRR / MRR@K, Hit@K for K ∈ {1,5,10,20}
- Source metrics: Source Accuracy, Source Precision, Source Recall (and @K)
- Per-question-type reporting
- Answer JSONL contract above (ranked `document_ids`)
- Optional comparative ablations

### Out of scope (for this spec)

- Re-generating the synthetic company corpus
- Training judge models from scratch
- Treating GitHub as a code-search benchmark (corpus is PR text)
- Claiming absolute “best docs in corpus” — gold is protocol-relevant, correction-aware

---

## 9. Minimal acceptance checklist for a system run

- [ ] Indexed all 511,962 documents (or a declared, versioned subset — not allowed for leaderboard-comparable runs)
- [ ] Produced answers for all 500 `question_id`s
- [ ] Emitted **ranked** `document_ids` for every answered question
- [ ] Reported Overall + 4 upstream metrics
- [ ] Reported Recall@K, Precision@K, MRR / MRR@K, Hit@K for K ∈ {1,5,10,20}
- [ ] Reported Source Accuracy, Source Precision, Source Recall (and @K)
- [ ] Reported Correctness + retrieval + source metrics by `question_type`
- [ ] Special-cased `info_not_found` (explicit “not answerable” behavior)
- [ ] Did not score Document Recall / @K / source metrics on `high_level` / `info_not_found`

---

## 10. Suggested local workflow

```bash
# 1) Explore corpus / questions
python Data/Prepocessing.py

# 2) Run your RAG pipeline → answer_evaluation/answers.jsonl

# 3) Score (upstream-compatible)
python -m src.scripts.answer_evaluation.metrics_based_eval \
  --answers-file answer_evaluation/answers.jsonl
```

Until local scoring scripts are vendored, treat Section 4–6 as the contract and reuse the official EnterpriseRAG-Bench evaluation utilities.
