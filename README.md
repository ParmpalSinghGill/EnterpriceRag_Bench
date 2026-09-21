# EnterpriseRAGBench

Experimentation workspace for building, ablating, and evaluating **enterprise RAG** systems on [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench) — a multi-source company-knowledge benchmark (~512k documents, 500 questions) from [onyx-dot-app](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) ([paper](https://arxiv.org/abs/2605.05253)).

This repo is technique-agnostic: use it to try retrieval, ranking, chunking, agentic search, and generation approaches against a **fixed evaluation contract**, then compare them fairly.

---

## GitHub About (paste into repo description)

```text
Enterprise RAG experimentation on EnterpriseRAG-Bench: multi-source retrieval + answer eval (correctness, completeness, Recall@K, MRR, Hit@K, source accuracy) over ~512k docs / 500 questions.
```

Suggested topics: `rag` `enterprise-search` `retrieval` `benchmark` `llm` `information-retrieval` `evaluation`

---

## Why this project

Real company knowledge is messy: Slack threads, Jira tickets, Confluence runbooks, Gmail, CRM notes, Drive drafts, meeting transcripts, Linear issues, and GitHub PR text — not clean Wikipedia pages.

This project exists to:

1. **Understand** the corpus and question mix
2. **Implement** multiple RAG / retrieval techniques
3. **Evaluate** them with the same metrics and catalog slices
4. **Compare** ablations until a production-worthy pipeline emerges

The evaluation rules stay stable; the methods under test do not.

---

## Dataset at a glance

| Asset | Size | Role |
|---|---:|---|
| Documents (`documents` / test) | 511,962 | Retrieval corpus |
| Questions (`questions` / test) | 500 | Eval set |
| Local catalog | `Data/benchmark_catalog.jsonl` | Per-question labels + applicable metrics |

### Connectors (`source_type`)

| Source | Role in corpus |
|---|---|
| `slack` | Channel threads |
| `gmail` | Email threads |
| `linear` | Internal eng/product work |
| `google_drive` | Working docs / memos |
| `hubspot` | CRM account notes |
| `fireflies` | Meeting summary + transcript |
| `github` | PR / release-note **text** (not source code) |
| `jira` | Customer / ops tickets |
| `confluence` | Formal runbooks / playbooks |

### Question slices (catalog tags)

Use `Data/benchmark_catalog.jsonl` to filter experiments by:

- `question_type` (basic, semantic, multi-doc, conflicting, unanswerable, …)
- `answerability` (`answerable` / `unanswerable`)
- `multi_hop` / `multi_hop_kind`
- `temporal`, `multi_document`, `multi_source`
- `source_types`, `number_of_reference_documents`

---

## Ingestion contract

How to load, chunk, and index the 511,962-document corpus so retrieval stays eval-compatible: **[`Docs/ingestion_spec.md`](Docs/ingestion_spec.md)**

Phase-1 chunk store (Docker Postgres + pgvector, **no embeddings yet**): **[`Docs/chunk_store.md`](Docs/chunk_store.md)**

Phase-1 audit (latest): **[`indexes/chunk_v1/audit_report.md`](indexes/chunk_v1/audit_report.md)** / [`audit_report.json`](indexes/chunk_v1/audit_report.json)

```bash
docker compose up -d
python -m ingestion.run_chunk --limit 1000 --out indexes/chunk_smoke   # size smoke test
python -m ingestion.run_chunk --out indexes/chunk_v1                   # full corpus split
python -m ingestion.audit_ingest --out indexes/chunk_v1/audit_report.json
```

## Evaluation contract

Full definition: **[`Docs/evaluation_spec.md`](Docs/evaluation_spec.md)**

### Answer quality (leaderboard-compatible)

| Metric | Meaning |
|---|---|
| **Overall Score** | `mean(correct × completeness)` — primary rank |
| **Correctness** | Binary LLM judge vs gold answer |
| **Completeness** | Fraction of gold `answer_facts` supported |
| **Document Recall** | Gold docs recovered (no fixed K) |
| **Invalid Extra Docs** | Irrelevant retrieved docs (lower better) |

### Ranked retrieval (required in this repo)

Report at **K ∈ {1, 5, 10, 20}**:

- **Recall@K**, **Precision@K**, **MRR** / **MRR@K**, **Hit@K**

### Source / connector quality

- **Source Accuracy**, **Source Precision**, **Source Recall** (and @K)

Questions without gold docs (`high_level`, `info_not_found`) skip retrieval/source metrics.

### Answer file format

```json
{"question_id": "qst_0001", "answer": "...", "document_ids": ["dsid_...", "dsid_..."]}
```

`document_ids` must be **rank-ordered** (best first).

---

## Repo layout

```text
EnterpriseRAGBench/
├── README.md
├── docker-compose.yml        # Postgres + pgvector (chunk store)
├── Docs/
│   ├── evaluation_spec.md
│   ├── ingestion_spec.md
│   └── chunk_store.md        # Docker + phase-1 chunking instructions
├── Data/
│   ├── Prepocessing.py
│   └── benchmark_catalog.jsonl
├── infra/postgres/init.sql   # DB schema (embeddings NULL for now)
├── ingestion/                # sentence_quarter_v1 chunker + CLI
├── indexes/                  # chunk builds (gitignored)
├── experiments/
│   └── _template/
└── answer_evaluation/
```

Suggested experiment folder pattern:

```text
experiments/<technique-name>/
  README.md          # hypothesis, config, how to run
  run.py             # or notebook
  configs/
  outputs/           # answers.jsonl, metrics.json
```

Examples of techniques you might put here over time (non-exhaustive):

- BM25 / sparse retrieval
- Dense / hybrid (BM25 + vector) + RRF
- Chunking strategies (fixed, semantic, late chunking)
- Rerankers (cross-encoder, LLM rerank)
- Query rewrite / HyDE / multi-query
- Source-aware routing (predict connector first)
- Agentic / tool-using search
- Long-context vs RAG tradeoffs
- Abstention for `info_not_found`

---

## Quickstart

```bash
# Env with `datasets`, numpy, etc.
conda activate ERAG   # or your env

# Explore corpus / questions
python Data/Prepocessing.py

# Catalog slices
python - <<'PY'
import json
from collections import Counter
rows = [json.loads(l) for l in open("Data/benchmark_catalog.jsonl")]
print(Counter(r["question_type"] for r in rows))
print(Counter(r["answerability"] for r in rows))
print(sum(r["multi_hop"] for r in rows), "multi-hop")
PY
```

Load upstream data:

```python
from datasets import load_dataset

documents = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "documents")["test"]
questions = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "questions")["test"]
```

---

## Experiment workflow (stable until end)

1. **Pick a technique** → create `experiments/<name>/`
2. **Ingest / index** the document corpus per [`Docs/ingestion_spec.md`](Docs/ingestion_spec.md) (declare any subset; full corpus required for comparable runs)
3. **Answer** all 500 questions → `answer_evaluation/<name>/answers.jsonl`
4. **Score** against the evaluation spec (Overall + retrieval @K + source metrics)
5. **Slice** results with the catalog (`semantic`, `multi_hop`, `temporal`, `unanswerable`, …)
6. **Log** config + metrics in that experiment’s README so later ablations stay comparable

Primary decision metric: **Overall Score**. Use @K / MRR / source metrics to explain *why* a method wins or loses.

---

## Upstream references

- Dataset: https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench  
- Upstream repo / methodology: https://github.com/onyx-dot-app/EnterpriseRAG-Bench  
- Paper: https://arxiv.org/abs/2605.05253  

This workspace builds on that benchmark; it does not replace the official leaderboard tooling unless/until those scripts are vendored here.

---

## Status

Active experimentation. Methods will change; **`Docs/ingestion_spec.md`**, **`Docs/evaluation_spec.md`**, and **`Data/benchmark_catalog.jsonl`** are the durable ground truth for “how to index” and “what good looks like.”
