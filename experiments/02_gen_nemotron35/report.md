
# End-to-End RAG Generation Benchmark Results

**Model**: `nvidia/nemotron-3.5-lightning-30b-a3b` | **Evaluated Questions**: 25 | **Context**: Top-5 Retrieved Docs

## Aggregate Leaderboard Metrics

| Metric | Score (%) | Meaning |
|---|---:|---|
| **Overall Score** | **16.0%** | Primary Leaderboard Rank: `mean(Correctness × Completeness)` |
| **Answer Correctness** | **72.0%** | Factual consistency with gold answer (no contradictions) |
| **Answer Completeness** | **16.0%** | Percentage of required gold facts captured |

## Breakdown by Question Type

| Question Type | n | Correctness | Completeness | Overall Score |
|---|---:|---:|---:|---:|
| `basic` | 7 | 71.4% | 14.3% | **14.3%** |
| `completeness` | 2 | 50.0% | 0.0% | **0.0%** |
| `conflicting_info` | 2 | 50.0% | 0.0% | **0.0%** |
| `constrained` | 2 | 100.0% | 0.0% | **0.0%** |
| `high_level` | 2 | 100.0% | 50.0% | **50.0%** |
| `info_not_found` | 2 | 100.0% | 0.0% | **0.0%** |
| `intra_document_reasoning` | 2 | 50.0% | 50.0% | **50.0%** |
| `miscellaneous` | 2 | 100.0% | 50.0% | **50.0%** |
| `project_related` | 2 | 50.0% | 0.0% | **0.0%** |
| `semantic` | 2 | 50.0% | 0.0% | **0.0%** |
