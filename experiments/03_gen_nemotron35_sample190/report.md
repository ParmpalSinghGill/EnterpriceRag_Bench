
# End-to-End RAG Generation Benchmark Results

**Model**: `nvidia/nemotron-3.5-lightning-30b-a3b` | **Evaluated Questions**: 190 | **Context**: Top-5 Retrieved Docs

## Aggregate Leaderboard Metrics

| Metric | Score (%) | Meaning |
|---|---:|---|
| **Overall Score** | **7.4%** | Primary Leaderboard Rank: `mean(Correctness × Completeness)` |
| **Answer Correctness** | **50.5%** | Factual consistency with gold answer (no contradictions) |
| **Answer Completeness** | **7.4%** | Percentage of required gold facts captured |

## Breakdown by Question Type

| Question Type | n | Correctness | Completeness | Overall Score |
|---|---:|---:|---:|---:|
| `basic` | 20 | 45.0% | 8.3% | **8.3%** |
| `completeness` | 20 | 45.0% | 0.0% | **0.0%** |
| `conflicting_info` | 20 | 55.0% | 0.0% | **0.0%** |
| `constrained` | 20 | 70.0% | 0.0% | **0.0%** |
| `high_level` | 10 | 70.0% | 33.5% | **33.5%** |
| `info_not_found` | 20 | 85.0% | 15.0% | **15.0%** |
| `intra_document_reasoning` | 20 | 45.0% | 5.0% | **5.0%** |
| `miscellaneous` | 20 | 30.0% | 15.0% | **15.0%** |
| `project_related` | 20 | 40.0% | 0.0% | **0.0%** |
| `semantic` | 20 | 30.0% | 10.0% | **10.0%** |
