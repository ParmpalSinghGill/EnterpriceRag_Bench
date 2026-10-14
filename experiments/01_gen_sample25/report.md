
# End-to-End RAG Generation Benchmark Results

**Model**: `space-bunny-free` | **Evaluated Questions**: 25 | **Context**: Top-5 Retrieved Docs

## Aggregate Leaderboard Metrics

| Metric | Score (%) | Meaning |
|---|---:|---|
| **Overall Score** | **27.0%** | Primary Leaderboard Rank: `mean(Correctness × Completeness)` |
| **Answer Correctness** | **48.0%** | Factual consistency with gold answer (no contradictions) |
| **Answer Completeness** | **28.3%** | Percentage of required gold facts captured |

## Breakdown by Question Type

| Question Type | n | Correctness | Completeness | Overall Score |
|---|---:|---:|---:|---:|
| `basic` | 7 | 42.9% | 28.6% | **28.6%** |
| `completeness` | 2 | 50.0% | 0.0% | **0.0%** |
| `conflicting_info` | 2 | 50.0% | 50.0% | **50.0%** |
| `constrained` | 2 | 100.0% | 37.1% | **37.1%** |
| `high_level` | 2 | 50.0% | 0.0% | **0.0%** |
| `info_not_found` | 2 | 100.0% | 100.0% | **100.0%** |
| `intra_document_reasoning` | 2 | 50.0% | 50.0% | **50.0%** |
| `miscellaneous` | 2 | 50.0% | 0.0% | **0.0%** |
| `project_related` | 2 | 0.0% | 0.0% | **0.0%** |
| `semantic` | 2 | 0.0% | 16.7% | **0.0%** |
