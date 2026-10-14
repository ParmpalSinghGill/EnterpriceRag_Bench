# EnterpriseRAG-Bench: Dense vs. Hybrid Retrieval Evaluation

- **Model**: `nvidia/nemotron-3.5-lightning-30b-a3b`
- **Evaluated Categories**: `['semantic', 'miscellaneous']`
- **Sample Size**: 40 questions
- **Context Documents**: Top-5
- **Retrieval Architecture**: Dense BGE-M3 (HNSW ef=128) + Sparse BM25 (SQLite FTS5) via RRF (k=20, w_sparse=1.2)
- **Total Time**: 1383.0s

## Head-to-Head Comparison: Baseline (Dense Only) vs. Hybrid (Dense + BM25)

| Category | Baseline Hit@5 | Hybrid Hit@5 | Baseline Correctness | Hybrid Correctness | Δ Correctness |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **miscellaneous** | 35.0% | **95.0%** | 30.0% | **85.0%** | **+55.0%** |
| **semantic** | 20.0% | **45.0%** | 30.0% | **75.0%** | **+45.0%** |
| **Combined Total** | ~27.5% | **70.0%** | 30.0% | **80.0%** | **+50.0%** |

## Category Breakdown (Hybrid)

| Category | N | Hit@5 | Correctness | Completeness | Overall Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `miscellaneous` | 20 | 95.0% | 85.0% | 0.3000 | 0.3000 |
| `semantic` | 20 | 45.0% | 75.0% | 0.0600 | 0.0600 |
