# EnterpriseRAG-Bench: Hybrid + HyDE Expansion Evaluation

- **Model**: `nvidia/nemotron-3.5-lightning-30b-a3b`
- **Evaluated Categories**: `['semantic']`
- **Sample Size**: 20 questions
- **Context Documents**: Top-5
- **Query Expansion**: Enabled (LLM HyDE + Keywords)
- **Retrieval Architecture**: Dense BGE-M3 (HNSW ef=128) + Sparse BM25 (SQLite FTS5) via RRF (k=20, w_sparse=1.2)
- **Total Time**: 4171.6s

## Head-to-Head Comparison: Baseline (Dense Only) vs. Hybrid vs. Hybrid + HyDE

| Category | Baseline Hit@5 | Evaluated Hit@5 | Baseline Correctness | Evaluated Correctness | Δ Correctness |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **semantic** | 20.0% | **100.0%** | 30.0% | **80.0%** | **+50.0%** |
