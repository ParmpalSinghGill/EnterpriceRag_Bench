# EnterpriseRAG-Bench: Hybrid + HyDE Expansion Evaluation

- **Model**: `nvidia/nemotron-3.5-lightning-30b-a3b`
- **Evaluated Categories**: `['basic', 'completeness', 'conflicting_info', 'constrained', 'high_level', 'info_not_found', 'intra_document_reasoning', 'miscellaneous', 'project_related', 'semantic']`
- **Sample Size**: 400 questions
- **Context Documents**: Top-7
- **Query Expansion**: Enabled (LLM HyDE + Keywords)
- **Retrieval Architecture**: Dense BGE-M3 (HNSW ef=128) + Sparse BM25 (SQLite FTS5) via RRF (k=20, w_sparse=1.2)
- **Total Time**: 5581.2s

## Head-to-Head Comparison: Baseline (Dense Only) vs. Hybrid vs. Hybrid + HyDE

| Category | Baseline Hit@5 | Evaluated Hit@5 | Baseline Correctness | Evaluated Correctness | Δ Correctness |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **basic** | 35.0% | **73.0%** | 45.0% | **81.0%** | **+36.0%** |
| **completeness** | 35.0% | **80.0%** | 45.0% | **70.0%** | **+25.0%** |
| **conflicting_info** | 35.0% | **90.0%** | 55.0% | **95.0%** | **+40.0%** |
| **constrained** | 35.0% | **96.7%** | 70.0% | **93.3%** | **+23.3%** |
| **high_level** | 35.0% | **0.0%** | 70.0% | **80.0%** | **+10.0%** |
| **info_not_found** | 35.0% | **0.0%** | 85.0% | **100.0%** | **+15.0%** |
| **intra_document_reasoning** | 35.0% | **97.5%** | 45.0% | **97.5%** | **+52.5%** |
| **miscellaneous** | 35.0% | **95.0%** | 30.0% | **95.0%** | **+65.0%** |
| **project_related** | 35.0% | **95.0%** | 40.0% | **95.0%** | **+55.0%** |
| **semantic** | 20.0% | **83.0%** | 30.0% | **88.0%** | **+58.0%** |
