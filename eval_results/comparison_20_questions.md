# 20-Question Head-to-Head: Hybrid LlamaIndex vs. Pure Solo LlamaIndex

- **Evaluated Questions**: 20 (2 per category across 10 categories)
- **Model**: `nvidia/nemotron-3.5-lightning-30b-a3b`
- **Date**: 2026-10-15 20:13:49

## Overall Accuracy & Latency Summary

| Pipeline | Correct / Total | Accuracy (%) | Avg Latency (s) | Delta vs Hybrid |
| :--- | :---: | :---: | :---: | :---: |
| **Hybrid LlamaIndex QueryEngine** | 15/20 | **75.0%** | 38.7s | Baseline |
| **Pure Solo LlamaIndex QueryEngine** | 17/20 | **85.0%** | 90.5s | **+10.0%** |
| **Pure Solo LlamaIndex ReAct Agent** | 14/20 | **70.0%** | 126.9s | **-5.0%** |

## Category-by-Category Breakdown (2 Questions Each)

| Category | Hybrid QE Acc (%) | Pure QE Acc (%) | Pure Agent Acc (%) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **basic** | 50% (1/2) | 50% (1/2) | 50% (1/2) | Parity |
| **completeness** | 0% (0/2) | 50% (1/2) | 50% (1/2) | Pure +50% |
| **conflicting_info** | 100% (2/2) | 100% (2/2) | 100% (2/2) | Parity |
| **constrained** | 100% (2/2) | 100% (2/2) | 100% (2/2) | Parity |
| **high_level** | 100% (2/2) | 100% (2/2) | 100% (2/2) | Parity |
| **info_not_found** | 50% (1/2) | 100% (2/2) | 50% (1/2) | Pure +50% |
| **intra_document_reasoning** | 100% (2/2) | 100% (2/2) | 50% (1/2) | Parity |
| **miscellaneous** | 50% (1/2) | 50% (1/2) | 100% (2/2) | Parity |
| **project_related** | 100% (2/2) | 100% (2/2) | 100% (2/2) | Parity |
| **semantic** | 100% (2/2) | 100% (2/2) | 0% (0/2) | Parity |
