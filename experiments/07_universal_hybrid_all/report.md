# EnterpriseRAG-Bench: Universal Multi-Stream Hybrid Evaluation Report

- **Model**: `nvidia/nemotron-3.5-lightning-30b-a3b`
- **Retrieval Engine**: Dense BGE-M3 (HNSW ef=128) + Sparse BM25 (SQLite FTS5) via Multi-Stream RRF (k=20, w_sparse=1.2) + Query Expansion
- **Evaluated Categories**: All 10 Benchmark Categories
- **Evaluated Samples**: 300 Questions (Capped at up to 50 samples per category)
- **Baseline Sample Comparison**: 190 Questions (20 samples per category baseline slice)

---

## 1. Executive Summary

| Evaluation Slice | Evaluated Samples | Mean Hit@5 | Mean Factual Correctness | Mean Completeness | Overall Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Dense BGE-M3 Only)** | 190 | 49.4% | 50.5% | 7.4% | 0.074 |
| **Universal Hybrid (20 Samples/Cat)** | 190 | **93.1%** | **82.6%** | **48.2%** | **0.421** |
| **Universal Hybrid (50 Samples/Cat)** | 300 | **84.1%** | **77.3%** | **41.5%** | **0.363** |

---

## 2. Category-by-Category Comparison: Baseline vs. 20-Sample vs. 50-Sample

| Category | Baseline Correctness (n=20) | **20-Sample Hybrid Correctness** | **50-Sample Hybrid Correctness** | Baseline Hit@5 | **20-Sample Hybrid Hit@5** | **50-Sample Hybrid Hit@5** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`miscellaneous`** | 30.0% | **80.0%** (20/20) | **80.0%** (16/20) | 35.0% | **95.0%** | **95.0%** |
| **`intra_document_reasoning`** | 45.0% | **95.0%** (19/20) | **90.0%** (36/40) | 35.0% | **95.0%** | **97.5%** |
| **`conflicting_info`** | 55.0% | **85.0%** (17/20) | **85.0%** (17/20) | 45.0% | **90.0%** | **90.0%** |
| **`basic`** | 45.0% | **95.0%** (19/20) | **76.0%** (38/50) | 50.0% | **85.0%** | **74.0%** |
| **`semantic`** | 30.0% | **95.0%** (19/20) | **76.0%** (38/50) | 20.0% | **100.0%** | **62.0%** |
| **`constrained`** | 70.0% | **85.0%** (17/20) | **73.3%** (22/30) | 80.0% | **100.0%** | **96.7%** |
| **`project_related`** | 40.0% | **70.0%** (14/20) | **72.5%** (29/40) | 70.0% | **100.0%** | **95.0%** |
| **`completeness`** | 45.0% | **55.0%** (11/20) | **55.0%** (11/20) | 65.0% | **80.0%** | **80.0%** |
| **`info_not_found`** | 85.0% | **90.0%** (18/20) | **90.0%** (18/20) | N/A | N/A | N/A |
| **`high_level`** | 70.0% | **70.0%** (7/10) | **70.0%** (7/10) | N/A | N/A | N/A |
| **OVERALL AVERAGE** | **50.5%** | **82.6%** | **77.3%** | **49.4%** | **93.1%** | **84.1%** |
