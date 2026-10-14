# Experiment 01 — Dense BGE-M3 Baseline (`dense_bge_m3`)

First end-to-end retrieval baseline on the 500 EnterpriseRAG-Bench catalog questions across the 511,958 document corpus.

---

## 1. Hypothesis

A dense retrieval baseline using `BAAI/bge-m3` embeddings over the `sentence_quarter_v1` chunk index will establish the ground-truth retrieval lower bound:
- Fast HNSW sub-linear vector retrieval (<200 ms/query across ~550k chunks).
- Strong performance on specific queries (`constrained`, `project_related`).
- Potential vulnerability on paraphrased questions (`semantic`) and multi-hop questions where sparse keyword overlap or query expansion may be needed.

---

## 2. Configuration & Method

- **Index Build**: `chunk_v1` (511,958 unique documents, 548,718 chunks).
- **Chunking**: `sentence_quarter_v1` (\(C=8192, T=2048, O=64\)) with per-source atomic rules.
- **Embedding Model**: `BAAI/bge-m3` (1024-dim, cls pooling, L2 normalized, float16).
- **Index**: Postgres pgvector HNSW (`chunks_embedding_hnsw_idx`, `m=16, ef_construction=64`).
- **Search Beam**: `hnsw.ef_search = 64`.
- **Top-K**: 20 collapsed unique documents per question (`fetch_k = 100`).
- **Collapse Rule**: Spec §6.5 (chunk hits collapsed to unique `doc_id` based on highest chunk score).

---

## 3. Results Summary

Total Questions: **500** | Retrieval-Scorable: **470** (excludes `high_level` [10] and `info_not_found` [20]).

### 3.1 Ranked IR Metrics

| Metric | K=1 | K=5 | K=10 | K=20 |
|---|---:|---:|---:|---:|
| **Recall@K** | 0.2074 | 0.3196 | **0.3659** | **0.3918** |
| **Precision@K** | 0.2702 | 0.0949 | 0.0553 | 0.0300 |
| **Hit@K** | 0.2702 | 0.3745 | **0.4191** | **0.4404** |
| **MRR@K** | 0.2702 | 0.3098 | **0.3156** | **0.3172** (MRR) |

*Total runtime for all 500 queries: **5.35s query encoding + 59.15s HNSW retrieval** (~120 ms/query).*

### 3.2 Connector Routing Metrics

| Metric | @1 | @5 | @10 | @20 |
|---|---:|---:|---:|---:|
| **Source Accuracy** | 0.4553 | 0.2489 | 0.1787 | 0.1447 |
| **Source Precision** | 0.5468 | 0.4491 | 0.4051 | 0.3743 |
| **Source Recall** | 0.4910 | 0.6439 | **0.6960** | **0.7376** |

---

## 4. Breakdown by Question Type

| Question Type | n | Recall@10 | Precision@10 | MRR | Hit@10 | Source Rec@10 |
|---|---:|---:|---:|---:|---:|---:|
| `constrained` | 30 | **0.7667** | 0.1000 | **0.7439** | **0.8333** | 0.8333 |
| `project_related` | 40 | **0.4060** | 0.1575 | **0.5698** | **0.7750** | 0.6871 |
| `completeness` | 20 | 0.2869 | 0.1600 | **0.5417** | **0.6500** | 0.7558 |
| `intra_document_reasoning` | 40 | 0.4750 | 0.0475 | 0.3840 | 0.4750 | 0.8000 |
| `basic` | 175 | 0.4400 | 0.0440 | 0.3203 | 0.4400 | 0.7600 |
| `conflicting_info` | 20 | 0.4000 | 0.0800 | 0.4125 | 0.4500 | 0.5750 |
| `miscellaneous` | 20 | 0.4000 | 0.0400 | 0.2681 | 0.4000 | 0.7000 |
| `semantic` | 125 | **0.1200** | 0.0120 | **0.0649** | **0.1200** | 0.5520 |
| `high_level` | 10 | *(unscorable)* | — | — | — | — |
| `info_not_found` | 20 | *(unscorable)* | — | — | — | — |

---

## 5. Breakdown by Slices

| Slice | n | Recall@10 | MRR | Hit@10 | Source Rec@10 |
|---|---:|---:|---:|---:|---:|
| `single_doc` | 377 | 0.3581 | 0.2616 | 0.3581 | 0.7003 |
| `multi_doc` | 93 | 0.3976 | **0.5424** | **0.6667** | 0.6785 |
| `multi_hop_true` | 133 | **0.4209** | **0.4947** | **0.6090** | 0.7150 |
| `multi_hop_false` | 337 | 0.3442 | 0.2471 | 0.3442 | 0.6884 |
| `multi_source_true` | 68 | 0.4041 | **0.5282** | **0.6471** | 0.6485 |
| `temporal_true` | 254 | 0.4034 | 0.3803 | 0.4882 | 0.7053 |

---

## 6. Key Insights & Diagnostic Takeaways

1. **Connector Routing is Strong**: In **69.6%** of questions, the retriever successfully surfaces the correct enterprise connector in the top 10 (and 73.8% in top 20).
2. **Superior Constrained & Multi-Doc Hit Rates**:
   - `constrained` queries achieve **83.3% Hit@10** and **0.7439 MRR**.
   - `project_related` queries achieve **77.5% Hit@10** and **0.5698 MRR**.
   - `multi_hop` queries achieve **60.9% Hit@10**.
3. **The `semantic` Slice Failure Mode (Critical Bottleneck)**:
   - For `semantic` questions (125 queries), dense retrieval collapses to **Recall@10 = 0.1200** and **MRR = 0.0649**.
   - Why: These questions feature heavy paraphrase and high conceptual drift from the raw corporate text.
   - **Actionable Next Step**: Experiment 02 should evaluate **Hybrid Retrieval (Dense + Sparse BM25 via RRF)** and/or **Query Expansion (HyDE / Multi-Query)** specifically targeting the `semantic` slice.
