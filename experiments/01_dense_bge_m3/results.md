# Retrieval Evaluation Results

## 1. Ranked Retrieval Metrics (Aggregated)

| Metric | K=1 | K=5 | K=10 | K=20 | Unbounded |
|---|---:|---:|---:|---:|---:|
| **Recall@K** | 0.2074 | 0.3196 | 0.3659 | 0.3918 | 0.3918 |
| **Precision@K** | 0.2702 | 0.0949 | 0.0553 | 0.0300 | — |
| **Hit@K** | 0.2702 | 0.3745 | 0.4191 | 0.4404 | — |
| **MRR@K** | 0.2702 | 0.3098 | 0.3156 | 0.3172 | **0.3172** (MRR) |

## 2. Connector Routing Metrics

| Metric | @1 | @5 | @10 | @20 | Unbounded |
|---|---:|---:|---:|---:|---:|
| **Source Accuracy** | 0.4553 | 0.2489 | 0.1787 | 0.1447 | 0.1447 |
| **Source Precision** | 0.5468 | 0.4491 | 0.4051 | 0.3743 | 0.3743 |
| **Source Recall** | 0.4910 | 0.6439 | 0.6960 | 0.7376 | 0.7376 |

## 3. Breakdown by Question Type

| Question Type | n | Recall@10 | Precision@10 | MRR | Hit@10 | Source Rec@10 |
|---|---:|---:|---:|---:|---:|---:|
| `basic` | 175 | 0.4400 | 0.0440 | 0.3203 | 0.4400 | 0.7600 |
| `completeness` | 20 | 0.2869 | 0.1600 | 0.5417 | 0.6500 | 0.7558 |
| `conflicting_info` | 20 | 0.4000 | 0.0800 | 0.4125 | 0.4500 | 0.5750 |
| `constrained` | 30 | 0.7667 | 0.1000 | 0.7439 | 0.8333 | 0.8333 |
| `high_level` | 10 | *(unscorable)* | — | — | — | — |
| `info_not_found` | 20 | *(unscorable)* | — | — | — | — |
| `intra_document_reasoning` | 40 | 0.4750 | 0.0475 | 0.3840 | 0.4750 | 0.8000 |
| `miscellaneous` | 20 | 0.4000 | 0.0400 | 0.2681 | 0.4000 | 0.7000 |
| `project_related` | 40 | 0.4060 | 0.1575 | 0.5698 | 0.7750 | 0.6871 |
| `semantic` | 125 | 0.1200 | 0.0120 | 0.0649 | 0.1200 | 0.5520 |

## 4. Breakdown by Slices

| Slice | n | Recall@10 | MRR | Hit@10 | Source Rec@10 |
|---|---:|---:|---:|---:|---:|
| `single_doc` | 377 | 0.3581 | 0.2616 | 0.3581 | 0.7003 |
| `multi_doc` | 93 | 0.3976 | 0.5424 | 0.6667 | 0.6785 |
| `multi_hop_true` | 133 | 0.4209 | 0.4947 | 0.6090 | 0.7150 |
| `multi_hop_false` | 337 | 0.3442 | 0.2471 | 0.3442 | 0.6884 |
| `multi_source_true` | 68 | 0.4041 | 0.5282 | 0.6471 | 0.6485 |
| `multi_source_false` | 402 | 0.3595 | 0.2815 | 0.3806 | 0.7040 |
| `temporal_true` | 254 | 0.4034 | 0.3803 | 0.4882 | 0.7053 |
