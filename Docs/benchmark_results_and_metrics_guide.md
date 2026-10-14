# EnterpriseRAGBench: Retrieval Benchmark Report & Metrics Guide

This document presents the complete results of **Experiment 01 (`dense_bge_m3`)** across all 500 questions of [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench) and provides an in-depth guide explaining **every metric**, its mathematical definition, why it matters in real-world enterprise RAG, and how to interpret the results.

---

## Part 1: Comprehensive Benchmark Scorecard (`dense_bge_m3`)

### 1.1 Experiment Overview
- **Retriever**: Dense Vector Search with `BAAI/bge-m3` (1024-dim, fp16 on CUDA).
- **Index**: Postgres 16 + pgvector HNSW cosine index (`chunks_embedding_hnsw_idx`, \(m=16, ef\_construction=64, ef\_search=64\)).
- **Corpus**: 511,958 unique documents partitioned into 548,718 chunks under `sentence_quarter_v1` (\(C=8192, T=2048, O=64\)).
- **Question Set**: 500 questions (470 retrieval-scorable; 30 unscorable `high_level` and `info_not_found`).
- **Speed**: Query encoding: **5.35s** (~10 ms/query) | HNSW Retrieval: **59.15s** (~118 ms/query across 548k vectors).

---

### 1.2 Ranked Information Retrieval (IR) Table

| Metric | K = 1 | K = 5 | K = 10 | K = 20 | Unbounded |
|---|---:|---:|---:|---:|---:|
| **Recall@K** | 0.2074 | 0.3196 | **0.3659** | **0.3918** | **0.3918** |
| **Precision@K** | **0.2702** | 0.0949 | 0.0553 | 0.0300 | — |
| **Hit@K** | 0.2702 | 0.3745 | **0.4191** | **0.4404** | — |
| **MRR@K** | 0.2702 | 0.3098 | **0.3156** | **0.3172** | **0.3172** (MRR) |

---

### 1.3 Connector Routing Metrics

Scored across the 9 enterprise tools (`slack`, `gmail`, `jira`, `linear`, `confluence`, `google_drive`, `hubspot`, `fireflies`, `github`):

| Metric | @ 1 | @ 5 | @ 10 | @ 20 | Unbounded |
|---|---:|---:|---:|---:|---:|
| **Source Accuracy** | 0.4553 | 0.2489 | 0.1787 | 0.1447 | 0.1447 |
| **Source Precision** | 0.5468 | 0.4491 | 0.4051 | 0.3743 | 0.3743 |
| **Source Recall** | 0.4910 | 0.6439 | **0.6960** | **0.7376** | **0.7376** |

---

### 1.4 Breakdown by Question Type (10 Categories)

| Question Type | Questions (\(n\)) | Recall@10 | Precision@10 | MRR | Hit@10 | Source Recall@10 |
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

### 1.5 Breakdown by Catalog Slices

| Slice | Questions (\(n\)) | Recall@10 | MRR | Hit@10 | Source Recall@10 |
|---|---:|---:|---:|---:|---:|
| `single_doc` (\(|G|=1\)) | 377 | 0.3581 | 0.2616 | 0.3581 | 0.7003 |
| `multi_doc` (\(|G|>1\)) | 93 | **0.3976** | **0.5424** | **0.6667** | 0.6785 |
| `multi_hop = True` | 133 | **0.4209** | **0.4947** | **0.6090** | 0.7150 |
| `multi_hop = False` | 337 | 0.3442 | 0.2471 | 0.3442 | 0.6884 |
| `multi_source = True` | 68 | 0.4041 | **0.5282** | **0.6471** | 0.6485 |
| `multi_source = False` | 402 | 0.3595 | 0.2815 | 0.3806 | 0.7040 |
| `temporal = True` | 254 | 0.4034 | 0.3803 | 0.4882 | 0.7053 |

---

## Part 2: What Are These Metrics & Why Do They Matter?

When evaluating search and RAG systems, no single metric tells the full story. We evaluate across **three levels**:
1. **Ranked Retrieval Quality** (Did we find the right documents, and where are they ranked?)
2. **Connector Routing Quality** (Did we search the right company systems?)
3. **End-to-End Generation Quality** (Is the final answer accurate and complete?)

```
┌────────────────────────────────────────────────────────┐
│                   Evaluation Hierarchy                 │
├────────────────────────────────────────────────────────┤
│  Level 1: Connector Routing (Source Accuracy / Recall) │
│             ▲  Did we search the right system?         │
│             │                                          │
│  Level 2: Document Retrieval (Recall@K, MRR, Hit@K)    │
│             ▲  Did we find the right document?         │
│             │                                          │
│  Level 3: Answer Generation (Correctness, Completeness)│
│                Did the LLM answer properly?            │
└────────────────────────────────────────────────────────┘
```

---

### 1. Recall@K (The Upper Bound of RAG)

#### Definition
$$\text{Recall@K}_q = \frac{|R_q@K \cap G_q|}{|G_q|}$$
- \(G_q\): Set of gold/expected reference documents for question \(q\).
- \(R_q@K\): Top-\(K\) documents retrieved by the system.
- \(\text{Recall@K}\): The fraction of all gold documents retrieved in the top-\(K\).

#### Why It Matters
In a RAG system, **the LLM can only read what the retriever provides**. 
If a question requires 2 documents to answer, and the retriever only finds 1 of them in the top-\(K\), the LLM is starved of evidence. It will either fail to answer or hallucinate. 
- **Recall@K is the glass ceiling of RAG**: The generator's factual completeness can never exceed the retriever's Recall@K.
- At \(K=10\), our baseline achieves **0.3659 (36.6%)**. At \(K=20\), it reaches **0.3918 (39.2%)**.

---

### 2. Precision@K (Context Window Hygiene)

#### Definition
$$\text{Precision@K}_q = \frac{|R_q@K \cap G_q|}{K}$$
- The fraction of the top-\(K\) retrieved items that are actually relevant.
- Note: Under standard IR conventions, the denominator is **strictly \(K\)**.

#### Why It Matters
While Recall measures *sufficiency*, Precision measures *noise*.
- If \(K=20\) and there is only 1 gold document, the maximum possible Precision@20 is \(1/20 = 0.05\) (5%). That is why Precision@K naturally decreases as \(K\) grows.
- **RAG Impact**: Feeding irrelevant, noisy documents to an LLM:
  1. Wastes token budget and increases latency/cost.
  2. Induces "distractor confusion" where the LLM quotes irrelevant details instead of the true answer.

---

### 3. Hit@K (Success Rate)

#### Definition
$$\text{Hit@K}_q = \begin{cases} 1 & \text{if } |R_q@K \cap G_q| \ge 1 \\ 0 & \text{otherwise} \end{cases}$$
- Did we retrieve **at least one** relevant document in the top-\(K\)?

#### Why It Matters
For single-document lookup questions (which make up 80%+ of typical enterprise queries), Hit@K is the bottom-line success rate:
- At \(K=1\): Hit@1 is **27.0%**. More than 1 in 4 queries gets the right document immediately at Rank 1.
- At \(K=10\): Hit@10 reaches **41.9%**.
- At \(K=20\): Hit@20 reaches **44.0%**.

---

### 4. MRR & MRR@K (Mean Reciprocal Rank — Position Sensitivity)

#### Definition
$$\text{RR}_q = \frac{1}{\text{rank}_q^*} \quad \text{where } \text{rank}_q^* \text{ is the rank of the first gold document}$$
$$\text{MRR} = \frac{1}{|Q|} \sum_{q} \text{RR}_q$$

- If the first gold document is at **Rank 1** \(\rightarrow \text{RR} = 1/1 = \mathbf{1.0}\)
- If the first gold document is at **Rank 2** \(\rightarrow \text{RR} = 1/2 = \mathbf{0.5}\)
- If the first gold document is at **Rank 4** \(\rightarrow \text{RR} = 1/4 = \mathbf{0.25}\)
- If no gold document is found \(\rightarrow \text{RR} = \mathbf{0.0}\)

#### Why It Matters
LLMs suffer from the well-documented **"Lost in the Middle"** phenomenon: they pay significantly more attention to documents placed at the very top (and bottom) of their context prompt than documents buried in the middle.
- A system with high Recall but terrible MRR (e.g., gold doc at rank 19) will often produce worse answers than a system that places the gold document at rank 1.
- Our baseline achieves an **MRR of 0.3172**, meaning on average, when a hit occurs, it appears very close to Rank 3.

---

### 5. Connector / Routing Metrics (Source Accuracy, Precision, Recall)

In modern enterprises, knowledge is fragmented across multiple siloed platforms: Slack, Jira, Confluence, Gmail, Google Drive, HubSpot, etc.

#### Definitions
- **Source Recall@K**: Did the retrieved list include documents from the correct tool?
  $$\text{SourceRecall}_q = \frac{|P_q@K \cap S_q|}{|S_q|}$$
- **Source Precision@K**: Of the tools represented in the retrieved list, what fraction was actually relevant?
  $$\text{SourcePrecision}_q = \frac{|P_q@K \cap S_q|}{|P_q@K|}$$
- **Source Accuracy@K**: Did the retrieved list match the gold tool set **exactly**?
  $$\text{SourceAccuracy}_q = 1 \text{ if } P_q@K == S_q \text{ else } 0$$

#### The Diagnostic Value of Source Metrics
Source metrics decouple **connector routing errors** from **document ranking errors**:
```text
SITUATION A: High Source Recall (80%) + Low Document Recall (30%)
Diagnosis: The retriever knows to look in Jira, but picks the wrong Jira tickets.
Remedy: Better chunking, hybrid keyword search, or ticket reranking.

SITUATION B: Low Source Recall (30%) + Low Document Recall (30%)
Diagnosis: The retriever doesn't even know which system contains the answer! 
Remedy: Add connector intent classification, source routing, or metadata pre-filtering.
```
In our experiment, **Source Recall@10 is 69.60%**. This proves our retriever is strong at navigating to the right tool, but needs better intra-tool precision.

---

### 6. End-to-End Generation Metrics (Spec §4)

When generation is connected, the benchmark evaluates the final answer text using LLM judges:

#### 1. Answer Correctness (Binary: 0 or 1)
- Evaluates whether the generated answer is factually consistent with the `gold_answer` without contradictions or hallucinations.

#### 2. Answer Completeness (\(0.0\) to \(1.0\))
- Checks how many of the discrete factual units in `answer_facts` are supported:
  $$\text{Completeness}_q = \frac{\# \text{ facts supported}}{\# \text{ total gold facts}}$$

#### 3. Overall Score (The Primary Leaderboard Metric)
$$\text{Overall Score}_q = \text{Correctness}_q \times \text{Completeness}_q$$
$$\text{Overall Score} = \text{mean}(\text{Overall Score}_q)$$
- **Crucial Rule**: If an answer is factually incorrect, it receives an Overall Score of **0.0**, regardless of how complete or detailed it is. Correctness is a non-negotiable prerequisite.

---

## Part 3: Deep Insights from Experiment 01

### 1. Where Pure Dense Retrieval Dominates
- **`constrained` (Recall@10 = 0.7667, MRR = 0.7439, Hit@10 = 83.3%)**:
  When queries have specific qualifiers (e.g. *"Show me closed Linear bugs related to memory leaks reported in Q3"*), the 1024-dim dense embedding effectively encodes the intersecting semantic constraints.
- **`project_related` (Hit@10 = 77.5%, MRR = 0.5698)**:
  Dense embeddings easily capture project topic clusters across multiple related tickets and documents.

### 2. The Critical Failure Mode: The `semantic` Slice
- On `semantic` queries (125 questions), pure dense search collapses:
  - **Recall@10**: **0.1200** (12%)
  - **MRR**: **0.0649**
  - **Hit@10**: **0.1200** (only 15 out of 125 questions found a gold document!)
- **Why?**: These questions use heavy paraphrase, synonym drift, and oblique descriptions (e.g. asking about "cost blowout" when the document says "unexpected infrastructure egress charge spike"). Single dense query vectors get drowned out by 548k noisy enterprise documents.

---

## Part 4: Roadmap to Higher Scores (Upcoming Experiments)

| Technique | Target Metric to Improve | How It Works |
|---|---|---|
| **BM25 / Sparse Search + Dense RRF** | `Recall@10`, `Hit@10`, `semantic` slice | Catches exact keyword hits, error codes, and IDs that dense vectors overlook; fused with Reciprocal Rank Fusion. |
| **HyDE / Query Expansion** | `semantic` slice (MRR 0.06 \(\rightarrow\) 0.35+) | Uses an LLM to generate hypothetical passages or alternate queries before embedding. |
| **Cross-Encoder Reranking** (`bge-reranker-v2-m3`) | `MRR@10`, `Precision@10` | Jointly encodes (query, doc) pairs to push the true gold document from Rank 8 to Rank 1. |
| **LLM Generation with Abstention** | `Overall Score`, `info_not_found` | Explicit prompt engineering to refuse unanswerable queries and summarize evidence into complete facts. |
