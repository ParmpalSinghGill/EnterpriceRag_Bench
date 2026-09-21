# Ingestion Spec — EnterpriseRAGBench

This document is the **ingestion contract** for this repo: how we load the corpus, how we **split** each `source_type`, what must stay in **one chunk**, which open-weight context budget we target, and **how we check** that an index build is valid before scoring under [`evaluation_spec.md`](evaluation_spec.md).

This is not a menu of options. Default runs follow **§5–§8** exactly. Ablations may change embedder or index backend, but must keep identity + atomic-unit rules unless the experiment README explicitly marks the run **non-comparable**.

Upstream:

- Dataset: [`onyx-dot-app/EnterpriseRAG-Bench`](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)
- Methodology: [EnterpriseRAG-Bench `methodology.md`](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/main/methodology.md)
- Paper: [arXiv:2605.05253](https://arxiv.org/abs/2605.05253)

---

## 1. Goal

Produce a searchable index where:

1. Every indexed unit maps to a corpus **`doc_id`** (`dsid_…`)
2. Answer files emit **ranked unique `doc_id`s** (never chunk ids)
3. Chunks keep **enough local context** that a hit is answerable without needing a sibling chunk for the same atomic fact
4. Builds are **auditable** via the checks in §9

---

## 2. Paths

```text
Docs/ingestion_spec.md          ← this contract
Docs/evaluation_spec.md         ← scoring contract
Data/Prepocessing.py            ← explore only (not the indexer)
Data/benchmark_catalog.jsonl    ← question labels (eval only; never used at ingest)
indexes/<build_id>/             ← gitignored index + manifest + audit report
experiments/<technique>/        ← points at a build_id
```

Artifact layout:

```text
indexes/<build_id>/
  manifest.json
  audit_report.json          # output of §9 checks
  docid_map.jsonl            # chunk_id → doc_id, source_type, chunk_index, n_tokens
  sparse/ | dense/
```

---

## 3. Corpus to ingest

| Asset | Location | Rows | Ingest? |
|---|---|---:|---|
| Documents | HF `documents` / `test` | 511,962 | **Yes — all rows** for comparable runs |
| Questions | HF `questions` / `test` | 500 | **No** |
| Catalog | `Data/benchmark_catalog.jsonl` | 500 | **No** (eval filters only) |

```python
from datasets import load_dataset
documents = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "documents")["test"]
```

Schema (only these fields exist in the release):

| Field | Rule |
|---|---|
| `doc_id` | Primary key. Preserve exactly. Never mint new ids. |
| `source_type` | Required metadata + routing key for §7 |
| `title` | Always available; prepended per §6.3 |
| `content` | Body; split only under §6–§7 |

`doc_id` uniqueness: 511,958 unique / 511,962 rows. On duplicate ids, **merge into one logical document** (concatenate content with `\n\n` in row order) before chunking; do not create a second id.

---

## 4. Analysis summary (why these split rules)

Token lengths below use **≈ chars/4** as a planning estimator. **Implementation must count with the chosen model tokenizer** (§5.2).

### 4.1 Length by `source_type` (full corpus)

| `source_type` | n | median toks | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| `slack` | 285,605 | 809 | 1,236 | 1,686 | 3,166 |
| `hubspot` | 15,017 | 756 | 1,089 | 1,412 | 2,084 |
| `github` | 8,052 | 1,154 | 1,609 | 1,930 | 3,443 |
| `linear` | 35,308 | 1,260 | 1,753 | 2,359 | 4,327 |
| `jira` | 6,120 | 1,389 | 1,700 | 2,045 | 2,608 |
| `gmail` | 121,390 | 1,759 | 2,205 | 2,662 | 4,358 |
| `google_drive` | 25,108 | 1,751 | 2,286 | 3,115 | 5,666 |
| `confluence` | 5,189 | 2,394 | 3,118 | 4,318 | 6,212 |
| `fireflies` | 10,173 | 2,770 | 3,678 | 5,813 | 10,329 |

### 4.2 Observed shapes (sampled)

| Type | Dominant structure | What must stay together |
|---|---|---|
| `slack` | `speaker: message` turns (~18 turns median) | A single speaker turn (do not cut mid-message) |
| `gmail` | Often one long body; sometimes `From/To/Date/Subject` + body | Header block + continuous reply body; do not orphan Subject from body when splitting |
| `jira` | Labeled sections: `description`, `impact`, `steps_to_reproduce`, `logs`, `investigation_notes`, `resolution`, `comments` | One labeled section = one atomic unit |
| `linear` | `description`, `acceptance_criteria`, `progress_notes`, `decisions`, `comments` | One labeled section = one atomic unit |
| `github` | `description` (+ Motivation/Summary), `review_comments`, `release_notes`, checklist | Keep `description` intact if ≤ soft target; `review_comments` may pack by comment |
| `hubspot` | `notes`, `timeline`, `next_step`, `blockers`, `use_case_summary` | One labeled section = one atomic unit |
| `fireflies` | `summary:` then `transcript:` with `[mm:ss] Speaker:` lines | **Always** emit summary as its own chunk(s); transcript packs by utterance |
| `confluence` | Markdown/`##` sections, runbook-style | Prefer `##` / labeled section boundaries |
| `google_drive` | Prose + `---` / heading-like blocks | Prefer blank-line / heading blocks, else sentences |

### 4.3 Fraction of docs that need a split at common budgets

Approx % of docs with content tokens **>** budget (planning estimate):

| Budget | slack | hubspot | jira | github | gmail | drive | confluence | fireflies |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 86% | 91% | 98% | 99% | 100% | 100% | 99% | 100% |
| 1024 | 25% | 15% | 93% | 66% | 99% | 96% | 97% | 100% |
| **2048** | **0.2%** | **0%** | **1%** | **0.4%** | **20%** | **24%** | **86%** | **92%** |
| 4096 | 0% | 0% | 0% | 0% | 0% | 0.1% | 1.7% | 5% |
| 8192 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0.1% |

**Judgment:** 512/1024 over-fragment tickets and runbooks (hurts `intra_document_reasoning` and multi-field Jira/Linear answers). Full 8192 almost never splits, but packs unrelated Slack turns / long transcripts into one vector (hurts precision). **2048** is the smallest budget where Slack/HubSpot/Jira/GitHub/Linear are almost always **one chunk**, while still forcing Fireflies/Confluence (and some Gmail/Drive) to split on natural boundaries.

---

## 5. Initial Embedding & Chunking Configuration

**`BAAI/bge-m3` is the initial local embedding model.** This is an experimental starting configuration, not a claim that it is the optimal embedding model.

### 5.1 Embedder class

Initial dense ingest uses an **open-weight long-context embedder with max sequence length C = 8192**. Compatible alternatives for later ablations include:

| Model | Open weights | Max seq len |
|---|---|---:|
| `BAAI/bge-m3` (**initial**) | yes | 8192 |
| `nomic-ai/nomic-embed-text-v1.5` (or v1) | yes | 8192 |
| `jinaai/jina-embeddings-v3` | yes | 8192 |

Shorter models (e.g. classic `bge-*-en-v1.5` at **512**) are allowed only as a **named ablation** and must recompute soft target as `T = C/4` (→ 128) **or** declare a different packing policy in the experiment README and mark the run non-default.

### 5.2 Initial packing numbers

| Symbol | Value | Meaning |
|---|---:|---|
| `C` | **8192** | Hard max tokens per embedded string (model context) |
| `T` | **2048** | Initial packing target = **one quarter of C** (`C/4`) — not a hard maximum |
| `O` | **64** | Sentence/unit overlap when a doc is split into multiple chunks |
| `H` | title + source tag | Header prepended to every chunk (§6.3); counted inside `C`/`T` |

**2048 tokens is the initial chunk packing target, not a hard maximum.** Atomic units may exceed 2048 tokens when they remain semantically coherent, provided they fit within the model's 8192-token context.

**"Remains coherent" (operational definition):** an atomic unit `ui` from §7 is considered coherent-as-is whenever it is `≤ C`. There is no additional judgment call — §7's separators are defined precisely so that every unit they produce (a Slack turn, a labeled Jira/Linear/Hubspot/Github section, a Confluence `##` block, a Fireflies summary or transcript utterance) is by construction a single semantic fact. The only units force-split below `C` are those exceeding it (§6.4 step "unit_tok > C" → hard-split on sentence/word boundaries, §7.3). Implementers must not add ad hoc coherence heuristics beyond the §7 separators.

**Strategy name:** **sentence-packed quarters** (`pack=sentence_quarter_v1`).

Meaning:

1. Prefer **whole document** if `tokens(header + content) ≤ T`
2. Else split into **atomic units** (§7), then **pack complete units** toward size **≈ T**
3. If a single atomic unit is `> T` but `≤ C`, emit it as its own chunk (do **not** subdivide)
4. If a single atomic unit is `> C`, hard-split on sentence → word boundaries (last resort only)

Token count = **tokenizer of the chosen embedder** (not chars/4). chars/4 is only for planning in this doc.

### 5.3 Why “quarter” (starting point)

- Matches open-weight 8k embedders without pretending 512-token models are the baseline
- Keeps ~¾ of headroom unused so headers, instruction prefixes, and tokenizer variance do not silently truncate
- Aligns with corpus: at 2048, high-volume Slack stays 1 chunk; only long formal / meeting docs systematically multi-chunk
- Atomic coherence still wins over the soft target when a unit is larger than T but ≤ C

---

## 6. Global chunking algorithm (`sentence_quarter_v1`)

Apply in order for every document.

### 6.1 Normalize (allowed only)

1. Unicode NFC; drop NULs; newlines → `\n`
2. Trim trailing whitespace per line
3. Do **not** summarize, translate, or rewrite content
4. Do **not** drop any `source_type`

### 6.2 Build atomic units

Call the **source-specific separator** in §7 → ordered list `U = [u0, u1, …]`.

Each `ui` is an atomic unit that should **not** be subdivided unless it exceeds **`C`**. Units larger than `T` but `≤ C` are emitted as their own chunk.

The **only blocking token requirement** is: no emitted chunk (header + body) may exceed **`C = 8192`** tokens according to the **actual embedder tokenizer**. `T` is a packing preference only.

### 6.3 Header (required on every emitted chunk)

```text
[{source_type}] {title}

{chunk_body}
```

If `title` is empty, use `doc_id` as the title substitute. Header tokens count toward `T`/`C`.

### 6.4 Pack

```text
chunks = []
cur = []
cur_tok = tokens(header)   # reserved; body packing accounts for header once per chunk

for unit in U:
  unit_tok = tokens(unit)
  if unit_tok > C:
      flush cur
      hard_split_unit(unit) into pieces ≤ T (sentence then word)
      continue
  if unit_tok > T:
      flush cur
      emit unit as its own chunk (still with header)
      continue
  if cur_tok + unit_tok > T and cur:
      flush cur
      # overlap: prepend last O tokens worth of text from previous chunk body (sentence-aligned)
  append unit to cur; cur_tok += unit_tok

flush cur
```

**Overlap rule:** when starting chunk `k>0`, prepend the **last complete sentence(s)** from chunk `k-1` body whose total tokens ≤ `O` (64). Do not overlap across different `doc_id`s.

### 6.5 Collapse at retrieval (required)

Hits are chunks → emit **unique `doc_id`**, best rank wins (highest score / earliest rank). Tie-break: lower `chunk_index`.

---

## 7. Per-`source_type` separation (atomic units)

### 7.1 Decision table

| `source_type` | Separator | Pack target | When to split | Must-keep-in-one-chunk |
|---|---|---|---|---|
| `slack` | speaker turns | whole thread if ≤T | only if thread >T | each `name: …` turn |
| `hubspot` | labeled sections | whole doc if ≤T | rare | each `notes` / `timeline` / `next_step` / `blockers` / … block |
| `jira` | labeled sections | whole ticket if ≤T | rare (~1%) | each `description`, `impact`, `logs`, `resolution`, … |
| `linear` | labeled sections | whole issue if ≤T | rare | each `description`, `acceptance_criteria`, … |
| `github` | PR fields | whole PR if ≤T | rare | full `description`; each review comment; full `release_notes` if ≤T else by paragraph |
| `gmail` | header block + body paragraphs/sentences | whole mail if ≤T | ~20% | `From/To/Cc/Date/Subject` as one unit; then sentence-pack body |
| `google_drive` | heading / `---` / blank-line blocks, else sentences | whole doc if ≤T | ~24% | a heading section; else consecutive sentences packed to T |
| `confluence` | `##` / `#` sections, else labeled blocks, else sentences | usually multi-chunk | ~86% | one markdown section per unit when present |
| `fireflies` | **mandatory two-phase** | always split summary vs transcript | ~92% | entire `summary` (split only if summary >T); each transcript utterance |

### 7.2 Rules by type

#### `slack`

**Corpus check (full `slack` split, 285,605 docs):** speaker-like prefixes are ubiquitous (285,577 docs match a turn pattern). Empirically, real speaker tokens are short — length median **7**, p90 **15**, p99 **32** — dominated by first names (`maria`, `sam`, `maya`, …) and bots (`deploy-bot`, `ops-bot`, `incident-bot`, `infra-bot`). A naïve `^([^\n:]{1,40}):\s` is **too loose**: it also matches action-item bullets, code, curl headers, and phrases like `owner:`, `Error:`, `Note:` (tens of thousands of “names” longer than 40 were almost all false positives under an unbounded `{1,200}` probe). Speaker display names with an embedded `:` do **not** appear in this corpus (and cannot with a colon-terminated pattern).

**Atomic unit regex (required):**

```text
^(?P<speaker>[A-Za-z][\w.@-]{0,38}):\s
```

Rules:

- Unit = that line through the line before the next match (multi-line bodies, code fences, stack traces stay in the turn)
- Reject / do not treat as a new turn if `speaker` is in the non-speaker denylist: `note`, `error`, `stack`, `warning`, `summary`, `status`, `subject`, `impact`, `expected`, `observed`, `input`, `output`, `logs`, `log`, `date`, `from`, `to`, `cc`, `description`, `context`, `example`, `steps`, `scope`, `docs`, `rollout`, `workaround`, `hypothesis`, `primary` (case-insensitive)
- `deploy-bot: deploy-bot: …` style double prefixes are still **one** turn (first match wins)
- If a doc has **no** speaker matches → one atomic unit = full content
- Prefer whole thread as **one chunk** when `tokens(header+content) ≤ T` (typical case; thread-level multi-chunk is rare)

Do **not** use unbounded `[^\n:]+` before `:`; the 40-char cap is evidence-backed for real speakers and reduces false turns.

#### `jira` / `linear` / `hubspot` / `github`

- Parse top-level labels: lines matching `^[a-zA-Z][a-zA-Z0-9_ /\-]{0,40}:\s` at section starts (case-insensitive known keys below)
- Known keys (non-exhaustive; unknown `label:` starts a new unit too):
  - **jira:** `description`, `impact`, `steps_to_reproduce`, `logs`, `investigation_notes`, `workaround`, `resolution`, `resolution_notes`, `comments`, `environment`
  - **linear:** `description`, `goal`, `scope`, `acceptance_criteria`, `tasks`, `decisions`, `progress_notes`, `next_steps`, `comments`, `risk`
  - **hubspot:** `use_case_summary`, `notes`, `recent_activities`, `timeline`, `next_step`, `blockers`
  - **github:** `description`, `review_comments`, `release_notes`
- Inside `review_comments`, sub-split on `Name:` reviewer lines if the whole comments block > T
- Prefer packing **all sections into one chunk** when ≤ T so description + resolution stay co-located

#### `gmail`

1. If present, unit0 = contiguous header lines (`From`, `To`, `Cc`, `Date`, `Subject`, `Attachments`)
2. Remainder = body; split on blank lines into paragraphs; if a paragraph > T, sentence-split
3. Pack paragraphs/sentences to ≤ T; **repeat Subject line in the header** of every body chunk (already covered by §6.3 title; if Subject differs from title, append `Subject: …` under the header)

#### `google_drive` / `confluence`

1. Split on markdown headings `^#{1,3}\s` or on lines that are only `---`
2. Else split on `\n\n` blocks
3. Else sentence-pack
4. Pack units to ≤ T

#### `fireflies` (strict)

1. Split content into **summary region** and **transcript region** on a line matching `^transcript:\s*` (case-insensitive)
2. Summary region (everything before transcript, including `summary:`, `topics:`, attendees, action_items):
   - Prefer **one chunk** for the whole summary if ≤ T; if summary is `> T` but `≤ C`, still emit as **one chunk**
   - Subdivide summary only if it exceeds **C**
3. Transcript region:
   - Atomic unit = one utterance `^\[\d{1,2}:\d{2}(?::\d{2})?\]\s*[A-Za-z][\w.@-]{0,38}:\s*` … until next timestamp utterance
   - Pack utterances toward ≈ T; a single utterance `> T` and `≤ C` stays one chunk
4. Never pack summary sentences into the same chunk as transcript utterances (keeps meeting decisions retrievable without diluting them in dialogue)

### 7.3 Sentence splitter (shared)

Used when a unit must be subdivided:

1. Split on `(?<=[.!?])\s+` and on `\n{2,}`
2. Keep numbered list items (`^\d+[.)]\s`) attached to the prior sentence when the item is < 40 tokens
3. Never break inside fenced code blocks (` ``` ` … ` ``` `); treat the fence as one unit

---

## 8. What “required context in one chunk” means

A chunk is valid only if a reader could use **that chunk alone** for the atomic fact it contains:

| Situation | Correct | Incorrect |
|---|---|---|
| Jira `resolution:` | Entire resolution section in one unit | Resolution split mid-sentence across chunks without overlap |
| Slack decision in one turn | Full `alice: …` turn | Cut after 80 tokens mid-message |
| Fireflies action item in summary | In a summary chunk | Only in a random transcript window |
| Confluence “Rollback steps” under `##` | Whole `## Rollback` section as unit (pack with neighbors only if ≤T) | Heading in chunk A, steps in chunk B with no overlap |
| GitHub motivation inside `description` | Keep with description | Description motivational bullets orphaned from Summary |

Multi-doc questions (`project_related`, etc.) still need **multiple parent docs**; this spec only guarantees intra-doc atomic coherence.

---

## 9. Checking strategy (how we verify)

Every index build must write `audit_report.json`. A build is **PASS** only if all **blocking** checks pass.

### 9.1 Blocking checks

| ID | Check | How |
|---|---|---|
| B1 | Row coverage | `n_docs_indexed == 511962` (or declared subset + `comparable=false`) |
| B2 | Id integrity | Every input `doc_id` appears in ≥1 chunk; no unknown `doc_id`s |
| B3 | Token ceiling (**only blocking token rule**) | No chunk may exceed `C=8192` tokens: `max(tokens(header+body)) ≤ C` for all chunks, counted with the **actual embedder tokenizer** |
| B5 | Atomic slack | Random sample 200 `slack` multi-chunk docs: no chunk boundary falls inside a speaker turn (using §7.2 slack regex + denylist) |
| B6 | Atomic fireflies | 100% of `fireflies` docs: no chunk mixes `region=summary` and `region=transcript` |
| B7 | Jira/Linear labels | Sample 100 split tickets: boundaries only at section starts or sentence boundaries inside an oversized section |
| B8 | Collapse map | `docid_map.jsonl` has unique `chunk_id`; every row has `doc_id`, `source_type`, `chunk_index`, `n_tokens` |
| B9 | No question leakage | Build inputs exclude HF questions / catalog files |

### 9.2 Non-blocking (warn)

| ID | Check | Expectation |
|---|---|---|
| W1 | Split rate by source | Approx: slack/hubspot/jira/github/linear **&lt; 5%** multi-chunk; gmail/drive **~15–30%**; confluence/fireflies **&gt; 80%** |
| W2 | Empty bodies | Count docs with empty `content` (still indexed via title header) |
| W3 | Duplicate ids merged | Report how many duplicate `doc_id`s were merged |
| W4 | Mean chunks/doc | Overall should be close to **1.0–1.3** at T=2048 |
| W5 | Packing target drift | Report median / p95 chunk tokens globally and per `source_type`. `T=2048` is a preference; units `> T` and `≤ C` are allowed and expected for long Confluence / Fireflies / some Gmail–Drive docs |

### 9.3 Audit procedure

```bash
# After ingest
python -m experiments.<technique>.audit_ingest \
  --index indexes/<build_id> \
  --documents-split test \
  --out indexes/<build_id>/audit_report.json
```

Minimum `audit_report.json` fields:

```json
{
  "build_id": "...",
  "comparable": true,
  "pack": "sentence_quarter_v1",
  "C": 8192,
  "T": 2048,
  "O": 64,
  "embedder": "BAAI/bge-m3",
  "tokenizer": "BAAI/bge-m3",
  "n_docs_indexed": 511962,
  "n_chunks": 0,
  "blocking": {"B1": "pass", "B2": "pass"},
  "warnings": {"W1": {"slack_multichunk_pct": 0.2}},
  "token_stats": {"median": 0, "p95": 0, "max": 0}
}
```

Do not score a run whose blocking status is not all `pass`.

---

## 10. Sparse / BM25 (future work)

Sparse/BM25 retrieval is reserved for a later retrieval experiment. When introduced, it should reuse the same atomic chunk representation unless explicitly defined as an ablation.

---

## 11. Manifest (required)

```json
{
  "dataset": "onyx-dot-app/EnterpriseRAG-Bench",
  "config": "documents",
  "split": "test",
  "comparable": true,
  "n_docs_expected": 511962,
  "n_docs_indexed": 511962,
  "n_chunks": 0,
  "pack": "sentence_quarter_v1",
  "C": 8192,
  "T": 2048,
  "O": 64,
  "embedder": {"name": "BAAI/bge-m3", "dim": 1024},
  "sparse": null,
  "title_header": true,
  "audit_report": "audit_report.json"
}
```

---

## 12. Handoff to answers / eval

```text
HF documents
  → normalize
  → source-specific atomic units (§7)
  → sentence-packed quarters T=C/4 (§5–§6)
  → audit (§9) → PASS
  → index
  → retrieve chunks → collapse to ranked doc_ids
  → answer_evaluation/<run>/answers.jsonl
  → Docs/evaluation_spec.md
```

---

## 13. Out of scope

- Regenerating the synthetic corpus
- Putting questions into the document index
- Using gold `expected_doc_ids` during ingest
- Claiming 512-token embedders as the default without marking non-default

---

## 14. Acceptance checklist

- [ ] Full 511,962 docs (or `comparable=false` subset)
- [ ] `pack=sentence_quarter_v1`, `C=8192`, `T=2048`, `O=64`
- [ ] Per-type separators from §7 applied
- [ ] Fireflies summary/transcript never mixed
- [ ] Header `[{source_type}] {title}` on every chunk
- [ ] `audit_report.json` all blocking checks `pass`
- [ ] Retrieval collapses to ranked unique `doc_id`s
- [ ] Experiment README points at `indexes/<build_id>/manifest.json`
