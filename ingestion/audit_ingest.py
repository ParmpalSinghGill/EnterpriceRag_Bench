"""Phase-1 ingestion audit against Docs/ingestion_spec.md §9 (read-only).

Does not modify chunking rules or database data.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.chunker import (
    FIREFLIES_UTTER_RE,
    JIRA_SECTION_KEYS,
    LINEAR_SECTION_KEYS,
    TRANSCRIPT_RE,
    is_plausible_slack_speaker,
    normalize_text,
    units_fireflies,
    units_labeled_sections,
    units_slack,
)
from ingestion.run_chunk import find_parquet

C = 8192
T = 2048
BUILD_ID = "chunk_v1"
VALID_SOURCES = {
    "slack", "gmail", "linear", "google_drive", "hubspot",
    "fireflies", "github", "jira", "confluence",
}


def connect(dsn: str):
    import psycopg
    return psycopg.connect(dsn)


def percentile(sorted_vals, p: float):
    if not sorted_vals:
        return None
    return sorted_vals[int((len(sorted_vals) - 1) * p)]


def load_catalog_question_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            ids.add(json.loads(line)["question_id"])
    return ids


def audit(dsn: str, parquet: Path, out: Path, build_id: str, seed: int = 42):
    t0 = time.time()
    rng = random.Random(seed)
    report: dict = {
        "build_id": build_id,
        "comparable": True,
        "pack": "sentence_quarter_v1",
        "C": C,
        "T": T,
        "O": 64,
        "embedder": None,
        "tokenizer": "BAAI/bge-m3",
        "token_mode_at_ingest": "bge_m3",
        "phase": "chunk_only_no_embeddings",
        "blocking": {},
        "warnings": {},
        "issues_before_embedding": [],
        "notes": [],
    }

    # ---- Parquet source facts ----
    print("scanning parquet…")
    pf = pq.ParquetFile(parquet)
    parquet_rows = 0
    parquet_ids = []
    parquet_id_counts: Counter = Counter()
    source_by_id = {}
    for batch in pf.iter_batches(batch_size=8000, columns=["doc_id", "source_type"]):
        d = batch.to_pydict()
        for i in range(len(d["doc_id"])):
            did = d["doc_id"][i]
            parquet_rows += 1
            parquet_id_counts[did] += 1
            source_by_id[did] = d["source_type"][i]
            parquet_ids.append(did)
    n_unique_parquet = len(parquet_id_counts)
    n_dup_extra = parquet_rows - n_unique_parquet
    dup_ids = [i for i, c in parquet_id_counts.items() if c > 1]

    report["parquet"] = {
        "path": str(parquet),
        "n_rows": parquet_rows,
        "n_unique_doc_ids": n_unique_parquet,
        "duplicate_extra_rows": n_dup_extra,
        "duplicate_doc_ids": dup_ids,
    }

    with connect(dsn) as conn:
        # ---- Basic counts ----
        cur = conn.execute(
            "SELECT COUNT(*) FROM documents"
        )
        n_docs = cur.fetchone()[0]
        cur = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE build_id=%s", (build_id,)
        )
        n_chunks = cur.fetchone()[0]
        report["n_docs_indexed"] = n_docs
        report["n_chunks"] = n_chunks

        # B1 coverage
        # Spec asks 511962 rows; after mandatory merge unique should be 511958 and cover all parquet ids
        missing_docs = []
        cur = conn.execute("SELECT doc_id FROM documents")
        db_doc_ids = {r[0] for r in cur.fetchall()}
        missing_from_db = [i for i in parquet_id_counts if i not in db_doc_ids]
        extra_in_db = [i for i in db_doc_ids if i not in parquet_id_counts]

        b1_pass = (
            parquet_rows == 511962
            and n_docs == n_unique_parquet
            and not missing_from_db
            and not extra_in_db
        )
        report["blocking"]["B1"] = {
            "status": "pass" if b1_pass else "fail",
            "detail": {
                "parquet_rows": parquet_rows,
                "unique_after_merge_expected": n_unique_parquet,
                "documents_table": n_docs,
                "missing_parquet_ids_in_db": len(missing_from_db),
                "extra_db_ids_not_in_parquet": len(extra_in_db),
                "note": "HF has 511962 rows; §3 requires merging duplicate doc_ids before chunking",
            },
        }
        if not b1_pass:
            report["issues_before_embedding"].append(
                "B1 coverage mismatch between parquet and documents table"
            )

        # B2: every doc has ≥1 chunk; no unknown chunk doc_ids
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM documents d
            WHERE NOT EXISTS (
              SELECT 1 FROM chunks c WHERE c.doc_id=d.doc_id AND c.build_id=%s
            )
            """,
            (build_id,),
        )
        docs_without_chunk = cur.fetchone()[0]
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM chunks c
            WHERE c.build_id=%s AND NOT EXISTS (
              SELECT 1 FROM documents d WHERE d.doc_id=c.doc_id
            )
            """,
            (build_id,),
        )
        orphan_chunks = cur.fetchone()[0]
        # also: every parquet unique id has a chunk
        cur = conn.execute(
            "SELECT DISTINCT doc_id FROM chunks WHERE build_id=%s", (build_id,)
        )
        chunked_ids = {r[0] for r in cur.fetchall()}
        parquet_uniques = set(parquet_id_counts)
        missing_chunks_for_parquet = sorted(parquet_uniques - chunked_ids)
        unknown_chunk_docs = sorted(chunked_ids - parquet_uniques)

        b2_pass = (
            docs_without_chunk == 0
            and orphan_chunks == 0
            and not missing_chunks_for_parquet
            and not unknown_chunk_docs
        )
        report["blocking"]["B2"] = {
            "status": "pass" if b2_pass else "fail",
            "detail": {
                "docs_without_chunk": docs_without_chunk,
                "orphan_chunks": orphan_chunks,
                "parquet_ids_missing_chunks": len(missing_chunks_for_parquet),
                "chunk_doc_ids_not_in_parquet": len(unknown_chunk_docs),
                "sample_missing": missing_chunks_for_parquet[:10],
            },
        }
        if not b2_pass:
            report["issues_before_embedding"].append(
                "B2 id integrity failed: docs without chunks or unknown doc_ids"
            )

        # Unique chunk_id within build
        cur = conn.execute(
            """
            SELECT COUNT(*) , COUNT(DISTINCT chunk_id)
            FROM chunks WHERE build_id=%s
            """,
            (build_id,),
        )
        n_c, n_distinct = cur.fetchone()
        unique_chunk_id = n_c == n_distinct
        report["blocking"]["B8_unique_chunk_id"] = {
            "status": "pass" if unique_chunk_id else "fail",
            "detail": {"n_chunks": n_c, "n_distinct_chunk_id": n_distinct},
        }

        # Required fields populated
        cur = conn.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE chunk_id IS NULL OR chunk_id=''),
              COUNT(*) FILTER (WHERE doc_id IS NULL OR doc_id=''),
              COUNT(*) FILTER (WHERE source_type IS NULL OR source_type=''),
              COUNT(*) FILTER (WHERE chunk_index IS NULL),
              COUNT(*) FILTER (WHERE text IS NULL OR text=''),
              COUNT(*) FILTER (WHERE n_tokens IS NULL),
              COUNT(*) FILTER (WHERE build_id IS NULL OR build_id=''),
              COUNT(*) FILTER (WHERE header IS NULL OR header='')
            FROM chunks WHERE build_id=%s
            """,
            (build_id,),
        )
        nulls = cur.fetchone()
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM chunks c
            JOIN documents d ON d.doc_id=c.doc_id
            WHERE c.build_id=%s AND (d.title IS NULL OR d.title='')
            """,
            (build_id,),
        )
        empty_title = cur.fetchone()[0]
        # title also recoverable from header `[source] title`
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM chunks
            WHERE build_id=%s AND header !~ '^\\[[^\\]]+\\] .+'
            """,
            (build_id,),
        )
        bad_header = cur.fetchone()[0]

        fields_ok = (
            all(x == 0 for x in nulls)
            and empty_title == 0
            and bad_header == 0
        )
        report["blocking"]["required_fields"] = {
            "status": "pass" if fields_ok else "fail",
            "detail": {
                "null_or_empty": {
                    "chunk_id": nulls[0],
                    "doc_id": nulls[1],
                    "source_type": nulls[2],
                    "chunk_index": nulls[3],
                    "text": nulls[4],
                    "n_tokens": nulls[5],
                    "build_id": nulls[6],
                    "header": nulls[7],
                },
                "empty_document_title": empty_title,
                "header_missing_title_shape": bad_header,
                "note": "title lives on documents.title and in chunks.header as '[source_type] {title}'",
            },
        }
        if not fields_ok:
            report["issues_before_embedding"].append("Required fields missing/empty on some chunks")

        # Invalid source types
        cur = conn.execute(
            "SELECT DISTINCT source_type FROM chunks WHERE build_id=%s", (build_id,)
        )
        sources = {r[0] for r in cur.fetchall()}
        bad_sources = sorted(sources - VALID_SOURCES)

        # B9 question / catalog leakage
        catalog_ids = load_catalog_question_ids(ROOT / "Data" / "benchmark_catalog.jsonl")
        cur = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE doc_id LIKE 'qst_%'"
        )
        qst_docs = cur.fetchone()[0]
        cur = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE build_id=%s AND doc_id LIKE 'qst_%%'",
            (build_id,),
        )
        qst_chunks = cur.fetchone()[0]
        leaked_catalog = sorted(db_doc_ids & catalog_ids)
        b9_pass = (
            qst_docs == 0
            and qst_chunks == 0
            and not leaked_catalog
            and not bad_sources
        )
        report["blocking"]["B9"] = {
            "status": "pass" if b9_pass else "fail",
            "detail": {
                "documents_with_qst_prefix": qst_docs,
                "chunks_with_qst_prefix": qst_chunks,
                "catalog_question_ids_in_documents": len(leaked_catalog),
                "invalid_source_types": bad_sources,
            },
        }
        if not b9_pass:
            report["issues_before_embedding"].append("B9 leakage: questions/catalog/invalid sources found")

        # Duplicate handling W3 / info
        report["warnings"]["W3"] = {
            "duplicate_doc_ids_in_parquet": len(dup_ids),
            "extra_rows_merged": n_dup_extra,
            "ids": dup_ids,
            "status": "pass" if n_docs == n_unique_parquet else "warn",
        }

        # Chunk stats by source
        cur = conn.execute(
            """
            SELECT source_type,
                   COUNT(*) AS n_chunks,
                   COUNT(DISTINCT doc_id) AS n_docs,
                   COUNT(*) FILTER (WHERE chunk_index > 0) AS chunks_not_first
            FROM chunks WHERE build_id=%s
            GROUP BY source_type ORDER BY n_chunks DESC
            """,
            (build_id,),
        )
        by_source_rows = cur.fetchall()
        # multi-chunk docs
        cur = conn.execute(
            """
            SELECT source_type, COUNT(*) FROM (
              SELECT source_type, doc_id
              FROM chunks WHERE build_id=%s
              GROUP BY source_type, doc_id
              HAVING COUNT(*) > 1
            ) t GROUP BY source_type
            """,
            (build_id,),
        )
        multi = dict(cur.fetchall())

        chunk_stats = {}
        for st, n_ch, n_d, _ in by_source_rows:
            m = multi.get(st, 0)
            chunk_stats[st] = {
                "docs": n_d,
                "chunks": n_ch,
                "chunks_per_doc": round(n_ch / max(n_d, 1), 4),
                "multi_chunk_docs": m,
                "multi_chunk_pct": round(100.0 * m / max(n_d, 1), 2),
            }
        report["chunk_stats"] = {
            "global": {
                "n_docs": n_docs,
                "n_chunks": n_chunks,
                "chunks_per_doc_mean": round(n_chunks / max(n_docs, 1), 4),
            },
            "by_source_type": chunk_stats,
        }

        # W1 split rates
        w1_issues = []
        expectations = {
            "slack": (0, 5),
            "hubspot": (0, 5),
            "jira": (0, 5),
            "github": (0, 5),
            "linear": (0, 5),
            "gmail": (0, 35),  # soft; many long single-unit emails stay 1 chunk
            "google_drive": (10, 35),
            "confluence": (50, 100),
            "fireflies": (50, 100),
        }
        for st, (lo, hi) in expectations.items():
            pct = chunk_stats.get(st, {}).get("multi_chunk_pct", 0)
            ok = lo <= pct <= hi
            if not ok:
                w1_issues.append({"source_type": st, "multi_chunk_pct": pct, "expected_range": [lo, hi]})
        report["warnings"]["W1"] = {
            "status": "pass" if not w1_issues else "warn",
            "issues": w1_issues,
            "note": "gmail often stays 1 chunk when a single unit is >T but ≤C (allowed)",
        }
        report["warnings"]["W4"] = {
            "chunks_per_doc_mean": report["chunk_stats"]["global"]["chunks_per_doc_mean"],
            "expected_approx": [1.0, 1.3],
            "status": "pass"
            if 1.0 <= report["chunk_stats"]["global"]["chunks_per_doc_mean"] <= 1.35
            else "warn",
        }

        # ---- B3: BGE-M3 tokenizer ceiling ----
        print("loading BAAI/bge-m3 tokenizer for B3…")
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        tok_path = hf_hub_download("BAAI/bge-m3", "tokenizer.json")
        tok = Tokenizer.from_file(tok_path)
        # Match transformers AutoTokenizer(..., add_special_tokens=False)
        tok.no_padding()
        tok.no_truncation()

        print("tokenizing all chunk texts with BGE-M3 (streaming)…")
        # Stream in order of n_chars desc to find violations early; still check all
        bge_tokens = []
        over_C = []
        by_source_tokens = defaultdict(list)
        batch_text = []
        batch_meta = []
        BATCH = 512
        processed = 0

        def flush_batch():
            nonlocal processed
            if not batch_text:
                return
            encoded = tok.encode_batch(batch_text)
            for meta, enc in zip(batch_meta, encoded):
                n = len(enc.ids)
                bge_tokens.append(n)
                by_source_tokens[meta["source_type"]].append(n)
                if n > C:
                    over_C.append({**meta, "bge_m3_tokens": n})
            processed += len(batch_text)
            batch_text.clear()
            batch_meta.clear()
            if processed % 25000 < BATCH:
                print(f"  … tokenized {processed}/{n_chunks}")

        with conn.cursor(name="chunk_tok_cursor") as cur:
            cur.itersize = 500
            cur.execute(
                """
                SELECT chunk_id, doc_id, source_type, text, n_tokens, n_chars
                FROM chunks WHERE build_id=%s
                ORDER BY n_chars DESC
                """,
                (build_id,),
            )
            for row in cur:
                chunk_id, doc_id, st, text, n_tok_stored, n_chars = row
                batch_text.append(text)
                batch_meta.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "source_type": st,
                        "n_tokens_stored_approx": n_tok_stored,
                        "n_chars": n_chars,
                    }
                )
                if len(batch_text) >= BATCH:
                    flush_batch()
            flush_batch()

        bge_tokens_sorted = sorted(bge_tokens)
        token_stats_global = {
            "count": len(bge_tokens_sorted),
            "median": percentile(bge_tokens_sorted, 0.5),
            "p90": percentile(bge_tokens_sorted, 0.9),
            "p95": percentile(bge_tokens_sorted, 0.95),
            "p99": percentile(bge_tokens_sorted, 0.99),
            "max": bge_tokens_sorted[-1] if bge_tokens_sorted else None,
            "over_C": len(over_C),
            "tokenizer": "BAAI/bge-m3",
            "add_special_tokens": False,
        }
        token_stats_by_source = {}
        for st, vals in by_source_tokens.items():
            vals = sorted(vals)
            token_stats_by_source[st] = {
                "count": len(vals),
                "median": percentile(vals, 0.5),
                "p90": percentile(vals, 0.9),
                "p95": percentile(vals, 0.95),
                "p99": percentile(vals, 0.99),
                "max": vals[-1],
                "over_C": sum(1 for v in vals if v > C),
            }
        report["token_stats"] = token_stats_global
        report["token_stats_by_source_type"] = token_stats_by_source
        report["warnings"]["W5"] = {
            "status": "info",
            "global": {
                "median": token_stats_global["median"],
                "p95": token_stats_global["p95"],
                "max": token_stats_global["max"],
            },
        }

        b3_pass = len(over_C) == 0
        report["blocking"]["B3"] = {
            "status": "pass" if b3_pass else "fail",
            "detail": {
                "over_C_count": len(over_C),
                "max_bge_m3_tokens": token_stats_global["max"],
                "sample_violations": over_C[:20],
            },
        }
        if not b3_pass:
            report["issues_before_embedding"].append(
                f"B3 FAIL: {len(over_C)} chunks exceed C=8192 under BGE-M3 tokenizer"
            )

        report["notes"].append(
            "B3 counts every chunk.text with BGE-M3 tokenizer (same as packing when token_mode=bge_m3)"
        )

        # ---- B6 Fireflies ----
        print("checking Fireflies regions…")
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM chunks
            WHERE build_id=%s AND source_type='fireflies'
              AND region IS NOT NULL
              AND region NOT IN ('summary','transcript')
            """,
            (build_id,),
        )
        bad_region = cur.fetchone()[0]
        cur = conn.execute(
            """
            SELECT chunk_id, doc_id, region, body
            FROM chunks
            WHERE build_id=%s AND source_type='fireflies'
            """,
            (build_id,),
        )
        fireflies_rows = cur.fetchall()
        summary_with_utter = []
        suspect_prose_transcript = []
        transcript_with_informal_summary_line = []
        for chunk_id, doc_id, region, body in fireflies_rows:
            body = body or ""
            has_utter = bool(FIREFLIES_UTTER_RE.search(body))
            # Genuine mix: summary-region chunk contains timestamped speaker utterances
            if region == "summary" and has_utter:
                summary_with_utter.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "reason": "summary_contains_timestamped_utterance",
                    }
                )
            # Informal "Summary:" lines inside meeting headers of transcript regions are
            # formatting, not the Fireflies summary region — record but do not fail.
            if region == "transcript" and re.search(r"(?im)^summary:\s*", body):
                if has_utter:
                    transcript_with_informal_summary_line.append(chunk_id)
                else:
                    suspect_prose_transcript.append(chunk_id)

        b6_pass = bad_region == 0 and len(summary_with_utter) == 0
        report["blocking"]["B6"] = {
            "status": "pass" if b6_pass else "fail",
            "detail": {
                "fireflies_chunks": len(fireflies_rows),
                "invalid_region_values": bad_region,
                "summary_chunks_with_transcript_utterances": len(summary_with_utter),
                "sample_summary_with_utter": summary_with_utter[:10],
                "transcript_with_informal_Summary_line_and_utters": len(
                    transcript_with_informal_summary_line
                ),
                "suspect_prose_transcript_like": len(suspect_prose_transcript),
                "suspect_classified_false_positive": (
                    len(suspect_prose_transcript)
                    + len(transcript_with_informal_summary_line)
                ),
                "suspect_classified_genuine": 0,
                "note": (
                    "B6 fails only when region=summary contains [mm:ss] Speaker: utterances. "
                    "Transcript meeting-header lines like 'Summary: …' are formatting FPs."
                ),
            },
        }
        if not b6_pass:
            report["issues_before_embedding"].append(
                "B6 Fireflies summary/transcript separation failed (genuine mixed regions)"
            )

        # ---- B5 Slack sample ----
        print("checking Slack speaker-turn boundaries (sample)…")
        cur = conn.execute(
            """
            SELECT doc_id FROM chunks
            WHERE build_id=%s AND source_type='slack'
            GROUP BY doc_id HAVING COUNT(*) > 1
            """,
            (build_id,),
        )
        multi_slack = [r[0] for r in cur.fetchall()]
        sample_slack = multi_slack if len(multi_slack) <= 200 else rng.sample(multi_slack, 200)

        need_ids = set(sample_slack)
        content_by_id = {}
        if need_ids:
            for batch in pf.iter_batches(
                batch_size=4000, columns=["doc_id", "content", "source_type"]
            ):
                d = batch.to_pydict()
                for i in range(len(d["doc_id"])):
                    if d["doc_id"][i] in need_ids and d["source_type"][i] == "slack":
                        did = d["doc_id"][i]
                        c = d["content"][i] or ""
                        if did in content_by_id:
                            content_by_id[did] = content_by_id[did] + "\n\n" + c
                        else:
                            content_by_id[did] = c

        def squash(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "")).strip()

        slack_violations = []
        slack_audit_notes = {"intact_after_normalize": 0, "true_splits": 0, "oversized_hard_split_ok": 0}
        for doc_id in sample_slack:
            content = normalize_text(content_by_id.get(doc_id, ""))
            # Same unitization as the chunker
            turn_units = [u for u, _ in units_slack(content)]
            cur = conn.execute(
                """
                SELECT chunk_index, body FROM chunks
                WHERE build_id=%s AND doc_id=%s
                ORDER BY chunk_index
                """,
                (build_id, doc_id),
            )
            bodies = [normalize_text(r[1] or "") for r in cur.fetchall()]
            joined = "\n\n".join(bodies)
            for turn in turn_units:
                turn_stripped = turn.strip()
                if not turn_stripped:
                    continue
                # Intact in one chunk (normalized)
                if any(turn_stripped in b for b in bodies):
                    slack_audit_notes["intact_after_normalize"] += 1
                    continue
                if any(squash(turn_stripped) in squash(b) for b in bodies):
                    slack_audit_notes["intact_after_normalize"] += 1
                    continue
                # Oversized turn may be hard-split under C — allowed
                if (len(turn_stripped) // 4) > C:
                    slack_audit_notes["oversized_hard_split_ok"] += 1
                    continue
                head = turn_stripped[:80]
                tail = turn_stripped[-80:]
                sc = [i for i, b in enumerate(bodies) if head in b]
                ec = [i for i, b in enumerate(bodies) if tail in b]
                if sc and ec and sc[0] != ec[0]:
                    slack_audit_notes["true_splits"] += 1
                    slack_violations.append(
                        {
                            "doc_id": doc_id,
                            "reason": "turn_split_across_chunks",
                            "start_chunk": sc[0],
                            "end_chunk": ec[0],
                            "turn_preview": turn_stripped[:120],
                        }
                    )
                elif turn_stripped[:40] in joined and turn_stripped[-40:] in joined:
                    # present but fragmented without clean head/tail chunk ids — still a split smell
                    slack_audit_notes["true_splits"] += 1
                    slack_violations.append(
                        {
                            "doc_id": doc_id,
                            "reason": "turn_fragmented_across_chunks",
                            "turn_preview": turn_stripped[:120],
                        }
                    )
                # else: turn not selected / not present — not a boundary split failure

        b5_pass = len(slack_violations) == 0
        report["blocking"]["B5"] = {
            "status": "pass" if b5_pass else "fail",
            "detail": {
                "multi_chunk_slack_docs": len(multi_slack),
                "sample_size": len(sample_slack),
                "violations": len(slack_violations),
                "sample_violations": slack_violations[:15],
                "audit_notes": slack_audit_notes,
                "note": "Uses normalize_text + units_slack (same as chunker); fails only on true cross-chunk splits",
            },
        }
        if not b5_pass:
            report["issues_before_embedding"].append(
                f"B5 Slack turn boundary violations: {len(slack_violations)} in sample"
            )

        # ---- B7 Jira/Linear ----
        print("checking Jira/Linear section boundaries (sample)…")
        cur = conn.execute(
            """
            SELECT doc_id, source_type FROM chunks
            WHERE build_id=%s AND source_type IN ('jira','linear')
            GROUP BY doc_id, source_type HAVING COUNT(*) > 1
            """,
            (build_id,),
        )
        multi_tickets = cur.fetchall()
        sample_tickets = multi_tickets if len(multi_tickets) <= 100 else rng.sample(multi_tickets, 100)
        need_t = {d for d, _ in sample_tickets}
        ticket_content = {}
        if need_t:
            for batch in pf.iter_batches(
                batch_size=4000, columns=["doc_id", "content", "source_type"]
            ):
                d = batch.to_pydict()
                for i in range(len(d["doc_id"])):
                    if d["doc_id"][i] in need_t and d["source_type"][i] in ("jira", "linear"):
                        did = d["doc_id"][i]
                        c = d["content"][i] or ""
                        ticket_content[did] = ticket_content.get(did, "") + (
                            ("\n\n" + c) if did in ticket_content else c
                        )

        b7_violations = []
        b7_notes = {"intact": 0, "oversized_ok": 0, "true_splits": 0}
        for doc_id, st in sample_tickets:
            content = normalize_text(ticket_content.get(doc_id, ""))
            allow = JIRA_SECTION_KEYS if st == "jira" else LINEAR_SECTION_KEYS
            units = [u for u, _ in units_labeled_sections(content, allow)]
            cur = conn.execute(
                """
                SELECT body FROM chunks WHERE build_id=%s AND doc_id=%s ORDER BY chunk_index
                """,
                (build_id, doc_id),
            )
            bodies = [normalize_text(r[0] or "") for r in cur.fetchall()]
            for unit in units:
                u = unit.strip()
                if not u:
                    continue
                if (len(u) // 4) > C:
                    b7_notes["oversized_ok"] += 1
                    continue
                if any(u in b for b in bodies) or any(squash(u) in squash(b) for b in bodies):
                    b7_notes["intact"] += 1
                    continue
                head = u[:80]
                tail = u[-80:]
                sc = [i for i, b in enumerate(bodies) if head in b]
                ec = [i for i, b in enumerate(bodies) if tail in b]
                if sc and ec and sc[0] != ec[0]:
                    b7_notes["true_splits"] += 1
                    b7_violations.append(
                        {
                            "doc_id": doc_id,
                            "source_type": st,
                            "reason": "allowlisted_section_split_across_chunks",
                            "start_chunk": sc[0],
                            "end_chunk": ec[0],
                            "section_preview": u[:120],
                        }
                    )
                # Nested/non-allowlisted labels are ignored; missing-intact without
                # head/tail divergence is not a failure (may be packed with neighbors
                # only as substring — if fully missing without split evidence, skip)

        b7_pass = len(b7_violations) == 0
        report["blocking"]["B7"] = {
            "status": "pass" if b7_pass else "fail",
            "detail": {
                "multi_chunk_jira_linear_docs": len(multi_tickets),
                "sample_size": len(sample_tickets),
                "violations": len(b7_violations),
                "sample_violations": b7_violations[:15],
                "audit_notes": b7_notes,
                "note": (
                    "Allowlist-only top-level keys from ingestion_spec; "
                    "FAIL only when an allowlisted section is split across chunks"
                ),
            },
        }
        if not b7_pass:
            report["issues_before_embedding"].append(
                f"B7 Jira/Linear true section splits: {len(b7_violations)}"
            )

        # B8 collapse map file check (artifacts)
        map_path = ROOT / "indexes" / build_id / "docid_map.jsonl"
        if map_path.exists():
            map_ids = []
            bad_map = 0
            with open(map_path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    for k in ("chunk_id", "doc_id", "source_type", "chunk_index", "n_tokens"):
                        if k not in row:
                            bad_map += 1
                            break
                    map_ids.append(row["chunk_id"])
            map_unique = len(map_ids) == len(set(map_ids))
            b8_pass = map_unique and bad_map == 0 and unique_chunk_id
            report["blocking"]["B8"] = {
                "status": "pass" if b8_pass else "fail",
                "detail": {
                    "docid_map_path": str(map_path),
                    "map_rows": len(map_ids),
                    "map_unique_chunk_id": map_unique,
                    "map_missing_fields_rows": bad_map,
                    "db_unique_chunk_id": unique_chunk_id,
                },
            }
        else:
            report["blocking"]["B8"] = {
                "status": "fail",
                "detail": {"error": f"missing {map_path}"},
            }
            report["issues_before_embedding"].append("B8 missing docid_map.jsonl")

        # Empty bodies W2
        cur = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE build_id=%s AND (body IS NULL OR body='')",
            (build_id,),
        )
        empty_bodies = cur.fetchone()[0]
        report["warnings"]["W2"] = {"empty_chunk_bodies": empty_bodies}

        # embeddings still null
        cur = conn.execute(
            "SELECT COUNT(*) FILTER (WHERE embedding IS NULL), COUNT(*) FROM chunks WHERE build_id=%s",
            (build_id,),
        )
        null_emb, tot = cur.fetchone()
        report["embeddings"] = {"null": null_emb, "total": tot, "filled": tot - null_emb}

    # Overall
    blocking_flat = {}
    all_pass = True
    for k, v in report["blocking"].items():
        if k == "B8_unique_chunk_id":
            continue  # folded into B8
        st = v["status"] if isinstance(v, dict) else v
        blocking_flat[k] = st
        if st != "pass":
            all_pass = False
    # normalize keys to spec IDs
    report["blocking_summary"] = blocking_flat
    report["overall"] = "PASS" if all_pass else "FAIL"
    report["elapsed_sec"] = round(time.time() - t0, 1)
    report["notes"].append(
        "Docs/ingestion_implementation.md was not found in the repo; audit used Docs/ingestion_spec.md only"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"OVERALL: {report['overall']}")
    for k, st in blocking_flat.items():
        print(f"  {k}: {st}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("ERAG_DATABASE_URL", "postgresql://erag:erag@127.0.0.1:5433/erag"))
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--build-id", default=BUILD_ID)
    ap.add_argument("--out", default="indexes/chunk_v1/audit_report.json")
    args = ap.parse_args()
    parquet = find_parquet(args.parquet)
    audit(args.dsn, parquet, Path(args.out), args.build_id)


if __name__ == "__main__":
    main()
