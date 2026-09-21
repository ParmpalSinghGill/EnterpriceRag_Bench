"""sentence_quarter_v1 chunker — implements Docs/ingestion_spec.md §5–§7.

Embeddings are out of scope here. This module only splits documents into chunks.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from ingestion.tokenize import TokenCounter

C_DEFAULT = 8192
T_DEFAULT = 2048
O_DEFAULT = 64
PACK = "sentence_quarter_v1"

# Spec §7.2 slack regex + optional "Name (role)" when present in corpus.
# Role chars exclude '%' etc. so phrases like "eu-west (with 7% extra):" do not match.
SLACK_SPEAKER_RE = re.compile(
    r"^(?P<speaker>[A-Za-z][\w.@-]{0,38}(?:\s*\([A-Za-z][A-Za-z0-9 /+._-]{0,38}\))?):\s",
    re.M,
)
# Spec denylist (case-insensitive). Kept as-is; parsing relies primarily on
# fence masking + speaker plausibility, not endless denylist growth.
SLACK_DENYLIST = {
    "note", "error", "stack", "warning", "summary", "status", "subject", "impact",
    "expected", "observed", "observations", "rationale", "input", "output", "logs",
    "log", "date", "from", "to", "cc", "description", "context", "example", "steps",
    "scope", "docs", "rollout", "workaround", "hypothesis", "primary",
}

# Spec §7.2 known top-level keys (allowlist-only for jira/linear/hubspot/github)
JIRA_SECTION_KEYS = {
    "description", "impact", "steps_to_reproduce", "logs", "investigation_notes",
    "workaround", "resolution", "resolution_notes", "comments", "environment",
}
LINEAR_SECTION_KEYS = {
    "description", "goal", "scope", "acceptance_criteria", "tasks", "decisions",
    "progress_notes", "next_steps", "comments", "risk",
}
HUBSPOT_SECTION_KEYS = {
    "use_case_summary", "notes", "recent_activities", "timeline", "next_step",
    "blockers",
}
GITHUB_SECTION_KEYS = {
    "description", "review_comments", "release_notes",
}

# Match label lines; callers filter by allowlist.
SECTION_LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_ /\-]{0,40}):\s", re.M)
MD_HEADING_RE = re.compile(r"^#{1,3}\s+.+$", re.M)
HR_RE = re.compile(r"^---\s*$", re.M)
TRANSCRIPT_RE = re.compile(r"(?im)^transcript:\s*")
FIREFLIES_UTTER_RE = re.compile(
    r"^\[\d{1,2}:\d{2}(?::\d{2})?\]\s*[A-Za-z][\w.@-]{0,38}:\s*",
    re.M,
)
EMAIL_HDR_RE = re.compile(
    r"^(From|To|Cc|Date|Subject|Attachments)\s*:",
    re.M | re.I,
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
FENCE_LINE_RE = re.compile(r"^```", re.M)
NUMBERED_ITEM_RE = re.compile(r"^\d+[.)]\s")
GITHUB_REVIEWER_RE = re.compile(r"^([A-Za-z][\w .@-]{0,60}):\s", re.M)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_type: str
    chunk_index: int
    header: str
    body: str
    text: str
    n_tokens: int
    n_chars: int
    region: Optional[str] = None


@dataclass
class DocRecord:
    doc_id: str
    source_type: str
    title: str
    content: str


@dataclass
class ChunkerConfig:
    C: int = C_DEFAULT
    T: int = T_DEFAULT
    O: int = O_DEFAULT
    pack: str = PACK


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(lines)


def make_header(source_type: str, title: str, doc_id: str) -> str:
    t = (title or "").strip() or doc_id
    return f"[{source_type}] {t}"


def _split_by_matches(text: str, matches: Sequence[re.Match]) -> List[str]:
    if not matches:
        return [text] if text.strip() else []
    units: List[str] = []
    if matches[0].start() > 0:
        pre = text[: matches[0].start()]
        if pre.strip():
            units.append(pre)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        unit = text[m.start() : end]
        if unit.strip():
            units.append(unit)
    return units


def _fence_spans(text: str) -> List[Tuple[int, int]]:
    """Byte ranges covered by fenced code blocks (``` … ```)."""
    spans: List[Tuple[int, int]] = []
    in_fence = False
    start = 0
    for m in FENCE_LINE_RE.finditer(text):
        if not in_fence:
            in_fence = True
            start = m.start()
        else:
            spans.append((start, m.end()))
            in_fence = False
    if in_fence:
        spans.append((start, len(text)))
    return spans


def _in_spans(pos: int, spans: Sequence[Tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


def slack_speaker_base(speaker: str) -> str:
    """Return the name token without an optional (role) suffix."""
    if "(" in speaker:
        return speaker.split("(", 1)[0].strip()
    return speaker.strip()


def is_plausible_slack_speaker(speaker: str) -> bool:
    """Tighten §7.2 matches: denylist + reject metric-like keys; keep Name (role)/bots."""
    base = slack_speaker_base(speaker)
    if not base:
        return False
    if base.lower() in SLACK_DENYLIST:
        return False
    # Metric / config keys: multiple dots, or underscore outside bot names
    if base.count(".") > 1:
        return False
    low = base.lower()
    if "_" in base and "bot" not in low:
        return False
    if "." in base:
        _left, right = base.split(".", 1)
        # Allow short dotted handles (maria.g); reject long dotted paths / filenames
        if len(base) > 16 or len(right) > 5 or not right.replace("-", "").isalnum():
            return False
    return True


def split_sentences(text: str) -> List[str]:
    """Spec §7.3 — avoid breaking fenced code; keep short numbered items with prior."""
    if not text.strip():
        return []

    fences = list(CODE_FENCE_RE.finditer(text))
    if not fences:
        parts = [p.strip() for p in SENTENCE_RE.split(text) if p.strip()]
        return _attach_numbered(parts)

    pieces: List[str] = []
    pos = 0
    for f in fences:
        if f.start() > pos:
            pieces.extend(
                p.strip() for p in SENTENCE_RE.split(text[pos : f.start()]) if p.strip()
            )
        pieces.append(text[f.start() : f.end()])
        pos = f.end()
    if pos < len(text):
        pieces.extend(p.strip() for p in SENTENCE_RE.split(text[pos:]) if p.strip())
    return _attach_numbered(pieces)


def _attach_numbered(parts: List[str]) -> List[str]:
    out: List[str] = []
    for p in parts:
        if out and NUMBERED_ITEM_RE.match(p) and len(p) // 4 < 40:
            out[-1] = out[-1] + "\n" + p
        else:
            out.append(p)
    return out


def hard_split_unit(text: str, max_tok: int, counter: TokenCounter) -> List[str]:
    """Last resort when a unit exceeds budget: sentence then word boundaries."""
    if max_tok < 1:
        max_tok = 1
    sents = split_sentences(text)
    if not sents:
        sents = [text]
    out: List[str] = []
    for s in sents:
        if counter.count(s) <= max_tok:
            out.append(s)
            continue
        words = s.split()
        buf: List[str] = []
        for w in words:
            trial = (" ".join(buf + [w])).strip()
            if buf and counter.count(trial) > max_tok:
                out.append(" ".join(buf))
                buf = [w]
            else:
                buf.append(w)
        if buf:
            out.append(" ".join(buf))
    packed: List[str] = []
    cur: List[str] = []
    cur_t = 0
    for piece in out:
        pt = counter.count(piece)
        if cur and cur_t + pt > max_tok:
            packed.append("\n".join(cur))
            cur, cur_t = [piece], pt
        else:
            cur.append(piece)
            cur_t += pt
    if cur:
        packed.append("\n".join(cur))
    return packed


# ---- source-specific atomic units ----

def units_slack(content: str) -> List[Tuple[str, Optional[str]]]:
    fences = _fence_spans(content)
    matches = []
    for m in SLACK_SPEAKER_RE.finditer(content):
        if _in_spans(m.start(), fences):
            continue
        if not is_plausible_slack_speaker(m.group("speaker")):
            continue
        matches.append(m)
    units = _split_by_matches(content, matches) if matches else ([content] if content.strip() else [])
    return [(u, None) for u in units]


def _normalize_section_key(raw: str) -> str:
    return re.sub(r"[\s/\-]+", "_", raw.strip().lower())


def units_labeled_sections(
    content: str,
    allowlist: Optional[Set[str]] = None,
) -> List[Tuple[str, Optional[str]]]:
    """Split on top-level labeled sections. If allowlist is set, only those keys split."""
    matches = []
    for m in SECTION_LABEL_RE.finditer(content):
        key = _normalize_section_key(m.group(1))
        if allowlist is None or key in allowlist:
            matches.append(m)
    units = _split_by_matches(content, matches) if matches else ([content] if content.strip() else [])
    return [(u, None) for u in units]


def units_github(content: str, counter: TokenCounter, T: int) -> List[Tuple[str, Optional[str]]]:
    base = units_labeled_sections(content, GITHUB_SECTION_KEYS)
    out: List[Tuple[str, Optional[str]]] = []
    for u, region in base:
        first = u.split("\n", 1)[0].lower()
        if first.startswith("review_comments:") and counter.count(u) > T:
            body = u.split(":", 1)[1] if ":" in u else u
            label = "review_comments:"
            matches = [
                m for m in GITHUB_REVIEWER_RE.finditer(body)
                if is_plausible_slack_speaker(m.group(1).strip())
            ]
            if not matches:
                out.append((u, region))
                continue
            subs = _split_by_matches(body, matches)
            for i, s in enumerate(subs):
                if i == 0:
                    out.append((f"{label}\n{s.lstrip()}", region))
                else:
                    out.append((s, region))
        else:
            out.append((u, region))
    return out


def units_gmail(content: str) -> List[Tuple[str, Optional[str]]]:
    lines = content.split("\n")
    hdr_end = 0
    for i, ln in enumerate(lines):
        if EMAIL_HDR_RE.match(ln):
            hdr_end = i + 1
        elif hdr_end > 0 and not ln.strip():
            hdr_end = i
            break
        elif hdr_end > 0 and not EMAIL_HDR_RE.match(ln) and i > 0:
            if i == hdr_end:
                break
            hdr_end = i
            break
    units: List[Tuple[str, Optional[str]]] = []
    if hdr_end > 0:
        header_block = "\n".join(lines[:hdr_end]).strip()
        if header_block:
            units.append((header_block, None))
        body = "\n".join(lines[hdr_end:]).strip()
    else:
        body = content
    if body:
        paras = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
        if not paras:
            paras = [body]
        for p in paras:
            units.append((p, None))
    return units if units else ([(content, None)] if content.strip() else [])


def units_drive_confluence(content: str) -> List[Tuple[str, Optional[str]]]:
    marks = []
    for m in MD_HEADING_RE.finditer(content):
        marks.append(m.start())
    for m in HR_RE.finditer(content):
        marks.append(m.start())
    marks = sorted(set(marks))
    if marks:
        units = []
        if marks[0] > 0:
            pre = content[: marks[0]]
            if pre.strip():
                units.append(pre)
        for i, start in enumerate(marks):
            end = marks[i + 1] if i + 1 < len(marks) else len(content)
            part = content[start:end]
            if part.strip():
                units.append(part)
        return [(u, None) for u in units]
    paras = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    if paras:
        return [(p, None) for p in paras]
    return [(content, None)] if content.strip() else []


def units_fireflies(content: str) -> List[Tuple[str, Optional[str]]]:
    """Summary/transcript split: earliest of `transcript:` or first timestamped utterance.

    Timestamped utterances are never labeled summary. Transcript regions without
    utterance markers are sentence-split so packing can respect T/C.
    """
    transcript_m = TRANSCRIPT_RE.search(content)
    first_utter = FIREFLIES_UTTER_RE.search(content)

    boundary_candidates: List[Tuple[int, str]] = []
    if transcript_m:
        boundary_candidates.append((transcript_m.start(), "marker"))
    if first_utter:
        boundary_candidates.append((first_utter.start(), "utterance"))

    if not boundary_candidates:
        # No marker and no utterances — single undifferentiated unit
        return [(content, None)] if content.strip() else []

    boundary_candidates.sort(key=lambda x: x[0])
    boundary, kind = boundary_candidates[0]

    summary = content[:boundary]
    if kind == "marker":
        assert transcript_m is not None
        transcript = content[transcript_m.end() :]
    else:
        transcript = content[boundary:]

    out: List[Tuple[str, Optional[str]]] = []
    if summary.strip():
        # By construction no timestamp utterance starts before boundary
        out.append((summary, "summary"))

    um = list(FIREFLIES_UTTER_RE.finditer(transcript))
    if not um:
        if transcript.strip():
            # Prose / header-style transcript: sentence units under transcript region
            for sent in split_sentences(transcript):
                if sent.strip():
                    out.append((sent.strip(), "transcript"))
            if not any(r == "transcript" for _, r in out):
                out.append((transcript, "transcript"))
    else:
        if um[0].start() > 0:
            pre = transcript[: um[0].start()]
            if pre.strip():
                for sent in split_sentences(pre):
                    if sent.strip():
                        out.append((sent.strip(), "transcript"))
        for i, mm in enumerate(um):
            end = um[i + 1].start() if i + 1 < len(um) else len(transcript)
            out.append((transcript[mm.start() : end], "transcript"))
    return out


def atomic_units(
    source_type: str,
    content: str,
    counter: TokenCounter,
    T: int,
) -> List[Tuple[str, Optional[str]]]:
    st = source_type
    if st == "slack":
        return units_slack(content)
    if st == "github":
        return units_github(content, counter, T)
    if st == "jira":
        return units_labeled_sections(content, JIRA_SECTION_KEYS)
    if st == "linear":
        return units_labeled_sections(content, LINEAR_SECTION_KEYS)
    if st == "hubspot":
        return units_labeled_sections(content, HUBSPOT_SECTION_KEYS)
    if st == "gmail":
        return units_gmail(content)
    if st in ("google_drive", "confluence"):
        return units_drive_confluence(content)
    if st == "fireflies":
        return units_fireflies(content)
    return [(content, None)] if content.strip() else []


def _overlap_prefix(prev_body: str, O: int, counter: TokenCounter) -> str:
    if not prev_body or O <= 0:
        return ""
    sents = split_sentences(prev_body)
    if not sents:
        return ""
    picked: List[str] = []
    tok = 0
    for s in reversed(sents):
        st = counter.count(s)
        if picked and tok + st > O:
            break
        picked.insert(0, s)
        tok += st
        if tok >= O:
            break
    return "\n".join(picked)


def pack_units(
    doc: DocRecord,
    units: List[Tuple[str, Optional[str]]],
    counter: TokenCounter,
    cfg: ChunkerConfig,
) -> List[Chunk]:
    """Pack atomic units. Guarantees tokens(header+body) <= C. Never mix Fireflies regions."""
    header = make_header(doc.source_type, doc.title, doc.doc_id)
    header_suffix = "\n\n"
    header_tok = counter.count(header + header_suffix)
    C, T, O = cfg.C, cfg.T, cfg.O
    # Max body tokens so header+body never exceeds C
    body_budget_C = max(1, C - header_tok)

    chunks: List[Chunk] = []
    cur_parts: List[str] = []
    cur_regions: List[Optional[str]] = []
    cur_tok = header_tok
    prev_body = ""

    def _emit_raw(body: str, region: Optional[str]) -> None:
        nonlocal prev_body
        text = f"{header}{header_suffix}{body}" if body else header
        n_tok = counter.count(text)
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::c{idx:04d}",
                doc_id=doc.doc_id,
                source_type=doc.source_type,
                chunk_index=idx,
                header=header,
                body=body,
                text=text,
                n_tokens=n_tok,
                n_chars=len(text),
                region=region,
            )
        )
        prev_body = body

    def _append_chunk(body: str, region: Optional[str]) -> None:
        text = f"{header}{header_suffix}{body}" if body else header
        n_tok = counter.count(text)
        if n_tok > C:
            for piece in hard_split_unit(body, max(1, body_budget_C // 2), counter):
                t2 = f"{header}{header_suffix}{piece}"
                if counter.count(t2) > C:
                    words = piece.split()
                    buf: List[str] = []
                    for w in words:
                        trial = " ".join(buf + [w])
                        if buf and counter.count(f"{header}{header_suffix}{trial}") > C:
                            _emit_raw(" ".join(buf), region)
                            buf = [w]
                        else:
                            buf.append(w)
                    if buf:
                        _emit_raw(" ".join(buf), region)
                else:
                    _emit_raw(piece, region)
            return
        _emit_raw(body, region)

    def make_chunk(body: str, region: Optional[str]) -> None:
        body = body.strip()
        text = f"{header}{header_suffix}{body}" if body else header
        n_tok = counter.count(text)
        if n_tok > C:
            for piece in hard_split_unit(body, min(T, body_budget_C), counter):
                _append_chunk(piece, region)
            return
        _append_chunk(body, region)

    def flush():
        nonlocal cur_parts, cur_regions, cur_tok
        if not cur_parts:
            return
        body = "\n\n".join(cur_parts).strip()
        region = None
        regions = {r for r in cur_regions if r}
        if len(regions) == 1:
            region = next(iter(regions))
        elif len(regions) > 1:
            region = cur_regions[-1]
        make_chunk(body, region)
        cur_parts, cur_regions, cur_tok = [], [], header_tok

    def emit_body(body: str, region: Optional[str]):
        flush()
        body = body.strip()
        # Do not overlap across Fireflies summary/transcript region changes
        prev_region = chunks[-1].region if chunks else None
        allow_overlap = not (
            prev_region in ("summary", "transcript")
            or region in ("summary", "transcript")
        ) or (prev_region == region and region is not None)
        # Same-region transcript/summary overlap is OK; cross-region is not
        allow_overlap = chunks and O > 0 and (
            prev_region == region
            or (prev_region is None and region is None)
        )
        if allow_overlap:
            ov = _overlap_prefix(prev_body, O, counter)
            if ov:
                candidate = ov + "\n\n" + body
                if counter.count(header + header_suffix + candidate) <= C:
                    body = candidate
        if counter.count(header + header_suffix + body) > C:
            for piece in hard_split_unit(body, min(T, body_budget_C), counter):
                make_chunk(piece, region)
            return
        make_chunk(body, region)

    for unit, region in units:
        unit = unit.strip()
        if not unit:
            continue
        # Fireflies: never pack summary with transcript
        if cur_parts and cur_regions:
            prev_r = cur_regions[-1]
            if (
                region in ("summary", "transcript")
                or prev_r in ("summary", "transcript")
            ) and region != prev_r:
                flush()

        unit_tok = counter.count(unit)
        # Hard ceiling uses header + body
        if header_tok + unit_tok > C:
            flush()
            pieces = hard_split_unit(unit, min(T, body_budget_C), counter)
            for piece in pieces:
                emit_body(piece, region)
            continue

        if unit_tok > T:
            flush()
            emit_body(unit, region)
            continue

        if cur_parts and cur_tok + unit_tok > T:
            flush()
            prev_region = chunks[-1].region if chunks else None
            same_region = prev_region == region
            if chunks and O > 0 and same_region:
                ov = _overlap_prefix(prev_body, O, counter)
                if ov:
                    ov_tok = counter.count(ov)
                    candidate_tok = header_tok + ov_tok + unit_tok
                    if candidate_tok <= C and ov_tok + unit_tok <= T:
                        cur_parts = [ov, unit]
                        cur_regions = [region, region]
                        cur_tok = candidate_tok
                        continue
            cur_parts = [unit]
            cur_regions = [region]
            cur_tok = header_tok + unit_tok
            continue

        cur_parts.append(unit)
        cur_regions.append(region)
        cur_tok += unit_tok

    flush()

    if not chunks and doc.content.strip():
        body = doc.content.strip()
        text = f"{header}{header_suffix}{body}"
        if counter.count(text) > C:
            for piece in hard_split_unit(body, min(T, body_budget_C), counter):
                make_chunk(piece, None)
        else:
            make_chunk(body, None)
    return chunks


def chunk_document(
    doc: DocRecord,
    counter: TokenCounter,
    cfg: Optional[ChunkerConfig] = None,
) -> List[Chunk]:
    cfg = cfg or ChunkerConfig()
    content = normalize_text(doc.content)
    title = normalize_text(doc.title)
    doc = DocRecord(doc.doc_id, doc.source_type, title, content)

    header = make_header(doc.source_type, doc.title, doc.doc_id)
    whole = f"{header}\n\n{content}".strip() if content else header
    whole_tok = counter.count(whole)

    # Prefer whole document when ≤ T (fireflies always two-phase)
    if doc.source_type != "fireflies" and whole_tok <= cfg.T:
        if whole_tok > cfg.C:
            return pack_units(doc, [(content, None)], counter, cfg)
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}::c0000",
                doc_id=doc.doc_id,
                source_type=doc.source_type,
                chunk_index=0,
                header=header,
                body=content,
                text=whole if content else header,
                n_tokens=whole_tok,
                n_chars=len(whole if content else header),
                region=None,
            )
        ]

    units = atomic_units(doc.source_type, content, counter, cfg.T)
    if not units and content.strip():
        units = [(content, None)]
    chunks = pack_units(doc, units, counter, cfg)
    for ch in chunks:
        if ch.n_tokens > cfg.C:
            raise RuntimeError(
                f"chunk {ch.chunk_id} has {ch.n_tokens} tokens > C={cfg.C}"
            )
    return chunks


def merge_duplicate_docs(rows: Iterable[DocRecord]) -> List[DocRecord]:
    """Merge duplicate doc_ids by concatenating content (spec §3)."""
    order: List[str] = []
    by_id: dict = {}
    for r in rows:
        if r.doc_id not in by_id:
            order.append(r.doc_id)
            by_id[r.doc_id] = DocRecord(r.doc_id, r.source_type, r.title, r.content or "")
        else:
            prev = by_id[r.doc_id]
            merged = (prev.content + "\n\n" + (r.content or "")).strip()
            title = prev.title or r.title
            by_id[r.doc_id] = DocRecord(r.doc_id, prev.source_type or r.source_type, title, merged)
    return [by_id[i] for i in order]
