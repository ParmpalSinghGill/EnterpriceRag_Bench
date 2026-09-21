-- EnterpriseRAGBench chunk store (phase 1: text chunks only)
-- Embedding column reserved for later; leave NULL for now.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS ingest_builds (
    build_id        TEXT PRIMARY KEY,
    pack            TEXT NOT NULL DEFAULT 'sentence_quarter_v1',
    c_tokens        INT  NOT NULL DEFAULT 8192,
    t_tokens        INT  NOT NULL DEFAULT 2048,
    o_tokens        INT  NOT NULL DEFAULT 64,
    token_mode      TEXT NOT NULL,          -- 'approx' | 'bge_m3'
    comparable      BOOLEAN NOT NULL DEFAULT TRUE,
    n_docs          INT,
    n_chunks        INT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    source_type     TEXT NOT NULL,
    title           TEXT NOT NULL DEFAULT '',
    content         TEXT NOT NULL DEFAULT '',
    content_chars   INT  NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS documents_source_type_idx ON documents (source_type);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT NOT NULL,
    build_id        TEXT NOT NULL REFERENCES ingest_builds (build_id) ON DELETE CASCADE,
    doc_id          TEXT NOT NULL REFERENCES documents (doc_id),
    source_type     TEXT NOT NULL,
    chunk_index     INT  NOT NULL,
    region          TEXT,                  -- 'summary' | 'transcript' | NULL
    n_tokens        INT  NOT NULL,
    n_chars         INT  NOT NULL,
    header          TEXT NOT NULL,
    body            TEXT NOT NULL,
    text            TEXT NOT NULL,          -- header + blank line + body (indexed string)
    -- Reserved for phase 2 (bge-m3 dim=1024). Stay NULL until embedding pass.
    embedding       vector(1024),
    PRIMARY KEY (build_id, chunk_id),
    UNIQUE (build_id, doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_build_id_idx ON chunks (build_id);
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_source_type_idx ON chunks (source_type);
CREATE INDEX IF NOT EXISTS chunks_build_source_idx ON chunks (build_id, source_type);

-- Optional: full-text search later (not required for phase 1 size check)
-- CREATE INDEX chunks_text_fts_idx ON chunks USING GIN (to_tsvector('english', text));
