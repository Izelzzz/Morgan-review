CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    repo TEXT NOT NULL,
    pr INTEGER NOT NULL CHECK (pr > 0),
    base_sha TEXT NOT NULL,
    head_sha TEXT NOT NULL,
    round INTEGER NOT NULL DEFAULT 1 CHECK (round >= 1),
    mode TEXT NOT NULL CHECK (mode IN ('quick', 'standard', 'deep')),
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'done', 'failed')),
    error TEXT,
    started_at TEXT,
    finished_at TEXT,
    limit_hit TEXT CHECK (limit_hit IN ('time', 'tokens', 'tool_calls')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX reviews_by_status ON reviews (status, id);
CREATE INDEX reviews_by_pr ON reviews (repo, pr);

CREATE TABLE findings (
    id INTEGER PRIMARY KEY,
    review_id INTEGER NOT NULL REFERENCES reviews (id) ON DELETE CASCADE,
    fingerprint TEXT NOT NULL,
    primary_dimension TEXT NOT NULL,
    dimensions TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(dimensions)),
    severity TEXT NOT NULL CHECK (severity IN ('blocking', 'important', 'robustness', 'nit')),
    preexisting INTEGER NOT NULL DEFAULT 0 CHECK (preexisting IN (0, 1)),
    file TEXT NOT NULL,
    line_start INTEGER NOT NULL CHECK (line_start >= 1),
    line_end INTEGER NOT NULL CHECK (line_end >= line_start),
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    evidence TEXT NOT NULL,
    verdict TEXT,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'fixed', 'dismissed'))
);
CREATE INDEX findings_by_fingerprint ON findings (fingerprint);

CREATE TABLE comment_links (
    finding_id INTEGER NOT NULL REFERENCES findings (id) ON DELETE CASCADE,
    github_comment_id INTEGER NOT NULL,
    thread_id TEXT,
    PRIMARY KEY (finding_id, github_comment_id)
);

CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    repo TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('false_positive', 'false_negative', 'convention')),
    paths TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(paths)),
    dimension TEXT,
    content TEXT NOT NULL,
    source_comment_url TEXT NOT NULL,
    author TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    revoked_at TEXT
);
CREATE INDEX memories_by_repo ON memories (repo) WHERE revoked_at IS NULL;

CREATE TABLE feedback_cases (
    id INTEGER PRIMARY KEY,
    repo TEXT NOT NULL,
    pr INTEGER NOT NULL CHECK (pr > 0),
    finding_id INTEGER REFERENCES findings (id) ON DELETE SET NULL,
    label TEXT NOT NULL CHECK (label IN ('false_positive', 'false_negative')),
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE usage (
    id INTEGER PRIMARY KEY,
    review_id INTEGER NOT NULL REFERENCES reviews (id) ON DELETE CASCADE,
    agent TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
    duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0)
);

CREATE TABLE steps (
    id INTEGER PRIMARY KEY,
    review_id INTEGER NOT NULL REFERENCES reviews (id) ON DELETE CASCADE,
    node TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'ok', 'error')),
    error TEXT,
    trace_id TEXT
);

CREATE TABLE context_cache (
    repo TEXT NOT NULL,
    layer TEXT NOT NULL CHECK (layer IN ('repo', 'pr')),
    cache_key TEXT NOT NULL,
    content TEXT NOT NULL,
    tokens INTEGER NOT NULL CHECK (tokens >= 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (repo, layer, cache_key)
);
