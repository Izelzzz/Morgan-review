-- How many times a worker has claimed the review: after a restart, one without a token
-- is queued once more, but not again and again (decision D-021, item 8).
ALTER TABLE reviews ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0);
-- The smallest fix a finding proposes; `body` holds why it matters (shared rules, Output).
ALTER TABLE findings ADD COLUMN suggestion TEXT NOT NULL DEFAULT '';
