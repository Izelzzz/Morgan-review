-- The report of a finished review: the summary comment, the inline comments and the
-- counts (report design §7), written in the same transaction as its findings. Reviews
-- that failed or were superseded have none.
ALTER TABLE reviews ADD COLUMN report TEXT CHECK (report IS NULL OR json_valid(report));
