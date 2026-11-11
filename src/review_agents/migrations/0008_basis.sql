-- Why a finding holds: a list of bases, each with how the program's check came out
-- (decision D-036). It replaces the free-text evidence, which is kept, written as the
-- bases' text, so an older version reading it still works; a later migration drops it.
ALTER TABLE findings ADD COLUMN basis TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(basis));
