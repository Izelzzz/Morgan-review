-- Why a finding holds: a list of bases, each with how the program's check came out
-- (decision D-036). It replaces the free-text evidence.
ALTER TABLE findings ADD COLUMN basis TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(basis));
ALTER TABLE findings DROP COLUMN evidence;
