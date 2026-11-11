-- Runs that reported the finding; Deep mode runs each dimension three times (D-022).
ALTER TABLE findings ADD COLUMN votes INTEGER NOT NULL DEFAULT 1 CHECK (votes >= 1);
-- Other places one kind of problem appears, as [file, line_start, line_end] triples
-- (architecture §4.4: report a kind of bug once, with every location).
ALTER TABLE findings ADD COLUMN locations TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(locations));
