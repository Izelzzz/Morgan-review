-- The level the dimension agent reported; `severity` is the one published, which a
-- verifier may have raised or lowered (decision D-027).
ALTER TABLE findings ADD COLUMN reported_severity TEXT
    CHECK (reported_severity IN ('blocking', 'important', 'robustness', 'nit'));
-- What a step found besides an error, such as a verifier's verdict on a finding that was
-- not published.
ALTER TABLE steps ADD COLUMN note TEXT;
