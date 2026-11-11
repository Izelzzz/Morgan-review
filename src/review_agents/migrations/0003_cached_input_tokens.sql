-- Input tokens a model service read from its prompt cache, part of input_tokens.
ALTER TABLE usage ADD COLUMN cached_input_tokens INTEGER NOT NULL DEFAULT 0
    CHECK (cached_input_tokens >= 0);
ALTER TABLE summary_usage ADD COLUMN cached_input_tokens INTEGER NOT NULL DEFAULT 0
    CHECK (cached_input_tokens >= 0);
