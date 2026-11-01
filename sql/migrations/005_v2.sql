-- V2 prompt provenance and retrieval-neighbor diagnostics. Historical rows are
-- v1 by construction; new runners always write their explicit prompt version.
ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS prompt_version TEXT NOT NULL DEFAULT 'v1'
        CHECK (prompt_version IN ('v1', 'v2')),
    ADD COLUMN IF NOT EXISTS knn_majority_intent TEXT,
    ADD COLUMN IF NOT EXISTS knn_agreement REAL
        CHECK (knn_agreement BETWEEN 0 AND 1);

-- E2E can classify without first writing a predictions row, so retain the same
-- diagnostics here for reproducibility and comparison audits.
ALTER TABLE e2e_results
    ADD COLUMN IF NOT EXISTS prompt_version TEXT NOT NULL DEFAULT 'v1'
        CHECK (prompt_version IN ('v1', 'v2')),
    ADD COLUMN IF NOT EXISTS knn_majority_intent TEXT,
    ADD COLUMN IF NOT EXISTS knn_agreement REAL
        CHECK (knn_agreement BETWEEN 0 AND 1);
