CREATE TABLE IF NOT EXISTS e2e_results (
    ticket_id INT NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL,
    pred_intent TEXT NOT NULL,
    true_intent TEXT NOT NULL,
    confidence REAL NOT NULL,
    reply TEXT NOT NULL,
    citations JSONB NOT NULL,
    grounded BOOLEAN NOT NULL,
    escalated BOOLEAN NOT NULL,
    reasons JSONB NOT NULL,
    top_kb_score REAL,
    kb_hit BOOLEAN NOT NULL,
    -- Incremental provider cost attributable to this run. Cached/reused work is
    -- zero; storing it is necessary for the requested report-level cost KPI.
    cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, ticket_id)
);

CREATE INDEX IF NOT EXISTS idx_e2e_results_run_id ON e2e_results(run_id);
