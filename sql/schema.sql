CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tickets (
    id SERIAL PRIMARY KEY,
    instruction TEXT NOT NULL,
    category TEXT,
    intent TEXT NOT NULL,
    flags TEXT,
    reference_response TEXT,
    split TEXT CHECK (split IN ('train', 'test')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    ticket_id INT REFERENCES tickets(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    pred_intent TEXT,
    pred_category TEXT,
    confidence REAL,
    escalated BOOLEAN,
    run_id TEXT,
    prompt_version TEXT NOT NULL DEFAULT 'v1' CHECK (prompt_version IN ('v1', 'v2')),
    knn_majority_intent TEXT,
    knn_agreement REAL CHECK (knn_agreement BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kb_articles (
    id SERIAL PRIMARY KEY,
    intent TEXT,
    title TEXT NOT NULL,
    body_md TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id SERIAL PRIMARY KEY,
    article_id INT REFERENCES kb_articles(id) ON DELETE CASCADE,
    chunk TEXT NOT NULL,
    embedding VECTOR(384)
);

CREATE TABLE IF NOT EXISTS ticket_embeddings (
    ticket_id INT REFERENCES tickets(id) ON DELETE CASCADE,
    embedding VECTOR(384),
    PRIMARY KEY (ticket_id)
);

CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    response JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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
    cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    prompt_version TEXT NOT NULL DEFAULT 'v1' CHECK (prompt_version IN ('v1', 'v2')),
    knn_majority_intent TEXT,
    knn_agreement REAL CHECK (knn_agreement BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, ticket_id)
);

-- Indexes for efficient querying, filtering, and vector similarity search
CREATE INDEX IF NOT EXISTS idx_tickets_split ON tickets(split);
CREATE INDEX IF NOT EXISTS idx_tickets_intent ON tickets(intent);
CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets(category);
CREATE INDEX IF NOT EXISTS idx_predictions_ticket_id ON predictions(ticket_id);
CREATE INDEX IF NOT EXISTS idx_predictions_run_id ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_kb_articles_intent ON kb_articles(intent);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_article_id ON kb_chunks(article_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding ON kb_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_ticket_embeddings_embedding ON ticket_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_e2e_results_run_id ON e2e_results(run_id);
