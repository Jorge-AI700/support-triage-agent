-- BGE-small produces 384-dimensional vectors for both ticket and KB retrieval.
-- Existing KB vectors were produced by a different embedder and cannot be resized,
-- so clear only those vectors while preserving their chunks for re-embedding.
DROP INDEX IF EXISTS idx_kb_chunks_embedding;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_attribute
        WHERE attrelid = 'kb_chunks'::regclass
          AND attname = 'embedding'
          AND NOT attisdropped
          AND format_type(atttypid, atttypmod) <> 'vector(384)'
    ) THEN
        ALTER TABLE kb_chunks
            ALTER COLUMN embedding TYPE vector(384)
            USING NULL::vector(384);
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding
    ON kb_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS ticket_embeddings (
    ticket_id INT REFERENCES tickets(id) ON DELETE CASCADE,
    embedding vector(384),
    PRIMARY KEY (ticket_id)
);

CREATE INDEX IF NOT EXISTS idx_ticket_embeddings_embedding
    ON ticket_embeddings USING hnsw (embedding vector_cosine_ops);
