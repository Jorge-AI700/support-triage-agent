"""Embed knowledge base chunks in PostgreSQL using BAAI/bge-small-en-v1.5.

Generates 384-dimensional normalized vector embeddings on CPU for each row in
kb_chunks and stores them in the embedding column for pgvector cosine search.
"""

from __future__ import annotations

import os
from pathlib import Path
import psycopg
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_DATABASE_URL = "postgresql://support:support@localhost:5433/support"


def embed_kb(db_url: str | None = None) -> int:
    """Embed all kb_chunks with BAAI/bge-small-en-v1.5 and update PostgreSQL."""
    db_url = db_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    print(f"Loading embedding model: {MODEL_NAME} on CPU...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, chunk FROM kb_chunks ORDER BY id;")
            rows = cur.fetchall()
            if not rows:
                print("No kb_chunks found to embed.")
                return 0

            chunk_ids = [r[0] for r in rows]
            chunk_texts = [r[1] for r in rows]

            print(f"Embedding {len(chunk_texts)} chunks (normalize_embeddings=True)...")
            embeddings = model.encode(
                chunk_texts,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

            print("Updating kb_chunks table with vector embeddings...")
            with conn.transaction():
                for chunk_id, emb in zip(chunk_ids, embeddings):
                    # pgvector accepts vector literal format: [0.123, -0.456, ...]
                    vector_literal = "[" + ",".join(f"{x:.8f}" for x in emb.tolist()) + "]"
                    cur.execute(
                        "UPDATE kb_chunks SET embedding = %s WHERE id = %s;",
                        (vector_literal, chunk_id),
                    )

            cur.execute("SELECT COUNT(*), COUNT(embedding) FROM kb_chunks;")
            total, embedded = cur.fetchone()
            print(f"Finished embedding kb_chunks: {embedded}/{total} populated.")
            return embedded


if __name__ == "__main__":
    embed_kb()
