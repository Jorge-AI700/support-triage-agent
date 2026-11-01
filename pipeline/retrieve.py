"""Vector retrieval for Knowledge Base chunks and similar historical tickets.

Uses pgvector cosine distance (<=>) in PostgreSQL with embeddings generated
by sentence-transformers (BAAI/bge-small-en-v1.5) on CPU.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Sequence

import psycopg
from psycopg.rows import dict_row

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_DATABASE_URL = "postgresql://support:support@localhost:5433/support"


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the sentence-transformers model once and cache it in memory."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_query(query: str) -> list[float]:
    """Generate a 384-dimensional normalized embedding for a text query."""
    model = get_embedding_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()


def vector_literal(vector: Sequence[float]) -> str:
    """Format a float vector into a pgvector text literal: [x1, x2, ...]."""
    return "[" + ",".join(f"{x:.8f}" for x in vector) + "]"


def retrieve_kb_chunks(
    query: str,
    k: int = 4,
    db_url: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k most relevant KB chunks using pgvector cosine distance.

    Each result includes chunk_id, article_id, kb_id (e.g. KB-01), title, intent,
    chunk text, and cosine similarity score.
    """
    db_url = db_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    query_vector = vector_literal(embed_query(query))

    # Plain, readable SQL with in-line explanation of each clause for interviews
    query_sql = """
        SELECT
            -- Primary key of the chunk in kb_chunks
            kc.id AS chunk_id,
            -- Foreign key to kb_articles
            kc.article_id,
            -- Formatted identifier used in citations (e.g., KB-01)
            CONCAT('KB-', LPAD(kc.article_id::text, 2, '0')) AS kb_id,
            -- Metadata from kb_articles for intent and title context
            ka.intent AS article_intent,
            ka.title AS article_title,
            -- The actual chunk content passed to the LLM context
            kc.chunk,
            -- Cosine similarity: pgvector's <=> operator returns cosine distance (1 - cos_sim).
            -- Subtracting from 1 yields standard cosine similarity where 1.0 is identical.
            (1 - (kc.embedding <=> %(vec)s::vector)) AS score
        FROM kb_chunks kc
        -- Inner join with kb_articles to attach article title and intent metadata
        JOIN kb_articles ka ON kc.article_id = ka.id
        -- Ensure chunk embedding is populated
        WHERE kc.embedding IS NOT NULL
        ORDER BY
            -- Rank by pgvector cosine distance ascending (<=>) to get highest similarity first
            kc.embedding <=> %(vec)s::vector ASC
        -- Limit to top-k most relevant chunks
        LIMIT %(limit)s;
    """

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query_sql, {"vec": query_vector, "limit": k})
            results = cur.fetchall()

    return [dict(row) for row in results]


def retrieve_similar_tickets(
    query: str,
    k: int = 3,
    db_url: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k similar historical tickets from the train split using pgvector.

    Each result includes ticket_id, instruction, intent, category, reference_response,
    and similarity score.
    """
    db_url = db_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    query_vector = vector_literal(embed_query(query))

    # Plain, readable SQL with in-line explanation of each clause for interviews
    query_sql = """
        SELECT
            -- Ticket primary key
            t.id AS ticket_id,
            -- The historical user request
            t.instruction,
            -- Ground-truth intent and category
            t.intent,
            t.category,
            -- The verified human reference response
            t.reference_response,
            -- Cosine similarity: 1 - cosine_distance (<=>)
            (1 - (te.embedding <=> %(vec)s::vector)) AS score
        FROM ticket_embeddings te
        -- Join tickets table to retrieve instruction text, labels, and reference response
        JOIN tickets t ON te.ticket_id = t.id
        -- Restrict strictly to 'train' split tickets to prevent data leakage from evaluation test sets
        WHERE t.split = 'train'
          AND te.embedding IS NOT NULL
        ORDER BY
            -- Rank by pgvector cosine distance ascending (<=>)
            te.embedding <=> %(vec)s::vector ASC
        -- Limit to top-k most similar past tickets
        LIMIT %(limit)s;
    """

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query_sql, {"vec": query_vector, "limit": k})
            results = cur.fetchall()

    return [dict(row) for row in results]


def retrieve(
    query: str,
    kb_k: int = 4,
    ticket_k: int = 3,
    db_url: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Retrieve both top-k KB chunks and top-k similar historical train tickets."""
    return {
        "kb_chunks": retrieve_kb_chunks(query, k=kb_k, db_url=db_url),
        "similar_tickets": retrieve_similar_tickets(query, k=ticket_k, db_url=db_url),
    }
