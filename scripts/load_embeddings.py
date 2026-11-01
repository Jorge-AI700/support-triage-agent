"""Load Colab-generated ticket embeddings into PostgreSQL/pgvector."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


EMBEDDINGS_FILE = Path(__file__).resolve().parent.parent / "data" / "embeddings.parquet"
DEFAULT_DATABASE_URL = "postgresql://support:support@localhost:5433/support"
REQUIRED_COLUMNS = {"instruction", "intent", "category", "embedding"}
EMBEDDING_DIMENSIONS = 384


def vector_literal(value: object) -> str:
    """Validate one embedding and format it for pgvector's text input."""
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (EMBEDDING_DIMENSIONS,):
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS} embedding dimensions, got {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ValueError("Embedding contains a non-finite value")
    return "[" + ",".join(map(str, vector.tolist())) + "]"


def load_embeddings(
    parquet_path: Path = EMBEDDINGS_FILE,
    database_url: str | None = None,
) -> None:
    """Match embedding rows to every train ticket with the same instruction and intent."""
    embeddings = pd.read_parquet(parquet_path)
    missing_columns = REQUIRED_COLUMNS.difference(embeddings.columns)
    if missing_columns:
        raise ValueError(f"Missing parquet columns: {', '.join(sorted(missing_columns))}")

    database_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, instruction, intent FROM tickets WHERE split = 'train'"
            )
            train_tickets = cur.fetchall()

            ticket_ids_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
            for ticket_id, instruction, intent in train_tickets:
                ticket_ids_by_key[(instruction, intent)].append(ticket_id)

            records_by_ticket: dict[int, object] = {}
            matched_source_rows: set[int] = set()
            for source_index, row in embeddings.iterrows():
                ticket_ids = ticket_ids_by_key.get((row["instruction"], row["intent"]), [])
                if ticket_ids:
                    matched_source_rows.add(source_index)
                for ticket_id in ticket_ids:
                    records_by_ticket[ticket_id] = row["embedding"]

            unmatched = embeddings.loc[~embeddings.index.isin(matched_source_rows)]

            cur.execute(
                """
                CREATE TEMP TABLE embedding_load (
                    ticket_id INT PRIMARY KEY,
                    embedding vector(384)
                ) ON COMMIT DROP
                """
            )
            with cur.copy(
                "COPY embedding_load (ticket_id, embedding) FROM STDIN"
            ) as copy:
                for ticket_id, embedding in records_by_ticket.items():
                    copy.write_row((ticket_id, vector_literal(embedding)))

            cur.execute(
                """
                INSERT INTO ticket_embeddings (ticket_id, embedding)
                SELECT ticket_id, embedding FROM embedding_load
                ON CONFLICT (ticket_id) DO UPDATE
                SET embedding = EXCLUDED.embedding
                """
            )
            upserted_count = cur.rowcount
            cur.execute("SELECT COUNT(*) FROM ticket_embeddings")
            total_count = cur.fetchone()[0]

    print(f"Parquet rows: {len(embeddings)}")
    print(f"Matched parquet rows: {len(matched_source_rows)}")
    print(f"Matched ticket IDs: {len(records_by_ticket)}")
    print(f"Inserted or updated embeddings: {upserted_count}")
    print(f"Total rows in ticket_embeddings: {total_count}")
    print(f"Unmatched parquet rows: {len(unmatched)}")
    if not unmatched.empty:
        print(unmatched[["instruction", "intent", "category"]].to_string(index=False))


if __name__ == "__main__":
    load_embeddings()
