"""Run the deterministic test-ticket classification, RAG, and escalation flow."""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from pipeline.citations import check_citations
from pipeline.classify import (
    DEFAULT_PROMPT_VERSION,
    DEFAULT_DATABASE_URL,
    PROMPT_VERSIONS,
    MemoryCache,
    cache_key as classification_cache_key,
    classify_instruction,
    knn_summary,
    load_cache,
    retrieve_training_examples,
)
from pipeline.escalate import should_escalate
from pipeline.llm import OpenRouterClient
from pipeline.rag import draft_reply, reply_cache_key
from pipeline.retrieve import retrieve_kb_chunks, retrieve_similar_tickets

load_dotenv()

DEFAULT_LIMIT = 200
DEFAULT_WORKERS = 8
SAMPLE_SEED = 42
KB_K = 4


@dataclass(frozen=True)
class Ticket:
    ticket_id: int
    text: str
    true_intent: str


@dataclass(frozen=True)
class Prediction:
    intent: str
    confidence: float
    knn_majority_intent: str | None = None
    knn_agreement: float | None = None


def fetch_test_tickets(
    conn: Any,
    limit: int = DEFAULT_LIMIT,
    seed: int = SAMPLE_SEED,
) -> list[Ticket]:
    """Fetch a stable random sample from the test split."""
    rows = conn.execute(
        "SELECT id, instruction, intent FROM tickets WHERE split = 'test' ORDER BY id"
    ).fetchall()
    tickets = [Ticket(int(row[0]), str(row[1]), str(row[2])) for row in rows]
    if limit < len(tickets):
        tickets = random.Random(seed).sample(tickets, limit)
        tickets.sort(key=lambda ticket: ticket.ticket_id)
    return tickets


def fetch_predictions(
    conn: Any,
    ticket_ids: Sequence[int],
    classify_run_id: str | None,
    prompt_version: str,
) -> dict[int, Prediction]:
    """Load the newest prediction per sampled ticket from a requested run."""
    if not classify_run_id or not ticket_ids:
        return {}
    rows = conn.execute(
        """
        SELECT DISTINCT ON (ticket_id)
            ticket_id, pred_intent, confidence,
            knn_majority_intent, knn_agreement
        FROM predictions
        WHERE run_id = %s AND prompt_version = %s AND ticket_id = ANY(%s)
        ORDER BY ticket_id, id DESC
        """,
        (classify_run_id, prompt_version, list(ticket_ids)),
    ).fetchall()
    return {
        int(ticket_id): Prediction(
            str(intent or "unknown"),
            float(confidence or 0.0),
            str(majority_intent) if majority_intent is not None else None,
            float(agreement) if agreement is not None else None,
        )
        for ticket_id, intent, confidence, majority_intent, agreement in rows
    }


def _usage_cost(usage: Mapping[str, Any]) -> float:
    try:
        return float(usage.get("cost") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _classify_missing(
    tickets: Sequence[Ticket],
    predictions: dict[int, Prediction],
    client: Any,
    cache: MemoryCache,
    workers: int,
    prompt_version: str,
    neighbors_by_ticket: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[int, float]:
    """Classify only missing tickets, once per unique cache key.

    Provider cost is assigned to one representative ticket for each unique call,
    which makes summing row costs equal the actual incremental run cost.
    """
    missing = [ticket for ticket in tickets if ticket.ticket_id not in predictions]
    tickets_by_key: dict[str, list[Ticket]] = {}
    for ticket in missing:
        key = classification_cache_key(
            client.model,
            ticket.text,
            prompt_version,
            neighbors_by_ticket.get(ticket.ticket_id, ()),
        )
        tickets_by_key.setdefault(key, []).append(ticket)

    cost_by_ticket: dict[int, float] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                classify_instruction,
                group[0].text,
                client,
                cache,
                prompt_version=prompt_version,
                similar_tickets=neighbors_by_ticket.get(group[0].ticket_id, ()),
            ): key
            for key, group in tickets_by_key.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            result = future.result()
            group = tickets_by_key[key]
            for ticket in group:
                predictions[ticket.ticket_id] = Prediction(
                    result.intent,
                    result.confidence,
                    result.knn_majority_intent,
                    result.knn_agreement,
                )
            cost_by_ticket[group[0].ticket_id] = _usage_cost(result.usage)
    return cost_by_ticket


def _retrieve_all(
    tickets: Sequence[Ticket],
    database_url: str,
    workers: int,
    retriever: Callable[..., list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """Retrieve the top four KB chunks for every sampled ticket in parallel."""
    retrieved: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(retriever, ticket.text, k=KB_K, db_url=database_url): ticket
            for ticket in tickets
        }
        for future in as_completed(futures):
            ticket = futures[future]
            retrieved[ticket.ticket_id] = future.result()
    return retrieved


def _draft_all(
    tickets: Sequence[Ticket],
    predictions: Mapping[int, Prediction],
    retrieved: Mapping[int, list[dict[str, Any]]],
    client: Any,
    cache: MemoryCache,
    workers: int,
    prompt_version: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, float]]:
    """Draft once per unique prompt and fan the result out to duplicate tickets."""
    tickets_by_key: dict[str, list[Ticket]] = {}
    for ticket in tickets:
        prediction = predictions[ticket.ticket_id]
        key = reply_cache_key(
            client.model,
            ticket.text,
            prediction.intent,
            retrieved[ticket.ticket_id],
            prompt_version,
        )
        tickets_by_key.setdefault(key, []).append(ticket)

    draft_by_ticket: dict[int, dict[str, Any]] = {}
    cost_by_ticket: dict[int, float] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for key, group in tickets_by_key.items():
            ticket = group[0]
            prediction = predictions[ticket.ticket_id]
            futures[
                executor.submit(
                    draft_reply,
                    ticket.text,
                    prediction.intent,
                    client=client,
                    retrieved_chunks=retrieved[ticket.ticket_id],
                    cache=cache,
                    prompt_version=prompt_version,
                )
            ] = key

        for future in as_completed(futures):
            key = futures[future]
            result = future.result()
            group = tickets_by_key[key]
            for ticket in group:
                draft_by_ticket[ticket.ticket_id] = result
            cost_by_ticket[group[0].ticket_id] = _usage_cost(result["usage"])
    return draft_by_ticket, cost_by_ticket


def run_e2e(
    *,
    run_id: str,
    limit: int = DEFAULT_LIMIT,
    classify_run_id: str | None = None,
    workers: int = DEFAULT_WORKERS,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    database_url: str | None = None,
    client: Any | None = None,
    retriever: Callable[..., list[dict[str, Any]]] = retrieve_kb_chunks,
    ticket_retriever: Callable[..., list[dict[str, Any]]] = retrieve_similar_tickets,
) -> dict[str, Any]:
    """Run E2E evaluation and replace the stored rows for ``run_id``."""
    import psycopg
    from psycopg.types.json import Jsonb

    if not run_id or not run_id.strip():
        raise ValueError("run_id must not be empty")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if workers <= 0:
        raise ValueError("workers must be greater than zero")
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(f"prompt_version must be one of {PROMPT_VERSIONS}")

    database_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    client = client or OpenRouterClient()

    with psycopg.connect(database_url) as conn:
        tickets = fetch_test_tickets(conn, limit=limit)
        if not tickets:
            raise ValueError("No tickets found in the test split")
        predictions = fetch_predictions(
            conn,
            [ticket.ticket_id for ticket in tickets],
            classify_run_id,
            prompt_version,
        )

    examples_by_text = (
        retrieve_training_examples(
            (ticket.text for ticket in tickets),
            database_url=database_url,
            workers=workers,
            retriever=ticket_retriever,
        )
        if prompt_version == "v2"
        else {}
    )
    neighbors_by_ticket = {
        ticket.ticket_id: examples_by_text.get(ticket.text, []) for ticket in tickets
    }
    if prompt_version == "v2":
        # A reused v2 prediction should already have these fields. Recomputing
        # missing values keeps older partial runs safe after the v2 migration.
        for ticket_id, prediction in list(predictions.items()):
            if prediction.knn_majority_intent is None:
                majority, agreement = knn_summary(neighbors_by_ticket[ticket_id])
                predictions[ticket_id] = Prediction(
                    prediction.intent,
                    prediction.confidence,
                    majority,
                    agreement,
                )

    with psycopg.connect(database_url) as conn:
        missing_keys = {
            classification_cache_key(
                client.model,
                ticket.text,
                prompt_version,
                neighbors_by_ticket.get(ticket.ticket_id, ()),
            )
            for ticket in tickets
            if ticket.ticket_id not in predictions
        }
        classification_cache = MemoryCache(load_cache(conn, missing_keys))

    reused_predictions = len(predictions)
    classification_costs = _classify_missing(
        tickets,
        predictions,
        client,
        classification_cache,
        workers,
        prompt_version,
        neighbors_by_ticket,
    )

    print(
        f"Running retrieval for {len(tickets)} tickets "
        f"({reused_predictions} reused predictions)..."
    )
    retrieved = _retrieve_all(tickets, database_url, workers, retriever)

    reply_keys = {
        reply_cache_key(
            client.model,
            ticket.text,
            predictions[ticket.ticket_id].intent,
            retrieved[ticket.ticket_id],
            prompt_version,
        )
        for ticket in tickets
    }
    with psycopg.connect(database_url) as conn:
        reply_cache = MemoryCache(load_cache(conn, reply_keys))

    drafts, reply_costs = _draft_all(
        tickets,
        predictions,
        retrieved,
        client,
        reply_cache,
        workers,
        prompt_version,
    )

    rows: list[tuple[Any, ...]] = []
    grounded_count = 0
    escalated_count = 0
    kb_hit_count = 0
    total_cost = 0.0
    for ticket in tickets:
        prediction = predictions[ticket.ticket_id]
        chunks = retrieved[ticket.ticket_id]
        draft = drafts[ticket.ticket_id]
        retrieved_article_ids = [
            chunk.get("kb_id", chunk.get("article_id")) for chunk in chunks
        ]
        citation_check = check_citations(
            draft["reply"],
            retrieved_article_ids,
            prompt_version=prompt_version,
        )
        scores = [float(chunk["score"]) for chunk in chunks if chunk.get("score") is not None]
        escalation = should_escalate(
            prediction.confidence,
            citation_check,
            scores,
            ticket.text,
            prediction.intent,
            prompt_version=prompt_version,
            knn_majority_intent=prediction.knn_majority_intent,
            knn_agreement=prediction.knn_agreement,
        )
        top_kb_score = max(scores, default=None)
        kb_hit = any(
            str(chunk.get("article_intent")) == ticket.true_intent for chunk in chunks
        )
        row_cost = classification_costs.get(ticket.ticket_id, 0.0) + reply_costs.get(
            ticket.ticket_id, 0.0
        )

        grounded_count += int(citation_check["grounded"])
        escalated_count += int(escalation["escalate"])
        kb_hit_count += int(kb_hit)
        total_cost += row_cost
        rows.append(
            (
                ticket.ticket_id,
                run_id,
                prediction.intent,
                ticket.true_intent,
                prediction.confidence,
                draft["reply"],
                Jsonb(draft["citations"]),
                citation_check["grounded"],
                escalation["escalate"],
                Jsonb(escalation["reasons"]),
                top_kb_score,
                kb_hit,
                row_cost,
                prompt_version,
                prediction.knn_majority_intent,
                prediction.knn_agreement,
            )
        )

    pending_cache = classification_cache.pending()
    pending_cache.update(reply_cache.pending())
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cursor:
            if pending_cache:
                cursor.executemany(
                    """
                    INSERT INTO llm_cache (key, response)
                    VALUES (%s, %s)
                    ON CONFLICT (key) DO NOTHING
                    """,
                    [(key, Jsonb(response)) for key, response in pending_cache.items()],
                )
            cursor.execute("DELETE FROM e2e_results WHERE run_id = %s", (run_id,))
            cursor.executemany(
                """
                INSERT INTO e2e_results (
                    ticket_id, run_id, pred_intent, true_intent, confidence,
                    reply, citations, grounded, escalated, reasons,
                    top_kb_score, kb_hit, cost_usd, prompt_version,
                    knn_majority_intent, knn_agreement
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s)
                """,
                rows,
            )

    stats = {
        "run_id": run_id,
        "tickets": len(tickets),
        "reused_predictions": reused_predictions,
        "grounded": grounded_count,
        "escalated": escalated_count,
        "kb_hits": kb_hit_count,
        "total_cost_usd": total_cost,
        "prompt_version": prompt_version,
    }
    print(f"Wrote {len(rows)} E2E results for run {run_id!r}.")
    print(f"Incremental cost: ${total_cost:.6f} USD")
    return stats


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=_positive_int, default=DEFAULT_LIMIT)
    parser.add_argument("--classify-run-id")
    parser.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--prompt-version",
        choices=PROMPT_VERSIONS,
        default=DEFAULT_PROMPT_VERSION,
    )
    args = parser.parse_args()
    run_e2e(
        run_id=args.run_id,
        limit=args.limit,
        classify_run_id=args.classify_run_id,
        workers=args.workers,
        prompt_version=args.prompt_version,
    )


if __name__ == "__main__":
    main()
