"""Classify Bitext tickets with an LLM and persist predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from dotenv import load_dotenv

from pipeline.llm import LLMResponse, OpenRouterClient

load_dotenv()

DEFAULT_DATABASE_URL = "postgresql://support:support@localhost:5433/support"
PROMPT_VERSION = "bitext-intent-v1"
DEFAULT_WORKERS = 16

# Complete Bitext taxonomy. Categories deliberately do not live here: each run
# derives its fixed intent/category mapping from the ingested tickets table.
INTENT_DESCRIPTIONS: dict[str, str] = {
    "cancel_order": "The customer wants to cancel an existing order.",
    "change_order": "The customer wants to modify items or quantities in an order.",
    "change_shipping_address": "The customer wants to change an order's delivery address.",
    "check_cancellation_fee": "The customer asks whether cancelling has a fee.",
    "check_invoice": "The customer wants to inspect or verify an invoice.",
    "check_payment_methods": "The customer asks which payment methods are accepted.",
    "check_refund_policy": "The customer asks about refund rules or eligibility.",
    "complaint": "The customer wants to make a complaint about service or an experience.",
    "contact_customer_service": "The customer asks how to contact customer support.",
    "contact_human_agent": "The customer explicitly wants a live human agent.",
    "create_account": "The customer wants to create a new account.",
    "delete_account": "The customer wants to delete or close an account.",
    "delivery_options": "The customer asks about available delivery methods.",
    "delivery_period": "The customer asks how long delivery will take.",
    "edit_account": "The customer wants to change account or profile details.",
    "get_invoice": "The customer wants to obtain, download, or print an invoice.",
    "get_refund": "The customer wants to request or receive a refund.",
    "newsletter_subscription": "The customer wants to manage a newsletter subscription.",
    "payment_issue": "The customer reports a payment or checkout problem.",
    "place_order": "The customer needs help placing a new order.",
    "recover_password": "The customer forgot or needs to reset a password.",
    "registration_problems": "The customer has a problem signing up or registering.",
    "review": "The customer wants to submit or discuss a product or service review.",
    "set_up_shipping_address": "The customer wants to add or configure a shipping address.",
    "switch_account": "The customer wants to switch between accounts.",
    "track_order": "The customer wants the status or location of an order.",
    "track_refund": "The customer wants the status of an already requested refund.",
}
ALLOWED_INTENTS = tuple(INTENT_DESCRIPTIONS)

SYSTEM_PROMPT = """You classify customer-support tickets into exactly one Bitext intent.
Return only a strict JSON object with this schema:
{"intent": "one_allowed_intent", "confidence": 0.0}
Confidence must be a number from 0 to 1. Do not add markdown or other keys.

Allowed intents:
""" + "\n".join(
    f"- {intent}: {description}" for intent, description in INTENT_DESCRIPTIONS.items()
)


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse: ...


class Cache(Protocol):
    def get(self, key: str) -> Mapping[str, Any] | None: ...

    def set(self, key: str, response: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class Classification:
    intent: str
    confidence: float
    usage: dict[str, Any]
    cache_hit: bool
    cache_key: str


class MemoryCache:
    """Thread-safe run-local cache, preloaded from and later flushed to Postgres."""

    def __init__(self, responses: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self._responses = {key: dict(value) for key, value in (responses or {}).items()}
        self._initial_keys = set(self._responses)
        self._lock = threading.Lock()

    def get(self, key: str) -> Mapping[str, Any] | None:
        with self._lock:
            response = self._responses.get(key)
            return dict(response) if response is not None else None

    def set(self, key: str, response: Mapping[str, Any]) -> None:
        with self._lock:
            self._responses[key] = dict(response)

    def pending(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                key: dict(value)
                for key, value in self._responses.items()
                if key not in self._initial_keys
            }


def cache_key(model: str, instruction: str) -> str:
    """Return the required model + prompt-version + instruction SHA-256 key."""
    raw = f"{model}{PROMPT_VERSION}{instruction}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_classification(content: str) -> tuple[str, float]:
    """Parse and validate the strict classification JSON response."""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return "unknown", 0.0
    return _validate_payload(payload)


def _validate_payload(payload: Any) -> tuple[str, float]:
    if not isinstance(payload, Mapping):
        return "unknown", 0.0

    intent = payload.get("intent")
    confidence = payload.get("confidence")
    if intent not in INTENT_DESCRIPTIONS:
        return "unknown", 0.0
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return "unknown", 0.0
    if not 0 <= float(confidence) <= 1:
        return "unknown", 0.0
    return str(intent), float(confidence)


def classify_instruction(instruction: str, client: LLMClient, cache: Cache) -> Classification:
    """Classify one instruction, consulting cache before the LLM."""
    key = cache_key(client.model, instruction)
    cached = cache.get(key)
    if cached is not None:
        intent, confidence = _validate_payload(cached)
        return Classification(intent, confidence, {}, True, key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Classify this customer-support ticket. Return only the required JSON.\n\n"
                f"Ticket: {instruction}"
            ),
        },
    ]
    response = client.complete(messages)
    intent, confidence = parse_classification(response.content)
    cache.set(key, {"intent": intent, "confidence": confidence})
    return Classification(intent, confidence, response.usage, False, key)


def get_intent_category_map(conn: Any) -> dict[str, str]:
    """Build and validate one immutable intent/category mapping for this run."""
    rows = conn.execute(
        """
        SELECT intent, category
        FROM tickets
        WHERE intent = ANY(%s)
        GROUP BY intent, category
        ORDER BY intent, category
        """,
        (list(ALLOWED_INTENTS),),
    ).fetchall()

    mapping: dict[str, str] = {}
    for intent, category in rows:
        if intent in mapping and mapping[intent] != category:
            raise ValueError(f"Intent {intent!r} has multiple categories in tickets")
        if category is None:
            raise ValueError(f"Intent {intent!r} has a null category in tickets")
        mapping[intent] = category

    missing = sorted(set(ALLOWED_INTENTS) - mapping.keys())
    if missing:
        raise ValueError(f"tickets table is missing intent/category mappings: {missing}")
    return mapping


def fetch_tickets(conn: Any, split: str, limit: int | None) -> list[tuple[int, str]]:
    rows = conn.execute(
        "SELECT id, instruction FROM tickets WHERE split = %s ORDER BY id",
        (split,),
    ).fetchall()
    tickets = [(int(ticket_id), instruction) for ticket_id, instruction in rows]
    if limit is not None and limit < len(tickets):
        tickets = random.Random(42).sample(tickets, limit)
        tickets.sort(key=lambda item: item[0])
    return tickets


def load_cache(conn: Any, keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    key_list = list(keys)
    if not key_list:
        return {}
    rows = conn.execute(
        "SELECT key, response FROM llm_cache WHERE key = ANY(%s)",
        (key_list,),
    ).fetchall()
    return {key: dict(response) for key, response in rows}


def _usage_tokens(usage: Mapping[str, Any]) -> int:
    total = usage.get("total_tokens")
    if isinstance(total, (int, float)):
        return int(total)
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    return int(prompt) + int(completion)


def _usage_cost(usage: Mapping[str, Any]) -> float:
    try:
        return float(usage.get("cost") or 0)
    except (TypeError, ValueError):
        return 0.0


def run_classification(
    *,
    split: str,
    limit: int | None,
    run_id: str,
    workers: int = DEFAULT_WORKERS,
    database_url: str | None = None,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Run classification and return aggregate run statistics."""
    import psycopg
    from psycopg.types.json import Jsonb

    database_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    client = client or OpenRouterClient()

    with psycopg.connect(database_url) as conn:
        intent_categories = get_intent_category_map(conn)
        tickets = fetch_tickets(conn, split, limit)
        keys = {cache_key(client.model, instruction) for _, instruction in tickets}
        stored_cache = load_cache(conn, keys)

    if not tickets:
        raise ValueError(f"No tickets found for split {split!r}")

    cache = MemoryCache(stored_cache)
    # One future per unique instruction prevents concurrent cache stampedes when
    # the dataset contains duplicate ticket text.
    instruction_by_key = {
        cache_key(client.model, instruction): instruction for _, instruction in tickets
    }
    results_by_key: dict[str, Classification] = {}
    total_tokens = 0
    total_cost = 0.0
    cache_hits = 0
    completed = 0
    progress_every = max(1, min(25, len(instruction_by_key)))

    print(
        f"Classifying {len(tickets)} tickets "
        f"({len(instruction_by_key)} unique) for run {run_id!r}..."
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(classify_instruction, instruction, client, cache): key
            for key, instruction in instruction_by_key.items()
        }
        for future in as_completed(futures):
            result = future.result()
            results_by_key[result.cache_key] = result
            total_tokens += _usage_tokens(result.usage)
            total_cost += _usage_cost(result.usage)
            cache_hits += int(result.cache_hit)
            completed += 1
            if completed % progress_every == 0 or completed == len(futures):
                print(f"Progress: {completed}/{len(futures)} unique tickets")

    prediction_rows = []
    for ticket_id, instruction in tickets:
        result = results_by_key[cache_key(client.model, instruction)]
        prediction_rows.append(
            (
                ticket_id,
                client.model,
                result.intent,
                intent_categories.get(result.intent, "unknown"),
                result.confidence,
                run_id,
            )
        )

    with psycopg.connect(database_url) as conn:
        pending_cache = cache.pending()
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
            # A run ID represents one reproducible result set; rerunning it replaces
            # only that run's prior predictions instead of double-counting them.
            cursor.execute("DELETE FROM predictions WHERE run_id = %s", (run_id,))
            cursor.executemany(
                """
                INSERT INTO predictions
                    (ticket_id, model, pred_intent, pred_category, confidence, run_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                prediction_rows,
            )

    stats = {
        "run_id": run_id,
        "tickets": len(tickets),
        "unique_instructions": len(instruction_by_key),
        "cache_hits": cache_hits,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
    }
    print(f"Wrote {len(prediction_rows)} predictions for run {run_id!r}.")
    print(f"Cache hits: {cache_hits}/{len(instruction_by_key)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total cost: ${total_cost:.6f} USD")
    return stats


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    run_classification(
        split=args.split,
        limit=args.limit,
        run_id=args.run_id,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
