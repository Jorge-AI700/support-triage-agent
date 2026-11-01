"""Classify Bitext tickets with an LLM and persist predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from dotenv import load_dotenv

from pipeline.llm import LLMResponse, OpenRouterClient

load_dotenv()

DEFAULT_DATABASE_URL = "postgresql://support:support@localhost:5433/support"
PROMPT_VERSIONS = ("v1", "v2")
DEFAULT_PROMPT_VERSION = "v2"
PROMPT_VERSION_TAGS = {
    "v1": "bitext-intent-v1",
    "v2": "bitext-intent-v2",
}
KNN_K = 5
DEFAULT_WORKERS = 16

# Complete Bitext taxonomy. Categories deliberately do not live here: each run
# derives its fixed intent/category mapping from the ingested tickets table.
INTENT_DESCRIPTIONS_V1: dict[str, str] = {
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
INTENT_DESCRIPTIONS_V2: dict[str, str] = {
    "cancel_order": "Cancel an existing order; not edit its items or address.",
    "change_order": "Change items or quantities in an existing order; not cancel it.",
    "change_shipping_address": "Change the delivery address on an existing order; not add an account address.",
    "check_cancellation_fee": "Ask whether cancelling an order incurs a fee.",
    "check_invoice": "Look up or view an existing invoice's details; not request or download a copy.",
    "check_payment_methods": "Ask which payment methods are accepted.",
    "check_refund_policy": "Ask about refund rules or eligibility; not request or track a refund.",
    "complaint": "Make a complaint about the service or customer experience.",
    "contact_customer_service": "Ask how to contact customer support; not explicitly demand a human agent.",
    "contact_human_agent": "Explicitly request a live human agent.",
    "create_account": "Create a new account; not fix a registration failure.",
    "delete_account": "Delete or permanently close an account.",
    "delivery_options": "Ask which delivery or shipping methods are available; not how to place an order.",
    "delivery_period": "Ask how long delivery normally takes or for an ETA; not the current location/status of a placed order.",
    "edit_account": "Change profile or account details; not switch to a different existing account.",
    "get_invoice": "Request, receive, print, or download a copy of an invoice; not inspect its existing details.",
    "get_refund": "Request or initiate a new refund; not track an already requested refund.",
    "newsletter_subscription": "Subscribe, unsubscribe, or otherwise manage a newsletter subscription.",
    "payment_issue": "Report a failed, declined, duplicated, or otherwise problematic payment.",
    "place_order": "Ask how to submit or purchase a new order; not choose among delivery methods.",
    "recover_password": "Recover or reset a forgotten password.",
    "registration_problems": "Report a problem while signing up or registering; not simply create an account.",
    "review": "Submit or discuss a product or service review.",
    "set_up_shipping_address": "Add or configure a shipping address on the account; not change an order already placed.",
    "switch_account": "Move or sign in to a different existing account; not edit account/profile details.",
    "track_order": "Ask for the current status or location of an already placed order; not the usual delivery duration.",
    "track_refund": "Ask for the status or timing of an already requested refund; not request a new refund.",
}
# Backwards-compatible export used by validation and callers that inspect the
# taxonomy. V2 is the default while the exact v1 definitions remain versioned.
INTENT_DESCRIPTIONS = INTENT_DESCRIPTIONS_V2
ALLOWED_INTENTS = tuple(INTENT_DESCRIPTIONS_V1)

SYSTEM_PROMPT_PREFIX = """You classify customer-support tickets into exactly one Bitext intent.
Return only a strict JSON object with this schema:
{"intent": "one_allowed_intent", "confidence": 0.0}
Confidence must be a number from 0 to 1. Do not add markdown or other keys.

Allowed intents:
"""
SYSTEM_PROMPTS = {
    "v1": SYSTEM_PROMPT_PREFIX
    + "\n".join(
        f"- {intent}: {description}"
        for intent, description in INTENT_DESCRIPTIONS_V1.items()
    ),
    "v2": SYSTEM_PROMPT_PREFIX
    + "\n".join(
        f"- {intent}: {description}"
        for intent, description in INTENT_DESCRIPTIONS_V2.items()
    ),
}
# Compatibility alias for code that imported the original prompt directly.
SYSTEM_PROMPT = SYSTEM_PROMPTS["v1"]


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
    knn_majority_intent: str | None = None
    knn_agreement: float | None = None


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


def _validate_prompt_version(prompt_version: str) -> str:
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(
            f"prompt_version must be one of {PROMPT_VERSIONS}, got {prompt_version!r}"
        )
    return prompt_version


def _example_payload(similar_tickets: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return only stable prompt inputs from retrieved train tickets."""
    return [
        {"instruction": str(ticket.get("instruction", "")), "intent": str(ticket.get("intent", ""))}
        for ticket in similar_tickets[:KNN_K]
    ]


def knn_summary(
    similar_tickets: Sequence[Mapping[str, Any]],
) -> tuple[str | None, float | None]:
    """Return nearest-neighbor majority and its share, breaking ties by rank."""
    labels = [
        str(ticket.get("intent"))
        for ticket in similar_tickets[:KNN_K]
        if ticket.get("intent") in ALLOWED_INTENTS
    ]
    if not labels:
        return None, None
    majority, count = Counter(labels).most_common(1)[0]
    return majority, count / len(labels)


def build_classification_messages(
    instruction: str,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    similar_tickets: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, str]]:
    """Build the versioned classifier prompt with v2 train-only examples."""
    prompt_version = _validate_prompt_version(prompt_version)
    user_parts = [
        "Classify this customer-support ticket. Return only the required JSON."
    ]
    if prompt_version == "v2":
        examples = _example_payload(similar_tickets)
        if examples:
            user_parts.append(
                "Here are the five most similar labeled examples from the training set:\n"
                + "\n".join(
                    f"Example {index}: Ticket={json.dumps(example['instruction'])}; "
                    f"Intent={example['intent']}"
                    for index, example in enumerate(examples, 1)
                )
            )
    user_parts.append(f"Ticket: {instruction}")
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def cache_key(
    model: str,
    instruction: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    similar_tickets: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Return a key containing the model, prompt version, ticket, and examples."""
    prompt_version = _validate_prompt_version(prompt_version)
    if prompt_version == "v1":
        # Preserve v1 cache compatibility exactly.
        raw = f"{model}{PROMPT_VERSION_TAGS['v1']}{instruction}".encode("utf-8")
    else:
        prompt_inputs = json.dumps(
            _example_payload(similar_tickets), sort_keys=True, separators=(",", ":")
        )
        raw = (
            f"{model}{PROMPT_VERSION_TAGS[prompt_version]}{instruction}{prompt_inputs}"
        ).encode("utf-8")
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


def classify_instruction(
    instruction: str,
    client: LLMClient,
    cache: Cache,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    similar_tickets: Sequence[Mapping[str, Any]] = (),
) -> Classification:
    """Classify one instruction, consulting cache before the LLM."""
    prompt_version = _validate_prompt_version(prompt_version)
    neighbors = similar_tickets if prompt_version == "v2" else ()
    majority_intent, agreement = knn_summary(neighbors)
    key = cache_key(client.model, instruction, prompt_version, neighbors)
    cached = cache.get(key)
    if cached is not None:
        intent, confidence = _validate_payload(cached)
        return Classification(
            intent, confidence, {}, True, key, majority_intent, agreement
        )

    messages = build_classification_messages(
        instruction,
        prompt_version=prompt_version,
        similar_tickets=neighbors,
    )
    response = client.complete(messages)
    intent, confidence = parse_classification(response.content)
    cache.set(key, {"intent": intent, "confidence": confidence})
    return Classification(
        intent,
        confidence,
        response.usage,
        False,
        key,
        majority_intent,
        agreement,
    )


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


def retrieve_training_examples(
    instructions: Iterable[str],
    *,
    database_url: str,
    workers: int,
    retriever: Callable[..., list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Retrieve five train tickets for each unique instruction in parallel."""
    unique_instructions = tuple(dict.fromkeys(instructions))
    examples: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                retriever, instruction, k=KNN_K, db_url=database_url
            ): instruction
            for instruction in unique_instructions
        }
        for future in as_completed(futures):
            examples[futures[future]] = future.result()
    return examples


def run_classification(
    *,
    split: str,
    limit: int | None,
    run_id: str,
    workers: int = DEFAULT_WORKERS,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    database_url: str | None = None,
    client: LLMClient | None = None,
    ticket_retriever: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run classification and return aggregate run statistics."""
    import psycopg
    from psycopg.types.json import Jsonb

    database_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    client = client or OpenRouterClient()
    prompt_version = _validate_prompt_version(prompt_version)
    if ticket_retriever is None:
        from pipeline.retrieve import retrieve_similar_tickets

        ticket_retriever = retrieve_similar_tickets

    with psycopg.connect(database_url) as conn:
        intent_categories = get_intent_category_map(conn)
        tickets = fetch_tickets(conn, split, limit)

    if not tickets:
        raise ValueError(f"No tickets found for split {split!r}")

    examples_by_instruction = (
        retrieve_training_examples(
            (instruction for _, instruction in tickets),
            database_url=database_url,
            workers=workers,
            retriever=ticket_retriever,
        )
        if prompt_version == "v2"
        else {}
    )
    keys = {
        cache_key(
            client.model,
            instruction,
            prompt_version,
            examples_by_instruction.get(instruction, ()),
        )
        for _, instruction in tickets
    }
    with psycopg.connect(database_url) as conn:
        stored_cache = load_cache(conn, keys)

    cache = MemoryCache(stored_cache)
    # One future per unique instruction prevents concurrent cache stampedes when
    # the dataset contains duplicate ticket text.
    instruction_by_key = {
        cache_key(
            client.model,
            instruction,
            prompt_version,
            examples_by_instruction.get(instruction, ()),
        ): instruction
        for _, instruction in tickets
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
            executor.submit(
                classify_instruction,
                instruction,
                client,
                cache,
                prompt_version=prompt_version,
                similar_tickets=examples_by_instruction.get(instruction, ()),
            ): key
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
        result = results_by_key[
            cache_key(
                client.model,
                instruction,
                prompt_version,
                examples_by_instruction.get(instruction, ()),
            )
        ]
        prediction_rows.append(
            (
                ticket_id,
                client.model,
                result.intent,
                intent_categories.get(result.intent, "unknown"),
                result.confidence,
                run_id,
                prompt_version,
                result.knn_majority_intent,
                result.knn_agreement,
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
                    (ticket_id, model, pred_intent, pred_category, confidence, run_id,
                     prompt_version, knn_majority_intent, knn_agreement)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        "prompt_version": prompt_version,
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
    parser.add_argument(
        "--prompt-version",
        choices=PROMPT_VERSIONS,
        default=DEFAULT_PROMPT_VERSION,
    )
    args = parser.parse_args()
    run_classification(
        split=args.split,
        limit=args.limit,
        run_id=args.run_id,
        workers=args.workers,
        prompt_version=args.prompt_version,
    )


if __name__ == "__main__":
    main()
