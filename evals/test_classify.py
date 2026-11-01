"""Unit tests for LLM classification parsing, validation, and caching."""

from pipeline.classify import (
    INTENT_DESCRIPTIONS_V2,
    MemoryCache,
    cache_key,
    classify_instruction,
    knn_summary,
    parse_classification,
    retrieve_training_examples,
)
from pipeline.llm import LLMResponse


class MockLLMClient:
    model = "mock/model"

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0
        self.last_messages = None

    def complete(self, messages):
        self.calls += 1
        self.last_messages = messages
        return LLMResponse(
            content=self.content,
            usage={"total_tokens": 12, "cost": 0.00001},
        )


def test_parse_classification_json():
    assert parse_classification('{"intent":"track_order","confidence":0.87}') == (
        "track_order",
        0.87,
    )


def test_invalid_label_becomes_unknown():
    assert parse_classification('{"intent":"shipping_problem","confidence":0.95}') == (
        "unknown",
        0.0,
    )


def test_malformed_json_becomes_unknown():
    assert parse_classification("```json\n{}\n```") == ("unknown", 0.0)


def test_cache_hit_does_not_call_llm():
    instruction = "Where is my package?"
    client = MockLLMClient('{"intent":"cancel_order","confidence":0.1}')
    cache = MemoryCache(
        {
            cache_key(client.model, instruction): {
                "intent": "track_order",
                "confidence": 0.99,
            }
        }
    )

    result = classify_instruction(instruction, client, cache)

    assert result.intent == "track_order"
    assert result.confidence == 0.99
    assert result.cache_hit is True
    assert result.usage == {}
    assert client.calls == 0


def test_cache_miss_stores_normalized_response():
    instruction = "Please cancel order 123"
    client = MockLLMClient('{"intent":"cancel_order","confidence":0.9}')
    cache = MemoryCache()

    result = classify_instruction(instruction, client, cache)

    assert result.intent == "cancel_order"
    assert result.cache_hit is False
    assert cache.get(result.cache_key) == {"intent": "cancel_order", "confidence": 0.9}
    assert client.calls == 1


def test_v2_prompt_uses_five_train_examples_and_records_knn_signal():
    neighbors = [
        {"instruction": f"train ticket {index}", "intent": intent}
        for index, intent in enumerate(
            ["track_order", "delivery_period", "track_order", "delivery_period", "track_order"],
            1,
        )
    ]
    client = MockLLMClient('{"intent":"track_order","confidence":0.8}')

    result = classify_instruction(
        "Where is order 42?",
        client,
        MemoryCache(),
        prompt_version="v2",
        similar_tickets=neighbors,
    )

    assert result.knn_majority_intent == "track_order"
    assert result.knn_agreement == 0.6
    assert client.last_messages is not None
    assert "not the usual delivery duration" in client.last_messages[0]["content"]
    assert client.last_messages[1]["content"].count("Example ") == 5
    assert "Intent=delivery_period" in client.last_messages[1]["content"]
    assert "request or download a copy" in INTENT_DESCRIPTIONS_V2["check_invoice"]


def test_prompt_version_and_neighbors_are_part_of_cache_key():
    neighbors = [{"instruction": "Where is it?", "intent": "track_order"}]

    assert cache_key("model", "ticket", "v1") != cache_key(
        "model", "ticket", "v2", neighbors
    )
    assert cache_key("model", "ticket", "v2", neighbors) != cache_key(
        "model",
        "ticket",
        "v2",
        [{"instruction": "How long?", "intent": "delivery_period"}],
    )


def test_knn_summary_breaks_ties_by_nearest_neighbor():
    neighbors = [
        {"intent": "track_order"},
        {"intent": "delivery_period"},
        {"intent": "delivery_period"},
        {"intent": "track_order"},
        {"intent": "place_order"},
    ]

    assert knn_summary(neighbors) == ("track_order", 0.4)


def test_v1_prompt_remains_without_retrieved_examples():
    client = MockLLMClient('{"intent":"track_order","confidence":0.8}')
    result = classify_instruction(
        "Where is it?",
        client,
        MemoryCache(),
        prompt_version="v1",
        similar_tickets=[{"instruction": "train", "intent": "delivery_period"}],
    )

    assert result.knn_majority_intent is None
    assert "Example 1" not in client.last_messages[1]["content"]


def test_training_example_retrieval_is_deduplicated_and_requests_five():
    calls = []

    def retriever(query, *, k, db_url):
        calls.append((query, k, db_url))
        return [{"instruction": query, "intent": "track_order"}]

    result = retrieve_training_examples(
        ["same ticket", "same ticket", "other ticket"],
        database_url="mock://database",
        workers=2,
        retriever=retriever,
    )

    assert set(result) == {"same ticket", "other ticket"}
    assert sorted(calls) == [
        ("other ticket", 5, "mock://database"),
        ("same ticket", 5, "mock://database"),
    ]
