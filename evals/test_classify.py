"""Unit tests for LLM classification parsing, validation, and caching."""

from pipeline.classify import MemoryCache, cache_key, classify_instruction, parse_classification
from pipeline.llm import LLMResponse


class MockLLMClient:
    model = "mock/model"

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
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
