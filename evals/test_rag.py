"""Unit tests for cached reply drafting with pre-retrieved KB evidence."""

from pipeline.classify import MemoryCache
from pipeline.llm import LLMResponse
from pipeline.rag import (
    NOT_COVERED_REPLY,
    SYSTEM_PROMPTS,
    draft_reply,
    reply_cache_key,
)


class MockReplyClient:
    model = "mock/reply-model"

    def __init__(self):
        self.calls = 0
        self.last_messages = None

    def complete(self, messages):
        self.calls += 1
        self.last_messages = messages
        return LLMResponse(
            content="Track the shipment from your account [KB-01].",
            usage={"cost": 0.001},
        )


def test_draft_reply_reuses_cache_and_supplied_chunks():
    client = MockReplyClient()
    cache = MemoryCache()
    chunks = [
        {
            "chunk_id": 10,
            "article_id": 1,
            "kb_id": "KB-01",
            "article_title": "Track an order",
            "chunk": "Open the order page to track the shipment.",
            "score": 0.91,
        }
    ]

    first = draft_reply(
        "Where is my order?",
        "track_order",
        client=client,
        retrieved_chunks=chunks,
        cache=cache,
    )
    second = draft_reply(
        "Where is my order?",
        "track_order",
        client=client,
        retrieved_chunks=chunks,
        cache=cache,
    )

    assert first["cache_hit"] is False
    assert first["usage"] == {"cost": 0.001}
    assert first["citations"] == ["KB-01"]
    assert first["used_chunk_ids"] == [10]
    assert second["cache_hit"] is True
    assert second["usage"] == {}
    assert second["reply"] == first["reply"]
    assert client.calls == 1


def test_v2_prompt_is_strict_and_cache_is_versioned():
    chunks = [
        {
            "chunk_id": 10,
            "article_id": 1,
            "kb_id": "KB-01",
            "article_title": "Track an order",
            "chunk": "Track it from the order page.",
            "score": 0.91,
        }
    ]

    assert NOT_COVERED_REPLY in SYSTEM_PROMPTS["v2"]
    assert "Every sentence and every list item MUST end" in SYSTEM_PROMPTS["v2"]
    assert "Do not write greetings" in SYSTEM_PROMPTS["v2"]
    assert reply_cache_key("model", "ticket", "track_order", chunks, "v1") != (
        reply_cache_key("model", "ticket", "track_order", chunks, "v2")
    )


def test_v1_and_v2_reply_cache_entries_remain_separate():
    client = MockReplyClient()
    cache = MemoryCache()
    chunks = [
        {
            "chunk_id": 10,
            "article_id": 1,
            "kb_id": "KB-01",
            "article_title": "Track an order",
            "chunk": "Track it from the order page.",
            "score": 0.91,
        }
    ]

    draft_reply(
        "Where is it?",
        "track_order",
        client=client,
        retrieved_chunks=chunks,
        cache=cache,
        prompt_version="v1",
    )
    draft_reply(
        "Where is it?",
        "track_order",
        client=client,
        retrieved_chunks=chunks,
        cache=cache,
        prompt_version="v2",
    )

    assert client.calls == 2


def test_v2_empty_retrieval_returns_exact_fallback_without_llm_call():
    client = MockReplyClient()

    result = draft_reply(
        "A question absent from the help center",
        "unknown",
        client=client,
        retrieved_chunks=[],
        prompt_version="v2",
    )

    assert result["reply"] == NOT_COVERED_REPLY
    assert result["citations"] == []
    assert client.calls == 0
