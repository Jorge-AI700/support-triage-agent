"""Unit tests for cached reply drafting with pre-retrieved KB evidence."""

from pipeline.classify import MemoryCache
from pipeline.llm import LLMResponse
from pipeline.rag import draft_reply


class MockReplyClient:
    model = "mock/reply-model"

    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
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
