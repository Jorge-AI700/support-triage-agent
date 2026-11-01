"""Unit tests for the OpenRouter wrapper without making network calls."""

from types import SimpleNamespace

from pipeline.llm import OpenRouterClient


class StatusError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class FakeCompletions:
    def __init__(self):
        self.calls = []
        self.attempts = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        self.attempts += 1
        if self.attempts == 1:
            raise StatusError(429)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage={"total_tokens": 7, "cost": 0.0002},
        )


def test_openrouter_wrapper_retries_and_returns_usage_cost():
    completions = FakeCompletions()
    sdk_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    sleeps = []
    client = OpenRouterClient(
        model="mock/model",
        client=sdk_client,
        max_retries=2,
        backoff_seconds=0.25,
        sleep=sleeps.append,
    )

    result = client.complete([{"role": "user", "content": "hello"}])

    assert result.content == "ok"
    assert result.usage == {"total_tokens": 7, "cost": 0.0002}
    assert sleeps == [0.25]
    assert len(completions.calls) == 2
    assert completions.calls[-1]["extra_body"] == {
        "reasoning": {"enabled": False},
        "usage": {"include": True},
    }
