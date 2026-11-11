import time

import pytest

from review_agents.context import tokens
from review_agents.context.tokens import estimate_tokens


def test_tokens_are_counted_with_a_real_tokenizer() -> None:
    assert estimate_tokens("hello world") == 2
    assert estimate_tokens("") == 0
    # Special-token text in a PR is counted as plain text, not refused.
    assert estimate_tokens("<|endoftext|>") > 1


def test_without_the_encoding_the_count_falls_back_to_an_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> None:
        raise OSError("no network and no cached encoding")

    monkeypatch.setattr(tokens, "_load_encoding", unavailable)
    tokens._encoding.cache_clear()

    try:
        assert estimate_tokens("abcdefgh") == 2
        assert estimate_tokens("你好") == 2
    finally:
        tokens._encoding.cache_clear()


def test_a_failed_load_is_retried_later_not_given_up_for_good(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    clock = [1000.0]

    def flaky() -> object:
        calls.append(clock[0])
        raise OSError("network blip")

    monkeypatch.setattr(tokens, "_load_encoding", flaky)
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    tokens._encoding.cache_clear()
    try:
        estimate_tokens("a")
        estimate_tokens("b")  # within the retry interval: no second attempt
        clock[0] += tokens.RETRY_SECONDS
        estimate_tokens("c")

        assert len(calls) == 2
    finally:
        tokens._encoding.cache_clear()
