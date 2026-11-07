"""Model access: OpenAI-compatible chat models, usage records and the token budget."""

import email.utils
import math
import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx2
import openai
from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult, LLMResult
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr

from review_agents.config import ModelEndpoint
from review_agents.db import connect

# Without streaming, a whole reply must arrive within this (decision D-029).
WHOLE_REPLY_SECONDS = 120.0
CONNECT_TIMEOUT_SECONDS = 10.0
# Waits between retries (decision D-030): from half a second, doubling, at most ten
# seconds, each ±10%; a service's Retry-After is followed up to thirty seconds.
FIRST_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 10.0
BACKOFF_JITTER = 0.1
MAX_RETRY_AFTER_SECONDS = 30.0
# Local model servers ignore the key, but the client refuses to start without one.
_NO_KEY = "not-needed"


class BudgetExceededError(RuntimeError):
    """The review has used its token budget; no further model call is made."""


class DeadlinePassedError(TimeoutError):
    """The call's deadline passed before its reply was complete."""


class AttemptTimedOutError(TimeoutError):
    """One attempt took longer than an attempt may (a whole reply's two minutes)."""


class EmptyReplyError(RuntimeError):
    """The model replied with nothing: no text and no tool call."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def _retry_after(headers: httpx2.Headers) -> float | None:
    """Seconds a service asks to wait, from `retry-after-ms` or `retry-after`."""
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is not None:
            try:
                return max(0.0, float(str(value)) * scale)
            except ValueError:
                pass
            try:  # an HTTP date, the other form Retry-After takes
                when = email.utils.parsedate_to_datetime(str(value))
            except (TypeError, ValueError):
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            return max(0.0, (when - datetime.now(UTC)).total_seconds())
    return None


def retry_wait(error: BaseException, retry: int, rng: random.Random) -> float | None:
    """How long to wait before retry number `retry` (from 0), or None not to retry.

    Retried: rate limits, server errors, a stall or a broken connection, and an empty
    reply with no finish reason. Not retried: request, permission and other client
    errors, and an empty reply cut by the output limit, which a resend would repeat.
    """
    if isinstance(error, AttemptTimedOutError):
        pass
    elif isinstance(error, EmptyReplyError):
        if not error.retryable:
            return None
    elif isinstance(error, openai.APIStatusError):
        if error.status_code != 429 and error.status_code < 500:
            return None
        asked = _retry_after(error.response.headers)
        if asked is not None:
            return min(asked, MAX_RETRY_AFTER_SECONDS)
    elif not isinstance(error, (openai.APIConnectionError, httpx2.TransportError)):
        # The SDK reports timeouts, stalls in mid-stream included, and broken connections
        # as APIConnectionError; a raw transport error escaping it is treated the same.
        return None
    backoff = min(MAX_BACKOFF_SECONDS, FIRST_BACKOFF_SECONDS * 2.0**retry)
    return backoff * rng.uniform(1 - BACKOFF_JITTER, 1 + BACKOFF_JITTER)


def _check_not_empty(result: ChatResult) -> None:
    for generation in result.generations:
        message = getattr(generation, "message", None)
        if isinstance(message, AIMessage) and not message.content and not message.tool_calls:
            # A streamed reply carries it in the generation, a whole one in the message.
            finish = (generation.generation_info or {}).get(
                "finish_reason"
            ) or message.response_metadata.get("finish_reason")
            if finish == "length":
                raise EmptyReplyError(
                    "the model replied with nothing before reaching its output limit",
                    retryable=False,
                )
            if finish is None:
                raise EmptyReplyError("the model replied with nothing", retryable=True)


class ReviewChatModel(ChatOpenAI):
    """A chat model that streams and retries by the review's own rules (decision D-030).

    The client's own retries are off; this one knows the call's deadline and never waits
    past it.
    """

    stream_replies: bool = True
    retries: int = 4
    deadline: float = math.inf
    # The longest one attempt may take; a whole reply's two minutes when not streaming.
    attempt_seconds: float = math.inf
    sleep_for: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    rng: random.Random = Field(default_factory=random.Random)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        retry = 0
        while True:
            try:
                result = self._within_deadline(
                    lambda cancelled: self._attempt(messages, stop, run_manager, cancelled, kwargs)
                )
                _check_not_empty(result)
                return result
            except DeadlinePassedError:
                raise
            except Exception as error:
                wait = retry_wait(error, retry, self.rng)
                if wait is None or retry >= self.retries or self.clock() + wait >= self.deadline:
                    raise
            retry += 1
            self.sleep_for(wait)

    def _attempt(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        run_manager: CallbackManagerForLLMRun | None,
        cancelled: threading.Event,
        kwargs: dict[str, Any],
    ) -> ChatResult:
        if not self.stream_replies:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        stream = self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

        def until_cancelled() -> Iterator[ChatGenerationChunk]:
            try:
                for piece in stream:
                    if cancelled.is_set():
                        return  # given up: stop reading, which closes the connection
                    yield piece
            finally:
                close = getattr(stream, "close", None)  # a generator: closing ends the request
                if close is not None:
                    close()

        result = generate_from_stream(until_cancelled())
        message = result.generations[0].message if result.generations else None
        # The usage record names the model; a whole reply carries it in llm_output.
        reported = message.response_metadata.get("model_name") if message else None
        result.llm_output = {"model_name": reported or self.model_name}
        return result

    def _within_deadline(self, attempt: Callable[[threading.Event], ChatResult]) -> ChatResult:
        """Run one attempt, giving it up at its own limit or the deadline, however busy.

        A reply that keeps coming is never stalled, so only this bounds it. A given-up
        attempt stops reading at its next byte, heartbeats included, or at its read
        timeout: the HTTP layer checks `cancel` on every piece it receives.
        """
        left = self.deadline - self.clock()
        if left <= 0:
            raise DeadlinePassedError("the call's deadline passed before it started")
        cancelled = threading.Event()
        limit = min(self.attempt_seconds, left)
        if math.isinf(limit):
            return _with_cancel(cancelled, attempt)
        outcome: dict[str, Any] = {}

        def run() -> None:
            try:
                outcome["result"] = _with_cancel(cancelled, attempt)
            except BaseException as error:  # handed to the caller below
                outcome["error"] = error

        worker = threading.Thread(target=run, name="model-call", daemon=True)
        worker.start()
        worker.join(limit)
        if worker.is_alive():
            cancelled.set()
            if limit < left:  # this attempt's own limit: worth another try
                raise AttemptTimedOutError(f"no complete reply within {limit:.0f} seconds")
            raise DeadlinePassedError(f"no complete reply within the {left:.0f} seconds left")
        if "error" in outcome:
            raise outcome["error"]
        result: ChatResult = outcome["result"]
        return result


# The cancel flag of the attempt running on this thread. The HTTP client's response hook
# runs on the thread that sends the request, so it finds the flag of its own attempt.
_attempt = threading.local()


def _with_cancel(
    cancelled: threading.Event, attempt: Callable[[threading.Event], ChatResult]
) -> ChatResult:
    _attempt.cancelled = cancelled
    try:
        return attempt(cancelled)
    finally:
        _attempt.cancelled = None


class _CancellableStream(httpx2.SyncByteStream):
    """A response body that ends as soon as its attempt is given up."""

    def __init__(self, inner: Any, cancelled: threading.Event) -> None:
        self._inner = inner
        self._cancelled = cancelled

    def __iter__(self) -> Iterator[bytes]:
        for piece in self._inner:
            if self._cancelled.is_set():
                raise httpx2.ReadError("the call was given up")
            yield piece

    def close(self) -> None:
        self._inner.close()


def _on_response(response: httpx2.Response) -> None:
    cancelled = getattr(_attempt, "cancelled", None)
    if cancelled is not None:
        response.stream = _CancellableStream(response.stream, cancelled)


# One client for every model call, as the SDK would share otherwise; proxies from the
# environment are kept, since only a response hook is added.
_HTTP_CLIENT = httpx2.Client(event_hooks={"response": [_on_response]})


def chat_model(
    endpoint: ModelEndpoint,
    *,
    time_left: float = math.inf,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ReviewChatModel:
    """A chat model for any OpenAI-compatible service: OpenRouter, Anthropic, Ollama, ...

    `time_left` is the seconds until the call's deadline: no wait, for data or before a
    retry, runs past it.
    """
    api_key = endpoint.api_key or SecretStr(_NO_KEY)
    # Streaming, the read timeout is the longest gap between two pieces of data.
    wait = min(endpoint.idle_seconds if endpoint.stream else WHOLE_REPLY_SECONDS, time_left)
    wait = max(wait, 0.001)
    return ReviewChatModel(
        base_url=str(endpoint.base_url),
        api_key=api_key,
        model=endpoint.model,
        http_client=_HTTP_CLIENT,
        attempt_seconds=math.inf if endpoint.stream else WHOLE_REPLY_SECONDS,
        timeout=httpx2.Timeout(wait, connect=min(CONNECT_TIMEOUT_SECONDS, wait)),
        max_retries=0,
        stream_usage=True,
        stream_replies=endpoint.stream,
        retries=endpoint.max_retries,
        deadline=clock() + time_left,
        sleep_for=sleep,
        clock=clock,
    )


# Cached input costs about a tenth of fresh input at most services (decision D-009).
CACHED_INPUT_WEIGHT = 0.1


def charged_tokens(input_tokens: int, cached_input_tokens: int, output_tokens: int) -> int:
    """What one call costs the budget: cached input at a tenth, the rest in full."""
    fresh = input_tokens - cached_input_tokens
    return fresh + math.ceil(cached_input_tokens * CACHED_INPUT_WEIGHT) + output_tokens


class TokenBudget:
    """Tokens one review may use, as `charged_tokens` counts them. Shared by its agents."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def charge(self, tokens: int) -> None:
        with self._lock:
            self._used += tokens


RecordUsage = Callable[[str, int, int, int, int], None]
"""Stores one call's usage: model, input, cached input and output tokens, duration in ms."""


def review_usage(db_path: Path, review_id: int, agent: str) -> RecordUsage:
    """Records into `usage`, the per-review table."""

    def record(model: str, input_tokens: int, cached: int, output_tokens: int, ms: int) -> None:
        # Callbacks run on the model call's thread, so each record opens its own connection.
        with closing(connect(db_path, create=False)) as conn:
            conn.execute(
                "INSERT INTO usage (review_id, agent, model, input_tokens, cached_input_tokens,"
                " output_tokens, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (review_id, agent, model, input_tokens, cached, output_tokens, ms),
            )

    return record


def summary_usage(db_path: Path, repo: str) -> RecordUsage:
    """Records into `summary_usage`, for module summaries that belong to no single review."""

    def record(model: str, input_tokens: int, cached: int, output_tokens: int, ms: int) -> None:
        with closing(connect(db_path, create=False)) as conn:
            conn.execute(
                "INSERT INTO summary_usage (repo, model, input_tokens, cached_input_tokens,"
                " output_tokens, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
                (repo, model, input_tokens, cached, output_tokens, ms),
            )

    return record


class UsageMeter(BaseCallbackHandler):
    """Refuses model calls once the budget is spent and records every call's usage.

    The budget is checked before each call, so a review overshoots it by at most the
    calls already in flight when it ran out.
    """

    # Exceptions raised here must reach the caller, not be logged and swallowed.
    raise_error = True

    def __init__(
        self,
        budget: TokenBudget,
        record: RecordUsage,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._budget = budget
        self._record = record
        self._clock = clock
        self._started: dict[UUID, float] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if self._budget.exhausted:
            raise BudgetExceededError(
                f"token budget of {self._budget.limit} used up ({self._budget.used} tokens)"
            )
        self._started[run_id] = self._clock()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        started = self._started.pop(run_id, self._clock())
        duration_ms = int((self._clock() - started) * 1000)
        input_tokens = cached = output_tokens = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    input_tokens += usage["input_tokens"]
                    output_tokens += usage["output_tokens"]
                    details = usage.get("input_token_details") or {}
                    cached += min(details.get("cache_read") or 0, usage["input_tokens"])
        model = (response.llm_output or {}).get("model_name", "")
        # Charged before the record is written: if writing fails, the budget still holds.
        self._budget.charge(charged_tokens(input_tokens, cached, output_tokens))
        self._record(model, input_tokens, cached, output_tokens, duration_ms)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._started.pop(run_id, None)
