import json
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import openai
import pytest
from pydantic import SecretStr

from review_agents import models
from review_agents.config import ModelEndpoint
from review_agents.db import connect, migrate
from review_agents.models import (
    BudgetExceededError,
    DeadlinePassedError,
    EmptyReplyError,
    TokenBudget,
    UsageMeter,
    chat_model,
    review_usage,
)


@dataclass
class FakeService:
    """A minimal OpenAI-compatible /chat/completions endpoint."""

    url: str = ""
    # One entry per request: (status, delay seconds); the last entry repeats.
    script: list[tuple[int, float]] = field(default_factory=lambda: [(200, 0.0)])
    requests: list[dict[str, str]] = field(default_factory=list)
    # Sent with error responses; the default keeps retries in tests fast.
    retry_headers: dict[str, str] = field(default_factory=lambda: {"retry-after-ms": "1"})
    # Input tokens the service reports as read from its prompt cache.
    cached: int = 0
    # How each streamed reply is sent, one entry per request; the last entry repeats.
    plans: list["Plan"] = field(default_factory=lambda: [Plan()])
    bodies: list[dict[str, object]] = field(default_factory=list)
    # Seconds between the pieces of a whole (unstreamed) reply, for the first request.
    trickle: float = 0.0
    # Replies the client hung up on before they were sent in full.
    hangups: int = 0


@dataclass
class Plan:
    """One streamed reply: silence, heartbeats, then content in chunks."""

    silence: float = 0.0  # seconds with no byte at all after the headers
    heartbeats: int = 0  # SSE comments sent before the content, `gap` apart
    chunks: int = 1
    gap: float = 0.0  # seconds between heartbeats and between chunks
    content: str = "no findings"
    finish: str | None = "stop"


def sse(data: dict[str, object]) -> bytes:
    return b"data: " + json.dumps(data).encode() + b"\n\n"


def chunk(model: str, delta: dict[str, object], finish: str | None = None) -> dict[str, object]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def streamed(handler: BaseHTTPRequestHandler, model: str, plan: Plan, cached: int) -> None:
    write = handler.wfile
    time.sleep(plan.silence)
    for _ in range(plan.heartbeats):
        write.write(b": keep-alive\n\n")
        write.flush()
        time.sleep(plan.gap)
    size = -(-len(plan.content) // plan.chunks) if plan.content else 1
    pieces = [plan.content[i : i + size] for i in range(0, len(plan.content), size)]
    write.write(sse(chunk(model, {"role": "assistant", "content": ""})))
    for piece in pieces:
        write.write(sse(chunk(model, {"content": piece})))
        write.flush()
        time.sleep(plan.gap)
    write.write(sse(chunk(model, {}, plan.finish)))
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    write.write(sse({**chunk(model, {}), "choices": [], "usage": usage}))
    write.write(b"data: [DONE]\n\n")
    write.flush()


def completion(model: str, cached: int = 0) -> bytes:
    return json.dumps(
        {
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "no findings"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }
    ).encode()


@pytest.fixture
def service() -> Iterator[FakeService]:
    fake = FakeService()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.requests.append({"path": self.path, "auth": self.headers.get("Authorization", "")})
            fake.bodies.append(body)
            n = len(fake.requests)
            status, delay = fake.script[min(n, len(fake.script)) - 1]
            time.sleep(delay)
            self.send_response(status)
            stream = status == 200 and bool(body.get("stream"))
            self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
            if status != 200:
                for name, value in fake.retry_headers.items():
                    self.send_header(name, value)
            self.end_headers()
            # The client may give up mid-reply after its timeout.
            try:
                if stream:
                    plan = fake.plans[min(n, len(fake.plans)) - 1]
                    streamed(self, body["model"], plan, fake.cached)
                elif status == 200 and fake.trickle and n == 1:
                    for byte in completion(body["model"], fake.cached):
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(fake.trickle)
                else:
                    self.wfile.write(
                        completion(body["model"], fake.cached)
                        if status == 200
                        else b'{"error": {}}'
                    )
            except (BrokenPipeError, ConnectionResetError):
                fake.hangups += 1

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fake.url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield fake
    server.shutdown()


@pytest.fixture
def db(tmp_path: Path) -> tuple[Path, int]:
    path = tmp_path / "db.sqlite3"
    with closing(connect(path)) as conn:
        migrate(conn)
        review_id = conn.execute(
            "INSERT INTO reviews (repo, pr, base_sha, head_sha, mode)"
            " VALUES ('o/r', 1, 'a', 'b', 'standard') RETURNING id"
        ).fetchone()["id"]
    return path, int(review_id)


def endpoint(service: FakeService, key: str | None = "sk-test", **extra: object) -> ModelEndpoint:
    return ModelEndpoint.model_validate(
        {
            "base_url": service.url,
            "model": "finder-model",
            "api_key": key and SecretStr(key),
            **extra,
        }
    )


class Sleeps:
    """Records the waits between retries instead of sleeping them."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def usage_rows(db_path: Path) -> list[tuple[str, str, int, int, int]]:
    with closing(connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT agent, model, input_tokens, cached_input_tokens, output_tokens FROM usage"
            " ORDER BY id"
        ).fetchall()
    return [tuple(row) for row in rows]


@pytest.mark.parametrize(
    ("key", "auth"), [("sk-test", "Bearer sk-test"), (None, "Bearer not-needed")]
)
def test_a_call_reaches_the_service_and_records_its_usage(
    service: FakeService, db: tuple[Path, int], key: str | None, auth: str
) -> None:
    budget = TokenBudget(500_000)
    meter = UsageMeter(budget, review_usage(db[0], db[1], "security"))

    reply = chat_model(endpoint(service, key)).invoke("review", config={"callbacks": [meter]})

    assert reply.content == "no findings"
    assert service.requests == [{"path": "/v1/chat/completions", "auth": auth}]
    assert usage_rows(db[0]) == [("security", "finder-model", 11, 0, 7)]
    assert budget.used == 18


def test_cached_input_counts_a_tenth_against_the_budget(
    service: FakeService, db: tuple[Path, int]
) -> None:
    service.cached = 10
    budget = TokenBudget(500_000)
    meter = UsageMeter(budget, review_usage(db[0], db[1], "security"))

    chat_model(endpoint(service)).invoke("review", config={"callbacks": [meter]})

    # 1 uncached input + 10 cached at a tenth + 7 output.
    assert budget.used == 9
    assert usage_rows(db[0]) == [("security", "finder-model", 11, 10, 7)]


def test_no_call_is_made_once_the_budget_is_used_up(
    service: FakeService, db: tuple[Path, int]
) -> None:
    budget = TokenBudget(20)
    meter = UsageMeter(budget, review_usage(db[0], db[1], "security"))
    model = chat_model(endpoint(service))

    model.invoke("first", config={"callbacks": [meter]})
    model.invoke("second", config={"callbacks": [meter]})  # 18 < 20: still allowed
    with pytest.raises(BudgetExceededError, match="token budget of 20"):
        model.invoke("third", config={"callbacks": [meter]})

    assert len(service.requests) == 2
    assert budget.used == 36


def test_agents_of_one_review_share_its_budget(service: FakeService, db: tuple[Path, int]) -> None:
    budget = TokenBudget(18)
    model = chat_model(endpoint(service))

    model.invoke(
        "a", config={"callbacks": [UsageMeter(budget, review_usage(db[0], db[1], "security"))]}
    )
    with pytest.raises(BudgetExceededError):
        model.invoke(
            "b",
            config={"callbacks": [UsageMeter(budget, review_usage(db[0], db[1], "edge_cases"))]},
        )

    assert [row[0] for row in usage_rows(db[0])] == ["security"]


def test_calls_stream_by_default_and_still_record_their_usage(
    service: FakeService, db: tuple[Path, int]
) -> None:
    budget = TokenBudget(500_000)
    meter = UsageMeter(budget, review_usage(db[0], db[1], "security"))
    service.plans = [Plan(chunks=3)]

    reply = chat_model(endpoint(service)).invoke("review", config={"callbacks": [meter]})

    assert reply.content == "no findings"
    assert service.bodies[0]["stream"] is True
    assert usage_rows(db[0]) == [("security", "finder-model", 11, 0, 7)]


def test_a_model_with_streaming_off_waits_for_the_whole_reply(service: FakeService) -> None:
    reply = chat_model(endpoint(service, stream=False)).invoke("review")

    assert reply.content == "no findings"
    assert not service.bodies[0].get("stream")


def test_a_stream_silent_for_the_idle_time_is_stalled_and_retried(service: FakeService) -> None:
    service.plans = [Plan(silence=1.0), Plan()]
    sleeps = Sleeps()

    reply = chat_model(endpoint(service, idle_seconds=0.3), sleep=sleeps).invoke("review")

    assert reply.content == "no findings"
    assert len(service.requests) == 2
    assert len(sleeps.waits) == 1


def test_a_stream_that_stops_mid_reply_is_stalled_and_retried(service: FakeService) -> None:
    # Two chunks, then a second of silence before the next: a stall in mid-stream.
    service.plans = [Plan(chunks=3, gap=1.0), Plan()]

    reply = chat_model(endpoint(service, idle_seconds=0.3), sleep=Sleeps()).invoke("review")

    assert reply.content == "no findings"
    assert len(service.requests) == 2


def test_a_reply_that_keeps_coming_is_never_cut_however_long(service: FakeService) -> None:
    # 1.2 seconds in all, but never 0.3 seconds without a byte.
    service.plans = [Plan(chunks=6, gap=0.2)]

    reply = chat_model(endpoint(service, idle_seconds=0.5)).invoke("review")

    assert reply.content == "no findings"
    assert len(service.requests) == 1


def test_a_reply_still_coming_at_the_deadline_is_given_up(service: FakeService) -> None:
    # Data every 0.2 seconds for 2 seconds: never stalled, but the call has 0.5 seconds.
    service.plans = [Plan(chunks=10, gap=0.2)]

    started = time.monotonic()
    with pytest.raises(DeadlinePassedError):
        chat_model(endpoint(service), time_left=0.5, sleep=Sleeps()).invoke("review")

    assert time.monotonic() - started < 1.0
    assert len(service.requests) == 1  # past the deadline, nothing is retried


def test_a_reply_trickling_in_without_streaming_is_cut_at_its_limit_and_retried(
    service: FakeService, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Never 0.5 seconds without a byte, but far longer than the whole-reply limit.
    monkeypatch.setattr(models, "WHOLE_REPLY_SECONDS", 0.5)
    service.trickle = 0.05

    reply = chat_model(endpoint(service, stream=False), sleep=Sleeps()).invoke("review")

    assert reply.content == "no findings"
    assert len(service.requests) == 2


def test_a_call_given_up_at_its_deadline_hangs_up_even_on_heartbeats(
    service: FakeService,
) -> None:
    # Heartbeats for 4 seconds keep the connection busy; the call has 0.5 seconds.
    service.plans = [Plan(heartbeats=40, gap=0.1)]
    before = set(threading.enumerate())

    with pytest.raises(DeadlinePassedError):
        chat_model(endpoint(service), time_left=0.5, sleep=Sleeps()).invoke("review")

    def calls_running() -> list[threading.Thread]:
        new = set(threading.enumerate()) - before
        return [t for t in new if t.name == "model-call" and t.is_alive()]

    hung_up = time.monotonic() + 1.5
    while (service.hangups == 0 or calls_running()) and time.monotonic() < hung_up:
        time.sleep(0.05)
    assert service.hangups == 1  # the request ended, not just the caller's wait
    assert calls_running() == []


def test_a_retry_after_date_is_followed(service: FakeService) -> None:
    when = datetime.now(UTC) + timedelta(seconds=3)
    service.script = [(429, 0.0), (200, 0.0)]
    service.retry_headers = {"Retry-After": format_datetime(when, usegmt=True)}
    sleeps = Sleeps()

    chat_model(endpoint(service), sleep=sleeps).invoke("review")

    [wait] = sleeps.waits
    assert 1.5 <= wait <= 3.5  # whole seconds in the header, and the request took a moment


def test_heartbeats_keep_a_thinking_model_alive(service: FakeService) -> None:
    service.plans = [Plan(heartbeats=6, gap=0.2)]

    chat_model(endpoint(service, idle_seconds=0.5)).invoke("review")

    assert len(service.requests) == 1


@pytest.mark.parametrize("status", [429, 500, 503])
def test_rate_limits_and_server_errors_are_retried(service: FakeService, status: int) -> None:
    service.script = [(status, 0.0), (200, 0.0)]

    chat_model(endpoint(service), sleep=Sleeps()).invoke("review")

    assert len(service.requests) == 2


@pytest.mark.parametrize("status", [400, 401, 403])
def test_request_and_permission_errors_are_not_retried(
    service: FakeService, db: tuple[Path, int], status: int
) -> None:
    service.script = [(status, 0.0)]
    meter = UsageMeter(TokenBudget(500_000), review_usage(db[0], db[1], "security"))

    with pytest.raises(openai.APIStatusError):
        chat_model(endpoint(service)).invoke("review", config={"callbacks": [meter]})

    assert len(service.requests) == 1
    assert usage_rows(db[0]) == []


def test_four_retries_back_off_from_half_a_second_and_then_fail(service: FakeService) -> None:
    service.script = [(500, 0.0)]
    service.retry_headers = {}
    sleeps = Sleeps()

    with pytest.raises(openai.InternalServerError):
        chat_model(endpoint(service), sleep=sleeps).invoke("review")

    assert len(service.requests) == 5
    for wait, base in zip(sleeps.waits, [0.5, 1.0, 2.0, 4.0], strict=True):
        assert base * 0.9 <= wait <= base * 1.1


def test_backoff_stops_growing_at_ten_seconds(service: FakeService) -> None:
    service.script = [(500, 0.0)]
    service.retry_headers = {}
    sleeps = Sleeps()

    with pytest.raises(openai.InternalServerError):
        chat_model(endpoint(service, max_retries=6), sleep=sleeps).invoke("review")

    assert 9.0 <= sleeps.waits[-1] <= 11.0  # 0.5 * 2**5 = 16, capped at 10


@pytest.mark.parametrize(("header", "wait"), [("2", 2.0), ("90", 30.0)])
def test_retry_after_sets_the_wait_up_to_thirty_seconds(
    service: FakeService, header: str, wait: float
) -> None:
    service.script = [(429, 0.0), (200, 0.0)]
    service.retry_headers = {"Retry-After": header}
    sleeps = Sleeps()

    chat_model(endpoint(service), sleep=sleeps).invoke("review")

    assert sleeps.waits == [wait]


def test_no_retry_waits_past_the_deadline(service: FakeService) -> None:
    service.script = [(500, 0.0)]
    service.retry_headers = {}
    sleeps = Sleeps()

    # 0.4 seconds left: the first backoff of about 0.5 seconds would end past it.
    with pytest.raises(openai.InternalServerError):
        chat_model(endpoint(service), time_left=0.4, sleep=sleeps).invoke("review")

    assert len(service.requests) == 1
    assert sleeps.waits == []


def test_an_empty_reply_without_a_reason_is_retried(service: FakeService) -> None:
    service.plans = [Plan(content="", finish=None), Plan()]

    reply = chat_model(endpoint(service), sleep=Sleeps()).invoke("review")

    assert reply.content == "no findings"
    assert len(service.requests) == 2


def test_an_empty_reply_cut_by_the_output_limit_is_not_retried(service: FakeService) -> None:
    service.plans = [Plan(content="", finish="length")]

    with pytest.raises(EmptyReplyError, match="output limit"):
        chat_model(endpoint(service), sleep=Sleeps()).invoke("review")

    assert len(service.requests) == 1
