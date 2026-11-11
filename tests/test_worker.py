import asyncio
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from review_agents.db import connect, migrate
from review_agents.worker import ReviewHandler, Worker, not_implemented, run_once


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "db.sqlite3"
    with closing(connect(path)) as conn:
        migrate(conn)
    return path


def enqueue(db_path: Path, status: str = "queued") -> int:
    with closing(connect(db_path)) as conn:
        row = conn.execute(
            "INSERT INTO reviews (repo, pr, base_sha, head_sha, mode, status)"
            " VALUES ('o/r', 1, 'a', 'b', 'standard', ?) RETURNING id",
            (status,),
        ).fetchone()
        return int(row["id"])


def review(db_path: Path, review_id: int) -> sqlite3.Row:
    with closing(connect(db_path)) as conn:
        row: sqlite3.Row = conn.execute(
            "SELECT status, error, started_at, finished_at FROM reviews WHERE id = ?", (review_id,)
        ).fetchone()
        return row


def test_run_once_processes_the_oldest_queued_review(db_path: Path) -> None:
    first, second = enqueue(db_path), enqueue(db_path)
    seen: list[int] = []

    assert run_once(db_path, lambda _, review_id: seen.append(review_id)) == first
    assert seen == [first]
    assert review(db_path, first)["status"] == "done"
    assert review(db_path, second)["status"] == "queued"


def test_run_once_returns_none_on_an_empty_queue(db_path: Path) -> None:
    assert run_once(db_path, not_implemented) is None


def test_a_failing_handler_marks_the_review_failed_with_the_reason(db_path: Path) -> None:
    review_id = enqueue(db_path)

    run_once(db_path, not_implemented)

    row = review(db_path, review_id)
    assert row["status"] == "failed"
    assert "not implemented" in row["error"]
    assert row["finished_at"] is not None


def test_a_handler_failing_mid_transaction_still_marks_the_review_failed(db_path: Path) -> None:
    review_id = enqueue(db_path)

    def fails_mid_transaction(conn: sqlite3.Connection, review_id: int) -> None:
        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO usage (review_id, agent, model, input_tokens, output_tokens, duration_ms)"
            " VALUES (?, 'security', 'm', 1, 1, 1)",
            (review_id,),
        )
        raise RuntimeError("model call failed")

    run_once(db_path, fails_mid_transaction)

    row = review(db_path, review_id)
    assert row["status"] == "failed"
    assert "model call failed" in row["error"]
    with closing(connect(db_path)) as conn:
        assert conn.execute("SELECT count(*) AS n FROM usage").fetchone()["n"] == 0


def test_a_handler_leaving_a_transaction_open_is_a_failure(db_path: Path) -> None:
    review_id = enqueue(db_path)

    def leaves_transaction_open(conn: sqlite3.Connection, review_id: int) -> None:
        conn.execute("BEGIN")

    run_once(db_path, leaves_transaction_open)

    row = review(db_path, review_id)
    assert row["status"] == "failed"
    assert "transaction open" in row["error"]


def test_worker_keeps_draining_the_queue_after_a_failed_iteration(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queued = enqueue(db_path)
    calls = 0
    real_run_once = run_once

    def flaky_run_once(path: Path, handler: ReviewHandler) -> int | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_run_once(path, handler)

    monkeypatch.setattr("review_agents.worker.run_once", flaky_run_once)

    async def run_worker() -> bool:
        worker = Worker(db_path, lambda conn, review_id: None, poll_seconds=0.01)
        worker.start()
        for _ in range(200):
            if review(db_path, queued)["status"] == "done":
                break
            await asyncio.sleep(0.01)
        alive = worker.alive
        await worker.stop()
        return alive

    assert asyncio.run(run_worker())
    assert review(db_path, queued)["status"] == "done"


def test_worker_recovers_interrupted_reviews_and_drains_the_queue(db_path: Path) -> None:
    # Reviews without a token: one on its first attempt is queued once more (D-021),
    # one already tried twice fails.
    retried = enqueue(db_path, status="running")
    interrupted = enqueue(db_path, status="running")
    queued = enqueue(db_path)
    with closing(connect(db_path)) as conn:
        conn.execute("UPDATE reviews SET attempts = 1 WHERE id = ?", (retried,))
        conn.execute("UPDATE reviews SET attempts = 2 WHERE id = ?", (interrupted,))

    async def run_worker_until_queue_drains() -> None:
        worker = Worker(db_path, lambda conn, review_id: None, poll_seconds=0.01)
        worker.start()
        for _ in range(200):
            if all(review(db_path, r)["status"] == "done" for r in (queued, retried)):
                break
            await asyncio.sleep(0.01)
        await worker.stop()

    asyncio.run(run_worker_until_queue_drains())

    assert review(db_path, queued)["status"] == "done"
    assert review(db_path, retried)["status"] == "done"
    assert review(db_path, interrupted)["status"] == "failed"
    assert "restart" in review(db_path, interrupted)["error"]


def test_a_repeating_failure_is_logged_once_at_error_level(
    db_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = 0

    def always_locked(path: Path, handler: ReviewHandler) -> int | None:
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("review_agents.worker.run_once", always_locked)

    async def run_a_few_iterations() -> None:
        worker = Worker(db_path, not_implemented, poll_seconds=0.001)
        worker.start()
        while calls < 5:
            await asyncio.sleep(0.005)
        await worker.stop()

    with caplog.at_level(logging.DEBUG, logger="review_agents.worker"):
        asyncio.run(run_a_few_iterations())

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert calls >= 5
