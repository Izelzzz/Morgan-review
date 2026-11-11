import shutil
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient

from review_agents import __version__
from review_agents.api import create_app
from review_agents.config import Settings
from review_agents.db import connect


def make_client(data_dir: Path) -> TestClient:
    return TestClient(create_app(Settings(data_dir=data_dir, worker_poll_seconds=0.05)))


def test_healthz_reports_ok_and_creates_the_database(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with make_client(data_dir) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
    assert (data_dir / "review-agents.sqlite3").is_file()


def test_healthz_reports_503_when_the_database_is_unusable(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with make_client(data_dir) as client:
        for leftover in data_dir.iterdir():
            leftover.unlink()
        (data_dir / "review-agents.sqlite3").mkdir()
        response = client.get("/healthz")
        shutil.rmtree(data_dir / "review-agents.sqlite3")

    assert response.status_code == 503
    assert response.json()["status"] == "database unavailable"


def test_healthz_reports_503_when_the_database_file_is_gone(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with make_client(data_dir) as client:
        for leftover in data_dir.iterdir():
            leftover.unlink()
        response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json()["status"] == "database unavailable"


class WorkerKilled(BaseException):
    pass


def test_healthz_reports_503_when_the_worker_has_stopped(tmp_path: Path) -> None:
    def kills_the_worker(conn: sqlite3.Connection, review_id: int) -> None:
        raise WorkerKilled

    data_dir = tmp_path / "data"
    app = create_app(Settings(data_dir=data_dir, worker_poll_seconds=0.01), kills_the_worker)
    with TestClient(app) as client:
        with closing(connect(data_dir / "review-agents.sqlite3")) as conn:
            conn.execute(
                "INSERT INTO reviews (repo, pr, base_sha, head_sha, mode)"
                " VALUES ('o/r', 1, 'a', 'b', 'quick')"
            )
        for _ in range(200):
            response = client.get("/healthz")
            if response.status_code == 503:
                break
            time.sleep(0.01)

    assert response.status_code == 503
    assert response.json()["status"] == "worker stopped"
