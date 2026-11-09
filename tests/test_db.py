import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from review_agents.db import _migrations, connect, migrate

LATEST = max(version for version, _ in _migrations())

TABLES = {
    "reviews",
    "findings",
    "comment_links",
    "memories",
    "feedback_cases",
    "usage",
    "steps",
    "context_cache",
    "summary_usage",
}


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row["name"] for row in rows}


def test_migrate_creates_every_table_on_an_empty_database(tmp_path: Path) -> None:
    with closing(connect(tmp_path / "db.sqlite3")) as conn:
        assert migrate(conn) == LATEST
        assert table_names(conn) >= TABLES


def test_connect_without_create_fails_on_a_missing_file_and_creates_nothing(
    tmp_path: Path,
) -> None:
    with pytest.raises(sqlite3.OperationalError):
        connect(tmp_path / "missing.sqlite3", create=False)

    assert list(tmp_path.iterdir()) == []


def test_connect_without_create_opens_an_existing_database(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    with closing(connect(path)) as conn:
        migrate(conn)
    with closing(connect(path, create=False)) as conn:
        assert table_names(conn) >= TABLES


def test_migrate_is_idempotent_across_restarts(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    with closing(connect(path)) as conn:
        migrate(conn)
        conn.execute(
            "INSERT INTO reviews (repo, pr, base_sha, head_sha, mode)"
            " VALUES ('o/r', 1, 'a', 'b', 'quick')"
        )
    with closing(connect(path)) as conn:
        assert migrate(conn) == LATEST
        assert conn.execute("SELECT count(*) AS n FROM reviews").fetchone()["n"] == 1
        assert conn.execute("SELECT count(*) AS n FROM schema_version").fetchone()["n"] == LATEST


@pytest.mark.parametrize(
    ("column", "value"),
    [("mode", "fast"), ("status", "paused"), ("pr", 0)],
)
def test_reviews_reject_values_outside_their_domain(
    tmp_path: Path, column: str, value: object
) -> None:
    row: dict[str, object] = {
        "repo": "o/r",
        "pr": 1,
        "base_sha": "a",
        "head_sha": "b",
        "mode": "quick",
        "status": "queued",
    }
    row[column] = value
    with closing(connect(tmp_path / "db.sqlite3")) as conn:
        migrate(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO reviews (repo, pr, base_sha, head_sha, mode, status)"
                " VALUES (:repo, :pr, :base_sha, :head_sha, :mode, :status)",
                row,
            )


def test_findings_reject_unknown_severity_and_orphans(tmp_path: Path) -> None:
    with closing(connect(tmp_path / "db.sqlite3")) as conn:
        migrate(conn)
        review_id = conn.execute(
            "INSERT INTO reviews (repo, pr, base_sha, head_sha, mode)"
            " VALUES ('o/r', 1, 'a', 'b', 'quick') RETURNING id"
        ).fetchone()["id"]
        insert = (
            "INSERT INTO findings (review_id, fingerprint, primary_dimension, severity, file,"
            " line_start, line_end, title, body) VALUES (?, 'f', 'security', ?, 'a.py',"
            " 1, 1, 't', 'b')"
        )
        conn.execute(insert, (review_id, "blocking"))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(insert, (review_id, "critical"))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(insert, (review_id + 1, "blocking"))
