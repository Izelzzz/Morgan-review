import sqlite3
from importlib.resources import files
from pathlib import Path

_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"""


def connect(path: Path, *, create: bool = True) -> sqlite3.Connection:
    """Open the database; with `create=False`, fail instead of creating a missing file."""
    target = str(path) if create else f"{path.resolve().as_uri()}?mode=rw"
    # Autocommit mode: transactions are opened explicitly where they are needed.
    conn = sqlite3.connect(target, isolation_level=None, uri=not create)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _migrations() -> list[tuple[int, str]]:
    folder = files("review_agents") / "migrations"
    found = [
        (int(entry.name.split("_", 1)[0]), entry.read_text(encoding="utf-8"))
        for entry in folder.iterdir()
        if entry.name.endswith(".sql")
    ]
    return sorted(found)


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations in order and return the schema version.

    One service process owns the database, so migrations run once at startup.
    """
    conn.execute(_SCHEMA_VERSION_TABLE)
    applied = {row["version"] for row in conn.execute("SELECT version FROM schema_version")}
    for version, sql in _migrations():
        if version in applied:
            continue
        # The version row commits in the same transaction as the schema change.
        # `version` is an int parsed from a bundled migration filename, never external input.
        record = f"INSERT INTO schema_version (version) VALUES ({version});"  # noqa: S608
        script = f"BEGIN IMMEDIATE;\n{sql}\n{record}\nCOMMIT;"
        try:
            conn.executescript(script)
        except sqlite3.Error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    row = conn.execute("SELECT max(version) AS version FROM schema_version").fetchone()
    return int(row["version"] or 0)
