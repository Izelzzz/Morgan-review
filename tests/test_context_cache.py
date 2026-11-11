import threading
import time
from contextlib import closing
from pathlib import Path

import pytest
from gitrepo import Upstream

from review_agents.context.cache import prune, repo_layer_for_review
from review_agents.context.modules import SUMMARY_VERSION
from review_agents.db import connect, migrate
from review_agents.mirror import MirrorStore


class Setup:
    def __init__(self, tmp_path: Path) -> None:
        self.upstream = Upstream(tmp_path, "octo/shop")
        self.upstream.commit({"README.md": "# Shop\n", "payments/refund.py": "x = 1\n"}, "init")
        self.store = MirrorStore(tmp_path / "repos", remote_base=str(self.upstream.base))
        self.db_path = tmp_path / "db.sqlite3"
        with closing(connect(self.db_path)) as conn:
            migrate(conn)
        self.sync()

    def sync(self) -> Path:
        return self.store.sync("octo/shop", ["refs/heads/main"], [], token=None)

    def review(self) -> int:
        with closing(connect(self.db_path)) as conn:
            row = conn.execute(
                "INSERT INTO reviews (repo, pr, base_sha, head_sha, mode)"
                " VALUES ('octo/shop', 1, 'a', 'b', 'standard') RETURNING id"
            ).fetchone()
        return int(row["id"])

    def build(self, review_id: int, summarize=None):  # type: ignore[no-untyped-def]
        with closing(connect(self.db_path)) as conn:
            return repo_layer_for_review(
                conn,
                review_id,
                "octo/shop",
                self.store.git_dir("octo/shop"),
                "main",
                [],
                summarize=summarize,
            )

    def steps(self, review_id: int) -> list[str]:
        with closing(connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT node, status FROM steps WHERE review_id = ? ORDER BY id", (review_id,)
            ).fetchall()
        return [f"{row['node']}:{row['status']}" for row in rows]


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


def test_a_second_review_of_the_repository_records_the_cache_hit(setup: Setup) -> None:
    first, second = setup.review(), setup.review()

    setup.build(first)
    setup.build(second)

    assert setup.steps(first) == ["context.repo_layer:ok"]
    assert setup.steps(second) == ["context.repo_layer.cache_hit:ok"]


def test_concurrent_reviews_of_one_repository_build_its_layer_once(setup: Setup) -> None:
    calls: list[str] = []
    lock = threading.Lock()

    def slow_summary(path: str, source: str) -> str:
        with lock:
            calls.append(path)
        time.sleep(0.2)
        return "summary"

    reviews = [setup.review() for _ in range(3)]
    threads = [threading.Thread(target=setup.build, args=(r, slow_summary)) for r in reviews]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(calls) == [".", "payments"]  # each module summarised once, not per review
    hits = [step for r in reviews for step in setup.steps(r)]
    assert sorted(hits).count("context.repo_layer:ok") == 1


def test_pruning_keeps_the_latest_layers_and_every_current_format_summary(setup: Setup) -> None:
    for i in range(5):
        setup.upstream.commit({"README.md": f"# Shop {i}\n"}, f"edit {i}")
        setup.sync()
        setup.build(setup.review(), summarize=lambda path, source: "summary")
    old_format = "module:0:payments:abc"
    gone_module = f"module:{SUMMARY_VERSION}:removed/module:def"
    with closing(connect(setup.db_path)) as conn:
        for repo, key in (
            ("octo/shop", old_format),
            ("octo/shop", gone_module),
            ("octo/other", "layer:keep-me"),
        ):
            conn.execute(
                "INSERT INTO context_cache (repo, layer, cache_key, content, tokens)"
                " VALUES (?, 'repo', ?, '{}', 1)",
                (repo, key),
            )

        prune(conn, "octo/shop", keep_layers=2)

        rows = conn.execute("SELECT repo, cache_key FROM context_cache").fetchall()
    keys = {(row["repo"], row["cache_key"]) for row in rows}
    layers = [key for repo, key in keys if repo == "octo/shop" and key.startswith("layer:")]
    assert len(layers) == 2
    # Summaries are a knowledge base that only grows: a module that is gone or comes
    # back keeps its summary; only summaries of an older format are dropped.
    assert ("octo/shop", gone_module) in keys
    assert ("octo/shop", old_format) not in keys
    assert ("octo/other", "layer:keep-me") in keys
