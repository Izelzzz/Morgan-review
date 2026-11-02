import json
import re
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from gitrepo import Upstream, git
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from review_agents.config import ModelEndpoint, Settings
from review_agents.context.git import Snapshot
from review_agents.credentials import TokenVault
from review_agents.db import connect, migrate
from review_agents.mirror import MirrorStore
from review_agents.models import MAX_RETRIES, BudgetExceededError
from review_agents.pipeline.agent import SUBMIT_TOOL
from review_agents.pipeline.merge import MERGE_TOOL
from review_agents.pipeline.run import ReviewSuperseded, run_review, wait_for_summaries
from review_agents.worker import claim_next, recover_interrupted

DIMENSION = re.compile(r"^name: (\w+)$", re.MULTILINE)


class ByDimension(BaseChatModel):
    """Answers each dimension agent from its own script, safe to share across threads."""

    scripts: dict[str, list[Any]]
    delay: dict[str, float] = Field(default_factory=dict)
    seen: list[str] = Field(default_factory=list)
    lock: Any = Field(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "by-dimension"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ByDimension":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        task = next((str(m.content) for m in messages if DIMENSION.search(str(m.content))), "")
        match = DIMENSION.search(task)
        if match is None:  # a module summary for the repository layer
            return ChatResult(generations=[ChatGeneration(message=AIMessage("A module."))])
        dimension = match.group(1)
        time.sleep(self.delay.get(dimension, 0.0))
        with self.lock:
            self.seen.append(dimension)
            script = self.scripts.get(dimension) or [submit()]
            reply = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


class JudgeAll(BaseChatModel):
    """A merge judge that puts every candidate into one group."""

    kind: bool = False

    @property
    def _llm_type(self) -> str:
        return "judge-all"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "JudgeAll":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        ids = [int(i) for i in re.findall(r"^#(\d+) ", str(messages[-1].content), re.MULTILINE)]
        args = (
            {"same_problem": [], "same_kind": [ids]}
            if self.kind
            else {"same_problem": [ids], "same_kind": []}
        )
        call = {"name": MERGE_TOOL, "args": args, "id": "m"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=[call]))])


def finding(dimension: str, **changes: Any) -> dict[str, Any]:
    return {
        "title": f"{dimension}: refunds can exceed the paid amount",
        "severity": "important",
        "dimension": dimension,
        "file": "payments/refund.py",
        "line_start": 1,
        "line_end": 2,
        "evidence": "refund(order, 10_000) on an order paid 100",
        "why": "Customers are refunded more than they paid.",
        "suggestion": "Clamp to order.paid.",
        **changes,
    }


def submit(*findings: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": SUBMIT_TOOL, "args": {"findings": list(findings)}, "id": "s"}],
    )


class Setup:
    def __init__(self, tmp_path: Path) -> None:
        self.upstream = Upstream(tmp_path, "octo/shop")
        self.base = self.upstream.commit(
            {"payments/refund.py": "def refund(order, cents):\n    return cents\n"}, "init"
        )
        git(self.upstream.work, "checkout", "-q", "-b", "pr")
        self.head = self.upstream.commit(
            {"payments/refund.py": "def refund(order, cents):\n    return min(cents, 1)\n"},
            "clamp",
            branch="pr",
        )
        git(self.upstream.work, "checkout", "-q", "main")
        # GitHub keeps every PR's head under refs/pull/<number>/head.
        git(self.upstream.bare, "update-ref", "refs/pull/7/head", self.head)
        self.settings = Settings(
            data_dir=tmp_path / "data",
            finder_model=ModelEndpoint.model_validate(
                {"base_url": "http://127.0.0.1:9/v1", "model": "finder"}
            ),
        )
        self.settings.data_dir.mkdir()
        with closing(connect(self.settings.db_path)) as conn:
            migrate(conn)
        self.store = MirrorStore(self.settings.repos_dir, remote_base=str(self.upstream.base))
        self.vault = TokenVault()

    def queue(self, config: str | None = None, head: str | None = None) -> int:
        if config is not None:
            self.upstream.commit({".github/review-agents.yml": config}, "config")
        with closing(connect(self.settings.db_path)) as conn:
            row = conn.execute(
                "INSERT INTO reviews (repo, pr, base_sha, head_sha, mode, default_branch, title,"
                " body, event, status, started_at) VALUES ('octo/shop', 7, ?, ?, 'standard',"
                " 'main', 'Clamp refunds', 'Never refund more than paid.', 'opened', 'running',"
                " strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) RETURNING id",
                (self.base, head or self.head),
            ).fetchone()
        return int(row["id"])

    def run(self, review_id: int, model: ByDimension, **options: Any) -> None:
        with closing(connect(self.settings.db_path)) as conn:
            run_review(
                conn,
                review_id,
                settings=self.settings,
                vault=self.vault,
                store=self.store,
                model_for=lambda timeout: model,
                **options,
            )

    def findings(self, review_id: int) -> list[dict[str, Any]]:
        with closing(connect(self.settings.db_path)) as conn:
            rows = conn.execute(
                "SELECT primary_dimension, dimensions, severity, file, title, body, suggestion,"
                " fingerprint, votes, locations FROM findings WHERE review_id = ? ORDER BY id",
                (review_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def steps(self, review_id: int) -> dict[str, tuple[str, str | None]]:
        with closing(connect(self.settings.db_path)) as conn:
            rows = conn.execute(
                "SELECT node, status, error FROM steps WHERE review_id = ? ORDER BY id",
                (review_id,),
            ).fetchall()
        return {row["node"]: (row["status"], row["error"]) for row in rows}

    def review(self, review_id: int) -> dict[str, Any]:
        with closing(connect(self.settings.db_path)) as conn:
            return dict(conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone())


@pytest.fixture
def setup(tmp_path: Path) -> Iterator[Setup]:
    yield Setup(tmp_path)


ALL = ("spec_drift", "edge_cases", "security", "maintainability", "verifiability")


def test_every_enabled_dimension_runs_and_its_findings_are_written(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    setup.run(review_id, model)

    assert sorted(set(model.seen)) == sorted(ALL)
    [written] = setup.findings(review_id)
    assert written["primary_dimension"] == "edge_cases"
    assert json.loads(written["dimensions"]) == ["edge_cases"]
    assert written["body"] == "Customers are refunded more than they paid."
    assert written["suggestion"] == "Clamp to order.paid."
    assert len(written["fingerprint"]) == 64
    steps = setup.steps(review_id)
    assert all(steps[f"dimension.{d}"] == ("ok", None) for d in ALL)
    assert steps["context.pr_layer"][0] == "ok"


def test_only_the_dimensions_the_repository_enables_run(setup: Setup) -> None:
    config = "dimensions:\n  spec_drift: false\n  maintainability: false\n  verifiability: false\n"
    review_id = setup.queue(config)
    model = ByDimension(scripts={})

    setup.run(review_id, model)

    assert sorted(set(model.seen)) == ["edge_cases", "security"]


def test_one_failing_dimension_leaves_the_others_and_says_why(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(
        scripts={
            "security": [RuntimeError("model service unavailable")],
            "edge_cases": [submit(finding("edge_cases"))],
        }
    )

    setup.run(review_id, model)

    assert [f["primary_dimension"] for f in setup.findings(review_id)] == ["edge_cases"]
    status, error = setup.steps(review_id)["dimension.security"]
    assert status == "error" and "model service unavailable" in (error or "")


def test_a_dimension_still_running_at_its_deadline_is_left_behind(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(
        scripts={"edge_cases": [submit(finding("edge_cases"))]}, delay={"security": 3.0}
    )

    started = time.monotonic()
    setup.run(review_id, model, review_seconds=2.5, wrap_up_seconds=1.0)

    assert time.monotonic() - started < 2.8  # it did not wait for the slow dimension
    assert [f["primary_dimension"] for f in setup.findings(review_id)] == ["edge_cases"]
    status, error = setup.steps(review_id)["dimension.security"]
    assert status == "error" and "deadline" in (error or "")
    assert setup.review(review_id)["limit_hit"] == "time"


def test_a_spent_token_budget_is_recorded_on_the_review(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={d: [BudgetExceededError("budget used up")] for d in ALL})

    setup.run(review_id, model)

    assert setup.review(review_id)["limit_hit"] == "tokens"


def test_a_superseded_review_stops_and_writes_nothing(setup: Setup) -> None:
    review_id = setup.queue()
    newer = setup.queue(head="c" * 40)
    with closing(connect(setup.settings.db_path)) as conn:
        conn.execute("UPDATE reviews SET superseded_by = ? WHERE id = ?", (newer, review_id))
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    with pytest.raises(ReviewSuperseded, match=f"review {newer}"):
        setup.run(review_id, model)

    assert setup.findings(review_id) == []
    assert model.seen == []


def test_a_review_whose_token_was_lost_fails_before_any_work(setup: Setup) -> None:
    review_id = setup.queue()
    with closing(connect(setup.settings.db_path)) as conn:
        conn.execute("UPDATE reviews SET has_token = 1 WHERE id = ?", (review_id,))

    with pytest.raises(RuntimeError, match="token"):
        setup.run(review_id, ByDimension(scripts={}))


def test_findings_are_written_all_at_once_or_not_at_all(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_id = setup.queue()
    # Two different problems, so the merge keeps both and there are two writes.
    model = ByDimension(
        scripts={
            "edge_cases": [submit(finding("edge_cases", line_start=1, line_end=1))],
            "security": [submit(finding("security", line_start=2, line_end=2))],
        }
    )
    from review_agents.pipeline import run as run_module

    original = run_module._insert_finding
    calls = []

    def fail_on_second(*args: Any) -> None:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        original(*args)

    monkeypatch.setattr(run_module, "_insert_finding", fail_on_second)

    with pytest.raises(RuntimeError, match="disk full"):
        setup.run(review_id, model)

    assert setup.findings(review_id) == []


def test_after_a_restart_a_review_without_a_token_is_queued_once_more(setup: Setup) -> None:
    review_id = setup.queue()
    with closing(connect(setup.settings.db_path)) as conn:
        conn.execute("UPDATE reviews SET attempts = 1 WHERE id = ?", (review_id,))
        recover_interrupted(conn)
        assert conn.execute("SELECT status FROM reviews").fetchone()[0] == "queued"
        assert claim_next(conn) == review_id  # a second attempt
        recover_interrupted(conn)
        row = conn.execute("SELECT status, error FROM reviews").fetchone()
    assert row["status"] == "failed" and "restart" in row["error"]


def test_a_review_shorter_than_the_wrap_up_reserve_still_reviews(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    # 60 seconds in all, less than the 240-second reserve on its own.
    setup.run(review_id, model, review_seconds=60.0)

    assert [f["primary_dimension"] for f in setup.findings(review_id)] == ["edge_cases"]


def test_missing_module_summaries_are_written_in_the_background(setup: Setup) -> None:
    review_id = setup.queue()
    slow_summaries = threading.Event()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})
    original = model._generate

    def summaries_wait(messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        if not any(DIMENSION.search(str(m.content)) for m in messages):
            slow_summaries.wait(5)  # a module summary: held until the review is done
        return original(messages, *args, **kwargs)

    object.__setattr__(model, "_generate", summaries_wait)

    setup.run(review_id, model)  # returns although the summaries are still waiting

    assert [f["primary_dimension"] for f in setup.findings(review_id)] == ["edge_cases"]
    slow_summaries.set()
    wait_for_summaries()
    with closing(connect(setup.settings.db_path)) as conn:
        written = conn.execute(
            "SELECT count(*) FROM context_cache WHERE cache_key LIKE 'module:%'"
        ).fetchone()[0]
    assert written >= 1


def test_findings_in_an_unreadable_file_do_not_share_a_fingerprint(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As for a file over the size limit or a binary one: there is no text to fingerprint.
    monkeypatch.setattr(Snapshot, "read_text", lambda self, path: None)
    review_id = setup.queue()
    first = finding("edge_cases", line_start=1, line_end=1, title="one")
    second = finding("edge_cases", line_start=2, line_end=2, title="two")
    model = ByDimension(scripts={"edge_cases": [submit(first, second)]})

    setup.run(review_id, model)

    prints = {f["fingerprint"] for f in setup.findings(review_id)}
    assert len(prints) == 2


def test_stored_findings_and_done_are_one_commit_so_a_restart_cannot_repeat_them(
    setup: Setup,
) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    setup.run(review_id, model)
    # The process dies before the worker's own bookkeeping; the next one recovers.
    with closing(connect(setup.settings.db_path)) as conn:
        recover_interrupted(conn)
        assert claim_next(conn) is None  # nothing to run again

    assert setup.review(review_id)["status"] == "done"
    assert len(setup.findings(review_id)) == 1


def test_a_single_enabled_dimension_completes_the_review(setup: Setup) -> None:
    others = "\n".join(f"  {d}: false" for d in ALL if d != "security")
    review_id = setup.queue(f"dimensions:\n{others}\n")
    model = ByDimension(scripts={"security": [submit(finding("security"))]})

    setup.run(review_id, model)

    assert set(model.seen) == {"security"}
    assert [f["primary_dimension"] for f in setup.findings(review_id)] == ["security"]
    assert setup.steps(review_id)["config"] == ("ok", None)


def test_quick_mode_runs_only_edge_cases_and_security(setup: Setup) -> None:
    review_id = setup.queue("mode: quick\n")
    model = ByDimension(scripts={})

    setup.run(review_id, model)

    assert set(model.seen) == {"edge_cases", "security"}


def test_quick_mode_falls_back_to_an_enabled_dimension(setup: Setup) -> None:
    others = "\n".join(f"  {d}: false" for d in ALL if d != "maintainability")
    review_id = setup.queue(f"mode: quick\ndimensions:\n{others}\n")
    model = ByDimension(scripts={})

    setup.run(review_id, model)

    assert set(model.seen) == {"maintainability"}


def test_deep_mode_runs_each_dimension_three_times_and_counts_votes(setup: Setup) -> None:
    others = "\n".join(f"  {d}: false" for d in ALL if d != "edge_cases")
    review_id = setup.queue(f"mode: deep\ndimensions:\n{others}\n")
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    setup.run(review_id, model)

    assert model.seen.count("edge_cases") == 3
    [stored] = setup.findings(review_id)  # the same problem, reported in all three runs
    assert stored["votes"] == 3
    steps = setup.steps(review_id)
    assert {"dimension.edge_cases.run1", "dimension.edge_cases.run3"} <= set(steps)


def test_the_judge_merges_one_problem_two_dimensions_found(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(
        scripts={
            "edge_cases": [submit(finding("edge_cases"))],
            "security": [submit(finding("security", line_end=1, evidence="much more evidence"))],
        }
    )
    judge = JudgeAll()

    setup.run(review_id, model, judge_for=lambda timeout: judge)

    [stored] = setup.findings(review_id)
    assert json.loads(stored["dimensions"]) == ["edge_cases", "security"]
    assert setup.steps(review_id)["merge"] == ("ok", None)


def test_one_kind_of_problem_is_stored_once_with_its_other_places(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(
        scripts={
            "edge_cases": [
                submit(
                    finding("edge_cases", line_start=1, line_end=1),
                    finding("edge_cases", line_start=2, line_end=2, title="again"),
                )
            ]
        }
    )

    setup.run(review_id, model, judge_for=lambda timeout: JudgeAll(kind=True))

    [stored] = setup.findings(review_id)
    assert json.loads(stored["locations"]) == [["payments/refund.py", 2, 2]]


def test_the_judge_is_not_asked_when_there_is_nothing_to_merge(setup: Setup) -> None:
    asked: list[float] = []

    def judge_for(timeout: float) -> JudgeAll:
        asked.append(timeout)
        return JudgeAll()

    review_id = setup.queue()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})
    setup.run(review_id, model, judge_for=judge_for)

    assert asked == []


def test_the_judge_and_its_retries_fit_in_the_time_left(setup: Setup) -> None:
    asked: list[float] = []

    def judge_for(timeout: float) -> JudgeAll:
        asked.append(timeout)
        return JudgeAll()

    review_id = setup.queue()
    model = ByDimension(scripts={d: [submit(finding(d))] for d in ("edge_cases", "security")})
    setup.run(review_id, model, judge_for=judge_for, review_seconds=30.0)

    assert len(asked) == 1 and asked[0] * (MAX_RETRIES + 1) <= 30.0
