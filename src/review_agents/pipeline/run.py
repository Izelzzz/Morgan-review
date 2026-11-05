"""One review from the queue to its stored findings (pipeline design §2, §6, §7).

The steps run in order in plain Python (decision D-023): sync the mirror, read the
config from the default branch, build the context, run the dimension agents side by
side, write the findings. Between steps the review checks that a newer push has not
superseded it (decision D-010).

A dimension that fails or runs out of time ends alone and is recorded in `steps`; only
a failure to get the code, a superseded review or a failed write ends the review.
Merging duplicate findings and the three modes come next (pipeline design §4, §5).
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from review_agents.config import Settings
from review_agents.context.cache import repo_layer_for_review
from review_agents.context.git import Snapshot
from review_agents.context.modules import Summarize, module_summaries, module_tokens
from review_agents.context.pack import DIMENSIONS, ContextPack, build_packs
from review_agents.context.pr_layer import PrRequest, build_pr_layer
from review_agents.context.summarizer import model_summarizer
from review_agents.credentials import TokenVault
from review_agents.db import connect
from review_agents.mirror import MirrorStore
from review_agents.models import (
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    TokenBudget,
    UsageMeter,
    chat_model,
    review_usage,
    summary_usage,
)
from review_agents.pipeline.agent import AgentRun, Finding, run_dimension
from review_agents.pipeline.conversation import ConversationLimits
from review_agents.pipeline.merge import Candidate, Merged, merge, model_judge
from review_agents.repo_config import RepoConfig, read_repo_config
from review_agents.tools import RepoTools

# Builds the finder model for one agent; its argument bounds each call in seconds.
ModelFor = Callable[[float], BaseChatModel]
log = logging.getLogger(__name__)
_summaries: dict[str, threading.Thread] = {}
_summaries_guard = threading.Lock()

# When one review hits several limits, the report names the most severe.
_LIMIT_ORDER = ("time", "tokens", "tool_calls")


class ReviewSuperseded(RuntimeError):
    """A newer push of the same PR replaced this review; nothing of it is published."""


@dataclass(frozen=True)
class Job:
    group: int
    groups: int
    dimension: str
    pack: ContextPack
    run: int = 1
    runs: int = 1

    @property
    def node(self) -> str:
        suffix = f".group{self.group}" if self.groups > 1 else ""
        suffix += f".run{self.run}" if self.runs > 1 else ""
        return f"dimension.{self.dimension}{suffix}"


def enabled_dimensions(config: RepoConfig) -> list[str]:
    return [d for d in DIMENSIONS if getattr(config.dimensions, d)]


# Quick mode looks at edge cases and security only (architecture §4.3); Deep mode runs
# each dimension three times and keeps what any run reports (D-022).
QUICK_DIMENSIONS = ("edge_cases", "security")
DEEP_RUNS = 3


def mode_plan(config: RepoConfig) -> tuple[list[str], int]:
    """The dimensions a review runs and how many times each, from the config's mode."""
    enabled = enabled_dimensions(config)
    if config.mode == "quick":
        # With neither of them enabled, the first enabled one still makes it a review.
        return [d for d in enabled if d in QUICK_DIMENSIONS] or enabled[:1], 1
    return enabled, DEEP_RUNS if config.mode == "deep" else 1


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _now(conn: sqlite3.Connection) -> str:
    return str(conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')").fetchone()[0])


def _step(
    conn: sqlite3.Connection, review_id: int, node: str, started: str, error: str | None = None
) -> None:
    conn.execute(
        "INSERT INTO steps (review_id, node, started_at, finished_at, status, error)"
        " VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?, ?)",
        (review_id, node, started, "error" if error else "ok", error),
    )


def _check_not_superseded(conn: sqlite3.Connection, review_id: int) -> None:
    row = conn.execute("SELECT superseded_by FROM reviews WHERE id = ?", (review_id,)).fetchone()
    if row["superseded_by"] is not None:
        raise ReviewSuperseded(f"superseded by review {row['superseded_by']}")


def fingerprint(snapshot: Snapshot, finding: Finding) -> str:
    """The same problem at the same code, wherever the lines moved (decision D-022)."""
    text = snapshot.read_text(finding.file)
    if text is None:  # too large or binary to read: the line range stands in for the code
        code = f"lines {finding.line_start}-{finding.line_end}"
    else:
        lines = text.splitlines()[finding.line_start - 1 : finding.line_end]
        code = "\n".join(" ".join(line.split()) for line in lines)
    return hashlib.sha256(f"{finding.file}\n{code}".encode()).hexdigest()


def _insert_finding(conn: sqlite3.Connection, review_id: int, merged: Merged) -> None:
    finding = merged.finding
    lead = next(c for c in merged.sources if c.finding is finding)
    conn.execute(
        "INSERT INTO findings (review_id, fingerprint, primary_dimension, dimensions, severity,"
        " preexisting, file, line_start, line_end, title, body, suggestion, evidence, votes,"
        " locations) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            review_id,
            lead.fingerprint,
            finding.dimension,
            json.dumps(merged.dimensions),
            finding.severity,
            finding.preexisting,
            finding.file,
            finding.line_start,
            finding.line_end,
            finding.title,
            finding.why,
            finding.suggestion,
            finding.evidence,
            merged.votes,
            json.dumps(merged.locations),
        ),
    )


def run_review(
    conn: sqlite3.Connection,
    review_id: int,
    *,
    settings: Settings,
    vault: TokenVault,
    store: MirrorStore,
    model_for: ModelFor,
    judge_for: ModelFor | None = None,
    clock: Callable[[], float] = time.monotonic,
    review_seconds: float | None = None,
    wrap_up_seconds: float | None = None,
) -> None:
    """Review one claimed review and store its findings; raise if it cannot be done."""
    if settings.finder_model is None:
        raise RuntimeError("no finder model is configured")
    started_at = clock()
    review = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
    token = vault.get(review_id)
    if review["has_token"] and token is None:
        raise RuntimeError("its token is no longer held; run the CI job again")
    repo, head = review["repo"], review["head_sha"]
    _check_not_superseded(conn, review_id)

    step = _now(conn)
    # The default branch for the config and repository layer; the PR's own head ref,
    # which GitHub keeps for forks too, for the commits under review.
    refs = [f"refs/heads/{review['default_branch']}", f"refs/pull/{review['pr']}/head"]
    git_dir = store.sync(repo, refs, [review["base_sha"], head], token)
    _step(conn, review_id, "mirror.sync", step)
    step = _now(conn)
    loaded = read_repo_config(git_dir, review["default_branch"])
    config = loaded.config
    # An invalid config falls back to the defaults as a whole (D-001); the step says why.
    _step(conn, review_id, "config", step, "; ".join(loaded.errors) or None)
    limits = config.limits
    deadline = started_at + (
        review_seconds if review_seconds is not None else limits.review_minutes * 60
    )
    total = deadline - started_at
    reserve = wrap_up_seconds if wrap_up_seconds is not None else settings.wrap_up_seconds
    # A repository may shorten the review below the reserve; the dimensions keep most of it.
    dimension_deadline = deadline - min(reserve, total / 3)
    budget = TokenBudget(limits.token_budget(settings.tokens_per_review))
    skip = config.effective_skip_paths
    window = settings.finder_model.context_tokens

    _check_not_superseded(conn, review_id)
    # The review reads the summaries that exist; missing ones are written in the
    # background, so a repository's first review does not spend its time on them and
    # git stats stand in until they are there (context design §5).
    repo_layer = repo_layer_for_review(
        conn,
        review_id,
        repo,
        git_dir,
        review["default_branch"],
        skip,
        None,
        module_tokens=module_tokens(window),
    )
    if any(m.summary is None for m in repo_layer.modules):
        summarize = model_summarizer(
            model_for(REQUEST_TIMEOUT_SECONDS),
            UsageMeter(
                TokenBudget(settings.summary_tokens_per_build),
                summary_usage(settings.db_path, repo),
            ),
        )
        summarize_in_background(
            settings.db_path, repo, git_dir, review["default_branch"], skip,
            module_tokens(window), summarize,
        )  # fmt: skip
    step = _now(conn)
    request = PrRequest(
        repo, review["pr"], review["title"], review["body"], review["base_sha"], head
    )
    pr_layer = build_pr_layer(
        conn, request, git_dir, config, token=token, surrounding_code=settings.surrounding_code
    )
    packs = build_packs(repo_layer, pr_layer, window)
    _step(conn, review_id, "context.pr_layer", step)
    if len(packs) > 1:
        _step(conn, review_id, f"context.groups.{len(packs)}", step)

    _check_not_superseded(conn, review_id)
    dimensions, repeats = mode_plan(config)
    jobs = [
        Job(index, len(packs), dimension, pack, run, repeats)
        for run in range(1, repeats + 1)  # every first run is queued before any second one
        for index, pack in enumerate(packs, start=1)
        for dimension in dimensions
    ]
    runs = _run_dimensions(
        conn,
        review_id,
        jobs,
        settings=settings,
        token=token,
        git_dir=git_dir,
        skip=skip,
        tool_calls=limits.tool_calls_per_agent,
        budget=budget,
        model_for=model_for,
        clock=clock,
        deadline=dimension_deadline,
    )

    _check_not_superseded(conn, review_id)
    snapshot = Snapshot(git_dir, head, tuple(skip))
    candidates = [
        Candidate(index, item, job.run, fingerprint(snapshot, item), job.node)
        for index, (job, item) in enumerate(
            ((job, item) for job, run in zip(jobs, runs, strict=True) for item in run.findings),
            start=1,
        )
    ]
    step = _now(conn)
    judge = None
    # One candidate has nothing to merge with; the judge is not asked.
    if judge_for is not None and len(candidates) > 1:
        left = (deadline - clock()) / (MAX_RETRIES + 1)  # the client's retries share it
        timeout = max(1.0, min(REQUEST_TIMEOUT_SECONDS, left))
        meter = UsageMeter(budget, review_usage(settings.db_path, review_id, "merge"))
        judge = model_judge(judge_for(timeout), callbacks=[meter])
    # A judge that fails or answers nonsense leaves the safe merge: the review still
    # completes, and the step says the merge was degraded.
    fallbacks: list[str] = []
    merged = merge(candidates, judge, fallbacks.append)
    _step(conn, review_id, "merge", step, fallbacks[0] if fallbacks else None)
    limit_hit = next((lim for lim in _LIMIT_ORDER if any(r.limit_hit == lim for r in runs)), None)
    # The findings and the review's end state commit together (reliability §5.3): never
    # half a review, and a review with stored findings is never claimed and run again.
    conn.execute("BEGIN IMMEDIATE")
    try:
        for item in merged:
            _insert_finding(conn, review_id, item)
        conn.execute(
            "UPDATE reviews SET limit_hit = ?, status = 'done', error = NULL,"
            " finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
            (limit_hit, review_id),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _run_dimensions(
    conn: sqlite3.Connection,
    review_id: int,
    jobs: list[Job],
    *,
    settings: Settings,
    token: str | None,
    git_dir: Path,
    skip: list[str],
    tool_calls: int,
    budget: TokenBudget,
    model_for: ModelFor,
    clock: Callable[[], float],
    deadline: float,
) -> list[AgentRun]:
    """Every job side by side, at most `model_concurrency` at once, until the deadline.

    A job still running at the deadline is left behind: every model call it makes is
    built with a timeout that fits, retries included, in the time left, so its thread
    ends by the deadline plus at most the one-second floor of a call's timeout.
    """
    repo = conn.execute("SELECT repo, head_sha FROM reviews WHERE id = ?", (review_id,)).fetchone()

    starts: dict[str, str] = {}
    assert settings.finder_model is not None  # noqa: S101 - checked when the review starts
    limits = ConversationLimits(
        window=settings.finder_model.context_tokens,
        result_tokens=settings.tool_result_tokens,
        shortened_tokens=settings.shortened_result_tokens,
        output_reserve=settings.output_reserve_tokens,
    )

    def work(job: Job) -> AgentRun:
        starts[job.node] = _utc_now()  # when the job runs, not when it was queued
        tools = RepoTools(git_dir, repo["repo"], repo["head_sha"], skip, token=token)
        meter = UsageMeter(budget, review_usage(settings.db_path, review_id, job.dimension))
        return run_dimension(
            model_for,
            job.pack,
            job.dimension,
            tools,
            tool_calls=tool_calls,
            callbacks=[meter],
            deadline=deadline,
            clock=clock,
            limits=limits,
        )

    pool = ThreadPoolExecutor(max_workers=settings.model_concurrency)
    futures: dict[Future[AgentRun], Job] = {pool.submit(work, job): job for job in jobs}
    try:
        wait(futures, timeout=max(0.0, deadline - clock()))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    runs = []
    for future, job in futures.items():
        if not future.done():
            run = AgentRun(job.dimension, "incomplete", limit_hit="time")
            run.reason = "did not finish before the dimension phase deadline"
        elif future.cancelled():
            run = AgentRun(job.dimension, "incomplete", limit_hit="time")
            run.reason = "not started before the dimension phase deadline"
        elif (exc := future.exception()) is not None:
            run = AgentRun(job.dimension, "incomplete", reason=f"{type(exc).__name__}: {exc}")
        else:
            run = future.result()
        error = None if run.status == "done" else (run.reason or "incomplete")
        _step(conn, review_id, job.node, starts.get(job.node, _now(conn)), error)
        runs.append(run)
    return runs


def summarize_in_background(
    db_path: Path,
    repo: str,
    git_dir: Path,
    default_branch: str,
    skip: list[str],
    tokens: int,
    summarize: Summarize,
) -> None:
    """Write a repository's missing module summaries on a thread of their own.

    One build per repository at a time; a review that finds one running leaves it be.
    The summaries' own budget (`summary_tokens_per_build`) bounds what a build spends.
    """

    def build() -> None:
        try:
            with closing(connect(db_path, create=False)) as conn:
                snapshot = Snapshot.at_branch(git_dir, default_branch, tuple(skip))
                module_summaries(conn, repo, snapshot, summarize, tokens)
        except Exception:
            log.exception("writing module summaries for %s failed", repo)

    with _summaries_guard:
        running = _summaries.get(repo)
        if running is not None and running.is_alive():
            return
        thread = threading.Thread(target=build, name=f"summaries:{repo}", daemon=True)
        _summaries[repo] = thread
        thread.start()


def wait_for_summaries(timeout: float | None = None) -> None:
    """Wait for the background summary builds; for tests and an orderly shutdown."""
    with _summaries_guard:
        threads = list(_summaries.values())
    for thread in threads:
        thread.join(timeout)


def review_handler(
    settings: Settings, vault: TokenVault
) -> Callable[[sqlite3.Connection, int], None]:
    """The worker's handler: the real pipeline, with the configured finder model."""
    if settings.finder_model is None:
        raise RuntimeError("REVIEW_AGENTS_FINDER_MODEL__* must be set to run reviews")
    endpoint = settings.finder_model
    verifier = settings.verifier_model
    store = MirrorStore(settings.repos_dir)

    def handle(conn: sqlite3.Connection, review_id: int) -> None:
        run_review(
            conn,
            review_id,
            settings=settings,
            vault=vault,
            store=store,
            model_for=lambda timeout: chat_model(endpoint, timeout=timeout),
            # The merge is judged by the strong model (D-022); without one, only the safe
            # merge runs: one dimension's agent runs agreeing on overlapping lines.
            judge_for=(lambda timeout: chat_model(verifier, timeout=timeout)) if verifier else None,
        )

    return handle
