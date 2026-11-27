"""Who writes the findings and the walkthrough (issue #83, D-038)."""

from pathlib import Path
from typing import Any

import pytest
import test_walkthrough as walkthrough_tests
from gitrepo import Upstream
from scripted import ScriptedModel, hand_in

from review_agents.context.pack import DIMENSIONS, GENERAL, SKILLS, task
from review_agents.mirror import MirrorStore
from review_agents.pipeline.agent import run_dimension
from review_agents.pipeline.run import plan_jobs
from review_agents.pipeline.strategy import Author, plan
from review_agents.repo_config import RepoConfig
from review_agents.tools import RepoTools


def config(**changes: object) -> RepoConfig:
    return RepoConfig.model_validate(changes)


def test_dimensions_gives_each_enabled_dimension_an_agent() -> None:
    planned = plan("dimensions", config())

    assert planned.authors == [Author(d, (d,)) for d in DIMENSIONS]
    assert planned.runs == 1


def test_one_general_agent_holds_every_enabled_dimension() -> None:
    planned = plan("general", config())

    assert planned.authors == [Author(GENERAL, DIMENSIONS)]
    assert planned.dimensions == list(DIMENSIONS)  # what the report says was reviewed


def test_the_mode_still_picks_the_dimensions_and_the_runs() -> None:
    quick = plan("general", config(mode="quick"))
    deep = plan("general", config(mode="deep"))

    assert quick.authors == [Author(GENERAL, ("edge_cases", "security"))]
    assert (deep.authors, deep.runs) == ([Author(GENERAL, DIMENSIONS)], 3)


def test_a_single_dimension_needs_no_general_agent() -> None:
    only = {d: d == "security" for d in DIMENSIONS}

    assert plan("general", config(dimensions=only)).authors == [Author("security", ("security",))]


def test_the_general_task_carries_the_enabled_dimensions_rules_only() -> None:
    text = task(GENERAL, ("edge_cases", "security"))

    assert SKILLS["edge_cases"] in text and SKILLS["security"] in text
    assert SKILLS["spec_drift"] not in text
    assert "`dimension`" in text  # each finding is tagged with the one it belongs to


def test_one_dimension_s_task_is_unchanged() -> None:
    assert task("security") == task("security", ("security",))


# --- the agents ----------------------------------------------------------------


@pytest.fixture
def tools(tmp_path: Path) -> RepoTools:
    upstream = Upstream(tmp_path, "octo/shop")
    head = upstream.commit(walkthrough_tests.FILES, "init")
    store = MirrorStore(tmp_path / "repos", remote_base=str(upstream.base))
    git_dir = store.sync("octo/shop", ["refs/heads/main"], [head], None)
    return RepoTools(git_dir, "octo/shop", head, [])


def general_finding(dimension: str) -> dict[str, Any]:
    return {
        "title": f"refunds can exceed the paid amount ({dimension})",
        "severity": "nit",
        "dimension": dimension,
        "file": "payments/refund.py",
        "line_start": 1,
        "line_end": 2,
        "why": "Customers are refunded more than they paid.",
        "suggestion": "Clamp to order.paid.",
    }


def test_a_general_agent_s_findings_keep_the_dimension_it_tagged(tools: RepoTools) -> None:
    reply = hand_in(general_finding("security"), general_finding("spec_drift"))
    model = ScriptedModel(script=[reply])

    run = run_dimension(
        lambda left: model,
        walkthrough_tests.pack(),
        GENERAL,
        tools,
        tool_calls=10,
        callbacks=[],
        clock=lambda: 0.0,
        dimensions=("edge_cases", "security"),
    )

    # A dimension it does not hold, or a misspelt one, is not taken for the truth.
    assert [f.dimension for f in run.findings] == ["security", GENERAL]


def test_a_dimension_agent_s_findings_take_its_own_dimension(tools: RepoTools) -> None:
    model = ScriptedModel(script=[hand_in(general_finding("x"))])

    run = run_dimension(
        lambda left: model,
        walkthrough_tests.pack(),
        "security",
        tools,
        tool_calls=10,
        callbacks=[],
        clock=lambda: 0.0,
    )

    assert [f.dimension for f in run.findings] == ["security"]


def test_deep_mode_queues_every_group_s_agent_run_by_run() -> None:
    packs = [walkthrough_tests.pack(), walkthrough_tests.pack("en")]

    jobs = plan_jobs(plan("general", config(mode="deep")), packs)

    assert [(j.group, j.run) for j in jobs] == [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]
    assert all(j.dimensions == DIMENSIONS and j.dimension == GENERAL for j in jobs)


def test_dimension_jobs_are_queued_run_by_run() -> None:
    jobs = plan_jobs(plan("dimensions", config()), [walkthrough_tests.pack()])

    assert [j.dimension for j in jobs] == list(DIMENSIONS)
