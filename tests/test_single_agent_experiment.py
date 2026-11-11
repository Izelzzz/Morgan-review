"""The single general agent experiment (not for main)."""

import pytest

from review_agents.context.pack import DIMENSIONS, GENERAL, SKILLS, task
from review_agents.pipeline.run import mode_plan
from review_agents.repo_config import RepoConfig


def test_the_general_task_carries_every_dimension_s_rules() -> None:
    text = task(GENERAL)

    for dimension in DIMENSIONS:
        assert SKILLS[dimension] in text and f"`{dimension}`" in text


def test_the_switch_runs_one_general_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_AGENTS_SINGLE_AGENT", "1")

    assert mode_plan(RepoConfig()) == ([GENERAL], 1)


def test_without_the_switch_the_five_dimensions_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REVIEW_AGENTS_SINGLE_AGENT", raising=False)

    assert mode_plan(RepoConfig()) == (list(DIMENSIONS), 1)
