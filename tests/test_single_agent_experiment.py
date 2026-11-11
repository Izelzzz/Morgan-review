"""The single general agent experiment (not for main)."""

from pathlib import Path

import pytest
import test_walkthrough as walkthrough_tests
from langchain_core.messages import ToolMessage
from scripted import ScriptedModel, call

from review_agents.context.pack import DIMENSIONS, GENERAL, SKILLS, task
from review_agents.pipeline.agent import SUBMIT_TOOL, run_dimension
from review_agents.pipeline.run import mode_plan
from review_agents.repo_config import RepoConfig
from review_agents.walkthrough import WALKTHROUGH_TOOL, run_walkthrough


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


def test_the_general_agent_carries_on_from_the_walkthrough(tmp_path: Path) -> None:
    # Walkthrough first: what it read and wrote is in front of the agent when it looks
    # for problems, so nothing is read twice.
    tools = walkthrough_tests.repo_tools.__wrapped__(tmp_path)  # type: ignore[attr-defined]
    writer = ScriptedModel(script=[call(WALKTHROUGH_TOOL, walkthrough_tests.answer(), "w1")])
    written = run_walkthrough(
        lambda left: writer,
        walkthrough_tests.pack(),
        tools,
        tool_calls=10,
        callbacks=[],
        clock=lambda: 0.0,
    )
    assert written.transcript is not None
    assert isinstance(written.transcript[-1], ToolMessage)
    assert written.transcript[-1].tool_call_id == "w1"

    finder = ScriptedModel(script=[call(SUBMIT_TOOL, {"findings": []}, "f1")])
    run = run_dimension(
        lambda left: finder,
        walkthrough_tests.pack(),
        GENERAL,
        tools,
        tool_calls=10,
        callbacks=[],
        clock=lambda: 0.0,
        opening=written.transcript,
    )

    assert run.status == "done"
    [sent] = finder.sent
    assert sent[: len(written.transcript)] == written.transcript
    assert "all five dimensions" in str(sent[-1].content)
