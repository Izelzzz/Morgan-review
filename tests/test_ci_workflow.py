"""The workflow template and its two actions, checked as files (issue #14, D-037).

GitHub offers no way for the service to ask what a token may do, so what the token can
do is checked where it is decided: in the permissions the template gives each job.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

CI = Path(__file__).parent.parent / "ci"
TEMPLATE = CI / "review-agents.yml"
ACTIONS = [CI / "review" / "action.yml", CI / "post" / "action.yml"]


def load(path: Path) -> dict[Any, Any]:
    data: dict[Any, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def steps(data: dict[str, Any]) -> list[dict[str, Any]]:
    if "jobs" in data:
        return [s for job in data["jobs"].values() for s in job.get("steps", [])]
    return list(data["runs"]["steps"])


def test_the_token_the_service_gets_can_only_read() -> None:
    jobs = load(TEMPLATE)["jobs"]

    assert load(TEMPLATE)["permissions"] == {}  # nothing unless a job asks
    assert jobs["review"]["permissions"] == {
        "contents": "read",
        "issues": "read",
        "pull-requests": "read",
    }


def test_only_the_post_job_writes_and_only_comments() -> None:
    assert load(TEMPLATE)["jobs"]["post"]["permissions"] == {"pull-requests": "write"}


def test_the_post_job_reads_no_secret() -> None:
    # Its token writes; the service's secret stays with the job that talks to it.
    assert "secrets." not in yaml.safe_dump(load(TEMPLATE)["jobs"]["post"])


def test_the_triggers_cover_forks_and_the_review_command() -> None:
    triggers = load(TEMPLATE)[True]  # YAML 1.1 reads the key `on` as true

    assert triggers["pull_request_target"]["types"] == ["opened", "synchronize", "reopened"]
    assert triggers["issue_comment"]["types"] == ["created"]


@pytest.mark.parametrize("path", [TEMPLATE, *ACTIONS], ids=lambda p: p.parent.name + "/" + p.name)
def test_nothing_checks_out_the_pull_request(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    assert not any("actions/checkout" in s.get("uses", "") for s in steps(load(path)))
    assert "github.event.pull_request.head" not in text


@pytest.mark.parametrize("path", [TEMPLATE, *ACTIONS], ids=lambda p: p.parent.name + "/" + p.name)
def test_every_action_is_pinned_to_a_commit(path: Path) -> None:
    for step in steps(load(path)):
        uses = step.get("uses")
        if uses is None:
            continue
        ref = uses.rsplit("@", 1)[1]
        if uses.startswith("Izelzzz/review-lab/"):
            assert ref == "<commit>"  # the user puts the commit they chose
        else:
            assert re.fullmatch(r"[0-9a-f]{40}", ref), uses


@pytest.mark.parametrize("path", [TEMPLATE, *ACTIONS], ids=lambda p: p.parent.name + "/" + p.name)
def test_no_event_text_is_put_into_a_shell_command(path: Path) -> None:
    # A PR title or comment in `run:` would be a script injection; the script reads the
    # event from its file instead.
    for step in steps(load(path)):
        assert "${{" not in step.get("run", ""), step["run"]


def test_a_broken_review_job_still_reports_red() -> None:
    post = load(TEMPLATE)["jobs"]["post"]

    # A skipped required check counts as passing; the post job must run and fail instead.
    assert post["needs"] == "review"
    assert "!cancelled()" in post["if"] and "needs.review.result != 'skipped'" in post["if"]


def test_posts_for_one_pr_run_one_at_a_time() -> None:
    # Not the review job's group: a queued post there would cancel the newer review.
    concurrency = load(TEMPLATE)["jobs"]["post"]["concurrency"]

    assert concurrency["group"] == (
        "review-agents-post-${{ github.event.pull_request.number || github.event.issue.number }}"
    )
    assert concurrency["cancel-in-progress"] is False


def test_the_result_is_kept_when_the_review_step_fails() -> None:
    # A crash or missing secrets still writes a result; the post job says 审查失败 with it.
    [_, upload] = steps(load(CI / "review" / "action.yml"))

    assert upload["if"] == "always()"
