from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from gitrepo import Upstream, git
from scripted import ScriptedModel, call

from review_agents.context.pack import SHARED_RULES, ContextPack
from review_agents.context.pr_layer import ChangedFile, PrLayer, PrRequest
from review_agents.context.repo_layer import RepoLayer
from review_agents.mirror import MirrorStore
from review_agents.tools import RepoTools
from review_agents.walkthrough import (
    MARKER,
    WALKTHROUGH_TOOL,
    PrFacts,
    Walkthrough,
    pr_facts,
    render_failed,
    render_walkthrough,
    run_walkthrough,
)

FILES: dict[str, str | bytes] = {
    "payments/refund.py": "def refund(order, cents):\n    total = cents\n    return total\n",
    "README.md": "# Shop\n",
}


@pytest.fixture
def repo_tools(tmp_path: Path) -> RepoTools:
    upstream = Upstream(tmp_path, "octo/shop")
    head = upstream.commit(FILES, "init")
    store = MirrorStore(tmp_path / "repos", remote_base=str(upstream.base))
    git_dir = store.sync("octo/shop", ["refs/heads/main"], [head], None)
    return RepoTools(git_dir, "octo/shop", head, [])


def pack(language: str = "zh") -> ContextPack:
    sha = "a" * 40
    repo = RepoLayer(commit=sha, conventions=[], tree="./ (2 files)", modules=[])
    pr = PrLayer(
        request=PrRequest(
            "octo/shop", 7, "Clamp refunds", "Never refund more than paid.", sha, sha
        ),
        issues=[],
        specs=[],
        files=[ChangedFile("M", "payments/refund.py")],
        diffs={"payments/refund.py": "+    total = cents\n"},
        excerpts=[],
        previous_findings=[],
    )
    return ContextPack(SHARED_RULES, repo, pr, language)  # type: ignore[arg-type]


def answer(**changes: Any) -> dict[str, Any]:
    card = {"headline": "h", "points": ["a", "b"]}
    base: dict[str, Any] = {
        "summary": "Refunds are clamped to what was paid.",
        "why": card,
        "what": card,
        "impact": card,
        "problems": [{"title": "Over-refunds", "explanation": "Customers got more back."}],
        "changes": [
            {
                "title": "Clamp in `refund`",
                "body": "The amount is clamped (`payments/refund.py:2`).",
            }
        ],
        "impacts": [{"aspect": "Results users see", "text": "Refunds never exceed payments."}],
    }
    return base | changes


def run(model: ScriptedModel, tools: RepoTools, language: str = "zh") -> Any:
    return run_walkthrough(
        lambda left: model, pack(language), tools, tool_calls=10, callbacks=[], clock=lambda: 0.0
    )


def test_a_walkthrough_whose_references_are_there_is_accepted(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[call(WALKTHROUGH_TOOL, answer())])

    result = run(model, repo_tools)

    assert result.status == "done" and result.walkthrough is not None
    assert result.walkthrough.summary == "Refunds are clamped to what was paid."


def test_the_walkthrough_follows_the_reviewers_opening_and_its_language(
    repo_tools: RepoTools,
) -> None:
    model = ScriptedModel(script=[call(WALKTHROUGH_TOOL, answer())])

    run(model, repo_tools, "en")

    sent = model.sent[0]
    assert [m.content for m in sent[:2]] == [m.content for m in pack().messages("security")[:2]]
    task = str(sent[2].content)
    assert "walkthrough" in task and "`submit_walkthrough`" in task
    assert "in English" in task.rsplit("\n\n", 1)[-1]


@pytest.mark.parametrize(
    ("where", "problem"),
    [
        ("`payments/missing.py:3`", "payments/missing.py is not in the reviewed commit"),
        ("`payments/refund.py:2-40`", "payments/refund.py has only 3 lines"),
    ],
)
def test_a_reference_that_is_not_there_is_asked_again_once(
    repo_tools: RepoTools, where: str, problem: str
) -> None:
    wrong = answer(changes=[{"title": "t", "body": f"See {where}."}])
    model = ScriptedModel(
        script=[call(WALKTHROUGH_TOOL, wrong, "w0"), call(WALKTHROUGH_TOOL, answer())]
    )

    result = run(model, repo_tools)

    assert result.status == "done"
    assert problem in str(model.sent[1][-1].content)


def test_a_walkthrough_still_wrong_after_one_correction_is_not_published(
    repo_tools: RepoTools,
) -> None:
    wrong = call(WALKTHROUGH_TOOL, answer(summary="See `nowhere.py:1`."))
    model = ScriptedModel(script=[wrong, wrong])

    result = run(model, repo_tools)

    assert result.status == "incomplete" and result.walkthrough is None
    assert result.reason is not None


def test_more_than_three_diagrams_are_asked_again(repo_tools: RepoTools) -> None:
    fenced = "```mermaid\nflowchart LR\n  A --> B\n```"
    many = answer(
        diagram="flowchart LR\n  A --> B",
        changes=[{"title": f"c{n}", "body": fenced} for n in range(3)],
    )
    model = ScriptedModel(
        script=[call(WALKTHROUGH_TOOL, many, "w0"), call(WALKTHROUGH_TOOL, answer())]
    )

    run(model, repo_tools)

    assert "4 Mermaid diagrams" in str(model.sent[1][-1].content)


FACTS = PrFacts(
    pr=7,
    files=[("payments/refund.py", 3, 1), ("docs/a.md", 10, 0)],
    commits=2,
    decisions=["D-031", "D-035"],
)


def walkthrough(**changes: Any) -> Walkthrough:
    return Walkthrough.model_validate(answer(**changes))


def test_the_comment_starts_with_its_marker_and_the_program_s_facts() -> None:
    text = render_walkthrough(walkthrough(), FACTS, "zh")

    lines = text.splitlines()
    assert lines[0] == MARKER
    facts = next(line for line in lines if "PR #7" in line)
    assert "2 个文件" in facts and "+13 −1" in facts and "2 次提交" in facts
    assert "D-031 / D-035" in facts
    # The files by lines changed, most first.
    table = text.split("改动分布")[1]
    assert table.index("docs/a.md") < table.index("payments/refund.py")


def test_the_comment_shows_the_cards_and_folds_the_details() -> None:
    text = render_walkthrough(
        walkthrough(diagram="flowchart LR\n  A --> B", deploy_notes=["Run migration 0010."]),
        FACTS,
        "zh",
    )

    assert "**一句话：** Refunds are clamped to what was paid." in text
    assert "```mermaid\nflowchart LR\n  A --> B\n```" in text
    assert "<summary><b>01　为什么要改</b>（1 个问题）</summary>" in text
    assert "<summary><b>02　怎么改的</b>" in text
    assert "> [!WARNING]" in text and "Run migration 0010." in text
    # Folded sections close.
    assert text.count("<details>") == text.count("</details>")


def test_an_english_comment_has_english_headings() -> None:
    text = render_walkthrough(walkthrough(), FACTS, "en")

    assert "**In one sentence:**" in text and "Why it was changed" in text
    assert "2 files" in text and "2 commits" in text


def test_model_text_cannot_add_a_marker_or_mention_someone() -> None:
    sneaky = walkthrough(summary="Done <!-- review-agents:summary --> cc @octocat")

    text = render_walkthrough(sneaky, FACTS, "zh")

    assert text.count("<!--") == 1 and "@octocat" not in text


def test_a_walkthrough_that_could_not_be_written_says_why() -> None:
    text = render_failed("did not hand in its walkthrough by the latest submit time", "zh")

    assert text.splitlines()[0] == MARKER
    assert "改动说明没能生成" in text and "没在最晚交卷时间前交卷" in text


def test_the_facts_come_from_git_and_the_pr_text(tmp_path: Path) -> None:
    upstream = Upstream(tmp_path, "octo/shop")
    base = upstream.commit({"a.py": "x = 1\n", "b.py": "y = 1\n"}, "init")
    git(upstream.work, "checkout", "-q", "-b", "pr")
    upstream.commit({"a.py": "x = 2\nz = 3\n"}, "change a (D-031)", branch="pr")
    head = upstream.commit({"c.py": "c = 1\n"}, "add c", branch="pr")
    store = MirrorStore(tmp_path / "repos", remote_base=str(upstream.base))
    git_dir = store.sync("octo/shop", ["refs/heads/pr"], [head], None)
    layer = replace(
        pack().pr,
        request=PrRequest("octo/shop", 7, "t", "Follows D-035 and D-031.", base, head),
        files=[ChangedFile("M", "a.py"), ChangedFile("A", "c.py")],
    )

    facts = pr_facts(git_dir, layer)

    assert facts.pr == 7 and facts.commits == 2
    assert sorted(facts.files) == [("a.py", 2, 1), ("c.py", 1, 0)]
    assert facts.decisions == ["D-031", "D-035"]


def test_a_host_and_port_is_not_taken_for_a_file(repo_tools: RepoTools) -> None:
    hosts = answer(changes=[{"title": "t", "body": "Calls `127.0.0.1:9` and `localhost:8080`."}])
    model = ScriptedModel(script=[call(WALKTHROUGH_TOOL, hosts)])

    assert run(model, repo_tools).status == "done"
