from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from gitrepo import Upstream
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from review_agents.context.pack import SHARED_RULES, ContextPack
from review_agents.context.pr_layer import PrLayer, PrRequest
from review_agents.context.repo_layer import RepoLayer
from review_agents.context.tokens import estimate_tokens
from review_agents.mirror import MirrorStore
from review_agents.models import MAX_RETRIES, REQUEST_TIMEOUT_SECONDS, BudgetExceededError
from review_agents.pipeline.agent import SUBMIT_TOOL, AgentRun, run_dimension
from review_agents.pipeline.conversation import (
    REPEATED,
    ConversationLimits,
    conversation_tokens,
    shorten,
)
from review_agents.tools import RepoTools

FILES: dict[str, str | bytes] = {
    "payments/refund.py": "def refund(order, cents):\n    total = cents\n    return total\n",
    "README.md": "# Shop\n",
}


class ScriptedModel(BaseChatModel):
    """Replies from a script; records what it was sent and which tools it could call."""

    script: list[Any]  # AIMessage replies, or exceptions to raise
    sent: list[list[BaseMessage]] = Field(default_factory=list)
    bound: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedModel":
        self.bound.append([t.name for t in tools])
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.sent.append(list(messages))
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


def call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def finding(**changes: Any) -> dict[str, Any]:
    base = {
        "title": "A refund can exceed the paid amount",
        "severity": "important",
        "dimension": "edge_cases",
        "file": "payments/refund.py",
        "line_start": 2,
        "line_end": 3,
        "evidence": "refund(order, 10_000) on an order paid 100 returns 10_000",
        "why": "Customers are refunded more than they paid.",
        "suggestion": "Clamp to order.paid.",
        "preexisting": False,
    }
    return {**base, **changes}


def submit(*findings: dict[str, Any], call_id: str = "s1") -> AIMessage:
    return call(SUBMIT_TOOL, {"findings": list(findings)}, call_id)


@pytest.fixture
def repo_tools(tmp_path: Path) -> RepoTools:
    upstream = Upstream(tmp_path, "octo/shop")
    head = upstream.commit(FILES, "init")
    store = MirrorStore(tmp_path / "repos", remote_base=str(upstream.base))
    git_dir = store.sync("octo/shop", ["refs/heads/main"], [head], None)
    return RepoTools(git_dir, "octo/shop", head, [])


def pack() -> ContextPack:
    sha = "a" * 40
    repo = RepoLayer(commit=sha, conventions=[], tree="./ (2 files)", modules=[])
    pr = PrLayer(
        request=PrRequest(
            "octo/shop", 7, "Clamp refunds", "Never refund more than paid.", sha, sha
        ),
        issues=[],
        specs=[],
        files=[],
        diffs={"payments/refund.py": "+    total = cents\n"},
        excerpts=[],
        previous_findings=[],
    )
    return ContextPack(SHARED_RULES, repo, pr)


def run(
    model: ScriptedModel,
    repo_tools: RepoTools,
    *,
    limit: int = 25,
    now: Callable[[], float] | None = None,
    deadline: float = float("inf"),
) -> AgentRun:
    return run_dimension(
        lambda timeout: model,
        pack(),
        "edge_cases",
        repo_tools,
        tool_calls=limit,
        callbacks=[],
        deadline=deadline,
        clock=now or (lambda: 0.0),
    )


def test_the_agent_reads_code_with_its_tools_then_submits_findings(repo_tools: RepoTools) -> None:
    model = ScriptedModel(
        script=[call("read_file", {"path": "payments/refund.py"}), submit(finding())]
    )

    result = run(model, repo_tools)

    assert result.status == "done"
    assert [f.title for f in result.findings] == ["A refund can exceed the paid amount"]
    tool_reply = model.sent[1][-1]
    assert isinstance(tool_reply, ToolMessage)
    assert "def refund(order, cents):" in str(tool_reply.content)
    assert '<untrusted source="tool:read_file">' in str(tool_reply.content)
    assert result.tool_calls == 1


def test_every_dimension_starts_from_the_shared_prefix(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[submit()])

    run(model, repo_tools)

    first = model.sent[0]
    assert [m.content for m in first] == [m.content for m in pack().messages("edge_cases")]


def test_an_empty_submission_is_a_valid_review(repo_tools: RepoTools) -> None:
    result = run(ScriptedModel(script=[submit()]), repo_tools)

    assert result.status == "done" and result.findings == []


def test_once_the_tool_budget_is_spent_only_submitting_remains(repo_tools: RepoTools) -> None:
    reads = [call("list_dir", {}, f"c{i}") for i in range(3)]
    model = ScriptedModel(script=[*reads, submit(finding())])

    result = run(model, repo_tools, limit=2)

    assert result.status == "done"
    assert model.bound[-1] == [SUBMIT_TOOL]  # the last turn could not call a tool
    assert "only submit" in str(model.sent[3][-1].content)  # the third call was not run
    assert result.tool_calls == 3


def test_the_loop_ends_even_if_the_model_never_submits(repo_tools: RepoTools) -> None:
    endless = [call("list_dir", {}, f"c{i}") for i in range(20)]
    model = ScriptedModel(script=endless)

    result = run(model, repo_tools, limit=2)

    assert result.status == "incomplete"
    assert "did not submit" in (result.reason or "")
    assert len(model.sent) <= 2 + 3  # the tool limit, the submit turn and two retries


def test_findings_that_point_nowhere_are_dropped_with_a_reason(repo_tools: RepoTools) -> None:
    model = ScriptedModel(
        script=[
            submit(
                finding(),
                finding(title="Missing file", file="payments/gone.py"),
                finding(title="Past the end", line_start=2, line_end=99),
                finding(title="No evidence", severity="blocking", evidence=""),
            )
        ]
    )

    result = run(model, repo_tools)

    assert [f.title for f in result.findings] == ["A refund can exceed the paid amount"]
    reasons = " ".join(result.dropped)
    for title in ("Missing file", "Past the end", "No evidence"):
        assert title in reasons


def test_an_unreadable_submission_gets_one_retry(repo_tools: RepoTools) -> None:
    broken = call(SUBMIT_TOOL, {"findings": [{"title": "only a title"}]})
    model = ScriptedModel(script=[broken, submit(finding(), call_id="s2")])

    result = run(model, repo_tools)

    assert result.status == "done" and len(result.findings) == 1
    assert "not in the expected format" in str(model.sent[1][-1].content)


def test_a_second_unreadable_submission_leaves_the_dimension_incomplete(
    repo_tools: RepoTools,
) -> None:
    broken = call(SUBMIT_TOOL, {"findings": "none"})
    model = ScriptedModel(script=[broken, broken])

    result = run(model, repo_tools)

    assert result.status == "incomplete"
    assert "format" in (result.reason or "")


def test_a_failing_model_ends_only_this_dimension(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[RuntimeError("model service unavailable")])

    result = run(model, repo_tools)

    assert result.status == "incomplete"
    assert "model service unavailable" in (result.reason or "")


def test_a_spent_review_budget_stops_the_dimension(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[BudgetExceededError("token budget of 10 used up")])

    result = run(model, repo_tools)

    assert result.status == "incomplete"
    assert result.limit_hit == "tokens"


def test_near_the_deadline_the_agent_must_submit(repo_tools: RepoTools) -> None:
    times = iter([0.0, 100.0, 100.0, 100.0])
    model = ScriptedModel(
        script=[call("list_dir", {}), submit(finding())],
    )

    result = run(model, repo_tools, now=lambda: next(times), deadline=110.0)

    assert result.status == "done"
    assert model.bound[-1] == [SUBMIT_TOOL]


def test_past_the_deadline_no_model_call_is_made(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[submit(finding())])

    result = run(model, repo_tools, now=lambda: 50.0, deadline=10.0)

    assert result.status == "incomplete"
    assert result.limit_hit == "time"
    assert model.sent == []


def test_a_tool_called_with_wrong_arguments_answers_with_an_error(
    repo_tools: RepoTools,
) -> None:
    model = ScriptedModel(script=[call("read_file", {"no_such_argument": 1}), submit()])

    result = run(model, repo_tools)

    assert result.status == "done"
    assert str(model.sent[1][-1].content).startswith("error:")
    assert result.tool_calls == 1


def test_findings_belong_to_the_agent_dimension_and_a_clean_path(repo_tools: RepoTools) -> None:
    model = ScriptedModel(
        script=[submit(finding(dimension="Edge cases", file="/payments/refund.py"))]
    )

    [kept] = run(model, repo_tools).findings

    assert kept.dimension == "edge_cases"
    assert kept.file == "payments/refund.py"


def test_a_tool_that_breaks_answers_with_an_error_instead_of_ending_the_review(
    repo_tools: RepoTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: RepoTools, path: str = "") -> str:
        """List a directory."""
        raise OSError("mirror vanished")

    monkeypatch.setattr(RepoTools, "list_dir", broken)
    model = ScriptedModel(script=[call("list_dir", {}), submit()])

    result = run(model, repo_tools)

    assert result.status == "done"
    assert "mirror vanished" in str(model.sent[1][-1].content)


def test_on_the_final_turn_tools_that_were_not_offered_are_not_run(
    repo_tools: RepoTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[str] = []

    def recorded(self: RepoTools, path: str = "") -> str:
        """List a directory."""
        ran.append(path)
        return ""

    monkeypatch.setattr(RepoTools, "list_dir", recorded)
    model = ScriptedModel(script=[call("list_dir", {}), submit(call_id="s2")])

    # 20 seconds before the deadline: inside the submit-only margin from the start.
    result = run(model, repo_tools, now=lambda: 0.0, deadline=20.0)

    assert ran == []
    assert "only submit" in str(model.sent[1][-1].content)
    assert result.status == "done"


def test_a_dimension_stopped_by_its_tool_limit_says_so(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[call("list_dir", {}, "c1"), call("list_dir", {}, "c2"), submit()])

    result = run(model, repo_tools, limit=2)

    assert result.status == "done" and result.limit_hit == "tool_calls"


def test_a_dimension_that_finished_early_hit_no_limit(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[call("list_dir", {}), submit()])

    assert run(model, repo_tools, limit=5).limit_hit is None


def test_a_finding_without_its_consequence_or_fix_is_dropped(repo_tools: RepoTools) -> None:
    model = ScriptedModel(
        script=[
            submit(
                finding(), finding(title="No why", why=" "), finding(title="No fix", suggestion="")
            )
        ]
    )

    result = run(model, repo_tools)

    assert [f.title for f in result.findings] == ["A refund can exceed the paid amount"]
    reasons = " ".join(result.dropped)
    assert "No why" in reasons and "No fix" in reasons


def test_every_submit_call_in_a_turn_gets_an_answer(repo_tools: RepoTools) -> None:
    two = AIMessage(
        content="",
        tool_calls=[
            {"name": SUBMIT_TOOL, "args": {"findings": []}, "id": "s1"},
            {"name": SUBMIT_TOOL, "args": {"findings": "broken"}, "id": "s2"},
        ],
    )
    model = ScriptedModel(script=[two, submit(call_id="s3")])

    result = run(model, repo_tools)

    assert result.status == "done"
    answered = {m.tool_call_id for m in model.sent[1] if isinstance(m, ToolMessage)}
    assert answered == {"s1", "s2"}  # the ignored one and the malformed one


def test_a_text_reply_does_not_use_up_the_format_retry(repo_tools: RepoTools) -> None:
    broken = call(SUBMIT_TOOL, {"findings": "none"})
    model = ScriptedModel(
        script=[AIMessage(content="I found nothing."), broken, submit(call_id="s2")]
    )

    assert run(model, repo_tools).status == "done"


def test_a_title_is_kept_on_one_line(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[submit(finding(title="Refund\n  exceeds paid"))])

    [kept] = run(model, repo_tools).findings

    assert kept.title == "Refund exceeds paid"


@pytest.mark.parametrize(
    "written", ["./payments/refund.py", "payments//refund.py", "/payments/refund.py"]
)
def test_a_path_is_stored_the_way_the_repository_spells_it(
    repo_tools: RepoTools, written: str
) -> None:
    [kept] = run(ScriptedModel(script=[submit(finding(file=written))]), repo_tools).findings

    assert kept.file == "payments/refund.py"


def test_a_path_outside_the_repository_is_dropped(repo_tools: RepoTools) -> None:
    result = run(ScriptedModel(script=[submit(finding(file="../payments/refund.py"))]), repo_tools)

    assert result.findings == []
    assert "inside the repository" in " ".join(result.dropped)


def test_calling_a_tool_that_does_not_exist_uses_the_budget(repo_tools: RepoTools) -> None:
    typos = [call("read_files", {}, f"c{i}") for i in range(2)]
    model = ScriptedModel(script=[*typos, submit()])

    result = run(model, repo_tools, limit=2)

    assert result.status == "done"
    assert model.bound[-1] == [SUBMIT_TOOL]  # the budget ran out, so only submit remained


def test_a_submission_forced_by_the_deadline_says_so(repo_tools: RepoTools) -> None:
    result = run(ScriptedModel(script=[submit()]), repo_tools, now=lambda: 100.0, deadline=110.0)

    assert result.status == "done" and result.limit_hit == "time"


def test_both_retries_still_fit_after_the_tools_are_used_up(repo_tools: RepoTools) -> None:
    model = ScriptedModel(
        script=[
            call("read_file", {"path": "payments/refund.py"}),
            AIMessage(content="Nothing to report."),
            call(SUBMIT_TOOL, {"findings": "none"}, "bad"),
            submit(finding(), call_id="good"),
        ]
    )

    result = run(model, repo_tools, limit=1)

    assert result.status == "done" and len(result.findings) == 1


def test_each_model_call_is_bounded_by_the_time_left(repo_tools: RepoTools) -> None:
    timeouts: list[float] = []
    model = ScriptedModel(script=[call("list_dir", {}), submit()])
    times = iter([0.0, 990.0])

    def model_for(timeout: float) -> ScriptedModel:
        timeouts.append(timeout)
        return model

    run_dimension(
        model_for,
        pack(),
        "edge_cases",
        repo_tools,
        tool_calls=25,
        callbacks=[],
        deadline=1000.0,
        clock=lambda: next(times),
    )

    # Early on a call may take its full limit; 10 seconds before the deadline, the call
    # and its retries together fit in those 10 seconds.
    assert timeouts[0] == REQUEST_TIMEOUT_SECONDS
    assert timeouts[1] * (MAX_RETRIES + 1) <= 10.0


LONG = "\n".join(f"value_{n} = compute({n}, factor={n * 7})" for n in range(1, 3001)) + "\n"


@pytest.fixture
def long_tools(tmp_path: Path) -> RepoTools:
    upstream = Upstream(tmp_path, "octo/shop")
    head = upstream.commit({**FILES, "big/table.py": LONG}, "init")
    store = MirrorStore(tmp_path / "repos", remote_base=str(upstream.base))
    git_dir = store.sync("octo/shop", ["refs/heads/main"], [head], None)
    return RepoTools(git_dir, "octo/shop", head, [])


def run_limited(model: ScriptedModel, tools: RepoTools, limits: ConversationLimits) -> AgentRun:
    return run_dimension(
        lambda left: model,
        pack(),
        "edge_cases",
        tools,
        tool_calls=25,
        callbacks=[],
        deadline=float("inf"),
        clock=lambda: 0.0,
        limits=limits,
    )


def tool_texts(model: ScriptedModel) -> list[str]:
    return [str(m.content) for m in model.sent[-1] if isinstance(m, ToolMessage)]


def test_a_long_tool_result_keeps_its_head_and_tail_and_says_how_to_read_the_rest(
    long_tools: RepoTools,
) -> None:
    model = ScriptedModel(script=[call("read_file", {"path": "big/table.py"}), submit()])

    run_limited(model, long_tools, ConversationLimits(result_tokens=1000))

    [text] = tool_texts(model)
    assert estimate_tokens(text) <= 1200
    assert "1: value_1 = " in text and "3000: value_3000 = " in text
    assert "1500: value_1500 = " not in text
    assert "start_line" in text  # how to read what was left out
    assert text.rstrip().endswith("</untrusted>")  # still inside its data block


def test_older_results_shrink_when_the_conversation_nears_the_window(
    long_tools: RepoTools,
) -> None:
    reads = [
        call("read_file", {"path": "big/table.py", "start_line": s, "end_line": s + 199}, f"c{s}")
        for s in (1, 1001, 2001)
    ]
    model = ScriptedModel(script=[*reads, submit()])
    limits = ConversationLimits(
        window=9000, result_tokens=4000, shortened_tokens=100, output_reserve=500
    )

    run_limited(model, long_tools, limits)

    sent = model.sent[-1]
    assert conversation_tokens(sent) + 500 <= 9000
    first, *_, last = tool_texts(model)
    assert "Shortened" in first  # the oldest result gave way
    assert "Shortened" not in last and "2200: value_2200" in last
    assert last.rstrip().endswith("</untrusted>") and first.rstrip().endswith("</untrusted>")


def test_the_third_identical_call_is_reminded_to_change_course(repo_tools: RepoTools) -> None:
    same = [call("list_dir", {}, f"c{n}") for n in range(3)]
    model = ScriptedModel(script=[*same, call("list_dir", {"path": "payments"}, "c9"), submit()])

    run_limited(model, repo_tools, ConversationLimits())

    texts = tool_texts(model)
    assert [REPEATED in t for t in texts] == [False, False, True, False]


def test_a_result_of_a_few_very_long_lines_is_still_cut_to_its_limit() -> None:
    minified = "\n".join(["x" * 30_000, "y" * 30_000])  # two lines, about 20k tokens

    text = shorten(minified, 500, "read_file")

    assert estimate_tokens(text) <= 1200
    assert text.startswith("[Shortened") and "x" in text and "y" in text


def test_a_cut_by_characters_stays_within_its_limit_in_any_script() -> None:
    chinese = "\n".join(["退款金额" * 5000, "订单状态" * 5000])  # about 1 token a character

    text = shorten(chinese, 500, "search_code")

    assert estimate_tokens(text) <= 1200
