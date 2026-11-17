from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from gitrepo import Upstream
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from scripted import ScriptedModel, add, call, done, hand_in

from review_agents.context.pack import SHARED_RULES, ContextPack
from review_agents.context.pr_layer import PrLayer, PrRequest
from review_agents.context.repo_layer import RepoLayer
from review_agents.context.tokens import estimate_tokens
from review_agents.mirror import MirrorStore
from review_agents.models import BudgetExceededError
from review_agents.pipeline.agent import (
    ADD_TOOL,
    FORMAT_RETRIES,
    SUBMIT_TOOL,
    AgentRun,
    run_dimension,
    turns_after_tools,
)
from review_agents.pipeline.conversation import (
    REMOVED,
    REPEATED,
    Conversation,
    ConversationLimits,
    conversation_tokens,
    shorten,
)
from review_agents.tools import RepoTools, ToolTimeoutError

FILES: dict[str, str | bytes] = {
    "payments/refund.py": "def refund(order, cents):\n    total = cents\n    return total\n",
    "README.md": "# Shop\n\nRefunds never exceed the amount paid.\n",
}


BASIS: dict[str, Any] = {
    "kind": "spec",
    "source": "pr",
    "quote": "Never refund more than paid.",
    "places": [{"file": "payments/refund.py", "line_start": 2, "line_end": 3, "shows": "no cap"}],
    "explanation": "The requested amount is refunded whatever was paid.",
}


def finding(**changes: Any) -> dict[str, Any]:
    base = {
        "title": "A refund can exceed the paid amount",
        "severity": "important",
        "dimension": "edge_cases",
        "file": "payments/refund.py",
        "line_start": 2,
        "line_end": 3,
        "basis": [BASIS],
        "why": "Customers are refunded more than they paid.",
        "suggestion": "Clamp to order.paid.",
        "preexisting": False,
    }
    return {**base, **changes}


def submit(*findings: dict[str, Any], call_id: str = "s1") -> AIMessage:
    return hand_in(*findings, call_id=call_id)


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
    assert model.bound[-1] == [ADD_TOOL, SUBMIT_TOOL]  # the last turn could not call a tool
    assert f"only {ADD_TOOL} and {SUBMIT_TOOL} can be called" in str(
        model.sent[3][-1].content
    )  # the third call was not run
    assert result.tool_calls == 3


def test_the_loop_ends_even_if_the_model_never_submits(repo_tools: RepoTools) -> None:
    endless = [call("list_dir", {}, f"c{i}") for i in range(20)]
    model = ScriptedModel(script=endless)

    result = run(model, repo_tools, limit=2)

    assert result.status == "incomplete"
    assert "did not submit" in (result.reason or "")
    # The tool limit, the submit turn, a text retry and the retries of each kind of mistake.
    assert len(model.sent) <= 2 + turns_after_tools(FORMAT_RETRIES)


def test_findings_that_point_nowhere_are_dropped_with_a_reason(repo_tools: RepoTools) -> None:
    model = ScriptedModel(
        script=[
            submit(
                finding(),
                finding(title="Missing file", file="payments/gone.py"),
                finding(title="Past the end", line_start=2, line_end=99),
            ),
            done(1),
        ]
    )

    result = run(model, repo_tools)

    assert [f.title for f in result.findings] == ["A refund can exceed the paid amount"]
    reasons = " ".join(result.dropped)
    for title in ("Missing file", "Past the end"):
        assert title in reasons


def test_an_unreadable_submission_gets_one_retry(repo_tools: RepoTools) -> None:
    broken = call(SUBMIT_TOOL, {"findings": [{"title": "only a title"}]})
    model = ScriptedModel(script=[broken, submit(finding(), call_id="s2")])

    result = run(model, repo_tools)

    assert result.status == "done" and len(result.findings) == 1
    assert "not in the expected format" in str(model.sent[1][-1].content)


def test_an_unreadable_submission_is_asked_again_up_to_its_retries(
    repo_tools: RepoTools,
) -> None:
    broken = call(SUBMIT_TOOL, {"count": "none"})
    model = ScriptedModel(script=[broken] * (FORMAT_RETRIES + 1))

    result = run(model, repo_tools)

    assert result.status == "incomplete"
    assert f"not in the expected format {FORMAT_RETRIES + 1} times" in (result.reason or "")


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
    assert model.bound[-1] == [ADD_TOOL, SUBMIT_TOOL]


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
    assert f"only {ADD_TOOL} and {SUBMIT_TOOL} can be called" in str(model.sent[1][-1].content)
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
            ),
            done(1),
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


def unparsable(name: str, call_id: str = "bad1") -> AIMessage:
    """A call whose arguments are not JSON: LangChain files it under invalid_tool_calls."""
    return AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": name, "args": '{"findings": [{"title": "cut', "id": call_id, "error": None}
        ],
    )


def test_a_call_whose_arguments_are_not_json_gets_an_answer(repo_tools: RepoTools) -> None:
    both = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "payments/refund.py"}, "id": "c1"}],
        invalid_tool_calls=[{"name": "read_file", "args": '{"path": ', "id": "c2", "error": None}],
    )
    model = ScriptedModel(script=[both, submit()])

    result = run(model, repo_tools)

    # The service refuses a request with a call left unanswered (issue #96).
    assert result.status == "done"
    answered = {m.tool_call_id: m for m in model.sent[1] if isinstance(m, ToolMessage)}
    assert set(answered) == {"c1", "c2"}
    assert "not valid JSON" in str(answered["c2"].content)
    assert result.tool_calls == 2  # counted like a call with wrong arguments


def test_a_turn_of_only_unparsable_calls_is_not_a_text_answer(repo_tools: RepoTools) -> None:
    model = ScriptedModel(
        script=[unparsable("read_file"), unparsable("read_file", "bad2"), submit()]
    )

    result = run(model, repo_tools)

    # Each call was answered, so the model is asked nothing more (review nit on PR #97).
    assert result.status == "done"
    asked = [sum(isinstance(m, HumanMessage) for m in sent) for sent in model.sent]
    assert asked[0] == asked[2]
    assert result.tool_calls == 2


def test_an_unparsable_call_in_a_final_turn_is_not_run(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[unparsable("read_file"), submit()])

    run(model, repo_tools, limit=0)

    answer = model.sent[1][-1]
    assert isinstance(answer, ToolMessage) and answer.tool_call_id == "bad1"
    assert f"only {ADD_TOOL} and {SUBMIT_TOOL} can be called" in str(answer.content)


def test_an_unparsable_call_without_an_id_counts_as_a_text_answer(repo_tools: RepoTools) -> None:
    no_id = AIMessage(
        content="",
        invalid_tool_calls=[{"name": "read_file", "args": "{", "id": None, "error": None}],
    )
    model = ScriptedModel(script=[no_id, submit()])

    run(model, repo_tools)

    assert f"by calling {SUBMIT_TOOL}" in str(model.sent[1][-1].content)


def test_an_unparsable_submission_is_answered_and_retried(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[unparsable(SUBMIT_TOOL), submit(finding(), call_id="s2")])

    result = run(model, repo_tools)

    assert result.status == "done" and len(result.findings) == 1
    last = model.sent[1][-1]
    assert isinstance(last, ToolMessage) and last.tool_call_id == "bad1"
    assert "not valid JSON" in str(last.content)


def test_two_unparsable_submissions_end_as_a_format_problem(repo_tools: RepoTools) -> None:
    bad = [unparsable(SUBMIT_TOOL, f"bad{n}") for n in range(FORMAT_RETRIES + 1)]
    model = ScriptedModel(script=bad)

    result = run(model, repo_tools)

    assert result.status == "incomplete"
    assert f"not in the expected format {FORMAT_RETRIES + 1} times" in (result.reason or "")
    assert "not valid JSON" in (result.reason or "")


def test_a_second_malformed_submission_says_what_was_wrong(repo_tools: RepoTools) -> None:
    broken = call(SUBMIT_TOOL, {"count": "none"})
    model = ScriptedModel(script=[broken] * (FORMAT_RETRIES + 1))

    result = run(model, repo_tools)

    # Recorded in the review's steps, so a failed agent can be looked into.
    assert "count: Input should be a valid integer" in (result.reason or "")


def cut_off(reply: AIMessage) -> AIMessage:
    """The same reply, ended by the output limit."""
    return reply.model_copy(update={"response_metadata": {"finish_reason": "length"}})


def test_a_submission_cut_off_at_the_output_limit_is_asked_again(repo_tools: RepoTools) -> None:
    # Whole as far as it goes: the rest of the findings were never written.
    # The finding added in that turn was sent whole; only finishing is not taken.
    model = ScriptedModel(script=[cut_off(submit(finding())), hand_in(finding(title="B"), count=2)])

    result = run(model, repo_tools)

    assert result.status == "done" and len(result.findings) == 2
    answer = model.sent[1][-1]
    assert isinstance(answer, ToolMessage) and answer.tool_call_id == "s1"
    assert "cut off at the output limit" in str(answer.content)


def test_two_cut_off_submissions_end_the_dimension_and_say_why(repo_tools: RepoTools) -> None:
    model = ScriptedModel(
        script=[cut_off(submit(call_id=f"s{n}")) for n in range(FORMAT_RETRIES + 1)]
    )

    result = run(model, repo_tools)

    assert result.status == "incomplete"
    assert "cut off at the output limit" in (result.reason or "")


def test_an_unreadable_call_in_a_cut_off_reply_is_told_it_was_cut_off(
    repo_tools: RepoTools,
) -> None:
    model = ScriptedModel(script=[cut_off(unparsable("read_file")), submit()])

    run(model, repo_tools)

    answer = model.sent[1][-1]
    assert isinstance(answer, ToolMessage) and answer.tool_call_id == "bad1"
    assert "cut off at the output limit" in str(answer.content)


def test_an_unreadable_submission_in_a_cut_off_reply_is_told_it_was_cut_off(
    repo_tools: RepoTools,
) -> None:
    model = ScriptedModel(script=[cut_off(unparsable(SUBMIT_TOOL)), submit(call_id="s2")])

    assert run(model, repo_tools).status == "done"
    assert "cut off at the output limit" in str(model.sent[1][-1].content)


def test_a_reply_that_ended_normally_is_kept(repo_tools: RepoTools) -> None:
    ended = submit(finding()).model_copy(update={"response_metadata": {"finish_reason": "stop"}})

    assert len(run(ScriptedModel(script=[ended]), repo_tools).findings) == 1


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
    outside = finding(file="../payments/refund.py")
    model = ScriptedModel(script=[submit(outside), done(0)])

    result = run(model, repo_tools)

    assert result.findings == []
    assert "inside the repository" in " ".join(result.dropped)


def test_calling_a_tool_that_does_not_exist_uses_the_budget(repo_tools: RepoTools) -> None:
    typos = [call("read_files", {}, f"c{i}") for i in range(2)]
    model = ScriptedModel(script=[*typos, submit()])

    result = run(model, repo_tools, limit=2)

    assert result.status == "done"
    assert model.bound[-1] == [ADD_TOOL, SUBMIT_TOOL]  # the budget ran out, so only submit remained


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
    # Read before each turn, and again once its model call has a slot.
    times = iter([0.0, 0.0, 990.0, 990.0])

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

    # Each turn's model gets the time to the deadline; its retries never wait past it.
    assert timeouts == [1000.0, 10.0]


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


def test_results_give_way_entirely_when_shortening_them_is_not_enough(
    long_tools: RepoTools,
) -> None:
    reads = [
        call("read_file", {"path": "big/table.py", "start_line": s, "end_line": s + 199}, f"c{s}")
        for s in (1, 1001, 2001)
    ]
    model = ScriptedModel(script=[*reads, submit()])
    # Room for the pack and about one shortened result, not three.
    limits = ConversationLimits(
        window=5200, result_tokens=4000, shortened_tokens=600, output_reserve=500
    )

    result = run_limited(model, long_tools, limits)

    assert result.status == "done"
    for sent in model.sent:  # no request ever exceeded the window
        assert conversation_tokens(sent) + 500 <= 5200
    first, *_ = tool_texts(model)
    assert first.startswith(REMOVED)


def test_an_agent_whose_conversation_cannot_fit_stops_instead_of_sending_it(
    repo_tools: RepoTools,
) -> None:
    model = ScriptedModel(script=[submit()])
    # The context pack alone is larger than the window leaves.
    limits = ConversationLimits(window=600, output_reserve=500)

    result = run_limited(model, repo_tools, limits)

    assert result.status == "incomplete"
    assert result.reason is not None and "window" in result.reason
    assert model.sent == []


def test_the_tool_definitions_count_against_the_window() -> None:
    # The tool definitions go with every request, so they take room like the messages do.
    text = "\n".join(f"line {n}" for n in range(400))
    messages: list[BaseMessage] = []
    Conversation(ConversationLimits()).answer(messages, "c1", "read_file", {}, text)
    size = conversation_tokens(messages)
    limits = ConversationLimits(window=size + 100, shortened_tokens=20, output_reserve=0)
    messages = []
    conversation = Conversation(limits, overhead=500)
    conversation.answer(messages, "c1", "read_file", {}, text)

    conversation.fit(messages)

    assert conversation_tokens(messages) + 500 <= size + 100
    assert "Shortened" in str(messages[0].content)


def test_a_call_whose_result_was_removed_may_be_made_again_without_a_reminder() -> None:
    big = "\n".join(f"line {n}" for n in range(400))
    # Room for two removal notes, not for two shortened results.
    limits = ConversationLimits(
        window=2 * estimate_tokens(REMOVED) + 5, shortened_tokens=20, output_reserve=0
    )
    conversation = Conversation(limits)
    messages: list[BaseMessage] = []
    for n in range(2):
        conversation.answer(messages, f"c{n}", "list_dir", {}, big)

    conversation.fit(messages)
    assert [str(m.content) for m in messages] == [REMOVED, REMOVED]
    conversation.answer(messages, "c2", "list_dir", {}, "payments/")

    # The agent was told to call again for what was removed; that is not repeating itself.
    assert REPEATED not in str(messages[-1].content)


def test_an_agent_counts_its_tool_definitions_before_sending(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[submit()])
    # Room for the context pack's messages, not for them and the tool definitions.
    pack_size = conversation_tokens(pack().messages("edge_cases"))
    limits = ConversationLimits(window=pack_size + 10, output_reserve=0)

    result = run_limited(model, repo_tools, limits)

    assert result.status == "incomplete" and model.sent == []


def test_a_finding_whose_place_cannot_be_checked_in_time_is_kept(
    repo_tools: RepoTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow(path: str) -> int | None:
        raise ToolTimeoutError("git cat-file timed out after 30s")

    monkeypatch.setattr(repo_tools, "line_count", slow)
    model = ScriptedModel(script=[submit(finding())])

    result = run(model, repo_tools)

    # Not known to be wrong, only not checked: the verifier reads it again later.
    assert [f.title for f in result.findings] == ["A refund can exceed the paid amount"]
    assert result.dropped == []


def test_each_model_call_waits_for_a_slot(repo_tools: RepoTools) -> None:
    entered: list[int] = []

    @contextmanager
    def slot() -> Iterator[None]:
        entered.append(len(model.sent))
        yield

    model = ScriptedModel(script=[call("list_dir", {}), submit()])

    run_dimension(
        lambda left: model,
        pack(),
        "edge_cases",
        repo_tools,
        tool_calls=25,
        callbacks=[],
        clock=lambda: 0.0,
        slot=slot,
    )

    assert entered == [0, 1]  # each call is made inside its slot


def test_time_spent_waiting_for_a_slot_counts_against_the_deadline(
    repo_tools: RepoTools,
) -> None:
    now = [0.0]

    @contextmanager
    def slow_slot() -> Iterator[None]:
        now[0] += 100.0  # waited behind calls of a higher priority
        yield

    model = ScriptedModel(script=[submit()])

    result = run_dimension(
        lambda left: model,
        pack(),
        "edge_cases",
        repo_tools,
        tool_calls=25,
        callbacks=[],
        deadline=60.0,
        clock=lambda: now[0],
        slot=slow_slot,
    )

    assert result.status == "incomplete" and result.limit_hit == "time"
    assert model.sent == []


def test_every_basis_is_checked_and_says_how_it_checked_out(repo_tools: RepoTools) -> None:
    moved = dict(BASIS, source="README.md:9", quote="Refunds never exceed the amount paid.")
    wrong = dict(BASIS, source="docs/policy.md:3")
    model = ScriptedModel(script=[submit(finding(basis=[BASIS, moved, wrong]))])

    result = run(model, repo_tools)

    [kept] = result.findings
    statuses = [(b.check.status if b.check else None) for b in kept.basis]
    assert statuses == ["ok", "corrected", "failed"]
    assert kept.basis[1].source == "README.md:3"  # the quote's real line
    problem = kept.basis[2].check.problem if kept.basis[2].check else ""
    assert "docs/policy.md is not in the reviewed commit" in (problem or "")


def test_a_finding_whose_basis_does_not_check_out_is_kept_for_the_verifier(
    repo_tools: RepoTools,
) -> None:
    # Decision D-036: the dimension agent does not submit again; the verifier decides.
    unbased = finding(title="No basis", severity="blocking", basis=[])
    misquoted = finding(title="Misquoted", basis=[dict(BASIS, quote="Refund at most once.")])
    model = ScriptedModel(script=[submit(unbased, misquoted)])

    result = run(model, repo_tools)

    assert [f.title for f in result.findings] == ["No basis", "Misquoted"]
    assert result.dropped == []
    assert len(model.sent) == 1  # not asked to submit again


def test_a_model_cannot_mark_its_own_basis_as_checked(repo_tools: RepoTools) -> None:
    claimed = dict(BASIS, source="docs/policy.md:3", check={"status": "ok"})
    model = ScriptedModel(script=[submit(finding(basis=[claimed]))])

    result = run(model, repo_tools)

    [kept] = result.findings
    assert kept.basis[0].check is not None and kept.basis[0].check.status == "failed"


def test_one_bad_finding_is_refused_alone_and_the_others_are_kept(repo_tools: RepoTools) -> None:
    bad = finding(title="Bad level", severity="huge")
    model = ScriptedModel(
        script=[
            hand_in(finding(), bad, finding(title="Third")),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": ADD_TOOL, "args": finding(title="Bad level"), "id": "a2"},
                    {"name": SUBMIT_TOOL, "args": {"count": 3}, "id": "d2"},
                ],
            ),
        ]
    )

    result = run(model, repo_tools)

    assert result.status == "done"
    titles = [f.title for f in result.findings]
    assert titles == ["A refund can exceed the paid amount", "Third", "Bad level"]
    told = {m.tool_call_id: str(m.content) for m in model.sent[1] if isinstance(m, ToolMessage)}
    assert told["s1.2"].startswith("Not added (severity:")
    assert told["s1.1"] == "Added (1 so far)." and told["s1.3"] == "Added (2 so far)."
    assert "you said 3, and 2 were added" in told["s1"] and "Bad level" in told["s1"]


def test_a_finding_refused_too_often_is_left_out_and_the_others_kept(
    repo_tools: RepoTools,
) -> None:
    bad = finding(title="Nowhere", line_end=99)
    tries = [add(bad, f"a{n}") for n in range(FORMAT_RETRIES)]
    model = ScriptedModel(script=[hand_in(finding(), bad), *tries, done(1)])

    result = run(model, repo_tools)

    assert result.status == "done" and len(result.findings) == 1
    assert result.dropped == ["Nowhere: payments/refund.py has only 3 lines"]
    last = model.sent[-1][-1]
    assert isinstance(last, ToolMessage)
    assert str(last.content).startswith(f"Left out after {FORMAT_RETRIES + 1} tries")


def test_a_misspelt_field_is_named_with_the_one_meant(repo_tools: RepoTools) -> None:
    misspelt = finding()
    misspelt["suggestions"] = misspelt.pop("suggestion")
    model = ScriptedModel(script=[add(misspelt), hand_in(finding())])

    result = run(model, repo_tools)

    assert result.status == "done" and len(result.findings) == 1
    told = str(model.sent[1][-1].content)
    assert "`suggestions` is not a field; did you mean `suggestion`?" in told


def test_finishing_with_the_wrong_count_is_asked_once(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[hand_in(finding(), count=2), done(2)])

    result = run(model, repo_tools)

    assert result.status == "done" and len(result.findings) == 1
    assert "you said 2, and 1 were added" in str(model.sent[1][-1].content)
    assert result.dropped == ["(count): said 2, 1 were added"]


def test_forty_five_findings_handed_in_one_a_turn_all_fit(repo_tools: RepoTools) -> None:
    adds = [add(finding(title=f"Finding {n}"), f"a{n}") for n in range(45)]
    model = ScriptedModel(script=[call("list_dir", {}), *adds, done(45)])

    result = run(model, repo_tools, limit=2)

    assert result.status == "done" and len(result.findings) == 45


def test_with_the_tools_used_up_findings_can_still_be_handed_in(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[call("list_dir", {}), add(finding()), done(1)])

    result = run(model, repo_tools, limit=1)

    assert result.status == "done" and len(result.findings) == 1
    assert model.bound[1:] == [[ADD_TOOL, SUBMIT_TOOL]] * 2


def test_no_retries_ends_at_the_first_malformed_answer(repo_tools: RepoTools) -> None:
    model = ScriptedModel(script=[call(SUBMIT_TOOL, {"count": "none"})])

    result = run_dimension(
        lambda timeout: model,
        pack(),
        "edge_cases",
        repo_tools,
        tool_calls=5,
        callbacks=[],
        clock=lambda: 0.0,
        retries=0,
    )

    assert result.status == "incomplete"
    assert "not in the expected format 1 times" in (result.reason or "")


def test_the_task_says_how_findings_are_handed_in() -> None:
    task = str(pack().messages("edge_cases")[-1].content)

    assert f"`{ADD_TOOL}`" in task and f"`{SUBMIT_TOOL}`" in task
