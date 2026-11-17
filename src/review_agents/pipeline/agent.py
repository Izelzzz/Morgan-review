"""One dimension agent: read the code with read-only tools, then hand in findings.

The loop is shared with the verifier, which hands in a verdict instead of findings.

A dimension agent hands in each finding with its own call (issue #108): one that does
not check out is refused alone, and the model is told why; the others stay. It then
says it is done, with how many it handed in.

Two layers make sure an agent stops (agent tools doc §5). The tool layer refuses calls
past the limit. The loop layer offers only the hand-in tools once the tool calls are
used up or the deadline is close, and gives up after a few more turns than tool calls:
the submit turn, a retry for a text reply, and the retries of each kind of malformed
answer. Turns that only hand in findings do not count, up to a limit of their own.
A failure ends this agent only; the pipeline carries on with the other dimensions.
"""

import difflib
import functools
import json
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from review_agents.context.pack import ContextPack
from review_agents.context.tokens import estimate_tokens
from review_agents.models import BudgetExceededError
from review_agents.pipeline.basis import Basis, Sources, check_basis
from review_agents.pipeline.conversation import (
    Conversation,
    ConversationLimits,
    ConversationTooLong,
)
from review_agents.tools import (
    LIMIT_REACHED,
    RepoTools,
    ToolBudget,
    ToolError,
    ToolTimeoutError,
    agent_tools,
    clean_path,
)

# Held around each model call: the review's share of the model service (decision D-031).
Slot = Callable[[], AbstractContextManager[object]]

ADD_TOOL = "add_finding"
SUBMIT_TOOL = "submit_findings"
# How many times one answer may be handed in again after each kind of mistake: arguments
# that are not JSON or were cut off, a schema not followed, a check of the program's own.
FORMAT_RETRIES = 2
Mistake = Literal["unreadable", "schema", "check"]
# A safety net, not a design limit: findings one agent may hand in.
MAX_FINDINGS = 100
# Below this much time before the deadline, the agent may only submit.
SUBMIT_MARGIN_SECONDS = 30.0
# The answer to a call whose arguments are not JSON (issue #96).
NOT_JSON_REASON = "its arguments are not valid JSON"
NOT_JSON = f"error: {NOT_JSON_REASON}; call it again with valid JSON arguments."
# A reply that ended at the output limit (issue #106): what it handed in may be whole as
# far as it goes, with the rest never written. The finish reasons services use for it.
CUT_OFF = frozenset({"length", "max_tokens", "max_output_tokens"})
CUT_OFF_REASON = "the reply was cut off at the output limit"
CUT_OFF_ANSWER = f"error: {CUT_OFF_REASON}; call it again with a shorter answer."

SUBMIT_NOW = (
    f"Stop investigating. Hand in each finding you have not handed in yet with its own"
    f" {ADD_TOOL} call, then call {SUBMIT_TOOL} with how many you handed in; none is a"
    " valid review."
)


def turns_after_tools(retries: int) -> int:
    """Turns an agent has beyond its tool calls: one to submit, one retry for a reply in
    text, one to answer a count that did not match, and the retries of each kind of
    mistake."""
    return 3 + 3 * retries


Severity = Literal["blocking", "important", "robustness", "nit"]


class Finding(BaseModel):
    """One candidate finding, in the fields the shared rules define."""

    # A misspelt field is said, not dropped: `suggestions` would leave the fix out.
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    # Set to the agent's own dimension: a model's spelling of it is not checked.
    severity: Severity
    dimension: str
    file: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    # Why it holds, as the shared rules define; required for 🔴 / 🟡 (decision D-036). The
    # program checks each one; one that does not check out still goes to the verifier.
    basis: list[Basis] = Field(default_factory=list)
    why: str = ""
    suggestion: str = ""
    preexisting: bool = False

    @field_validator("title")
    @classmethod
    def _one_line(cls, value: str) -> str:
        return " ".join(value.split())


class Done(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0, description=f"How many findings you handed in with {ADD_TOOL}.")


@dataclass
class AgentRun:
    dimension: str
    status: Literal["done", "incomplete"]
    findings: list[Finding] = field(default_factory=list)
    # Findings left out, each with its title and the reason.
    dropped: list[str] = field(default_factory=list)
    reason: str | None = None
    limit_hit: Literal["time", "tokens", "tool_calls"] | None = None
    tool_calls: int = 0


def _add_tool() -> StructuredTool:
    def add_finding(**kwargs: Any) -> str:
        return "received"

    return StructuredTool.from_function(
        add_finding,
        name=ADD_TOOL,
        description="Hand in one finding. Call it once for each finding; several calls in"
        " one turn are fine. A finding that does not check out is refused alone, with why.",
        args_schema=Finding,
    )


def _submit_tool() -> StructuredTool:
    def submit_findings(count: int) -> str:
        return "received"

    return StructuredTool.from_function(
        submit_findings,
        name=SUBMIT_TOOL,
        description=f"Finish the review once every finding is handed in with {ADD_TOOL}:"
        " say how many you handed in. None is a valid review.",
        args_schema=Done,
    )


def check(finding: Finding, line_count: Callable[[str], int | None]) -> str | None:
    """Why a finding cannot be kept, or None when it is well-formed."""
    if finding.line_end < finding.line_start:
        return "ends before it starts"
    lines = line_count(finding.file)
    if lines is None:
        return f"{finding.file} is not in the reviewed commit"
    if finding.line_end > lines:
        return f"{finding.file} has only {lines} lines"
    if not finding.why.strip():
        return "it does not say what goes wrong for users or maintainers (why)"
    if not finding.suggestion.strip():
        return "it does not suggest a fix"
    return None


@dataclass(frozen=True)
class Collect[I]:
    """Answers handed in one by one before the submit call: the tool and how each is read."""

    tool: StructuredTool
    parse: Callable[[dict[str, Any]], I]  # raises ValueError, said to the model

    @property
    def name(self) -> str:
        return self.tool.name


@dataclass(frozen=True)
class Submit[T]:
    """How an agent hands in its answer: the tool, how its arguments are read, and words."""

    tool: StructuredTool
    parse: Callable[[dict[str, Any]], T]  # raises ValueError on a malformed answer
    what: str  # what is handed in, a noun taking "was": "findings list", "verdict"
    now: str  # told once only the submit tool is left
    # Items handed in one call each before it; `parse` then reads how many there were.
    collect: Collect[Any] | None = None

    @property
    def name(self) -> str:
        return self.tool.name


@dataclass
class LoopEnd[T]:
    """How a tool loop ended: with the parsed answer, or with the reason it did not."""

    answer: T | None = None
    reason: str | None = None
    limit_hit: Literal["time", "tokens", "tool_calls"] | None = None
    tool_calls: int = 0
    # The submit call the answer came in, for a conversation that carries on after it.
    submit_id: str | None = None
    # Its tool results, so an agent carrying on can still cut them to fit its window.
    conversation: Conversation | None = None
    # Items handed in one by one and kept, and those left out with their reason.
    collected: list[Any] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def _field_names(tools: Sequence[StructuredTool]) -> list[str]:
    """Every field name in the tools' schemas, nested ones included."""

    def walk(node: object) -> Iterator[str]:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                yield from properties
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    return sorted({n for t in tools for n in walk(convert_to_openai_tool(t))})


def describe(exc: ValueError, fields: Sequence[str]) -> str:
    """What is wrong with an answer, field by field, in words a model can act on."""
    if not isinstance(exc, ValidationError):
        return str(exc)
    parts = []
    for error in exc.errors()[:10]:
        where = ".".join(map(str, error["loc"]))
        if error["type"] == "extra_forbidden":
            name = str(error["loc"][-1])
            close = difflib.get_close_matches(name, fields, n=1)
            hint = f"; did you mean `{close[0]}`?" if close else ""
            parts.append(f"{where}: `{name}` is not a field{hint}")
        else:
            parts.append(f"{where}: {error['msg']}")
    return "; ".join(parts)


def _mistake(exc: ValueError) -> Mistake:
    return "schema" if isinstance(exc, ValidationError) else "check"


def _key(args: dict[str, Any]) -> tuple[str, str]:
    """What tells one finding from another as it is handed in again: its title and file.
    A generic title is common across files; the lines may be what gets corrected."""
    title, file = args.get("title"), args.get("file")
    title = " ".join(title.split()) if isinstance(title, str) and title.strip() else "(untitled)"
    return title, file if isinstance(file, str) else ""


def tool_loop[T](
    model_for: Callable[[float], BaseChatModel],
    messages: list[BaseMessage],
    tools: RepoTools,
    submit: Submit[T],
    *,
    tool_calls: int,
    callbacks: Sequence[BaseCallbackHandler],
    deadline: float,
    clock: Callable[[], float],
    limits: ConversationLimits,
    slot: Slot = nullcontext,
    carried: Conversation | None = None,
    retries: int = FORMAT_RETRIES,
) -> LoopEnd[T]:
    """Let a model read the code with read-only tools until it hands in its answer.

    `model_for` builds the model for each turn from the seconds left to the deadline, so
    no call, nor a wait before one of its retries, runs past it. `retries` is how many
    times an answer may be handed in again after each kind of mistake.
    """
    budget = ToolBudget(tool_calls)
    available = agent_tools(tools, budget)
    schemas = [StructuredTool.from_function(f, name=n) for n, f in available.items()]
    collect = submit.collect
    hand_in = [collect.tool, submit.tool] if collect is not None else [submit.tool]
    definitions = [convert_to_openai_tool(t) for t in [*schemas, *hand_in]]
    overhead = estimate_tokens(json.dumps(definitions))
    fields = _field_names(hand_in)
    if carried is not None:
        # The messages carry on from that conversation: its results stay cuttable, under
        # this loop's window and with this loop's tool definitions. A copy of its own:
        # Deep mode's runs all carry on from the same walkthrough.
        conversation = replace(deepcopy(carried), limits=limits, overhead=overhead)
    else:
        conversation = Conversation(limits, overhead=overhead)
    end: LoopEnd[T] = LoopEnd(conversation=conversation)
    only = "Not run: only " + " and ".join(t.name for t in hand_in) + " can be called now."
    text_retry_used = False
    mistakes: dict[Mistake, int] = {"unreadable": 0, "schema": 0, "check": 0}
    told_to_submit = False
    count_checked = False
    # Items handed in one by one: refused ones by title, until resent or left out.
    failures: dict[tuple[str, str], int] = {}
    refused: dict[tuple[str, str], str] = {}
    left_out: set[tuple[str, str]] = set()
    add_calls = 0
    turns = 0
    max_turns = tool_calls + turns_after_tools(retries)

    def leave_out(key: tuple[str, str], why: str) -> None:
        left_out.add(key)
        refused.pop(key, None)
        end.dropped.append(f"{key[0]}: {why}")

    def add(args: object, unreadable: str | None) -> str:
        """Read one item handed in; what the model is told about it."""
        assert collect is not None  # noqa: S101 - only called with items to collect
        if len(end.collected) >= MAX_FINDINGS:
            return f"Not added: at most {MAX_FINDINGS} can be handed in. Call {submit.name}."
        if unreadable is not None or not isinstance(args, dict):
            # Nothing to tell it by: each one is answered, and a count that does not
            # match says what never came.
            why = unreadable or NOT_JSON_REASON
            return f"Not added ({why}). Call {collect.name} again; the ones added are kept."
        key = _key(args)
        if key in left_out:
            return "Not added: this one was already left out after too many tries."
        try:
            item = collect.parse(args)
        except ValueError as exc:
            why = describe(exc, fields)
        else:
            end.collected.append(item)
            refused.pop(key, None)
            failures.pop(key, None)
            return f"Added ({len(end.collected)} so far)."
        failures[key] = failures.get(key, 0) + 1
        if failures[key] > retries:
            failures.pop(key)
            leave_out(key, why)
            return f"Left out after {retries + 1} tries ({why}). Go on with the others."
        refused[key] = why
        return (
            f"Not added ({why}). Fix it and call {collect.name} again; the ones already"
            " added are kept."
        )

    while turns < max_turns:
        now = clock()
        if now >= deadline:
            end.reason, end.limit_hit = "the review's deadline passed", "time"
            return end
        forced_by_time = deadline - now < SUBMIT_MARGIN_SECONDS
        final = budget.used >= tool_calls or forced_by_time
        if final and not told_to_submit:
            messages.append(HumanMessage(submit.now))
            told_to_submit = True
        try:
            conversation.fit(messages)
        except ConversationTooLong as exc:  # sending it would only fail at the service
            end.reason = str(exc)
            return end
        offered = hand_in if final else [*schemas, *hand_in]
        # The review's model calls are shared out by priority (D-031); the time spent
        # waiting for one counts, so the time left is measured once it is ours.
        with slot():
            left = deadline - clock()
            if left <= 0:
                end.reason, end.limit_hit = "the review's deadline passed", "time"
                return end
            try:
                reply = (
                    model_for(left)
                    .bind_tools(offered)
                    .invoke(messages, config={"callbacks": list(callbacks)})
                )
            except BudgetExceededError as exc:
                end.reason, end.limit_hit = str(exc), "tokens"
                return end
            except Exception as exc:  # one agent failing must not fail the review
                end.reason = f"{type(exc).__name__}: {exc}"
                return end
        if not isinstance(reply, AIMessage):
            end.reason = "the model returned no message"
            return end
        messages.append(reply)
        cut_off = reply.response_metadata.get("finish_reason") in CUT_OFF
        names = [c["name"] for c in reply.tool_calls] + [
            b.get("name") for b in reply.invalid_tool_calls
        ]
        # A turn that only hands in items is free, up to a limit of its own: an agent
        # with forty findings may hand in one a turn.
        if not (collect is not None and names and set(names) == {collect.name}) or (
            add_calls + len(names) > MAX_FINDINGS * (retries + 1)
        ):
            turns += 1

        submitted: ToolCall | None = None
        for tool_call in reply.tool_calls:
            if tool_call["name"] == submit.name:
                if submitted is not None:  # every call needs an answer, the ignored ones too
                    ignored = "Ignored: only the last submission of a turn counts."
                    messages.append(ToolMessage(ignored, tool_call_id=submitted["id"]))
                submitted = tool_call
                continue
            if collect is not None and tool_call["name"] == collect.name:
                add_calls += 1
                told = add(tool_call["args"], None)
                messages.append(ToolMessage(told, tool_call_id=tool_call["id"]))
                continue
            end.tool_calls += 1
            if final:  # only submit was offered: a tool must not eat the time that is left
                messages.append(ToolMessage(only, tool_call_id=tool_call["id"]))
            else:
                _answer(tool_call, available, budget, messages, conversation)
        # Calls whose arguments are not JSON land here, not in tool_calls, yet they go back
        # to the service with the reply: one left unanswered makes it refuse the next
        # request (issue #96). An unreadable submission is a malformed one.
        unreadable: str | None = None
        answered_bad = False
        for bad in reply.invalid_tool_calls:
            call_id = bad.get("id")
            if call_id is None:  # nothing to answer it by; services always give one
                continue
            answered_bad = True
            if bad.get("name") == submit.name and submitted is None and unreadable is None:
                unreadable = call_id
                continue
            if collect is not None and bad.get("name") == collect.name:
                add_calls += 1
                why = CUT_OFF_REASON if cut_off else NOT_JSON_REASON
                messages.append(ToolMessage(add(None, why), tool_call_id=call_id))
                continue
            answer = CUT_OFF_ANSWER if cut_off else NOT_JSON
            if bad.get("name") != submit.name:
                end.tool_calls += 1
                if final:  # as for a readable call: retrying the tool would not run it
                    answer = only
                elif budget.used < budget.limit:
                    budget.used += 1  # counted like a call with wrong arguments
            messages.append(ToolMessage(answer, tool_call_id=call_id))

        if not reply.tool_calls and not answered_bad:  # no call that got an answer
            if text_retry_used:
                end.reason = f"the model answered in text twice instead of calling {submit.name}"
                return end
            text_retry_used = True
            messages.append(HumanMessage(f"Report your {submit.what} by calling {submit.name}."))
            continue
        if submitted is None and unreadable is None:
            continue
        reply_to = submitted["id"] if submitted is not None else unreadable
        problems: str | None = None
        mistake: Mistake = "unreadable"
        if cut_off:  # not taken, however whole it reads
            problems = CUT_OFF_REASON
        elif submitted is None:
            problems = NOT_JSON_REASON
        else:
            try:
                end.answer = submit.parse(submitted["args"])
            except ValueError as exc:  # ValidationError, or a check of the program's own
                problems, mistake = describe(exc, fields), _mistake(exc)
        if problems is not None:
            end.answer = None
            # Said in the reason too, so a dimension that failed can be looked into.
            if mistakes[mistake] >= retries:
                end.reason = (
                    f"the {submit.what} was not in the expected format"
                    f" {mistakes[mistake] + 1} times: {problems}"
                )
                return end
            mistakes[mistake] += 1
            again = (
                f"Call {submit.name} again with a shorter {submit.what}: fewer or briefer entries."
                if cut_off
                else f"Call {submit.name} again with every required field."
            )
            messages.append(
                ToolMessage(
                    f"The {submit.what} you submitted is not in the expected format"
                    f" ({problems}). {again}",
                    tool_call_id=reply_to,
                )
            )
            continue
        assert submitted is not None  # noqa: S101 - parsed above
        if collect is not None:
            said = cast(int, end.answer)  # the count, with items handed in one by one
            kept = len(end.collected)
            if said != kept and not count_checked:
                # Asked once: a finding refused and never sent again, or one not sent.
                count_checked = True
                end.answer = None
                waiting = "".join(f"\n- {t}: {why}" for (t, _), why in refused.items())
                messages.append(
                    ToolMessage(
                        f"Not finished: you said {said}, and {kept} were added."
                        + (f" Refused and not sent again:{waiting}" if waiting else "")
                        + f"\nHand in what is missing with {collect.name}, or call"
                        f" {submit.name} with count {kept} to finish with what was added.",
                        tool_call_id=submitted["id"],
                    )
                )
                continue
            for key, why in list(refused.items()):
                leave_out(key, f"refused and not sent again: {why}")
            if said != kept:
                end.dropped.append(f"(count): said {said}, {kept} were added")
        end.submit_id = submitted["id"]
        # A forced answer is complete but cut short; the report says by what.
        if budget.used >= tool_calls:
            end.limit_hit = "tool_calls"
        elif forced_by_time:
            end.limit_hit = "time"
        return end

    end.reason = f"the agent did not submit its {submit.what} within {max_turns} turns"
    end.limit_hit = "tool_calls"
    return end


def pack_sources(pack: ContextPack, tools: RepoTools) -> Sources:
    """What a basis may quote: the reviewed commit, the PR description, its linked issues."""
    request = pack.pr.request
    return Sources(
        text=functools.cache(tools.text),  # a timeout raises, so it is never remembered
        pr_text=f"{request.title}\n\n{request.body}",
        issues=dict(pack.pr.issues),
    )


def run_dimension(
    model_for: Callable[[float], BaseChatModel],
    pack: ContextPack,
    dimension: str,
    tools: RepoTools,
    *,
    tool_calls: int,
    callbacks: Sequence[BaseCallbackHandler],
    deadline: float = math.inf,
    clock: Callable[[], float],
    limits: ConversationLimits = ConversationLimits(),  # noqa: B008 - frozen, shared safely
    slot: Slot = nullcontext,
    dimensions: tuple[str, ...] | None = None,
    opening: Sequence[BaseMessage] | None = None,
    carried: Conversation | None = None,
    retries: int = FORMAT_RETRIES,
) -> AgentRun:
    """Run one agent to its findings, or to a recorded reason it stopped.

    `dimensions` are the rules it holds: its own dimension unless given (a general agent
    holds several, D-038). Given an `opening`, the walkthrough's conversation once it is
    written, the agent carries on from it with its own task, so the code it read and
    what it wrote are already in front of it.
    """
    held = dimensions or (dimension,)
    messages = list(pack.messages(dimension, held))
    if opening is not None:
        messages = [*opening, messages[-1]]
    line_count = functools.cache(tools.line_count)
    sources = pack_sources(pack, tools)

    def one(args: dict[str, Any]) -> Finding:
        """A finding as kept, or ValueError with why it cannot be: said to the model."""
        item = Finding.model_validate(args)
        # A general agent tags each finding; a tag outside what it holds is not trusted.
        if len(held) == 1 or item.dimension not in held:
            item.dimension = dimension
        try:
            # The same file must be spelled the same in every finding, as git spells
            # it: merging compares paths, and GitHub rejects others in comments.
            item.file = clean_path(item.file)
        except ToolError as exc:
            raise ValueError(str(exc)) from None
        # Checked, not judged: a basis that does not check out says why, for the verifier.
        item.basis = [check_basis(b, sources) for b in item.basis]
        try:
            problem = check(item, line_count)
        except ToolTimeoutError:
            # Not known to be wrong, only not checked now; the verifier reads it again.
            problem = None
        if problem is not None:
            raise ValueError(problem)
        return item

    submit = Submit(
        _submit_tool(),
        lambda args: Done.model_validate(args).count,
        "findings count",
        SUBMIT_NOW,
        collect=Collect(_add_tool(), one),
    )
    end = tool_loop(
        model_for,
        messages,
        tools,
        submit,
        tool_calls=tool_calls,
        callbacks=callbacks,
        deadline=deadline,
        clock=clock,
        limits=limits,
        slot=slot,
        carried=carried if opening is not None else None,
        retries=retries,
    )
    run = AgentRun(
        dimension,
        "incomplete",
        dropped=list(end.dropped),
        reason=end.reason,
        limit_hit=end.limit_hit,
        tool_calls=end.tool_calls,
    )
    # Kept however the agent ended: each one passed every check (review on PR-B).
    run.findings = list(end.collected)
    if end.answer is not None:
        run.status = "done"
    return run


def _answer(
    tool_call: ToolCall,
    available: dict[str, Callable[..., str]],
    budget: ToolBudget,
    messages: list[BaseMessage],
    conversation: Conversation,
) -> None:
    name, call_id = tool_call["name"], tool_call["id"]
    tool = available.get(name)
    if tool is None:
        # Counted like any refused call, so a model stuck on a misspelt name still runs
        # out of calls and is told to submit.
        if budget.used >= budget.limit:
            messages.append(ToolMessage(LIMIT_REACHED, tool_call_id=call_id))
            return
        budget.used += 1
        messages.append(
            ToolMessage(f"error: there is no tool named {name!r}", tool_call_id=call_id)
        )
        return
    try:
        # Called through the budget wrapper, so a call with wrong arguments still counts.
        result = tool(**tool_call["args"])
    except Exception as exc:  # wrong arguments, or a tool that broke: this call fails only
        messages.append(ToolMessage(f"error: {exc}", tool_call_id=call_id))
        return
    conversation.answer(messages, call_id, name, tool_call["args"], result)
