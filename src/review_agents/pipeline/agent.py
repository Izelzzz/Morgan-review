"""One dimension agent: read the code with read-only tools, then submit findings.

The loop is shared with the verifier, which hands in a verdict instead of findings.

Two layers make sure an agent stops (agent tools doc §5). The tool layer refuses calls
past the limit. The loop layer offers only the submit tool once the tool calls are
used up or the deadline is close, and gives up after `MAX_TURNS_AFTER_TOOLS` more
turns than tool calls: the submit turn and one retry each for a text reply and for a
malformed submission.
A failure ends this agent only; the pipeline carries on with the other dimensions.
"""

import functools
import json
import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from review_agents.context.pack import DIMENSIONS, GENERAL, ContextPack
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

SUBMIT_TOOL = "submit_findings"
# Turns beyond the tool calls: one to submit, one retry for a reply in text, one for a
# malformed submission.
MAX_TURNS_AFTER_TOOLS = 3
# Below this much time before the deadline, the agent may only submit.
SUBMIT_MARGIN_SECONDS = 30.0
SUBMIT_NOW = (
    f"Stop investigating. Call {SUBMIT_TOOL} now with the findings you have; an empty"
    " list is a valid review."
)


Severity = Literal["blocking", "important", "robustness", "nit"]


class Finding(BaseModel):
    """One candidate finding, in the fields the shared rules define."""

    model_config = ConfigDict(extra="ignore")

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


class Submission(BaseModel):
    findings: list[Finding] = Field(description="Every finding; an empty list if none.")


@dataclass
class AgentRun:
    dimension: str
    status: Literal["done", "incomplete"]
    findings: list[Finding] = field(default_factory=list)
    # Findings the program refused, each with its title and the reason.
    dropped: list[str] = field(default_factory=list)
    reason: str | None = None
    limit_hit: Literal["time", "tokens", "tool_calls"] | None = None
    tool_calls: int = 0


def _submit_tool() -> StructuredTool:
    def submit_findings(findings: list[Finding]) -> str:
        return "received"

    return StructuredTool.from_function(
        submit_findings,
        name=SUBMIT_TOOL,
        description="Submit your findings for this dimension and finish the review.",
        args_schema=Submission,
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
class Submit[T]:
    """How an agent hands in its answer: the tool, how its arguments are read, and words."""

    tool: StructuredTool
    parse: Callable[[dict[str, Any]], T]  # raises ValueError on a malformed answer
    what: str  # what is handed in, a noun taking "was": "findings list", "verdict"
    now: str  # told once only the submit tool is left

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
) -> LoopEnd[T]:
    """Let a model read the code with read-only tools until it hands in its answer.

    `model_for` builds the model for each turn from the seconds left to the deadline, so
    no call, nor a wait before one of its retries, runs past it.
    """
    budget = ToolBudget(tool_calls)
    available = agent_tools(tools, budget)
    schemas = [StructuredTool.from_function(f, name=n) for n, f in available.items()]
    definitions = [convert_to_openai_tool(t) for t in [*schemas, submit.tool]]
    conversation = Conversation(limits, overhead=estimate_tokens(json.dumps(definitions)))
    end: LoopEnd[T] = LoopEnd()
    # One retry each: a reply in text instead of a call, and a malformed answer.
    text_retry_used = False
    format_retry_used = False
    told_to_submit = False

    for _ in range(tool_calls + MAX_TURNS_AFTER_TOOLS):
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
        offered = [submit.tool] if final else [*schemas, submit.tool]
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

        submitted: ToolCall | None = None
        for tool_call in reply.tool_calls:
            if tool_call["name"] == submit.name:
                if submitted is not None:  # every call needs an answer, the ignored ones too
                    ignored = "Ignored: only the last submission of a turn counts."
                    messages.append(ToolMessage(ignored, tool_call_id=submitted["id"]))
                submitted = tool_call
                continue
            end.tool_calls += 1
            if final:  # only submit was offered: a tool must not eat the time that is left
                only = f"Not run: only {submit.name} is available now."
                messages.append(ToolMessage(only, tool_call_id=tool_call["id"]))
            else:
                _answer(tool_call, available, budget, messages, conversation)

        if submitted is None and not reply.tool_calls:
            if text_retry_used:
                end.reason = f"the model answered in text twice instead of calling {submit.name}"
                return end
            text_retry_used = True
            messages.append(HumanMessage(f"Report your {submit.what} by calling {submit.name}."))
            continue
        if submitted is None:
            continue
        try:
            end.answer = submit.parse(submitted["args"])
        except ValueError as exc:  # ValidationError, or a check of the program's own
            if format_retry_used:
                end.reason = f"the {submit.what} was not in the expected format twice"
                return end
            format_retry_used = True
            problems = (
                "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:10])
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            messages.append(
                ToolMessage(
                    f"The {submit.what} you submitted is not in the expected format"
                    f" ({problems}). Call {submit.name} again with every required field.",
                    tool_call_id=submitted["id"],
                )
            )
            continue
        # A forced answer is complete but cut short; the report says by what.
        if budget.used >= tool_calls:
            end.limit_hit = "tool_calls"
        elif forced_by_time:
            end.limit_hit = "time"
        return end

    end.reason = (
        f"the agent did not submit its {submit.what} within"
        f" {tool_calls + MAX_TURNS_AFTER_TOOLS} turns"
    )
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
) -> AgentRun:
    """Run one dimension agent to its findings, or to a recorded reason it stopped."""
    submit = Submit(_submit_tool(), Submission.model_validate, "findings list", SUBMIT_NOW)
    end = tool_loop(
        model_for,
        list(pack.messages(dimension)),
        tools,
        submit,
        tool_calls=tool_calls,
        callbacks=callbacks,
        deadline=deadline,
        clock=clock,
        limits=limits,
        slot=slot,
    )
    run = AgentRun(
        dimension,
        "incomplete",
        reason=end.reason,
        limit_hit=end.limit_hit,
        tool_calls=end.tool_calls,
    )
    if end.answer is None:
        return run
    line_count = functools.cache(tools.line_count)
    sources = pack_sources(pack, tools)
    for item in end.answer.findings:
        # The general agent tags its findings; anything else is taken as `general`.
        if dimension != GENERAL or item.dimension not in DIMENSIONS:
            item.dimension = dimension
        try:
            # The same file must be spelled the same in every finding, as git spells
            # it: merging compares paths, and GitHub rejects others in comments.
            item.file = clean_path(item.file)
        except ToolError as exc:
            run.dropped.append(f"{item.title}: {exc}")
            continue
        # Checked, not judged: a basis that does not check out says why, for the verifier.
        item.basis = [check_basis(b, sources) for b in item.basis]
        try:
            problem = check(item, line_count)
        except ToolTimeoutError:
            # Not known to be wrong, only not checked now; the verifier reads it again.
            problem = None
        if problem is None:
            run.findings.append(item)
        else:
            run.dropped.append(f"{item.title}: {problem}")
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
