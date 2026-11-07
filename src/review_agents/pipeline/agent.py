"""One dimension agent: read the code with read-only tools, then submit findings.

Two layers make sure an agent stops (agent tools doc §5). The tool layer refuses calls
past the limit. The loop layer offers only the submit tool once the tool calls are
used up or the deadline is close, and gives up after `MAX_TURNS_AFTER_TOOLS` more
turns than tool calls: the submit turn and one retry each for a text reply and for a
malformed submission.
A failure ends this agent only; the pipeline carries on with the other dimensions.
"""

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from review_agents.context.pack import ContextPack
from review_agents.models import MAX_RETRIES, REQUEST_TIMEOUT_SECONDS, BudgetExceededError
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
    agent_tools,
    clean_path,
)

SUBMIT_TOOL = "submit_findings"
# Turns beyond the tool calls: one to submit, one retry for a reply in text, one for a
# malformed submission.
MAX_TURNS_AFTER_TOOLS = 3
# Below this much time before the deadline, the agent may only submit.
SUBMIT_MARGIN_SECONDS = 30.0
SUBMIT_ONLY = f"Not run: only {SUBMIT_TOOL} is available now."
SUBMIT_IGNORED = "Ignored: only the last submission of a turn counts."
SUBMIT_NOW = (
    f"Stop investigating. Call {SUBMIT_TOOL} now with the findings you have; an empty"
    " list is a valid review."
)


class Finding(BaseModel):
    """One candidate finding, in the fields the shared rules define."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1)
    # Set to the agent's own dimension: a model's spelling of it is not checked.
    severity: Literal["blocking", "important", "robustness", "nit"]
    dimension: str
    file: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    evidence: str = ""
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
    if finding.severity in ("blocking", "important") and not finding.evidence.strip():
        return "a blocking or important finding needs evidence"
    if not finding.why.strip():
        return "it does not say what goes wrong for users or maintainers (why)"
    if not finding.suggestion.strip():
        return "it does not suggest a fix"
    return None


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
) -> AgentRun:
    """Run one dimension agent to its findings, or to a recorded reason it stopped.

    `model_for` builds the model for each turn from the seconds that call may take, so
    no call, with its retries, runs past the deadline.
    """
    budget = ToolBudget(tool_calls)
    conversation = Conversation(limits)
    available = agent_tools(tools, budget)
    schemas = [StructuredTool.from_function(f, name=n) for n, f in available.items()]
    submit = _submit_tool()
    messages: list[BaseMessage] = list(pack.messages(dimension))
    run = AgentRun(dimension, "incomplete")
    # One retry each: a reply in text instead of a call, and a malformed submission.
    text_retry_used = False
    format_retry_used = False
    told_to_submit = False

    for _ in range(tool_calls + MAX_TURNS_AFTER_TOOLS):
        now = clock()
        if now >= deadline:
            run.reason, run.limit_hit = "the review's deadline passed", "time"
            return run
        forced_by_time = deadline - now < SUBMIT_MARGIN_SECONDS
        final = budget.used >= tool_calls or forced_by_time
        if final and not told_to_submit:
            messages.append(HumanMessage(SUBMIT_NOW))
            told_to_submit = True
        # The client retries a failed call, so the time left is shared by every attempt.
        timeout = max(1.0, min(REQUEST_TIMEOUT_SECONDS, (deadline - now) / (MAX_RETRIES + 1)))
        try:
            conversation.fit(messages)
        except ConversationTooLong as exc:  # sending it would only fail at the service
            run.reason = str(exc)
            return run
        bound = model_for(timeout).bind_tools([submit] if final else [*schemas, submit])
        try:
            reply = bound.invoke(messages, config={"callbacks": list(callbacks)})
        except BudgetExceededError as exc:
            run.reason, run.limit_hit = str(exc), "tokens"
            return run
        except Exception as exc:  # one agent failing must not fail the review
            run.reason = f"{type(exc).__name__}: {exc}"
            return run
        if not isinstance(reply, AIMessage):
            run.reason = "the model returned no message"
            return run
        messages.append(reply)

        submitted: ToolCall | None = None
        for tool_call in reply.tool_calls:
            if tool_call["name"] == SUBMIT_TOOL:
                if submitted is not None:  # every call needs an answer, the ignored ones too
                    messages.append(ToolMessage(SUBMIT_IGNORED, tool_call_id=submitted["id"]))
                submitted = tool_call
                continue
            run.tool_calls += 1
            if final:  # only submit was offered: a tool must not eat the time that is left
                messages.append(ToolMessage(SUBMIT_ONLY, tool_call_id=tool_call["id"]))
            else:
                _answer(tool_call, available, budget, messages, conversation)

        if submitted is None and not reply.tool_calls:
            if text_retry_used:
                run.reason = f"the model answered in text twice instead of calling {SUBMIT_TOOL}"
                return run
            text_retry_used = True
            messages.append(HumanMessage(f"Report your findings by calling {SUBMIT_TOOL}."))
            continue
        if submitted is None:
            continue
        try:
            submission = Submission.model_validate(submitted["args"])
        except ValidationError as exc:
            if format_retry_used:
                run.reason = "the findings were not in the expected format twice"
                return run
            format_retry_used = True
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:10]
            )
            messages.append(
                ToolMessage(
                    f"The findings are not in the expected format ({problems}). Call"
                    f" {SUBMIT_TOOL} again with every required field.",
                    tool_call_id=submitted["id"],
                )
            )
            continue
        line_count = functools.cache(tools.line_count)
        for item in submission.findings:
            item.dimension = dimension
            try:
                # The same file must be spelled the same in every finding, as git spells
                # it: merging compares paths, and GitHub rejects others in comments.
                item.file = clean_path(item.file)
            except ToolError as exc:
                run.dropped.append(f"{item.title}: {exc}")
                continue
            problem = check(item, line_count)
            if problem is None:
                run.findings.append(item)
            else:
                run.dropped.append(f"{item.title}: {problem}")
        run.status = "done"
        # A forced submission is complete but cut short; the report says by what.
        if budget.used >= tool_calls:
            run.limit_hit = "tool_calls"
        elif forced_by_time:
            run.limit_hit = "time"
        return run

    run.reason = (
        f"the agent did not submit findings within {tool_calls + MAX_TURNS_AFTER_TOOLS} turns"
    )
    run.limit_hit = "tool_calls"
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
