"""The walkthrough: the first comment, explaining a pull request (walkthrough design, D-035).

A walkthrough agent writes it alongside the dimension agents, with the same opening and
the same priority, and hands it in by the same latest submit time. It is not verified:
the program checks every `path:line` it cites and how many diagrams it draws, and asks
once for a correction. The program adds the facts it knows itself (files, lines added
and removed, commits, decisions) and lays the comment out.
"""

import functools
import math
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from review_agents.context.pack import LANGUAGE_NAMES, ContextPack
from review_agents.context.pr_layer import GIT_TIMEOUT_SECONDS, PrLayer
from review_agents.mirror import run_git
from review_agents.pipeline.agent import Slot, Submit, tool_loop
from review_agents.pipeline.conversation import ConversationLimits
from review_agents.repo_config import Language
from review_agents.report import defused
from review_agents.tools import RepoTools, ToolTimeoutError

MARKER = "<!-- review-agents:walkthrough -->"
WALKTHROUGH_TOOL = "submit_walkthrough"
MAX_DIAGRAMS = 3
SKILL = (files("review_agents") / "skills" / "_walkthrough.md").read_text(encoding="utf-8")
WALKTHROUGH_TASK = (
    "You are now writing the walkthrough of this pull request, not reviewing it: the"
    " findings and output rules above do not apply to you. Follow these instructions"
    f" and hand in with `{WALKTHROUGH_TOOL}`.\n\n{SKILL}"
)
WALKTHROUGH_NOW = f"Stop investigating. Call {WALKTHROUGH_TOOL} now with what you know."
# A citation in code font: `path:line`, `path:start-end`, or GitHub's `path#L12-L20`.
# A name with neither a folder nor a dot is checked only when the reviewed commit has
# such a file: `Dockerfile:999999` is caught, while `postgres:16`, `node:20` or `03:30`
# are an image tag and a time. An IP address and port is not a file either.
_REFERENCE = re.compile(r"`([^`\s#]+?)(?::(\d+)(?:-(\d+))?|#L(\d+)(?:-L(\d+))?)`")
_HOST = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
# The walkthrough describes; it needs fewer reads than a dimension agent, and what it
# spends is taken from the review's shared budget.
WALKTHROUGH_TOOL_CALLS = 20
# Git too slow to say whether a cited file is there gets one more try.
CHECK_TRIES = 2


class Card(BaseModel):
    headline: str = Field(min_length=1, description="One sentence.")
    points: list[str] = Field(min_length=2, max_length=4, description="One line each.")


class Problem(BaseModel):
    title: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class Change(BaseModel):
    title: str = Field(min_length=1)
    body: str = Field(min_length=1, description="Markdown: a table, a Mermaid diagram, a list.")


class Impact(BaseModel):
    aspect: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Tradeoff(BaseModel):
    situation: str = Field(min_length=1)
    choice: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class Walkthrough(BaseModel):
    summary: str = Field(min_length=1, description="The whole change in one sentence.")
    why: Card
    what: Card
    impact: Card
    diagram: str = Field(default="", description="Optional: Mermaid source, without fences.")
    problems: list[Problem] = Field(min_length=2, max_length=5)
    changes: list[Change] = Field(min_length=1)
    impacts: list[Impact] = Field(default_factory=list)
    deploy_notes: list[str] = Field(default_factory=list)
    tradeoffs: list[Tradeoff] = Field(default_factory=list)

    def texts(self) -> Iterator[str]:
        """Every text the agent wrote."""

        def walk(value: object) -> Iterator[str]:
            if isinstance(value, str):
                yield value
            elif isinstance(value, BaseModel):
                for item in value.__dict__.values():
                    yield from walk(item)
            elif isinstance(value, list):
                for item in value:
                    yield from walk(item)

        yield from walk(self)

    @property
    def diagrams(self) -> int:
        fenced = sum(text.count("```mermaid") for text in self.texts())
        return fenced + (1 if self.diagram.strip() and "```mermaid" not in self.diagram else 0)


@dataclass
class WalkthroughRun:
    status: Literal["done", "incomplete"]
    walkthrough: Walkthrough | None = None
    reason: str | None = None
    limit_hit: Literal["time", "tokens", "tool_calls"] | None = None
    # Experiment: the conversation once the walkthrough is in, for the general agent to
    # carry on from.
    transcript: list[BaseMessage] | None = None
    tool_calls: int = 0


def reference_problems(
    walkthrough: Walkthrough, line_count: Callable[[str], int | None]
) -> list[str]:
    """What is wrong with the places the walkthrough cites.

    Raises ToolTimeoutError when git is still too slow on the last try: a place that
    could not be checked is not taken for one that checked out.
    """
    problems = []
    seen = set()
    for text in walkthrough.texts():
        for match in _REFERENCE.finditer(text):
            path = match.group(1)
            start = int(match.group(2) or match.group(4))
            end = int(match.group(3) or match.group(5) or start)
            if "://" in path or _HOST.fullmatch(path) or (path, start, end) in seen:
                continue
            seen.add((path, start, end))
            lines = _counted(line_count, path)
            if lines is None and "/" not in path and "." not in path:
                continue  # an image tag, a time: no such file, so not a citation of one
            cited = match.group(0).strip("`")
            if start < 1:
                problems.append(f"{cited} has no line 0")
                continue
            if end < start:
                problems.append(f"{cited} ends before it starts")
                continue
            if lines is None:
                problems.append(f"{path} is not in the reviewed commit")
            elif end > lines:
                problems.append(f"{path} has only {lines} lines, not {end}")
    return problems


class _Unchecked(RuntimeError):
    """Git was too slow to check a cited place: not the agent's mistake to correct."""


def _counted(line_count: Callable[[str], int | None], path: str) -> int | None:
    for tries in range(1, CHECK_TRIES + 1):
        try:
            return line_count(path)
        except ToolTimeoutError:
            if tries == CHECK_TRIES:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


def _submit_tool() -> StructuredTool:
    def submit_walkthrough(**kwargs: Any) -> str:
        return "received"

    return StructuredTool.from_function(
        submit_walkthrough,
        name=WALKTHROUGH_TOOL,
        description="Submit the walkthrough of the pull request, and finish.",
        args_schema=Walkthrough,
    )


def walkthrough_messages(pack: ContextPack) -> list[BaseMessage]:
    """The dimension agents' opening, then the walkthrough's task in the review's language."""
    opening = pack.messages("security")[:2]
    language = (
        f"Write the walkthrough in {LANGUAGE_NAMES[pack.language]}. Keep file paths, code and"
        " identifiers as they are."
    )
    return [*opening, HumanMessage(f"{WALKTHROUGH_TASK}\n\n{language}")]


def run_walkthrough(
    model_for: Callable[[float], BaseChatModel],
    pack: ContextPack,
    tools: RepoTools,
    *,
    tool_calls: int,
    callbacks: Sequence[BaseCallbackHandler],
    deadline: float = math.inf,
    clock: Callable[[], float],
    limits: ConversationLimits = ConversationLimits(),  # noqa: B008 - frozen, shared safely
    slot: Slot = nullcontext,
) -> WalkthroughRun:
    """Write the walkthrough, or record why it could not be written."""
    line_count = functools.cache(tools.line_count)  # a timeout raises, so it is not kept

    def parse(args: dict[str, Any]) -> Walkthrough:
        walkthrough = Walkthrough.model_validate(args)
        try:
            problems = reference_problems(walkthrough, line_count)
        except ToolTimeoutError as exc:  # a ValueError, which the loop takes for a mistake
            raise _Unchecked(str(exc)) from exc
        if walkthrough.diagrams > MAX_DIAGRAMS:
            problems.append(f"{walkthrough.diagrams} Mermaid diagrams, more than {MAX_DIAGRAMS}")
        if problems:
            raise ValueError("; ".join(problems))
        return walkthrough

    messages = walkthrough_messages(pack)
    try:
        end = tool_loop(
            model_for,
            messages,
            tools,
            Submit(_submit_tool(), parse, "walkthrough", WALKTHROUGH_NOW),
            tool_calls=tool_calls,
            callbacks=callbacks,
            deadline=deadline,
            clock=clock,
            limits=limits,
            slot=slot,
        )
    except _Unchecked as exc:
        return WalkthroughRun("incomplete", reason=f"its references could not be checked: {exc}")
    received = (
        [*messages, ToolMessage("received", tool_call_id=end.submit_id)]
        if end.answer is not None and end.submit_id is not None
        else None
    )
    return WalkthroughRun(
        "done" if end.answer is not None else "incomplete",
        walkthrough=end.answer,
        reason=end.reason,
        limit_hit=end.limit_hit,
        tool_calls=end.tool_calls,
        transcript=received,
    )


@dataclass(frozen=True)
class PrFacts:
    """What the program knows of a PR itself, for the walkthrough's header and table."""

    pr: int
    files: list[tuple[str, int, int]] = field(default_factory=list)  # path, added, removed
    commits: int = 0
    decisions: list[str] = field(default_factory=list)


_DECISION = re.compile(r"\bD-\d{3}\b")


def pr_facts(git_dir: Path, layer: PrLayer) -> PrFacts:
    """Files with lines added and removed, commits and decisions named, over the whole PR."""
    request = layer.request
    span = f"{request.base_sha}...{request.head_sha}"

    def git(*args: str) -> str:
        return run_git(git_dir, *args, timeout=GIT_TIMEOUT_SECONDS).decode("utf-8", "replace")

    numstat = git("diff", "--numstat", "-z", "-M", "--no-ext-diff", span).split("\0")
    changed = []
    i = 0
    while i < len(numstat) and numstat[i]:
        added, removed, path = numstat[i].split("\t", 2)
        if not path:  # a rename: the old and the new path follow
            path, i = numstat[i + 2], i + 3
        else:
            i += 1
        # A binary file has no line counts.
        changed.append(
            (path, int(added) if added != "-" else 0, int(removed) if removed != "-" else 0)
        )
    count = git("rev-list", "--count", f"{request.base_sha}..{request.head_sha}").strip()
    messages = git("log", "--format=%B", f"{request.base_sha}..{request.head_sha}")
    named = _DECISION.findall(f"{request.title}\n{request.body}\n{messages}")
    return PrFacts(request.pr, changed, int(count or 0), sorted(set(named)))


_TEXT: dict[Language, dict[str, str]] = {
    "zh": {
        "title": "## 📖 改动说明",
        "facts": "PR #{pr} · {files} 个文件 · **+{added} −{removed}** · {commits} 次提交",
        "decisions": " · 决策 {names}",
        "summary": "**一句话：** ",
        "why": "为什么改",
        "what": "改了什么",
        "impact": "有什么影响",
        "problems": "<summary><b>01　为什么要改</b>（{n} 个问题）</summary>",
        "changes": "<summary><b>02　怎么改的</b>（{n} 处改动）</summary>",
        "impacts": "#### 有什么影响",
        "aspect": "| 方面 | 影响 |",
        "deploy": "**部署时注意**",
        "tradeoffs": "<summary><b>取舍</b>（{n} 条）</summary>",
        "tradeoff_head": "| 情况 | 这次的做法 | 理由 |",
        "files": "<summary><b>改动分布</b>（{n} 个文件）</summary>",
        "files_head": "| 文件 | 增 | 删 |",
        "failed": "改动说明没能生成：{reason}",
        "did not hand in its walkthrough by the latest submit time": "没在最晚交卷时间前交卷",
        "not started before the latest submit time": "到最晚交卷时间还没开始",
    },
    "en": {
        "title": "## 📖 Walkthrough",
        "facts": "PR #{pr} · {files} files · **+{added} −{removed}** · {commits} commits",
        "decisions": " · decisions {names}",
        "summary": "**In one sentence:** ",
        "why": "Why it was changed",
        "what": "What changed",
        "impact": "What it affects",
        "problems": "<summary><b>01　Why it was changed</b> ({n} problems)</summary>",
        "changes": "<summary><b>02　How it was changed</b> ({n} changes)</summary>",
        "impacts": "#### What it affects",
        "aspect": "| Aspect | Effect |",
        "deploy": "**Before deploying**",
        "tradeoffs": "<summary><b>Trade-offs</b> ({n})</summary>",
        "tradeoff_head": "| Situation | What this change does | Why |",
        "files": "<summary><b>Files changed</b> ({n} files)</summary>",
        "files_head": "| File | Added | Removed |",
        "failed": "The walkthrough could not be written: {reason}",
        "did not hand in its walkthrough by the latest submit time": "it was not handed in by"
        " the latest submit time",
        "not started before the latest submit time": "it had not started by the latest submit time",
    },
}


def _cell(text: str) -> str:
    return defused(" ".join(text.split())).replace("|", "\\|")


def _details(summary: str, body: list[str]) -> list[str]:
    return ["<details>", summary, "", *body, "", "</details>", ""]


def render_walkthrough(walkthrough: Walkthrough, facts: PrFacts, language: Language) -> str:
    """The walkthrough comment, in the order of the PR #66 sample (D-035)."""
    t = _TEXT[language]
    w = walkthrough
    added = sum(a for _, a, _ in facts.files)
    removed = sum(r for _, _, r in facts.files)
    head = t["facts"].format(
        pr=facts.pr, files=len(facts.files), added=added, removed=removed, commits=facts.commits
    )
    if facts.decisions:
        head += t["decisions"].format(names=" / ".join(facts.decisions))
    lines = [
        MARKER,
        t["title"],
        "",
        head,
        "",
        t["summary"] + defused(" ".join(w.summary.split())),
        "",
    ]
    for key, card in (("why", w.why), ("what", w.what), ("impact", w.impact)):
        lines.append(f"#### {t[key]} — {_cell(card.headline)}")
        lines += [f"- {defused(' '.join(p.split()))}" for p in card.points]
        lines.append("")
    if w.diagram.strip():
        source = w.diagram.strip()
        if not source.startswith("```"):
            source = f"```mermaid\n{source}\n```"
        lines += [defused(source), ""]
    problems = [
        f"{n}. **{_cell(p.title)}** — {defused(' '.join(p.explanation.split()))}"
        for n, p in enumerate(w.problems, start=1)
    ]
    lines += _details(t["problems"].format(n=len(w.problems)), problems)
    changes: list[str] = []
    for n, change in enumerate(w.changes, start=1):
        changes += [f"**{n}. {_cell(change.title)}**", "", defused(change.body.strip()), ""]
    lines += _details(t["changes"].format(n=len(w.changes)), changes[:-1])
    if w.impacts:
        lines += [t["impacts"], "", t["aspect"], "| --- | --- |"]
        lines += [f"| {_cell(i.aspect)} | {_cell(i.text)} |" for i in w.impacts]
        lines.append("")
    if w.deploy_notes:
        lines += ["> [!WARNING]", f"> {t['deploy']}"]
        lines += [f"> - {_cell(note)}" for note in w.deploy_notes]
        lines.append("")
    if w.tradeoffs:
        rows = [t["tradeoff_head"], "| --- | --- | --- |"]
        rows += [
            f"| {_cell(x.situation)} | {_cell(x.choice)} | {_cell(x.reason)} |" for x in w.tradeoffs
        ]
        lines += _details(t["tradeoffs"].format(n=len(w.tradeoffs)), rows)
    ordered = sorted(facts.files, key=lambda f: (-(f[1] + f[2]), f[0]))
    rows = [t["files_head"], "| --- | ---: | ---: |"]
    rows += [f"| `{_cell(path)}` | {a} | {r} |" for path, a, r in ordered]
    lines += _details(t["files"].format(n=len(facts.files)), rows)
    return "\n".join(lines).rstrip() + "\n"


def render_failed(reason: str, language: Language) -> str:
    """What replaces the walkthrough when none could be written: the last one is outdated."""
    t = _TEXT[language]
    shown = t.get(reason) or _cell(reason)
    return "\n".join([MARKER, t["title"], "", t["failed"].format(reason=shown)]) + "\n"
