"""The basis of a finding: why it holds, in a form a reader and the program can check (D-036).

A basis is one of four kinds. `spec`, `contract` and `convention` quote a written rule
exactly as written in its source and name the code that breaks it; `path` gives the steps
from a caller or input to the failure, each with its lines. The program checks every
basis: the files and lines exist, and the quote is in its source. A quote that is in its
source file at other lines gets those lines; a basis that does not check out says why
and goes to the verifier like any other, which decides whether the finding holds.
"""

import bisect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from review_agents.tools import ToolError, ToolTimeoutError, clean_path

# A quote is a sentence or a few lines, not a section copied in (decision D-036).
MAX_QUOTE_CHARS = 300
Kind = Literal["spec", "contract", "convention", "path"]
Status = Literal["ok", "corrected", "failed", "unchecked"]

_ISSUE = re.compile(r"^(?:issue\s*)?#?\s*(\d+)$", re.IGNORECASE)
_PR = {"pr", "pr description", "the pr description", "pull request", "description"}
_LINES = re.compile(r"^(.+?):(\d+)(?:-(\d+))?$")


class Place(BaseModel):
    """Lines of the reviewed commit and what they show."""

    file: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    shows: str = Field(min_length=1, description="What these lines show, in one sentence.")

    @field_validator("file")
    @classmethod
    def _as_git_spells_it(cls, value: str) -> str:
        try:
            return clean_path(value)
        except ToolError as exc:
            raise ValueError(str(exc)) from None


class Check(BaseModel):
    """What the program found when it checked a basis; never written by a model."""

    status: Status
    problem: str | None = None


class Basis(BaseModel):
    kind: Kind = Field(
        description="spec, contract or convention for a written rule broken; path for a"
        " failure traced from a caller or input."
    )
    source: str | None = Field(
        default=None,
        description="Where the rule is written: path:start-end of a file in the reviewed"
        " commit, pr for the PR description, or issue#N for a linked issue. Not for path.",
    )
    quote: str | None = Field(
        default=None,
        max_length=MAX_QUOTE_CHARS,
        description="The rule exactly as written in its source. Not for path.",
    )
    places: list[Place] = Field(
        min_length=1,
        description="The code that breaks the rule, or the steps of the path in order.",
    )
    explanation: str = Field(
        min_length=1, description="Why the code does not match, or why the path fails."
    )
    # Filled in by the program; a model's value is replaced.
    check: SkipJsonSchema[Check | None] = None

    @model_validator(mode="after")
    def _rules_are_quoted(self) -> Self:
        if self.kind != "path" and not (self.source and self.quote and self.quote.strip()):
            raise ValueError(f"a {self.kind} basis needs a source and a quote")
        return self


@dataclass(frozen=True)
class Sources:
    """What a basis may cite besides code: the reviewed commit, the PR and its issues.

    `text` gives a file of the reviewed commit, None when it is not there, and raises
    ToolTimeoutError when git is too slow to say.
    """

    text: Callable[[str], str | None]
    pr_text: str
    issues: Mapping[int, str]


def _squash(text: str) -> str:
    return " ".join(text.split())


def _find(quote: str, lines: Sequence[str]) -> tuple[int, int] | None:
    """The 1-based lines a quote spans in `lines`, whitespace not counted; None if absent."""
    starts: list[int] = []
    parts: list[str] = []
    length = 0
    for line in lines:
        starts.append(length)
        part = _squash(line)
        parts.append(part)
        length += len(part) + 1
    at = " ".join(parts).find(_squash(quote))
    if at < 0:
        return None
    first = bisect.bisect_right(starts, at)
    last = bisect.bisect_right(starts, at + len(_squash(quote)) - 1)
    return first, last


def _place_problem(place: Place, text: Callable[[str], str | None]) -> str | None:
    if place.line_end < place.line_start:
        return f"{place.file}:{place.line_start}-{place.line_end} ends before it starts"
    content = text(place.file)
    if content is None:
        return f"{place.file} is not in the reviewed commit"
    lines = len(content.splitlines())
    if place.line_end > lines:
        return f"{place.file} has only {lines} lines, not {place.line_end}"
    return None


def _source_problem(basis: Basis, sources: Sources) -> tuple[str | None, str | None]:
    """A problem with the quoted source, and the source with its real lines if it moved."""
    assert basis.source is not None and basis.quote is not None  # noqa: S101 - validated
    source = basis.source.strip()
    if source.lower() in _PR:
        found = _squash(basis.quote) in _squash(sources.pr_text)
        return (None if found else "the quote is not in the PR description"), None
    if match := _ISSUE.match(source):
        number = int(match.group(1))
        if number not in sources.issues:
            return f"issue #{number} is not linked to this PR", None
        found = _squash(basis.quote) in _squash(sources.issues[number])
        return (None if found else f"the quote is not in issue #{number}"), None
    lines_given = _LINES.match(source)
    path = lines_given.group(1) if lines_given else source
    try:
        path = clean_path(path)
    except ToolError as exc:
        return f"{source}: {exc}", None
    content = sources.text(path)
    if content is None:
        return f"{path} is not in the reviewed commit", None
    lines = content.splitlines()
    where = _find(basis.quote, lines)
    if where is None:
        return f"the quote is not in {path}", None
    start, end = where
    if lines_given:
        given_start = int(lines_given.group(2))
        given_end = int(lines_given.group(3) or given_start)
        if given_start <= start and end <= given_end:
            return None, None
    # Exactly as written in the file, at other lines than given: the lines are wrong, not the quote.
    return None, f"{path}:{start}" if start == end else f"{path}:{start}-{end}"


def check_basis(basis: Basis, sources: Sources) -> Basis:
    """`basis` with its check, and with its source's real lines when only those were off."""
    try:
        problems = [p for place in basis.places if (p := _place_problem(place, sources.text))]
        moved = None
        if basis.kind != "path":
            problem, moved = _source_problem(basis, sources)
            if problem:
                problems.append(problem)
    except ToolTimeoutError as exc:  # says nothing about the basis: it is just not checked
        check = Check(status="unchecked", problem=f"could not be checked in time: {exc}")
        return basis.model_copy(update={"check": check})
    if problems:
        check = Check(status="failed", problem="; ".join(problems))
        return basis.model_copy(update={"check": check})
    if moved is not None:
        return basis.model_copy(update={"source": moved, "check": Check(status="corrected")})
    return basis.model_copy(update={"check": Check(status="ok")})


def supported(bases: Sequence[Basis]) -> bool:
    """Whether at least one basis checked out (corrected lines count)."""
    return any(checked_out(b) for b in bases)


def checked_out(basis: Basis) -> bool:
    return basis.check is not None and basis.check.status in ("ok", "corrected")


def describe(basis: Basis) -> str:
    """A basis as text for a model to read, with how the program's check came out."""
    head = basis.kind if basis.kind == "path" else f"{basis.kind} from {basis.source}"
    lines = [head + (f': "{basis.quote}"' if basis.quote else "")]
    lines += [f"  - {p.file}:{p.line_start}-{p.line_end}: {p.shows}" for p in basis.places]
    lines.append(f"  because: {basis.explanation}")
    check = basis.check
    if check is None or check.status == "unchecked":
        lines.append("  checked: not checked" + (f" ({check.problem})" if check else ""))
    elif check.status == "failed":
        lines.append(f"  checked: the program could not confirm it: {check.problem}")
    else:
        lines.append("  checked: the files, lines and quote are there")
    return "\n".join(lines)
