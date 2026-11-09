"""The golden set's format: one case per pull request, its input and its answer apart.

A case is a directory named after its id. `case.yaml` holds what the review gets: the
repository, the two commits, the PR's description and linked issues, and the repository
configuration. `answer.yaml` holds what a good review finds, and nothing in it reaches the
reviewer. The answers can live under another root than the cases, so the held-out cases'
answers stay out of reach of whoever tunes the skills.

Loading checks everything that can be checked without the repository. What needs it,
that the commits exist, that every location is inside its file at the head commit and
that every proof fails on the head commit, is the harness's check.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from review_agents.pipeline.agent import Severity
from review_agents.repo_config import Dimension, RepoConfig
from review_agents.tools import ToolError, clean_path

SCHEMA_VERSION = 1
CASE_FILE = "case.yaml"
ANSWER_FILE = "answer.yaml"

Id = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=80)]
Sha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Text = Annotated[str, Field(min_length=1)]
# dev: tune the skills on it and read every failure. test: held out, run only to score a
# version, its failures not read. smoke: a quick check that the pipeline runs at all.
Split = Literal["dev", "test", "smoke"]
# An expected finding is never a nit: a nit nobody raises is not missed.
ExpectedSeverity = Literal["blocking", "important", "robustness"]


class _Strict(BaseModel):
    # Unknown keys are errors, so a misspelt field is reported instead of silently ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


def _repository_path(value: str) -> str:
    try:
        cleaned = clean_path(value)
    except ToolError as exc:
        raise ValueError(str(exc)) from None
    if not cleaned:
        raise ValueError("must name a file, not the repository root")
    return cleaned


# The case: what the review gets.


class Repo(_Strict):
    url: Text = Field(description="Where to clone it from: a URL git accepts, or a local path.")
    base: Sha = Field(description="The commit the PR starts from, in full.")
    head: Sha = Field(description="The PR's last commit, in full: the commit under review.")
    previous_head: Sha | None = Field(
        default=None,
        description="For a second-round case: the head the previous round reviewed.",
    )

    @model_validator(mode="after")
    def _distinct_commits(self) -> Self:
        if self.base == self.head:
            raise ValueError("base and head must be different commits")
        if self.previous_head in (self.base, self.head):
            raise ValueError("previous_head must differ from base and head")
        return self


class Issue(_Strict):
    """A linked issue as it read when the case was made; the live one may have changed."""

    number: int = Field(ge=1)
    title: Text
    body: str = ""


class Pull(_Strict):
    title: Text
    body: Text = Field(
        description="The PR description. It must state the usage scenario: a blocking"
        " finding is one that happens in it."
    )
    issues: list[Issue] = Field(default_factory=list)

    @field_validator("issues")
    @classmethod
    def _one_copy_per_issue(cls, value: list[Issue]) -> list[Issue]:
        numbers = [issue.number for issue in value]
        if len(numbers) != len(set(numbers)):
            raise ValueError("each issue is listed once")
        return value


class Case(_Strict):
    schema_version: Literal[1]
    id: Id
    split: Split
    repo: Repo
    pr: Pull
    config: RepoConfig = Field(
        default_factory=RepoConfig,
        description="The repository's .github/review-agents.yml, in the same form.",
    )


# The answer: what a good review finds.


class Location(_Strict):
    """Lines of a file at the head commit."""

    file: Text
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @field_validator("file")
    @classmethod
    def _inside_repository(cls, value: str) -> str:
        return _repository_path(value)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("line_end must not be before line_start")
        return self


class Proof(_Strict):
    """How the problem was shown to be real.

    - `fail_to_pass`: take `test_paths` from `fix_commit` into the head commit and run
      `command`; it exits with `head_exit_code`. At `fix_commit` it exits with 0.
    - `repro`: `command`, run at the head commit, exits with `head_exit_code`.
    - `manual`: no command; `note` says who showed it and how.

    `command` runs at the root of a checkout. `head_exit_code` is the code of the failure
    itself: pytest exits with 1 when tests fail and with 2 when they cannot be collected,
    so a test file that does not import at the head commit proves nothing.
    """

    kind: Literal["fail_to_pass", "repro", "manual"]
    command: list[Text] = Field(default_factory=list)
    head_exit_code: int = Field(default=1, ge=1, le=255)
    fix_commit: Sha | None = None
    test_paths: list[Text] = Field(default_factory=list)
    note: str = ""

    @field_validator("test_paths")
    @classmethod
    def _inside_repository(cls, value: list[str]) -> list[str]:
        return [_repository_path(path) for path in value]

    @model_validator(mode="after")
    def _fields_of_its_kind(self) -> Self:
        if self.kind == "fail_to_pass":
            if not (self.command and self.fix_commit and self.test_paths):
                raise ValueError("fail_to_pass needs command, fix_commit and test_paths")
        elif self.kind == "repro":
            if not self.command:
                raise ValueError("repro needs command")
            if self.fix_commit or self.test_paths:
                raise ValueError("repro takes no fix_commit or test_paths")
        else:
            if not self.note.strip():
                raise ValueError("manual needs a note: who showed it, and how")
            if self.command or self.fix_commit or self.test_paths:
                raise ValueError("manual takes no command, fix_commit or test_paths")
        return self


class Expected(_Strict):
    """A problem a good review reports."""

    id: Id
    locations: list[Location] = Field(
        min_length=1,
        description="Every place the problem is; a finding at any of them may be this one.",
    )
    must_say: Text = Field(
        description="The cause a finding must name to count, not only its symptom or"
        " 'add a test'. The grader's rubric."
    )
    severities: list[ExpectedSeverity] = Field(
        min_length=1, description="Every level that counts as ranked right."
    )
    dimensions: list[Dimension] = Field(
        min_length=1, description="Every dimension that may rightly find it."
    )
    preexisting: bool = False
    proof: Proof

    @field_validator("severities", "dimensions")
    @classmethod
    def _no_repeats(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("each value is listed once")
        return value


class Acceptable(_Strict):
    """A point a review may raise without it counting as a false positive."""

    about: Text = Field(description="What the point is; the grader's rubric.")
    locations: list[Location] = Field(default_factory=list)
    max_severity: Severity = Field(
        default="robustness",
        description="Raised above this level, the point counts as a false positive.",
    )


class Origin(_Strict):
    # introduced_bug: a real commit a later commit fixed. clean: a real change nothing
    # later fixed or reverted. accepted_review: a human review comment the author acted on.
    # planted: a bug written for the set.
    kind: Literal["introduced_bug", "clean", "accepted_review", "planted"]
    url: str = ""
    note: str = ""


class Tags(_Strict):
    languages: list[Text] = Field(min_length=1)
    context_hops: int = Field(
        ge=0,
        description="How far from the diff the code needed to see the problems is:"
        " 0 inside the changed functions, 1 a direct caller, callee or test, and so on.",
    )
    labels: list[Text] = Field(default_factory=list)


class Verified(_Strict):
    by: Text = Field(description="The person who checked the answer, not a model.")
    on: date


class Answer(_Strict):
    schema_version: Literal[1]
    case: Id
    origin: Origin
    expected: list[Expected] = Field(default_factory=list)
    acceptable: list[Acceptable] = Field(default_factory=list)
    tags: Tags
    verified: Verified | None = Field(
        default=None, description="Unset while the answer is a draft; a draft is not scored."
    )

    @model_validator(mode="after")
    def _fits_its_origin(self) -> Self:
        ids = [e.id for e in self.expected]
        if len(ids) != len(set(ids)):
            raise ValueError("expected findings need distinct ids")
        if self.origin.kind == "clean" and self.expected:
            raise ValueError("a clean case expects no findings")
        if self.origin.kind != "clean" and not self.expected:
            raise ValueError(f"an {self.origin.kind} case expects at least one finding")
        return self


# Loading.


@dataclass(frozen=True)
class GoldenCase:
    case: Case
    answer: Answer | None
    directory: Path


@dataclass(frozen=True)
class GoldenSet:
    """The cases that loaded, and an error for each problem found; `errors` empty when sound."""

    cases: list[GoldenCase] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def drafts(self) -> list[str]:
        return [c.case.id for c in self.cases if c.answer and c.answer.verified is None]


def _messages(exc: ValidationError) -> list[str]:
    out = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"]) or "(top level)"
        out.append(f"{where}: {error['msg']}")
    return out


def _load[T: BaseModel](model: type[T], path: Path) -> tuple[T | None, list[str]]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return None, [f"{path}: cannot read it: {exc}"]
    try:
        return model.model_validate(raw), []
    except ValidationError as exc:
        return None, [f"{path}: {message}" for message in _messages(exc)]


def load_case(path: Path) -> tuple[Case | None, list[str]]:
    return _load(Case, path)


def load_answer(path: Path) -> tuple[Answer | None, list[str]]:
    return _load(Answer, path)


def _case_directories(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def load_set(cases_root: Path, answers_root: Path | None = None) -> GoldenSet:
    """Every case under `cases_root`, each with its answer under `answers_root`.

    With `answers_root` unset the answers are not read, as when the held-out answers are
    out of reach. With it set, every case needs an answer there and every answer a case.
    """
    if not cases_root.is_dir():
        return GoldenSet(errors=[f"{cases_root}: not a directory"])
    errors: list[str] = []
    cases: list[GoldenCase] = []
    for directory in _case_directories(cases_root):
        path = directory / CASE_FILE
        if not path.is_file():
            errors.append(f"{directory}: no {CASE_FILE}")
            continue
        case, problems = load_case(path)
        errors += problems
        if case is None:
            continue
        if case.id != directory.name:
            errors.append(f"{path}: id {case.id!r} is not the directory's name")
            continue
        answer = None
        if answers_root is not None:
            answer, problems = _answer_for(case, answers_root / case.id / ANSWER_FILE)
            errors += problems
        cases.append(GoldenCase(case, answer, directory))
    if answers_root is not None and answers_root.is_dir():
        known = {c.case.id for c in cases} | {d.name for d in _case_directories(cases_root)}
        errors += [
            f"{d}: an answer with no case" for d in _case_directories(answers_root)
            if d.name not in known
        ]  # fmt: skip
    return GoldenSet(cases, errors)


def _answer_for(case: Case, path: Path) -> tuple[Answer | None, list[str]]:
    if not path.is_file():
        return None, [f"{path}: no answer for case {case.id!r}"]
    answer, problems = load_answer(path)
    if answer is None:
        return None, problems
    if answer.case != case.id:
        return None, [f"{path}: answers case {answer.case!r}, not {case.id!r}"]
    commits = {case.repo.base, case.repo.head}
    for expected in answer.expected:
        if expected.proof.fix_commit in commits:
            problems.append(
                f"{path}: expected.{expected.id}: fix_commit must be a commit after the PR,"
                " not its base or head"
            )
    return (None if problems else answer), problems
