import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from review_eval.__main__ import main
from review_eval.case import ANSWER_FILE, CASE_FILE, Answer, Case, Proof, load_set

EXAMPLE = Path(__file__).parent.parent / "golden" / "cases"
BASE = "a" * 40
HEAD = "b" * 40
FIX = "c" * 40


def case_data(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": 1,
        "id": "demo-case",
        "split": "dev",
        "repo": {"url": "https://example.com/r.git", "base": BASE, "head": HEAD},
        "pr": {"title": "Add mean", "body": "Called on every comment a user posts."},
    }
    return data | changes


def expected(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "empty-input",
        "locations": [{"file": "src/mean.py", "line_start": 3, "line_end": 5}],
        "must_say": "Empty text divides by zero.",
        "severities": ["blocking"],
        "dimensions": ["edge_cases"],
        "proof": {"kind": "repro", "command": ["python", "-c", "..."]},
    }
    return data | changes


def answer_data(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": 1,
        "case": "demo-case",
        "origin": {"kind": "planted"},
        "expected": [expected()],
        "tags": {"languages": ["python"], "context_hops": 0},
    }
    return data | changes


def at(file: str) -> dict[str, Any]:
    return {"file": file, "line_start": 1, "line_end": 1}


def write(root: Path, case: dict[str, Any], answer: dict[str, Any] | None = None) -> Path:
    directory = root / str(case["id"])
    directory.mkdir(parents=True)
    (directory / CASE_FILE).write_text(yaml.safe_dump(case, allow_unicode=True))
    if answer is not None:
        (directory / ANSWER_FILE).write_text(yaml.safe_dump(answer, allow_unicode=True))
    return directory


def test_the_example_case_loads_with_its_answer_as_a_draft() -> None:
    golden = load_set(EXAMPLE, EXAMPLE)

    assert golden.errors == []
    assert [c.case.id for c in golden.cases] == ["review-lab-basis-checks"]
    answer = golden.cases[0].answer
    assert answer is not None and len(answer.expected) == 3
    assert golden.drafts == ["review-lab-basis-checks"]


def test_a_sound_case_and_answer_load(tmp_path: Path) -> None:
    write(tmp_path, case_data(), answer_data(verified={"by": "izel", "on": "2026-11-09"}))

    golden = load_set(tmp_path, tmp_path)

    assert golden.errors == []
    assert golden.drafts == []
    assert golden.cases[0].case.config.mode == "standard"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"repo": {"url": "u", "base": BASE, "head": BASE}}, "base and head"),
        ({"repo": {"url": "u", "base": "abc123", "head": HEAD}}, "String should match"),
        ({"split": "train"}, "split"),
        ({"pr": {"title": "t", "body": ""}}, "at least 1 character"),
        ({"config": {"mode": "fast"}}, "config.mode"),
        ({"reviewer": "x"}, "Extra inputs are not permitted"),
    ],
)
def test_a_case_that_breaks_the_format_is_rejected(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Case.model_validate(case_data(**changes))


@pytest.mark.parametrize(
    ("proof", "message"),
    [
        ({"kind": "fail_to_pass", "command": ["pytest"]}, "fix_commit and test_paths"),
        ({"kind": "repro"}, "repro needs command"),
        ({"kind": "repro", "command": ["x"], "fix_commit": FIX}, "takes no fix_commit"),
        ({"kind": "manual"}, "needs a note"),
        ({"kind": "manual", "note": "seen", "command": ["x"]}, "takes no command"),
        ({"kind": "repro", "command": ["x"], "head_exit_code": 0}, "greater than or equal to 1"),
        (
            {"kind": "fail_to_pass", "command": ["x"], "fix_commit": FIX, "test_paths": ["../t"]},
            "inside the repository",
        ),
    ],
)
def test_a_proof_needs_the_fields_of_its_kind(proof: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Proof.model_validate(proof)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"origin": {"kind": "clean"}}, "a clean case expects no findings"),
        ({"expected": []}, "expects at least one finding"),
        ({"expected": [expected(), expected()]}, "distinct ids"),
        ({"expected": [expected(severities=["nit"])]}, "severities"),
        ({"expected": [expected(dimensions=["style"])]}, "dimensions"),
        ({"expected": [expected(severities=["blocking", "blocking"])]}, "listed once"),
        ({"expected": [expected(locations=[])]}, "at least 1 item"),
        (
            {"expected": [expected(locations=[{"file": "a.py", "line_start": 5, "line_end": 4}])]},
            "line_end must not be before line_start",
        ),
        ({"expected": [expected(locations=[at("/")])]}, "must name a file"),
        ({"expected": [expected(locations=[at("src/../../x.py")])]}, "inside the repository"),
    ],
)
def test_an_answer_that_breaks_the_format_is_rejected(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Answer.model_validate(answer_data(**changes))


def test_an_absolute_path_is_spelled_inside_the_repository() -> None:
    # clean_path strips the leading slash: the location is read inside the repository.
    location = {"file": "/src/mean.py", "line_start": 1, "line_end": 1}
    answer = Answer.model_validate(answer_data(expected=[expected(locations=[location])]))
    assert answer.expected[0].locations[0].file == "src/mean.py"


def test_the_set_reports_mismatches_between_cases_and_answers(tmp_path: Path) -> None:
    cases, answers = tmp_path / "cases", tmp_path / "answers"
    write(cases, case_data(id="no-answer"))
    write(cases, case_data(id="wrong-name"))
    (cases / "wrong-name" / CASE_FILE).write_text(yaml.safe_dump(case_data(id="other")))
    write(cases, case_data(id="answers-another"))
    write(answers, case_data(id="answers-another"), answer_data(case="not-this-one"))
    write(answers, case_data(id="orphan"), answer_data(case="orphan"))
    write(cases, case_data(id="fix-is-head"))
    fix_is_head = expected(
        proof={"kind": "fail_to_pass", "command": ["x"], "fix_commit": HEAD, "test_paths": ["t.py"]}
    )
    write(
        answers,
        case_data(id="fix-is-head"),
        answer_data(case="fix-is-head", expected=[fix_is_head]),
    )

    errors = "\n".join(load_set(cases, answers).errors)

    assert "no answer for case 'no-answer'" in errors
    assert "id 'other' is not the directory's name" in errors
    assert "answers case 'not-this-one', not 'answers-another'" in errors
    assert "orphan: an answer with no case" in errors
    assert "fix_commit must be a commit after the PR" in errors


def test_without_answers_only_the_cases_are_checked(tmp_path: Path) -> None:
    write(tmp_path, case_data())

    golden = load_set(tmp_path)

    assert golden.errors == []
    assert golden.cases[0].answer is None


def test_an_unreadable_file_is_an_error_not_a_crash(tmp_path: Path) -> None:
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / CASE_FILE).write_text("id: [unclosed\n")
    (tmp_path / "empty").mkdir()

    errors = load_set(tmp_path).errors

    assert any("cannot read it" in e for e in errors)
    assert any("empty: no case.yaml" in e for e in errors)


def test_the_check_command_exits_non_zero_on_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copytree(EXAMPLE, tmp_path / "cases")
    assert main(["check", str(tmp_path / "cases"), "--answers", str(tmp_path / "cases")]) == 0
    assert "1 cases loaded (1 dev); 0 errors" in capsys.readouterr().out

    (tmp_path / "cases" / "review-lab-basis-checks" / ANSWER_FILE).unlink()
    assert main(["check", str(tmp_path / "cases"), "--answers", str(tmp_path / "cases")]) == 1
    assert "no answer for case" in capsys.readouterr().err
