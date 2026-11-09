from typing import Any

import pytest
from pydantic import ValidationError

from review_agents.pipeline.basis import (
    MAX_QUOTE_CHARS,
    Basis,
    Sources,
    check_basis,
    supported,
)
from review_agents.tools import ToolTimeoutError

SPEC = "# Refunds\n\nRules:\n\n- A refund is at most\n  the paid amount.\n- Refunds are in cents.\n"
CODE = "def refund(order, cents):\n    total = cents\n    return total\n"
FILES = {"docs/refunds.md": SPEC, "payments/refund.py": CODE}


def sources(**changes: Any) -> Sources:
    def text(path: str) -> str | None:
        return FILES.get(path)

    base: dict[str, Any] = {
        "text": text,
        "pr_text": "Clamp refunds\n\nNever refund more than was paid.",
        "issues": {12: "Refunds must not exceed the amount paid."},
    }
    return Sources(**(base | changes))


PLACE = {"file": "payments/refund.py", "line_start": 2, "line_end": 2, "shows": "no cap"}


def basis(**changes: Any) -> Basis:
    base: dict[str, Any] = {
        "kind": "spec",
        "source": "docs/refunds.md:5-6",
        "quote": "A refund is at most the paid amount.",
        "places": [PLACE],
        "explanation": "The amount is returned as requested.",
    }
    return Basis.model_validate(base | changes)


def test_a_rule_based_basis_needs_a_source_and_a_quote() -> None:
    for kind in ("spec", "contract", "convention"):
        with pytest.raises(ValidationError, match="source and a quote"):
            basis(kind=kind, quote=None)
    path = basis(kind="path", source=None, quote=None)
    assert path.kind == "path"


def test_a_quote_is_at_most_three_hundred_characters() -> None:
    assert MAX_QUOTE_CHARS == 300  # decision D-036
    with pytest.raises(ValidationError):
        basis(quote="x" * 301)


def test_a_basis_whose_quote_is_where_it_says_checks_out() -> None:
    # The quote spans two lines and its spacing differs: whitespace does not count.
    checked = check_basis(basis(), sources())

    assert checked.check is not None and checked.check.status == "ok"
    assert checked.check.problem is None


def test_a_quote_found_elsewhere_in_the_file_gets_its_real_lines() -> None:
    checked = check_basis(basis(source="docs/refunds.md:1-2"), sources())

    assert checked.check is not None and checked.check.status == "corrected"
    assert checked.source == "docs/refunds.md:5-6"


def test_a_source_file_given_without_lines_gets_them() -> None:
    checked = check_basis(basis(source="docs/refunds.md"), sources())

    assert checked.check is not None and checked.check.status == "corrected"
    assert checked.source == "docs/refunds.md:5-6"


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"source": "docs/policy.md:3"}, "docs/policy.md is not in the reviewed commit"),
        ({"quote": "Refunds can never exceed the payment."}, "quote is not in docs/refunds.md"),
        ({"source": "issue#99", "quote": "anything"}, "issue #99 is not linked to this PR"),
        ({"source": "issue#12", "quote": "not in it"}, "quote is not in issue #12"),
        ({"source": "pr", "quote": "not in it"}, "quote is not in the PR description"),
        (
            {"places": [dict(PLACE, file="payments/missing.py")]},
            "payments/missing.py is not in the reviewed commit",
        ),
        ({"places": [dict(PLACE, line_end=40)]}, "payments/refund.py has only 3 lines"),
    ],
)
def test_a_basis_that_does_not_check_out_says_why(changes: dict[str, Any], problem: str) -> None:
    checked = check_basis(basis(**changes), sources())

    assert checked.check is not None and checked.check.status == "failed"
    assert checked.check.problem is not None and problem in checked.check.problem


def test_the_pr_description_and_linked_issues_are_sources() -> None:
    pr = check_basis(basis(source="pr", quote="Never refund more than was paid."), sources())
    issue = check_basis(
        basis(source="#12", quote="Refunds must not exceed the amount paid."), sources()
    )

    assert pr.check is not None and pr.check.status == "ok"
    assert issue.check is not None and issue.check.status == "ok"


def test_a_path_basis_checks_only_its_places() -> None:
    steps = [PLACE, dict(PLACE, line_start=3, line_end=3, shows="returned as is")]
    checked = check_basis(basis(kind="path", source=None, quote=None, places=steps), sources())

    assert checked.check is not None and checked.check.status == "ok"


def test_a_git_timeout_leaves_a_basis_unchecked_not_failed() -> None:
    def slow(path: str) -> str | None:
        raise ToolTimeoutError("git cat-file timed out after 30s")

    checked = check_basis(basis(), sources(text=slow))

    assert checked.check is not None and checked.check.status == "unchecked"
    assert checked.check.problem is not None and "timed out" in checked.check.problem


def test_a_finding_is_supported_when_one_basis_checks_out() -> None:
    good = check_basis(basis(), sources())
    bad = check_basis(basis(source="docs/policy.md:3"), sources())

    assert supported([bad, good])
    assert not supported([bad])
    assert not supported([])
