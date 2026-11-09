from typing import Any

import pytest
from pydantic import ValidationError

from review_agents.pipeline.basis import (
    MAX_QUOTE_CHARS,
    MIN_QUOTE_CHARS,
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
        (
            {"source": "issue#12", "quote": "a sentence that is not in it"},
            "quote is not in issue #12",
        ),
        (
            {"source": "pr", "quote": "a sentence that is not in it"},
            "quote is not in the PR description",
        ),
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


def test_a_quote_that_appears_twice_keeps_the_lines_given_when_it_is_there() -> None:
    twice = {**FILES, "docs/twice.md": "Amounts are in cents.\n\nAgain:\n\nAmounts are in cents.\n"}

    def text(path: str) -> str | None:
        return twice.get(path)

    checked = check_basis(
        basis(source="docs/twice.md:5", quote="Amounts are in cents."), sources(text=text)
    )

    assert checked.check is not None and checked.check.status == "ok"
    assert checked.source == "docs/twice.md:5"


RULES = "# Rules\n\nFailures are logged.\n\nRetries stop after three tries.\n"


def rules_sources() -> Sources:
    files = {**FILES, "docs/rules.md": RULES}
    return sources(text=lambda path: files.get(path))


@pytest.mark.parametrize(
    "quote",
    [
        "Failures are logged.\n\nRetries stop after three tries.",  # copied as written
        "Failures are logged. Retries stop after three tries.",  # joined into one line
    ],
)
def test_a_quote_across_a_blank_line_checks_out(quote: str) -> None:
    checked = check_basis(basis(source="docs/rules.md:3-5", quote=quote), rules_sources())

    assert checked.check is not None and checked.check.status == "ok"


def test_a_quote_too_short_to_be_a_rule_does_not_check_out() -> None:
    assert MIN_QUOTE_CHARS == 8
    checked = check_basis(basis(source="docs/refunds.md:5", quote="refund"), sources())

    assert checked.check is not None and checked.check.status == "failed"
    assert "too short" in (checked.check.problem or "")


def test_a_quote_matches_whole_words_only() -> None:
    # "is at most the paid" is there; "s at most the pai" is only inside other words.
    checked = check_basis(basis(source="docs/refunds.md:5-6", quote="s at most the pai"), sources())

    assert checked.check is not None and checked.check.status == "failed"


def test_quoting_the_code_under_review_is_not_a_rule() -> None:
    checked = check_basis(basis(source="payments/refund.py:2", quote="total = cents"), sources())

    assert checked.check is not None and checked.check.status == "failed"
    assert "the code under review" in (checked.check.problem or "")


# Chinese punctuation on purpose: it is what real Chinese rules use.
ZH = "# 规则\n\n本 PR 规定：所有请求必须带超时时间。重试最多三次。\n\nretry_limit_total = 3\n"


def zh_sources() -> Sources:
    files = {**FILES, "docs/zh.md": ZH}
    return sources(
        text=lambda path: files.get(path),
        pr_text="本 PR 规定：所有请求必须带超时时间。",
    )


@pytest.mark.parametrize("source", ["docs/zh.md:3", "pr"])
def test_half_a_chinese_sentence_is_a_quote_chinese_has_no_spaces_between_words(
    source: str,
) -> None:
    checked = check_basis(basis(source=source, quote="请求必须带超时时间"), zh_sources())

    assert checked.check is not None and checked.check.status == "ok"


def test_part_of_an_identifier_is_not_a_quote_of_it() -> None:
    # "retry_limit" is only there inside "retry_limit_total".
    checked = check_basis(basis(source="docs/zh.md:5", quote="retry_limit"), zh_sources())
    whole = check_basis(basis(source="docs/zh.md:5", quote="retry_limit_total = 3"), zh_sources())

    assert checked.check is not None and checked.check.status == "failed"
    assert whole.check is not None and whole.check.status == "ok"
