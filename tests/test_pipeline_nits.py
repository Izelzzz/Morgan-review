from typing import Any

from scripted import ScriptedModel, call

from review_agents.pipeline.nits import NIT_TOOL, merge_nits
from review_agents.report import Item


def nit(n: int, **changes: Any) -> Item:
    base: dict[str, Any] = {
        "fingerprint": f"{n}" * 64,
        "severity": "nit",
        "title": f"Unused import {n}",
        "file": f"pkg/m{n}.py",
        "line_start": n,
        "line_end": n,
        "why": "It is never used.",
        "suggestion": "Remove it.",
        "dimensions": ["maintainability"],
        "preexisting": False,
        "locations": [],
        "bases": [],
    }
    return Item(**(base | changes))


SEVEN = [nit(n) for n in range(1, 8)]
BLOCKING = nit(9, severity="blocking", title="Overflow")


def groups(*ids: list[int]) -> Any:
    return call(
        NIT_TOOL,
        {
            "groups": [
                {"ids": g, "title": f"Group {g[0]}", "why": "w", "suggestion": "s"} for g in ids
            ]
        },
    )


def merged(model: ScriptedModel, items: list[Item], limit: int = 5) -> Any:
    return merge_nits(items, lambda: model, limit=limit, callbacks=[], language="zh")


def test_nits_within_the_limit_are_not_sent_to_a_model() -> None:
    model = ScriptedModel(script=[])

    result = merged(model, [BLOCKING, *SEVEN[:5]])

    assert result.items == [BLOCKING, *SEVEN[:5]] and result.problem is None
    assert model.sent == []


def test_nits_past_the_limit_are_merged_and_no_place_is_lost() -> None:
    model = ScriptedModel(script=[groups([1, 2, 3], [4], [5, 6, 7])])

    result = merged(model, [BLOCKING, *SEVEN])

    assert result.problem is None
    assert result.items[0] is BLOCKING  # only nits are merged
    nits = result.items[1:]
    assert [i.title for i in nits] == ["Group 1", "Group 4", "Group 5"]
    first = nits[0]
    assert (first.file, first.line_start, first.fingerprint) == ("pkg/m1.py", 1, "1" * 64)
    assert first.locations == [("pkg/m2.py", 2, 2), ("pkg/m3.py", 3, 3)]
    places = {(i.file, i.line_start) for i in nits} | {
        (f, s) for i in nits for f, s, _ in i.locations
    }
    assert places == {(f"pkg/m{n}.py", n) for n in range(1, 8)}


def test_the_model_is_given_the_nits_as_data_and_the_language() -> None:
    model = ScriptedModel(script=[groups([1, 2, 3, 4, 5, 6, 7])])

    merged(model, SEVEN)

    text = "\n".join(str(m.content) for m in model.sent[0])
    assert '<untrusted source="nits">' in text and "Unused import 7" in text
    assert "Simplified Chinese" in text and "at most 5 groups" in text


def test_a_merge_that_drops_or_repeats_a_nit_is_not_used() -> None:
    for answer in (groups([1, 2], [3, 4, 5, 6]), groups([1, 2, 3], [3, 4, 5, 6, 7])):
        result = merged(ScriptedModel(script=[answer]), SEVEN)

        assert result.items == SEVEN  # the report then shows the first five
        assert result.problem is not None


def test_a_merge_with_too_many_groups_is_not_used() -> None:
    result = merged(ScriptedModel(script=[groups(*[[n] for n in range(1, 8)])]), SEVEN)

    assert result.items == SEVEN and result.problem is not None and "7 groups" in result.problem


def test_a_failed_or_missing_model_leaves_the_nits_as_they_are() -> None:
    failed = merged(ScriptedModel(script=[TimeoutError("timed out after 120s")]), SEVEN)
    no_call = merged(ScriptedModel(script=[call("other", {})]), SEVEN)

    assert failed.items == SEVEN and "timed out" in (failed.problem or "")
    assert no_call.items == SEVEN and no_call.problem is not None


def test_a_model_that_cannot_be_built_leaves_the_nits_as_they_are() -> None:
    def broken() -> ScriptedModel:
        raise ValueError("no such endpoint")

    result = merge_nits(SEVEN, broken, limit=5, callbacks=[], language="zh")

    assert result.items == SEVEN and "no such endpoint" in (result.problem or "")
