"""Strict tool schemas (issue #109): the subset strict modes share, checked as written."""

from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from review_agents.pipeline import merge as merge_module
from review_agents.pipeline import nits as nits_module
from review_agents.pipeline.agent import _add_tool, _submit_tool
from review_agents.pipeline.verify import _verdict_tool
from review_agents.strict import UNSUPPORTED, drop_nulls, strict_tool
from review_agents.tools import RepoTools, ToolBudget, agent_tools
from review_agents.walkthrough import _submit_tool as walkthrough_tool


def read_tools() -> list[StructuredTool]:
    tools = RepoTools.__new__(RepoTools)  # only the signatures are read
    return [
        StructuredTool.from_function(f, name=n)
        for n, f in agent_tools(tools, ToolBudget(1)).items()
    ]


def merge_tools() -> list[StructuredTool]:
    """The judge's and the nit merge's tools, built as their modules build them."""

    def merge(**kwargs: Any) -> str:
        """Merge."""
        return ""

    return [
        StructuredTool.from_function(merge, name="merge", args_schema=merge_module._MergeArgs),
        StructuredTool.from_function(merge, name="merge_nits", args_schema=nits_module._Args),
    ]


ALL = [
    _add_tool(),
    _submit_tool(),
    _verdict_tool(),
    walkthrough_tool(),
    *read_tools(),
    *merge_tools(),
]


def objects(node: object) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            yield node
        for value in node.values():
            yield from objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from objects(value)


def keys(node: object) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties":  # field names are not keywords
                for field in value.values():
                    yield from keys(field)
                continue
            yield key
            yield from keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from keys(value)


@pytest.mark.parametrize("tool", ALL, ids=lambda t: t.name)
def test_every_object_lists_every_field_as_required_and_no_other(tool: StructuredTool) -> None:
    definition = strict_tool(convert_to_openai_tool(tool))

    assert definition["function"]["strict"] is True
    found = list(objects(definition["function"]["parameters"]))
    assert found
    for node in found:
        assert node["additionalProperties"] is False
        assert sorted(node["required"]) == sorted(node["properties"])


@pytest.mark.parametrize("tool", ALL, ids=lambda t: t.name)
def test_no_keyword_a_strict_mode_may_refuse_is_left(tool: StructuredTool) -> None:
    definition = strict_tool(convert_to_openai_tool(tool))

    assert not set(keys(definition["function"]["parameters"])) & UNSUPPORTED


def test_an_optional_field_may_be_null_and_a_required_one_may_not() -> None:
    properties = strict_tool(convert_to_openai_tool(_add_tool()))["function"]["parameters"][
        "properties"
    ]

    assert {"type": "null"} in properties["suggestion"]["anyOf"]  # it has a default
    assert "anyOf" not in properties["title"]


def test_a_constraint_left_out_is_said_in_the_description() -> None:
    properties = strict_tool(convert_to_openai_tool(_add_tool()))["function"]["parameters"][
        "properties"
    ]

    assert "minimum 1" in properties["line_start"]["description"]


def test_the_programs_own_fields_are_not_offered() -> None:
    basis = strict_tool(convert_to_openai_tool(_add_tool()))["function"]["parameters"][
        "properties"
    ]["basis"]

    assert "check" not in str(basis)


def test_a_definition_is_left_as_it_was() -> None:
    original = convert_to_openai_tool(_add_tool())
    copy = convert_to_openai_tool(_add_tool())

    strict_tool(copy)

    assert copy == original


def test_null_values_are_dropped_at_every_level_but_not_from_lists() -> None:
    args = {"a": None, "b": {"c": None, "d": 1}, "e": [None, {"f": None}]}

    assert drop_nulls(args) == {"b": {"d": 1}, "e": [None, {}]}


def test_a_field_named_like_a_keyword_is_kept() -> None:
    def search(pattern: str, path: str = "") -> str:
        """Search."""
        return ""

    parameters = strict_tool(convert_to_openai_tool(search))["function"]["parameters"]

    assert sorted(parameters["properties"]) == ["path", "pattern"]
    assert sorted(parameters["required"]) == ["path", "pattern"]


@pytest.mark.parametrize(
    "build",
    [
        lambda: {
            "type": "object",
            "properties": {"m": {"type": "object", "additionalProperties": {"type": "string"}}},
        },
        lambda: {"type": "object", "properties": {"k": {"const": "x"}}},
        lambda: {
            "type": "object",
            "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
            "required": ["x"],
        },
    ],
    ids=["a map", "a constant", "a field that must be sent as null"],
)
def test_a_schema_strict_modes_cannot_take_as_meant_is_refused(build: Any) -> None:
    # A map would come back empty, a constant is not in every subset, and a null sent on
    # purpose would read as left out: better to fail where the tool is defined.
    definition = {"type": "function", "function": {"name": "t", "parameters": build()}}

    with pytest.raises(ValueError, match="strict"):
        strict_tool(definition)
