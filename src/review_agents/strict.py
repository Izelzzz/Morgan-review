"""Tool definitions for a service's strict mode, in the subset strict modes share (#109).

A service in strict mode decodes tool arguments to the schema it is sent, so they come
back in shape. Services take different parts of JSON Schema: OpenAI and DeepSeek both
want every field required and no field beyond those listed, OpenAI refuses string
lengths, DeepSeek refuses string lengths and item counts. What is sent here is what
both take: every field required, a field that was optional may be null, nothing beyond
the fields, and no length, count, range, pattern, format or default. A constraint left
out is said in the field's description, and the answer is still checked against the
full schema when it comes back, nulls dropped first so a field left out reads as left
out.
"""

from copy import deepcopy
from typing import Any

# Keywords a strict mode may refuse; checked again, in full, when the answer comes back.
UNSUPPORTED = frozenset(
    {
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "pattern",
        "format",
        "default",
        "minProperties",
        "maxProperties",
        "uniqueItems",
    }
)
# Said in the description instead; a default is not a constraint.
_SAID = UNSUPPORTED - {"default"}


def strict_tool(definition: dict[str, Any]) -> dict[str, Any]:
    """An OpenAI-format tool definition, as strict mode takes it; the input is not changed."""
    tool = deepcopy(definition)
    function = tool["function"]
    function["parameters"] = _tighten(function.get("parameters") or {"type": "object"})
    function["strict"] = True
    return tool


def _tighten(node: Any) -> Any:
    if isinstance(node, list):
        return [_tighten(n) for n in node]
    if not isinstance(node, dict):
        return node
    _refuse_what_cannot_be_meant(node)
    said = [f"{key} {node[key]}" for key in sorted(_SAID) if key in node]
    # Under `properties` the keys are field names, which may well be `pattern`.
    node = {
        k: {name: _tighten(f) for name, f in v.items()} if k == "properties" else _tighten(v)
        for k, v in node.items()
        if k not in UNSUPPORTED
    }
    if node.get("type") == "object" or "properties" in node:
        properties = node.get("properties") or {}
        required = set(node.get("required") or [])
        node["properties"] = {
            name: value if name in required else _nullable(value)
            for name, value in properties.items()
        }
        node["required"] = list(properties)
        node["additionalProperties"] = False
    if said:
        described = node.get("description", "")
        node["description"] = f"{described} ({'; '.join(said)})".strip()
    return node


def _refuse_what_cannot_be_meant(node: dict[str, Any]) -> None:
    """Fail where a tool is defined rather than send a schema that reads otherwise: a map
    could only come back empty, a constant is not in every strict subset, and a null
    sent on purpose would read as left out."""
    if isinstance(node.get("additionalProperties"), dict):
        raise ValueError("a map of any keys cannot be sent for strict mode")
    if "const" in node:
        raise ValueError("a constant cannot be sent for strict mode in every service")
    for name in node.get("required") or []:
        field = (node.get("properties") or {}).get(name, {})
        if {"type": "null"} in field.get("anyOf", []):
            raise ValueError(
                f"{name} must be sent as null at times, which strict mode reads as left out"
            )


def _nullable(node: dict[str, Any]) -> dict[str, Any]:
    if "anyOf" in node:
        if {"type": "null"} not in node["anyOf"]:
            node = {**node, "anyOf": [*node["anyOf"], {"type": "null"}]}
        return node
    outer: dict[str, Any] = {k: node[k] for k in ("description", "title") if k in node}
    inner = {k: v for k, v in node.items() if k not in outer}
    return {**outer, "anyOf": [inner, {"type": "null"}]}


def drop_nulls(value: Any) -> Any:
    """A tool's arguments with null fields left out, at every level; list items stay."""
    if isinstance(value, dict):
        return {k: drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [drop_nulls(v) for v in value]
    return value
