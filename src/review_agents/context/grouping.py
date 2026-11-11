"""Split a pull request too large for one review into groups of related changes.

A group is only made when the whole PR does not fit (decision D-008). Changes that
must be read together stay together:

- strong ties, never separated: one file is never split; files that use a definition
  another file changes; a test and the code it tests;
- a weaker tie, separated only when a group would not fit otherwise: files in the same
  module.

Each group still sees the PR's full intent and file list, plus the definitions every
other group changes, so a contract changed in one group is visible to the others.

`capacity_tokens` is the room for a group's own changes and the code around them; the
caller subtracts what every group carries anyway: the intent, the file list, the
repository layer and `signature_tokens` for the definitions changed elsewhere.
"""

import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from review_agents.context.pr_layer import (
    DEFINITION,
    TEST_PATH,
    PrLayer,
    names_in_patch,
    types_in_patch,
)
from review_agents.context.tokens import estimate_tokens

_IDENTIFIER = re.compile(r"[A-Za-z_]\w{2,}")
# Language hooks such as __init__ are defined everywhere and tie nothing together.
_DUNDER = re.compile(r"^__\w+__$")
OVERSIZED_NOTE = " This group is larger than one review window."


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list[str]]:
        found: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            found[self.find(item)].append(item)
        return [sorted(members) for members in found.values()]


def _definition_name(text: str) -> str | None:
    match = DEFINITION.search(text)
    return (match.group(1) or match.group(2)) if match else None


def changed_definitions(patch: str) -> list[tuple[str, str]]:
    """(name, line) of each definition a patch adds, removes or changes the body of.

    A `git diff -W` hunk also shows untouched definitions as context. A change belongs
    to the nearest definition above it in the hunk; a decorator, to the one below it.
    Layout is not read, so a multi-line signature or string does not end a definition.
    """
    lines = [
        line
        for line in patch.splitlines()
        if not line.startswith(("+++", "---")) and (line.startswith("@@") or line[1:].strip())
    ]
    found: dict[tuple[str, str], None] = {}
    current: tuple[str, str] | None = None
    decorated = False
    for index, line in enumerate(lines):
        if line.startswith("@@"):
            header = line.split("@@", 2)[-1].strip()
            name = _definition_name(header)
            current = (name, f"  {header}") if name else None
            decorated = False
            continue
        if line[:1] not in "+- ":
            continue
        changed, body = line[0] != " ", line[1:]
        name = _definition_name(body)
        if name:
            current = (name, line.rstrip())
            if changed or decorated:
                found[current] = None
            decorated = False
        elif decorated or (changed and body.lstrip().startswith("@") and _decorates(lines, index)):
            # The decorator and its arguments belong to the definition below.
            decorated = decorated or changed
        elif changed and current:
            found[current] = None
    return list(found)


def _decorates(lines: list[str], index: int) -> bool:
    """Whether the `@` line at `index` sits on top of a definition, not inside code.

    A decorator's arguments may run over several lines; brackets tell where it ends.
    """
    depth = 0
    for line in lines[index:]:
        if line.startswith("@@") or line[:1] not in "+- ":
            return False
        body = line[1:]
        if depth == 0 and _definition_name(body):
            return True
        if depth == 0 and not body.lstrip().startswith("@"):
            return False
        depth = max(
            0, depth + sum(body.count(c) for c in "([{") - sum(body.count(c) for c in ")]}")
        )
    return False


@dataclass(frozen=True)
class _File:
    tokens: int
    defines: frozenset[str]  # definitions this file's patch changes
    uses: frozenset[str]  # every identifier in the patch's code, context included
    context_names: frozenset[str]  # names whose excerpts belong with this file
    definitions: tuple[str, ...]  # lines of the definitions it changes, for other groups


def _code_lines(patch: str) -> str:
    return "\n".join(
        line[1:]
        for line in patch.splitlines()
        if line[:1] in "+- " and not line.startswith(("+++", "---"))
    )


def _facts(layer: PrLayer) -> dict[str, _File]:
    facts = {}
    for path, patch in layer.diffs.items():
        changed = changed_definitions(patch)
        defines = {name for name, _ in changed if len(name) >= 3}
        facts[path] = _File(
            tokens=estimate_tokens(patch),
            defines=frozenset(n for n in defines if not _DUNDER.match(n)),
            uses=frozenset(_IDENTIFIER.findall(_code_lines(patch))),
            # The PR layer finds excerpts by these names, so each one has a home.
            context_names=frozenset(names_in_patch(patch) | types_in_patch(patch)),
            definitions=tuple(line for _, line in changed),
        )
    return facts


def _tested_stem(path: str) -> str | None:
    """`test_refund.py`, `refund_test.go` and `refund.spec.ts` all test `refund`."""
    if not TEST_PATH.search(path):
        return None
    stem = PurePosixPath(path).name.split(".")[0]
    stem = stem.removeprefix("test_").removesuffix("_test")
    # tests/__init__.py or tests/conftest.py test nothing in particular.
    return None if not stem or _DUNDER.match(stem) or stem == "conftest" else stem


_TEST_DIRS = frozenset({"tests", "test", "spec", "specs", "__tests__"})


def _nearest(test: str, sources: list[str]) -> list[str]:
    """The sources sharing the most directories with the test; all of them on a tie."""
    where = set(PurePosixPath(test).parent.parts) - _TEST_DIRS

    def shared(source: str) -> int:
        return len(where & set(PurePosixPath(source).parent.parts))

    best = max(map(shared, sources), default=0)
    return [s for s in sources if shared(s) == best]


def core_units(layer: PrLayer, facts: dict[str, _File] | None = None) -> list[list[str]]:
    """Files joined by strong ties: changed definitions in use, and tests with their code."""
    facts = facts or _facts(layer)
    paths = sorted(layer.diffs)
    union = _UnionFind(paths)
    definers: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        for name in facts[path].defines:
            definers[name].append(path)
    for path in paths:
        for name in facts[path].uses & definers.keys():
            for definer in definers[name]:
                union.union(path, definer)
    sources: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        if not TEST_PATH.search(path):
            sources[PurePosixPath(path).name.split(".")[0]].append(path)
    for path in paths:
        stem = _tested_stem(path)
        if stem:
            for source in _nearest(path, sources.get(stem, [])):
                union.union(path, source)
    return union.groups()


def _signature_list(paths: Iterable[str], facts: dict[str, "_File"]) -> list[str]:
    return [
        f"{path}:\n" + "\n".join(f"  {line}" for line in facts[path].definitions)
        for path in sorted(paths)
        if facts[path].definitions
    ]


def signature_tokens(layer: PrLayer) -> int:
    """At most what any group is shown of the definitions other groups change.

    The caller reserves this, with the intent and file list every group carries, when
    working out `capacity_tokens`.
    """
    return estimate_tokens("\n".join(_signature_list(layer.diffs, _facts(layer))))


class _Sizer:
    """Tokens of a set of files plus the excerpts about them, each excerpt counted once."""

    def __init__(self, layer: PrLayer, facts: dict[str, _File]) -> None:
        self._facts = facts
        self._excerpts: dict[str, int] = defaultdict(int)
        for excerpt in layer.excerpts:
            self._excerpts[excerpt.name] += estimate_tokens(excerpt.text)

    def names(self, paths: Iterable[str]) -> set[str]:
        return set().union(*(self._facts[p].context_names for p in paths))

    def __call__(self, paths: Iterable[str]) -> int:
        paths = list(paths)
        diff_tokens = sum(self._facts[p].tokens for p in paths)
        return diff_tokens + sum(self._excerpts.get(n, 0) for n in self.names(paths))


def group_changes(
    layer: PrLayer, module_of: Callable[[str], str], capacity_tokens: int
) -> list[list[str]]:
    """Partition the changed files into groups that each fit in `capacity_tokens`.

    Returns one group holding every file when the whole PR fits. A strongly tied unit
    larger than the capacity becomes a group of its own rather than being split.
    """
    facts = _facts(layer)
    return _group(layer, facts, _Sizer(layer, facts), module_of, capacity_tokens)


def _group(
    layer: PrLayer,
    facts: dict[str, _File],
    size: _Sizer,
    module_of: Callable[[str], str],
    capacity_tokens: int,
) -> list[list[str]]:
    if size(layer.diffs) <= capacity_tokens:
        return [sorted(layer.diffs)] if layer.diffs else []
    by_module: dict[str, list[list[str]]] = defaultdict(list)
    for unit in core_units(layer, facts):
        by_module[module_of(unit[0])].append(unit)
    # Merge units of one module while they fit, then pack the rest first-fit decreasing.
    merged: list[list[str]] = []
    for module_units in by_module.values():
        current: list[str] = []
        for unit in sorted(module_units, key=lambda u: -size(u)):
            if current and size(current + unit) > capacity_tokens:
                merged.append(current)
                current = []
            current = current + unit
        if current:
            merged.append(current)
    groups: list[list[str]] = []
    for unit in sorted(merged, key=lambda u: -size(u)):
        for group in groups:
            if size(group + unit) <= capacity_tokens:
                group.extend(unit)
                break
        else:
            groups.append(list(unit))
    return [sorted(g) for g in groups]


def _copy(layer: PrLayer) -> PrLayer:
    """A layer whose lists are its own, so one group's changes stay in that group."""
    return replace(
        layer,
        diffs=dict(layer.diffs),
        issues=list(layer.issues),
        specs=list(layer.specs),
        files=list(layer.files),
        excerpts=list(layer.excerpts),
        previous_findings=list(layer.previous_findings),
        notes=list(layer.notes),
        signatures_elsewhere=list(layer.signatures_elsewhere),
    )


def grouped_layers(
    layer: PrLayer, module_of: Callable[[str], str], capacity_tokens: int
) -> list[PrLayer]:
    """One PR layer per group; the original layer when the whole PR fits."""
    facts = _facts(layer)
    size = _Sizer(layer, facts)
    groups = _group(layer, facts, size, module_of, capacity_tokens)
    if len(groups) <= 1:
        if size(layer.diffs) <= capacity_tokens:
            return [layer]
        return [replace(_copy(layer), group=OVERSIZED_NOTE.strip())]
    layers = []
    for index, group in enumerate(groups, start=1):
        members = set(group)
        names = size.names(group)
        elsewhere = _signature_list(set(layer.diffs) - members, facts)
        layers.append(
            replace(
                _copy(layer),
                diffs={p: layer.diffs[p] for p in group},
                excerpts=[e for e in layer.excerpts if e.name in names],
                group=(
                    f"Group {index} of {len(groups)}: this group reviews {len(group)} of the"
                    f" {len(layer.diffs)} changed files; the others are reviewed in parallel."
                    + (OVERSIZED_NOTE if size(group) > capacity_tokens else "")
                ),
                signatures_elsewhere=elsewhere,
            )
        )
    return layers
