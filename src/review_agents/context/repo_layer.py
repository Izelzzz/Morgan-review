"""The repository layer: what a reviewer knows about the repository before reading a PR.

It is read from the default branch, never from the pull request, so a PR cannot
rewrite the conventions it is reviewed against (decision D-005).
"""

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from review_agents.context.git import Snapshot
from review_agents.context.markup import untrusted
from review_agents.context.modules import (
    MODULE_TOKENS,
    ModuleSummary,
    Summarize,
    module_summaries,
)
from review_agents.context.tokens import estimate_tokens

# 2: the directory tree has no depth limit.
LAYER_VERSION = "2"
# The tree is folded only past this size, so it cannot crowd out the review's own
# content; a safety net, not a design limit (decision D-016).
TREE_TOKENS = 20_000
# Root files that state how the repository is meant to be written, in reading order.
CONVENTION_FILES = (
    "CLAUDE.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "README.md",
    "README.rst",
    "README.txt",
    "README",
    ".editorconfig",
    "pyproject.toml",
    "setup.cfg",
    "ruff.toml",
    ".ruff.toml",
    "package.json",
    "tsconfig.json",
    ".eslintrc",
    ".eslintrc.json",
    ".eslintrc.js",
    ".eslintrc.cjs",
    "eslint.config.js",
    "eslint.config.mjs",
    ".prettierrc",
    ".golangci.yml",
    ".golangci.yaml",
    "rustfmt.toml",
    "clippy.toml",
    ".rubocop.yml",
)
# Agent instructions that apply to a subdirectory, wherever they live.
NESTED_CONVENTION_NAMES = frozenset({"CLAUDE.md", "AGENTS.md"})

MODULE_NOTE = "Summaries are model-written and may be wrong; the code is authoritative."


@dataclass(frozen=True)
class RepoLayer:
    commit: str
    conventions: list[tuple[str, str]]
    tree: str
    modules: list[ModuleSummary]
    memories: list[str] = field(default_factory=list)
    cache_hit: bool = False

    def render(self) -> str:
        parts = [untrusted(f"convention:{path}", text) for path, text in self.conventions]
        parts.append(untrusted("directory-tree", self.tree))
        module_map = "\n".join(m.render() for m in self.modules)
        parts.append(f"{MODULE_NOTE}\n{untrusted('module-map', module_map)}")
        if self.memories:
            parts.append(untrusted("memories", "\n".join(f"- {m}" for m in self.memories)))
        return "\n\n".join(parts)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.render())


def convention_paths(snapshot: Snapshot) -> list[str]:
    paths = {entry.path for entry in snapshot.files()}
    root = [name for name in CONVENTION_FILES if name in paths]
    nested = sorted(
        p for p in paths if "/" in p and PurePosixPath(p).name in NESTED_CONVENTION_NAMES
    )
    return root + nested


def directory_tree(snapshot: Snapshot, max_tokens: int = TREE_TOKENS) -> str:
    """Every directory, with the number of files beneath it.

    Only a tree larger than `max_tokens` is folded, deepest levels first, and says so.
    """
    counts: Counter[str] = Counter()
    for entry in snapshot.files():
        parts = PurePosixPath(entry.path).parts[:-1]
        for level in range(1, len(parts) + 1):
            counts["/".join(parts[:level])] += 1
    header = f"./ ({len(snapshot.files())} files)"

    def render(depth: int | None) -> str:
        lines = [header]
        for path in sorted(counts):
            level = path.count("/")
            if depth is None or level < depth:
                name = PurePosixPath(path).name
                lines.append(f"{'  ' * (level + 1)}{name}/ ({counts[path]} files)")
        if depth is not None:
            hidden = sum(1 for p in counts if p.count("/") >= depth)
            # Files beneath the folded directories: those under the first hidden level.
            files = sum(n for p, n in counts.items() if p.count("/") == depth)
            folded = f"deeper than {depth} levels" if depth else "at every level"
            lines.append(
                f"(directories {folded} are folded to fit: {hidden} directories, {files} files)"
            )
        return "\n".join(lines)

    tree = render(None)
    deepest = max((p.count("/") + 1 for p in counts), default=0)
    depth = deepest - 1
    while estimate_tokens(tree) > max_tokens and depth >= 0:
        tree = render(depth)
        depth -= 1
    return tree


def _key(snapshot: Snapshot, memory_version: str, tree_tokens: int) -> str:
    raw = f"{LAYER_VERSION}:{snapshot.tree_id}:{memory_version}:{tree_tokens}"
    return "layer:" + hashlib.sha256(raw.encode()).hexdigest()


def build_repo_layer(
    conn: sqlite3.Connection,
    repo: str,
    git_dir: Path,
    default_branch: str,
    skip_paths: list[str],
    summarize: Summarize | None = None,
    memories: list[str] | None = None,
    memory_version: str = "",
    module_tokens: int = MODULE_TOKENS,
    tree_tokens: int = TREE_TOKENS,
) -> RepoLayer:
    """The repository layer at the tip of the default branch.

    Conventions and the directory tree are cached under the default branch's tree id,
    so any change to the branch, a README edit included, rebuilds them. Module summaries
    are cached per module and only regenerated for modules that changed; pass
    `summarize=None` to use only the summaries that already exist.
    """
    snapshot = Snapshot.at_branch(git_dir, default_branch, tuple(skip_paths))
    key = _key(snapshot, memory_version, tree_tokens)
    row = conn.execute(
        "SELECT content FROM context_cache WHERE repo = ? AND layer = 'repo' AND cache_key = ?",
        (repo, key),
    ).fetchone()
    if row:
        cached = json.loads(row["content"])
        conventions = [(path, text) for path, text in cached["conventions"]]
        tree = cached["tree"]
    else:
        conventions = []
        for path in convention_paths(snapshot):
            text = snapshot.read_text(path)
            if text is not None:
                conventions.append((path, text))
        tree = directory_tree(snapshot, tree_tokens)
        content = json.dumps({"conventions": conventions, "tree": tree})
        conn.execute(
            "INSERT OR REPLACE INTO context_cache (repo, layer, cache_key, content, tokens)"
            " VALUES (?, 'repo', ?, ?, ?)",
            (repo, key, content, estimate_tokens(content)),
        )
    return RepoLayer(
        commit=snapshot.commit,
        conventions=conventions,
        tree=tree,
        modules=module_summaries(conn, repo, snapshot, summarize, module_tokens),
        memories=memories or [],
        cache_hit=row is not None,
    )
