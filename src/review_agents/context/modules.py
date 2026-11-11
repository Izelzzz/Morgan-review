"""Split a repository into modules and keep a model-written summary of each.

Every module is summarised, with no cap on how many: the review needs the whole
structure to judge module boundaries (decision D-007). A module is as large as one
summary can read in full, and the summary reads all of it (decisions D-014, D-016).
A summary is cached by the content of the module's files (their git blob ids), so
only modules whose files changed are summarised again; a change to a skipped file,
such as a lockfile, does not count.
"""

import hashlib
import json
import logging
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

from review_agents.context.git import Entry, Snapshot
from review_agents.context.tokens import estimate_tokens
from review_agents.models import BudgetExceededError
from review_agents.repo_config import MAX_FILE_BYTES

log = logging.getLogger(__name__)


def module_tokens(context_tokens: int) -> int:
    """Room for one module's source in a summary request to a model with this window.

    A fifth is left for the instructions, the reply, and tokenizers that count more
    than ours does.
    """
    return context_tokens * 4 // 5


MODULE_TOKENS = module_tokens(128_000)
# Before the files are read, a file's size gives its tokens: three bytes per token
# overestimates code a little and is about right for CJK text.
_BYTES_PER_TOKEN = 3
# A directory holding one of these starts its own module, whatever its size.
PACKAGE_MARKERS = frozenset(
    {
        "package.json",
        "pyproject.toml",
        "setup.py",
        "go.mod",
        "Cargo.toml",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "composer.json",
        "Gemfile",
        "mix.exs",
        "CMakeLists.txt",
    }
)
_ENTRY_NAMES = (
    "__init__.py",
    "__main__.py",
    "main.py",
    "index.js",
    "index.ts",
    "main.go",
    "lib.rs",
    "main.rs",
    "mod.rs",
)
_LANGUAGES = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".go": "Go", ".rs": "Rust", ".java": "Java", ".kt": "Kotlin",
    ".rb": "Ruby", ".php": "PHP", ".cs": "C#", ".c": "C", ".h": "C", ".cc": "C++",
    ".cpp": "C++", ".hpp": "C++", ".swift": "Swift", ".scala": "Scala", ".ex": "Elixir",
    ".sh": "Shell", ".sql": "SQL",
}  # fmt: skip

# 2: summaries read the whole module instead of a sample.
SUMMARY_VERSION = "2"


@dataclass(frozen=True)
class Module:
    path: str  # "" for files at the repository root
    files: tuple[Entry, ...]
    # (n, of): set when a directory's own files are too many for one summary.
    part: tuple[int, int] | None = None

    @property
    def language(self) -> str:
        counts = Counter(_LANGUAGES.get(PurePosixPath(f.path).suffix, "") for f in self.files)
        counts.pop("", None)
        return counts.most_common(1)[0][0] if counts else "-"

    @property
    def entry(self) -> str | None:
        here = PurePosixPath(self.path or ".")
        own = {
            PurePosixPath(f.path).name: f.path
            for f in self.files
            if PurePosixPath(f.path).parent == here
        }
        return next((own[name] for name in _ENTRY_NAMES if name in own), None)

    @property
    def name(self) -> str:
        return f"{self.path or '.'}{self._part_label}"

    @property
    def _part_label(self) -> str:
        return f" (part {self.part[0]} of {self.part[1]})" if self.part else ""

    def stats(self) -> str:
        entry = f" · entry {self.entry}" if self.entry else ""
        name = f"{self.path or '.'}/{self._part_label}"
        return f"{name} · {len(self.files)} files · {self.language}{entry}"


def size_tokens(entry: Entry) -> int:
    """A file's share of a summary request, judged before reading it.

    Its content counts from its size (files too large to read have none), and its path
    twice: once in the file list and once in the header above its content.
    """
    content = 0 if entry.size > MAX_FILE_BYTES else math.ceil(entry.size / _BYTES_PER_TOKEN)
    return content + 2 * math.ceil(len(entry.path) / _BYTES_PER_TOKEN) + 2


def split_modules(files: Iterable[Entry], module_tokens: int = MODULE_TOKENS) -> list[Module]:
    """Directories become modules once one summary can read their whole subtree.

    Larger directories are split into their subdirectories; files directly inside a
    split directory form a module of their own, read in parts when even they are too
    large. A package marker always starts a module.
    """
    by_dir: dict[str, list[Entry]] = defaultdict(list)
    children: dict[str, set[str]] = defaultdict(set)
    for entry in files:
        parent = str(PurePosixPath(entry.path).parent)
        parent = "" if parent == "." else parent
        by_dir[parent].append(entry)
        parts = PurePosixPath(parent).parts
        for depth in range(len(parts)):
            above = "/".join(parts[:depth])
            children[above].add("/".join(parts[: depth + 1]))

    def subtree(path: str) -> list[Entry]:
        found = list(by_dir.get(path, []))
        for child in sorted(children.get(path, ())):
            found.extend(subtree(child))
        return found

    def has_marker(path: str) -> bool:
        return any(PurePosixPath(f.path).name in PACKAGE_MARKERS for f in by_dir.get(path, []))

    def marked_below(path: str) -> bool:
        return any(has_marker(c) or marked_below(c) for c in children.get(path, ()))

    def tokens(entries: Iterable[Entry]) -> int:
        return sum(size_tokens(e) for e in entries)

    modules: list[Module] = []

    def visit(path: str, is_root: bool) -> None:
        all_files = subtree(path)
        if not is_root and tokens(all_files) <= module_tokens and not marked_below(path):
            modules.append(Module(path, tuple(all_files)))
            return
        own = by_dir.get(path, [])
        if tokens(own) <= module_tokens:
            if own:
                modules.append(Module(path, tuple(own)))
        else:
            modules.extend(_parts(path, own, module_tokens))
        for child in sorted(children.get(path, ())):
            visit(child, is_root=False)

    visit("", is_root=True)
    return sorted(modules, key=lambda m: (m.path, m.part or (0, 0)))


def _parts(path: str, files: list[Entry], module_tokens: int) -> list[Module]:
    """A directory's own files in path order, cut into runs that each fit one summary."""
    runs: list[list[Entry]] = [[]]
    used = 0
    for entry in sorted(files, key=lambda e: e.path):
        size = size_tokens(entry)
        if runs[-1] and used + size > module_tokens:
            runs.append([])
            used = 0
        runs[-1].append(entry)
        used += size
    if len(runs) == 1:  # one file too large on its own: there is nothing to split
        return [Module(path, tuple(runs[0]))]
    return [Module(path, tuple(run), (i, len(runs))) for i, run in enumerate(runs, start=1)]


@dataclass(frozen=True)
class ModuleSummary:
    path: str
    stats: str
    summary: str | None  # None until a summary exists; the stats stand in for it

    def render(self) -> str:
        if self.summary is None:
            return f"- {self.stats}"
        return f"- {self.stats}\n  {self.summary}"


Summarize = Callable[[str, str], str]
"""Takes a module's name and its whole source; returns a 3-5 sentence summary."""


def module_source(snapshot: Snapshot, module: Module) -> str:
    """What the model reads for a summary: the file list, then every readable file in full."""
    listing = "\n".join(f.path for f in module.files)
    parts = [f"Files:\n{listing}"]
    for entry in module.files:
        text = snapshot.read_text(entry.path)
        if text:
            parts.append(f"--- {entry.path} ---\n{text}")
    return "\n\n".join(parts)


def _cache_key(module: Module) -> str:
    content = hashlib.sha256("\n".join(f"{f.path}={f.blob}" for f in module.files).encode())
    part = f"#{module.part[0]}" if module.part else ""
    return f"module:{SUMMARY_VERSION}:{module.path or '.'}{part}:{content.hexdigest()}"


def module_summaries(
    conn: sqlite3.Connection,
    repo: str,
    snapshot: Snapshot,
    summarize: Summarize | None,
    module_tokens: int = MODULE_TOKENS,
) -> list[ModuleSummary]:
    """Summaries for every module, generating the missing ones when `summarize` is given.

    A failed summary falls back to the module's git stats and does not stop the others.
    """
    results = []
    for module in split_modules(snapshot.files(), module_tokens):
        key = _cache_key(module)
        row = conn.execute(
            "SELECT content FROM context_cache WHERE repo = ? AND layer = 'repo' AND cache_key = ?",
            (repo, key),
        ).fetchone()
        summary: str | None = json.loads(row["content"])["summary"] if row else None
        if summary is None and summarize is not None:
            source = module_source(snapshot, module)
            # The size was judged before reading; a request that cannot fit is not sent.
            if estimate_tokens(source) > module_tokens:
                log.warning("module %s of %s is too large for one summary", module.name, repo)
                results.append(ModuleSummary(module.path, module.stats(), None))
                continue
            try:
                summary = summarize(module.name, source).strip()
            except BudgetExceededError:
                log.warning(
                    "summary budget used up; %s and later modules keep their stats",
                    module.name,
                )
                summarize = None
            except Exception:
                log.exception("summarising module %s of %s failed", module.name, repo)
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO context_cache (repo, layer, cache_key, content, tokens)"
                    " VALUES (?, 'repo', ?, ?, ?)",
                    (repo, key, json.dumps({"summary": summary}), estimate_tokens(summary)),
                )
        results.append(ModuleSummary(module.path, module.stats(), summary))
    return results


def to_json(summaries: list[ModuleSummary]) -> list[dict[str, str | None]]:
    return [asdict(s) for s in summaries]
