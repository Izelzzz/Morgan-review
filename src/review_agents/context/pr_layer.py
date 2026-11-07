"""The pull request layer: the PR's intent and its complete changes.

Read from the PR's own commits. Nothing in the changes is dropped (decision D-008):
files the skip rules exclude are still listed, with the reason. The code around the
changes is left to the agents and their tools unless asked for (decision D-015).
"""

import re
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from review_agents.context.markup import untrusted
from review_agents.context.tokens import estimate_tokens
from review_agents.mirror import GitError, check_commit_sha, check_repo_name, run_git
from review_agents.repo_config import RepoConfig
from review_agents.tools import MAX_ISSUE_CHARS, RepoTools, ToolError, glob_match

GIT_TIMEOUT_SECONDS = 30.0
# A name used in more places than this is too generic to expand usefully; the agent
# can still search for it.
MAX_USAGES_PER_NAME = 30
# A safety net against a description that links thousands of issues (decision D-017).
MAX_LINKED_ISSUES = 100
ISSUE_FETCH_WORKERS = 8
TYPE_DEFINITION_LINES = 20

NO_SPEC_NOTE = "No specification found; review against the PR description only."

_ISSUE_REF = re.compile(r"(?<![\w/#])#(\d+)\b")
DEFINITION = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|protected\s+)?(?:static\s+)?"
    r"(?:async\s+)?(?:def|class|function|interface|type|struct|enum|trait|fn)\s+([A-Za-z_]\w*)"
    r"|^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"
)
_TYPE_NAME = re.compile(r"\b([A-Z][A-Za-z0-9_]*[a-z][A-Za-z0-9_]*)\b")
TEST_PATH = re.compile(
    r"(^|/)(tests?|spec|specs|__tests__)/|(^|/)test_[^/]+$|_test\.[^/]+$|\.(test|spec)\.[^/]+$"
)


@dataclass(frozen=True)
class PrRequest:
    repo: str
    pr: int
    title: str
    body: str
    base_sha: str
    head_sha: str
    # Set for rounds 2 and 3: the head the previous round reviewed.
    previous_head: str | None = None


@dataclass(frozen=True)
class ChangedFile:
    status: str  # A, M, D, R (renamed), C (copied), T (type changed)
    path: str
    old_path: str | None = None
    skipped: bool = False

    def describe(self) -> str:
        name = f"{self.old_path} -> {self.path}" if self.old_path else self.path
        note = " (matches a skip rule; not shown)" if self.skipped else ""
        return f"{self.status} {name}{note}"


@dataclass(frozen=True)
class Excerpt:
    kind: str  # "definition", "usage" or "test"
    name: str
    path: str
    text: str


@dataclass
class PrLayer:
    request: PrRequest
    issues: list[tuple[int, str]]
    specs: list[tuple[str, str, bool]]  # path, text, changed in this PR
    files: list[ChangedFile]
    diffs: dict[str, str]
    excerpts: list[Excerpt]
    previous_findings: list[str]
    notes: list[str] = field(default_factory=list)
    # Set when a large PR is reviewed in groups: what the other groups change (D-008).
    group: str | None = None
    signatures_elsewhere: list[str] = field(default_factory=list)

    def render(self) -> str:
        req = self.request
        parts = [
            "## Pull request",
            untrusted("pr-description", f"#{req.pr} {req.title}\n\n{req.body}".strip()),
        ]
        for number, text in self.issues:
            parts.append(untrusted(f"issue:{number}", text))
        if self.specs:
            for path, text, changed in self.specs:
                label = f"spec:{path}" + (" (modified by this PR)" if changed else "")
                parts.append(untrusted(label, text))
        else:
            parts.append(NO_SPEC_NOTE)
        parts.append("## Changed files")
        parts.append("\n".join(f.describe() for f in self.files))
        heading = "## Changes"
        if req.previous_head:
            heading += f" since the previous review ({req.previous_head[:12]})"
        parts.append(heading)
        parts.extend(untrusted(f"diff:{path}", text) for path, text in self.diffs.items())
        if self.excerpts:
            parts.append("## Code around the changes")
            for excerpt in self.excerpts:
                source = f"{excerpt.kind}:{excerpt.name}:{excerpt.path}"
                parts.append(untrusted(source, excerpt.text))
        if self.previous_findings:
            parts.append("## Findings still open from earlier rounds")
            parts.append(untrusted("previous-findings", "\n".join(self.previous_findings)))
        if self.group:
            parts.append(f"## Review group\n{self.group}")
        if self.signatures_elsewhere:
            parts.append("## Definitions changed elsewhere in this PR")
            parts.append(untrusted("other-groups", "\n".join(self.signatures_elsewhere)))
        if self.notes:
            parts.append("## Notes")
            parts.append("\n".join(f"- {n}" for n in self.notes))
        return "\n\n".join(parts)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.render())


class _Git:
    def __init__(self, git_dir: Path) -> None:
        self.git_dir = git_dir

    def __call__(self, *args: str, ok_codes: tuple[int, ...] = (0,)) -> str:
        output = run_git(self.git_dir, *args, ok_codes=ok_codes, timeout=GIT_TIMEOUT_SECONDS)
        return output.decode("utf-8", "replace")


def changed_files(git: _Git, base: str, head: str, skip_paths: list[str]) -> list[ChangedFile]:
    output = git("diff", "--name-status", "-z", "-M", "--no-ext-diff", f"{base}...{head}")
    fields = output.split("\0")
    files = []
    i = 0
    while i < len(fields) and fields[i]:
        status = fields[i]
        if status[0] in "RC":
            old, new = fields[i + 1], fields[i + 2]
            i += 3
        else:
            old, new = None, fields[i + 1]
            i += 2
        skipped = any(glob_match(pattern, new) for pattern in skip_paths)
        files.append(ChangedFile(status[0], new, old, skipped))
    return files


def file_diff(git: _Git, since: str, head: str, changed: ChangedFile, three_dot: bool) -> str:
    """The file's patch with whole functions around each change (`git diff -W`)."""
    span = f"{since}...{head}" if three_dot else f"{since}..{head}"
    paths = [changed.path] if changed.old_path is None else [changed.old_path, changed.path]
    return git("diff", "-W", "-M", "--no-color", "--no-ext-diff", span, "--", *paths).strip()


def names_in_patch(patch: str) -> set[str]:
    """Functions and types a patch changes.

    A `git diff -W` hunk spans the whole function around a change, so every definition
    inside a hunk, changed or not, is one the change touches; hunk headers name the
    enclosing function when the hunk starts inside it.
    """
    names: set[str] = set()
    in_hunk = False
    for line in patch.splitlines():
        if line.startswith("@@"):
            in_hunk = True
            match = DEFINITION.search(line.split("@@", 2)[-1])
        elif in_hunk and line[:1] in "+- ":
            match = DEFINITION.search(line[1:])
        else:
            continue
        if match:
            names.add(match.group(1) or match.group(2))
    return {n for n in names if len(n) >= 3}


def types_in_patch(patch: str) -> set[str]:
    names: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            names.update(_TYPE_NAME.findall(line[1:]))
    return names


def _grep_blocks(output: str, head: str) -> dict[str, list[str]]:
    """Group `git grep --null` output with context into blocks per file.

    With --null, path, line number and text are separated by NUL bytes, so paths
    containing "-" or ":" parse safely.
    """
    blocks: dict[str, list[str]] = defaultdict(list)
    current: list[str] = []
    current_path = ""
    prefix = f"{head}:"
    for line in output.splitlines():
        if line == "--" or "\0" not in line:
            if current:
                blocks[current_path].append("\n".join(current))
            current = []
            continue
        fields = line.removeprefix(prefix).split("\0", 2)
        if len(fields) != 3 or not fields[1].isdigit():
            continue
        path, number, text = fields
        if current and path != current_path:
            blocks[current_path].append("\n".join(current))
            current = []
        current_path = path
        current.append(f"{number}: {text}")
    if current:
        blocks[current_path].append("\n".join(current))
    return blocks


def find_surrounding_code(
    git: _Git,
    head: str,
    names: set[str],
    types: set[str],
    changed_paths: set[str],
    skip_paths: list[str],
    notes: list[str],
) -> list[Excerpt]:
    def keep(path: str) -> bool:
        return path not in changed_paths and not any(glob_match(p, path) for p in skip_paths)

    excerpts: list[Excerpt] = []
    for name in sorted(names):
        matches = git("grep", "-I", "-w", "-c", "--null", "-e", name, head, ok_codes=(0, 1))
        count = sum(int(line.split("\0", 1)[1]) for line in matches.splitlines() if "\0" in line)
        if count > MAX_USAGES_PER_NAME:
            notes.append(
                f"`{name}` is used in {count} places; usages not expanded, use search_code."
            )
            continue
        # -W shows the whole function around each use: the caller, or the test.
        output = git("grep", "-n", "-I", "-w", "-W", "--null", "-e", name, head, ok_codes=(0, 1))
        for path, blocks in sorted(_grep_blocks(output, head).items()):
            if keep(path):
                kind = "test" if TEST_PATH.search(path) else "usage"
                excerpts.append(Excerpt(kind, name, path, "\n...\n".join(blocks)))
    for type_name in sorted(types - names):
        pattern = rf"(class|interface|type|struct|enum|trait)\s+{type_name}\b"
        output = git(
            "grep", "-n", "-I", "-E", "--null", f"-A{TYPE_DEFINITION_LINES}", "-e", pattern, head,
            ok_codes=(0, 1),
        )  # fmt: skip
        for path, blocks in sorted(_grep_blocks(output, head).items()):
            if keep(path):
                excerpts.append(Excerpt("definition", type_name, path, "\n...\n".join(blocks)))
    return excerpts


def linked_issues(tools: RepoTools, body: str, notes: list[str]) -> list[tuple[int, str]]:
    # Each linked issue is read in full (decision D-016).
    numbers = list(dict.fromkeys(int(n) for n in _ISSUE_REF.findall(body)))
    if len(numbers) > MAX_LINKED_ISSUES:
        notes.append(
            f"The description links {len(numbers)} issues; {len(numbers) - MAX_LINKED_ISSUES}"
            " more were not read; read_issue fetches one, its body cut at"
            f" {MAX_ISSUE_CHARS:,} characters."
        )
        numbers = numbers[:MAX_LINKED_ISSUES]

    def fetch(number: int) -> str | ToolError:
        try:
            return tools.issue_text(number)
        except ToolError as exc:
            return exc

    # Fetched side by side: one at a time, 100 slow answers would take many minutes.
    with ThreadPoolExecutor(max_workers=ISSUE_FETCH_WORKERS) as pool:
        results = list(pool.map(fetch, numbers))
    issues = []
    for number, result in zip(numbers, results, strict=True):
        if isinstance(result, ToolError):
            notes.append(str(result))
        else:
            issues.append((number, result))
    return issues


def find_specs(
    git: _Git,
    head: str,
    config: RepoConfig,
    files: list[ChangedFile],
    texts: list[str],
) -> list[tuple[str, str, bool]]:
    """Spec files this PR changes, or that its description or linked issues mention.

    Read from the PR's commit; a spec the PR modifies is marked so (decision D-006).
    """
    if not config.specs:
        return []
    listing = git("ls-tree", "-r", "--name-only", "-z", head).split("\0")
    candidates = [p for p in listing if p and any(glob_match(g, p) for g in config.specs)]
    changed = {f.path for f in files}
    mentioned = "\n".join(texts)
    specs = []
    for path in candidates:
        if path in changed or path in mentioned:
            try:
                text = git("cat-file", "blob", f"{head}:{path}")
            except GitError:
                continue
            specs.append((path, text, path in changed))
    return specs


def previous_findings(conn: sqlite3.Connection, repo: str, pr: int) -> list[str]:
    # The same finding is stored once per round; keep its latest copy.
    rows = conn.execute(
        "SELECT f.severity, f.title, f.file, f.line_start, f.line_end FROM findings f"
        " WHERE f.id IN (SELECT max(f2.id) FROM findings f2"
        "   JOIN reviews r ON r.id = f2.review_id WHERE r.repo = ? AND r.pr = ?"
        "   GROUP BY f2.fingerprint)"
        " AND f.status = 'open' ORDER BY f.id",
        (repo, pr),
    ).fetchall()
    return [
        f"[{row['severity']}] {row['title']} ({row['file']}:{row['line_start']}-{row['line_end']})"
        for row in rows
    ]


def build_pr_layer(
    conn: sqlite3.Connection,
    request: PrRequest,
    git_dir: Path,
    config: RepoConfig,
    token: str | None = None,
    api_base: str = "https://api.github.com",
    surrounding_code: bool = False,
) -> PrLayer:
    """`surrounding_code` also preloads the callers, tests and types of changed code."""
    check_repo_name(request.repo)
    for sha in (request.base_sha, request.head_sha, request.previous_head):
        if sha is not None:
            check_commit_sha(sha)
    git = _Git(git_dir)
    skip = config.effective_skip_paths
    notes: list[str] = []
    head = request.head_sha

    files = changed_files(git, request.base_sha, head, skip)
    since = request.previous_head or request.base_sha
    diffs = {}
    for changed in files:
        if changed.skipped:
            continue
        patch = file_diff(git, since, head, changed, three_dot=request.previous_head is None)
        if patch:
            diffs[changed.path] = patch

    tools = RepoTools(git_dir, request.repo, head, skip, token=token, api_base=api_base)
    issues = linked_issues(tools, request.body, notes)
    specs = find_specs(git, head, config, files, [request.body, *(t for _, t in issues)])
    excerpts: list[Excerpt] = []
    if surrounding_code:
        names = set().union(*map(names_in_patch, diffs.values()))
        types = set().union(*map(types_in_patch, diffs.values()))
        changed_paths = {f.path for f in files}
        excerpts = find_surrounding_code(git, head, names, types, changed_paths, skip, notes)
    return PrLayer(
        request=request,
        issues=issues,
        specs=specs,
        files=files,
        diffs=diffs,
        excerpts=excerpts,
        previous_findings=previous_findings(conn, request.repo, request.pr),
        notes=notes,
    )
