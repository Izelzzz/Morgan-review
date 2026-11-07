"""Read-only tools the review agents use to look at a repository at one commit.

Everything is read straight from git objects in the mirror. No files are checked
out, so a symlink or a crafted path in the pull request cannot reach the host.
"""

import functools
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from functools import cache
from http.client import HTTPException
from pathlib import Path, PurePosixPath

from review_agents.mirror import (
    GitError,
    GitTimeoutError,
    check_commit_sha,
    check_repo_name,
    run_git,
)
from review_agents.repo_config import MAX_FILE_BYTES

GIT_TIMEOUT_SECONDS = 30.0
MAX_SEARCH_MATCHES = 100
MAX_LOG_ENTRIES = 50
MAX_ISSUE_CHARS = 20_000
_BINARY_PROBE_BYTES = 8_000

LIMIT_REACHED = (
    "Tool call limit reached. Do not call any more tools; "
    "finish now with the findings you already have."
)


class ToolError(ValueError):
    """A request the tool refuses; its message is returned to the agent."""


class ToolTimeoutError(ToolError):
    """Git took too long: the request may be fine, it could only not be answered now."""


@cache
def _glob_regex(pattern: str) -> re.Pattern[str]:
    # `**/` matches any number of directories, `**` anything, `*` and `?` stay in one segment.
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out)


def glob_match(pattern: str, path: str) -> bool:
    return _glob_regex(pattern).fullmatch(path) is not None


def clean_path(path: str) -> str:
    """A path inside the repository, spelled one way; ToolError for one that leaves it."""
    stripped = path.strip().strip("/")
    if stripped in ("", "."):
        return ""
    if "\\" in stripped or ".." in PurePosixPath(stripped).parts:
        raise ToolError(f"path must stay inside the repository: {path!r}")
    return str(PurePosixPath(stripped))


class RepoTools:
    """The agents' only view of the repository: read files, list, search, history, issues."""

    NAMES = ("read_file", "list_dir", "search_code", "git_log", "read_issue")

    def __init__(
        self,
        git_dir: Path,
        repo: str,
        commit: str,
        skip_paths: Sequence[str],
        token: str | None = None,
        api_base: str = "https://api.github.com",
        git_timeout: float = GIT_TIMEOUT_SECONDS,
    ) -> None:
        self._git_dir = git_dir
        self._repo = check_repo_name(repo)
        self._commit = check_commit_sha(commit)
        self._skip_paths = tuple(skip_paths)
        self._token = token
        self._api_base = api_base.rstrip("/")
        self._git_timeout = git_timeout

    def is_skipped(self, path: str) -> bool:
        return any(glob_match(pattern, path) for pattern in self._skip_paths)

    def _git(self, *args: str, ok_codes: tuple[int, ...] = (0,)) -> bytes:
        try:
            return run_git(self._git_dir, *args, ok_codes=ok_codes, timeout=self._git_timeout)
        except GitTimeoutError as exc:
            # Not a GitError for the callers below, which read those as "not found".
            raise ToolTimeoutError(str(exc)) from None

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a text file at the reviewed commit, optionally a 1-based line range."""
        clean = clean_path(path)
        if not clean:
            raise ToolError("give a file path")
        if self.is_skipped(clean):
            raise ToolError(f"{clean} is in the skip list and is not read")
        try:
            kind = self._git("cat-file", "-t", f"{self._commit}:{clean}").decode().strip()
            size = int(self._git("cat-file", "-s", f"{self._commit}:{clean}"))
        except GitError:
            raise ToolError(f"{clean} does not exist at this commit") from None
        if kind != "blob":
            raise ToolError(f"{clean} is a directory; use list_dir")
        if size > MAX_FILE_BYTES:
            raise ToolError(f"{clean} is larger than {MAX_FILE_BYTES} bytes and is not read")
        data = self._git("cat-file", "blob", f"{self._commit}:{clean}")
        if b"\0" in data[:_BINARY_PROBE_BYTES]:
            raise ToolError(f"{clean} is a binary file and is not read")
        lines = data.decode("utf-8", "replace").splitlines()
        start = max(start_line, 1)
        end = len(lines) if end_line is None else min(end_line, len(lines))
        return "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, end + 1))

    def list_dir(self, path: str = "") -> str:
        """List one directory at the reviewed commit: type, size and name per entry."""
        clean = clean_path(path)
        if clean and self.is_skipped(f"{clean}/"):
            raise ToolError(f"{clean} is in the skip list and is not read")
        target = f"{self._commit}:{clean}" if clean else f"{self._commit}^{{tree}}"
        try:
            output = self._git("ls-tree", "-l", target).decode("utf-8", "replace")
        except GitError:
            raise ToolError(f"{clean or '.'} is not a directory at this commit") from None
        entries = []
        for line in output.splitlines():
            meta, name = line.split("\t", 1)
            _, kind, _, size = meta.split()
            full = f"{clean}/{name}" if clean else name
            note = " (skipped)" if self.is_skipped(full) else ""
            entries.append(f"{'dir ' if kind == 'tree' else 'file'} {size:>8} {full}{note}")
        return "\n".join(entries) or "(empty)"

    def search_code(self, pattern: str, path_glob: str | None = None) -> str:
        """Search text files for a regular expression; returns `path:line: text` matches."""
        if not pattern:
            raise ToolError("give a pattern to search for")
        try:
            # Exit code 1 means no match.
            output = self._git(
                "grep",
                "-n",
                "-I",
                "-E",
                "-e",
                pattern,
                self._commit,
                ok_codes=(0, 1),
            ).decode("utf-8", "replace")
        except GitError as exc:
            raise ToolError(f"search failed: {exc}") from None
        matches = []
        for line in output.splitlines():
            location = line.removeprefix(f"{self._commit}:")
            path = location.split(":", 1)[0]
            if self.is_skipped(path) or (path_glob and not glob_match(path_glob, path)):
                continue
            matches.append(location)
            if len(matches) == MAX_SEARCH_MATCHES:
                matches.append(f"(stopped after {MAX_SEARCH_MATCHES} matches; narrow the search)")
                break
        return "\n".join(matches) or "no matches"

    def git_log(self, path: str | None = None, max_entries: int = 20) -> str:
        """Show recent commits up to the reviewed commit, optionally only those touching a path."""
        count = min(max(max_entries, 1), MAX_LOG_ENTRIES)
        args = ["log", f"--max-count={count}", "--format=%h %ad %an%n    %s", "--date=short"]
        args.append(self._commit)
        clean = clean_path(path) if path else ""
        if clean:
            args += ["--", clean]
        return self._git(*args).decode("utf-8", "replace").strip() or "no commits"

    def line_count(self, path: str) -> int | None:
        """Lines in a file of the reviewed commit, or None when it does not exist.

        Not an agent tool: the pipeline uses it to check where a finding points. A git
        that times out raises ToolTimeoutError: that says nothing about the file.
        """
        try:
            clean = clean_path(path)
        except ToolError:
            return None
        try:
            data = self._git("cat-file", "blob", f"{self._commit}:{clean}")
        except ToolTimeoutError:
            raise
        except (GitError, ToolError):
            return None
        return data.count(b"\n") + (0 if data.endswith(b"\n") or not data else 1)

    def read_issue(self, number: int) -> str:
        """Read an issue or pull request of this repository: title, state, labels and body."""
        return self.issue_text(number, MAX_ISSUE_CHARS)

    def issue_text(self, number: int, max_chars: int | None = None) -> str:
        """`read_issue` without the agents' cap on the body: the PR layer reads it all."""
        if number < 1:
            raise ToolError("issue numbers start at 1")
        request = urllib.request.Request(  # noqa: S310 - fixed https API base, GET only
            f"{self._api_base}/repos/{self._repo}/issues/{number}",
            headers={"Accept": "application/vnd.github+json"}
            | ({"Authorization": f"Bearer {self._token}"} if self._token else {}),
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                issue = json.load(response)
            body = (issue.get("body") or "")[:max_chars]
            labels = ", ".join(label["name"] for label in issue.get("labels", []))
            title, state = issue["title"], issue["state"]
        except urllib.error.HTTPError as exc:
            raise ToolError(f"issue #{number} could not be read: HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError):
            raise ToolError(f"issue #{number} could not be read: GitHub is unreachable") from None
        # A proxy's error page, a cut-off answer or JSON of another shape.
        except (ValueError, KeyError, TypeError, AttributeError, OSError, HTTPException):
            raise ToolError(f"issue #{number} could not be read: not an issue") from None
        return f"#{number} {title} [{state}] labels: {labels or '-'}\n\n{body}"


class ToolBudget:
    """Caps one agent's tool calls; past the cap a call returns `LIMIT_REACHED` instead."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def wrap(self, tool: Callable[..., str]) -> Callable[..., str]:
        @functools.wraps(tool)
        def limited(*args: object, **kwargs: object) -> str:
            if self.used >= self.limit:
                return LIMIT_REACHED
            self.used += 1
            try:
                return tool(*args, **kwargs)
            except ToolError as exc:
                return f"error: {exc}"

        return limited


def agent_tools(tools: RepoTools, budget: ToolBudget) -> dict[str, Callable[..., str]]:
    """The complete tool set one agent receives. Nothing here writes or executes."""
    return {name: budget.wrap(getattr(tools, name)) for name in RepoTools.NAMES}
