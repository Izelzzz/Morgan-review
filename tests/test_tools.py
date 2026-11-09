import inspect
import json
import os
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from gitrepo import Upstream, install_fake_git

from review_agents.mirror import MirrorStore
from review_agents.repo_config import DEFAULT_SKIP_PATHS, MAX_FILE_BYTES
from review_agents.tools import (
    LIMIT_REACHED,
    RepoTools,
    ToolBudget,
    ToolError,
    ToolTimeoutError,
    agent_tools,
    glob_match,
)


@pytest.fixture
def checkout(tmp_path: Path) -> tuple[Path, str]:
    upstream = Upstream(tmp_path)
    upstream.commit(
        {
            "src/app.py": "def total(items):\n    return sum(items)\n\nTOKEN_NAME = 'x'\n",
            "README.md": "# App\nCall total() to add numbers.\n",
            "package-lock.json": '{"total": 1}\n',
            "fixtures/data.txt": "total in a fixture\n",
            "big.txt": "a" * (MAX_FILE_BYTES + 1),
            "logo.png": b"\x89PNG\r\n\x1a\n\0\0\0total",
        },
        "init",
    )
    os.symlink("/etc/passwd", upstream.work / "passwd-link")
    head = upstream.commit({"src/util.py": "def helper():\n    return total([1])\n"}, "add util")
    git_dir = MirrorStore(tmp_path / "repos", remote_base=str(upstream.base)).sync(
        "octo/app", ["refs/heads/main"], [head], token=None
    )
    return git_dir, head


@pytest.fixture
def tools(checkout: tuple[Path, str]) -> RepoTools:
    git_dir, head = checkout
    return RepoTools(git_dir, "octo/app", head, [*DEFAULT_SKIP_PATHS, "fixtures/**"])


def test_read_file_returns_numbered_lines_and_ranges(tools: RepoTools) -> None:
    assert tools.read_file("src/app.py").splitlines()[0] == "1: def total(items):"
    assert tools.read_file("src/app.py", start_line=2, end_line=2) == "2:     return sum(items)"
    assert tools.read_file("/src/app.py", end_line=1) == "1: def total(items):"


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("package-lock.json", "skip list"),
        ("fixtures/data.txt", "skip list"),
        ("big.txt", "larger than"),
        ("logo.png", "binary"),
        ("missing.py", "does not exist"),
        ("src", "directory"),
        ("../outside.txt", "inside the repository"),
        ("src/../../x", "inside the repository"),
        ("", "file path"),
    ],
)
def test_read_file_refuses_what_it_must_not_read(tools: RepoTools, path: str, reason: str) -> None:
    with pytest.raises(ToolError, match=reason):
        tools.read_file(path)


def test_a_symlink_is_read_as_its_target_text_not_followed(tools: RepoTools) -> None:
    assert tools.read_file("passwd-link") == "1: /etc/passwd"


def test_list_dir_shows_entries_and_marks_skipped_ones(tools: RepoTools) -> None:
    root = tools.list_dir()
    assert "dir " in root and " src" in root
    assert "package-lock.json (skipped)" in root
    assert tools.list_dir("src").splitlines()[0].endswith("src/app.py")
    with pytest.raises(ToolError, match="not a directory"):
        tools.list_dir("README.md")


def test_search_code_skips_skipped_and_binary_files(tools: RepoTools) -> None:
    result = tools.search_code("total")

    paths = {line.split(":", 1)[0] for line in result.splitlines()}
    assert paths == {"src/app.py", "src/util.py", "README.md"}


def test_search_code_filters_by_path_and_reports_no_matches(tools: RepoTools) -> None:
    assert {
        line.split(":", 1)[0] for line in tools.search_code("total", "src/**").splitlines()
    } == {
        "src/app.py",
        "src/util.py",
    }
    assert tools.search_code("no_such_symbol") == "no matches"
    assert tools.search_code("--output") == "no matches"


def test_git_log_shows_history_for_a_path(tools: RepoTools) -> None:
    assert "add util" in tools.git_log()
    assert "add util" not in tools.git_log("src/app.py")


@pytest.fixture
def issue_api() -> Iterator[tuple[str, list[dict[str, str]]]]:
    seen: list[dict[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen.append({"path": self.path, "auth": self.headers.get("Authorization", "")})
            if self.path.endswith("/issues/12"):
                body = {
                    "title": "Add totals",
                    "state": "open",
                    "labels": [{"name": "P0"}],
                    "body": "Totals must ignore negatives.",
                }
                payload = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            elif self.path.endswith(("/issues/20", "/issues/21")):
                # A proxy's HTML error page, and JSON that is not an issue.
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<html>bad gateway</html>" if "20" in self.path else b"{}")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", seen
    server.shutdown()


def test_read_issue_returns_the_issue_using_the_token(
    checkout: tuple[Path, str], issue_api: tuple[str, list[dict[str, str]]]
) -> None:
    base, seen = issue_api
    tools = RepoTools(*checkout[:1], "octo/app", checkout[1], [], token="fake-token", api_base=base)

    text = tools.read_issue(12)

    assert text.startswith("#12 Add totals [open] labels: P0")
    assert "ignore negatives" in text
    assert seen == [{"path": "/repos/octo/app/issues/12", "auth": "Bearer fake-token"}]
    with pytest.raises(ToolError, match="HTTP 404"):
        tools.read_issue(13)
    for number in (20, 21):
        with pytest.raises(ToolError, match="not an issue"):
            tools.issue_text(number)


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("**/vendor/**", "vendor/x.go", True),
        ("**/vendor/**", "a/vendor/b/x.go", True),
        ("**/vendor/**", "vendors/x.go", False),
        ("**/*.min.js", "app.min.js", True),
        ("**/*.min.js", "web/app.min.js", True),
        ("fixtures/**", "fixtures/a/b.txt", True),
        ("fixtures/**", "src/fixtures/b.txt", False),
        ("src/*.py", "src/a/b.py", False),
        ("**/*.generated.*", "api/types.generated.ts", True),
    ],
)
def test_glob_match(pattern: str, path: str, expected: bool) -> None:
    assert glob_match(pattern, path) is expected


def test_the_budget_stops_tool_calls_at_the_limit(tools: RepoTools) -> None:
    budget = ToolBudget(limit=25)
    calls = 0

    def counted() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    limited = budget.wrap(counted)
    results = [limited() for _ in range(30)]

    assert results[:25] == ["ok"] * 25
    assert results[25:] == [LIMIT_REACHED] * 5
    assert calls == 25


def test_refused_requests_count_and_come_back_as_error_text(tools: RepoTools) -> None:
    budget = ToolBudget(limit=2)
    read = agent_tools(tools, budget)["read_file"]

    assert read("package-lock.json").startswith("error: package-lock.json is in the skip list")
    assert budget.used == 1


def test_agents_only_get_read_only_tools(tools: RepoTools) -> None:
    given = agent_tools(tools, ToolBudget(limit=25))

    assert set(given) == {"read_file", "list_dir", "search_code", "git_log", "read_issue"}
    public = {name for name in dir(RepoTools) if not name.startswith("_")}
    # A new public method must be added here on purpose, after checking it cannot write.
    # issue_text is read_issue without the body cap, for the PR layer, not for agents.
    # line_count lets the pipeline check where a finding points; agents never get it.
    assert public == {*given, "NAMES", "is_skipped", "issue_text", "line_count", "text"}


def test_an_invalid_pattern_and_an_unreachable_api_come_back_as_errors(
    checkout: tuple[Path, str],
) -> None:
    git_dir, head = checkout
    tools = RepoTools(git_dir, "octo/app", head, [], api_base="http://127.0.0.1:9")

    with pytest.raises(ToolError, match="search failed"):
        tools.search_code("(unclosed")
    with pytest.raises(ToolError, match="unreachable"):
        tools.read_issue(1)


def test_wrapped_tools_keep_their_signature_and_docs(tools: RepoTools) -> None:
    # LangChain builds each tool's argument schema from the signature.
    read = agent_tools(tools, ToolBudget(limit=1))["read_file"]

    assert list(inspect.signature(read).parameters) == ["path", "start_line", "end_line"]
    assert read.__doc__ == RepoTools.read_file.__doc__


@pytest.mark.parametrize("commit", ["HEAD", "abc123", "--output=x", "main"])
def test_tools_only_accept_a_full_commit_sha(checkout: tuple[Path, str], commit: str) -> None:
    with pytest.raises(ValueError, match="full commit sha"):
        RepoTools(checkout[0], "octo/app", commit, [])


def test_list_dir_refuses_a_skipped_directory(tools: RepoTools) -> None:
    with pytest.raises(ToolError, match="skip list"):
        tools.list_dir("fixtures")


def test_a_hanging_git_command_is_killed_and_reported_to_the_agent(
    checkout: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_dir, head = checkout
    monkeypatch.setenv("PATH", os.environ["PATH"])  # restored after the test
    install_fake_git(tmp_path / "fakebin", "exec sleep 30")
    tools = RepoTools(git_dir, "octo/app", head, [], git_timeout=0.3)
    read = agent_tools(tools, ToolBudget(limit=25))["read_file"]

    started = time.monotonic()
    result = read("src/app.py")

    assert result.startswith("error: git cat-file timed out")
    assert time.monotonic() - started < 3


def test_line_count_reads_the_reviewed_commit(tools: RepoTools) -> None:
    assert tools.line_count("src/app.py") == 4
    assert tools.line_count("big.txt") == 1  # no trailing newline
    assert tools.line_count("src/missing.py") is None
    assert tools.line_count("../outside") is None


def test_line_count_reports_a_timeout_instead_of_a_missing_file(
    checkout: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_dir, head = checkout
    monkeypatch.setenv("PATH", os.environ["PATH"])  # restored after the test
    install_fake_git(tmp_path / "fakebin", "exec sleep 30")
    tools = RepoTools(git_dir, "octo/app", head, [], git_timeout=0.3)

    # A slow git says nothing about whether the file exists.
    with pytest.raises(ToolTimeoutError, match="timed out"):
        tools.line_count("src/app.py")


def test_text_reads_a_file_of_the_reviewed_commit(tools: RepoTools) -> None:
    # Not an agent tool: the pipeline uses it to check what a finding quotes.
    assert (
        tools.text("src/app.py") == "def total(items):\n    return sum(items)\n\nTOKEN_NAME = 'x'\n"
    )
    assert tools.text("src/missing.py") is None
    assert tools.text("../outside") is None


def test_text_reports_a_timeout_instead_of_a_missing_file(
    checkout: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_dir, head = checkout
    monkeypatch.setenv("PATH", os.environ["PATH"])  # restored after the test
    install_fake_git(tmp_path / "fakebin", "exec sleep 30")
    tools = RepoTools(git_dir, "octo/app", head, [], git_timeout=0.3)

    with pytest.raises(ToolTimeoutError, match="timed out"):
        tools.text("src/app.py")
