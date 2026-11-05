"""One bare git mirror per reviewed repository, fetched incrementally."""

import base64
import fcntl
import math
import os
import random
import re
import signal
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

_REPO_NAME = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA = re.compile(r"^[0-9a-f]{40}$")

FETCH_TIMEOUT_SECONDS = 300.0
FETCH_ATTEMPTS = 3
# Errors a later attempt can get past. Authentication, permission and missing-ref
# errors are not here: retrying them only delays the failure.
_TRANSIENT_FETCH_ERRORS = re.compile(
    r"Could not resolve host|Connection reset|Connection refused|Connection timed out"
    r"|Failed to connect|Operation timed out|early EOF|unexpected disconnect|RPC failed"
    r"|returned error: 5\d\d",
)


class GitError(RuntimeError):
    pass


class MissingCommitError(GitError):
    pass


class GitTimeoutError(GitError):
    pass


def check_repo_name(repo: str) -> str:
    if not _REPO_NAME.fullmatch(repo) or ".." in repo.split("/"):
        raise ValueError(f"not an owner/name repository: {repo!r}")
    return repo


def check_commit_sha(sha: str) -> str:
    if not _SHA.fullmatch(sha):
        raise ValueError(f"not a full commit sha: {sha!r}")
    return sha


def git_env(token: str | None = None, remote_prefix: str = "") -> dict[str, str]:
    """Environment for every git call: no user or system config, no prompts.

    The token goes into an HTTP header through `GIT_CONFIG_*` variables, so it is
    never written to the mirror's config, a remote URL or the process arguments.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
    }
    if token:
        credentials = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env |= {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": f"http.{remote_prefix}.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {credentials}",
        }
    return env


def run_git(
    git_dir: Path,
    *args: str,
    env: dict[str, str] | None = None,
    secret: str | None = None,
    ok_codes: tuple[int, ...] = (0,),
    timeout: float | None = None,
) -> bytes:
    # Its own process group, so a timeout also kills what git started, such as the
    # HTTPS remote helper during a fetch.
    process = subprocess.Popen(  # noqa: S603 - fixed git binary, arguments are not shell-parsed
        ["git", f"--git-dir={git_dir}", *args],  # noqa: S607
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env or git_env(),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise GitTimeoutError(f"git {args[0]} timed out after {timeout:g}s") from None
    except BaseException:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise
    if process.returncode not in ok_codes:
        message = stderr.decode("utf-8", "replace").strip()
        if secret:
            message = message.replace(secret, "***")
        raise GitError(f"git {args[0]} failed: {message}")
    return stdout


class MirrorStore:
    """Bare mirrors under `root`, one per repository, reused across reviews."""

    def __init__(
        self,
        root: Path,
        remote_base: str = "https://github.com",
        fetch_timeout: float = FETCH_TIMEOUT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._root = root
        self._remote_base = remote_base.rstrip("/")
        self._fetch_timeout = fetch_timeout
        self._sleep = sleep

    def git_dir(self, repo: str) -> Path:
        owner, name = check_repo_name(repo).split("/")
        return self._root / owner / f"{name}.git"

    def remote_url(self, repo: str) -> str:
        return f"{self._remote_base}/{check_repo_name(repo)}.git"

    @contextmanager
    def _locked(self, repo: str) -> Iterator[Path]:
        # Two reviews of the same repository must not fetch into one mirror at once.
        git_dir = self.git_dir(repo)
        git_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(git_dir.parent / f"{git_dir.name}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield git_dir
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def sync(
        self,
        repo: str,
        refs: Sequence[str],
        commits: Sequence[str],
        token: str | None,
        *,
        deadline: float = math.inf,
        clock: Callable[[], float] = time.monotonic,
    ) -> Path:
        """Fetch `refs` into the mirror, creating it on first use; return its git dir.

        Later syncs only transfer objects the mirror does not have yet. Every commit in
        `commits` must be present afterwards. No fetch, nor the wait before a retry, runs
        past `deadline` (a `clock` time).
        """
        for ref in refs:
            if not ref.startswith("refs/") or any(c in ref for c in " :^~?*[\\"):
                raise ValueError(f"not a full ref name: {ref!r}")
        for sha in commits:
            check_commit_sha(sha)
        url = self.remote_url(repo)
        env = git_env(token, remote_prefix=self._remote_base)
        with self._locked(repo) as git_dir:
            if not (git_dir / "HEAD").exists():
                run_git(git_dir, "init", "--bare", "--quiet")
            refspecs = [f"+{ref}:{ref}" for ref in refs]
            self._fetch(git_dir, [url, *refspecs], env, token, deadline, clock)
            for sha in commits:
                if not self.has_commit(git_dir, sha):
                    raise MissingCommitError(f"{repo} has no commit {sha} after fetching")
        return git_dir

    def _fetch(
        self,
        git_dir: Path,
        args: list[str],
        env: dict[str, str],
        token: str | None,
        deadline: float,
        clock: Callable[[], float],
    ) -> None:
        """Fetch, retrying transient failures with exponential backoff and jitter."""
        for attempt in range(1, FETCH_ATTEMPTS + 1):
            left = deadline - clock()
            if left <= 0:
                raise GitTimeoutError("no time left in the review to fetch the repository")
            try:
                run_git(
                    git_dir,
                    "fetch",
                    "--quiet",
                    "--no-tags",
                    *args,
                    env=env,
                    secret=token,
                    timeout=min(self._fetch_timeout, left),
                )
                return
            except GitError as exc:
                transient = isinstance(exc, GitTimeoutError) or bool(
                    _TRANSIENT_FETCH_ERRORS.search(str(exc))
                )
                backoff = 2**attempt + random.random()  # noqa: S311 - jitter, not security
                # A retry whose wait would end past the deadline is not worth starting.
                if not transient or attempt == FETCH_ATTEMPTS or clock() + backoff >= deadline:
                    raise
            self._sleep(backoff)

    @staticmethod
    def has_commit(git_dir: Path, sha: str) -> bool:
        try:
            run_git(git_dir, "cat-file", "-e", f"{sha}^{{commit}}")
        except GitError:
            return False
        return True
