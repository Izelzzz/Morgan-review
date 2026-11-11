"""Read-only access to one commit of a mirror, for building context."""

from dataclasses import dataclass
from pathlib import Path

from review_agents.mirror import GitError, check_commit_sha, run_git
from review_agents.repo_config import MAX_FILE_BYTES
from review_agents.tools import glob_match

GIT_TIMEOUT_SECONDS = 30.0
_BINARY_PROBE_BYTES = 8_000


@dataclass(frozen=True)
class Entry:
    path: str
    blob: str
    size: int


class Snapshot:
    """The files of one commit, with the skip rules already applied."""

    def __init__(self, git_dir: Path, commit: str, skip_paths: tuple[str, ...]) -> None:
        self.git_dir = git_dir
        self.commit = check_commit_sha(commit)
        self._skip_paths = skip_paths
        self._files: list[Entry] | None = None
        self._by_path: dict[str, Entry] = {}

    @classmethod
    def at_branch(cls, git_dir: Path, branch: str, skip_paths: tuple[str, ...]) -> "Snapshot":
        sha = run_git(
            git_dir,
            "rev-parse",
            "--verify",
            f"refs/heads/{branch}^{{commit}}",
            timeout=GIT_TIMEOUT_SECONDS,
        )
        return cls(git_dir, sha.decode().strip(), skip_paths)

    def _git(self, *args: str) -> bytes:
        return run_git(self.git_dir, *args, timeout=GIT_TIMEOUT_SECONDS)

    def is_skipped(self, path: str) -> bool:
        return any(glob_match(pattern, path) for pattern in self._skip_paths)

    @property
    def tree_id(self) -> str:
        return self._git("rev-parse", f"{self.commit}^{{tree}}").decode().strip()

    def files(self) -> list[Entry]:
        """Every file that is not skipped, in path order."""
        if self._files is None:
            output = self._git("ls-tree", "-r", "-l", "-z", self.commit)
            files = []
            for record in filter(None, output.decode("utf-8", "replace").split("\0")):
                meta, path = record.split("\t", 1)
                _, kind, object_id, size = meta.split()
                if kind == "blob" and not self.is_skipped(path):
                    files.append(Entry(path, object_id, int(size)))
            self._files = files
            self._by_path = {f.path: f for f in files}
        return self._files

    def read_text(self, path: str) -> str | None:
        """A text file's content, or None when it is missing, skipped, too large or binary."""
        self.files()
        entry = self._by_path.get(path)
        if entry is None or entry.size > MAX_FILE_BYTES:
            return None
        try:
            data = self._git("cat-file", "blob", entry.blob)
        except GitError:
            return None
        if b"\0" in data[:_BINARY_PROBE_BYTES]:
            return None
        return data.decode("utf-8", "replace")
