import subprocess
from pathlib import Path


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "-c", "commit.gpgsign=false", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class Upstream:
    """A bare repository standing in for GitHub, plus a work tree to commit from."""

    def __init__(self, root: Path, repo: str = "octo/app") -> None:
        self.base = root / "remote"
        self.bare = self.base / f"{repo}.git"
        self.bare.mkdir(parents=True)
        git(self.bare, "init", "--bare", "-q", "-b", "main")
        self.work = root / "work"
        self.work.mkdir()
        git(self.work, "init", "-q", "-b", "main")
        git(self.work, "config", "user.email", "t@example.com")
        git(self.work, "config", "user.name", "t")
        git(self.work, "remote", "add", "origin", str(self.bare))

    def commit(self, files: dict[str, str | bytes], message: str, branch: str = "main") -> str:
        for name, content in files.items():
            path = self.work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")
        git(self.work, "add", "-A")
        git(self.work, "commit", "-q", "-m", message)
        git(self.work, "push", "-q", "origin", f"HEAD:refs/heads/{branch}")
        return git(self.work, "rev-parse", "HEAD")


def install_fake_git(bin_dir: Path, body: str) -> None:
    """Put a `git` wrapper first on PATH; `$REAL_GIT` runs the real one."""
    import os
    import shutil

    real = shutil.which("git")
    assert real
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "git"
    script.write_text(f'#!/bin/sh\nREAL_GIT="{real}"\n{body}\n', encoding="utf-8")
    script.chmod(0o755)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
