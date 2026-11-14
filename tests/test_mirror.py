import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from gitrepo import Upstream, git, install_fake_git

from review_agents.mirror import (
    GitError,
    GitTimeoutError,
    MirrorStore,
    MissingCommitError,
    check_repo_name,
    git_env,
    run_git,
)

MAIN = ["refs/heads/main"]


@pytest.fixture
def upstream(tmp_path: Path) -> Upstream:
    return Upstream(tmp_path)


@pytest.fixture
def store(tmp_path: Path, upstream: Upstream) -> MirrorStore:
    return MirrorStore(tmp_path / "repos", remote_base=str(upstream.base))


def object_ids(git_dir: Path) -> set[str]:
    listing = git(git_dir, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)")
    return set(listing.split())


def test_first_sync_creates_a_bare_mirror_with_the_requested_commits(
    store: MirrorStore, upstream: Upstream
) -> None:
    head = upstream.commit({"a.py": "x = 1\n"}, "init")

    git_dir = store.sync("octo/app", MAIN, [head], token=None)

    assert git_dir == store.git_dir("octo/app")
    assert git(git_dir, "rev-parse", "--is-bare-repository") == "true"
    assert store.has_commit(git_dir, head)


def test_a_second_sync_only_fetches_the_new_objects(store: MirrorStore, upstream: Upstream) -> None:
    upstream.commit({f"f{i}.py": f"v = {i}\n" for i in range(20)}, "init")
    git_dir = store.sync("octo/app", MAIN, [], token=None)
    before = object_ids(git_dir)
    (git_dir / "marker").write_text("kept", encoding="utf-8")

    # One changed file: one commit, one root tree and one blob.
    head = upstream.commit({"f0.py": "v = 100\n"}, "change one file")
    store.sync("octo/app", MAIN, [head], token=None)

    after = object_ids(git_dir)
    assert (git_dir / "marker").read_text(encoding="utf-8") == "kept"
    assert before < after
    assert len(after - before) == 3
    assert store.has_commit(git_dir, head)


def test_a_pull_request_ref_is_fetched_next_to_the_default_branch(
    store: MirrorStore, upstream: Upstream
) -> None:
    upstream.commit({"a.py": "x = 1\n"}, "init")
    head = upstream.commit({"a.py": "x = 2\n"}, "pr change", branch="pull/7/head")

    git_dir = store.sync("octo/app", [*MAIN, "refs/heads/pull/7/head"], [head], token=None)

    assert store.has_commit(git_dir, head)


def test_a_commit_missing_after_the_fetch_is_an_error(
    store: MirrorStore, upstream: Upstream
) -> None:
    upstream.commit({"a.py": "x = 1\n"}, "init")

    with pytest.raises(MissingCommitError):
        store.sync("octo/app", MAIN, ["0" * 40], token=None)


@pytest.mark.parametrize("repo", ["octo", "octo/app/x", "../app", "octo/..", "octo/a b", "/etc"])
def test_repository_names_must_be_owner_slash_name(repo: str) -> None:
    with pytest.raises(ValueError, match="owner/name"):
        check_repo_name(repo)


@pytest.mark.parametrize("ref", ["main", "refs/heads/main:refs/x", "refs/heads/a b", "HEAD"])
def test_refs_must_be_full_ref_names(store: MirrorStore, ref: str) -> None:
    with pytest.raises(ValueError, match="full ref"):
        store.sync("octo/app", [ref], [], token=None)


def test_commits_must_be_full_shas(store: MirrorStore) -> None:
    with pytest.raises(ValueError, match="full commit sha"):
        store.sync("octo/app", MAIN, ["HEAD"], token=None)


def test_the_token_travels_only_in_the_environment_and_never_reaches_the_mirror(
    store: MirrorStore, upstream: Upstream
) -> None:
    upstream.commit({"a.py": "x = 1\n"}, "init")
    token = "fake-token-value-123"  # secret-scan: allow

    env = git_env(token, remote_prefix="https://github.com")
    git_dir = store.sync("octo/app", MAIN, [], token=token)

    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com.extraheader"
    assert token not in env["GIT_CONFIG_VALUE_0"]  # base64-encoded in the header
    for path in git_dir.rglob("*"):
        if path.is_file():
            assert token.encode() not in path.read_bytes()


NETWORK = {
    "HTTPS_PROXY": "http://proxy.internal:3128",
    "https_proxy": "http://proxy.internal:3128",
    "HTTP_PROXY": "http://proxy.internal:3128",
    "http_proxy": "http://proxy.internal:3128",
    "NO_PROXY": "localhost",
    "no_proxy": "localhost",
    "GIT_SSL_CAINFO": "/etc/ssl/corp.pem",
    "SSL_CERT_FILE": "/etc/ssl/corp.pem",
    "SSL_CERT_DIR": "/etc/ssl/certs",
}


def test_git_reaches_the_network_through_the_service_s_proxy_and_cas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A service behind a proxy could fetch nothing (issue #80).
    for name, value in NETWORK.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HOME", "/home/someone")  # anything else stays out

    env = git_env("fake-token-value-123")  # secret-scan: allow

    assert {name: env[name] for name in NETWORK} == NETWORK
    assert "HOME" not in env and env["GIT_CONFIG_GLOBAL"] == os.devnull


def test_without_a_proxy_git_gets_no_network_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in NETWORK:
        monkeypatch.delenv(name, raising=False)

    assert not set(NETWORK) & set(git_env())


def test_git_errors_never_show_the_token(store: MirrorStore, upstream: Upstream) -> None:
    upstream.commit({"a.py": "x = 1\n"}, "init")
    git_dir = store.sync("octo/app", MAIN, [], token=None)
    token = "fake-token-value-123"  # secret-scan: allow

    with pytest.raises(GitError) as raised:
        run_git(git_dir, "rev-parse", token, secret=token)

    assert token not in str(raised.value)
    assert "***" in str(raised.value)


def test_concurrent_syncs_of_one_repository_both_succeed(
    store: MirrorStore, upstream: Upstream
) -> None:
    head = upstream.commit({f"f{i}.py": "x\n" * 200 for i in range(50)}, "init")
    errors: list[BaseException] = []

    def sync() -> None:
        try:
            store.sync("octo/app", MAIN, [head], token=None)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=sync) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []


@pytest.fixture
def fake_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Path]:
    monkeypatch.setenv("PATH", os.environ["PATH"])  # restored after the test
    counter = tmp_path / "fetch-count"

    def install(on_fetch: str) -> Path:
        install_fake_git(
            tmp_path / "fakebin",
            'case " $* " in *" fetch "*)\n'
            f'  n=$(cat "{counter}" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "{counter}"\n'
            f"  {on_fetch}\n"
            ";; esac\n"
            'exec "$REAL_GIT" "$@"',
        )
        return counter

    return install


def store_with_fake_sleep(
    tmp_path: Path, upstream: Upstream, **kwargs: float
) -> tuple[MirrorStore, list[float]]:
    slept: list[float] = []
    store = MirrorStore(
        tmp_path / "repos", remote_base=str(upstream.base), sleep=slept.append, **kwargs
    )
    return store, slept


def test_a_transient_fetch_failure_is_retried_with_backoff(
    tmp_path: Path, upstream: Upstream, fake_git: Callable[[str], Path]
) -> None:
    head = upstream.commit({"a.py": "x = 1\n"}, "init")
    counter = fake_git(
        'if [ "$n" -le 2 ]; then echo "fatal: unable to access: Could not resolve host:'
        ' github.com" >&2; exit 128; fi'
    )
    store, slept = store_with_fake_sleep(tmp_path, upstream)

    git_dir = store.sync("octo/app", MAIN, [head], token=None)

    assert store.has_commit(git_dir, head)
    assert counter.read_text().strip() == "3"
    assert len(slept) == 2
    assert 2 <= slept[0] < 3
    assert 4 <= slept[1] < 5


def test_a_permission_error_is_not_retried(
    tmp_path: Path, upstream: Upstream, fake_git: Callable[[str], Path]
) -> None:
    upstream.commit({"a.py": "x = 1\n"}, "init")
    counter = fake_git(
        'echo "fatal: unable to access: The requested URL returned error: 403" >&2; exit 128'
    )
    store, slept = store_with_fake_sleep(tmp_path, upstream)

    with pytest.raises(GitError, match="403"):
        store.sync("octo/app", MAIN, [], token=None)

    assert counter.read_text().strip() == "1"
    assert slept == []


def test_a_hanging_fetch_is_killed_and_retried_then_fails(
    tmp_path: Path, upstream: Upstream, fake_git: Callable[[str], Path]
) -> None:
    upstream.commit({"a.py": "x = 1\n"}, "init")
    counter = fake_git("exec sleep 30")
    store, slept = store_with_fake_sleep(tmp_path, upstream, fetch_timeout=0.3)

    started = time.monotonic()
    with pytest.raises(GitTimeoutError):
        store.sync("octo/app", MAIN, [], token=None)

    assert time.monotonic() - started < 5
    assert counter.read_text().strip() == "3"
    assert len(slept) == 2


def test_a_timeout_also_kills_the_processes_git_started(
    tmp_path: Path, upstream: Upstream, fake_git: Callable[[str], Path]
) -> None:
    # git fetch runs a remote helper as a child process; it must not outlive the timeout.
    upstream.commit({"a.py": "x = 1\n"}, "init")
    pid_file = tmp_path / "helper.pid"
    fake_git(f'sleep 30 & echo $! > "{pid_file}"; wait')
    store, _ = store_with_fake_sleep(tmp_path, upstream, fetch_timeout=0.3)

    with pytest.raises(GitTimeoutError):
        store.sync("octo/app", MAIN, [], token=None)

    helper = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(helper, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(helper, 9)
        pytest.fail("the helper process outlived the timeout")


def test_a_fetch_never_outlasts_the_review(
    tmp_path: Path, upstream: Upstream, fake_git: Callable[[str], Path]
) -> None:
    # The fetch alone may wait five minutes, but the review has 0.3 seconds left.
    upstream.commit({"a.py": "x = 1\n"}, "init")
    counter = fake_git("exec sleep 30")
    store, slept = store_with_fake_sleep(tmp_path, upstream)

    started = time.monotonic()
    with pytest.raises(GitTimeoutError):
        store.sync("octo/app", MAIN, [], token=None, deadline=started + 0.3)

    assert time.monotonic() - started < 5
    assert counter.read_text().strip() == "1"  # no time left to try again
    assert slept == []


def test_no_fetch_starts_once_the_review_has_no_time_left(
    tmp_path: Path, upstream: Upstream, fake_git: Callable[[str], Path]
) -> None:
    upstream.commit({"a.py": "x = 1\n"}, "init")
    counter = fake_git("true")
    store, _ = store_with_fake_sleep(tmp_path, upstream)

    with pytest.raises(GitTimeoutError, match="no time left"):
        store.sync("octo/app", MAIN, [], token=None, deadline=time.monotonic() - 1)

    assert not counter.exists()
