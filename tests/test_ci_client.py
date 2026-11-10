import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from review_agents.auth import verify


def _load() -> ModuleType:
    path = Path(__file__).parent.parent / "ci" / "review_agents_ci.py"
    spec = importlib.util.spec_from_file_location("review_agents_ci", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ci = _load()
HEAD = "b" * 40
BASE = "a" * 40
FP_OLD = "1" * 64
FP_NEW = "2" * 64
SECRET = "s" * 40
GITHUB = "https://api.github.test"
SERVICE = "https://review.example.test"


class Fake:
    """Answers HTTP calls by (method, path), and records them."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Callable[[dict[str, str], Any], tuple[int, Any]]] = {}
        self.calls: list[tuple[str, str, dict[str, str], Any]] = []

    def on(self, method: str, path: str, *answers: Any) -> None:
        queue = list(answers)

        def respond(headers: dict[str, str], body: Any) -> tuple[int, Any]:
            reply = queue.pop(0) if len(queue) > 1 else queue[0]
            answer: tuple[int, Any] = reply(headers, body) if callable(reply) else reply
            return answer

        self.routes[(method, path)] = respond

    def call(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None = None
    ) -> tuple[int, Any]:
        path = url.split("://", 1)[1].split("/", 1)[1]
        path = "/" + path.split("?")[0]
        data = json.loads(body) if body else None
        self.calls.append((method, path, headers, data))
        return self.routes[(method, path)](headers, data)

    def made(self, method: str, path: str) -> list[Any]:
        return [body for m, p, _, body in self.calls if (m, p) == (method, path)]


@pytest.fixture
def fake() -> Fake:
    return Fake()


def github(fake: Fake) -> Any:
    return ci.GitHub(GITHUB, "octo/shop", "ghs_readonly", http=fake)


def service(fake: Fake) -> Any:
    return ci.Service(SERVICE, SECRET, http=fake, clock=lambda: 1_700_000_000)


def pr(head: str = HEAD) -> dict[str, Any]:
    return {
        "number": 7,
        "title": "Clamp refunds",
        "body": "Never refund more than paid.",
        "head": {"sha": head},
        "base": {"sha": BASE},
    }


def pr_event(action: str = "opened") -> dict[str, Any]:
    return {"action": action, "pull_request": pr(), "repository": {"default_branch": "main"}}


PULL = "/repos/octo/shop/pulls/7"


def run_review(fake: Fake, tmp_path: Path, name: str, event: dict[str, Any], **kw: Any) -> Any:
    out = tmp_path / "result.json"
    code = ci.review(name, event, github(fake), service(fake), out, sleep=lambda s: None, **kw)
    assert code == 0
    return json.loads(out.read_text())


def test_a_pr_is_submitted_with_the_read_only_token_and_waited_for(
    fake: Fake, tmp_path: Path
) -> None:
    fake.on("GET", PULL, (200, pr()))
    fake.on("POST", "/v1/reviews", (202, {"id": 42, "status": "queued"}))
    fake.on(
        "GET",
        "/v1/reviews/42",
        (200, {"id": 42, "status": "running"}),
        (200, {"id": 42, "status": "done", "report": {}}),
    )

    result = run_review(fake, tmp_path, "pull_request_target", pr_event())

    [sent] = fake.made("POST", "/v1/reviews")
    assert sent["token"] == "ghs_readonly" and sent["event"] == "opened"
    assert (sent["head_sha"], sent["base_sha"], sent["default_branch"]) == (HEAD, BASE, "main")
    assert result["status"] == "done" and result["review"]["id"] == 42
    assert (result["pr"], result["head_sha"]) == (7, HEAD)


def test_every_request_to_the_service_is_signed(fake: Fake, tmp_path: Path) -> None:
    fake.on("GET", PULL, (200, pr()))
    fake.on("POST", "/v1/reviews", (202, {"id": 42, "status": "queued"}))
    fake.on("GET", "/v1/reviews/42", (200, {"id": 42, "status": "done", "report": {}}))

    run_review(fake, tmp_path, "pull_request_target", pr_event())

    for method, path, headers, body in fake.calls:
        if not path.startswith("/v1/"):
            continue
        raw = json.dumps(body).encode() if body is not None else b""
        assert verify(
            SECRET,
            headers["X-Review-Agents-Timestamp"],
            headers["X-Review-Agents-Signature"],
            method,
            path,
            raw,
            clock=lambda: 1_700_000_000,
        )


def test_a_run_for_an_older_head_is_not_submitted(fake: Fake, tmp_path: Path) -> None:
    fake.on("GET", PULL, (200, pr(head="c" * 40)))

    result = run_review(fake, tmp_path, "pull_request_target", pr_event("synchronize"))

    assert result["status"] == "skipped" and "newer push" in result["reason"]
    assert fake.made("POST", "/v1/reviews") == []


def comment_event(body: str, association: str, on_pr: bool = True) -> dict[str, Any]:
    issue: dict[str, Any] = {"number": 7}
    if on_pr:
        issue["pull_request"] = {"url": "x"}
    return {
        "comment": {"body": body, "author_association": association},
        "issue": issue,
        "repository": {"default_branch": "main"},
    }


def test_a_collaborator_s_review_command_starts_a_manual_review(fake: Fake, tmp_path: Path) -> None:
    fake.on("GET", PULL, (200, pr()))
    fake.on("POST", "/v1/reviews", (202, {"id": 43, "status": "queued"}))
    fake.on("GET", "/v1/reviews/43", (200, {"id": 43, "status": "done", "report": {}}))

    run_review(fake, tmp_path, "issue_comment", comment_event("/review please", "MEMBER"))

    [sent] = fake.made("POST", "/v1/reviews")
    assert sent["event"] == "manual" and sent["head_sha"] == HEAD


@pytest.mark.parametrize(
    ("body", "association", "on_pr"),
    [
        ("/review", "NONE", True),  # anyone could spend the budget
        ("/review", "CONTRIBUTOR", True),
        ("looks good", "OWNER", True),
        ("/review", "OWNER", False),  # an issue, not a PR
    ],
)
def test_other_comments_start_nothing(
    fake: Fake, tmp_path: Path, body: str, association: str, on_pr: bool
) -> None:
    result = run_review(fake, tmp_path, "issue_comment", comment_event(body, association, on_pr))

    assert result["status"] == "skipped" and fake.calls == []


def test_a_refusal_is_recorded_for_the_post_job(fake: Fake, tmp_path: Path) -> None:
    fake.on("GET", PULL, (200, pr()))
    fake.on("POST", "/v1/reviews", (401, {"detail": "invalid or missing signature"}))

    result = run_review(fake, tmp_path, "pull_request_target", pr_event())

    assert result["status"] == "error" and "401" in result["reason"]
    assert "ghs_readonly" not in json.dumps(result)


def test_a_review_with_no_result_in_45_minutes_times_out(fake: Fake, tmp_path: Path) -> None:
    fake.on("GET", PULL, (200, pr()))
    fake.on("POST", "/v1/reviews", (202, {"id": 42, "status": "queued"}))
    fake.on("GET", "/v1/reviews/42", (200, {"id": 42, "status": "running"}))
    now = iter(range(0, 10_000, 15 * 60))

    result = run_review(
        fake, tmp_path, "pull_request_target", pr_event(), clock=lambda: float(next(now))
    )

    assert result["status"] == "timeout" and "45 minutes" in result["reason"]


# --- post -------------------------------------------------------------------

ISSUE_COMMENTS = "/repos/octo/shop/issues/7/comments"
REVIEW_COMMENTS = "/repos/octo/shop/pulls/7/comments"
REVIEWS = "/repos/octo/shop/pulls/7/reviews"
BOT = {"login": "github-actions[bot]"}
SUMMARY = '<!-- review-agents:summary -->\n## ok\n\n<!-- review-agents:counts {"blocking": 0} -->\n'


def inline(fp: str, line: int = 2) -> dict[str, Any]:
    return {
        "path": "payments/refund.py",
        "start_line": None,
        "line": line,
        "body": f"**🟡 x**\n\n<!-- review-agents:finding {fp} -->\n",
    }


def done(counts: dict[str, Any] | None = None, comments: list[Any] | None = None) -> dict[str, Any]:
    report = {
        "summary": SUMMARY,
        "comments": comments if comments is not None else [],
        "counts": counts if counts is not None else {"blocking": 0, "complete": True},
        "walkthrough": "<!-- review-agents:walkthrough -->\n## 📖\n",
    }
    return {"status": "done", "pr": 7, "head_sha": HEAD, "review": {"id": 42, "report": report}}


def run_post(fake: Fake, result: dict[str, Any]) -> tuple[int, list[str]]:
    said: list[str] = []
    code = ci.post(result, github(fake), write_summary=said.append)
    return code, said


def test_a_finished_review_posts_both_comments_and_new_inline_ones(fake: Fake) -> None:
    fake.on("GET", ISSUE_COMMENTS, (200, []))
    fake.on("POST", ISSUE_COMMENTS, (201, {"id": 1}))
    fake.on(
        "GET",
        REVIEW_COMMENTS,
        (
            200,
            [
                {"user": BOT, "body": f"<!-- review-agents:finding {FP_OLD} -->"},
                # Someone else writing a marker does not count as ours.
                {"user": {"login": "mallory"}, "body": f"<!-- review-agents:finding {FP_NEW} -->"},
            ],
        ),
    )
    fake.on("POST", REVIEWS, (200, {"id": 9}))

    code, said = run_post(fake, done(comments=[inline(FP_OLD), inline(FP_NEW, 3)]))

    assert code == 0 and said == ["No 🔴, and the review finished."]
    walkthrough, summary = (b["body"] for b in fake.made("POST", ISSUE_COMMENTS))
    assert walkthrough.startswith("<!-- review-agents:walkthrough -->")  # first, before results
    assert summary == SUMMARY
    [batch] = fake.made("POST", REVIEWS)
    assert batch["commit_id"] == HEAD and batch["event"] == "COMMENT"
    assert [c["line"] for c in batch["comments"]] == [3]  # the old one is there already
    assert batch["comments"][0]["side"] == "RIGHT"


def test_the_comments_are_updated_in_place(fake: Fake) -> None:
    fake.on(
        "GET",
        ISSUE_COMMENTS,
        (
            200,
            [
                {"id": 5, "user": BOT, "body": "<!-- review-agents:summary -->\nold"},
                {
                    "id": 6,
                    "user": {"login": "mallory"},
                    "body": "<!-- review-agents:walkthrough -->",
                },
            ],
        ),
    )
    fake.on("POST", ISSUE_COMMENTS, (201, {"id": 7}))
    fake.on("PATCH", "/repos/octo/shop/issues/comments/5", (200, {}))
    fake.on("GET", REVIEW_COMMENTS, (200, []))

    run_post(fake, done())

    assert fake.made("PATCH", "/repos/octo/shop/issues/comments/5") == [{"body": SUMMARY}]
    # mallory's comment is not edited: a new walkthrough is posted.
    assert [b["body"][:34] for b in fake.made("POST", ISSUE_COMMENTS)] == [
        "<!-- review-agents:walkthrough -->"
    ]


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({"blocking": 2, "complete": True}, "2 🔴 to address first."),
        ({"blocking": 0, "complete": False}, "did not finish"),
    ],
)
def test_a_red_or_an_unfinished_review_fails_the_job(
    fake: Fake, counts: dict[str, Any], message: str
) -> None:
    fake.on("GET", ISSUE_COMMENTS, (200, []))
    fake.on("POST", ISSUE_COMMENTS, (201, {"id": 1}))
    fake.on("GET", REVIEW_COMMENTS, (200, []))

    code, said = run_post(fake, done(counts=counts))

    assert code == 1 and message in said[0]


def test_inline_comments_github_rejects_go_into_the_summary(fake: Fake) -> None:
    fake.on("GET", ISSUE_COMMENTS, (200, []), (200, [{"id": 8, "user": BOT, "body": SUMMARY}]))
    fake.on("POST", ISSUE_COMMENTS, (201, {"id": 1}))
    fake.on("PATCH", "/repos/octo/shop/issues/comments/8", (200, {}))
    fake.on("GET", REVIEW_COMMENTS, (200, []))
    fake.on("POST", REVIEWS, (422, {"message": "Unprocessable"}))
    fake.on("POST", REVIEW_COMMENTS, (201, {}), (422, {"message": "line must be part of the diff"}))

    code, _ = run_post(fake, done(comments=[inline(FP_OLD), inline(FP_NEW, 30)]))

    assert code == 0
    assert len(fake.made("POST", REVIEW_COMMENTS)) == 2  # each tried alone
    [patched] = fake.made("PATCH", "/repos/octo/shop/issues/comments/8")
    body = patched["body"]
    assert "payments/refund.py:30-30" in body
    assert body.index("payments/refund.py:30-30") < body.index("<!-- review-agents:counts")
    assert f"finding {FP_NEW}" not in body


def test_a_failed_review_replaces_the_last_summary_and_fails(fake: Fake) -> None:
    fake.on("GET", ISSUE_COMMENTS, (200, [{"id": 5, "user": BOT, "body": SUMMARY}]))
    fake.on("PATCH", "/repos/octo/shop/issues/comments/5", (200, {}))
    failed = {
        "status": "failed",
        "pr": 7,
        "head_sha": HEAD,
        "review": {"id": 42, "status": "failed", "error": "model service unavailable"},
    }

    code, said = run_post(fake, failed)

    assert code == 1 and "model service unavailable" in said[0]
    [patched] = fake.made("PATCH", "/repos/octo/shop/issues/comments/5")
    assert "审查失败" in patched["body"] and "review-agents:counts" not in patched["body"]


@pytest.mark.parametrize(
    ("result", "code"),
    [
        ({"status": "skipped", "reason": "a newer push"}, 0),
        ({"status": "superseded", "pr": 7, "head_sha": HEAD, "review": {"id": 42}}, 1),
    ],
)
def test_a_skipped_or_superseded_run_touches_no_comment(
    fake: Fake, result: dict[str, Any], code: int
) -> None:
    assert run_post(fake, result)[0] == code
    assert fake.calls == []


def test_only_https_is_opened() -> None:
    with pytest.raises(ci.Failure, match="only https"):
        ci.Http().call("GET", "file:///etc/passwd", {})
