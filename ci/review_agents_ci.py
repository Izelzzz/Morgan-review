"""The CI side of review-agents: submit a review, then post its comments (issue #14).

Two commands, one per job of the workflow (CI design, D-037):

- `review` runs with a read-only GITHUB_TOKEN. It checks that the PR's head is still the
  commit this run is for (D-021), submits the review with that token, waits for the
  report and writes what came back to a file, the artifact the next job reads.
- `post` runs with a token that can write comments, which never leaves GitHub. It updates
  the walkthrough and the summary in place, posts the inline comments not posted yet
  (D-033), and exits red unless the review finished with no 🔴.

Standard library only: GitHub's runners have Python, and nothing is installed. The PR's
code is never checked out or run here; only the service reads it.
"""

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WALKTHROUGH_MARKER = "<!-- review-agents:walkthrough -->"
SUMMARY_MARKER = "<!-- review-agents:summary -->"
_FINGERPRINT = re.compile(r"<!-- review-agents:finding ([0-9a-f]{64}) -->")
# The comments GITHUB_TOKEN posts are this bot's; only those are taken for ours.
BOT = "github-actions[bot]"
# Who may ask for a review with `/review`: a comment costs the model budget.
COLLABORATORS = {"OWNER", "MEMBER", "COLLABORATOR"}
WAIT_SECONDS = 45 * 60  # a review lasts at most 30 minutes; the rest is queue and report
POLL_SECONDS = 15
EVENTS = {"opened": "opened", "synchronize": "synchronize", "reopened": "reopened"}


class Failure(Exception):
    """A request that did not work; the message says what, never a token or a secret."""


class SameHostRedirect(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to the same host over HTTPS.

    A request carries a token or a signature; a URL secret pointing at an address that
    redirects elsewhere must not hand them on.
    """

    def redirect_request(
        self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> urllib.request.Request | None:
        old, new = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(newurl)
        if new.scheme != "https" or new.netloc != old.netloc:
            raise Failure(
                f"{req.get_method()} {old.netloc}{old.path}: a redirect to "
                f"{new.scheme}://{new.netloc} is not followed"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass
class Http:
    """JSON over HTTPS, with the headers each side needs."""

    opener: Callable[..., Any] = field(
        default_factory=lambda: urllib.request.build_opener(SameHostRedirect()).open
    )

    def call(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None = None
    ) -> tuple[int, Any]:
        # HTTPS only: the request carries a token or a signature, and no other scheme is
        # opened (a file: URL from a mistyped secret, say).
        if not url.startswith("https://"):
            raise Failure(f"{method} {url.split('?')[0]}: only https:// is allowed")
        request = urllib.request.Request(url, data=body, method=method, headers=headers)  # noqa: S310
        try:
            with self.opener(request, timeout=60) as response:
                raw = response.read()
                return response.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                return exc.code, json.loads(raw) if raw else None
            except ValueError:
                return exc.code, None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise Failure(f"{method} {url.split('?')[0]}: {exc}") from None


@dataclass
class GitHub:
    api: str
    repo: str
    token: str
    http: Http = field(default_factory=Http)

    def request(self, method: str, path: str, body: object | None = None) -> tuple[int, Any]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "review-agents-ci",
        }
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        return self.http.call(method, f"{self.api}/repos/{self.repo}{path}", headers, data)

    def ok(self, method: str, path: str, body: object | None = None) -> Any:
        code, data = self.request(method, path, body)
        if code >= 300:
            message = data.get("message") if isinstance(data, dict) else ""
            raise Failure(f"GitHub {method} {path}: {code} {message}".rstrip())
        return data

    def pages(self, path: str) -> list[Any]:
        found: list[Any] = []
        page = 1
        while True:
            batch = self.ok("GET", f"{path}?per_page=100&page={page}")
            found += batch
            if len(batch) < 100:
                return found
            page += 1


@dataclass
class Service:
    url: str
    secret: str
    http: Http = field(default_factory=Http)
    clock: Callable[[], float] = time.time

    def request(self, method: str, path: str, body: bytes = b"") -> tuple[int, Any]:
        timestamp = str(int(self.clock()))
        message = f"{timestamp}\n{method}\n{path}\n".encode() + body
        signature = "sha256=" + hmac.new(self.secret.encode(), message, hashlib.sha256).hexdigest()
        headers = {
            "X-Review-Agents-Timestamp": timestamp,
            "X-Review-Agents-Signature": signature,
            "Content-Type": "application/json",
            "User-Agent": "review-agents-ci",
        }
        return self.http.call(method, self.url.rstrip("/") + path, headers, body or None)


def _write(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result), encoding="utf-8")


def review(
    event_name: str,
    event: dict[str, Any],
    github: GitHub,
    service: Service,
    result_path: Path,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Submit the PR and wait for its report. The job itself fails only on a broken run."""
    if event_name == "issue_comment":
        comment, issue = event["comment"], event["issue"]
        asked = comment.get("body", "").lstrip().startswith("/review")
        if not issue.get("pull_request") or not asked:
            _write(result_path, {"status": "skipped", "reason": "not a /review comment on a PR"})
            return 0
        if comment.get("author_association") not in COLLABORATORS:
            reason = "only the repository's collaborators can ask for a review"
            _write(result_path, {"status": "skipped", "reason": reason})
            return 0
        pr = github.ok("GET", f"/pulls/{issue['number']}")
        kind = "manual"
    else:
        pr = event["pull_request"]
        kind = EVENTS.get(event.get("action", ""), "")
        if not kind:
            _write(result_path, {"status": "skipped", "reason": f"action {event.get('action')}"})
            return 0
        # D-021: the service takes what it is sent for the PR's head. A re-run of an older
        # commit's job would supersede the newest push's review, so it is not sent.
        current = github.ok("GET", f"/pulls/{pr['number']}")
        if current["head"]["sha"] != pr["head"]["sha"]:
            reason = f"the PR's head is now {current['head']['sha'][:12]}, a newer push"
            _write(result_path, {"status": "skipped", "reason": reason})
            return 0
        pr = current
    request = {
        "repo": github.repo,
        "pr": pr["number"],
        "base_sha": pr["base"]["sha"],
        "head_sha": pr["head"]["sha"],
        "default_branch": event["repository"]["default_branch"],
        "title": pr.get("title") or "",
        "body": pr.get("body"),
        "event": kind,
        "token": github.token,
    }
    base = {"pr": pr["number"], "head_sha": pr["head"]["sha"]}
    try:
        code, created = service.request("POST", "/v1/reviews", json.dumps(request).encode())
        if code not in (200, 202) or not isinstance(created, dict):
            raise Failure(f"the service refused the review: {code} {_detail(created)}")
        review_id = created["id"]
        start = clock()
        while True:
            code, polled = service.request("GET", f"/v1/reviews/{review_id}")
            if code != 200 or not isinstance(polled, dict):
                raise Failure(f"polling review {review_id}: {code} {_detail(polled)}")
            if polled["status"] in ("done", "failed", "superseded"):
                _write(result_path, base | {"status": polled["status"], "review": polled})
                return 0
            if clock() - start >= WAIT_SECONDS:
                reason = f"review {review_id} had no result after {WAIT_SECONDS // 60} minutes"
                _write(result_path, base | {"status": "timeout", "reason": reason})
                return 0
            sleep(POLL_SECONDS)
    except Failure as exc:  # recorded for the post job, which says so and goes red
        _write(result_path, base | {"status": "error", "reason": str(exc)})
        return 0


def _detail(data: object) -> str:
    if isinstance(data, dict) and "detail" in data:
        return json.dumps(data["detail"])[:300]
    return ""


@dataclass
class Outcome:
    passed: bool
    message: str


def gate(result: dict[str, Any]) -> Outcome:
    """Red or green, by D-037: green only for a finished review with no 🔴."""
    status = result.get("status")
    if status == "skipped":
        return Outcome(True, f"Not reviewed: {result.get('reason')}.")
    review_ = result.get("review") or {}
    if status != "done":
        reason = result.get("reason") or review_.get("error") or status
        return Outcome(False, f"No result: {reason}.")
    counts = (review_.get("report") or {}).get("counts")
    if not isinstance(counts, dict):
        return Outcome(False, "No result: the report has no counts.")
    if counts.get("blocking"):
        return Outcome(False, f"{counts['blocking']} 🔴 to address first.")
    if counts.get("complete") is not True:
        return Outcome(False, "The review did not finish: see what did not finish in its summary.")
    return Outcome(True, "No 🔴, and the review finished.")


def _ours(comments: list[dict[str, Any]], marker: str) -> dict[str, Any] | None:
    return next(
        (
            c
            for c in comments
            if (c.get("user") or {}).get("login") == BOT and marker in (c.get("body") or "")
        ),
        None,
    )


def _upsert(
    github: GitHub, number: int, comments: list[dict[str, Any]], marker: str, body: str
) -> None:
    existing = _ours(comments, marker)
    if existing is None:
        github.ok("POST", f"/issues/{number}/comments", {"body": body})
    elif existing.get("body") != body:
        github.ok("PATCH", f"/issues/comments/{existing['id']}", {"body": body})


def _inline(comment: dict[str, Any], head: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": comment["path"],
        "line": comment["line"],
        "side": "RIGHT",
        "body": comment["body"],
    }
    if comment.get("start_line"):
        out |= {"start_line": comment["start_line"], "start_side": "RIGHT"}
    return out


def _rejected_section(rejected: list[dict[str, Any]]) -> str:
    lines = [
        "",
        "### GitHub 未接受的行间评论 / Inline comments GitHub did not accept",
        "",
    ]
    for comment in rejected:
        where = (
            f"{comment['path']}:{comment.get('start_line') or comment['line']}-{comment['line']}"
        )
        body = _FINGERPRINT.sub("", comment["body"]).strip()
        lines += [f"<details><summary><code>{where}</code></summary>", "", body, "", "</details>"]
    return "\n".join(lines)


def post(result: dict[str, Any], github: GitHub, *, write_summary: Callable[[str], None]) -> int:
    """Post the review's comments and set the job's colour. 0 is green, 1 is red."""
    outcome = gate(result)
    status = result.get("status")
    number = result.get("pr")
    if status in ("skipped", "superseded") or number is None:
        write_summary(outcome.message)  # a newer run updates the comments
        return 0 if outcome.passed else 1
    # D-021: an older run's post can run after a newer one's. Its colour stays on its own
    # commit, but the comments belong to the PR's head now.
    # A run that broke before it knew its commit has no head_sha: it leaves them alone too.
    current = (github.ok("GET", f"/pulls/{number}").get("head") or {}).get("sha")
    if current != result.get("head_sha"):
        why = f"the PR's head is now {current}" if result.get("head_sha") else "no commit known"
        write_summary(f"{outcome.message} Comments not updated: {why}.")
        return 0 if outcome.passed else 1
    issue_comments = github.pages(f"/issues/{number}/comments")
    report = (result.get("review") or {}).get("report")
    if status != "done" or not isinstance(report, dict):
        # The last round's conclusion must not stay up as if it were this one's.
        review_id = (result.get("review") or {}).get("id")
        line = f"## 审查失败 / Review failed\n\n{outcome.message.removeprefix('No result: ')}"
        if review_id is not None:
            line += f"\n\n审查 / review #{review_id}"
        _upsert(github, number, issue_comments, SUMMARY_MARKER, f"{SUMMARY_MARKER}\n{line}\n")
        write_summary(outcome.message)
        return 1
    if report.get("walkthrough"):
        _upsert(github, number, issue_comments, WALKTHROUGH_MARKER, report["walkthrough"])
    summary = report["summary"]
    _upsert(github, number, issue_comments, SUMMARY_MARKER, summary)

    posted = {
        m.group(1)
        for c in github.pages(f"/pulls/{number}/comments")
        if (c.get("user") or {}).get("login") == BOT
        for m in _FINGERPRINT.finditer(c.get("body") or "")
    }
    head = result["head_sha"]
    new = [
        c
        for c in report.get("comments", [])
        if not ((m := _FINGERPRINT.search(c["body"])) and m.group(1) in posted)
    ]
    rejected: list[dict[str, Any]] = []
    if new:
        batch = {"commit_id": head, "event": "COMMENT", "comments": [_inline(c, head) for c in new]}
        code, _ = github.request("POST", f"/pulls/{number}/reviews", batch)
        if code >= 300:  # one line GitHub will not take sinks the batch: try each alone
            for comment in new:
                code, _ = github.request(
                    "POST",
                    f"/pulls/{number}/comments",
                    _inline(comment, head) | {"commit_id": head},
                )
                if code >= 300:
                    rejected.append(comment)
    if rejected:
        # D-033 ③: none is lost; they go into the summary, before its hidden counts line.
        head_part, sep, tail = summary.rpartition("\n<!-- review-agents:counts")
        updated = (head_part + _rejected_section(rejected) + "\n" + sep + tail) if sep else summary
        refreshed = github.pages(f"/issues/{number}/comments")
        _upsert(github, number, refreshed, SUMMARY_MARKER, updated)
    write_summary(outcome.message)
    return 0 if outcome.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["review", "post"])
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    env = os.environ
    github = GitHub(
        env.get("GITHUB_API_URL", "https://api.github.com"),
        env["GITHUB_REPOSITORY"],
        env["GITHUB_TOKEN"],
    )

    def write_summary(text: str) -> None:
        print(text)
        if env.get("GITHUB_STEP_SUMMARY"):
            with open(env["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as out:
                out.write(f"review-agents: {text}\n")

    if args.command == "review":
        event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
        # What the post job needs to say 审查失败 if this run breaks before a result.
        pr = event.get("pull_request") or {}
        base = {
            "pr": pr.get("number") or (event.get("issue") or {}).get("number"),
            "head_sha": (pr.get("head") or {}).get("sha"),
        }
        url, secret = env.get("REVIEW_AGENTS_URL", ""), env.get("REVIEW_AGENTS_SECRET", "")
        if not url or not secret:
            reason = "REVIEW_AGENTS_URL and REVIEW_AGENTS_SECRET must be set as secrets"
            print(reason, file=sys.stderr)
            _write(args.result, base | {"status": "error", "reason": reason})
            return 2
        try:
            return review(
                env["GITHUB_EVENT_NAME"], event, github, Service(url, secret), args.result
            )
        except Exception as exc:
            reason = f"the review step broke: {type(exc).__name__}"
            _write(args.result, base | {"status": "error", "reason": reason})
            raise
    result = json.loads(args.result.read_text(encoding="utf-8"))
    return post(result, github, write_summary=write_summary)


if __name__ == "__main__":
    sys.exit(main())
