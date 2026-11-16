"""Run evaluation cases against a review-agents service (issue #17, D-040).

Each case is a PR in the evaluation repository: `case/<id>/base` holds the PR's base,
`case/<id>/head` its head, and a PR goes from head into base. The service is asked to
review it as a manual `/review`, which is always a fresh full review, so every run is
independent; the default branch it reads its context from is the case's base branch.

    python eval/run.py --service http://127.0.0.1:8080 --secret-file secret \
        --out eval/results/2026-11-16-walkthrough_first py-multipart-traversal-001 ...

Each run's result (report, findings, usage as the service returns them) is written to
`<out>/<case>/run-<n>.json`; `<out>/summary.json` lists them. Judging is a separate
step (eval design §8).
"""

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

CASES = Path(__file__).parent / "cases"
GITHUB = "https://api.github.com"
POLL_SECONDS = 15
WAIT_SECONDS = 45 * 60
# Must-pass cases run five times (pass^5), the others three (eval design §3).
RUNS = {"must_pass": 5}
DEFAULT_RUNS = 3


def http(method: str, url: str, headers: dict[str, str], body: bytes | None = None) -> Any:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {url}: {exc.code} {exc.read()[:300]!r}") from None
    return json.loads(raw) if raw else None


def github(path: str, token: str) -> Any:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    return http("GET", f"{GITHUB}{path}", headers)


def signed(service: str, secret: str, method: str, path: str, body: bytes = b"") -> Any:
    """A request signed the way `review_agents.auth` verifies it."""
    timestamp = str(int(time.time()))
    message = f"{timestamp}\n{method}\n{path}\n".encode() + body
    signature = "sha256=" + hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    headers = {
        "X-Review-Agents-Timestamp": timestamp,
        "X-Review-Agents-Signature": signature,
        "Content-Type": "application/json",
    }
    return http(method, service.rstrip("/") + path, headers, body or None)


def find_pr(repo: str, case: str, token: str) -> dict[str, Any]:
    owner = repo.split("/")[0]
    head = f"{owner}:case/{case}/head"
    pulls = github(f"/repos/{repo}/pulls?state=open&head={head}", token)
    if not pulls:
        raise SystemExit(f"{case}: no open PR from case/{case}/head in {repo}")
    pr: dict[str, Any] = pulls[0]
    return pr


def review_once(
    service: str, secret: str, repo: str, case: dict[str, Any], pr: dict[str, Any], token: str
) -> dict[str, Any]:
    request = {
        "repo": repo,
        "pr": pr["number"],
        "base_sha": pr["base"]["sha"],
        "head_sha": pr["head"]["sha"],
        "default_branch": pr["base"]["ref"],  # the case's base: context as of the PR
        "title": case["pr"]["title"],
        "body": case["pr"]["body"],
        "event": "manual",  # a fresh full review every run
        "token": token,
    }
    created = signed(service, secret, "POST", "/v1/reviews", json.dumps(request).encode())
    started = time.monotonic()
    while True:
        polled: dict[str, Any] = signed(service, secret, "GET", f"/v1/reviews/{created['id']}")
        if polled["status"] in ("done", "failed", "superseded"):
            polled["seconds"] = round(time.monotonic() - started, 1)
            return polled
        if time.monotonic() - started > WAIT_SECONDS:
            return {"id": created["id"], "status": "timeout"}
        time.sleep(POLL_SECONDS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="+", help="case ids, as in eval/cases/<id>.yaml")
    parser.add_argument("--service", required=True)
    parser.add_argument("--secret-file", type=Path, required=True)
    parser.add_argument("--repo", default="izelcompany/morgan-eval")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, help="runs per case, instead of the tier's")
    args = parser.parse_args(argv)
    token = os.environ["GITHUB_TOKEN"]
    secret = args.secret_file.read_text(encoding="utf-8").strip()
    summary = []
    for case_id in args.cases:
        case = yaml.safe_load((CASES / f"{case_id}.yaml").read_text(encoding="utf-8"))
        pr = find_pr(args.repo, case_id, token)
        runs = args.runs or RUNS.get(case["tier"], DEFAULT_RUNS)
        for n in range(1, runs + 1):
            result = review_once(args.service, secret, args.repo, case, pr, token)
            path = args.out / case_id / f"run-{n}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
            summary.append({"case": case_id, "run": n, "status": result["status"]})
            print(f"{case_id} run {n}: {result['status']}", flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
