"""The weak baseline of the discrimination check (D-041): one model call, diff only.

The model sees the PR's title, description and diff (base...head of the case's branches
in a local clone of the evaluation repository) and nothing else: no tools, no repository
context, no walkthrough. A medium or hard case this baseline hits in every run is too easy.

    python eval/baseline.py --eval-repo ../morgan-eval --out eval/results/<dir> <case> ...

The model's key is not read here: the network proxy adds it to requests to the API host.
Each run is written to `<out>/<case>/baseline-<n>.json`; judging is a separate step.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

CASES = Path(__file__).parent / "cases"
RUNS = 3

PROMPT = """You review a pull request. Below are its title, description and diff; you have
nothing else. Report every real problem the change introduces: bugs, races, security holes,
broken contracts, missing handling. Do not report style.

Answer with JSON only: {{"findings": [{{"file": "<path>", "line_start": <n>, "line_end": <n>,
"severity": "blocking|important|robustness|nit", "title": "<one line>",
"why": "<the root cause and its consequence>"}}]}}. Lines are head-side lines of the diff.

# Title
{title}

# Description
{body}

# Diff
{diff}
"""


def diff(eval_repo: Path, case: str) -> str:
    args = ["git", "-C", str(eval_repo), "-c", "core.quotePath=false", "diff", "--no-ext-diff",
            f"case/{case}/base...case/{case}/head"]  # fmt: skip
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout  # noqa: S603


def call(base_url: str, model: str, prompt: str) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(  # noqa: S310 - fixed https endpoint
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        method="POST",
        # The proxy replaces the placeholder with the real key.
        headers={"Content-Type": "application/json", "Authorization": "Bearer placeholder"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
        reply: dict[str, Any] = json.loads(response.read())
    return reply


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--eval-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--first", type=int, default=1, help="number of the first run")
    args = parser.parse_args(argv)
    for case_id in args.cases:
        case = yaml.safe_load((CASES / f"{case_id}.yaml").read_text(encoding="utf-8"))
        prompt = PROMPT.format(
            title=case["pr"]["title"], body=case["pr"]["body"], diff=diff(args.eval_repo, case_id)
        )
        for n in range(args.first, args.first + args.runs):
            started = time.monotonic()
            reply: dict[str, Any] = {}
            content: str | None = None
            findings: list[Any] | None = None
            error: str | None = None
            try:
                reply = call(args.base_url, args.model, prompt)
                content = reply["choices"][0]["message"]["content"]
                parsed = json.loads(content or "")
                if isinstance(parsed, dict) and isinstance(parsed.get("findings"), list):
                    findings = parsed["findings"]
                else:
                    error = "the reply is not a findings object"
            except (OSError, ValueError, KeyError, IndexError) as exc:
                # A failed run is written down, and the case's other runs still go ahead.
                error = f"{type(exc).__name__}: {exc}"
            result = {
                "case": case_id,
                "run": n,
                "model": args.model,
                "seconds": round(time.monotonic() - started, 1),
                "usage": reply.get("usage"),
                "findings": findings,
                "content": content if findings is None else None,  # judged by hand
                "error": error,
            }
            path = args.out / case_id / f"baseline-{n}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{case_id} baseline {n}: {len(findings or [])} findings", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
