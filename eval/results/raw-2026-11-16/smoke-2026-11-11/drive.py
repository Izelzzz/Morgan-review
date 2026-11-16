"""Smoke driver: the CI script's review() and post() against a local service."""
import importlib.util, json, subprocess, sys, time, urllib.request
from pathlib import Path

spec = importlib.util.spec_from_file_location("ci", "/home/user/review-lab-wt/ci/ci/review_agents_ci.py")
ci = importlib.util.module_from_spec(spec); spec.loader.exec_module(ci)

class LocalHttp(ci.Http):
    def call(self, method, url, headers, body=None):
        if url.startswith("http://127.0.0.1:"):
            req = urllib.request.Request(url, data=body, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    raw = r.read(); return r.status, json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                raw = e.read(); return e.code, (json.loads(raw) if raw else None)
        return super().call(method, url, headers, body)

here = Path(__file__).parent
token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
gh = ci.GitHub("https://api.github.com", "Izelzzz/review-lab", token)
cmd = sys.argv[1]
result = here / "result.json"
if cmd == "review":
    pr = gh.ok("GET", f"/pulls/{sys.argv[2]}")
    event = {"action": "synchronize", "pull_request": pr, "repository": {"default_branch": "main"}}
    svc = ci.Service("http://127.0.0.1:8080", (here / "secret").read_text().strip(), http=LocalHttp())
    t = time.time()
    def sleep(s): print(f"  waiting… {int(time.time()-t)}s", flush=True); time.sleep(s)
    code = ci.review("pull_request_target", event, gh, svc, result, sleep=sleep)
    r = json.loads(result.read_text())
    print("exit", code, "status", r["status"], r.get("reason") or (r.get("review") or {}).get("error"))
else:
    code = ci.post(json.loads(result.read_text()), gh, write_summary=lambda m: print("summary:", m))
    print("exit", code)
