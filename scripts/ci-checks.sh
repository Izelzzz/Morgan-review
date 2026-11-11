#!/usr/bin/env bash
# Checks every branch on origin. Local hooks can be skipped (--no-verify) or never installed; this can't.
# Run from a clone that has fetched all branches with full history.
set -uo pipefail
. "$(dirname "$0")/../.githooks/lib/common.sh" || { echo "ci-checks: can't load .githooks/lib/common.sh" >&2; exit 2; }

failures=0
problem() { printf '  ✗ %s\n' "$*"; failures=$((failures + 1)); }
section() { printf '\n== %s\n' "$*"; }

branches="$(git for-each-ref --format='%(refname:short)' refs/remotes/origin | grep -v '^origin$' | grep -v '/HEAD$')"

section "Commit messages on all branches"
for c in $(git rev-list $branches); do
  [ -n "$(git rev-list --parents -n1 "$c" | cut -s -d' ' -f2)" ] || continue   # root commit
  msg="$(git log -1 --format=%B "$c")"
  is_git_generated_subject "$(printf '%s\n' "$msg" | subject_of)" && continue
  printf '%s\n' "$msg" | "$HOOK_LIB_DIR/validate-commit-msg.sh" 2>/dev/null \
    || problem "${c:0:7} $(git log -1 --format=%s "$c")"
done

printf '\n'
[ "$failures" -eq 0 ] && { echo "All checks passed."; exit 0; }
echo "$failures problem(s) found."; exit 1
