# Shared helpers for review-lab's git hooks and CI checks. Source, don't run.

HOOK_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fail() { printf 'review-lab hook: %s\n' "$*" >&2; exit 1; }
warn() { printf 'review-lab hook: warning: %s\n' "$*" >&2; }

# Git writes these subjects itself (merge, revert, rebase --autosquash), so they can't follow our format.
is_git_generated_subject() {
  case "$1" in
    "Merge "* | "Revert \""* | "fixup! "* | "squash! "* | "amend! "*) return 0 ;;
  esac
  return 1
}

# First non-comment, non-blank line of a commit message.
subject_of() { grep -v '^#' | grep -m1 -v '^[[:space:]]*$' || true; }

# The installed copy goes stale when main's hooks change; warn instead of failing so commits still work.
warn_if_stale() {
  local installed current
  installed="$(cat "$HOOK_LIB_DIR/../.source-tree" 2>/dev/null || true)"
  current="$(git rev-parse -q --verify main:.githooks 2>/dev/null || true)"
  if [ -n "$installed" ] && [ -n "$current" ] && [ "$installed" != "$current" ]; then
    warn "hooks are older than main's .githooks; run: git show main:scripts/install-hooks.sh | bash"
  fi
}

# Reads `git diff -U0` output on stdin; prints added lines that look like credentials.
# A line containing "secret-scan: allow" is skipped, for deliberate test fixtures.
find_secrets() {
  grep -E '^\+' | grep -vE '^\+\+\+ ' | grep -v 'secret-scan: allow' | grep -nE \
    -e 'AKIA[0-9A-Z]{16}' \
    -e '-----BEGIN [A-Z ]*PRIVATE KEY-----' \
    -e 'gh[pousr]_[A-Za-z0-9]{36}' \
    -e 'github_pat_[A-Za-z0-9_]{22,}' \
    -e 'xox[abprs]-[A-Za-z0-9-]{10,}' \
    -e 'sk-(ant-)?[A-Za-z0-9_-]{20,}' \
    -e 'AIza[0-9A-Za-z_-]{35}' \
    -e '(api[_-]?key|secret|passw(or)?d|token)["'\'']?[[:space:]]*[:=][[:space:]]*["'\''][^"'\'' ]{12,}["'\'']' \
    || true
}

# Files that are almost always credentials, whatever their content.
SECRET_FILE_NAMES='(^|/)(\.env(\..*)?|id_rsa|id_ed25519|.*\.pem|.*\.p12|.*\.key)$'
