---
name: verifiability
description: Checks whether the change comes with enough evidence that it works, and whether everything that depends on it was updated with it.
---

# Verifiability

## Mission

Find claims in the change that nothing proves, and contracts that changed without their dependents. Ask: if this change broke tomorrow, what would notice?

## What to check

- New or changed behavior with no test, or with tests that never reach the changed lines.
- Changed contracts, such as a function signature, an API response, a configuration key, a database column or an event shape, whose callers, consumers or documentation were not updated. This dimension is the primary owner of these.
- Backward compatibility: existing data, stored configuration or older clients that the change no longer handles, and migrations that are missing or cannot be rolled back.
- Dependency changes that are not reflected in the lockfile, or lockfile changes the PR does not explain.
- Observability: a new failure mode with no log, metric or error message that would let someone diagnose it.
- When sandbox results are provided, failing or skipped checks that the PR does not mention.

## Not this dimension

Tests that pass without proving the requirement belong to spec drift.

## Shared rules

Follow `_shared.md` for severity, basis and output.
