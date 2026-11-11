---
name: spec_drift
description: Checks whether the change does what its pull request, linked issue and spec say it should, and nothing else.
---

# Spec drift

## Mission

Find places where the code and its stated purpose disagree. Read the PR description, the linked issue and the spec first, then read the diff against them.

If no PR description, issue or spec states the intended behavior, judge against the PR title and description alone, and say in each finding that no spec was available.

## What to check

- Behavior the PR, issue or spec asks for that the change does not implement, or implements differently.
- Branches the spec names that the code leaves out: a status, a role, an error case, a configuration option.
- Changes outside the stated scope, such as an unrelated refactor or a feature nobody asked for.
- Behavior changes the PR does not declare, such as a changed default, a renamed field or a different error.
- Tests that pass without proving the requirement: they assert on a mock instead of the behavior, assert something that is always true, are skipped, skip the branch they claim to cover, update a snapshot to hide a change, or would still pass if the feature were removed. This dimension is the primary owner of these tests.
- Documentation, comments or examples that now contradict the new behavior.

## Not this dimension

Missing tests for behavior that is correct belong to verifiability. Hostile input belongs to security.

## Shared rules

Follow `_shared.md` for severity, basis and output.
