---
name: edge_cases
description: Checks how the change behaves at the limits of its inputs, under concurrency, and when the things it depends on fail.
---

# Edge cases

## Mission

Find inputs and conditions inside the PR's usage scenario that make the change fail or, worse, return a wrong result without any error. For each one, trace a concrete value through the code to the failure.

## What to check

- Empty, missing and zero values; the first and last item of a range; values at the size limit of their type.
- Number handling: overflow, rounding and precision, unit mismatches.
- Dates and times: time zones, daylight saving changes, month and year boundaries.
- Text: encoding, unusual characters, very long input.
- Concurrency: two requests at once, retries, ordering assumptions, shared state without a lock.
- Error handling: errors that are swallowed, retried forever, or reported with the wrong cause.
- Resources such as files, connections and locks that are not released on the error path.
- Calls to networks, processes or locks with no timeout.
- Operations that are not safe to repeat: a double submit, a retried request or a replayed event that applies twice.
- External dependencies that are slow, unavailable or return something unexpected.
- Missing permissions, files or configuration at runtime.
- Performance cliffs: work that grows much faster than the input, unbounded memory, a query per item.
- Release risk: the change misbehaving while old and new versions run side by side, or on existing data.

A silent wrong result is worse than an explicit failure. When the change turns an error into a quiet wrong answer, say so and rate it accordingly.

## Not this dimension

Input crafted by an attacker belongs to security. Missing tests belong to the finding about the bug they would catch.

## Shared rules

Follow `_shared.md` for severity, basis and output.
