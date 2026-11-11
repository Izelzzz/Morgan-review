---
name: maintainability
description: Checks whether the next person can understand, change and extend this code safely.
---

# Maintainability

## Mission

Find places where the change will make future work slower or riskier. A finding here names the concrete future change it makes harder.

## What to check

- Names that hide or misstate what a thing does.
- Module boundaries: logic placed in the wrong layer, new dependencies between modules that should stay separate.
- Duplicated logic that will drift apart, and code the change leaves unused.
- Temporary workarounds shipped as the real solution, with nothing tracking their removal. This dimension is the primary owner of these.
- Abstraction that the change does not need yet: layers, options or interfaces with a single use.
- Departures from the conventions the repository already follows, without a reason.
- Comments that restate what the code does instead of explaining why, and comments that are out of date.

## Not this dimension

Formatting and style enforced by a formatter or linter are out of scope. Behavior that differs from the spec belongs to spec drift.

## Shared rules

Follow `_shared.md` for severity, basis and output.
