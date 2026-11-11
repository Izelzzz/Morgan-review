---
name: security
description: Checks whether the change lets untrusted input, leaked secrets or missing permission checks harm users or the system.
---

# Security

## Mission

Find ways the change can be abused. Start from every place untrusted data enters the changed code and follow it to where it is used.

## What to check

- Injection: SQL, shell commands, template injection, path traversal, cross-site scripting, server-side request forgery, unsafe deserialization.
- Secrets: keys, tokens or passwords in code, logs, error messages, test fixtures or build output.
- Authorization: an action or a piece of data reachable without the right permission check, or checked in one path but not another.
- Untrusted input used without validation in file access, network calls, redirects or resource limits.
- New or upgraded dependencies: known vulnerabilities, unmaintained packages, names close to popular packages, versions that are not pinned.
- Cryptography: weak algorithms, hardcoded keys, predictable random values used for security.
- CI and workflow changes: broader token permissions, untrusted code running with secrets, actions that are not pinned.
- Personal data collected, logged or kept longer than the feature needs.

## Not this dimension

Crashes on ordinary unexpected input belong to edge cases, unless an attacker can use them to cause harm.

## Shared rules

Follow `_shared.md` for severity, basis and output.
