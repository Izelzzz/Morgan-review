"""HMAC signatures between the CI job and the service.

The CI job signs every request with the shared secret: the timestamp, method, path and
raw body, joined by newlines. A request is accepted only with a valid signature made
within `MAX_SKEW_SECONDS`, so a captured request cannot be replayed later.
"""

import hashlib
import hmac
import time
from collections.abc import Callable

TIMESTAMP_HEADER = "X-Review-Agents-Timestamp"
SIGNATURE_HEADER = "X-Review-Agents-Signature"
MAX_SKEW_SECONDS = 300
_PREFIX = "sha256="


def sign(secret: str, timestamp: str, method: str, path: str, body: bytes) -> str:
    message = f"{timestamp}\n{method.upper()}\n{path}\n".encode() + body
    digest = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    return _PREFIX + digest


def verify(
    secret: str,
    timestamp: str | None,
    signature: str | None,
    method: str,
    path: str,
    body: bytes,
    clock: Callable[[], float] = time.time,
) -> bool:
    if not timestamp or not signature or not timestamp.isdigit():
        return False
    if abs(clock() - int(timestamp)) > MAX_SKEW_SECONDS:
        return False
    expected = sign(secret, timestamp, method, path, body)
    return hmac.compare_digest(expected, signature)
