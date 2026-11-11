"""Read-only GitHub tokens the CI job hands over, held in memory and never stored.

The service keeps no GitHub credentials (architecture §3): a token lives here from the
request that brought it until its review finishes, and is gone after a restart.
"""

import threading


class TokenVault:
    def __init__(self) -> None:
        self._tokens: dict[int, str | None] = {}
        self._lock = threading.Lock()

    def put(self, review_id: int, token: str | None) -> None:
        with self._lock:
            self._tokens[review_id] = token

    def get(self, review_id: int) -> str | None:
        with self._lock:
            return self._tokens.get(review_id)

    def discard(self, review_id: int) -> None:
        with self._lock:
            self._tokens.pop(review_id, None)
