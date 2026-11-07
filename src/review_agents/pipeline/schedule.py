"""Who gets the next model call of a review (decision D-031).

A review has a fixed number of model calls in flight (`model_concurrency`), shared by
the dimension agents and the verifiers. Phases have no time limits of their own: when a
call ends, its slot goes to the waiting call of the highest priority. Only a verified
finding is published, so verifying what was found comes before finding more, and the
more severe a finding, the sooner it is verified. A call that is running is never
interrupted.
"""

import heapq
import itertools
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from review_agents.pipeline.agent import Severity

# Lower runs first: verify 🔴, verify 🟡, a dimension agent's next turn, verify 🔵.
_VERIFY: dict[Severity, int] = {"blocking": 0, "important": 1, "robustness": 3, "nit": 3}
DIMENSION_TURN = 2


def verify_priority(severity: Severity) -> int:
    """The priority of a verifier's model calls for a finding of this level."""
    return _VERIFY[severity]


class Slots:
    """`count` model calls at once; a freed slot goes to the highest priority (lowest number).

    Waiters of the same priority are served in the order they came.
    """

    def __init__(self, count: int) -> None:
        self._free = count
        self._waiting: list[tuple[int, int]] = []
        self._order = itertools.count()
        self._changed = threading.Condition()

    def waiting(self) -> int:
        with self._changed:
            return len(self._waiting)

    @contextmanager
    def held(self, priority: int) -> Iterator[None]:
        with self._changed:
            me = (priority, next(self._order))
            heapq.heappush(self._waiting, me)
            while not (self._free > 0 and self._waiting[0] == me):
                self._changed.wait()
            heapq.heappop(self._waiting)
            self._free -= 1
            # Another slot may be free for the next waiter in line.
            self._changed.notify_all()
        try:
            yield
        finally:
            with self._changed:
                self._free += 1
                self._changed.notify_all()
