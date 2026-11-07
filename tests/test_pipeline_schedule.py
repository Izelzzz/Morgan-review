import threading
import time

from review_agents.pipeline.schedule import (
    DIMENSION_TURN,
    Slots,
    verify_priority,
)


def test_verifying_comes_before_finding_more_and_robustness_after() -> None:
    # Decision D-031: the closer to publishable and the more severe, the sooner.
    order = sorted(
        [
            ("robustness", verify_priority("robustness")),
            ("dimension", DIMENSION_TURN),
            ("important", verify_priority("important")),
            ("blocking", verify_priority("blocking")),
        ],
        key=lambda item: item[1],
    )

    assert [name for name, _ in order] == ["blocking", "important", "dimension", "robustness"]


def test_a_free_slot_goes_to_the_waiter_of_the_highest_priority() -> None:
    slots = Slots(1)
    got: list[str] = []
    holder = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with slots.held(DIMENSION_TURN):
            holder.set()
            release.wait(5)

    def take(name: str, priority: int) -> None:
        with slots.held(priority):
            got.append(name)

    threads = [threading.Thread(target=hold)]
    threads[0].start()
    holder.wait(5)
    # Queued in the worst order: the lowest priority first.
    for name, priority in [
        ("robustness", verify_priority("robustness")),
        ("dimension", DIMENSION_TURN),
        ("blocking", verify_priority("blocking")),
    ]:
        thread = threading.Thread(target=take, args=(name, priority))
        thread.start()
        threads.append(thread)
        while slots.waiting() < len(threads) - 1:
            time.sleep(0.01)
    release.set()
    for thread in threads:
        thread.join(5)

    assert got == ["blocking", "dimension", "robustness"]


def test_waiters_of_one_priority_are_served_in_turn() -> None:
    slots = Slots(1)
    got: list[int] = []
    release = threading.Event()
    held = threading.Event()

    def hold() -> None:
        with slots.held(DIMENSION_TURN):
            held.set()
            release.wait(5)

    def take(n: int) -> None:
        with slots.held(DIMENSION_TURN):
            got.append(n)

    first = threading.Thread(target=hold)
    first.start()
    held.wait(5)
    threads = []
    for n in range(3):
        thread = threading.Thread(target=take, args=(n,))
        thread.start()
        threads.append(thread)
        while slots.waiting() < n + 1:
            time.sleep(0.01)
    release.set()
    for thread in [first, *threads]:
        thread.join(5)

    assert got == [0, 1, 2]


def test_no_more_than_the_slots_run_at_once() -> None:
    slots = Slots(2)
    running = 0
    most = 0
    lock = threading.Lock()

    def work() -> None:
        nonlocal running, most
        with slots.held(DIMENSION_TURN):
            with lock:
                running += 1
                most = max(most, running)
            time.sleep(0.02)
            with lock:
                running -= 1

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert most == 2
