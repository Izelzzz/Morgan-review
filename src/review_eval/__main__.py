"""`python -m review_eval check CASES [--answers ANSWERS]`: check the golden set's files."""

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from review_eval.case import load_set


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m review_eval")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="check every case, and its answer when given")
    check.add_argument("cases", type=Path, help="the directory holding one directory per case")
    check.add_argument(
        "--answers",
        type=Path,
        help="the directory holding the answers; leave out to check the cases alone",
    )
    args = parser.parse_args(argv)

    golden = load_set(args.cases, args.answers)
    for error in golden.errors:
        print(error, file=sys.stderr)
    splits = Counter(c.case.split for c in golden.cases)
    summary = ", ".join(f"{n} {split}" for split, n in sorted(splits.items())) or "none"
    print(f"{len(golden.cases)} cases loaded ({summary}); {len(golden.errors)} errors")
    if golden.drafts:
        print(f"draft answers, not scored: {', '.join(golden.drafts)}")
    return 1 if golden.errors else 0


if __name__ == "__main__":
    sys.exit(main())
