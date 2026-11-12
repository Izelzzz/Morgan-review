"""Who writes a review's findings and its walkthrough (issue #83, D-038).

A strategy decides only the writing: which agents look for problems, with which
dimensions' rules, and whether they start from the walkthrough's conversation. What
follows is the same for every strategy: each agent's findings are verified as soon as
it hands them in, then merged, ranked and reported.
"""

from dataclasses import dataclass
from typing import Literal

from review_agents.context.pack import DIMENSIONS, GENERAL
from review_agents.repo_config import RepoConfig

Strategy = Literal["dimensions", "general", "walkthrough_first"]

# Quick mode looks at edge cases and security only (architecture §4.3); Deep mode runs
# each agent three times and keeps what any run reports (D-022).
QUICK_DIMENSIONS = ("edge_cases", "security")
DEEP_RUNS = 3


@dataclass(frozen=True)
class Author:
    """One agent that looks for problems: its name, and whose rules it holds."""

    name: str
    dimensions: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    authors: list[Author]
    runs: int
    # The authors carry on from the walkthrough's conversation once it is written.
    walkthrough_first: bool = False

    @property
    def dimensions(self) -> list[str]:
        """Every dimension some author holds, as the report lists them."""
        return [d for d in DIMENSIONS if any(d in a.dimensions for a in self.authors)]


def mode_plan(config: RepoConfig) -> tuple[list[str], int]:
    """The dimensions a review covers and how many times each, from the config's mode."""
    enabled = config.enabled_dimensions
    if config.mode == "quick":
        # With neither of them enabled, the first enabled one still makes it a review.
        return [d for d in enabled if d in QUICK_DIMENSIONS] or enabled[:1], 1
    return enabled, DEEP_RUNS if config.mode == "deep" else 1


def plan(strategy: Strategy, config: RepoConfig) -> Plan:
    dimensions, runs = mode_plan(config)
    if strategy == "dimensions" or len(dimensions) == 1:
        authors = [Author(d, (d,)) for d in dimensions]
    else:
        authors = [Author(GENERAL, tuple(dimensions))]
    return Plan(authors, runs, walkthrough_first=strategy == "walkthrough_first")
