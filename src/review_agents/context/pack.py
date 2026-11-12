"""The context pack: what each dimension agent of a review reads, in cache-friendly order.

Every dimension gets the same prefix: the shared rules, then the repository layer, then
the pull request layer. Only the last message, the dimension's SKILL.md and its task,
differs, so a model service with prompt caching reuses the prefix across the five
agents and across each agent's turns (context design §5).
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import PurePosixPath

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from review_agents.context.grouping import grouped_layers, signature_tokens
from review_agents.context.pr_layer import PrLayer
from review_agents.context.repo_layer import RepoLayer
from review_agents.context.tokens import estimate_tokens
from review_agents.repo_config import Language

DIMENSIONS = ("spec_drift", "edge_cases", "security", "maintainability", "verifiability")
_SKILLS = files("review_agents") / "skills"
SHARED_RULES = (_SKILLS / "_shared.md").read_text(encoding="utf-8")
SKILLS = {d: (_SKILLS / d / "SKILL.md").read_text(encoding="utf-8") for d in DIMENSIONS}
# One agent holding several dimensions' rules (strategies `general` and
# `walkthrough_first`, D-038) tags each finding with the dimension it belongs to.
GENERAL = "general"
GENERAL_TASK = "Review the pull request above for each of these dimensions"

# The part of the window kept free for what an agent adds while it works: tool results,
# its reasoning and its findings.
AGENT_SHARE = 0.3
NO_ROOM_NOTE = (
    "The intent, repository layer and file list alone leave no room for the changes in"
    " one review window; each group is larger than the window."
)


LANGUAGE_NAMES: dict[Language, str] = {"zh": "Simplified Chinese", "en": "English"}


def language_note(language: Language) -> str:
    """The language an agent writes in, told after its task (decision D-034)."""
    return (
        f"Write every title, why, suggestion and reason, and the explanation and what each"
        f" place shows in a basis, in {LANGUAGE_NAMES[language]}. Keep file paths, code and"
        " identifiers as they are, and copy each quote exactly as it is written in its"
        " source: a quote is never translated."
    )


def task(dimension: str, dimensions: tuple[str, ...] | None = None) -> str:
    """The agent's own task: one dimension's rules, or several for a general agent."""
    held = dimensions or (dimension,)
    if len(held) > 1:
        rules = "\n\n".join(f"# Dimension `{d}`\n\n{SKILLS[d]}" for d in held)
        return (
            f"{rules}\n\n"
            f"{GENERAL_TASK}, whose rules are given above: {', '.join(held)}. Read more"
            " code with your tools wherever the context is not enough. Report each problem"
            " once, and set its `dimension` field to the one dimension above it belongs to"
            " most. Report findings in the output format of the shared rules."
        )
    dimension = held[0]
    return (
        f"{SKILLS[dimension]}\n\n"
        f"Review the pull request above for the {dimension} dimension only. Read more code"
        " with your tools wherever the context is not enough. Report findings in the output"
        " format of the shared rules."
    )


@dataclass(frozen=True)
class ContextPack:
    shared_rules: str
    repo: RepoLayer
    pr: PrLayer
    language: Language = "zh"

    @property
    def prefix(self) -> str:
        """Everything before the dimension's own message, identical for every dimension."""
        return f"{self.repo.render()}\n\n{self.pr.render()}"

    def messages(
        self, dimension: str, dimensions: tuple[str, ...] | None = None
    ) -> list[BaseMessage]:
        return [
            SystemMessage(self.shared_rules),
            HumanMessage(self.prefix),
            HumanMessage(f"{task(dimension, dimensions)}\n\n{language_note(self.language)}"),
        ]


def fixed_tokens(base: ContextPack) -> int:
    """What the one pack carries whatever its changes: rules, repository, intent, task.

    Split packs carry the definitions other groups change on top of this, at most
    `signature_tokens`; a pack that is not split has no other groups.
    """
    longest_note = max(estimate_tokens(language_note(lang)) for lang in LANGUAGE_NAMES)
    tasks = [task(d) for d in DIMENSIONS] + [task(GENERAL, DIMENSIONS)]
    longest_task = max(estimate_tokens(t) for t in tasks) + longest_note
    return estimate_tokens(base.shared_rules) + estimate_tokens(base.prefix) + longest_task


def module_finder(repo: RepoLayer) -> Callable[[str], str]:
    """The deepest module of the repository layer holding a path; "" for the root."""
    paths = sorted((m.path for m in repo.modules if m.path), key=len, reverse=True)

    def module_of(path: str) -> str:
        parents = {str(p) for p in PurePosixPath(path).parents}
        return next((m for m in paths if m in parents), "")

    return module_of


def pack_budget(window_tokens: int, output_reserve: int = 0) -> int:
    """Tokens one pack may take: `AGENT_SHARE` of the window is left for the agent's work,
    and never less than the room kept for its reply (decision D-030)."""
    return min(int(window_tokens * (1 - AGENT_SHARE)), window_tokens - output_reserve)


def build_packs(
    repo: RepoLayer,
    layer: PrLayer,
    window_tokens: int,
    output_reserve: int = 0,
    language: Language = "zh",
) -> list[ContextPack]:
    """One pack per group of changes; a single pack when the whole PR fits (decision D-008)."""
    base = ContextPack(
        SHARED_RULES, repo, replace(layer, diffs={}, excerpts=[], notes=[]), language
    )
    budget = pack_budget(window_tokens, output_reserve) - fixed_tokens(base)
    module_of = module_finder(repo)
    # Only a PR that does not fit is split (D-008), so the room for other groups'
    # signatures is kept back only once splitting is certain.
    whole = grouped_layers(layer, module_of, max(budget, 1))
    if budget >= 1 and len(whole) == 1 and whole[0].group is None:
        return [ContextPack(SHARED_RULES, repo, layer, language)]
    room = budget - signature_tokens(layer)
    if room < 1:
        # Every unit then gets a group of its own, flagged as larger than the window.
        layer = replace(layer, notes=[*layer.notes, NO_ROOM_NOTE])
        room = 1
    groups = grouped_layers(layer, module_of, room)
    return [ContextPack(SHARED_RULES, repo, group, language) for group in groups]
