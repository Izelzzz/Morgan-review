from dataclasses import replace

from langchain_core.messages import HumanMessage, SystemMessage

from review_agents.context.grouping import signature_tokens
from review_agents.context.modules import ModuleSummary
from review_agents.context.pack import (
    AGENT_SHARE,
    DIMENSIONS,
    GENERAL,
    SHARED_RULES,
    ContextPack,
    build_packs,
    fixed_tokens,
    module_finder,
    pack_budget,
)
from review_agents.context.pr_layer import ChangedFile, PrLayer, PrRequest
from review_agents.context.repo_layer import RepoLayer
from review_agents.context.tokens import estimate_tokens

SHA = "a" * 40


def repo_layer() -> RepoLayer:
    return RepoLayer(
        commit=SHA,
        conventions=[("AGENTS.md", "Never use floats for money.\n")],
        tree="./ (3 files)",
        modules=[
            ModuleSummary("payments", "payments/ · 2 files · Python", "Handles refunds."),
            ModuleSummary("payments/gateway", "payments/gateway/ · 1 files · Python", None),
            ModuleSummary("", "./ · 1 files · Python", None),
        ],
    )


def patch(path: str, body: str, size: int = 0) -> str:
    filler = "".join(f"+    pad_{i} = {i}\n" for i in range(size))
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n{body}{filler}"


def pr_layer(diffs: dict[str, str]) -> PrLayer:
    return PrLayer(
        request=PrRequest("octo/shop", 7, "Clamp refunds", "Refunds never exceed paid.", SHA, SHA),
        issues=[(42, "#42 Partial refunds [open] labels: -\n\nNo more than paid.")],
        specs=[],
        files=[ChangedFile("M", path) for path in diffs],
        diffs=diffs,
        excerpts=[],
        previous_findings=[],
    )


def test_every_dimension_reads_the_same_prefix_and_its_own_skill_last() -> None:
    [pack] = build_packs(repo_layer(), pr_layer({"a.py": patch("a.py", "+x = 1\n")}), 128_000)

    prompts = {d: pack.messages(d) for d in DIMENSIONS}

    prefixes = {tuple(m.content for m in messages[:-1]) for messages in prompts.values()}
    assert len(prefixes) == 1  # identical, so a prompt cache can reuse it across dimensions
    for dimension, messages in prompts.items():
        assert f"name: {dimension}" in str(messages[-1].content)


def test_the_prefix_runs_from_shared_rules_to_the_repository_to_the_pull_request() -> None:
    [pack] = build_packs(repo_layer(), pr_layer({"a.py": patch("a.py", "+x = 1\n")}), 128_000)

    system, context, task = pack.messages("security")

    assert isinstance(system, SystemMessage) and system.content == SHARED_RULES
    assert isinstance(context, HumanMessage) and isinstance(task, HumanMessage)
    text = str(context.content)
    assert text.index("Never use floats for money.") < text.index("Refunds never exceed paid.")
    assert text.index("Handles refunds.") < text.index("diff --git a/a.py")


def test_a_pr_that_fits_is_one_pack_and_a_larger_one_is_split_with_full_intent() -> None:
    diffs = {p: patch(p, f"+def f_{p[0]}():\n", size=300) for p in ("a.py", "b.py", "c.py")}
    pr = pr_layer(diffs)

    assert len(build_packs(repo_layer(), pr, 128_000)) == 1

    packs = build_packs(repo_layer(), pr, window_tokens=pack_overhead(pr) + 4_000)

    assert len(packs) > 1
    for pack in packs:
        assert "Refunds never exceed paid." in pack.prefix
        assert "M a.py" in pack.prefix  # every group sees the whole file list


def pack_overhead(pr: PrLayer) -> int:
    """Window tokens that leave nothing for diffs once the agents' share is kept free."""
    base = ContextPack(SHARED_RULES, repo_layer(), replace(pr, diffs={}))
    return int((fixed_tokens(base) + signature_tokens(pr)) / (1 - AGENT_SHARE)) + 1


def test_when_the_shared_context_alone_fills_the_window_each_unit_is_its_own_group() -> None:
    diffs = {p: patch(p, f"+def f_{p[0]}():\n", size=50) for p in ("a.py", "b.py")}

    packs = build_packs(repo_layer(), pr_layer(diffs), window_tokens=1_000)

    assert len(packs) == 2
    for pack in packs:
        assert "larger than one review window" in (pack.pr.group or "")
        assert any("leave no room" in note for note in pack.pr.notes)


def test_a_file_belongs_to_the_deepest_module_holding_it() -> None:
    module_of = module_finder(repo_layer())

    assert module_of("payments/refund.py") == "payments"
    assert module_of("payments/gateway/stripe.py") == "payments/gateway"
    assert module_of("payments-old/x.py") == ""
    assert module_of("setup.py") == ""


def test_the_room_kept_for_other_groups_signatures_does_not_split_a_pr_that_fits() -> None:
    # Many changed definitions: their signatures only matter once there are groups.
    many = "".join(f"+def function_{i}(value):\n" for i in range(80))
    diffs = {"a.py": patch("a.py", many), "b.py": patch("b.py", "+x = 1\n", size=40)}
    pr = pr_layer(diffs)
    base = ContextPack(SHARED_RULES, repo_layer(), replace(pr, diffs={}))
    changes = sum(estimate_tokens(p) for p in diffs.values())
    without_signatures = fixed_tokens(base)
    # Room for every change, but not for the changes plus the signature allowance.
    window = int((without_signatures + changes + signature_tokens(pr) // 2) / (1 - AGENT_SHARE))

    assert len(build_packs(repo_layer(), pr, window_tokens=window)) == 1


def test_a_pack_leaves_the_reply_its_reserve_even_in_a_small_window() -> None:
    # Seventy percent of the window normally; never so much that the reply has no room.
    assert pack_budget(128_000, output_reserve=4_000) == 89_600
    assert pack_budget(8_000, output_reserve=4_000) == 4_000


def test_every_agent_is_told_the_configured_language_after_its_task() -> None:
    layer = pr_layer({"a.py": patch("a.py", "+x = 1\n")})
    [zh] = build_packs(repo_layer(), layer, 128_000)
    [en] = build_packs(repo_layer(), layer, 128_000, language="en")

    zh_task, en_task = (str(p.messages("security")[-1].content) for p in (zh, en))

    assert "Simplified Chinese" in zh_task.rsplit("\n\n", 1)[-1]  # the configured default
    assert "English" in en_task.rsplit("\n\n", 1)[-1]
    # A quote is the rule as written in its source, whatever the language.
    assert "quote" in zh_task.rsplit("\n\n", 1)[-1]
    # The language comes after the shared opening, which stays the same.
    assert [m.content for m in zh.messages("security")[:2]] == [
        m.content for m in en.messages("security")[:2]
    ]


def test_the_room_kept_for_the_task_follows_the_agents_that_run() -> None:
    # Issue #83: the default strategy keeps the room it kept before; a general agent's
    # longer task is counted only when one runs.
    base = ContextPack(SHARED_RULES, repo_layer(), pr_layer({}))
    single = fixed_tokens(base)
    general = fixed_tokens(base, [(GENERAL, DIMENSIONS)])

    assert single == fixed_tokens(base, [(d, (d,)) for d in DIMENSIONS])
    assert general > single
    assert general == fixed_tokens(base, [(GENERAL, DIMENSIONS), ("security", ("security",))])
