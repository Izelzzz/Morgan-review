from review_agents.context.grouping import (
    changed_definitions,
    core_units,
    group_changes,
    grouped_layers,
    signature_tokens,
)
from review_agents.context.pr_layer import ChangedFile, Excerpt, PrLayer, PrRequest
from review_agents.context.tokens import estimate_tokens

SHA = "a" * 40


def patch(path: str, body: str, size: int = 0) -> str:
    filler = "".join(f"+    pad_{i} = {i}\n" for i in range(size))
    return (
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1,3 +1,4 @@\n{body}{filler}"
    )


def layer(diffs: dict[str, str], excerpts: list[Excerpt] | None = None) -> PrLayer:
    return PrLayer(
        request=PrRequest("octo/shop", 7, "Split orders", "Move orders out.", SHA, SHA),
        issues=[],
        specs=[],
        files=[ChangedFile("M", path) for path in diffs],
        diffs=diffs,
        excerpts=excerpts or [],
        previous_findings=[],
    )


def room_for(units: float, diffs: dict[str, str]) -> int:
    """A capacity that holds `units` of the largest patch."""
    return int(units * max(estimate_tokens(p) for p in diffs.values()))


def module_of(path: str) -> str:
    return path.split("/")[0]


def where(groups: list[list[str]]) -> dict[str, int]:
    return {path: i for i, group in enumerate(groups) for path in group}


def test_a_pr_that_fits_is_reviewed_as_one_group() -> None:
    pr = layer(
        {"a/x.py": patch("a/x.py", "+def alpha():\n"), "b/y.py": patch("b/y.py", "+x = 1\n")}
    )

    assert grouped_layers(pr, module_of, capacity_tokens=100_000) == [pr]


def test_independent_changes_are_packed_into_groups_that_fit() -> None:
    diffs = {p: patch(p, f"+def f_{p[0]}():\n", size=40) for p in ("a/x.py", "b/y.py", "c/z.py")}

    groups = group_changes(layer(diffs), module_of, capacity_tokens=room_for(2.5, diffs))

    assert len(groups) == 2
    assert sorted(p for g in groups for p in g) == sorted(diffs)


def sized(path: str, body: str, tokens: int) -> str:
    """A patch of about `tokens` tokens, so tests can control how files pack."""
    lines = body
    while estimate_tokens(patch(path, lines)) < tokens:
        lines += "+    pad = 0\n"
    return patch(path, lines)


def test_a_caller_stays_with_the_code_it_calls_across_modules() -> None:
    # Packed by size alone, refund (100) and notes (90) would not fit together, so
    # orders (70) would join notes; the reference to `refund` must keep it with refund.
    diffs = {
        "payments/refund.py": sized("payments/refund.py", "+def refund(order):\n", 100),
        "docs/notes.py": sized("docs/notes.py", "+def unrelated_thing():\n", 90),
        "api/orders.py": sized("api/orders.py", "+    return refund(order)\n", 70),
    }

    found = where(group_changes(layer(diffs), module_of, capacity_tokens=165))

    assert found["payments/refund.py"] == found["api/orders.py"]
    assert found["docs/notes.py"] != found["api/orders.py"]


def test_a_test_stays_with_the_code_it_tests() -> None:
    diffs = {
        "payments/refund.py": sized("payments/refund.py", "+x = 1\n", 100),
        "docs/notes.py": sized("docs/notes.py", "+z = 3\n", 90),
        "tests/test_refund.py": sized("tests/test_refund.py", "+y = 2\n", 70),
    }

    found = where(group_changes(layer(diffs), module_of, capacity_tokens=165))

    assert found["payments/refund.py"] == found["tests/test_refund.py"]


def test_a_file_larger_than_a_group_is_never_split() -> None:
    diffs = {
        "big/huge.py": patch("big/huge.py", "+def huge():\n", size=400),
        "a/x.py": patch("a/x.py", "+def alpha():\n"),
    }

    layers = grouped_layers(layer(diffs), module_of, capacity_tokens=500)

    huge = next(g for g in layers if "big/huge.py" in g.diffs)
    assert huge.diffs["big/huge.py"] == diffs["big/huge.py"]
    assert "larger than one review window" in (huge.group or "")


def test_small_changes_in_one_module_are_reviewed_together() -> None:
    # By size alone two of the four orders files would join the big file's group.
    diffs = {f"orders/m{i}.py": sized(f"orders/m{i}.py", f"+v{i} = {i}\n", 40) for i in range(4)}
    diffs["other/big.py"] = sized("other/big.py", "+w = 1\n", 200)

    found = where(group_changes(layer(diffs), module_of, capacity_tokens=290))

    assert len({found[f"orders/m{i}.py"] for i in range(4)}) == 1


def test_each_group_sees_the_whole_pr_and_what_other_groups_change() -> None:
    diffs = {
        "orders/service.py": patch(
            "orders/service.py", "-def cancel(self):\n+def cancel(self, reason):\n", size=40
        ),
        "notify/events.py": patch("notify/events.py", "+def on_paid():\n", size=40),
    }
    excerpts = [Excerpt("usage", "cancel", "api/orders.py", "1: service.cancel()")]

    layers = grouped_layers(layer(diffs, excerpts), module_of, capacity_tokens=room_for(1.5, diffs))

    assert len(layers) == 2
    notify = next(g for g in layers if "notify/events.py" in g.diffs)
    rendered = notify.render()
    assert "M orders/service.py" in rendered  # full file list
    assert "Move orders out." in rendered  # full intent
    assert "+def cancel(self, reason):" in rendered  # the contract another group changes
    assert notify.excerpts == []  # excerpts follow the group whose names they are about
    assert "Group " in rendered


def method_change(path: str, method: str) -> str:
    """A `-W` style patch: a one-line change inside `method` of a class with `__init__`."""
    return patch(
        path,
        "  class Thing:\n"
        "      def __init__(self):\n"
        "          self.value = 0\n"
        f"      def {method}(self):\n"
        "-         return 1\n"
        "+         return 2\n"
        "      def untouched_helper(self):\n"
        "          return 3\n",
    )


def test_definitions_that_only_appear_as_context_do_not_tie_files() -> None:
    diffs = {
        "a/x.py": method_change("a/x.py", "first_method"),
        "b/y.py": method_change("b/y.py", "second_method"),
    }

    groups = group_changes(layer(diffs), module_of, capacity_tokens=room_for(1.2, diffs))

    assert len(groups) == 2


def test_other_groups_see_only_definitions_that_changed() -> None:
    diffs = {
        "a/x.py": method_change("a/x.py", "first_method"),
        "b/y.py": patch("b/y.py", "+def other():\n", size=10),
    }

    layers = grouped_layers(layer(diffs), module_of, capacity_tokens=room_for(1.2, diffs))

    other = next(g for g in layers if "b/y.py" in g.diffs)
    listed = "\n".join(other.signatures_elsewhere)
    assert "def first_method(self):" in listed  # holds the change
    assert "__init__" not in listed
    assert "untouched_helper" not in listed


def test_a_type_definition_follows_the_group_that_uses_the_type() -> None:
    diffs = {
        "a/x.py": patch("a/x.py", "+    status = OrderStatus.PAID\n", size=40),
        "b/y.py": patch("b/y.py", "+def other():\n", size=40),
    }
    excerpts = [Excerpt("definition", "OrderStatus", "a/types.py", "1: class OrderStatus:")]

    layers = grouped_layers(layer(diffs, excerpts), module_of, capacity_tokens=room_for(1.5, diffs))

    uses = next(g for g in layers if "a/x.py" in g.diffs)
    assert [e.name for e in uses.excerpts] == ["OrderStatus"]


def test_a_package_init_in_tests_is_not_tied_to_one_in_the_code() -> None:
    diffs = {
        "pkg/__init__.py": sized("pkg/__init__.py", "+x = 1\n", 100),
        "tests/__init__.py": sized("tests/__init__.py", "+y = 2\n", 100),
    }

    assert len(group_changes(layer(diffs), module_of, capacity_tokens=150)) == 2


def test_a_test_stays_with_every_source_of_the_same_name() -> None:
    diffs = {
        "orders/models.py": sized("orders/models.py", "+a = 1\n", 60),
        "payments/models.py": sized("payments/models.py", "+b = 2\n", 60),
        "tests/test_models.py": sized("tests/test_models.py", "+c = 3\n", 60),
        "docs/notes.py": sized("docs/notes.py", "+d = 4\n", 60),
    }

    found = where(group_changes(layer(diffs), module_of, capacity_tokens=200))

    assert found["orders/models.py"] == found["tests/test_models.py"]
    assert found["payments/models.py"] == found["tests/test_models.py"]


def test_shared_context_is_counted_once_when_checking_whether_a_pr_fits() -> None:
    # Two unrelated files use the same type; its definition goes into the context once.
    diffs = {p: patch(p, "+    status = OrderStatus.PAID\n") for p in ("a/x.py", "b/y.py")}
    big = "1: class OrderStatus:\n" + "2:     PAID = 1\n" * 200
    pr = layer(diffs, [Excerpt("definition", "OrderStatus", "a/types.py", big)])
    once = sum(estimate_tokens(p) for p in diffs.values()) + estimate_tokens(big)

    assert grouped_layers(pr, module_of, capacity_tokens=once + 10) == [pr]


def test_a_pr_that_is_one_oversized_unit_is_flagged() -> None:
    diffs = {"big/huge.py": patch("big/huge.py", "+def huge():\n", size=400)}

    [only] = grouped_layers(layer(diffs), module_of, capacity_tokens=100)

    assert "larger than one review window" in (only.group or "")


def test_groups_do_not_share_their_notes() -> None:
    diffs = {p: patch(p, f"+def f_{p[0]}():\n", size=40) for p in ("a/x.py", "b/y.py")}
    pr = layer(diffs)

    first, second = grouped_layers(pr, module_of, capacity_tokens=room_for(1.5, diffs))
    first.notes.append("only for the first group")

    assert second.notes == [] and pr.notes == []


def test_a_dotfile_ties_nothing() -> None:
    diffs = {
        "a/x.py": sized("a/x.py", "+x = 1\n", 100),
        "b/y.py": sized("b/y.py", "+y = 2\n", 100),
        ".gitignore": sized(".gitignore", "+*.pyc\n", 10),
    }

    assert len(group_changes(layer(diffs), module_of, capacity_tokens=150)) == 2


def test_the_diff_header_is_not_a_use() -> None:
    # Every patch's header names `diff` and `git`; a changed `def diff` ties nothing.
    diffs = {
        "a/x.py": sized("a/x.py", "+def diff(a, b):\n", 100),
        "b/y.py": sized("b/y.py", "+y = 2\n", 100),
    }

    assert len(group_changes(layer(diffs), module_of, capacity_tokens=150)) == 2


def test_every_excerpt_lands_in_the_group_of_its_file() -> None:
    # `helper` is untouched context next to the change; the PR layer still finds its
    # callers, so the group holding x.py must carry them.
    body = "  def helper():\n      return 1\n  def changed():\n-     return 1\n+     return 2\n"
    diffs = {
        "a/x.py": patch("a/x.py", body, size=40),
        "b/y.py": patch("b/y.py", "+def other():\n", size=40),
    }
    excerpts = [Excerpt("usage", "helper", "api/call.py", "1: helper()")]

    layers = grouped_layers(layer(diffs, excerpts), module_of, capacity_tokens=room_for(1.5, diffs))

    holder = next(g for g in layers if "a/x.py" in g.diffs)
    assert [e.name for e in holder.excerpts] == ["helper"]


def test_a_decorator_belongs_to_the_definition_below_it() -> None:
    diff = (
        "@@ -1,6 +1,8 @@ def prev():\n"
        "      return 0\n"
        "+@cache\n"
        "  def foo():\n"
        "      return 1\n"
        "+LIMIT = 3\n"
    )

    assert changed_definitions(diff) == [("foo", "  def foo():")]


def test_a_test_mirroring_the_source_tree_ties_to_that_source() -> None:
    diffs = {
        "orders/models.py": sized("orders/models.py", "+a = 1\n", 60),
        "payments/models.py": sized("payments/models.py", "+b = 2\n", 60),
        "tests/orders/test_models.py": sized("tests/orders/test_models.py", "+c = 3\n", 60),
    }

    units = core_units(layer(diffs))

    assert ["orders/models.py", "tests/orders/test_models.py"] in units
    assert ["payments/models.py"] in units


def test_a_change_belongs_to_the_definition_above_it_whatever_the_layout() -> None:
    multi_line_signature = (
        "@@ -1,6 +1,6 @@\n  def build(\n      a,\n  ) -> int:\n-     return a\n+     return b\n"
    )
    string_at_column_zero = (
        "@@ -1,6 +1,6 @@\n  def build():\n      x = '''\n hello\n '''\n"
        "-     return a\n+     return b\n"
    )
    ruby_instance_variable = (
        "@@ -1,6 +1,6 @@\n  def bump\n-    @count = 0\n+    @count = 1\n  end\n  def other\n  end\n"
    )

    assert [n for n, _ in changed_definitions(multi_line_signature)] == ["build"]
    assert [n for n, _ in changed_definitions(string_at_column_zero)] == ["build"]
    assert [n for n, _ in changed_definitions(ruby_instance_variable)] == ["bump"]


def test_signature_tokens_bound_what_other_groups_are_shown() -> None:
    diffs = {
        "a/x.py": method_change("a/x.py", "first_method"),
        "b/y.py": patch("b/y.py", "+def other():\n", size=10),
    }
    pr = layer(diffs)

    layers = grouped_layers(pr, module_of, capacity_tokens=room_for(1.2, diffs))

    assert all(
        estimate_tokens("\n".join(g.signatures_elsewhere)) <= signature_tokens(pr) for g in layers
    )


def test_groups_do_not_share_any_list() -> None:
    diffs = {p: patch(p, f"+def f_{p[0]}():\n", size=40) for p in ("a/x.py", "b/y.py")}
    pr = layer(diffs)

    first, second = grouped_layers(pr, module_of, capacity_tokens=room_for(1.5, diffs))

    for name in ("diffs", "issues", "specs", "files", "previous_findings", "notes"):
        assert getattr(first, name) is not getattr(second, name)
        assert getattr(first, name) is not getattr(pr, name)


def test_a_decorator_over_several_lines_belongs_to_the_definition_below() -> None:
    diff = (
        "@@ -1,8 +1,8 @@\n  def prev():\n      return 0\n"
        '-@app.route(\n-    "/a",\n-)\n+@app.route(\n+    "/b",\n+)\n  def foo():\n      return 1\n'
    )

    assert [n for n, _ in changed_definitions(diff)] == ["foo"]


def test_signature_tokens_hold_every_definition_another_group_lists() -> None:
    many = "".join(f"+def function_{i}(value):\n" for i in range(60))
    diffs = {"a/x.py": patch("a/x.py", "+x = 1\n", size=200), "b/y.py": patch("b/y.py", many)}
    pr = layer(diffs)

    layers = grouped_layers(pr, module_of, capacity_tokens=room_for(1.0, diffs))

    shown = max(estimate_tokens("\n".join(g.signatures_elsewhere)) for g in layers)
    assert 0 < shown <= signature_tokens(pr)
