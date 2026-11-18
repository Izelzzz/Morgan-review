"""Traces of a review (issue #16): one trace, its nodes, model and tool calls, kept private."""

import subprocess
import sys
from collections.abc import Iterator
from contextlib import closing
from typing import Any

import pytest
from gitrepo import git
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from scripted import call
from test_context_cache import Setup as CacheSetup
from test_pipeline_run import ByDimension, Setup, finding, submit

from review_agents import observability
from review_agents.config import Observability
from review_agents.db import connect
from review_agents.pipeline.run import run_review, wait_for_summaries

TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
CODE = "return min(cents, 1)"  # the head's changed line
PR_BODY = "Never refund more than paid."


def enabled(content: str) -> Observability:
    return Observability.model_validate(
        {"enabled": True, "content": content, "exporter": "otlp", "endpoint": "http://127.0.0.1:9"}
    )


@pytest.fixture
def traced() -> Iterator[Any]:
    """Turns tracing on with an in-memory exporter; `traced(content)` returns the exporter."""
    exporters: list[InMemorySpanExporter] = []

    def turn_on(content: str) -> InMemorySpanExporter:
        exporter = InMemorySpanExporter()
        observability.setup(enabled(content), exporter=exporter, batch=False)
        exporters.append(exporter)
        return exporter

    yield turn_on
    observability.shutdown()


def reviewed(setup: Setup, model: ByDimension) -> int:
    review_id = setup.queue("dimensions:\n  " + "\n  ".join(OFF) + "\n")
    setup.run(review_id, model)
    return review_id


OFF = [f"{d}: false" for d in ("spec_drift", "security", "maintainability", "verifiability")]


def reads_then_submits() -> ByDimension:
    return ByDimension(
        scripts={
            "edge_cases": [
                call("read_file", {"path": "payments/refund.py"}),
                submit(finding("edge_cases")),
            ]
        }
    )


def by_name(spans: tuple[ReadableSpan, ...]) -> dict[str, ReadableSpan]:
    return {s.name: s for s in spans}


def ancestors(span: ReadableSpan, spans: tuple[ReadableSpan, ...]) -> list[str]:
    ids = {s.context.span_id: s for s in spans if s.context is not None}
    names, parent = [], span.parent
    while parent is not None and parent.span_id in ids:
        found = ids[parent.span_id]
        names.append(found.name)
        parent = found.parent
    return names


def everything(spans: tuple[ReadableSpan, ...]) -> str:
    """Every name, attribute, event and status text the spans would export."""
    parts: list[str] = []
    for span in spans:
        parts += [span.name, str(dict(span.attributes or {})), str(span.status.description)]
        parts += [f"{e.name} {dict(e.attributes or {})}" for e in span.events]
    return "\n".join(parts)


def test_one_review_is_one_trace_of_its_nodes_model_calls_and_tool_calls(
    tmp_path: Any, traced: Any
) -> None:
    exporter = traced("full")
    setup = Setup(tmp_path)

    review_id = reviewed(setup, reads_then_submits())

    wait_for_summaries()
    every = exporter.get_finished_spans()
    roots = [s for s in every if s.parent is None]
    # The module summaries, written in the background, are the repository's own trace.
    assert sorted({s.name for s in roots}) == ["review", "summaries"]
    root = next(s for s in roots if s.name == "review")
    assert root.context is not None
    spans = tuple(s for s in every if s.context and s.context.trace_id == root.context.trace_id)
    named = by_name(spans)
    assert root.parent is None
    assert (root.attributes or {})["review.id"] == review_id
    nodes: tuple[str, ...] = ("mirror.sync", "config", "context.repo_layer", "context.pr_layer")
    nodes += ("dimension.edge_cases", "walkthrough", "verify.1", "merge", "report", "write")
    for node in nodes:
        assert "review" in ancestors(named[node], spans), node
    llm = [s for s in spans if (s.attributes or {}).get("openinference.span.kind") == "LLM"]
    # Each model call sits under the agent that made it.
    agents = [ancestors(s, spans)[0] for s in llm]
    assert "dimension.edge_cases" in agents
    assert all(a.startswith(("dimension.", "walkthrough", "verify.")) for a in agents), agents
    tool = named["read_file"]
    assert (tool.attributes or {})["openinference.span.kind"] == "TOOL"
    assert "dimension.edge_cases" in ancestors(tool, spans)
    # The steps point at the trace, so a failed node leads to its spans.
    with closing(connect(setup.settings.db_path)) as conn:
        ids = {
            r[0]
            for r in conn.execute("SELECT trace_id FROM steps WHERE review_id = ?", (review_id,))
        }
    assert ids == {format(root.context.trace_id, "032x")}


def test_a_full_trace_holds_the_prompts_and_what_the_tools_read(tmp_path: Any, traced: Any) -> None:
    exporter = traced("full")

    reviewed(Setup(tmp_path), reads_then_submits())

    text = everything(exporter.get_finished_spans())
    assert CODE in text and PR_BODY in text


def test_a_metadata_trace_holds_no_code_prompt_or_model_output(tmp_path: Any, traced: Any) -> None:
    exporter = traced("metadata")

    reviewed(Setup(tmp_path), reads_then_submits())

    spans = exporter.get_finished_spans()
    text = everything(spans)
    assert CODE not in text and PR_BODY not in text
    assert "refunds can exceed the paid amount" not in text  # the model's finding
    # What is left still says what ran and what it cost.
    assert {"review", "dimension.edge_cases", "read_file"} <= set(by_name(spans))
    llm = [s for s in spans if (s.attributes or {}).get("openinference.span.kind") == "LLM"]
    assert llm


def test_a_full_trace_keeps_every_message_of_a_long_conversation(
    traced: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # Each message is several attributes; the SDK's default of 128 per span dropped the
    # earliest ones (issue #112).
    exporter = traced("full")
    messages = [HumanMessage(f"message {i}") for i in range(200)]

    FakeListChatModel(responses=["done"]).invoke(messages)

    [llm] = [
        s
        for s in exporter.get_finished_spans()
        if (s.attributes or {}).get("openinference.span.kind") == "LLM"
    ]
    attributes = llm.attributes or {}
    assert attributes["llm.input_messages.0.message.content"] == "message 0"
    assert attributes["llm.input_messages.199.message.content"] == "message 199"
    assert "Attributes dict is full" not in caplog.text


def test_a_cache_hit_span_is_named_like_its_step(tmp_path: Any, traced: Any) -> None:
    exporter = traced("metadata")
    cache = CacheSetup(tmp_path)
    first, second = cache.review(), cache.review()

    cache.build(first)
    cache.build(second)

    names = [s.name for s in exporter.get_finished_spans()]
    assert names.count("context.repo_layer") == 1
    assert names.count("context.repo_layer.cache_hit") == 1
    assert cache.steps(second) == ["context.repo_layer.cache_hit:ok"]


def test_a_secret_in_the_reviewed_code_is_masked_in_a_full_trace(
    tmp_path: Any, traced: Any
) -> None:
    exporter = traced("full")
    setup = Setup(tmp_path)
    git(setup.upstream.work, "checkout", "-q", "pr")
    head = setup.upstream.commit({"payments/keys.py": f"TOKEN = '{TOKEN}'\n"}, "keys", branch="pr")
    git(setup.upstream.work, "checkout", "-q", "main")
    git(setup.upstream.bare, "update-ref", "refs/pull/7/head", head)
    model = ByDimension(
        scripts={"edge_cases": [call("read_file", {"path": "payments/keys.py"}), submit()]}
    )
    review_id = setup.queue("dimensions:\n  " + "\n  ".join(OFF) + "\n", head=head)

    setup.run(review_id, model)

    text = everything(exporter.get_finished_spans())
    assert TOKEN not in text
    assert "TOKEN = '***'" in text  # it was read, and masked on the way out


def test_without_observability_nothing_is_instrumented_or_exported() -> None:
    # The SDK itself may be loaded anyway: LangSmith, under LangChain, imports it.
    script = (
        "import sys\n"
        "from review_agents import observability\n"
        "from review_agents.config import Observability\n"
        "import review_agents.pipeline.run\n"
        "observability.setup(Observability())\n"
        "with observability.span('review'):\n"
        "    assert observability.current_trace_id() is None\n"
        "prefixes = ('openinference', 'opentelemetry.exporter')\n"
        "loaded = [m for m in sys.modules if m.startswith(prefixes)]\n"
        "assert not loaded, loaded\n"
    )
    done = subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def test_without_observability_the_steps_have_no_trace(tmp_path: Any) -> None:
    setup = Setup(tmp_path)

    review_id = reviewed(setup, reads_then_submits())

    with closing(connect(setup.settings.db_path)) as conn:
        ids = {
            r[0]
            for r in conn.execute("SELECT trace_id FROM steps WHERE review_id = ?", (review_id,))
        }
    assert ids == {None}


@pytest.mark.parametrize(
    ("exporter", "endpoint", "expected"),
    [
        (
            "langfuse",
            "https://us.cloud.langfuse.com/api/public/otel",
            "https://us.cloud.langfuse.com/api/public/otel/v1/traces",
        ),
        ("langsmith", None, "https://api.smith.langchain.com/otel/v1/traces"),
        ("phoenix", None, "http://localhost:6006/v1/traces"),
        ("otlp", "http://collector:4318/", "http://collector:4318/v1/traces"),
        ("otlp", "http://collector:4318/v1/traces", "http://collector:4318/v1/traces"),
    ],
)
def test_each_exporter_preset_sends_to_its_traces_endpoint(
    exporter: str, endpoint: str | None, expected: str
) -> None:
    settings = Observability.model_validate(
        {"enabled": True, "exporter": exporter, "endpoint": endpoint}
    )

    assert observability.traces_endpoint(settings) == expected


@pytest.mark.parametrize("exporter", ["langfuse", "otlp"])
def test_a_preset_without_a_default_endpoint_needs_one(exporter: str) -> None:
    with pytest.raises(ValueError, match="no endpoint"):
        Observability.model_validate({"enabled": True, "exporter": exporter})


def test_a_node_that_raises_is_recorded_as_the_failed_step(tmp_path: Any) -> None:
    setup = Setup(tmp_path)
    review_id = setup.queue()

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("git fetch failed")

    setup.store.sync = broken  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="git fetch failed"):
        setup.run(review_id, ByDimension(scripts={}))

    # Without a trace, the steps alone say which node failed and why (issue #16).
    assert setup.steps(review_id)["mirror.sync"] == ("error", "RuntimeError: git fetch failed")


def test_a_failing_pr_layer_is_recorded_as_the_failed_step(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = Setup(tmp_path)
    review_id = setup.queue()

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr("review_agents.pipeline.run.build_pr_layer", broken)
    with pytest.raises(OSError, match="disk full"):
        setup.run(review_id, ByDimension(scripts={}))

    assert setup.steps(review_id)["context.pr_layer"] == ("error", "OSError: disk full")


def test_run_review_is_importable_without_tracing() -> None:
    assert callable(run_review)


@pytest.mark.parametrize(
    ("content", "expected"), [("metadata", "ValueError"), ("full", f"ValueError: {CODE}")]
)
def test_a_metadata_span_keeps_only_the_lead_of_an_error(
    traced: Any, content: str, expected: str
) -> None:
    exporter = traced(content)

    with observability.span("read_file") as current:
        # Exception text may quote what a model sent; the step keeps the whole reason.
        observability.failed(current, f"ValueError: {CODE}")

    [span] = exporter.get_finished_spans()
    assert span.status.description == expected


def test_a_failed_write_is_recorded_as_the_failed_step(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = Setup(tmp_path)
    review_id = setup.queue()

    def broken(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("review_agents.pipeline.run._insert_finding", broken)
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})
    with pytest.raises(OSError, match="disk full"):
        setup.run(review_id, model)

    assert setup.steps(review_id)["write"] == ("error", "OSError: disk full")
    assert setup.findings(review_id) == []  # rolled back, the step written after
