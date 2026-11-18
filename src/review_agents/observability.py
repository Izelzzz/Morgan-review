"""Traces of reviews over OpenTelemetry (issue #16, architecture §15, decision D-042).

One review is one trace: a root span `review`, a span per node (the same names as the
review's `steps`), and under each agent's node its model calls, recorded by OpenInference's
LangChain instrumentation, and its tool calls. Spans are sent over OTLP/HTTP to the
exporter's preset endpoint.

Off by default; off, nothing is instrumented or exported and every span is a no-op. On, every
span passes `_Private` on its way out: each string is masked by REDACT, and in `metadata`
mode only the attributes on `METADATA_KEYS` are kept, so no code, prompt or model output
leaves the service. `metadata` also has OpenInference not record them in the first place.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

from opentelemetry import context as otel_context
from opentelemetry import trace

from review_agents.config import Observability
from review_agents.redact import REDACT

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import SpanExporter

SERVICE = "review-agents"
TRACES_PATH = "/v1/traces"
DEFAULT_ENDPOINTS = {
    "langsmith": "https://api.smith.langchain.com/otel",
    "phoenix": "http://localhost:6006",
}
# What a metadata trace keeps: what ran, how it ended, what it cost; never what it read.
METADATA_KEYS = (
    "openinference.span.kind",
    "llm.model_name",
    "llm.provider",
    "llm.system",
    "llm.token_count.",
    "tool.name",
    "review.",
    "session.id",
    "exception.type",
    "exception.escaped",
)

# A metadata span's error keeps its first words, up to the first ": ".
REASON_CHARS = 120
# Spans per OTLP request: in `full` a model call's span holds its whole conversation,
# so a request of the SDK's default 512 could pass an endpoint's size limit.
EXPORT_BATCH = 32

_provider: TracerProvider | None = None
_instrumentor: Any = None
_full = False
_NO_OP = trace.NoOpTracer()


def traces_endpoint(settings: Observability) -> str:
    """Where spans are sent: the preset's or the configured endpoint, ending in /v1/traces."""
    base = str(settings.endpoint) if settings.endpoint else DEFAULT_ENDPOINTS[settings.exporter]
    base = base.rstrip("/")
    return base if base.endswith(TRACES_PATH) else base + TRACES_PATH


def setup(
    settings: Observability, *, exporter: SpanExporter | None = None, batch: bool = True
) -> None:
    """Start tracing as configured; does nothing when observability is off.

    `exporter` replaces the OTLP one, and `batch=False` exports each span as it ends: tests.
    """
    global _provider, _instrumentor, _full
    if not settings.enabled:
        return
    shutdown()
    # Loaded only when tracing is on.
    from openinference.instrumentation import TraceConfig
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import SpanLimits, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    full = settings.content == "full"
    if exporter is None:
        headers = {k: v.get_secret_value() for k, v in settings.headers.items()}
        exporter = OTLPSpanExporter(endpoint=traces_endpoint(settings), headers=headers)
    private = cast("SpanExporter", _Private(exporter, full=full))  # an exporter by its methods
    # In `full` a model call's span holds its conversation, several attributes per message
    # (`llm.input_messages.N.*`), and the SDK's default of 128 per span or event dropped a
    # long one's earliest messages: no count limit there (UNSET, not an env var).
    # `metadata` keeps only a few attributes, under the SDK's defaults.
    limits = (
        SpanLimits(max_span_attributes=SpanLimits.UNSET, max_event_attributes=SpanLimits.UNSET)
        if full
        else SpanLimits()
    )
    provider = TracerProvider(
        resource=Resource.create({"service.name": SERVICE}), span_limits=limits
    )
    provider.add_span_processor(
        BatchSpanProcessor(private, max_export_batch_size=EXPORT_BATCH)
        if batch
        else SimpleSpanProcessor(private)
    )
    hide = not full
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(
        tracer_provider=provider,
        config=TraceConfig(
            hide_inputs=hide,
            hide_outputs=hide,
            hide_input_messages=hide,
            hide_output_messages=hide,
            hide_llm_invocation_parameters=hide,
            hide_llm_tools=hide,
        ),
    )
    _provider, _instrumentor, _full = provider, instrumentor, full


def shutdown() -> None:
    """Send what is left and stop tracing."""
    global _provider, _instrumentor, _full
    if _instrumentor is not None:
        _instrumentor.uninstrument()
    if _provider is not None:
        _provider.shutdown()
    _provider, _instrumentor, _full = None, None, False


def records_content() -> bool:
    """Whether spans may carry what was read and written (`full`); skip the work otherwise."""
    return _full


def tracer() -> trace.Tracer:
    return _provider.get_tracer(__name__) if _provider is not None else _NO_OP


@contextmanager
def span(
    name: str,
    attributes: Mapping[str, Any] | None = None,
    *,
    parent: otel_context.Context | None = None,
    kind: str = "CHAIN",
) -> Iterator[trace.Span]:
    """A span under the current one, or under `parent`: a context taken on another thread."""
    attrs = {"openinference.span.kind": kind, **(attributes or {})}
    with tracer().start_as_current_span(name, context=parent, attributes=attrs) as current:
        yield current


def failed(current: trace.Span, error: str) -> None:
    """Mark a span failed without an exception: an agent that did not hand in."""
    current.set_status(trace.Status(trace.StatusCode.ERROR, error))


def current_context() -> otel_context.Context:
    return otel_context.get_current()


def current_trace_id() -> str | None:
    """The trace the current code runs in, as Langfuse and the others show it; None if off."""
    found = trace.get_current_span().get_span_context()
    return format(found.trace_id, "032x") if found.is_valid else None


def _keep(key: str) -> bool:
    return key.startswith(METADATA_KEYS)


def _mask(value: Any) -> Any:
    if isinstance(value, str):
        return REDACT(value)
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return tuple(REDACT(v) if isinstance(v, str) else v for v in value)
    return value


def _reason(text: str, full: bool) -> str:
    """An error as a span may carry it: whole in `full`; only its lead in `metadata`.

    Exception text can quote what a model sent or a tool read, and the step's error
    keeps the whole reason (same trace id); the lead, such as the exception's type,
    says what failed.
    """
    if not full:
        text = text.split(": ", 1)[0][:REASON_CHARS]
    return str(REDACT(text))


def _filtered(attributes: Mapping[str, Any] | None, full: bool) -> dict[str, Any]:
    return {k: _mask(v) for k, v in (attributes or {}).items() if full or _keep(k)}


class _Private:
    """Wraps the exporter: masks every string, and in `metadata` drops all but metadata."""

    def __init__(self, exporter: SpanExporter, *, full: bool) -> None:
        self._exporter, self._full = exporter, full

    def export(self, spans: Sequence[ReadableSpan]) -> Any:
        return self._exporter.export([self._private(s) for s in spans])

    def _private(self, span: ReadableSpan) -> ReadableSpan:
        from opentelemetry.sdk.trace import Event, ReadableSpan

        full = self._full
        events = [Event(e.name, _filtered(e.attributes, full), e.timestamp) for e in span.events]
        status = span.status
        if status.description:
            status = trace.Status(status.status_code, _reason(status.description, full))
        return ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=_filtered(span.attributes, full),
            events=events,
            links=span.links,
            kind=span.kind,
            status=status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return bool(self._exporter.force_flush(timeout_millis))
