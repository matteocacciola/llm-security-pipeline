"""
tracing.py
Where the time goes, per request.

Metrics say how much: p95 of `scan_duration_seconds` is 400 ms. They do
not say where: whether that is the lexical scan, a slow model server
behind a detector, a Redis round trip that waited on a failover, or the
process pool queueing a large input. A trace does. Each guard the pipeline
runs becomes a span under the request's span, so a slow request shows its
shape in whatever tracing backend the application already sends to.

--- Why this takes a tracer object and not a flag ---------------------------

The application already has a tracer, or it does not have tracing. Either
way, this library creating its own provider would be wrong: it would
either compete with the application's for the global registration or
export to nowhere. So the pipeline takes a tracer — anything with the
`start_as_current_span` context-manager shape OpenTelemetry defines — and
does nothing when given none. `opentelemetry-api` is an optional extra;
the library never imports it unless asked.

--- Why span attributes go through the metrics label guard -----------------

A span is indexed and searched like a metric label, and ends up in a
backend that is as widely readable as the metrics one. The same
identifiers are refused for the same reasons: see metrics.FORBIDDEN_LABELS.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

from .metrics import FORBIDDEN_LABELS, _warn_once

SPAN_PREFIX = "llm_security"


@runtime_checkable
class Span(Protocol):
    def set_attribute(self, key: str, value: Any) -> Any: ...


@runtime_checkable
class Tracer(Protocol):
    """The subset of the OpenTelemetry tracer shape this library uses."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> Any: ...


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Iterator[_NoopSpan]:
        yield _NoopSpan()


class SafeTracer:
    """Wraps a tracer so it can neither break a request nor carry an
    identifier. A tracer that raises is swallowed and logged once."""

    def __init__(self, tracer: Tracer | None = None):
        self.tracer = tracer or _NoopTracer()

    @property
    def enabled(self) -> bool:
        return not isinstance(self.tracer, _NoopTracer)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        """A child span named `llm_security.<name>` with the given
        attributes, plus whatever the body sets on it afterwards."""
        clean = {k: v for k, v in attributes.items() if _allowed(k)}
        try:
            cm = self.tracer.start_as_current_span(f"{SPAN_PREFIX}.{name}")
            span = cm.__enter__()
        except Exception as exc:
            _warn_once(f"tracer:{name}", "Tracer failed starting span %r: %s", name, exc)
            yield _NoopSpan()
            return
        guarded = _GuardedSpan(span)
        try:
            for key, value in clean.items():
                guarded.set_attribute(key, value)
            yield guarded
        except BaseException as exc:
            # Record the failure on the span and let it propagate: the
            # span's job is to show what happened, not to change it.
            guarded.set_attribute("error", True)
            guarded.set_attribute("error.type", type(exc).__name__)
            cm.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            cm.__exit__(None, None, None)


class _GuardedSpan:
    def __init__(self, span: Any):
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        if not _allowed(key):
            return
        try:
            self._span.set_attribute(f"{SPAN_PREFIX}.{key}", _scalar(value))
        except Exception as exc:
            _warn_once(f"span:{key}", "Tracer failed setting attribute %r: %s", key, exc)


def _allowed(key: str) -> bool:
    if key in FORBIDDEN_LABELS:
        _warn_once(
            f"span-forbidden:{key}",
            "Refusing span attribute %r: it is an identifier. Dropped.", key,
        )
        return False
    return True


def _scalar(value: Any) -> Any:
    """OpenTelemetry attributes are scalars or lists of scalars; anything
    else is stringified rather than rejected, since a trace attribute is
    for reading, not for querying by structure."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [x if isinstance(x, (str, bool, int, float)) else str(x) for x in value]
    return str(value)
