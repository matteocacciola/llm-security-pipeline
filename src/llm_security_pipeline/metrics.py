"""
metrics.py
Numbers you can put on a dashboard, rather than grep out of a log.

Three features in this library produce measurements whose whole purpose is
to be looked at before a decision is taken: shadow mode records what
enforcement *would* have done, a newly registered detector runs advisory
until someone has seen its scores, and every threshold in here is
documented as a product decision to be measured rather than a fact. All
three currently end up in the audit stream, which is a log — good for
answering "what happened to this request", useless for answering "what is
the distribution of risk scores this week and where would I put the
threshold".

So the audit log and this are not the same thing and should not be made
into the same thing. The audit log is per-request, keeps identifiers, and
is written somewhere durable and access-controlled. Metrics are aggregate,
carry no identifiers at all, and go to a system that is usually readable by
the whole engineering org. Blurring that line is how a user identifier ends
up in a Grafana dashboard.

--- Why the sink is synchronous -------------------------------------------

`AuditLogger.log` is async because it may write to Redis. `MetricsSink` is
deliberately not: it sits on the hot path of every request, and making it
awaitable would invite implementations that do I/O there. A sink must be
cheap — increment something in memory and return. If yours needs to reach
the network, buffer and flush from a background task; do not do it here.

--- Why there is no fail-open/fail-closed policy for metrics ---------------

Every other backend in this library has one, because for every other
backend both answers are defensible. Here they are not: refusing a user's
request because a counter could not be incremented is never the right
call. A knob whose only sensible setting is "open" is a knob that should
not exist, so instead a sink that raises is swallowed, logged once, and
the request proceeds.

--- Why identifying labels are refused, not discouraged -------------------

A label like `session_id` does two bad things at once: it makes the metric
cardinality unbounded, which is how a Prometheus server falls over, and it
copies an identifier into a store that is usually less protected than the
audit log. Documenting "don't do that" would not be enough, so the names
are refused at the sink and the offending label is dropped with a warning.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------
# Declared up front rather than created on first use. Prometheus needs the
# label names in advance anyway, and having the list in one place is what
# makes the cardinality guard below possible: a metric whose labels are
# whatever the call site happened to pass cannot be checked.

COUNTER = "counter"
HISTOGRAM = "histogram"

# Risk scores are on a 0..1 scale, so the default Prometheus buckets (which
# are tuned for seconds-scale latencies) would put everything in one place.
RISK_BUCKETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

METRICS: dict[str, tuple[str, tuple[str, ...], str, tuple[float, ...] | None]] = {
    "requests_total": (
        COUNTER, ("stage", "outcome", "enforcement"),
        "Requests seen by the pipeline, by stage and what was decided.", None,
    ),
    "blocks_total": (
        COUNTER, ("stage", "reason"),
        "Blocking decisions, by which signal caused them.", None,
    ),
    "would_block_total": (
        COUNTER, ("stage", "reason"),
        "Decisions enforcement would have taken but did not (shadow mode). "
        "Compare against blocks_total before switching a deployment over.", None,
    ),
    "risk_score": (
        HISTOGRAM, ("stage", "kind"),
        "Distribution of risk scores. `kind` separates the lexical score "
        "from the score the decision was actually taken on, so a threshold "
        "can be chosen against the right one.", RISK_BUCKETS,
    ),
    "findings_total": (
        COUNTER, ("stage", "kind", "category"),
        "Detections by category: which patterns are actually earning their "
        "place and which only ever fire on false positives.", None,
    ),
    "detector_score": (
        HISTOGRAM, ("detector", "mode"),
        "Per-detector score distribution. The number to look at when "
        "deciding whether to promote an advisory detector to enforcing.",
        RISK_BUCKETS,
    ),
    "detector_errors_total": (
        COUNTER, ("detector",),
        "Detectors that did not answer. These contribute nothing rather "
        "than zero, so a rise here means the ensemble is quietly weaker.", None,
    ),
    "degradations_total": (
        COUNTER, ("operation", "decision"),
        "Guards that could not run because a backend was unreachable. "
        "Every one of these with decision=open is a request served with "
        "part of the checking skipped.", None,
    ),
    "rate_limited_total": (
        COUNTER, ("stage",),
        "Requests or tool calls refused for being over budget.", None,
    ),
    "scan_duration_seconds": (
        HISTOGRAM, ("stage",),
        "Wall-clock cost of the security layer itself.", LATENCY_BUCKETS,
    ),
    "stream_leaks_total": (
        COUNTER, (),
        "Streamed responses where part of a match had already been emitted "
        "before it was recognizable, i.e. the hold-back was too small. "
        "Should be zero; if it is not, raise holdback_chars.", None,
    ),
}

# Label names that must never reach a metrics backend. Not a style rule:
# each of these is unbounded in cardinality, and each is an identifier that
# has no business being copied out of the audit log into a dashboard.
FORBIDDEN_LABELS = frozenset({
    "session_id", "principal", "actor_id", "subject", "user", "user_id",
    "agent_id", "source_id", "document_id", "text", "content", "nonce",
    "ip", "email", "token", "trace_id", "request_id",
})


# ---------------------------------------------------------------------------
# Protocol and bundled sinks
# ---------------------------------------------------------------------------


@runtime_checkable
class MetricsSink(Protocol):
    """Somewhere to put a number. Must be cheap and must not raise.

    Implementations are called on the hot path of every request. See the
    module docstring on why this is synchronous.
    """

    def increment(self, metric: str, value: float = 1.0, **labels: str) -> None:
        ...

    def observe(self, metric: str, value: float, **labels: str) -> None:
        ...


class NullMetricsSink:
    """The default. Does nothing, as fast as possible."""

    def increment(self, metric: str, value: float = 1.0, **labels: str) -> None:
        return None

    def observe(self, metric: str, value: float, **labels: str) -> None:
        return None


class InMemoryMetricsSink:
    """Keeps everything in dictionaries.

    For tests, and for anyone who wants the numbers without running a
    Prometheus. Unbounded, so not for production with high-cardinality
    labels — which the guard already refuses anyway.
    """

    def __init__(self) -> None:
        self.counters: dict[tuple, float] = defaultdict(float)
        self.observations: dict[tuple, list[float]] = defaultdict(list)

    @staticmethod
    def _key(metric: str, labels: dict) -> tuple:
        return (metric, tuple(sorted(labels.items())))

    def increment(self, metric: str, value: float = 1.0, **labels: str) -> None:
        self.counters[self._key(metric, labels)] += value

    def observe(self, metric: str, value: float, **labels: str) -> None:
        self.observations[self._key(metric, labels)].append(value)

    # -- reading it back -------------------------------------------------

    def count(self, metric: str, **labels: str) -> float:
        if labels:
            return self.counters.get(self._key(metric, labels), 0.0)
        return sum(v for (name, _), v in self.counters.items() if name == metric)

    def values(self, metric: str, **labels: str) -> list[float]:
        if labels:
            return list(self.observations.get(self._key(metric, labels), []))
        return [v for (name, _), vs in self.observations.items() if name == metric for v in vs]


class PrometheusMetricsSink:
    """Adapter for `prometheus_client`.

        pip install "llm-security-pipeline[metrics]"

        from prometheus_client import start_http_server
        start_http_server(9100)
        pipeline = SecurityPipeline(metrics=PrometheusMetricsSink())

    Metrics are declared from the catalogue above at construction, not on
    first use, so a scrape before the first request returns zeros rather
    than nothing — the difference between "no attacks" and "no data".
    """

    def __init__(self, namespace: str = "llm_security", registry=None):
        try:
            from prometheus_client import Counter, Histogram
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                "PrometheusMetricsSink needs prometheus_client. Install it with "
                "pip install 'llm-security-pipeline[metrics]', or use "
                "InMemoryMetricsSink, or write your own MetricsSink."
            ) from exc

        self._namespace = namespace
        self._metrics: dict[str, object] = {}
        kwargs = {"registry": registry} if registry is not None else {}
        for name, (kind, labels, doc, buckets) in METRICS.items():
            full = f"{namespace}_{name}"
            if kind == COUNTER:
                self._metrics[name] = Counter(full, doc, labels, **kwargs)
            else:
                # buckets is never None for a histogram entry; the type is
                # optional because counters share the catalogue tuple.
                self._metrics[name] = Histogram(
                    full, doc, labels, buckets=buckets or RISK_BUCKETS, **kwargs,
                )

    def _child(self, metric: str, labels: dict):
        declared = METRICS.get(metric)
        if declared is None:
            _warn_once(f"metric:{metric}", "Unknown metric %r; dropping.", metric)
            return None
        names = declared[1]
        clean = _sanitize(metric, labels, names)
        collector = self._metrics[metric]
        if not names:
            return collector
        return collector.labels(**{n: clean.get(n, "") for n in names})  # type: ignore[attr-defined]

    def increment(self, metric: str, value: float = 1.0, **labels: str) -> None:
        child = self._child(metric, labels)
        if child is not None:
            child.inc(value)  # type: ignore[attr-defined]

    def observe(self, metric: str, value: float, **labels: str) -> None:
        child = self._child(metric, labels)
        if child is not None:
            child.observe(value)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

_warned: set[str] = set()


def _warn_once(key: str, message: str, *args: object) -> None:
    """A broken metrics call is on the hot path; warning every time would
    turn a small mistake into a log flood."""
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message, *args)


def _sanitize(metric: str, labels: dict, declared: tuple[str, ...]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for name, value in labels.items():
        if name in FORBIDDEN_LABELS:
            _warn_once(
                f"forbidden:{name}",
                "Refusing metric label %r on %r: it is an identifier, so it would "
                "make the series cardinality unbounded and copy an identifier out "
                "of the audit log into the metrics backend. Dropped.",
                name, metric,
            )
            continue
        if name not in declared:
            _warn_once(
                f"undeclared:{metric}:{name}",
                "Metric %r has no label %r declared in the catalogue; dropped.",
                metric, name,
            )
            continue
        clean[name] = "" if value is None else str(value)
    return clean


class SafeMetricsSink:
    """Wraps a sink so it can never break a request.

    There is no fail-closed option here on purpose; see the module
    docstring. Errors are logged once per metric and then swallowed.
    """

    def __init__(self, sink: MetricsSink | None = None):
        self.sink = sink or NullMetricsSink()

    @property
    def enabled(self) -> bool:
        return not isinstance(self.sink, NullMetricsSink)

    def increment(self, metric: str, value: float = 1.0, **labels: str) -> None:
        try:
            self.sink.increment(metric, value, **_checked(metric, labels))
        except Exception as exc:
            _warn_once(f"sink:{metric}", "Metrics sink failed on %r: %s", metric, exc)

    def observe(self, metric: str, value: float, **labels: str) -> None:
        try:
            self.sink.observe(metric, value, **_checked(metric, labels))
        except Exception as exc:
            _warn_once(f"sink:{metric}", "Metrics sink failed on %r: %s", metric, exc)

    @contextmanager
    def timed(self, metric: str, **labels: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(metric, time.perf_counter() - started, **labels)


def _checked(metric: str, labels: dict) -> dict[str, str]:
    """Apply the cardinality guard for every sink, not just the Prometheus
    one: a custom sink is exactly as capable of blowing up its backend."""
    declared = METRICS.get(metric)
    if declared is None:
        _warn_once(f"metric:{metric}", "Unknown metric %r; labels passed through.", metric)
        return {k: str(v) for k, v in labels.items() if k not in FORBIDDEN_LABELS}
    return _sanitize(metric, labels, declared[1])
