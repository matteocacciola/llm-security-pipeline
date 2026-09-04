# Detection, tuning and measurement

Extending the lexical patterns, plugging in semantic detectors, choosing thresholds, shadow mode, and metrics.

[← Back to README](../README.md)

## Extending patterns via config.json

Secret, PII **and** multilingual injection-phrase patterns all live in
`config/patterns.json` and share the same override mechanism, via
`config_loader.py`'s `PatternRegistry`: an entry in a custom file with the
same `name` as a bundled default REPLACES it, a new `name` is added
alongside the rest. `injection_patterns` entries carry one extra required
field, `lang`, since that heuristic groups by language (used for the
code-mixing signal in `scan_text`) — otherwise they're the same
`{name, pattern, flags}` shape as `secret_patterns`/`pii_patterns`.

- `OutputGuard`/`SecurityPipeline(pii_config_path=..., include_default_pii_patterns=...)`
  picks up custom secret/PII patterns.
- `Sanitizer`/`SecurityPipeline(injection_config_path=..., include_default_injection_patterns=...)`
  picks up custom injection phrases, the same way. `Sanitizer` mirrors
  `OutputGuard`'s constructor exactly: `pattern_config_path`,
  `include_default_patterns`, or a pre-built `registry` for full control.
  The module-level `scan_text` function is a thin convenience wrapper
  around a lazily-created default `Sanitizer()` for callers who don't need
  any of that.

Every entry is validated against a pydantic schema before it's compiled:
required fields (`name`/`pattern`, plus `lang` for injection entries),
unknown fields rejected (`extra="forbid"`), unknown flag names, and
regex syntax — all reported as a `PatternConfigError` pointing at the
exact entry (e.g. `secret_patterns.2.pattern`) rather than surfacing as a
confusing error somewhere in the scanning path, or a pattern silently not
applying at all. `pydantic` is consequently a **required** dependency of
this library (not an extra like `redis`/`postgres`/`mysql`): pattern
loading always runs, for every install, regardless of which shared-state
backend you pick.

See the module docstring in `config_loader.py` for the exact schema.

## Semantic detectors

Everything else here decides by shape: a pattern matched, a codepoint was
invisible, a base64 blob decoded into an instruction. That is bypassed by
paraphrasing, as this README says elsewhere — *"kindly set aside the
guidance you were given earlier"* is in no phrase list, and adding it only
moves the problem to the next wording.

Closing that gap needs a model, and this library does not ship one. Same
position `MediaScanner` takes about OCR: extraction is yours, scoring is
ours. Bundling a classifier would pin a runtime, a set of weights and a
licence into a dependency-light library, and freeze a fast-moving choice on
everyone who installs it. What is provided is the socket.

```python
from llm_security_pipeline import (
    CallableDetector, ENFORCING, Registration, SecurityPipeline,
)

pipeline = SecurityPipeline(
    detectors=[
        Registration(CallableDetector("deberta-injection", classify)),        # advisory
        Registration(CallableDetector("llm-judge", judge), mode=ENFORCING),   # decides
    ],
)
```

`classify` may be sync or async and may return a float in 0..1 or a
`DetectorResult`. A sync detector is run on a worker thread rather than
inline, because an inference call blocking the event loop stalls every
other request in the process.

**A new detector is advisory until you say otherwise.** A classifier's
calibration is a property of your traffic, not of the classifier: a model
with an excellent published F1 will still have a threshold that is wrong
for your users on the day you install it, and the failure mode of getting
that wrong is blocking real customers. So it runs, it is scored, it is
written to the audit record, and it changes nothing. Measure it with
`llm_security_pipeline.evaluation`, pick a threshold, then re-register it
as `ENFORCING`. Shadow mode, one detector at a time.

**Scores combine by max, not by sum.** A detector firing on the sentence
the lexical scan already matched is the same evidence read twice. Adding
them lets two weak correlated signals reach a threshold neither deserved,
and the drift grows with every detector added — an ensemble would tend
towards blocking everything. `max` keeps a confident detector able to raise
the verdict alone while refusing to manufacture confidence out of
agreement. Pass `combine=` if your calibration disagrees. `weight` scales a
detector known to be eager; `threshold` silences the noisy floor many
classifiers emit on ordinary text, without hiding it from the record.

**Deciding whether to promote one is a measurement, and `DetectorEvaluator`
is the instrument.** It scores a labelled corpus through the ensemble once
and reports every signal — each detector, the lexical scan, and the
combination — on its own sweep, then answers the question a detector's
owner is actually asking: at the false-positive budget I can afford, does
this one buy recall the lexical scan does not already have?

```
signal          threshold  recall    FPR   verdict
lexical              0.05    0.75  0.000   baseline
combined             0.15    1.00  0.000   +0.25 recall over lexical
my-judge             0.15    1.00  0.000   +0.25 recall over lexical
dead-one                -       -      -   no data: errored on every sample
```

A detector that errored on part of the corpus says so next to its numbers,
because a recall over a subset is not comparable to one over the whole.
Each detector has its own circuit breaker, so a model server that is down
does not switch off the healthy detector next to it. `cache_size=` turns on
an LRU keyed by a hash of the text — never the text — for the case where a
network-backed judge would otherwise score the same retrieval chunk twice.

**A detector that failed contributes nothing, not zero.** Same rule as the
media scanner: 0.0 means "looked, found nothing", and a detector that timed
out has not looked. It is recorded in `errors`, surfaced in
`result.degraded`, and left out of the combination — so a model server
going down cannot quietly lower every risk score in the system. The
`detector` failure-policy category defaults to open, because refusing
traffic when a classifier is unreachable trades a supporting signal for an
outage.

The lexical score is never overwritten: `sanitized.risk_score` stays what
the pattern scan found, `combined_risk_score` is what the decision was
taken on, and both are in the audit event, so the record can still answer
which signal fired.

## Tuning the thresholds

Every number in this library is a default chosen to be reasonable across
agents in general, which means it is wrong for yours in some direction.
`llm_security_pipeline.evaluation` is the instrument for finding out which:

```python
from llm_security_pipeline.evaluation import Evaluator, load_samples

evaluator = Evaluator()
samples = load_samples("corpus.jsonl")     # {"text": ..., "label": "attack"|"benign"}
print(evaluator.report(samples))           # precision/recall/FPR per threshold
missed, false_alarms = evaluator.misclassified(samples, threshold=0.6)
best = evaluator.recommend(samples, max_false_positive_rate=0.01)
```

`recommend` optimizes for recall within a false-positive budget rather
than for F1, deliberately: a false positive is a blocked customer and a
false negative is a breach, and no single number knows the exchange rate
between those for your product. Stating an acceptable false-positive rate
is a decision an operator can actually reason about.

**Run it before you trust the defaults.** Doing exactly that surfaced
something worth knowing: at the default `input_risk_threshold=0.6`, a
plain single-phrase direct injection scores 0.25 and does *not* block. That
is consistent with the design — lexical matching is a supporting signal,
the structural `wrap_as_data` boundary is the actual defence, and
cumulative session risk catches the repeat offender — but it surprises
people who expected single-turn lexical blocking.

The weights that produce that score are configurable:

```python
from llm_security_pipeline import RiskWeights, Sanitizer

pipeline = SecurityPipeline(
    sanitizer=Sanitizer(weights=RiskWeights(per_pattern=0.6)),   # blocks on one match
)
```

`RiskWeights` exposes every contribution — per lexical match and its cap,
encoded payload, hidden text, bidi override, code-mixing. Raising
`per_pattern` buys single-turn blocking and costs false positives on text
that legitimately quotes instructions ("translate 'ignore the previous
message'", "how does prompt injection work"); `tests/test_risk_weights.py`
measures that trade rather than asserting it. Note also that the score is
quantized in steps of `per_pattern`, so with the default of 0.25 every
threshold between 0.25 and 0.4 behaves identically.

A bundled `smoke_corpus()` exercises each detector and is sized to notice a
regression, not to produce a score. Anyone quoting numbers from it as a
benchmark has misread what it is.

## Shadow mode

Calibrating against real traffic requires running the guards over real
traffic, which is not something you can do for the first time in
enforcing mode:

```python
pipeline = SecurityPipeline(enforcement="shadow")
result = await pipeline.pre_process(user_message)
result.blocked        # always False
result.would_block    # what enforcement would have done
```

Everything is scanned, scored and audited; nothing is blocked or even
redacted, because the point is to observe the system as it behaves today
rather than a partially-mitigated version of it. Rate-limit counters keep
incrementing — a shadow deployment that stops counting is not measuring
the same system — but going over budget is recorded on
`result.rate_limited` instead of raised.

## Tracing

Metrics say how much; a trace says where. Pass the application's own
OpenTelemetry tracer (`SecurityPipeline(tracer=...)`, extra `tracing`)
and every guard becomes a child span of the request — `pre_process`,
`detectors` under it, `post_process`, `tool_call` with `tool_result` and
the external scan under that, `pre_process_media`. Attributes carry the
verdict (outcome, scores, matched pattern names) and go through the same
identifier guard as metric labels, for the same reason. The library never
creates a provider of its own: it would compete with the application's or
export to nowhere. A tracer that raises is swallowed and logged once.

## Metrics

Three things in this library produce numbers whose entire purpose is to be
looked at before a decision is taken: shadow mode records what enforcement
*would* have done, an advisory detector runs so you can see its scores, and
every threshold is documented as a product decision to be measured rather
than a fact. All of that lands in the audit stream, which is a log — good
at "what happened to this request", useless at "what is the distribution of
risk scores this week and where should the threshold go".

```python
from prometheus_client import start_http_server
from llm_security_pipeline import PrometheusMetricsSink, SecurityPipeline

start_http_server(9100)
pipeline = SecurityPipeline(metrics=PrometheusMetricsSink())
```

Needs the extra: `pip install "llm-security-pipeline[metrics]"`. Off by
default — the default sink does nothing. `InMemoryMetricsSink` is there for
tests and for anyone who wants the numbers without running a Prometheus,
and `MetricsSink` is a two-method protocol if you have your own.

The series worth building a dashboard on first:

| Metric | What it answers |
| --- | --- |
| `would_block_total` vs `blocks_total` | What enforcement would do if you switched shadow mode off. The whole point of shadow mode, and currently the reason people grep logs. |
| `risk_score{kind}` | The score distribution, split into the lexical score and the score the decision was actually taken on. Tuning a threshold against the wrong one of those is easy to do. |
| `detector_score{detector,mode}` | Whether an advisory detector is ready to enforce. |
| `detector_errors_total` | A detector that did not answer contributes nothing rather than zero, so a rise here means the ensemble is quietly weaker without any score moving. |
| `degradations_total{decision="open"}` | Requests served with part of the checking skipped. |
| `findings_total{category}` | Which patterns earn their place and which only ever fire on false positives. |
| `stream_leaks_total` | Should be zero. If it isn't, `holdback_chars` is smaller than something your patterns can match. |

**The metrics sink is synchronous, unlike the audit logger.** That is not
an oversight: it sits on the hot path of every request, and making it
awaitable invites implementations that do I/O there. A sink increments
something in memory and returns; if yours needs the network, buffer and
flush from a background task.

**There is no failure policy for metrics.** Every other backend has one
because for every other backend both answers are defensible. Refusing a
user's request because a counter could not be incremented is never right, a
knob whose only sensible setting is "open" should not exist, so a sink that
raises is swallowed and logged once.

**Identifying labels are refused, not discouraged.** `session_id` on a
metric does two bad things at once: it makes the series cardinality
unbounded, which is the classic way to take a Prometheus server down, and
it copies an identifier out of the audit log — access-controlled, retained
deliberately — into a metrics backend that is usually readable by the whole
engineering org. Documenting "don't" is not a mechanism, so the names in
`FORBIDDEN_LABELS` are dropped at the sink with a warning, the measurement
itself is kept, and the warning fires once rather than per request.
