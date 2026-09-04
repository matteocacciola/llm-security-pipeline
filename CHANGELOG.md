# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project uses [Semantic Versioning](https://semver.org/) — with the caveat
that pre-1.0 minor releases may include breaking changes, called out below.

## [0.3.0] — 2026-09-04

The library has no external users yet, so this release squashes everything
done in one working session — a correctness review of the 0.2.0 codebase,
followed by five feature additions — into a single version rather than the
string of point releases it was actually built as. Test count over the
session: **234 → 474**.

### Fixed

Found by re-reading the library end to end and confirming each finding by
running it, not by inspection alone.

- **`OutputGuard.find_secrets` silently failed to redact every occurrence
  of a secret matched by a pattern with capture groups.** `findall()`
  returns capture groups instead of the whole match once a pattern has any;
  the old fallback resolved each group with `pattern.search(text)`, which
  restarts from position 0 and so returned the *first* match for every
  occurrence, leaving every occurrence after the first unredacted in the
  output. A pattern with exactly one group had a quieter version of the
  same bug: the bare group was replaced, but the identifying prefix (e.g.
  `key-`) survived. Custom patterns are the entire point of
  `pattern_config_path`, so this needed no exotic input to trigger — an
  ordinary regex with a group in it was enough. Rewritten around
  `finditer()`/span-based redaction throughout; see `test_redaction.py`.
- **`OutputGuard.redact` used `str.replace` per finding**, so a finding
  that was a substring of another corrupted the longer redaction depending
  on dict iteration order. Redaction is now computed from `(start, end,
  category)` spans, resolved by keeping the widest overlapping span and
  applying right-to-left in one pass, so ordering can no longer affect the
  result. Secrets outrank PII on an exact-position tie.
- **`ScopeGuard` generated a random per-process HMAC key when none was
  supplied**, so a capability token issued on one process failed
  verification — as "possible tampering" — on any other process sharing no
  key. Now logs an explicit warning at construction, and the signature
  error names the real cause (a process-local key) instead of accusing the
  token of forgery.
- **`CapabilityToken` had no `from_str()`** to pair with `to_str()`, so a
  token could be written down but never read back in another process —
  exactly the deployment this library targets. Added, with defensive
  parsing (16 KB size cap before touching untrusted input, `ScopeError` for
  every malformed shape rather than a raw `ValueError`/`JSONDecodeError`).
  The signature is verified over the exact bytes the token arrived as
  (`token.raw`), not a re-serialization of the parsed payload, removing an
  implicit dependency on the caller's JSON encoder producing byte-identical
  output to the issuer's.
- **`py.typed` was missing** from the package despite `pyproject.toml`
  declaring `Typing :: Typed`, so installed consumers received no type
  information. Added.

### Added — hardening

- **Input size caps.** `Sanitizer` and `OutputGuard` now refuse text over
  `max_scan_chars` (200,000 by default, both scanners) instead of scanning
  it, and mark the result `oversized`; `post_process` withholds an
  oversized model response instead of forwarding it unscanned. Refusing
  rather than truncating is deliberate: a truncated-but-reported-clean scan
  publishes an offset past which nothing is inspected. `max_scan_chars=None`
  opts out.
- **Bounded encoded-payload scanning.** `find_encoded_payloads` now caps
  the number of base64/hex-shaped candidates it decodes
  (`MAX_ENCODED_CANDIDATES = 256`), so a message built entirely of
  base64-shaped tokens can no longer turn into unbounded decode work.
- **External-content risk can be charged to a session, opt-in.**
  `pre_process_external`/`pre_process_external_batch` accept `session_id`
  (plus `principal`/`actor_id`); when given, the risk feeds the session's
  cumulative accumulator the same way `pre_process`/`pre_process_media` do.
  Off by default, preserving prior behaviour, because a poisoned page the
  user never chose is not evidence about the user while a user steering
  retrieval at a planted document is — only the caller knows which applies.
  A batch charges its worst chunk, not the sum, so a wide retrieval does
  not flag a session for being wide.
- **`ruff` and `mypy` configured as bug-finding gates**, not formatters:
  pyflakes/bugbear/bandit/comprehension families only, deliberately
  excluding the stylistic rule families that would rewrite the tree without
  catching anything executable. Both run clean with zero CLI flags. Two
  real `mypy` findings surfaced and were fixed along the way: a
  `None`-attribute risk in `IngestGuard.evaluate` masked by dataclass
  defaults, and a `zip()` without `strict=True` in `MediaScanner` where
  `gather()` already guarantees a 1:1 correspondence.
- **CI gained a dedicated `lint` job**, separate from `test`: no Docker, no
  database containers, so it reports a broken import in under a minute
  instead of waiting on the backend matrix.
- **`SECURITY.md`** — private disclosure channel, response-time
  expectations, and an explicit in-scope/out-of-scope list (the capture-
  group redaction bug is cited as the canonical in-scope example; jailbreak
  phrasings and threshold tuning are explicitly out of scope).
- **`Makefile`** gained `make lint` and `make format` targets matching CI.

### Added — key rotation

Capability tokens are signed once and verified later, possibly by another
process, and remain valid for their TTL — with a single key there is no way
to change it without a window in which every token in flight is rejected.

- **`SigningKeyring`**: holds several keys, signs with exactly one (`.active`),
  verifies against any it holds. Rotation is `with_key()` → `with_active()`
  → `without_key()`, three deploys and no downtime; the order is
  load-bearing (promoting or retiring out of order raises `UnknownKeyId`
  naming the key, not a generic signature error).
- Every token now carries `kid` inside its **signed** payload
  (`CapabilityToken.key_id`); verification looks the id up directly — never
  a try-every-key fallback, which would turn each verification into a
  keyring-wide search and let a retired key behave like a current one.
- `ScopeGuard(secret_key=...)` still works, as shorthand for
  `SigningKeyring.single(key, kid="default")`.
- Audit `tool_call` events now record `token_key_id`, which is how a
  deployment knows when retiring an old key is actually safe (no spent
  token has named it for longer than the longest TTL issued).
- Keys under 32 bytes are refused (HMAC-SHA256 floor), key ids are
  restricted to `[A-Za-z0-9._:-]` (≤64 chars — they end up in JSON
  payloads, error messages, and log lines), and `SigningKeyring` is
  picklable so it survives the process-pool workers `pre_process_media`
  already uses.

**⚠ Breaking:** `kid` is now a required field in a token's signed payload.
Tokens issued before this release do not verify after it. Irrelevant with
zero deployed users; called out for anyone who has since minted tokens
against a pre-release build.

### Added — backend failure policy (fail-open / fail-closed)

Every cross-process guarantee here is a round trip to a shared backend, and
until now an unreachable backend simply propagated a driver exception,
which made the fail-open/fail-closed decision by accident and made it the
same way for every operation.

- **`resilience.py`**: `FailurePolicy`, one decision per operation category
  (`token_replay`, `provenance`, `rate_limit`, `session_risk`, `audit`, and
  — added later in the same session — `detector`), each independently
  `"open"` or `"closed"`. Defaults are asymmetric on purpose: closed where
  refusing is bounded and letting through is not (`token_replay`,
  `provenance`), open where refusing would turn a dependency outage into a
  product outage (`rate_limit`, `session_risk`, `audit`, `detector`).
- **`ResilientBackend`** wraps every backend round trip with a timeout
  (`timeout_seconds`, default 2s — a hung backend never produces a failure
  to have a policy about) and a per-operation circuit breaker
  (`failure_threshold` consecutive failures trips it, `recovery_seconds`
  before one probe call is let through) so a dead backend does not keep
  paying its timeout on every request.
- **Fail-open is never silent.** Every degraded decision is logged, emitted
  as a `backend_degraded` audit event, and surfaced on the result the
  caller is holding (`PreProcessResult.degraded` /
  `PostProcessResult.degraded`, a tuple of `Degradation`). A caller who
  ignores this is running unguarded without knowing it.
- Wired into `ScopeGuard` (token replay), `SessionRateLimiter` (rate limit
  + session risk), `IngestGuard` (provenance), and the audit write path
  itself (a failing audit logger no longer refuses the request it was
  describing, unless `audit="closed"`).
- A guard supplied by the caller keeps its own `FailurePolicy` — that
  choice is the caller's — but the pipeline **adopts its reporting**, so a
  degraded check on a caller-supplied `SessionRateLimiter` still reaches
  `result.degraded` and not only the log.
- `BackendUnavailable` is the exception raised under a closed policy;
  carries `.operation`, documented as a 503 (not evaluated) rather than a
  403 (evaluated and denied).

### Added — streaming output guard

`OutputGuard.scan` needed the finished response; almost every deployed
chat surface streams instead.

- **`services/streaming_guard.py`**: `StreamingOutputGuard`, fed chunk by
  chunk via `.feed()`, finalized via `.finish()`. Scans over a sliding
  window (unemitted buffer + a tail of already-emitted text) so a secret
  split across a chunk boundary is still found, holds back the most recent
  `holdback_chars` (256 default) so a matched pattern is always seen whole
  *before* any of it is emitted, and defers the system-prompt overlap
  ratio until `min_chars_for_overlap` has arrived so a three-token prefix
  cannot trip it.
- **`leaked_before_holdback`**: the one limit that cannot be engineered
  away — a match longer than the hold-back has already had its prefix
  emitted by the time it is recognizable. Still detected (the detection
  tail is sized independently of the hold-back, `≥512` chars by default —
  tying the two together was an actual bug caught by this session's own
  tests: lowering the hold-back for latency silently stopped detecting
  long credentials instead of detecting-and-reporting a partial leak) and
  reported as a distinct incident from a clean block, extended to PII
  redaction crossing the boundary as well as secrets.
- **`SecurityPipeline.guard_stream()` / `GuardedStream`**: async-iterable
  wrapper so the verdict (`blocked`, `reason`, `replacement_text`) survives
  the loop for a caller who has already forwarded chunks. The refusal is
  never yielded as a final chunk — appending it would leave the offending
  text above it on screen; replacing the message is the caller's job.
  Shadow mode disables the hold-back entirely (chunks forward exactly as
  produced) so observing a stream cannot make it feel slower than the
  stream being measured.
- `output_scan` audit events gain `streamed`, `leaked_before_holdback`, and
  `emitted_chars`.

### Added — semantic detector seam

Every existing check decides by shape (a pattern matched, a codepoint was
invisible); paraphrased attacks are documented as bypassing all of it, and
closing that gap needs a model this library deliberately does not ship —
the same position `MediaScanner` already takes on OCR.

- **`services/detectors.py`**: `SemanticDetector` protocol (sync or async,
  returns a float in 0..1 or a `DetectorResult`), `CallableDetector`
  adapter for a plain function, `DetectorEnsemble` to run several
  concurrently — with each other and, at the pipeline level, with the
  lexical scan, since a network-backed judge would otherwise add its full
  latency to every turn.
- **Advisory by default.** `Registration(detector, mode=ADVISORY)` runs,
  scores, and is written to the audit record without changing any
  decision — a classifier's calibration is a property of the deployment's
  traffic, not the classifier, and the failure mode of trusting it too
  early is blocking real customers. `mode=ENFORCING` is opt-in per
  detector, the same shape as the pipeline's existing shadow mode.
- **Scores combine by `max`, not by sum** (`combine_max`, overridable via
  `combine=`): summing lets two weak, correlated detectors reach a
  threshold neither deserved on its own, and the drift compounds with
  every detector added.
- **A failed detector contributes nothing, not zero.** Excluded from the
  combination and recorded in `EnsembleResult.errors`; a model server going
  down cannot quietly lower every risk score in the system. Governed by the
  new `FailurePolicy.detector` category (default open).
- Wired into `pre_process` (`PreProcessResult.detectors`,
  `.combined_risk_score`, kept separate from `sanitized.risk_score` so the
  audit record can still say which signal fired) and
  `pre_process_external`/`_batch`.

### Added — metrics sink with Prometheus adapter

Shadow mode, advisory detectors, and every documented threshold in this
library produce numbers meant to be looked at before a decision is taken;
all of it previously only reached the audit log, which answers "what
happened to this request" and not "what is this week's risk-score
distribution."

- **`metrics.py`**: `MetricsSink` protocol (`increment`/`observe`,
  synchronous by design — it sits on the hot path of every request, and an
  awaitable sink would invite implementations that do network I/O there),
  `NullMetricsSink` (default, does nothing), `InMemoryMetricsSink` (tests,
  or metrics without a Prometheus server), `PrometheusMetricsSink` (needs
  the new `[metrics]` extra: `prometheus-client`).
- **A declared catalogue** (`METRICS`) of every metric this library emits —
  `requests_total`, `blocks_total`, `would_block_total`, `risk_score{kind}`
  (lexical vs. combined, so a threshold is never tuned against the wrong
  one), `detector_score{detector,mode}`, `detector_errors_total`,
  `degradations_total`, `rate_limited_total`, `scan_duration_seconds`,
  `stream_leaks_total` — each declared up front so a scrape before the
  first request returns zeros, not nothing (the difference between "no
  attacks" and "no data"). Risk histograms use 0..1-scale buckets rather
  than the client library's seconds-scale defaults.
- **`SafeMetricsSink`** wraps any sink: no failure policy exists for
  metrics (unlike every other backend in this library) because refusing a
  user's request over a broken counter is never the right call — a sink
  that raises is swallowed and logged once, not per request.
- **Identifying labels are refused, not discouraged.** `FORBIDDEN_LABELS`
  (`session_id`, `principal`, `email`, `token`, …) are dropped at the sink
  with a rate-limited warning; the measurement itself is kept, only the
  label is stripped. This is a mechanism, not documentation, because a
  label like `session_id` both makes a Prometheus series' cardinality
  unbounded and copies an identifier out of the access-controlled audit
  log into a metrics backend that usually is not.
- Wired into every entry point: `pre_process` (including the rate-limit
  exception path, which previously would have made a refused request
  invisible to `requests_total`), `post_process`, `guard_stream`,
  `pre_process_external`, and `authorized_tool_call` (labelled by exception
  *type*, never by message — the type is low-cardinality, the message may
  quote the input and stays in the audit log where identifiers already
  live).

### Added — follow-ups from the post-session review

- **System-prompt canary** (`Canary`, `SecurityPipeline(canary=True)`,
  `pipeline.planted_system_prompt`). An unguessable, letters-only marker
  planted in the system prompt and treated as a secret category, so its
  appearance in output blocks and redacts through the same path as a
  credential, buffered or streamed. Letters only so a leak does not also
  trip the phone-number detector.
- **`DetectorEvaluator`** in `evaluation.py`: scores a corpus through an
  ensemble once, sweeps every signal separately (each detector, lexical,
  combined), and `promotion_report()` answers whether a detector buys
  recall over the lexical scan at a stated false-positive budget. The
  shared confusion-matrix arithmetic was pulled out into `sweep_scores`,
  `recommend_from`, `format_sweep`; `Evaluator` delegates to them.
- **Per-instance circuit breakers.** `ResilientBackend.run()` gained an
  `instance` key; detectors use their name. Found by an evaluator smoke
  test: with one breaker per category, a detector that always failed
  opened the circuit for every healthy detector beside it.
- **Detector result cache** (`DetectorEnsemble(cache_size=, cache_ttl_seconds=)`),
  off by default, LRU keyed by a hash of text plus registration config,
  errors never cached.
- **`AUDIT_SCHEMA_VERSION`** (now `2`) stamped on every audit event.
  Version 1 is the 0.2.0 event shape; 2 adds `combined_risk_score`,
  `detectors`, `degraded`, `token_key_id`, `streamed`,
  `leaked_before_holdback`, `emitted_chars`, `oversized`.
- **`PipelineConfig`** (`config.py`): the pipeline's settings as grouped,
  validated, JSON/YAML-round-trippable dataclasses, separated from the
  per-process objects; `SecurityPipeline.from_config(config, **objects)`
  refuses a setting passed beside the config; `from_dict` refuses unknown
  keys; `pipeline.config_summary` returns the running posture as data.
- **Property-based tests** (`tests/test_properties.py`, Hypothesis, new
  dev dependency): parsers of untrusted input never raise anything but
  their documented error; streaming equals buffered for any text and any
  chunking.

### Changed

- `README.md` rewritten alongside every feature above, then **split**: it
  had grown past 1,200 lines. The README is now ~150 lines — what the
  library is, install, quick start, threat coverage, a documentation index,
  and the three things people get wrong — and the rest lives in `docs/`
  as seven pages (architecture, installation, deployment, identity and
  tokens, guards, detection and tuning, testing). Every cross-reference is
  a link, and a check in the split script verified every link and anchor
  resolves.
- Package metadata: `dev` extra gained `mypy`, `ruff`, `prometheus-client`;
  a new `metrics` extra was added; `all` now includes it.

### Security

- The redaction and `ScopeGuard` fixes under **Fixed** above are the
  security-relevant entries in this release. Anyone who deployed 0.2.0
  with custom secret patterns containing capture groups should treat
  matched secrets as having been logged/forwarded unredacted, and rotate
  them.

[0.3.0]: https://github.com/matteocacciola/llm-security-pipeline/releases/tag/v0.3.0
