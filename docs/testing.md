# Testing

What each test file checks and why, the backend integration tests, lint and type gates, and the audit schema version.

[← Back to README](../README.md)

## Testing

Tests live in `tests/` and run with pytest (`asyncio_mode = "auto"`, see
`pyproject.toml`):

- `test_redis_cluster_keys.py` – the Redis Cluster key layout: slot
  invariants computed locally (no server needed) plus, when a
  cluster-enabled Redis is reachable at `REDIS_CLUSTER_URL`, the same
  operations against a real one — including a test that the *old* key
  layout is genuinely rejected, so the hash tags can't quietly stop
  being load-bearing.
- `test_risk_weights.py` – the scoring knobs, including a measured
  demonstration of the recall/false-positive trade they control.
- `test_identity_binding.py` – token subject binding, principal-keyed
  session state, and the actor accumulator. Includes a test that documents
  the residual gap: with no principal and no actor id, rotating the session
  id defeats cumulative risk entirely.
- `test_ingest_guard.py` – ingest verdicts, trust tiers, and provenance:
  in particular that content modified *after* a clean ingest is detected at
  retrieval, and that "never ingested" is reported differently from
  "tampered".
- `test_media_guard.py` – the extractor seam: plugged-in output is scanned
  like ordinary text, a failing extractor degrades visibly rather than
  reporting clean, and media risk feeds the same session budget as text.
- `test_evaluation.py` – the threshold sweep and shadow mode.
- `test_exfil_guard.py` – side-channel URL detection: zero-click channels,
  payload shapes, allowlist behaviour, and the pipeline wiring for both
  output and tool arguments.
- `test_unicode_smuggling.py` – Unicode Tag-block and bidi handling,
  including that the hidden run reaches the lexical scan and is gone from
  the text handed to the model.
- `test_in_memory_stores.py` – plain unit tests for
  `InMemoryNonceStore`/`InMemorySessionStore`: no infrastructure, no
  `ProcessPoolExecutor`, just the single-process logic they're actually
  meant for (counter increments, window reset, risk decay/flagging,
  session reset).
- `test_redaction.py` – that a detected secret cannot reach the user.
  Covers patterns with zero, one and two capture groups (`findall` reports
  groups rather than whole matches, which used to let every occurrence
  after the first through unredacted), repeated matches, and findings that
  overlap each other.
- `test_properties.py` – Hypothesis over the parsers that read untrusted
  input (`from_str`, the sanitizer, the hidden-text and encoded-payload
  finders, the output guard) asserting they never raise anything but their
  documented error, and the property that matters most for streaming: for
  any text and any chunking, the streamed result equals the buffered one.
- `test_session_additions.py` – canary, detector cache, per-detector
  breakers, the detector evaluator, audit `schema_version`, and
  `PipelineConfig` round-trips and refusals.
- `test_metrics.py` – that the numbers a person would act on actually come
  out (the shadow comparison, both risk scores separately, per-detector
  distributions), and that the cardinality guard holds: every forbidden
  label parametrized, and an assertion that the pipeline's own call sites
  never make the guard fire in the first place.
- `test_detectors.py` – the socket, not a model: detectors are functions
  returning a number. Covers advisory-by-default, that max does not become
  a sum, that a failed detector is excluded rather than scored zero (with
  an averaging combiner to prove the difference is real), and that a sync
  detector does not block the event loop.
- `test_streaming_guard.py` – every case fed at four chunk sizes including
  one character at a time, because a seam bug shows up at one size and not
  another. Asserts on what was *emitted* rather than on what was detected,
  since detection after emission is not a save, and pins the equivalence
  between a streamed and a buffered response so the choice of transport
  does not become a security decision.
- `test_failure_policy.py` – stores that raise and stores that hang, per
  operation: that the policy is obeyed, that a domain exception is never
  mistaken for a backend failure, that cancellation is not converted into a
  security decision, that the breaker stops calling a dead backend and lets
  one call through after recovery, and above all that no fail-open decision
  is silent.
- `test_key_rotation.py` – the three-step rotation walked end to end,
  asserting that a token issued before each step still verifies after it,
  plus both ways of getting the order wrong, and that the key id is
  untrusted input rather than a hint.
- `test_token_serialization.py` – the `to_str`/`from_str` round trip
  across two independently constructed guards, malformed and oversized
  input, an edited payload failing verification, and the warning and error
  message produced when a guard is using a process-local key.
- `test_input_limits.py` – the size cap, including that a payload placed
  past it does not earn a clean verdict, and both branches of charging
  external-content risk to a session.
- `test_config_loader.py` – plain unit tests for `PatternRegistry`: default
  loading, override-by-name and add-alongside semantics, and every
  `PatternConfigError` path (malformed JSON, missing file, missing
  fields, unknown flag, invalid regex, zero patterns).
- `test_sanitizer.py` – plain unit tests for `scan_text`/`Sanitizer`:
  multilingual phrase matching (by config-defined name, not raw regex
  text), the code-mixing score bonus, encoded-payload detection, the
  block threshold, and `Sanitizer`'s override knobs (custom
  `pattern_config_path`, `include_default_patterns=False`, an explicit
  `registry`).
- `test_token_replay.py` – 5 real OS processes race to redeem the same
  `max_uses=1` capability token; asserts exactly 1 succeeds. Parametrized
  over all three backends (`redis`, `postgres`, `mysql`) — each is
  supposed to provide the identical guarantee on different technology.
- `test_rate_limit.py` – 6 real OS processes each send 1 request against a
  shared session budget of 3; asserts exactly 3 are allowed. Same
  three-backend parametrization as the token-replay test.
- `test_parallel_scanning.py` – asserts a batch scanned via
  `ProcessPoolExecutor` matches a sequential scan of the same chunks,
  including a deliberately injected malicious one. Redis-only: this test
  is about CPU parallelism, not about the state backend.
- `test_end_to_end.py` – exercises `SecurityPipeline` front-to-back: a
  benign message, indirect injection in external content, and a leaked
  secret in model output. Redis-only, for the same reason.

The cross-process tests are marked `@pytest.mark.integration` plus one of
`@pytest.mark.redis` / `postgres` / `mysql`: they need a reachable
instance of that backend (real cross-process state, not an in-memory
stand-in) and spawn real OS processes, so they don't fit the "unit test"
label despite using `assert`. `tests/conftest.py` checks reachability for
each backend once at collection time and skips only the parametrized
cases for backends that aren't reachable — so, for example, running with
only Redis up still exercises the Redis cases and cleanly skips the
PostgreSQL/MySQL ones instead of failing.

Every audit event carries `schema_version` (`AUDIT_SCHEMA_VERSION`,
currently 5); it is bumped whenever a field is added, renamed or changes
meaning, so a consumer can branch on it instead of finding out when its
parser breaks. The history is in `CHANGELOG.md`.

`tests/test_performance.py` is a regression gate on *shape*, not speed:
doubling the input must roughly double the time, because the failure worth
catching is a pattern that turns a scanner quadratic, invisible at twenty
characters and very visible at twenty thousand. It fails on a synthetic
quadratic and passes on the real scanners; a loose absolute floor catches
a scanner that became pathologically slow at every size. Run it alone with
`pytest -m perf` when touching a pattern or a scanner.

The example service under `examples/` has its own tests, run in CI: an
example that no longer runs teaches the wrong integration.

`make mutate` (`tools/mutate.py`) breaks one security decision at a time —
the expiry check, the hold-back, the fail-closed policy, twenty-five in
all — and reports whether any test noticed. A green suite proves nothing
about what it would catch; a surviving mutant is a decision no test
protects. It is slow (the suite runs once per mutation) and deliberately
not a CI gate; run it when you touch a guard. The first run found one:
the forbidden-label guard on custom metrics could be deleted without a
test failing, because the catalogue masked it for every known metric.

Supply chain: every GitHub Action is pinned to a commit SHA (the tag is a
comment beside it, and Dependabot keeps both current), releases use PyPI
trusted publishing with no stored token and refuse a tag that does not
match `pyproject.toml`, an SBOM is attached to each release, and
`make audit` (`pip-audit`, also in CI) checks every dependency against the
advisory databases.

Lint and type checking run as their own CI job and are reproducible
locally with `make lint` (ruff + mypy, both configured in
`pyproject.toml`). Ruff is deliberately configured as a bug-finding gate
rather than a formatter: the pyflakes/bugbear/bandit families are on, the
stylistic ones are not, because a lint job whose output is mostly noise is
one people learn to skip.

```bash
uv sync --extra redis --extra postgres --extra mysql   # drivers used by the test suite

docker run -d --rm -p 6379:6379 redis:7-alpine
docker run -d --rm -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16-alpine
docker run -d --rm -e MYSQL_ALLOW_EMPTY_PASSWORD=yes -e MYSQL_DATABASE=test -p 3306:3306 mysql:8

uv run pytest
```

Override `REDIS_URL` / `POSTGRES_DSN` / `MYSQL_HOST` (+ `MYSQL_PORT`,
`MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB`) if your instances aren't on
the defaults above. `uv run pytest` is always safe to run even with no
backend available at all — every integration case skips instead of
failing, and the unit tests still run and report real coverage on the
in-memory stores.
